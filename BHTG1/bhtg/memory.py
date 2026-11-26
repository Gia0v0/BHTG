"""Typed temporal memory with explicit memory/timestamp/seen state snapshots.

Event updates follow strict time ordering. Opt-in saved z-score inputs retain
their retrospective preprocessing semantics; they are not causal raw amounts.

All events at a timestamp read the same prior state. Aggregation is keep-last
by (node type, local ID) across relations, and timestamps replay in order.
Changing train/eval mode never commits or resets runtime state.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bhtg.config import NODE_TYPES, EDGE_TYPES, MEM_DIM, TEMPORAL_DIM

MEMORY_SEMANTICS = 'causal_seen_typed_last_timestamp_v1'
MEMORY_VALUE_POLICY = 'zero_until_causal_raw_values_available'
MEMORY_AMOUNT_SEMANTICS = 'retrospective_saved_amount_typed_last_timestamp_v2'
MEMORY_AMOUNT_MODES = ('legacy', 'disabled', 'continuous_zscore')


def memory_amount_inputs(stream, rows=None, device=None, amount_mode='legacy'):
    """Read explicit numeric event fields; presence is distinct from known.

    New architecture runs require both fields on every edge, including events
    whose amount is unknown. Legacy streams keep their original zero path.
    """
    if amount_mode not in MEMORY_AMOUNT_MODES:
        raise ValueError('unsupported Memory amount mode: %s' % amount_mode)
    if amount_mode == 'legacy':
        return {}
    count = len(stream['ts'])
    attributes = stream.get('edge_attributes', {})
    presence = stream.get('edge_attribute_present', {})
    result = {}
    for key in ('amount_z', 'amount_known'):
        tensor, present = attributes.get(key), presence.get(key)
        if not isinstance(tensor, torch.Tensor) or tensor.shape != (count,):
            raise ValueError('Memory amount protocol requires a full 1D field: %s' % key)
        if (not isinstance(present, torch.Tensor) or present.shape != (count,) or
                present.dtype != torch.bool):
            raise ValueError('Memory amount field requires boolean presence: %s' % key)
        selected = tensor if rows is None else tensor[rows]
        selected_presence = present if rows is None else present[rows]
        if not bool(selected_presence.all()):
            raise ValueError('Memory amount field is absent for some relations: %s' % key)
        if key == 'amount_z':
            if not selected.is_floating_point() or not bool(torch.isfinite(selected).all()):
                raise ValueError('Memory amount_z must be finite floating point')
        elif selected.dtype != torch.bool:
            raise ValueError('Memory amount_known must have boolean dtype')
        result[key] = selected.to(device=device)
    if not bool(torch.isfinite(result['amount_z'].float()).all()):
        raise ValueError('Memory amount_z overflows model float32 precision')
    return result


class TimeEncodeF32(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension
        self.w = nn.Linear(1, dimension)
        weights = (1 / 10 ** np.linspace(0, 9, dimension)).astype(np.float32)
        self.w.weight = nn.Parameter(torch.from_numpy(weights.reshape(-1, 1)))
        self.w.bias = nn.Parameter(torch.zeros(dimension))

    def forward(self, times):
        return torch.cos(self.w(times.float().unsqueeze(-1)))


class HeteroTemporalMemory(nn.Module):
    def __init__(self, num_nodes_per_type, mem_dim=MEM_DIM,
                 temporal_dim=TEMPORAL_DIM, node_types=NODE_TYPES,
                 edge_types=EDGE_TYPES, value_dim=None, *, amount_mode='legacy',
                 amount_source_policy='rugscreener_saved_zscore_v1'):
        super().__init__()
        if amount_mode not in MEMORY_AMOUNT_MODES:
            raise ValueError('unsupported Memory amount mode: %s' % amount_mode)
        if value_dim is None:
            value_dim = 1 if amount_mode == 'legacy' else 2
        if amount_mode != 'legacy' and value_dim != 2:
            raise ValueError('new Memory amount architecture requires value_dim=2')
        self.amount_mode = amount_mode
        self.amount_source_policy = amount_source_policy
        self.memory_semantics = (MEMORY_SEMANTICS if amount_mode == 'legacy'
                                 else MEMORY_AMOUNT_SEMANTICS)
        self.mem_dim, self.value_dim = mem_dim, value_dim
        self.node_types, self.edge_type_names = list(node_types), list(edge_types)
        self.n_edge_types = len(edge_types)
        self.value_policy = (MEMORY_VALUE_POLICY if amount_mode == 'legacy' else
                             'saved_zscore_and_known_identity_v1' if amount_mode == 'continuous_zscore'
                             else 'amount_and_known_disabled_v1')
        self.memory_tables, self.last_update_t, self.seen = (
            nn.ParameterDict(), nn.ParameterDict(), nn.ParameterDict())
        for ti in range(len(self.node_types)):
            n = num_nodes_per_type[ti]
            self.memory_tables[str(ti)] = nn.Parameter(torch.zeros(n, mem_dim), requires_grad=False)
            self.last_update_t[str(ti)] = nn.Parameter(torch.zeros(n, dtype=torch.float64), requires_grad=False)
            self.seen[str(ti)] = nn.Parameter(torch.zeros(n, dtype=torch.bool), requires_grad=False)
        self.time_enc = TimeEncodeF32(temporal_dim)
        self.updaters = nn.ModuleDict({name: nn.GRUCell(
            2 * mem_dim + value_dim + self.n_edge_types + 2 * temporal_dim,
            mem_dim) for name in self.edge_type_names})
        self._override = None

    def get_memory(self, ntype_idx, locs):
        table = (self._override[ntype_idx] if self._override is not None
                 else self.memory_tables[str(ntype_idx)])
        return table[locs]

    def set_override(self, tables_dict):
        self._override = tables_dict

    def clear_override(self):
        self._override = None

    def state_tables(self, clone=False, detach=False):
        result = {}
        for ti in range(len(self.node_types)):
            values = (self.memory_tables[str(ti)], self.last_update_t[str(ti)], self.seen[str(ti)])
            if detach:
                values = tuple(value.detach() for value in values)
            if clone:
                values = tuple(value.clone() for value in values)
            result[ti] = values
        return result

    def empty_state(self):
        return {ti: tuple(torch.zeros_like(value) for value in values)
                for ti, values in self.state_tables().items()}

    def reset(self):
        self.clear_override()
        with torch.no_grad():
            for values in self.state_tables().values():
                for value in values:
                    value.zero_()

    def backup(self):
        tables = self.state_tables(clone=True, detach=True)
        return tuple({ti: values[column] for ti, values in tables.items()} for column in range(3))

    def restore(self, backup):
        if len(backup) != 3:
            raise ValueError('causal snapshots require memory, timestamp, and seen state')
        self.clear_override()
        with torch.no_grad():
            for ti, values in self.state_tables().items():
                for column, value in enumerate(values):
                    value.copy_(backup[column][ti].to(value.device))

    def detach(self):
        with torch.no_grad():
            for values in self.state_tables().values():
                for value in values:
                    value.detach_()

    def protocol_metadata(self):
        metadata = {'memory_semantics': self.memory_semantics,
                    'value_policy': self.value_policy, 'amount_mode': self.amount_mode,
                    'value_dim': self.value_dim, 'edge_types': self.edge_type_names,
                    'node_types': self.node_types, 'mem_dim': self.mem_dim,
                    'temporal_dim': self.time_enc.dimension}
        if self.amount_mode != 'legacy':
            metadata.update(amount_source_policy=self.amount_source_policy,
                            amount_transform='identity', amount_preprocessing_causal=False,
                            evaluation_scope='retrospective_saved_preprocessed_amounts',
                            event_time_policy='strict_before_cutoff')
        return metadata

    def load_weights(self, checkpoint):
        if checkpoint.get('memory_semantics') != self.memory_semantics:
            raise ValueError('incompatible memory semantics; retrain Phase 2 memory under the causal protocol')
        if checkpoint.get('value_policy') != self.value_policy:
            raise ValueError('incompatible Memory value policy; select matching amount mode')
        if (checkpoint.get('amount_mode', 'legacy') != self.amount_mode or
                checkpoint.get('value_dim', 1) != self.value_dim):
            raise ValueError('incompatible Memory amount mode/value_dim')
        if self.amount_mode != 'legacy':
            for key, expected in self.protocol_metadata().items():
                if checkpoint.get(key) != expected:
                    raise ValueError('incompatible Memory amount metadata: %s' % key)
        declared = checkpoint.get('edge_types')
        if declared is not None and list(declared) != self.edge_type_names:
            raise ValueError('incompatible Memory relation schema; retrain for the new event types')
        if declared is None and any(name.startswith('lp_') for name in self.edge_type_names):
            raise ValueError('v3 Memory weights require an explicit relation schema')
        state = checkpoint.get('memory_state', checkpoint.get('model_state'))
        if not isinstance(state, dict):
            raise ValueError('missing Memory weights')
        updater_names = {key.split('.')[1] for key in state if key.startswith('updaters.')}
        if updater_names != set(self.edge_type_names):
            raise ValueError('Memory updater weights differ from relation schema')
        result = self.load_state_dict(state, strict=False)
        prefixes = ('memory_tables.', 'last_update_t.', 'seen.')
        missing = [key for key in result.missing_keys if not key.startswith(prefixes)]
        if missing or result.unexpected_keys:
            raise ValueError('incompatible memory weights: %s' % (missing + result.unexpected_keys,))
        self.reset()
        return result

    def save(self, path):
        torch.save({'model_state': self.state_dict(), 'mem_dim': self.mem_dim,
                    'temporal_dim': self.time_enc.dimension,
                    **self.protocol_metadata()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.load_weights(checkpoint)
        # save/load is the full-state round trip; load_weights deliberately
        # resets runtime state for freshly replayed pretraining checkpoints.
        if 'model_state' in checkpoint:
            self.load_state_dict(checkpoint['model_state'])
        return checkpoint

    def _single_timestamp(self, tables, src_nt, dst_nt, src, dst, et, ts, value_features):
        device = ts.device
        src_mem = torch.empty(len(src), self.mem_dim, device=device)
        dst_mem = torch.empty_like(src_mem)
        src_time = torch.zeros(len(src), dtype=torch.float64, device=device)
        dst_time = torch.zeros_like(src_time)
        src_seen = torch.zeros(len(src), dtype=torch.bool, device=device)
        dst_seen = torch.zeros_like(src_seen)
        for ti, (memory, times, seen) in tables.items():
            sm, dm = src_nt == ti, dst_nt == ti
            src_mem[sm], dst_mem[dm] = memory[src[sm]], memory[dst[dm]]
            src_time[sm], dst_time[dm] = times[src[sm]], times[dst[dm]]
            src_seen[sm], dst_seen[dm] = seen[src[sm]], seen[dst[dm]]
        if bool(((src_seen & (src_time > ts)) | (dst_seen & (dst_time > ts))).any()):
            raise ValueError('memory replay cannot move backwards; restore a historical snapshot first')
        src_delta = torch.where(src_seen, (ts - src_time) / 86400., torch.zeros_like(ts))
        dst_delta = torch.where(dst_seen, (ts - dst_time) / 86400., torch.zeros_like(ts))
        edge_features = torch.cat([value_features, F.one_hot(et, self.n_edge_types).float()], dim=-1)
        src_message = torch.cat([src_mem, dst_mem, edge_features,
                                 self.time_enc(src_delta), self.time_enc(dst_delta)], dim=-1)
        dst_message = torch.cat([dst_mem, src_mem, edge_features,
                                 self.time_enc(dst_delta), self.time_enc(src_delta)], dim=-1)
        ids = torch.stack([src, dst], dim=1).reshape(-1)
        types = torch.stack([src_nt, dst_nt], dim=1).reshape(-1)
        relations = et.repeat_interleave(2)
        messages = torch.stack([src_message, dst_message], dim=1).reshape(-1, src_message.shape[1])
        times = ts.repeat_interleave(2)
        already_seen = torch.stack([src_seen, dst_seen], dim=1).reshape(-1)
        previous_times = torch.stack([src_time, dst_time], dim=1).reshape(-1)
        updates = {}
        for ti in tables:
            rows = ((types == ti) & (~already_seen | (times > previous_times))).nonzero(as_tuple=True)[0]
            if not len(rows):
                continue
            # Cross-relation typed keep-last. Input order resolves equal-time
            # ties; destination role is last for a same-type self-loop.
            ordered = rows[torch.sort(ids[rows], stable=True).indices]
            keep = torch.ones(len(ordered), dtype=torch.bool, device=device)
            keep[:-1] = ids[ordered[:-1]] != ids[ordered[1:]]
            chosen = ordered[keep]
            groups = []
            for relation_idx, name in enumerate(self.edge_type_names):
                rows = chosen[relations[chosen] == relation_idx]
                if not len(rows):
                    continue
                nodes = ids[rows]
                vectors = self.updaters[name](messages[rows], tables[ti][0][nodes])
                groups.append((nodes, vectors, times[rows]))
            updates[ti] = groups
        return updates

    @staticmethod
    def _apply_updates(tables, updates, *, inplace=False):
        if inplace:
            if torch.is_grad_enabled():
                raise ValueError('in-place Memory replay requires torch.no_grad()')
            if any(value.requires_grad or value.grad_fn is not None
                   for values in tables.values() for value in values):
                raise ValueError('in-place Memory replay requires detached state')
        result = dict(tables)
        for ti, groups in updates.items():
            memory, times, seen = result[ti]
            for ids, vectors, timestamps in groups:
                if inplace:
                    memory.index_copy_(0, ids, vectors)
                    times.index_copy_(0, ids, timestamps.to(dtype=torch.float64))
                    seen.index_fill_(0, ids, True)
                else:
                    memory = memory.index_copy(0, ids, vectors)
                    times = times.index_copy(0, ids, timestamps.to(dtype=torch.float64))
                    seen = seen.index_fill(0, ids, True)
            result[ti] = memory, times, seen
        return result

    def _value_features(self, count, device, *, amount_z=None, amount_known=None):
        if self.amount_mode == 'legacy':
            if amount_z is not None or amount_known is not None:
                raise ValueError('explicit amount tensors require a new Memory amount mode')
            return torch.zeros(count, self.value_dim, device=device)
        if (not isinstance(amount_z, torch.Tensor) or amount_z.shape != (count,) or
                not amount_z.is_floating_point()):
            raise ValueError('new Memory requires amount_z floating tensor of shape [N]')
        if (not isinstance(amount_known, torch.Tensor) or amount_known.shape != (count,) or
                amount_known.dtype != torch.bool):
            raise ValueError('new Memory requires amount_known boolean tensor of shape [N]')
        z = amount_z.to(device=device, dtype=torch.float32)
        known = amount_known.to(device=device)
        if not bool(torch.isfinite(z).all()):
            raise ValueError('Memory amount_z must be finite after float32 conversion')
        if self.amount_mode == 'disabled':
            return torch.zeros(count, 2, device=device)
        safe_z = torch.where(known, z, torch.zeros_like(z))
        return torch.stack((safe_z, known.to(dtype=z.dtype)), dim=-1)

    def compute_new_memory(self, src_nt, dst_nt, src, dst, et, ts, value, tables=None,
                           *, amount_z=None, amount_known=None):
        value_features = self._value_features(len(src), ts.device,
                                               amount_z=amount_z, amount_known=amount_known)
        if len(src) == 0:
            return {}
        ts = ts.to(dtype=torch.float64)
        base = self.state_tables() if tables is None else tables
        if any(len(values) != 3 for values in base.values()):
            raise ValueError('roll state must include (memory, float64 timestamp, seen)')
        # Roll small incident-node tables per timestamp, then write back the
        # full graph once. This avoids a graph-sized copy for every timestamp.
        global_ids, compact = {}, {}
        local_src, local_dst = torch.empty_like(src), torch.empty_like(dst)
        for ti, values in base.items():
            sm, dm = src_nt == ti, dst_nt == ti
            ids = torch.unique(torch.cat([src[sm], dst[dm]]), sorted=True)
            global_ids[ti] = ids
            compact[ti] = tuple(value[ids].to(ts.device) for value in values)
            local_src[sm] = torch.searchsorted(ids, src[sm])
            local_dst[dm] = torch.searchsorted(ids, dst[dm])
        order = torch.sort(ts, stable=True).indices
        _, counts = torch.unique_consecutive(ts[order], return_counts=True)
        start = 0
        for count in counts.tolist():
            rows = order[start:start + count]
            updates = self._single_timestamp(compact, src_nt[rows], dst_nt[rows],
                                             local_src[rows], local_dst[rows], et[rows],
                                             ts[rows], value_features[rows])
            # These incident-node tables are fresh indexed copies. During
            # no-grad replay they can be reused without keeping a second copy
            # for every timestamp; BPTT still uses functional updates.
            compact = self._apply_updates(compact, updates, inplace=not torch.is_grad_enabled())
            start += count
        return {ti: [(ids, compact[ti][0], compact[ti][1])]
                for ti, ids in global_ids.items() if len(ids)}

    def commit(self, updates):
        with torch.no_grad():
            for ti, groups in updates.items():
                memory, times, seen = self.state_tables()[ti]
                for ids, vectors, timestamps in groups:
                    ids = ids.to(memory.device)
                    memory[ids] = vectors.detach().to(memory.device)
                    times[ids] = timestamps.detach().to(times.device, dtype=torch.float64)
                    seen[ids] = True

    def commit_state(self, tables, node_ids=None):
        """Commit a complete state, optionally indexed by global node IDs."""
        with torch.no_grad():
            for ti, values in tables.items():
                for target, value in zip(self.state_tables()[ti], values):
                    value = value.detach().to(target.device)
                    if node_ids is None:
                        target.copy_(value)
                    else:
                        target[node_ids[ti].to(target.device)] = value

    def get_memory_with_updates(self, updates):
        # Targets must read their pre-event state; these tables are for later
        # targets whose history includes the events that produced the updates.
        result = self._apply_updates(self.state_tables(clone=True, detach=True), updates)
        return {ti: values[0] for ti, values in result.items()}

    def roll_batch(self, src_nt, dst_nt, src, dst, et, ts, value,
                   *, amount_z=None, amount_known=None):
        self.commit(self.compute_new_memory(src_nt, dst_nt, src, dst, et, ts, value,
                                            amount_z=amount_z, amount_known=amount_known))

    def roll_diff(self, tables, src_nt, dst_nt, src, dst, et, ts, value,
                  *, amount_z=None, amount_known=None, inplace=False):
        """Replay a block, optionally reusing an owned, detached prefix state.

        ``inplace=True`` is only for a caller-owned state during no-grad replay.
        Differentiable tail blocks retain the default functional updates, and
        callers that need historical snapshots must leave this disabled.
        """
        if inplace and torch.is_grad_enabled():
            raise ValueError('in-place Memory replay requires torch.no_grad()')
        updates = self.compute_new_memory(src_nt, dst_nt, src, dst, et, ts, value, tables=tables,
                                          amount_z=amount_z, amount_known=amount_known)
        return self._apply_updates(tables, updates, inplace=inplace)
