"""Point-in-time node statistics with a fixed training-history scaler.

Archived ``feat`` columns use a single training-end snapshot even for earlier
queries. Archived ``value`` is a per-token *full-history* Z-score, so it cannot
be used to reconstruct historical monetary features. By default the three
amount columns are masked to zero. A caller with genuine event-local amounts
may explicitly provide ``raw_value_key``; its provenance is their contract.

No graph/node labels, archived features, or files are modified or consumed.
"""
import math
import numpy as np
import torch

from bhtg.temporal_time import conservative_timestamps, TIME_PROTOCOL


FEATURE_PROTOCOL = 'point_in_time_15d_training_active_node_scaler_v1'
FEATURE_NAMES = (
    'in_degree', 'out_degree', 'total_degree', 'pool_in_degree',
    'mint_in_degree', 'burn_in_degree', 'incoming_transfer_value_mean',
    'log1p_total_degree', 'incoming_value_sum', 'outgoing_value_sum',
    'temporal_degree_density', 'contract_pool_incident_ratio',
    'unique_neighbour_count', 'max_neighbour_total_degree',
    'mean_neighbour_total_degree',
)


class TemporalFeatureProvider:
    """Compute features at arbitrary cutoffs, fit once on training history.

    ``train_cutoff`` is exclusive. Mean/std are fit per node type using only
    nodes active strictly before it, excluding future/isolated node counts.
    All later calls only transform with these same statistics. Serialized
    scaler state is required when restoring a checkpoint. Chronologically
    increasing queries update event aggregates incrementally; earlier queries
    reset and replay visible history. Repeated cutoff calls reuse one snapshot.
    The graph must remain immutable for the provider's lifetime.
    """

    def __init__(self, g, train_cutoff, raw_value_key=None, *, scaler_state=None):
        self.graph = g
        self.train_cutoff = self._cutoff(train_cutoff)
        if raw_value_key in ('value', 'amount_z', 'amount_known'):
            raise ValueError('archived value/amount_z is a full-history Z-score, not a causal raw amount')
        self.raw_value_key = raw_value_key
        self.ntypes = list(g.ntypes)
        self.offsets, total = {}, 0
        for ntype in self.ntypes:
            self.offsets[ntype] = total
            total += g.num_nodes(ntype)
        self.num_nodes = total
        self._events = []
        neighbour_keys, neighbour_times = [], []
        for canonical in g.canonical_etypes:
            source, destination = g.edges(etype=canonical)
            source = source.detach().cpu().numpy() + self.offsets[canonical[0]]
            destination = destination.detach().cpu().numpy() + self.offsets[canonical[2]]
            times = conservative_timestamps(g.edges[canonical].data['timestamp']).cpu().numpy()
            order = np.argsort(times, kind='stable')
            values = np.zeros(len(source), dtype=np.float64)
            if raw_value_key is not None and raw_value_key in g.edges[canonical].data:
                values = g.edges[canonical].data[raw_value_key].detach().cpu().numpy().astype(np.float64)
                if not np.isfinite(values).all():
                    raise ValueError('raw event amounts must be finite')
            self._events.append((canonical, source[order], destination[order],
                                 times[order], values[order]))
            if len(source):
                neighbour_keys.extend((source * total + destination,
                                       destination * total + source))
                neighbour_times.extend((times, times))
        if neighbour_keys:
            keys = np.concatenate(neighbour_keys)
            times = np.concatenate(neighbour_times)
            order = np.argsort(keys, kind='stable')
            keys = keys[order]
            starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
            first_times = np.minimum.reduceat(times[order], starts)
            keys = keys[starts]
            self._neighbour_nodes = keys // total
            self._neighbours = keys % total
            self._neighbour_first_times = first_times
        else:
            self._neighbour_nodes = np.empty(0, dtype=np.int64)
            self._neighbours = np.empty(0, dtype=np.int64)
            self._neighbour_first_times = np.empty(0, dtype=np.float64)
        self._reset_history()
        self._cached_cutoff = None
        self._cached_raw = None
        self.mean, self.scale, self.fit_counts = {}, {}, {}
        if scaler_state is None:
            raw = self.raw_features(self.train_cutoff)
            for ntype, features in raw.items():
                active = features[:, 2] > 0
                history = features[active].double()
                self.fit_counts[ntype] = int(active.sum())
                if len(history):
                    self.mean[ntype] = history.mean(dim=0)
                    deviation = history.std(dim=0, unbiased=False)
                    self.scale[ntype] = torch.where(deviation > 1e-8,
                                                    deviation, torch.ones_like(deviation))
                else:
                    self.mean[ntype] = torch.zeros(15, dtype=torch.float64)
                    self.scale[ntype] = torch.ones(15, dtype=torch.float64)
        else:
            self._load_scaler(scaler_state)

    @staticmethod
    def _cutoff(value):
        cutoff = float(value)
        if not math.isfinite(cutoff):
            raise ValueError('feature cutoff must be finite')
        return cutoff

    def _reset_history(self):
        n = self.num_nodes
        self._stats = {name: np.zeros(n, dtype=np.float64) for name in (
            'in_degree', 'out_degree', 'pool_in', 'mint_in', 'burn_in',
            'transfer_sum', 'transfer_count', 'in_value', 'out_value', 'contract_inc')}
        self._first = np.full(n, np.inf)
        self._last = np.full(n, -np.inf)
        self._positions = [0] * len(self._events)
        self._history_cutoff = -np.inf

    def _advance(self, cutoff):
        if cutoff < self._history_cutoff:
            self._reset_history()
        stats = self._stats
        for index, (canonical, source, destination, times, values) in enumerate(self._events):
            end = int(np.searchsorted(times, cutoff, side='left'))
            start = self._positions[index]
            if start == end:
                continue
            src, dst = source[start:end], destination[start:end]
            ts, val = times[start:end], values[start:end]
            np.add.at(stats['out_degree'], src, 1)
            np.add.at(stats['in_degree'], dst, 1)
            np.add.at(stats['out_value'], src, val)
            np.add.at(stats['in_value'], dst, val)
            np.minimum.at(self._first, src, ts)
            np.minimum.at(self._first, dst, ts)
            np.maximum.at(self._last, src, ts)
            np.maximum.at(self._last, dst, ts)
            if canonical[0] == 'pool':
                np.add.at(stats['pool_in'], dst, 1)
            if canonical[1] in ('mint', 'burn'):
                np.add.at(stats[canonical[1] + '_in'], dst, 1)
            if canonical[1] == 'transfer':
                np.add.at(stats['transfer_sum'], dst, val)
                np.add.at(stats['transfer_count'], dst, 1)
            if canonical[0] in ('contract', 'pool'):
                np.add.at(stats['contract_inc'], dst, 1)
            if canonical[2] in ('contract', 'pool'):
                np.add.at(stats['contract_inc'], src, 1)
            self._positions[index] = end
        self._history_cutoff = cutoff

    def raw_features(self, cutoff):
        """Return unscaled 15-column historical features for every node type."""
        cutoff = self._cutoff(cutoff)
        if cutoff == self._cached_cutoff:
            return self._cached_raw
        self._advance(cutoff)
        stats = self._stats
        degree = stats['in_degree'] + stats['out_degree']
        n = self.num_nodes
        mean_value = np.divide(stats['transfer_sum'], stats['transfer_count'],
                               out=np.zeros(n), where=stats['transfer_count'] > 0)
        span = np.ones(n)
        active = np.isfinite(self._first)
        span[active] = self._last[active] - self._first[active] + 1.0
        density = np.divide(degree, span, out=np.zeros(n), where=active)
        cir = np.divide(stats['contract_inc'], degree,
                        out=np.zeros(n), where=degree > 0)
        visible = self._neighbour_first_times < cutoff
        nodes = self._neighbour_nodes[visible]
        neighbours = self._neighbours[visible]
        counts = np.bincount(nodes, minlength=n).astype(np.float64)
        max_degree = np.zeros(n)
        np.maximum.at(max_degree, nodes, degree[neighbours])
        degree_sum = np.bincount(nodes, weights=degree[neighbours], minlength=n)
        mean_degree = np.divide(degree_sum, counts, out=np.zeros(n), where=counts > 0)
        matrix = np.stack((stats['in_degree'], stats['out_degree'], degree,
                           stats['pool_in'], stats['mint_in'], stats['burn_in'],
                           mean_value, np.log1p(degree), stats['in_value'],
                           stats['out_value'], density, cir, counts, max_degree,
                           mean_degree), axis=1)
        raw = {}
        for ntype in self.ntypes:
            start = self.offsets[ntype]
            raw[ntype] = torch.from_numpy(matrix[start:start + self.graph.num_nodes(ntype)].copy())
        self._cached_cutoff, self._cached_raw = cutoff, raw
        return raw

    def get_features(self, cutoff, node_ids=None):
        """Transform using the frozen scaler, optionally selecting local IDs."""
        raw = self.raw_features(cutoff)
        features = {}
        for ntype in self.ntypes:
            tensor = raw[ntype]
            if node_ids is not None:
                ids = torch.as_tensor(node_ids.get(ntype, []), dtype=torch.long).cpu()
                if bool(((ids < 0) | (ids >= len(tensor))).any()):
                    raise ValueError('%s feature node ID out of bounds' % ntype)
                tensor = tensor[ids]
            features[ntype] = ((tensor - self.mean[ntype]) / self.scale[ntype]).float()
        return features

    def state_dict(self):
        """Plain tensor state suitable for torch checkpoint serialization."""
        return {
            'protocol': FEATURE_PROTOCOL, 'time_protocol': TIME_PROTOCOL,
            'feature_names': list(FEATURE_NAMES), 'train_cutoff': self.train_cutoff,
            'raw_value_key': self.raw_value_key, 'fit_counts': dict(self.fit_counts),
            'mean': {nt: value.clone() for nt, value in self.mean.items()},
            'scale': {nt: value.clone() for nt, value in self.scale.items()},
        }

    def _load_scaler(self, state):
        if state.get('protocol') != FEATURE_PROTOCOL or state.get('time_protocol') != TIME_PROTOCOL:
            raise ValueError('incompatible temporal feature/scaler protocol; retraining is required')
        if (state.get('feature_names') != list(FEATURE_NAMES) or
                float(state['train_cutoff']) != self.train_cutoff or
                state.get('raw_value_key') != self.raw_value_key):
            raise ValueError('feature scaler metadata does not match this run')
        for ntype in self.ntypes:
            mean = torch.as_tensor(state['mean'][ntype], dtype=torch.float64).cpu()
            scale = torch.as_tensor(state['scale'][ntype], dtype=torch.float64).cpu()
            if mean.shape != (15,) or scale.shape != (15,) or not bool(
                    torch.isfinite(mean).all() & torch.isfinite(scale).all() & (scale > 0).all()):
                raise ValueError('invalid %s feature scaler' % ntype)
            self.mean[ntype], self.scale[ntype] = mean.clone(), scale.clone()
        self.fit_counts = dict(state['fit_counts'])

    @classmethod
    def from_state_dict(cls, g, state):
        return cls(g, state['train_cutoff'], state.get('raw_value_key'), scaler_state=state)
