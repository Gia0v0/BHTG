"""Replay an exact BPTT tail on its incident nodes, without dense table grads.

The detached prefix remains a complete typed snapshot. Tail state is indexed
only by the union of nodes incident to the tail's events. Queries outside that
union read detached prefix rows; queries inside read differentiable tail rows.
No parameter or persistent checkpoint schema is introduced by these views.
"""
import torch


def _state(values, expected_rows=None, dimension=None):
    if not isinstance(values, (tuple, list)) or len(values) != 3:
        raise ValueError('compact tail state requires memory, timestamp and seen')
    memory, times, seen = values
    if (not all(torch.is_tensor(value) for value in values)
            or memory.ndim != 2 or times.ndim != 1 or seen.ndim != 1
            or times.shape[0] != memory.shape[0] or seen.shape[0] != memory.shape[0]
            or not memory.is_floating_point() or times.dtype != torch.float64
            or seen.dtype != torch.bool):
        raise ValueError('invalid compact tail state shape or dtype')
    if expected_rows is not None and memory.shape[0] != expected_rows:
        raise ValueError('compact tail state does not match incident node IDs')
    if dimension is not None and memory.shape[1] != dimension:
        raise ValueError('compact tail memory dimension differs from prefix')
    if any(value.device != memory.device for value in values):
        raise ValueError('compact tail state tensors must share a device')
    return memory, times, seen


def _ids(values, memory):
    ids = torch.as_tensor(values, device=memory.device)
    if (ids.ndim != 1 or ids.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
        raise ValueError('compact tail node IDs must be one-dimensional integers')
    ids = ids.long()
    if (bool(((ids < 0) | (ids >= len(memory))).any())
            or (len(ids) > 1 and not bool((ids[1:] > ids[:-1]).all()))):
        raise ValueError('compact tail node IDs must be sorted, unique and in bounds')
    return ids


def _prefix_state(prefix):
    result = {}
    for node_type, values in prefix.items():
        values = _state(values)
        if any(value.requires_grad or value.grad_fn is not None for value in values):
            raise ValueError('compact tail prefix must be detached from autograd')
        result[node_type] = tuple(value.detach() for value in values)
    return result


class _IndexedMemoryView:
    """Support the existing ``memory_override[type][local_ids]`` interface."""

    def __init__(self, prefix_memory, node_ids, tail_memory):
        self.prefix_memory = prefix_memory
        self.node_ids = node_ids
        self.tail_memory = tail_memory

    def __getitem__(self, locations):
        locations = torch.as_tensor(locations, device=self.prefix_memory.device)
        if (locations.ndim != 1 or locations.dtype not in (
                torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
            raise ValueError('compact memory lookup requires one-dimensional integer IDs')
        locations = locations.long()
        if bool(((locations < 0) | (locations >= len(self.prefix_memory))).any()):
            raise IndexError('compact memory lookup outside prefix node range')
        # Only the queried rows are materialized. This detached gather cannot
        # create a gradient with the full prefix table's shape.
        result = self.prefix_memory[locations]
        if not len(locations) or not len(self.node_ids):
            return result
        positions = torch.searchsorted(self.node_ids, locations)
        safe_positions = positions.clamp(max=len(self.node_ids) - 1)
        hits = (positions < len(self.node_ids)) & (
            self.node_ids[safe_positions] == locations)
        query_rows = hits.nonzero(as_tuple=True)[0]
        if not len(query_rows):
            return result
        tail_rows = positions[query_rows].to(self.tail_memory.device)
        vectors = self.tail_memory[tail_rows].to(result.device)
        # The differentiable copy's leading dimension is the query size, not
        # the graph size. Repeated query IDs accumulate gradients normally.
        return result.index_copy(0, query_rows, vectors)


class CompactMemoryTail:
    """A detached full prefix plus one differentiable, typed incident-node tail."""

    @staticmethod
    def compact_state(prefix, node_ids):
        prefix = _prefix_state(prefix)
        if set(prefix) != set(node_ids):
            raise ValueError('compact tail node types differ from prefix')
        result = {}
        for node_type, values in prefix.items():
            ids = _ids(node_ids[node_type], values[0])
            # Advanced indexing already owns its storage; no full-table clone
            # or differentiable connection to the prefix is needed.
            result[node_type] = tuple(value[ids] for value in values)
        return result

    def __init__(self, prefix, node_ids, tables):
        self.prefix = _prefix_state(prefix)
        if set(self.prefix) != set(node_ids) or set(tables) != set(self.prefix):
            raise ValueError('compact tail node types differ from prefix')
        self.node_ids, self.tables = {}, {}
        for node_type, values in self.prefix.items():
            ids = _ids(node_ids[node_type], values[0])
            tail = _state(tables[node_type], len(ids), values[0].shape[1])
            if tail[0].dtype != values[0].dtype:
                raise ValueError('compact tail memory dtype differs from prefix')
            self.node_ids[node_type], self.tables[node_type] = ids, tail

    @torch.no_grad()
    def commit(self, memory):
        """Populate preallocated runtime tables for Phase A, without a grad graph."""
        memory.clear_override()
        memory.commit_state(self.prefix)
        memory.commit_state(self.tables, node_ids=self.node_ids)

    def overrides(self):
        """Return indexed views; never reconstruct a differentiable full table."""
        return {node_type: _IndexedMemoryView(values[0], self.node_ids[node_type],
                                               self.tables[node_type][0])
                for node_type, values in self.prefix.items()}
