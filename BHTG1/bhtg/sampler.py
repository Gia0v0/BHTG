"""Strictly historical sampling with a separate event budget per destination.

The archived graph is immutable. Only explicitly selected edges are returned;
inducing all edges between retained nodes would undo the sampling budget.
Historical ``_all_gids`` names hold full-graph LOCAL IDs, not address-map IDs.
"""
import math
import numpy as np
import torch
import dgl

from bhtg.temporal_time import conservative_timestamps


SAMPLER_PROTOCOL = 'per_destination_relation_quota_selected_events_v2'


def _sampling_index(g):
    """Index an immutable graph once; all per-query operations use CPU IDs."""
    cached = getattr(g, '_temporal_sampling_index', None)
    if cached is not None:
        return cached
    entries = []
    for canonical in g.canonical_etypes:
        src, dst = g.edges(etype=canonical)
        times = conservative_timestamps(
            g.edges[canonical].data['timestamp']).detach().cpu().numpy()
        source = src.detach().cpu().numpy()
        destination = dst.detach().cpu().numpy()
        edge_ids = np.arange(len(source), dtype=np.int64)
        order = np.lexsort((edge_ids, -times, destination))
        offsets = np.searchsorted(destination[order],
                                  np.arange(g.num_nodes(canonical[2]) + 1))
        entries.append((canonical, source, times, order, offsets, -times[order], destination))
    g._temporal_sampling_index = entries
    return entries


def _validate_quotas(g, relation_quotas, k):
    quotas = dict(relation_quotas or {})
    for key, count in quotas.items():
        if key not in g.etypes and key not in g.canonical_etypes:
            raise ValueError('unknown relation quota: %r' % (key,))
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError('relation quotas must be non-negative integers')
        if isinstance(key, tuple) and key[1] in quotas:
            raise ValueError('use either canonical or relation-name quotas, not both')
    if sum(quotas.values()) > k:
        raise ValueError('relation quotas exceed the per-destination budget k')
    return quotas


def sample_token_subgraph(g, seed_contracts, current_time, k=10, n_hop=1,
                          relation_quotas=None, *, feature_provider=None,
                          seed_nodes=None):
    """Return selected incoming events with ``timestamp < current_time``.

    Each destination gets up to ``k`` events across ALL canonical relations.
    Relation quotas reserve recent events first; unused slots are filled by the
    most recent remaining events. Quotas keyed by relation name are shared
    across its canonical types. Ties use canonical type then original edge ID.
    Each newly reached node is expanded once, up to ``n_hop``. Repeated events
    from one neighbour consume slots, preserving transaction multiplicity.

    ``seed_nodes`` preserves additional typed prediction endpoints even when
    they have no history. ``feature_provider`` supplies cutoff-specific,
    training-scaled features, replacing archived full-history node features in
    the returned graph only. The input graph and files are never rewritten.
    """
    if int(k) != k or int(k) < 1 or int(n_hop) != n_hop or int(n_hop) < 0:
        raise ValueError('k must be positive and n_hop non-negative integers')
    cutoff = float(current_time)
    if not math.isfinite(cutoff):
        raise ValueError('current_time must be finite')
    k, n_hop = int(k), int(n_hop)
    quotas = _validate_quotas(g, relation_quotas, k)
    seed_contracts = torch.as_tensor(seed_contracts, dtype=torch.long).cpu()
    active = {nt: set() for nt in g.ntypes}
    if 'contract' not in active:
        raise ValueError('graph must contain contract nodes')
    active['contract'].update(seed_contracts.tolist())
    for ntype, node_ids in (seed_nodes or {}).items():
        if ntype not in active:
            raise ValueError('unknown seed node type: %s' % ntype)
        active[ntype].update(torch.as_tensor(node_ids, dtype=torch.long).cpu().tolist())
    for ntype, node_ids in active.items():
        if any(node < 0 or node >= g.num_nodes(ntype) for node in node_ids):
            raise ValueError('%s seed out of bounds' % ntype)

    index = _sampling_index(g)
    chosen = {canonical: set() for canonical in g.canonical_etypes}
    frontier = {nt: set(ids) for nt, ids in active.items()}
    expanded = {nt: set() for nt in g.ntypes}
    for _ in range(n_hop):
        next_frontier = {nt: set() for nt in g.ntypes}
        for dst_type, destinations in frontier.items():
            for destination in sorted(destinations - expanded[dst_type]):
                candidates = []
                for relation_index, (canonical, src, times, order, offsets, negative_times, _) in enumerate(index):
                    if canonical[2] != dst_type:
                        continue
                    incoming = order[offsets[destination]:offsets[destination + 1]]
                    # Descending timestamps: right excludes exact cutoff ties.
                    start = np.searchsorted(
                        negative_times[offsets[destination]:offsets[destination + 1]],
                        -cutoff, side='right')
                    for edge_id in incoming[start:start + k]:
                        candidates.append((-float(times[edge_id]), relation_index,
                                           int(edge_id), int(src[edge_id])))
                candidates.sort()
                selected = []
                selected_keys = set()
                for relation_key, quota in quotas.items():
                    count = 0
                    for event in candidates:
                        canonical = index[event[1]][0]
                        matches = (canonical == relation_key if isinstance(relation_key, tuple)
                                   else canonical[1] == relation_key)
                        event_key = event[1:3]
                        if matches and event_key not in selected_keys and count < quota:
                            selected.append(event)
                            selected_keys.add(event_key)
                            count += 1
                for event in candidates:
                    if len(selected) >= k:
                        break
                    if event[1:3] not in selected_keys:
                        selected.append(event)
                        selected_keys.add(event[1:3])
                for _, relation_index, edge_id, source in selected:
                    canonical = index[relation_index][0]
                    chosen[canonical].add(edge_id)
                    active[canonical[0]].add(source)
                    next_frontier[canonical[0]].add(source)
                expanded[dst_type].add(destination)
        frontier = next_frontier

    all_ids = {nt: sorted(ids) for nt, ids in active.items()}
    mappings = {nt: {node: i for i, node in enumerate(ids)}
                for nt, ids in all_ids.items()}
    edge_dict, selected_edges = {}, {}
    # The index already owns the CPU endpoints in original EID order. Gathering
    # the selected rows avoids asking DGL to materialize every relation again.
    for canonical, src, _, _, _, _, dst in index:
        edge_ids = torch.tensor(sorted(chosen[canonical]), dtype=torch.long)
        selected_edges[canonical] = edge_ids
        rows = edge_ids.numpy()
        source = src[rows].tolist()
        destination = dst[rows].tolist()
        edge_dict[canonical] = (
            torch.tensor([mappings[canonical[0]][v] for v in source], dtype=torch.long),
            torch.tensor([mappings[canonical[2]][v] for v in destination], dtype=torch.long))
    result = dgl.heterograph(edge_dict,
                           num_nodes_dict={nt: len(ids) for nt, ids in all_ids.items()})
    for ntype, ids in all_ids.items():
        for name, tensor in g.nodes[ntype].data.items():
            result.nodes[ntype].data[name] = tensor[
                torch.tensor(ids, dtype=torch.long, device=tensor.device)].cpu()
    if feature_provider is not None:
        features = feature_provider.get_features(cutoff, all_ids)
        for ntype, tensor in features.items():
            result.nodes[ntype].data['feat'] = tensor.cpu()
    visibility_times = {entry[0]: entry[2] for entry in index}
    for canonical, edge_ids in selected_edges.items():
        for name, tensor in g.edges[canonical].data.items():
            result.edges[canonical].data[name] = tensor[edge_ids.to(tensor.device)].cpu()
        # Propagate an exact float64 visibility bound to downstream encoders.
        result.edges[canonical].data['timestamp'] = torch.from_numpy(
            visibility_times[canonical][edge_ids.numpy()].copy())
        result.edges[canonical].data[dgl.EID] = edge_ids
    result._all_gids = all_ids
    result._gid_maps = mappings
    result._cid_map = mappings['contract']
    result._sampler_protocol = SAMPLER_PROTOCOL
    result._sampling_hops = n_hop
    seed_positions = torch.tensor([mappings['contract'][int(v)]
                                   for v in seed_contracts], dtype=torch.long)
    return result, seed_positions
