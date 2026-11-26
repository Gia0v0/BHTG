"""Data views for unlabeled Phase 2 and full-history Phase 3."""

import math

import dgl
import numpy as np
import torch

from bhtg.node_feature_data import load_nine_feature_inputs
from bhtg.node_feature_provider import BehaviorNineFeatureProvider
from bhtg.temporal_time import conservative_timestamps


EXPERIMENT_PROTOCOL = 'unlabeled_phase2_full_history_phase3_v1'


def _phase2_source(g):
    events, is_token, is_pool, binding = load_nine_feature_inputs(g, include_source_roles=True)
    roles = binding['source_role_vocabulary']
    unlabeled_code = roles.index('unlabeled')
    selected_edges, positive_times = {}, []
    unlabeled_actions = np.zeros(len(events), dtype=bool)
    other_actions = np.zeros_like(unlabeled_actions)
    for canonical in g.canonical_etypes:
        count = g.num_edges(canonical)
        if not count:
            selected_edges[canonical] = torch.empty(0, dtype=g.idtype)
            continue
        data = g.edges[canonical].data
        codes = data.get('source_role_code')
        row_ids = data.get('source_row_id_code')
        for field, values, limit in (('source role', codes, len(roles)),
                                     ('source row', row_ids, len(events))):
            if (values is None or values.shape != (count,) or values.dtype != torch.int64
                    or bool(((values < 0) | (values >= limit)).any())):
                raise ValueError('invalid %s provenance for %r' % (field, canonical))
        keep = codes == unlabeled_code
        if canonical[1] == 'create' and bool(keep.any()):
            raise ValueError('create edge cannot have an unlabeled source role')
        unlabeled_actions[row_ids[keep].numpy()] = True
        other_actions[row_ids[~keep].numpy()] = True
        selected_edges[canonical] = torch.where(keep)[0].to(g.idtype)
        if canonical[2] == 'contract':
            positive_times.append(conservative_timestamps(data['timestamp'][keep]).numpy())

    if not bool(np.logical_xor(unlabeled_actions, other_actions).all()):
        raise ValueError('missing or inconsistent graph-leg source roles for canonical actions')
    if not any(len(times) for times in positive_times):
        raise ValueError('no unlabeled graph edges targeting contracts')
    times = np.concatenate(positive_times)
    train, val = map(float, np.quantile(times, (0.70, 0.85)))
    if not (times.min() < train < val <= times.max()
            and np.any((times >= train) & (times < val))):
        raise ValueError('unlabeled contract-target timestamps cannot form three chronological cohorts')
    binding = dict(binding, experiment_protocol=EXPERIMENT_PROTOCOL,
                   scaler_fit_source_roles=['unlabeled'], phase2_val_cutoff=val,
                   cutoff_basis='unlabeled_contract_target_graph_edges')
    return (events, is_token, is_pool, binding), unlabeled_actions, selected_edges, train, val


def unlabeled_phase2_inputs(g):
    """Preserve node IDs and relation schema while isolating unlabeled events."""
    (events, is_token, is_pool, binding), keep, selected_edges, train, val = _phase2_source(g)
    unlabeled = dgl.edge_subgraph(g, selected_edges, relabel_nodes=False, store_ids=False)
    for ntype in unlabeled.ntypes:
        # Archived labels and aggregate features belong to the full-history graph.
        data = unlabeled.nodes[ntype].data
        for key in ('label', 'feat'):
            if key in data:
                del data[key]
    events = events[keep]
    unlabeled_tokens = np.zeros_like(is_token)
    unlabeled_tokens[events['token']] = True
    provider = BehaviorNineFeatureProvider(
        unlabeled, train, feature_inputs=(events, unlabeled_tokens, is_pool, binding))
    return unlabeled, provider, train, val


def restore_full_history_feature_provider(g, state):
    """Apply the proven unlabeled Phase 2 scaler to the full event index."""
    if not isinstance(state, dict):
        raise ValueError('full-history Phase 3 requires the saved unlabeled Phase 2 scaler')
    inputs, _, _, train, _ = _phase2_source(g)
    return BehaviorNineFeatureProvider(g, train, scaler_state=state, feature_inputs=inputs)


def full_history_cutoff(g):
    """Return the earliest float64 cutoff that admits every archived edge."""
    latest = max((conservative_timestamps(g.edges[canonical].data['timestamp']).max().item()
                  for canonical in g.canonical_etypes if g.num_edges(canonical)), default=-math.inf)
    cutoff = float(np.nextafter(latest, math.inf))
    if not (math.isfinite(latest) and math.isfinite(cutoff)):
        raise ValueError('full-history graph requires a finite nonempty event timeline')
    return cutoff
