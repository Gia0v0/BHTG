"""Versioned node-feature providers with backward-compatible legacy scalers.

``legacy15`` delegates to the historical provider without changing its saved
state. ``behavior9`` consumes hash-bound canonical actions, scales nine values
on valid, active training nodes, and appends two unscaled amount-validity masks.
The nine-feature index owns all event semantics and strict-before filtering.
"""
import hashlib
import json
import math
import os

import torch

from bhtg.temporal_features import (
    FEATURE_NAMES as LEGACY_FEATURE_NAMES,
    FEATURE_PROTOCOL as LEGACY_FEATURE_PROTOCOL,
    TemporalFeatureProvider,
)
from bhtg.temporal_time import TIME_PROTOCOL


BEHAVIOR_FEATURE_PROTOCOL = 'point_in_time_behavior9_11d_training_valid_scaler_v1'
SCALER_PROTOCOL = 'per_type_active_valid_population_std_v1'
MASK_FEATURE_NAMES = (
    'transfer_percentile_mean_valid',
    'transfer_percentile_out_in_gap_valid',
)
_AMOUNT_COLUMNS = (7, 8)


def _cutoff(value):
    if isinstance(value, bool):
        raise ValueError('feature cutoff must be a finite number')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('feature cutoff must be finite')
    return result


def _settings():
    mode = os.environ.get('BHTG_NODE_FEATURE_MODE', 'legacy15').strip().lower()
    if mode not in ('legacy15', 'behavior9'):
        raise ValueError('BHTG_NODE_FEATURE_MODE must be legacy15 or behavior9')
    if mode == 'legacy15':
        return mode, None, None
    try:
        window = float(os.environ.get('BHTG_NODE_FEATURE_WINDOW_SECONDS', '604800'))
        minimum = int(os.environ.get('BHTG_NODE_FEATURE_MIN_HISTORY', '5'))
    except (TypeError, ValueError):
        raise ValueError('invalid behavior9 window or minimum history') from None
    if not math.isfinite(window) or window <= 0 or minimum < 1:
        raise ValueError('behavior9 window and minimum history must be positive')
    return mode, window, minimum


def _nine_schema():
    from bhtg.node_features_nine import FEATURE_NAMES, PROTOCOL
    names = tuple(FEATURE_NAMES)
    if len(names) != 9 or len(set(names)) != 9:
        raise ValueError('behavior9 index must expose nine distinct feature names')
    if names[7:] != ('transfer_percentile_mean', 'transfer_percentile_out_in_gap'):
        raise ValueError('behavior9 amount-validity columns differ from the index schema')
    return str(PROTOCOL), names


def feature_spec():
    """Return the selected protocol and ordered model input names (15 or 11)."""
    mode, _, _ = _settings()
    if mode == 'legacy15':
        return LEGACY_FEATURE_PROTOCOL, tuple(LEGACY_FEATURE_NAMES)
    _, names = _nine_schema()
    return BEHAVIOR_FEATURE_PROTOCOL, names + MASK_FEATURE_NAMES


def _binding(value):
    if not isinstance(value, dict) or not value:
        raise ValueError('behavior9 requires a nonempty input binding')
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=False, allow_nan=False)
        frozen = json.loads(encoded)
    except (TypeError, ValueError):
        raise ValueError('behavior9 input binding must be finite JSON data') from None
    return frozen, hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def _legacy_provider(provider):
    # Keep state_dict() byte-for-byte structurally compatible with old callers.
    provider.mode = 'legacy15'
    provider.feature_names = tuple(LEGACY_FEATURE_NAMES)
    provider.input_dim = len(LEGACY_FEATURE_NAMES)
    return provider


class BehaviorNineFeatureProvider:
    """Point-in-time nine values plus two masks, with a frozen training scaler.

    ``node_ids`` uses the same {node_type: local_ids} convention as the legacy
    provider. Validity and sample counts are available separately; counts never
    enter model features. Graph type blocks follow ``g.ntypes`` exactly.
    """

    def __init__(self, g, train_cutoff, *, scaler_state=None, feature_inputs=None):
        mode, self.window_seconds, self.min_history = _settings()
        if mode != 'behavior9':
            raise ValueError('behavior9 provider requires BHTG_NODE_FEATURE_MODE=behavior9')
        from bhtg import config
        if (not getattr(config, 'FOUR_ACTIONS_SELECTED', False)
                or not getattr(config, 'AMOUNT_EXPERIMENT', False)
                or getattr(config, 'V3_BUNDLE_SELECTED', False)):
            raise ValueError('behavior9 requires a hash-bound four-actions amount experiment')
        from bhtg.node_feature_data import load_nine_feature_inputs
        from bhtg.node_features_nine import NineFeatureIndex

        self.graph, self.train_cutoff = g, _cutoff(train_cutoff)
        self.mode = mode
        self.behavior_protocol, self.raw_feature_names = _nine_schema()
        self.feature_names = self.raw_feature_names + MASK_FEATURE_NAMES
        self.input_dim = len(self.feature_names)
        self.raw_value_key = None
        self.ntypes = list(g.ntypes)
        self.offsets, self.node_layout, total = {}, [], 0
        for ntype in self.ntypes:
            count = int(g.num_nodes(ntype))
            self.offsets[ntype] = total
            self.node_layout.append([ntype, count])
            total += count
        self.num_nodes = total
        if feature_inputs is None:
            feature_inputs = load_nine_feature_inputs(g)
        events, is_token, is_pool, binding = feature_inputs
        if len(is_token) != total or len(is_pool) != total:
            raise ValueError('behavior9 role arrays do not match graph node layout')
        self.binding, self.binding_sha256 = _binding(binding)
        self.index = NineFeatureIndex(events, is_token, is_pool,
                                      window_seconds=self.window_seconds,
                                      min_history=self.min_history)
        self._cached_cutoff = None
        self._cached_snapshot = None
        self.mean, self.scale, self.fit_counts, self.fit_node_counts = {}, {}, {}, {}
        if scaler_state is None:
            self._fit_scaler()
        else:
            self._load_scaler(scaler_state)

    def _snapshot(self, cutoff, node_ids=None):
        cutoff = _cutoff(cutoff)
        if node_ids is None and self._cached_cutoff == cutoff:
            return self._cached_snapshot
        if node_ids is None:
            values, valid, counts = self.index.at(cutoff)
        else:
            values, valid, counts = self.index.at(cutoff, node_ids=node_ids.numpy())
        # The index owns immutable cached arrays. Read them without copying the
        # full graph; public raw/validity/count accessors return owned tensors.
        values = torch.as_tensor(values, dtype=torch.float64).detach().cpu()
        valid = torch.as_tensor(valid, dtype=torch.bool).detach().cpu()
        counts = torch.as_tensor(counts).detach().cpu()
        n_rows = self.num_nodes if node_ids is None else len(node_ids)
        if values.shape != (n_rows, 9) or valid.shape != values.shape:
            raise ValueError('behavior9 index returned an incompatible value/validity shape')
        if counts.shape != (n_rows, 2):
            raise ValueError('behavior9 index returned incompatible sample counts')
        if (not bool(torch.isfinite(values[valid]).all())
                or not bool(torch.isfinite(counts).all())
                or bool((counts < 0).any())
                or not bool((counts == counts.long()).all())):
            raise ValueError('behavior9 values/counts must be finite and counts nonnegative integers')
        # A missing value must never become -mean/std after transformation.
        values = torch.where(valid, values, torch.zeros_like(values))
        snapshot = values, valid, counts.long()
        if node_ids is None:
            self._cached_snapshot, self._cached_cutoff = snapshot, cutoff
        return snapshot

    def _type_slice(self, matrix, ntype, node_ids=None):
        start = self.offsets[ntype]
        tensor = matrix[start:start + self.graph.num_nodes(ntype)]
        if node_ids is not None:
            ids = torch.as_tensor(node_ids.get(ntype, []), dtype=torch.long).cpu()
            if ids.ndim != 1 or bool(((ids < 0) | (ids >= len(tensor))).any()):
                raise ValueError('%s feature node ID out of bounds' % ntype)
            tensor = tensor[ids]
        return tensor

    def _query(self, cutoff, node_ids=None):
        """Use a full cached snapshot or ask the index only for requested IDs."""
        cutoff = _cutoff(cutoff)
        if node_ids is None or self._cached_cutoff == cutoff:
            matrices = self._snapshot(cutoff)
            return tuple({nt: self._type_slice(x, nt, node_ids) for nt in self.ntypes}
                         for x in matrices)
        selections, sizes = [], {}
        for ntype in self.ntypes:
            ids = torch.as_tensor(node_ids.get(ntype, []), dtype=torch.long).cpu()
            if ids.ndim != 1 or bool(((ids < 0) | (ids >= self.graph.num_nodes(ntype))).any()):
                raise ValueError('%s feature node ID out of bounds' % ntype)
            selections.append(ids + self.offsets[ntype])
            sizes[ntype] = len(ids)
        selected = torch.cat(selections) if selections else torch.empty(0, dtype=torch.long)
        matrices = self._snapshot(cutoff, selected)
        results = []
        for matrix in matrices:
            blocks, start = {}, 0
            for ntype in self.ntypes:
                blocks[ntype] = matrix[start:start + sizes[ntype]]
                start += sizes[ntype]
            results.append(blocks)
        return tuple(results)

    def _fit_scaler(self):
        values, valid, _ = self._snapshot(self.train_cutoff)
        for ntype in self.ntypes:
            x = self._type_slice(values, ntype)
            available = self._type_slice(valid, ntype)
            active = (x[:, 0] + x[:, 1]) > 0
            self.fit_node_counts[ntype] = int(active.sum())
            mean = torch.zeros(9, dtype=torch.float64)
            scale = torch.ones(9, dtype=torch.float64)
            count = torch.zeros(9, dtype=torch.int64)
            for column in range(9):
                sample = x[active & available[:, column], column]
                count[column] = len(sample)
                if len(sample):
                    mean[column] = sample.mean()
                    deviation = sample.std(unbiased=False)
                    if deviation > 1e-8:
                        scale[column] = deviation
            self.mean[ntype], self.scale[ntype], self.fit_counts[ntype] = mean, scale, count

    def raw_features(self, cutoff, node_ids=None):
        """Return nine unscaled values; missing values are zero with separate validity."""
        values, _, _ = self._query(cutoff, node_ids)
        return {nt: x.clone() for nt, x in values.items()}

    def get_validity(self, cutoff, node_ids=None):
        """Return the nine per-value validity flags, before scaling."""
        _, valid, _ = self._query(cutoff, node_ids)
        return {nt: x.clone() for nt, x in valid.items()}

    def get_sample_counts(self, cutoff, node_ids=None):
        """Return mean-event and gap-token support counts, never model inputs."""
        _, _, counts = self._query(cutoff, node_ids)
        return {nt: x.clone() for nt, x in counts.items()}

    def get_features(self, cutoff, node_ids=None):
        """Return nine standardized numbers followed by two unscaled 0/1 masks."""
        values, valid, _ = self._query(cutoff, node_ids)
        result = {}
        for ntype in self.ntypes:
            x, available = values[ntype], valid[ntype]
            scaled = (x - self.mean[ntype]) / self.scale[ntype]
            scaled = torch.where(available, scaled, torch.zeros_like(scaled))
            result[ntype] = torch.cat((scaled, available[:, _AMOUNT_COLUMNS].double()), dim=1).float()
        return result

    def _metadata(self):
        return {
            'protocol': BEHAVIOR_FEATURE_PROTOCOL, 'mode': self.mode,
            'behavior_protocol': self.behavior_protocol, 'time_protocol': TIME_PROTOCOL,
            'scaler_protocol': SCALER_PROTOCOL,
            'feature_names': list(self.feature_names),
            'raw_feature_names': list(self.raw_feature_names),
            'mask_feature_names': list(MASK_FEATURE_NAMES),
            'train_cutoff': self.train_cutoff, 'scaler_fit_cutoff': self.train_cutoff,
            'window_seconds': self.window_seconds,
            'min_history': self.min_history, 'node_layout': self.node_layout,
            'binding': self.binding, 'binding_sha256': self.binding_sha256,
        }

    def state_dict(self):
        """Serialize source/layout/schema/settings and the frozen per-column scaler."""
        state = json.loads(json.dumps(self._metadata(), allow_nan=False))
        state.update(mean={nt: x.clone() for nt, x in self.mean.items()},
                     scale={nt: x.clone() for nt, x in self.scale.items()},
                     fit_counts={nt: x.clone() for nt, x in self.fit_counts.items()},
                     fit_node_counts=dict(self.fit_node_counts))
        return state

    def _load_scaler(self, state):
        if not isinstance(state, dict):
            raise ValueError('behavior9 scaler state must be a dictionary')
        for key, expected in self._metadata().items():
            if state.get(key) != expected:
                raise ValueError('behavior9 scaler %s mismatch; retraining is required' % key)
        for key in ('mean', 'scale', 'fit_counts', 'fit_node_counts'):
            if not isinstance(state.get(key), dict) or set(state[key]) != set(self.ntypes):
                raise ValueError('behavior9 scaler %s node types mismatch' % key)
        for ntype in self.ntypes:
            mean = torch.as_tensor(state['mean'][ntype], dtype=torch.float64).cpu()
            scale = torch.as_tensor(state['scale'][ntype], dtype=torch.float64).cpu()
            count = torch.as_tensor(state['fit_counts'][ntype]).cpu()
            nodes = state['fit_node_counts'][ntype]
            if (mean.shape != (9,) or scale.shape != (9,) or count.shape != (9,)
                    or not bool(torch.isfinite(mean).all() & torch.isfinite(scale).all())
                    or not bool((scale > 0).all()) or not bool(torch.isfinite(count).all())
                    or bool((count < 0).any()) or not bool((count == count.long()).all())
                    or isinstance(nodes, bool) or not isinstance(nodes, int)
                    or not 0 <= nodes <= self.graph.num_nodes(ntype)
                    or bool((count > nodes).any())):
                raise ValueError('invalid %s behavior9 scaler' % ntype)
            self.mean[ntype], self.scale[ntype] = mean.clone(), scale.clone()
            self.fit_counts[ntype], self.fit_node_counts[ntype] = count.long().clone(), nodes


def make_feature_provider(g, train_cutoff):
    """Create the selected feature provider, fitting only training history."""
    mode, _, _ = _settings()
    if mode == 'legacy15':
        return _legacy_provider(TemporalFeatureProvider(g, train_cutoff))
    return BehaviorNineFeatureProvider(g, train_cutoff)


def restore_feature_provider(g, state):
    """Restore only a scaler compatible with the explicitly selected feature mode."""
    if not isinstance(state, dict):
        raise ValueError('node feature state must be a dictionary')
    mode, _, _ = _settings()
    if mode == 'legacy15':
        if state.get('protocol') != LEGACY_FEATURE_PROTOCOL:
            raise ValueError('node feature checkpoint is not legacy15; select its matching mode')
        return _legacy_provider(TemporalFeatureProvider.from_state_dict(g, state))
    if state.get('protocol') != BEHAVIOR_FEATURE_PROTOCOL or state.get('mode') != 'behavior9':
        raise ValueError('node feature checkpoint is not behavior9; retraining is required')
    return BehaviorNineFeatureProvider(g, state['train_cutoff'], scaler_state=state)
