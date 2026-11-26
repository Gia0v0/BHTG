"""Nine causal node statistics over canonical four-action records.

The caller supplies dense global node IDs (including the node-type offsets)
and one row per canonical source event.  Transfer graph legs are reconstructed
only for the five retained structural statistics.  Amount statistics use the
economic sender/recipient, never the synthetic token-hub legs.

All time bounds are exclusive at the observation time.  Window statistics use
``[cutoff - window_seconds, cutoff)``.  An event's amount percentile is computed
once against *all* known transfers of its token at strictly earlier timestamps;
concurrent events never become each other's reference observations.
"""
from collections import OrderedDict
import math

import numpy as np


PROTOCOL = 'causal_node_features_nine_v1'
FEATURE_PROTOCOL = PROTOCOL
FEATURE_NAMES = (
    'in_degree', 'out_degree', 'unique_neighbour_count', 'pool_in_degree',
    'burn_in_degree', 'mint_role_count', 'canonical_activity_rate_per_day',
    'transfer_percentile_mean', 'transfer_percentile_out_in_gap',
)
AMOUNT_FEATURE_COLUMNS = (7, 8)
_REQUIRED = ('ts', 'op', 'token', 'src', 'dst', 'amount_z', 'amount_known')
_KEY_DTYPE = np.dtype([('group', np.int64), ('time', np.float64)])


def _integer_column(values, name):
    values = np.asarray(values)
    if values.dtype.kind not in 'iu' or values.dtype.kind == 'b':
        raise ValueError('%s must contain integer IDs' % name)
    if values.dtype.kind == 'u' and values.size and np.max(values) > np.iinfo(np.int64).max:
        raise ValueError('%s exceeds int64' % name)
    return values.astype(np.int64, copy=False)


def _boolean_array(values, name):
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError('%s must be a one-dimensional boolean array' % name)
    if not values.size:
        return np.empty(0, dtype=bool)
    if values.dtype.kind not in 'biu':
        raise ValueError('%s must be a one-dimensional boolean array' % name)
    if values.size and np.any((values != 0) & (values != 1)):
        raise ValueError('%s must contain boolean values' % name)
    return values.astype(bool, copy=True)


class _TimeSeries:
    """CSR-like grouped event histories with vectorized prefix queries.

    A query performs binary searches in indexed histories; it does not iterate
    over addresses and scan their full event history.  Prefix sums are only
    stored for weighted series (the amount percentiles).
    """

    def __init__(self, groups, times, n_groups, weights=None):
        groups = np.asarray(groups, dtype=np.int64)
        times = np.asarray(times, dtype=np.float64)
        self.n_groups = n_groups
        order = np.lexsort((times, groups))
        groups, times = groups[order], times[order]
        self.keys = np.empty(len(groups), dtype=_KEY_DTYPE)
        self.keys['group'], self.keys['time'] = groups, times
        sizes = np.bincount(groups, minlength=n_groups)
        self.offsets = np.r_[0, np.cumsum(sizes, dtype=np.int64)]
        self.query_keys = np.empty(n_groups, dtype=_KEY_DTYPE)
        self.query_keys['group'] = np.arange(n_groups, dtype=np.int64)
        self.prefix = None
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)[order]
            # Cumulative sums restart at each group to avoid cancellation
            # between unrelated tokens/addresses with very different counts.
            self.prefix = np.zeros(len(weights) + 1, dtype=np.float64)
            if len(weights):
                starts = self.offsets[:-1][sizes > 0]
                ends = self.offsets[1:][sizes > 0]
                for start, end in zip(starts, ends):
                    self.prefix[start + 1:end + 1] = np.cumsum(weights[start:end])

    def before(self, cutoff, ids=None):
        if ids is None:
            queries = self.query_keys
            offsets = self.offsets[:-1]
        else:
            ids = np.asarray(ids, dtype=np.int64)
            queries = np.empty(len(ids), dtype=_KEY_DTYPE)
            queries['group'] = ids
            offsets = self.offsets[ids]
        queries['time'] = cutoff
        positions = np.searchsorted(self.keys, queries, side='left')
        counts = positions - offsets
        if self.prefix is None:
            return counts
        sums = np.zeros(len(counts), dtype=np.float64)
        nonempty = counts > 0
        sums[nonempty] = self.prefix[positions[nonempty]]
        return counts, sums

    def window(self, lower, upper, ids=None):
        right = self.before(upper, ids)
        left = self.before(lower, ids)
        if self.prefix is None:
            return right - left
        return right[0] - left[0], right[1] - left[1]


def _strict_transfer_percentiles(times, tokens, amounts, minimum):
    """Return midrank percentiles with token-local Fenwick count trees."""
    ranks = np.full(len(times), np.nan, dtype=np.float64)
    order = np.lexsort((times, tokens))
    ordered_tokens = tokens[order]
    token_starts = np.r_[0, np.flatnonzero(ordered_tokens[1:] != ordered_tokens[:-1]) + 1,
                         len(order)]
    for token_start, token_end in zip(token_starts[:-1], token_starts[1:]):
        rows = order[token_start:token_end]
        if not len(rows):
            continue
        token_times, token_amounts = times[rows], amounts[rows]
        levels, compressed = np.unique(token_amounts, return_inverse=True)
        tree = np.zeros(len(levels) + 1, dtype=np.int64)
        frequencies = np.zeros(len(levels), dtype=np.int64)
        time_starts = np.r_[0, np.flatnonzero(token_times[1:] != token_times[:-1]) + 1,
                            len(rows)]
        past_count = 0
        for start, end in zip(time_starts[:-1], time_starts[1:]):
            if past_count >= minimum:
                for position in range(start, end):
                    level = int(compressed[position])
                    less, index = 0, level
                    while index:
                        less += int(tree[index])
                        index -= index & -index
                    ranks[rows[position]] = (less + .5 * frequencies[level]) / past_count
            # Insert the whole timestamp only after all its ranks are read.
            levels_at_time, repetitions = np.unique(compressed[start:end], return_counts=True)
            for level, repetitions_at_level in zip(levels_at_time, repetitions):
                level, repetitions_at_level = int(level), int(repetitions_at_level)
                frequencies[level] += repetitions_at_level
                index = level + 1
                while index < len(tree):
                    tree[index] += repetitions_at_level
                    index += index & -index
            past_count += end - start
    return ranks


class NineFeatureIndex:
    """Indexed raw features for arbitrary historical observation cutoffs.

    ``at`` returns ``(values, valid, counts)``.  Values are float64 ``[N, 9]``;
    valid is boolean with the same shape.  The seven count/rate columns are
    valid even when zero.  The amount mean is valid only with at least one
    ranked window observation.  The gap is invalid for target-token nodes and
    for addresses without a token having both incoming and outgoing ranked
    window observations.  Counts ``[N, 2]`` hold the number of ranked mean
    observations and the number of common valid gap tokens, respectively.

    Per-token means are averaged equally across tokens.  The gap is the equal
    mean of (outgoing percentile mean - incoming percentile mean) over common
    valid tokens, not a difference of independently pooled token sets.
    """

    def __init__(self, events, is_token, is_pool, window_seconds=604800, min_history=5):
        self.is_token = _boolean_array(is_token, 'is_token')
        self.is_pool = _boolean_array(is_pool, 'is_pool')
        if self.is_token.shape != self.is_pool.shape:
            raise ValueError('node role masks must have the same length')
        self.n_nodes = len(self.is_token)
        if isinstance(window_seconds, (bool, np.bool_)):
            raise ValueError('window_seconds must be positive and finite')
        self.window_seconds = float(window_seconds)
        if not math.isfinite(self.window_seconds) or self.window_seconds <= 0:
            raise ValueError('window_seconds must be positive and finite')
        if (isinstance(min_history, (bool, np.bool_)) or not isinstance(min_history, (int, np.integer))
                or min_history < 1):
            raise ValueError('min_history must be a positive integer')
        self.min_history = int(min_history)
        events = np.asarray(events)
        if events.ndim != 1 or events.dtype.names is None or set(_REQUIRED) - set(events.dtype.names):
            raise ValueError('events require canonical ts/op/token/src/dst/amount_z/amount_known columns')
        times = np.asarray(events['ts'], dtype=np.float64)
        if not np.isfinite(times).all():
            raise ValueError('event times must be finite')
        operations = _integer_column(events['op'], 'op')
        tokens = _integer_column(events['token'], 'token')
        sources = _integer_column(events['src'], 'src')
        destinations = _integer_column(events['dst'], 'dst')
        for name, ids in (('token', tokens), ('src', sources), ('dst', destinations)):
            if ids.size and np.any((ids < 0) | (ids >= self.n_nodes)):
                raise ValueError('%s outside dense global node IDs' % name)
        if np.any((operations < 0) | (operations > 3)):
            raise ValueError('op must be 0 transfer, 1 mint, 2 burn or 3 create')
        known = _boolean_array(events['amount_known'], 'amount_known')
        amounts = np.asarray(events['amount_z'], dtype=np.float64)
        if not np.isfinite(amounts[known]).all():
            raise ValueError('known amount_z must be finite')
        transfer, mint, burn = operations == 0, operations == 1, operations == 2

        # These are exactly the graph legs used by the legacy four-action graph.
        edge_src = np.concatenate((sources[transfer], tokens[transfer], sources[~transfer]))
        edge_dst = np.concatenate((tokens[transfer], destinations[transfer], destinations[~transfer]))
        edge_times = np.concatenate((times[transfer], times[transfer], times[~transfer]))
        pool_edges = self.is_pool[edge_src]
        self._in = _TimeSeries(edge_dst, edge_times, self.n_nodes)
        self._out = _TimeSeries(edge_src, edge_times, self.n_nodes)
        self._pool_in = _TimeSeries(edge_dst[pool_edges], edge_times[pool_edges], self.n_nodes)
        self._burn_in = _TimeSeries(destinations[burn], times[burn], self.n_nodes)

        # One unordered pair contributes once to each endpoint, including a
        # self-neighbour exactly once, matching the legacy unique-neighbour set.
        low, high = np.minimum(edge_src, edge_dst), np.maximum(edge_src, edge_dst)
        pair_order = np.lexsort((edge_times, high, low))
        low, high, pair_times = low[pair_order], high[pair_order], edge_times[pair_order]
        first = np.ones(len(low), dtype=bool)
        if len(low) > 1:
            first[1:] = (low[1:] != low[:-1]) | (high[1:] != high[:-1])
        low, high, pair_times = low[first], high[first], pair_times[first]
        different = low != high
        self._unique = _TimeSeries(np.r_[low, high[different]],
                                   np.r_[pair_times, pair_times[different]], self.n_nodes)

        token_mints = mint & self.is_token[tokens]
        recipient_mints = mint & ~self.is_token[destinations]
        self._mint_role = _TimeSeries(np.r_[tokens[token_mints], destinations[recipient_mints]],
                                      np.r_[times[token_mints], times[recipient_mints]], self.n_nodes)
        distinct_source = sources != tokens
        distinct_destination = (destinations != tokens) & (destinations != sources)
        self._activity = _TimeSeries(
            np.r_[tokens, sources[distinct_source], destinations[distinct_destination]],
            np.r_[times, times[distinct_source], times[distinct_destination]], self.n_nodes)

        ranked_rows = np.flatnonzero(transfer & known)
        ranks = _strict_transfer_percentiles(times[ranked_rows], tokens[ranked_rows],
                                             amounts[ranked_rows], self.min_history)
        usable = np.isfinite(ranks)
        ranked_rows, ranks = ranked_rows[usable], ranks[usable]
        rt, token, src, dst = (column[ranked_rows] for column in (times, tokens, sources, destinations))
        target = self.is_token[token]
        recipient = ~self.is_token[dst]
        sender = ~self.is_token[src]
        mean_nodes = np.r_[token[target], dst[recipient]]
        mean_tokens = np.r_[token[target], token[recipient]]
        mean_times = np.r_[rt[target], rt[recipient]]
        mean_ranks = np.r_[ranks[target], ranks[recipient]]
        out_nodes, out_tokens, out_times, out_ranks = src[sender], token[sender], rt[sender], ranks[sender]
        pairs = np.rec.fromarrays((np.r_[mean_nodes, out_nodes], np.r_[mean_tokens, out_tokens]),
                                  names=('node', 'token'))
        groups, inverse = np.unique(pairs, return_inverse=True)
        self._amount_nodes = np.asarray(groups['node'], dtype=np.int64)
        self._amount_offsets = np.r_[0, np.cumsum(
            np.bincount(self._amount_nodes, minlength=self.n_nodes), dtype=np.int64)]
        n_groups = len(groups)
        self._mean = _TimeSeries(inverse[:len(mean_nodes)], mean_times, n_groups, mean_ranks)
        self._send = _TimeSeries(inverse[len(mean_nodes):], out_times, n_groups, out_ranks)
        self._cache = OrderedDict()

    def at(self, cutoff, node_ids=None):
        """Return historical features, optionally for selected global node IDs.

        Subset queries preserve order and duplicates and touch only the selected
        node histories and their amount groups.  Only full-table results enter
        the two-snapshot cache, so P2 batch queries cannot grow it without bound.
        """
        if isinstance(cutoff, (bool, np.bool_)):
            raise ValueError('cutoff must be finite')
        cutoff = float(cutoff)
        if not math.isfinite(cutoff):
            raise ValueError('cutoff must be finite')
        requested = None
        if node_ids is not None:
            requested = np.asarray(node_ids)
            if requested.ndim != 1:
                raise ValueError('node_ids must be one-dimensional integer IDs')
            requested = (_integer_column(requested, 'node_ids') if requested.size
                         else np.empty(0, dtype=np.int64))
            if np.any((requested < 0) | (requested >= self.n_nodes)):
                raise ValueError('node_ids outside dense global node IDs')
        if cutoff in self._cache:
            self._cache.move_to_end(cutoff)
            cached = self._cache[cutoff]
            return cached if requested is None else tuple(array[requested] for array in cached)
        if requested is None:
            query_nodes, inverse, group_ids = None, None, None
            n_rows = self.n_nodes
            group_nodes = self._amount_nodes
        else:
            query_nodes, inverse = np.unique(requested, return_inverse=True)
            n_rows = len(query_nodes)
            starts = self._amount_offsets[query_nodes]
            lengths = self._amount_offsets[query_nodes + 1] - starts
            group_nodes = np.repeat(np.arange(n_rows, dtype=np.int64), lengths)
            group_bases = starts - np.r_[0, np.cumsum(lengths, dtype=np.int64)[:-1]]
            group_ids = np.repeat(group_bases, lengths) + np.arange(len(group_nodes), dtype=np.int64)
        lower = cutoff - self.window_seconds
        values = np.zeros((n_rows, len(FEATURE_NAMES)), dtype=np.float64)
        valid = np.ones(values.shape, dtype=bool)
        valid[:, 7:] = False
        counts = np.zeros((n_rows, 2), dtype=np.int64)
        for column, history in enumerate((self._in, self._out, self._unique,
                                           self._pool_in, self._burn_in, self._mint_role)):
            values[:, column] = history.before(cutoff, query_nodes)
        values[:, 6] = self._activity.window(lower, cutoff, query_nodes).astype(np.float64) * 86400. / self.window_seconds

        incoming_count, incoming_sum = self._mean.window(lower, cutoff, group_ids)
        outgoing_count, outgoing_sum = self._send.window(lower, cutoff, group_ids)
        incoming_valid, outgoing_valid = incoming_count > 0, outgoing_count > 0
        incoming_mean = np.divide(incoming_sum, incoming_count, out=np.zeros_like(incoming_sum),
                                  where=incoming_valid)
        outgoing_mean = np.divide(outgoing_sum, outgoing_count, out=np.zeros_like(outgoing_sum),
                                  where=outgoing_valid)
        valid_token_counts = np.bincount(group_nodes[incoming_valid], minlength=n_rows)
        mean_sums = np.bincount(group_nodes[incoming_valid], weights=incoming_mean[incoming_valid],
                                minlength=n_rows)
        valid[:, 7] = valid_token_counts > 0
        values[:, 7] = np.divide(mean_sums, valid_token_counts, out=np.zeros(n_rows),
                                 where=valid[:, 7])
        np.add.at(counts[:, 0], group_nodes, incoming_count)
        token_nodes = self.is_token if query_nodes is None else self.is_token[query_nodes]
        common = incoming_valid & outgoing_valid & ~token_nodes[group_nodes]
        counts[:, 1] = np.bincount(group_nodes[common], minlength=n_rows)
        gap_sums = np.bincount(group_nodes[common],
                               weights=(outgoing_mean - incoming_mean)[common], minlength=n_rows)
        valid[:, 8] = counts[:, 1] > 0
        values[:, 8] = np.divide(gap_sums, counts[:, 1], out=np.zeros(n_rows), where=valid[:, 8])
        result = values, valid, counts
        if requested is not None:
            return tuple(array[inverse] for array in result)
        # Train and validation often alternate between two fixed snapshots.
        # Bound the cache instead of retaining an N-by-9 matrix per P2 batch.
        self._cache[cutoff] = result
        if len(self._cache) > 2:
            self._cache.popitem(last=False)
        return result
