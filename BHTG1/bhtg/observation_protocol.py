"""Label-blind existence and activity at an exclusive observation cutoff.

Identity registries describe which node is which, not when a project exists.
Exact chain creation times take precedence over archived create edges. Missing
creation metadata may be replaced by strictly visible incident activity, which
proves existence without inventing an earlier deployment date. Nodes with no
such evidence are excluded; future activity never proves past existence.

``has_history`` counts all incident graph edges, including create. The separate
``has_activity_history`` excludes create and defines active/cold-start groups.
"""
import csv
import hashlib
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from bhtg.temporal_time import conservative_timestamps, TIME_PROTOCOL


OBSERVATION_PROTOCOL = 'exclusive_cutoff_creation_and_incident_activity_v1'
OBSERVATION_GROUPS = (
    'active_history', 'cold_start', 'not_yet_observable',
    'unknown_existence', 'inconsistent_metadata',
)


def _finite_time(value, description):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('%s must be finite' % description)
    return result


def read_creator_times(path):
    """Read exact chain seconds; conflicting duplicate addresses are errors."""
    times, records = {}, {}
    with open(path, encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'token_address', 'timestamp'}.issubset(reader.fieldnames or []):
            raise ValueError('creator CSV requires token_address and timestamp')
        for row in reader:
            address = str(row['token_address']).strip().lower()
            if not address:
                raise ValueError('creator address must not be empty')
            time = _finite_time(row['timestamp'], 'chain creation timestamp')
            record = (time,) + tuple(str(row.get(field, '')).strip().lower()
                                     for field in ('creator_address', 'tx_hash', 'block'))
            if address in records and records[address] != record:
                raise ValueError('conflicting chain creation times for %s' % address)
            times[address] = time
            records[address] = record
    return times


def load_creator_times(path):
    """Return (exact-time mapping, source metadata); missing permits fallback."""
    path = Path(path).resolve()
    metadata = {'source_path': str(path), 'source_sha256': None,
                'source_status': 'missing', 'unique_tokens': 0}
    if not path.is_file():
        return {}, metadata
    times = read_creator_times(path)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    metadata.update({'source_sha256': digest.hexdigest(),
                     'source_status': 'present', 'unique_tokens': len(times)})
    return times, metadata


def _node_ids(values, num_nodes, description):
    ids = torch.as_tensor(values).detach().cpu()
    if ids.ndim != 1:
        raise ValueError('%s must be a one-dimensional ID sequence' % description)
    if ids.numel() and (ids.dtype == torch.bool or ids.is_floating_point()
                        or ids.is_complex()):
        raise ValueError('%s must contain integer local IDs' % description)
    ids = ids.to(dtype=torch.long)
    if bool(((ids < 0) | (ids >= num_nodes)).any()):
        raise ValueError('%s is outside its node type' % description)
    return ids


class ObservationIndex:
    """Immutable typed-node observation evidence from an immutable graph.

    ``node_addresses`` maps node type to {local ID: address}; it may also contain
    address lists in local-ID order. ``creator_times`` maps token address to
    exact chain creation seconds, usually returned by ``read_creator_times``.
    Both are optional. Chain token creation metadata applies only to contract
    nodes, never to an EOA that happens to be the deployer of a token.

    Results preserve the requested local-ID order and carry per-node reasons
    and provenance. Activity evidence uses every incoming and outgoing relation
    rather than the sampled neighbourhood or any graph/node label.
    """

    def __init__(self, g, node_addresses=None, creator_times=None):
        self.num_nodes = {ntype: g.num_nodes(ntype) for ntype in g.ntypes}
        self.first_history = {ntype: np.full(size, np.inf)
                              for ntype, size in self.num_nodes.items()}
        self.first_activity = {ntype: np.full(size, np.inf)
                               for ntype, size in self.num_nodes.items()}
        self.graph_creation = {ntype: np.full(size, np.inf)
                               for ntype, size in self.num_nodes.items()}
        self.chain_creation = {ntype: np.full(size, np.inf)
                               for ntype, size in self.num_nodes.items()}
        for canonical in g.canonical_etypes:
            src_type, relation, dst_type = canonical
            source, destination = g.edges(etype=canonical)
            source = source.detach().cpu().numpy()
            destination = destination.detach().cpu().numpy()
            if not len(source):
                continue
            if 'timestamp' not in g.edges[canonical].data:
                raise ValueError('incident edges require timestamps: %s' % (canonical,))
            times = conservative_timestamps(
                g.edges[canonical].data['timestamp']).detach().cpu().numpy()
            np.minimum.at(self.first_history[src_type], source, times)
            np.minimum.at(self.first_history[dst_type], destination, times)
            if relation == 'create':
                # The created object is the destination, not its creator.
                np.minimum.at(self.graph_creation[dst_type], destination, times)
            else:
                np.minimum.at(self.first_activity[src_type], source, times)
                np.minimum.at(self.first_activity[dst_type], destination, times)

        creators = {}
        for address, value in (creator_times or {}).items():
            normalized = str(address).strip().lower()
            if not normalized:
                raise ValueError('creator address must not be empty')
            time = _finite_time(value, 'chain creation timestamp')
            if normalized in creators and creators[normalized] != time:
                raise ValueError('conflicting chain creation times for %s' % normalized)
            creators[normalized] = time
        for ntype, addresses in (node_addresses or {}).items():
            if ntype not in self.num_nodes:
                raise ValueError('unknown address node type: %s' % ntype)
            pairs = addresses.items() if hasattr(addresses, 'items') else enumerate(addresses)
            for node_id, address in pairs:
                ids = _node_ids([node_id], self.num_nodes[ntype], '%s address ID' % ntype)
                address = str(address).strip().lower()
                if ntype == 'contract' and address in creators:
                    self.chain_creation[ntype][int(ids[0])] = creators[address]

    def query_batch(self, ntype, node_ids, cutoff):
        """Classify existence and cold-start status strictly before ``cutoff``.

        An exact chain creation date later than an already visible activity is
        contradictory metadata and is excluded explicitly. Without chain data,
        earlier visible activity can establish existence even if an archived
        create timestamp has a later conservative visibility bound. This is an
        evidence fallback, not a claim to know the deployment time.
        """
        if ntype not in self.num_nodes:
            raise ValueError('unknown query node type: %s' % ntype)
        cutoff = _finite_time(cutoff, 'observation cutoff')
        ids = _node_ids(node_ids, self.num_nodes[ntype], '%s query IDs' % ntype)
        index = ids.numpy()
        history = self.first_history[ntype][index]
        activity = self.first_activity[ntype][index]
        chain = self.chain_creation[ntype][index]
        create = self.graph_creation[ntype][index]
        has_history, has_activity = history < cutoff, activity < cutoff
        eligible = np.zeros(len(ids), dtype=bool)
        existence = np.full(len(ids), np.inf)
        creation = np.where(np.isfinite(chain), chain, create)
        groups, reasons, provenance = [], [], []
        for row in range(len(ids)):
            if np.isfinite(chain[row]):
                existence[row] = chain[row]
                source = 'chain_creators_exact'
                if has_activity[row] and activity[row] < chain[row]:
                    group, reason = 'inconsistent_metadata', 'activity_precedes_chain_creation'
                elif chain[row] < cutoff:
                    eligible[row] = True
                    group = 'active_history' if has_activity[row] else 'cold_start'
                    reason = 'chain_creation_before_cutoff'
                else:
                    group, reason = 'not_yet_observable', 'chain_creation_at_or_after_cutoff'
            elif np.isfinite(create[row]) and create[row] < cutoff:
                existence[row] = create[row]
                eligible[row] = True
                group = 'active_history' if has_activity[row] else 'cold_start'
                reason, source = 'graph_create_before_cutoff', 'graph_create_conservative_upper_bound'
            elif has_history[row]:
                existence[row] = activity[row] if has_activity[row] else history[row]
                eligible[row] = True
                group = 'active_history' if has_activity[row] else 'cold_start'
                reason = 'visible_activity_proves_existence' if has_activity[row] else 'visible_event_proves_existence'
                source = ('first_incident_activity_conservative_upper_bound' if has_activity[row]
                          else 'first_incident_event_conservative_upper_bound')
            elif np.isfinite(create[row]):
                existence[row] = create[row]
                group, reason, source = ('not_yet_observable', 'graph_create_not_strictly_before_cutoff',
                                         'graph_create_conservative_upper_bound')
            else:
                group, reason, source = 'unknown_existence', 'no_visible_existence_evidence', 'missing'
            groups.append(group)
            reasons.append(reason)
            provenance.append(source)
        return {
            'protocol': OBSERVATION_PROTOCOL, 'time_protocol': TIME_PROTOCOL,
            'node_type': ntype, 'node_ids': ids.clone(), 'cutoff': cutoff,
            'eligible': torch.from_numpy(eligible),
            'has_history': torch.from_numpy(has_history),
            'has_activity_history': torch.from_numpy(has_activity),
            'creation_time': torch.from_numpy(creation.copy()),
            'existence_time': torch.from_numpy(existence),
            'first_history_time': [float(value) if math.isfinite(value) else None
                                   for value in history],
            'first_activity_time': [float(value) if math.isfinite(value) else None
                                    for value in activity],
            'graph_creation_time': [float(value) if math.isfinite(value) else None
                                    for value in create],
            'group': groups, 'reason': reasons, 'provenance': provenance,
        }

    def query(self, ntype, node_ids, cutoff):
        """Return JSON-safe per-node records in the requested ID order."""
        result = self.query_batch(ntype, node_ids, cutoff)
        records = []
        for row, node_id in enumerate(result['node_ids'].tolist()):
            creation, existence = (float(result[name][row])
                                   for name in ('creation_time', 'existence_time'))
            records.append({
                'node_id': node_id, 'node_type': ntype,
                'cutoff': result['cutoff'], 'protocol': OBSERVATION_PROTOCOL,
                'eligible': bool(result['eligible'][row]),
                'has_history': bool(result['has_history'][row]),
                'has_activity_history': bool(result['has_activity_history'][row]),
                'group': result['group'][row], 'reason': result['reason'][row],
                'provenance': result['provenance'][row],
                'creation_time': creation if math.isfinite(creation) else None,
                'existence_time': existence if math.isfinite(existence) else None,
                'first_history_time': result['first_history_time'][row],
                'first_activity_time': result['first_activity_time'][row],
                'graph_creation_time': result['graph_creation_time'][row],
            })
        return records


def observation_groups(result):
    """Return local-ID partitions without altering the original split order."""
    if isinstance(result, list):
        return {group: torch.tensor([row['node_id'] for row in result if row['group'] == group],
                                     dtype=torch.long) for group in OBSERVATION_GROUPS}
    return {group: result['node_ids'][torch.tensor(
        [value == group for value in result['group']], dtype=torch.bool)]
            for group in OBSERVATION_GROUPS}


def observation_summary(result):
    """Compact, serializable counts for saved manifests and user reports."""
    if isinstance(result, list):
        return {'protocol': OBSERVATION_PROTOCOL,
                'cutoff': result[0]['cutoff'] if result else None,
                'node_type': result[0]['node_type'] if result else None,
                'total': len(result), 'eligible': sum(row['eligible'] for row in result),
                'groups': {group: sum(row['group'] == group for row in result)
                           for group in OBSERVATION_GROUPS},
                'reasons': dict(Counter(row['reason'] for row in result)),
                'provenance': dict(Counter(row['provenance'] for row in result))}
    return {'protocol': result['protocol'], 'cutoff': result['cutoff'],
            'node_type': result['node_type'], 'total': len(result['node_ids']),
            'eligible': int(result['eligible'].sum()),
            'groups': {group: result['group'].count(group) for group in OBSERVATION_GROUPS},
            'reasons': dict(Counter(result['reason'])),
            'provenance': dict(Counter(result['provenance']))}
