"""Bind authenticated, train-snapshot pool identities to a static action graph.

This is a retrospective identity overlay.  A train-snapshot verification does
not establish the role at every earlier block, and timestamp-only actions do
not establish transaction order within the pool's creation second.
"""
import hashlib
import json
import math
import os
import re
from collections import Counter


SCHEMA = 'bhtg_train_pool_overlay_v1'
AUTHENTICATION = 'rpc_receipt_and_canonical_header'
_ADDRESS = re.compile(r'0x[0-9a-f]{40}')
_HASH = re.compile(r'0x[0-9a-f]{64}')
_SHA = re.compile(r'[0-9a-f]{64}')


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('pool overlay requires a nonnegative integer: ' + field)
    return value


def _matches(value, pattern, field):
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError('pool overlay has a noncanonical ' + field)
    return value


def _validate_train_role_records(path, pools, train_cutoff):
    """Cross-check selected identities while retaining only address keys.

    Hashing a role table binds its bytes; this second check binds each overlay
    claim to the corresponding authenticated train-snapshot record in it.
    """
    expected = {pool['address']: pool for pool in pools}
    matched = set()
    fields = ('candidate_identity_sha256', 'creation_block_number',
              'creation_block_timestamp', 'creation_log_authentication',
              'verification_block_number', 'verification_block_hash')
    optional_fields = ('creation_block_hash', 'creation_transaction_hash',
                       'creation_transaction_index', 'creation_log_index',
                       'factory', 'protocol', 'token0', 'token1')
    with open(path, encoding='utf-8-sig') as source:
        for line in source:
            if not line.strip():
                continue
            role = json.loads(line)
            if not isinstance(role, dict):
                raise ValueError('pool snapshot role must be a JSON object')
            address = role.get('address')
            if role.get('snapshot') != 'train' or address not in expected:
                continue
            if address in matched:
                raise ValueError('duplicate train snapshot role for pool overlay address')
            matched.add(address)
            pool = expected[address]
            if role.get('status') != 'verified' or role.get('pool_role') is not True:
                raise ValueError('pool overlay train role is not a verified pool')
            for field in ('creation_block_number', 'creation_block_timestamp', 'verification_block_number'):
                _integer(role.get(field), 'train role.' + field)
            for field in fields + tuple(field for field in optional_fields if field in pool):
                if role.get(field) != pool[field]:
                    raise ValueError('pool overlay/train role field mismatch: ' + field)
            if role.get('cutoff', train_cutoff) != train_cutoff:
                raise ValueError('pool overlay/train role cutoff mismatch')
            evidence = pool['evidence']
            if isinstance(evidence, dict):
                if ('address' in evidence and evidence['address'] != address) or (
                        'snapshot' in evidence and evidence['snapshot'] != 'train'):
                    raise ValueError('pool overlay evidence address/snapshot mismatch')
                if 'source_verification_record_sha256' in evidence:
                    proof_hash = _matches(evidence['source_verification_record_sha256'], _SHA,
                                          'source verification record SHA256')
                    if proof_hash != role.get('source_verification_record_sha256'):
                        raise ValueError('pool overlay/train role source verification record SHA256 mismatch')
    if set(expected) - matched:
        raise ValueError('pool overlay is missing a train snapshot role')
    return len(matched)


def load_train_pool_overlay(path, source_actions_sha256, train_cutoff):
    """Read and validate a source- and cutoff-bound pool identity sidecar.

    The caller must check ``sha256`` again before publication.  No chain calls,
    account secrets, or mutable scanner state are consumed here.
    """
    path = os.path.abspath(os.fspath(path))
    digest = file_sha256(path)
    with open(path, encoding='utf-8-sig') as source:
        overlay = json.load(source)
    if not isinstance(overlay, dict) or overlay.get('schema', overlay.get('schema_version')) != SCHEMA:
        raise ValueError('unsupported train pool overlay schema')
    if ('schema' in overlay and 'schema_version' in overlay and
            overlay['schema'] != overlay['schema_version']):
        raise ValueError('conflicting pool overlay schema fields')
    bound_hash = _matches(overlay.get('source_actions_sha256'), _SHA, 'source actions SHA256')
    if bound_hash != source_actions_sha256:
        raise ValueError('pool overlay source actions SHA256 mismatch')
    cutoff = overlay.get('train_cutoff')
    if (isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)) or
            not math.isfinite(cutoff) or cutoff != train_cutoff):
        raise ValueError('pool overlay training cutoff mismatch')
    snapshot = overlay.get('snapshot')
    if not isinstance(snapshot, dict) or snapshot.get('name') != 'train':
        raise ValueError('pool overlay requires the train snapshot')
    number = _integer(snapshot.get('number'), 'snapshot.number')
    block_hash = _matches(snapshot.get('hash'), _HASH, 'snapshot hash')
    timestamp = _integer(snapshot.get('timestamp'), 'snapshot.timestamp')
    if (snapshot.get('cutoff') != cutoff or isinstance(snapshot.get('cutoff'), bool) or
            timestamp >= cutoff):
        raise ValueError('pool overlay train snapshot/cutoff mismatch')
    pools = overlay.get('pools')
    if not isinstance(pools, list):
        raise ValueError('pool overlay pools must be a list')
    seen = set()
    for pool in pools:
        if not isinstance(pool, dict):
            raise ValueError('pool overlay pool entry must be an object')
        address = _matches(pool.get('address'), _ADDRESS, 'pool address')
        if address == '0x' + '0' * 40 or address in seen:
            raise ValueError('zero or duplicate pool overlay address')
        seen.add(address)
        birth_block = _integer(pool.get('creation_block_number'), 'creation_block_number')
        birth_time = _integer(pool.get('creation_block_timestamp'), 'creation_block_timestamp')
        if birth_block > number or birth_time > timestamp or birth_time >= cutoff:
            raise ValueError('pool creation is later than the verified train snapshot/cutoff')
        if pool.get('creation_log_authentication') != AUTHENTICATION:
            raise ValueError('pool creation lacks receipt and canonical header authentication')
        if (pool.get('verified_snapshot') != 'train' or
                _integer(pool.get('verification_block_number'), 'verification_block_number') != number or
                _matches(pool.get('verification_block_hash'), _HASH, 'verification block hash') != block_hash):
            raise ValueError('pool verification does not match the train snapshot block/hash')
        _matches(pool.get('candidate_identity_sha256'), _SHA, 'candidate identity SHA256')
        evidence = pool.get('evidence')
        if not isinstance(evidence, (dict, list)) or not evidence:
            raise ValueError('pool overlay requires nonempty evidence references')
    role_table = overlay.get('pool_roles_by_snapshot')
    if role_table is not None:
        if not isinstance(role_table, dict) or not isinstance(role_table.get('path'), str) or not role_table['path']:
            raise ValueError('pool overlay snapshot roles require a path/hash binding')
        role_hash = _matches(role_table.get('sha256'), _SHA, 'snapshot roles SHA256')
        role_path = role_table['path']
        if not os.path.isabs(role_path):
            role_path = os.path.join(os.path.dirname(path), role_path)
        role_path = os.path.abspath(role_path)
        if file_sha256(role_path) != role_hash:
            raise ValueError('pool overlay snapshot roles SHA256 mismatch')
        verified_records = _validate_train_role_records(role_path, pools, cutoff)
        if file_sha256(role_path) != role_hash:
            raise ValueError('pool overlay snapshot roles changed while being read')
        role_table = dict(role_table, path=role_path, validated_train_pool_records=verified_records)
    if file_sha256(path) != digest:
        raise ValueError('pool overlay changed while being read')
    return {'path': path, 'sha256': digest, 'schema': SCHEMA,
            'source_actions_sha256': bound_hash, 'train_cutoff': cutoff,
            'snapshot': dict(snapshot), 'pools': pools, 'pool_roles_by_snapshot': role_table}


def select_train_pool_overlay(overlay, frame, target_tokens, internal_factories):
    """Select static pool nodes without changing actions or supervision roles.

    Every occurrence as src, dst, or target token contributes to the temporal
    lower-bound check. Equal-second observations are retained and marked unknown
    in order; source flags remain available and are counted without inferring
    a native creation/action ordering.
    """
    earliest = {}
    for column in ('src_addr', 'dst_addr', 'token_addr'):
        for address, timestamp in frame.groupby(column, sort=False)['timestamp'].min().items():
            earliest[address] = min(int(timestamp), earliest.get(address, int(timestamp)))
    accepted, records, reasons = {}, [], Counter()
    same_second_addresses = {}
    same_second_native = Counter()
    for pool in overlay['pools']:
        address = pool['address']
        birth_time = pool['creation_block_timestamp']
        observed = earliest.get(address)
        reason = None
        if address in target_tokens:
            reason = 'target_token_type_conflict'
        elif address in internal_factories:
            reason = 'internal_factory_type_conflict'
        elif observed is None:
            reason = 'outside_graph'
        elif observed < birth_time:
            reason = 'action_before_pool_creation'
        equal = observed == birth_time
        record = {'address': address, 'decision': 'excluded' if reason else 'accepted',
                  'reason': reason or 'authenticated_train_pool',
                  'earliest_action_timestamp': observed,
                  'creation_block_number': pool['creation_block_number'],
                  'creation_block_timestamp': birth_time,
                  'pool_role': True,
                  'pool_role_verification_scope': 'train_snapshot_retrospective_identity',
                  'same_second_order_unknown': equal,
                  'event_order': 'timestamp_only',
                  'candidate_identity_sha256': pool['candidate_identity_sha256']}
        if reason:
            reasons[reason] += 1
        else:
            accepted[address] = pool
            if equal:
                same_second_addresses[address] = birth_time
        records.append(record)
    # One vectorized pass per endpoint column, rather than one graph scan per
    # pool (thousands of pools can have their first action in the birth second).
    same_second_sources = set()
    for column in ('src_addr', 'dst_addr', 'token_addr'):
        relevant = frame.loc[frame[column].isin(same_second_addresses),
                             [column, 'timestamp', 'source_row_id', 'native_order_available']]
        equal_rows = relevant.loc[relevant['timestamp'].eq(relevant[column].map(same_second_addresses)),
                                  ['source_row_id', 'native_order_available']].drop_duplicates()
        for source_id, native in equal_rows.itertuples(index=False, name=None):
            if source_id not in same_second_sources:
                same_second_native[str(native)] += 1
                same_second_sources.add(source_id)
    audit = {'requested_count': len(overlay['pools']), 'accepted_count': len(accepted),
             'excluded_count': len(records) - len(accepted), 'excluded_by_reason': dict(reasons),
             'same_second_order_unknown_pool_count': sum(
                 row['decision'] == 'accepted' and row['same_second_order_unknown'] for row in records),
             'same_second_source_actions_count': len(same_second_sources),
             'same_second_source_native_order_available': dict(same_second_native),
             'time_lower_bound_policy': 'earliest_selected_action_timestamp_gte_authenticated_creation_timestamp',
             'identity_scope': 'authenticated_creation_and_train_snapshot_retrospective_identity',
             'earlier_block_pool_identity_fully_verified': False,
             'records': records}
    return accepted, audit
