"""Recover canonical four-action rows from the bound amount graph's edge stream."""
import json
from pathlib import Path

import numpy as np


OPERATIONS = ('transfer', 'mint', 'burn', 'create')
EVENT_DTYPE = np.dtype([
    ('ts', 'f8'), ('op', 'i1'), ('token', 'i8'), ('src', 'i8'), ('dst', 'i8'),
    ('amount_z', 'f4'), ('amount_known', '?'),
])


def _canonical_events(g, stream, vocabulary, action_counts):
    meta = stream['_meta']
    offsets = {}
    node_count = 0
    for nt in g.ntypes:
        offsets[nt] = node_count
        node_count += g.num_nodes(nt)
    gids = np.concatenate([g.nodes[nt].data['global_id'].cpu().numpy() for nt in g.ntypes])
    gid_to_dense = np.argsort(gids)
    if not np.array_equal(gids[gid_to_dense], np.arange(node_count)):
        raise ValueError('invalid global node mapping in behavior9 graph')
    is_pool = np.zeros(node_count, dtype=bool)
    is_pool[offsets['pool']:offsets['pool'] + g.num_nodes('pool')] = True
    type_offsets = np.asarray([offsets[nt] for nt in meta['node_types']])
    src = type_offsets[stream['src_ntype'].numpy()] + stream['src_lid'].numpy()
    dst = type_offsets[stream['dst_ntype'].numpy()] + stream['dst_lid'].numpy()
    fields = stream['edge_attributes']
    for key in ('leg_role_code', 'source_row_id_code', 'token_gid', 'amount_z', 'amount_known'):
        if key not in fields or not stream['edge_attribute_present'][key].all().item():
            raise ValueError('missing canonical action field: ' + key)
    row_id = fields['source_row_id_code'].numpy()
    token_gid = fields['token_gid'].numpy()
    if np.any(token_gid < 0) or np.any(token_gid >= node_count):
        raise ValueError('invalid canonical action token ID')
    token = gid_to_dense[token_gid]
    leg_names = vocabulary['leg_role']
    leg_codes = fields['leg_role_code'].numpy()
    if np.any(leg_codes < 0) or np.any(leg_codes >= len(leg_names)):
        raise ValueError('invalid canonical action leg code')

    role_names = ('sender_to_token', 'token_to_receiver', 'mint', 'burn', 'create')
    role_ops = np.array([0, 0, 1, 2, 3])
    role = np.asarray([role_names.index(name) for name in leg_names], dtype=int)[leg_codes]
    op = np.asarray([OPERATIONS.index(name) for name in meta['relations']])[stream['etype_idx'].numpy()]
    if not np.array_equal(role_ops[role], op):
        raise ValueError('canonical action relation differs from its leg role')
    to_token = np.isin(role, (0, 3, 4))
    if np.any(dst[to_token] != token[to_token]) or np.any(src[~to_token] != token[~to_token]):
        raise ValueError('canonical action leg has wrong token endpoint')

    # A transfer has a sender leg and a receiver leg. Every other operation
    # contributes exactly one row. Identity also disambiguates self-transfers.
    primary = role != 1
    primary_ids = row_id[primary]
    count = len(vocabulary['source_row_id'])
    if not np.array_equal(np.sort(primary_ids), np.arange(count)):
        raise ValueError('missing or duplicate primary canonical action')
    events = np.empty(count, dtype=EVENT_DTYPE)
    for key, array in (('ts', stream['ts'].numpy()), ('op', op), ('src', src),
                       ('dst', dst), ('token', token), ('amount_z', fields['amount_z'].numpy()),
                       ('amount_known', fields['amount_known'].numpy())):
        events[key][primary_ids] = array[primary]
    second = ~primary
    ids = row_id[second]
    if not np.array_equal(np.sort(ids), np.flatnonzero(events['op'] == 0)):
        raise ValueError('missing or duplicate transfer receiver leg')
    for key, data in (('ts', stream['ts'].numpy()), ('token', token),
                      ('amount_z', fields['amount_z'].numpy()),
                      ('amount_known', fields['amount_known'].numpy())):
        if not np.array_equal(events[key][ids], data[second]):
            raise ValueError('transfer legs disagree on ' + key)
    events['dst'][ids] = dst[second]
    expected = [action_counts.get(name, 0) for name in OPERATIONS]
    if not np.array_equal(np.bincount(events['op'], minlength=4), expected):
        raise ValueError('canonical action counts differ from graph manifest')
    # build_four_actions_bundle defines the project-token registry as exactly
    # the set of distinct token_addr values, including unlabeled tokens.
    is_token = np.zeros(node_count, dtype=bool)
    is_token[events['token']] = True
    return events, is_token, is_pool


def load_nine_feature_inputs(g, *, include_source_roles=False):
    from bhtg import config
    from bhtg.action_csv import file_sha
    from bhtg.amounts import amount_metadata
    from bhtg.edge_stream import load_edge_stream

    if not (config.FOUR_ACTIONS_SELECTED and config.AMOUNT_EXPERIMENT):
        raise ValueError('behavior9 requires a selected four-actions amount graph')
    graph_path = Path(config.GRAPH_BIN)
    manifest_path = graph_path.with_name('manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    provenance = manifest['provenance']
    actions_sha = config.ACTIONS_SELECTION['actions_csv']['sha256']
    selection_manifest_sha = config.ACTIONS_SELECTION['manifest']['sha256']
    audit = provenance.get('amount_audit', {})
    if (provenance.get('source') != 'canonical_four_actions_csv'
            or manifest['source_hashes'].get('actions_csv') != actions_sha
            or manifest['source_hashes'].get('actions_manifest') != selection_manifest_sha
            or provenance.get('amount_protocol') != amount_metadata()
            or audit.get('actions_sha256') != actions_sha
            or audit.get('policy', {}).get('verified') is not True):
        raise ValueError('behavior9 graph lacks the selected verified amount source')
    # This also checks the in-memory graph fingerprint, local/global mappings,
    # stream hash and each bundle artifact against the manifest.
    stream = load_edge_stream(cache_path=graph_path.with_name('edge_stream.pt'), g=g,
                              graph_path=graph_path, bundle_manifest=manifest_path)
    vocabulary_path = graph_path.with_name('provenance_vocabulary.json')
    vocabulary = json.loads(vocabulary_path.read_text(encoding='utf-8'))
    events, is_token, is_pool = _canonical_events(
        g, stream, vocabulary, provenance['canonical_action_counts'])
    binding = {'actions_sha256': actions_sha,
               'actions_manifest_sha256': selection_manifest_sha,
               'graph_sha256': manifest['files']['bhtg_graph.bin']['sha256'],
               'graph_manifest_sha256': file_sha(manifest_path),
               'graph_fingerprint': stream['_meta']['graph_fingerprint'],
               'amount_protocol': amount_metadata()}
    if include_source_roles:
        roles = vocabulary.get('source_role')
        if (not isinstance(roles, list) or 'unlabeled' not in roles
                or any(role not in ('creation', 'labeled', 'unlabeled') for role in roles)
                or len(roles) != len(set(roles))):
            raise ValueError('behavior9 graph lacks a valid unlabeled source role vocabulary')
        binding['source_role_vocabulary'] = roles
    return events, is_token, is_pool, binding
