"""Audited saved z-scores shared by semantic histories and graph conversion.

Source hashes and evidence, never supervision labels or filenames alone, decide
whether a value is usable. Unknown values keep their original text in actions.
"""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


POLICY = 'rugscreener_saved_zscore_v1'
SCHEMA = 'four_actions_amount_fields_v1'
SOURCE_COLUMNS = ('value_text', 'source_file_sha256', 'source_row')
# This source was independently inspected in transfer_preprocess_audit.json.
# Its feature() writer standardizes contiguous token runs (constant runs -> 1).
CONFIRMED_SOURCE = '16453ad8feed338692f91502278f7cc1b3c5b0862524df0ae1d28820eeb78fcb'
PREPROCESS_CODE = 'a6806149d566b467c52618772e477096c0a78b1ce3a61eaa32493946b2d972b3'
AUDIT_EVIDENCE = '328c76ab122eda9a12309fcd366f9780257311ce1f14be25d35e9f8bd2fa77f4'


def amount_metadata():
    return {'schema_version': SCHEMA, 'event_time_policy': 'strict_before_cutoff',
            'amount_source_policy': POLICY, 'amount_transform': 'identity',
            'amount_preprocessing_causal': False,
            'evaluation_scope': 'retrospective_saved_preprocessed_amounts',
            'fields': {'amount_z': 'float32', 'amount_known': 'bool',
                       'source_row_id_code': 'int64_audit_only'}}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def public_policy(policy):
    return {key: value for key, value in policy.items() if not key.startswith('_')}


def source_policy_for_actions(path, selection=None):
    """Verify selection, source ledger, source files and normalization evidence.

    Unselected diagnostic CSVs have no confirmed sources. Callers of synthetic
    tests may pass an explicit policy to parse_amount_frame/read_actions.
    """
    import pandas as pd
    from bhtg.action_csv import active_actions_selection
    path = Path(path).resolve()
    selection = selection if selection is not None else active_actions_selection()
    policy = dict(amount_metadata(), verified=False, sources={}, _source_rows={})
    if selection is None or Path(selection['actions_csv']['path']).resolve() != path:
        policy['reason'] = 'unselected_diagnostic_source'
        return policy
    if _sha(path) != selection['actions_csv']['sha256']:
        raise ValueError('amount actions source hash mismatch')
    dataset = selection['dataset_manifest']
    ledger_path = Path(selection['manifest']['path']).parent / 'source_manifest.json'
    expected = dataset.get('files', {}).get('source_manifest.json', {}).get('sha256')
    if not expected or _sha(ledger_path) != expected:
        raise ValueError('amount source manifest hash mismatch')
    ledger = json.loads(ledger_path.read_text(encoding='utf-8-sig'))
    if ledger.get('schema_version') != 'bhtg_four_actions_sources_v1':
        raise ValueError('unsupported amount source ledger')
    audit_path = Path(ledger['expected_source_sha_from'])
    if _sha(audit_path) != AUDIT_EVIDENCE:
        raise ValueError('saved amount source audit evidence hash mismatch')
    audit = json.loads(audit_path.read_text(encoding='utf-8-sig'))
    evidence = {item.get('sha256'): item for item in audit.get('files', []) if item.get('sha256')}
    code = evidence.get(PREPROCESS_CODE)
    if not code or _sha(code['path']) != PREPROCESS_CODE:
        raise ValueError('saved amount preprocessing evidence hash mismatch')
    statistics = audit.get('labeled_transfer_std', {})
    if (statistics.get('values_nonfinite') != 0 or
            statistics.get('tokens_with_std_within_1e_8_of_1', 0) < 1):
        raise ValueError('saved amount normalization audit evidence missing')
    policy.update(verified=True, actions_sha256=selection['actions_csv']['sha256'],
                  actions_manifest_sha256=selection['manifest']['sha256'],
                  source_manifest_sha256=expected,
                  evidence={'audit_sha256': _sha(audit_path), 'preprocess_sha256': PREPROCESS_CODE,
                            'method': 'load_data.feature; contiguous token run mean/std; constant -> 1'})
    for role, record in ledger.get('sources', {}).items():
        if dataset.get('source_inputs', {}).get(role) != record:
            raise ValueError('dataset/source amount ledger binding mismatch: ' + role)
        sha = record['sha256']
        if _sha(record['path']) != sha:
            raise ValueError('amount source file hash mismatch: ' + role)
        confirmed = sha == CONFIRMED_SOURCE and sha in evidence
        policy['sources'][sha] = {'role': role, 'path': record['path'], 'sha256': sha,
                                 'confirmed': confirmed,
                                 'reason': ('verified_saved_standardization' if confirmed else
                                            'normalization_semantics_unconfirmed')}
        # Keep only model-input audit fields; labels never enter this path.
        source = pd.read_csv(record['path'], dtype=str, keep_default_na=False,
                             usecols=['token_address', 'from_address', 'to_address',
                                      'value', 'block_timestamp'])
        policy['_source_rows'][sha] = source.to_dict('list')
    creation = ledger.get('creation', {})
    if dataset.get('source_inputs', {}).get('creation') != creation:
        raise ValueError('dataset/source amount creation ledger binding mismatch')
    if not creation.get('path') or _sha(creation['path']) != creation.get('sha256'):
        raise ValueError('amount creation source file hash mismatch')
    sha = creation['sha256']
    policy['sources'][sha] = {'role': 'creation', 'path': creation['path'], 'sha256': sha,
                             'confirmed': False, 'reason': 'creation_amount_unknown'}
    return policy


def parse_amount_frame(frame, policy):
    """Return validated float32 amount_z/boolean known plus an audit reason.

    Known zero remains known. Nonempty illegal values in confirmed sources fail
    with source identity. A stored amount column cannot override source parsing.
    """
    if policy.get('amount_source_policy') != POLICY:
        raise ValueError('unsupported amount source policy')
    missing = set(SOURCE_COLUMNS) - set(frame)
    if missing:
        raise ValueError('continuous amount input is missing source fields: ' + ', '.join(sorted(missing)))
    out = frame.copy()
    values, known, reasons = [], [], []
    fields = ['edge_type', 'source_file_sha256', 'source_row', 'source_row_id',
              'value_text', 'token_addr', 'src_addr', 'dst_addr', 'timestamp']
    for kind, sha, row, identity, text, token, src, dst, timestamp in out[fields].itertuples(index=False, name=None):
        source = policy.get('sources', {}).get(sha)
        z, usable = 0.0, False
        if kind == 'create':
            reason = 'creation_amount_unknown'
        else:
            if not str(row).isdigit() or int(row) < 1 or identity != '%s:%s' % (sha, row):
                raise ValueError('amount source-row identity mismatch: ' + str(identity))
            confirmed = bool(source and source.get('confirmed') is True)
            reason = 'normalization_semantics_unconfirmed' if not confirmed else 'missing_value_text'
            if confirmed and str(text).strip():
                try:
                    z = float(text)
                except (TypeError, ValueError):
                    raise ValueError('invalid confirmed amount at ' + str(identity))
                if not math.isfinite(z) or abs(z) > np.finfo(np.float32).max:
                    raise ValueError('nonfinite/float32 overflow confirmed amount at ' + str(identity))
                z, usable, reason = float(np.float32(z)), True, 'known'
            source_rows = policy.get('_source_rows', {}).get(sha)
            if source_rows is not None:
                i = int(row) - 1
                if i >= len(source_rows['value']):
                    raise ValueError('amount source-row outside source file: ' + str(identity))
                expected = (source_rows['token_address'][i].lower(), source_rows['from_address'][i].lower(),
                            source_rows['to_address'][i].lower(), source_rows['value'][i],
                            source_rows['block_timestamp'][i])
                if (token, src, dst, str(text), str(timestamp)) != expected:
                    raise ValueError('amount source-row content mismatch: ' + str(identity))
        values.append(z); known.append(usable); reasons.append(reason)
    z = np.asarray(values, dtype=np.float32)
    flags = np.asarray(known, dtype=bool)
    if 'amount_z' in out:
        try:
            stored = out['amount_z'].to_numpy(dtype=np.float32)
        except (ValueError, TypeError):
            raise ValueError('invalid stored amount_z')
        if not np.array_equal(stored, z):
            raise ValueError('stored amount_z disagrees with audited source')
    if 'amount_known' in out:
        stored = out['amount_known'].map(lambda x: x if isinstance(x, (bool, np.bool_)) else
                                         {'true': True, 'false': False}.get(str(x).lower(), None))
        if stored.isna().any() or not np.array_equal(stored.to_numpy(dtype=bool), flags):
            raise ValueError('stored amount_known disagrees with audited source')
    out['amount_z'], out['amount_known'], out['amount_reason'] = z, flags, reasons
    return out


def audit_actions(path, policy=None):
    from bhtg.action_csv import iter_action_chunks, file_sha
    policy = source_policy_for_actions(path) if policy is None else policy
    groups, known_values = defaultdict(lambda: Counter(total=0, known=0, unknown=0)), []
    seen = set()
    columns = ['edge_type', 'source_file_sha256', 'source_role', 'source_row_id',
               'amount_z', 'amount_known', 'amount_reason']
    for frame in iter_action_chunks(path, columns=columns, amount_mode='continuous_zscore', amount_policy=policy):
        identities = frame['source_row_id'].tolist()
        if seen.intersection(identities):
            raise ValueError('repeated amount source-row identity across chunks')
        seen.update(identities)
        for kind, sha, role, identity, z, usable, reason in frame.itertuples(index=False, name=None):
            group = groups[(sha, role, kind)]
            group['total'] += 1
            group['known' if usable else 'unknown'] += 1
            group['reason:' + reason] += 1
            if usable:
                known_values.append(z)
    distribution = {'count': len(known_values)}
    if known_values:
        values = np.asarray(known_values, dtype=np.float64)
        distribution.update(min=float(values.min()), max=float(values.max()), mean=float(values.mean()),
                            std=float(values.std()), negative=int((values < 0).sum()),
                            zero=int((values == 0).sum()),
                            quantiles={str(q): float(np.quantile(values, q)) for q in (0, .01, .5, .99, 1)})
    return {'schema_version': 'four_actions_amount_source_audit_v1', 'policy': public_policy(policy),
            'actions_sha256': file_sha(path), 'rows': len(seen),
            'source_row_alignment': 'checked_against_source_files' if policy.get('verified') else 'diagnostic',
            'groups': [dict(source_sha256=sha, source_role=role, action=kind, **dict(counts))
                       for (sha, role, kind), counts in sorted(groups.items())],
            'known_distribution': distribution, 'training_started': False}


def main(argv=None):
    from bhtg.action_csv import active_actions_selection
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actions-csv')
    parser.add_argument('--output', required=True, help='new audit JSON path')
    args = parser.parse_args(argv)
    selected = active_actions_selection()
    path = args.actions_csv or (selected['actions_csv']['path'] if selected else None)
    if not path:
        parser.error('select a four-actions CSV')
    report = audit_actions(path)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
    print(json.dumps({'output': str(target.resolve()), 'rows': report['rows'],
                      'known': report['known_distribution']['count']}))
    return report


if __name__ == '__main__':
    main()
