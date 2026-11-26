"""Versioned provenance for retrospective saved continuous amount experiments."""
import hashlib
import json
from pathlib import Path


AMOUNT_PROTOCOL = 'retrospective_saved_amounts_v1'
SEMANTIC_PROTOCOL = 'retrospective_amount_semantic_v2'


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_metadata(config=None):
    """Bind new weights to the actual source, graph and typed node ordering."""
    if config is None:
        from bhtg import config
    if not config.AMOUNT_EXPERIMENT:
        return None
    selected = config.ACTIONS_SELECTION
    bindings = {
        'actions_csv': selected['actions_csv']['sha256'],
        'actions_manifest': selected['manifest']['sha256'],
        'graph': file_sha256(config.GRAPH_BIN),
        'address_map': file_sha256(config.ADDR_TO_ID_CSV),
        'contract_gids': file_sha256(Path(config.BHTG_DATA_DIR) / 'contract_gids.pt'),
        'graph_manifest': file_sha256(config.GRAPH_MANIFEST_PATH),
    }
    graph_manifest = json.loads(Path(config.GRAPH_MANIFEST_PATH).read_text(encoding='utf-8-sig'))
    from bhtg.amount_edge_features import graph_amount_metadata
    return {
        'protocol': AMOUNT_PROTOCOL,
        'architecture': 'four_actions_continuous_amount_v2',
        'semantic_amount_mode': config.SEMANTIC_AMOUNT_MODE,
        'memory_amount_mode': config.MEMORY_AMOUNT_MODE,
        'event_time_policy': 'strict_before_cutoff',
        'amount_source_policy': config.AMOUNT_SOURCE_POLICY,
        'amount_transform': 'identity',
        'amount_preprocessing_causal': False,
        'evaluation_scope': 'retrospective_saved_preprocessed_amounts',
        'amount_schema': {'amount_z': 'float32', 'amount_known': 'bool',
                          'source_row_id': 'audit_only'},
        'memory_value_dim': 2,
        'memory_message': '[z,1] if known; [0,0] if missing or disabled',
        'graph_amount_contract': graph_amount_metadata(config.MEMORY_AMOUNT_MODE),
        'amount_mask_rate': config.AMOUNT_MASK_RATE,
        'amount_loss_weight': (config.AMOUNT_LOSS_WEIGHT
                               if config.SEMANTIC_AMOUNT_MODE == 'continuous_zscore' else 0.),
        'amount_mlp_hidden': config.AMOUNT_MLP_HIDDEN,
        'bindings': bindings,
        'node_types': list(config.NODE_TYPES),
        'relations': list(config.EDGE_TYPES),
        'seed': config.SEED,
        'graph_source_audit': graph_manifest.get('provenance', {}).get('amount_audit'),
    }


def validate_amount_metadata(actual, expected):
    if expected is None:
        if actual is not None:
            raise RuntimeError('amount checkpoint cannot be loaded in legacy mode')
        return
    if not isinstance(actual, dict):
        raise RuntimeError('missing amount experiment metadata; regenerate this artifact')
    for key, value in expected.items():
        if actual.get(key) != value:
            raise RuntimeError('amount experiment mismatch: %s' % key)


def upstream_identity(path):
    return {'path': str(Path(path).resolve()), 'sha256': file_sha256(path)}
