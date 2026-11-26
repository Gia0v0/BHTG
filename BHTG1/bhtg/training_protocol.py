"""Small, import-safe contracts shared by the corrected training runners."""
from contextlib import contextmanager
import math
from pathlib import Path

import torch


PROTOCOL_VERSION = 'causal_training_v2'
FULL_SEMANTIC_SCOPE = 'retrospective_full_history_v1'


def is_memory_state(key):
    """Runtime state is replayed, never selected or persisted as model weights."""
    return any(key.startswith(prefix) for prefix in (
        'memory_tables.', 'last_update_t.', 'seen.',
        'memory.memory_tables.', 'memory.last_update_t.', 'memory.seen.',
    ))


def model_weights(module):
    return {key: value.detach().cpu().clone()
            for key, value in module.state_dict().items()
            if not is_memory_state(key)}


def load_model_weights(module, weights):
    missing, unexpected = module.load_state_dict(weights, strict=False)
    missing_weights = [key for key in missing if not is_memory_state(key)]
    if missing_weights or unexpected:
        raise RuntimeError('Checkpoint weights mismatch: missing=%s unexpected=%s'
                           % (missing_weights, unexpected))


@contextmanager
def memory_snapshot(memory, snapshot):
    """Evaluation borrows a complete state and always returns the caller's state."""
    previous = memory.backup()
    override = memory._override
    training = memory.training
    try:
        memory.clear_override()
        memory.restore(snapshot)
        memory.eval()
        yield
    finally:
        memory.clear_override()
        memory.restore(previous)
        memory.set_override(override)
        memory.train(training)


def _validate_semantic_metadata(bundle, *, semantic_amount_mode, amount_experiment, allow_smoke):
    from bhtg.amount_protocol import SEMANTIC_PROTOCOL, validate_amount_metadata
    expected_protocol = ('causal_semantic_v1' if semantic_amount_mode == 'legacy'
                         else SEMANTIC_PROTOCOL)
    if not isinstance(bundle, dict) or bundle.get('protocol') != expected_protocol:
        raise RuntimeError('Semantic embeddings require the causal Phase1 bundle; retrain Phase1')
    if semantic_amount_mode not in ('legacy', 'disabled', 'continuous_zscore'):
        raise RuntimeError('Unknown semantic amount mode')
    if semantic_amount_mode != 'legacy':
        if amount_experiment is None:
            raise RuntimeError('new semantic protocol requires explicit amount provenance')
        validate_amount_metadata(bundle.get('amount_experiment'), amount_experiment)
        if bundle.get('semantic_amount_mode') != semantic_amount_mode:
            raise RuntimeError('Semantic amount mode mismatch')
        vocab = bundle.get('vocab')
        if not isinstance(vocab, dict) or not all(
                marker in vocab for marker in ('[AMT_CONT]', '[AMT_MISSING]', '[AMT_DISABLED]')):
            raise RuntimeError('Semantic continuous vocabulary is missing amount markers')
        if len(set(vocab.values())) != len(vocab) or set(vocab.values()) != set(range(len(vocab))):
            raise RuntimeError('Semantic vocabulary mapping is invalid')
        from bhtg.encoder.semantic_lmae import FieldVocab
        if vocab != FieldVocab(amount_policy=semantic_amount_mode).token_to_id:
            raise RuntimeError('Semantic vocabulary mapping differs from the expected amount schema')
    else:
        validate_amount_metadata(bundle.get('amount_experiment'), None)
    if not bundle.get('projection_trained'):
        raise RuntimeError('Semantic output projection was not trained')
    if bundle.get('smoke_subset') and not allow_smoke:
        raise RuntimeError('Smoke-subset semantic embeddings cannot be used for a full experiment')


def validate_semantic_bundle(bundle, n_contracts, hidden_dim, train_cutoff,
                             val_cutoff, *, semantic_amount_mode='legacy',
                             amount_experiment=None, allow_smoke=False):
    """Reject unversioned tensors and embeddings observed after the query time."""
    _validate_semantic_metadata(bundle, semantic_amount_mode=semantic_amount_mode,
                                amount_experiment=amount_experiment, allow_smoke=allow_smoke)
    outputs = {}
    for name, cutoff in (('train', train_cutoff), ('val', val_cutoff)):
        tensor = bundle['embeddings'][name]
        observed = float(bundle['cutoffs'][name])
        if (not math.isfinite(observed) or observed != float(cutoff)
                or tensor.shape != (n_contracts, hidden_dim)):
            raise RuntimeError('Invalid semantic %s snapshot or cutoff' % name)
        if not torch.isfinite(tensor).all():
            raise RuntimeError('Nonfinite semantic embeddings')
        outputs[name] = tensor
    if float(bundle['training_cutoff']) != float(train_cutoff):
        raise RuntimeError('Semantic pretraining cutoff differs from the training protocol')
    return outputs


def semantic_history_input_bindings():
    """Hash the selected files that determine semantic row identities and histories."""
    from bhtg import config
    from bhtg.amount_protocol import file_sha256

    data_dir = Path(config.BHTG_DATA_DIR)
    paths = {'graph': config.GRAPH_BIN, 'address_map': config.ADDR_TO_ID_CSV,
             'token_list': config.TOKEN_LIST_CSV, 'contract_gids': data_dir / 'contract_gids.pt',
             'node_types': data_dir / 'chain_node_types.csv'}
    if config.ACTION_CSV:
        paths['actions_csv'] = config.ACTION_CSV
    else:
        paths.update(labeled_csv=config.LABELED_EDGES_CSV, unlabeled_csv=config.UNLABELED_EDGES_CSV,
                     creators=data_dir / 'chain_creators.csv')
    return {name: {'path': str(Path(path).resolve()), 'sha256': file_sha256(path)}
            for name, path in paths.items()}


def validate_full_semantic_bundle(bundle, n_contracts, hidden_dim, train_cutoff,
                                  val_cutoff, full_history_cutoff, *, expected_input_bindings,
                                  semantic_amount_mode='legacy', amount_experiment=None):
    """Validate a full retrospective snapshot before indexing its graph-aligned rows."""
    _validate_semantic_metadata(bundle, semantic_amount_mode=semantic_amount_mode,
                                amount_experiment=amount_experiment, allow_smoke=False)
    if bundle.get('history_snapshot_scope') != FULL_SEMANTIC_SCOPE:
        raise RuntimeError('Semantic embeddings require an explicit retrospective full-history snapshot')
    if bundle.get('strict_temporal') is not True or bundle.get('instance_limit', 0):
        raise RuntimeError('Full-history semantics require a compatible, non-subset Phase1 checkpoint')
    recorded_cutoffs = bundle.get('cutoffs')
    if (not isinstance(recorded_cutoffs, dict)
            or recorded_cutoffs.get('train') != float(train_cutoff)
            or recorded_cutoffs.get('val') != float(val_cutoff)
            or bundle.get('training_cutoff') != float(train_cutoff)):
        raise RuntimeError('Full-history semantic pretraining cutoffs differ from Phase1')
    if (not math.isfinite(full_history_cutoff) or full_history_cutoff <= val_cutoff
            or bundle.get('full_history_cutoff') != float(full_history_cutoff)):
        raise RuntimeError('Full-history semantic snapshot cutoff differs from Phase3')
    bindings = bundle.get('export_input_bindings')
    if not isinstance(bindings, dict) or bindings.keys() != expected_input_bindings.keys():
        raise RuntimeError('Full-history semantic snapshot input bindings are incomplete')
    for name, expected in expected_input_bindings.items():
        binding = bindings[name]
        if not isinstance(binding, dict) or binding.get('sha256') != expected['sha256']:
            raise RuntimeError('Full-history semantic snapshot input mismatch: ' + name)
    tensor = bundle.get('h_sem_full')
    if (not torch.is_tensor(tensor) or not tensor.is_floating_point()
            or tensor.shape != (n_contracts, hidden_dim) or not torch.isfinite(tensor).all()):
        raise RuntimeError('Invalid full-history semantic embeddings')
    return tensor
