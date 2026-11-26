"""Re-export all visible history using an existing trained Phase1 encoder.

No optimizer or training is run. Select the same data/amount environment as the
source checkpoint, then pass a separate, new output directory. Phase3 can read
the resulting h_sem_temporal_lmae_strict.pt through --sem-path. For full-history
Phase3, --full-history-cutoff auto exports a separate h_sem_full_history.pt.
"""
import argparse
import copy
import json
import math
import os
import pickle
from pathlib import Path

import torch

import run_phase1_lmae as phase1
from bhtg.amount_protocol import file_sha256, validate_amount_metadata
from bhtg.encoder.semantic_lmae import build_semantic_lmae, load_semantic_checkpoint
from bhtg.training_protocol import (
    FULL_SEMANTIC_SCOPE, semantic_history_input_bindings,
    validate_full_semantic_bundle, validate_semantic_bundle,
)


def _validate_checkpoint(checkpoint, index, vocab):
    if not isinstance(checkpoint, dict) or checkpoint.get('protocol') != phase1.PROTOCOL:
        raise RuntimeError('Phase1 checkpoint protocol differs from selected inputs')
    if checkpoint.get('projection_trained') is not True:
        raise RuntimeError('Phase1 output projection was not trained')
    if checkpoint.get('strict_temporal') is not True:
        raise RuntimeError('Phase1 checkpoint must use strict temporal inputs')
    if checkpoint.get('smoke_subset') or checkpoint.get('instance_limit', 0):
        raise RuntimeError('Smoke/subset checkpoints cannot produce formal semantic snapshots')
    if checkpoint.get('vocab') != vocab.token_to_id:
        raise RuntimeError('Phase1 checkpoint vocabulary mismatch')
    if checkpoint.get('amount_policy') != phase1.AMOUNT_POLICY:
        raise RuntimeError('Phase1 checkpoint amount policy mismatch')
    if checkpoint.get('source') != index['source']:
        raise RuntimeError('Phase1 checkpoint data source mismatch')
    for name in ('train', 'val'):
        expected = float(index[name + '_cutoff'])
        observed = float(checkpoint.get('cutoffs', {}).get(name, float('nan')))
        if not math.isfinite(expected) or observed != expected:
            raise RuntimeError('Phase1 checkpoint %s cutoff mismatch' % name)
    if float(checkpoint.get('training_cutoff', float('nan'))) != float(index['train_cutoff']):
        raise RuntimeError('Phase1 checkpoint training cutoff mismatch')
    expected_amount = phase1.experiment_metadata()
    validate_amount_metadata(checkpoint.get('amount_experiment'), expected_amount)
    return expected_amount


def _validate_legacy_builder(checkpoint_path, index, vocab):
    # Old legacy checkpoints lack source hashes and do not store contract order.
    # The accompanying builder protects against reusing weights with a different
    # contract registry; it cannot establish the original CSV's content hash.
    path = checkpoint_path.parent / 'phase1_builder_lmae_strict.pkl'
    if not path.is_file():
        raise RuntimeError('Legacy re-export requires adjacent phase1_builder_lmae_strict.pkl')
    with path.open('rb') as handle:
        builder = pickle.load(handle)
    if builder.get('cid_to_addr') != index['cid_to_addr']:
        raise RuntimeError('Legacy builder contract registry/order mismatch')
    if builder.get('vocab') != vocab.token_to_id:
        raise RuntimeError('Legacy builder vocabulary mismatch')
    if float(builder.get('train_cutoff', float('nan'))) != float(index['train_cutoff']):
        raise RuntimeError('Legacy builder training cutoff mismatch')
    return {'path': str(path), 'sha256': file_sha256(path)}


def _restore_encoder(checkpoint, vocab, expected_amount):
    if phase1.SEMANTIC_AMOUNT_MODE != 'legacy':
        return load_semantic_checkpoint(checkpoint, expected_mode=phase1.SEMANTIC_AMOUNT_MODE,
                                        vocab=vocab, expected_amount_metadata=expected_amount)
    cfg = checkpoint['cfg']
    model = build_semantic_lmae(
        cfg['vocab_size'], d_model=cfg['hidden_size'],
        num_layers=cfg['num_hidden_layers'], num_heads=cfg['num_attention_heads'],
        ffn=cfg['intermediate_size'], max_seq_len=cfg['max_position_embeddings'],
        dropout=cfg['hidden_dropout_prob'], out_dim=phase1.HIDDEN_DIM,
        projection_loss_weight=checkpoint['projection_loss_weight'],
        projection_temperature=checkpoint['projection_temperature'])
    model.model.load_state_dict(checkpoint['model_state'], strict=True)
    model.out_proj.load_state_dict(checkpoint['out_proj'], strict=True)
    model.anchor_bert.load_state_dict(checkpoint['anchor_bert'], strict=True)
    model.anchor_out_proj.load_state_dict(checkpoint['anchor_out_proj'], strict=True)
    return model


def _current_bindings():
    """Hash the current export inputs; these are not retroactive training proof."""
    paths = {'graph': phase1.GRAPH_BIN, 'address_map': phase1.ADDR_TO_ID_CSV,
             'contract_gids': os.path.join(phase1.BHTG_DATA_DIR, 'contract_gids.pt')}
    if phase1.ACTION_CSV:
        paths['actions_csv'] = phase1.ACTION_CSV
    return {name: {'path': str(Path(path).resolve()), 'sha256': file_sha256(path)}
            for name, path in paths.items() if Path(path).is_file()}


def export_checkpoint(checkpoint_path, output_dir, *, window_size=16,
                      window_batch_size=4, device=None, log=print, full_history_cutoff=None):
    """Validate and export a trained checkpoint without changing its weights."""
    if 'BHTG_BUNDLE_DIR' in os.environ:
        raise RuntimeError('This exporter uses the four-actions/legacy Phase1 route, not the v3 runner')
    for name, value in (('window_size', window_size), ('window_batch_size', window_batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('%s must be a positive integer' % name)
    checkpoint_path = Path(checkpoint_path).resolve()
    output_dir = Path(output_dir).resolve()
    full_history = full_history_cutoff is not None
    if not full_history and checkpoint_path.parent == output_dir:
        raise RuntimeError('Re-export requires a separate output directory')
    if full_history:
        bundle_name, manifest_name = 'h_sem_full_history.pt', 'semantic_full_history_export.json'
        names = (bundle_name, manifest_name)
    else:
        bundle_name, manifest_name = 'h_sem_temporal_lmae_strict.pt', 'semantic_history_export.json'
        names = (bundle_name, manifest_name, 'h_sem_all_lmae_strict.pt')
    if any((output_dir / name).exists() for name in names):
        raise FileExistsError('Semantic re-export outputs already exist; choose a new output directory')
    phase1.require_graph_binding()
    checkpoint_digest = file_sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    snapshot_cutoffs = None
    if full_history:
        from bhtg.retrospective_protocol import full_history_cutoff as graph_cutoff

        graph = phase1.dgl.load_graphs(phase1.GRAPH_BIN)[0][0]
        n_contracts = graph.num_nodes('contract')
        minimum = graph_cutoff(graph)
        del graph
        if full_history_cutoff == 'auto':
            full_history_cutoff = minimum
        else:
            full_history_cutoff = float(full_history_cutoff)
        if not math.isfinite(full_history_cutoff) or full_history_cutoff < minimum:
            raise ValueError('Full-history cutoff must include every archived graph edge')
        snapshot_cutoffs = {'full': full_history_cutoff}
        index = phase1.load_causal_index(full_history_cutoff=full_history_cutoff)
    else:
        index = phase1.load_causal_index()
    vocab = phase1.legacy_vocab(index['node_types'])
    expected_amount = _validate_checkpoint(checkpoint, index, vocab)
    builder_identity = (_validate_legacy_builder(checkpoint_path, index, vocab)
                        if phase1.SEMANTIC_AMOUNT_MODE == 'legacy' else None)
    sem = _restore_encoder(checkpoint, vocab, expected_amount)
    if sem.out_proj.out_features != phase1.HIDDEN_DIM:
        raise RuntimeError('Semantic output dimension differs from Phase3')
    sem.history_encoding_config(window_size)
    if not full_history:
        n_contracts = phase1.dgl.load_graphs(phase1.GRAPH_BIN)[0][0].num_nodes('contract')
    sem.to(phase1.DEVICE if device is None else device).eval()
    snapshots, coverage = phase1.export_semantic_snapshots(
        sem, vocab, index, n_contracts, window_size=window_size,
        window_batch_size=window_batch_size, log=log, snapshot_cutoffs=snapshot_cutoffs)
    weights = {'model_state', 'state_dict', 'out_proj', 'anchor_bert', 'anchor_out_proj', 'embeddings'}
    metadata = {key: copy.deepcopy(value) for key, value in checkpoint.items() if key not in weights}
    input_bindings = semantic_history_input_bindings() if full_history else _current_bindings()
    metadata.update(history_encoding=sem.history_encoding_config(window_size),
                    history_coverage=coverage, export_window_batch_size=window_batch_size,
                    source_checkpoint={'path': str(checkpoint_path), 'sha256': checkpoint_digest},
                    export_input_bindings=input_bindings, legacy_builder=builder_identity)
    manifest_fields = ['history_encoding', 'history_coverage', 'export_window_batch_size',
                       'source_checkpoint', 'export_input_bindings', 'legacy_builder']
    if full_history:
        metadata.update(full_history_cutoff=full_history_cutoff, history_snapshot_scope=FULL_SEMANTIC_SCOPE)
        manifest_fields.extend(('full_history_cutoff', 'history_snapshot_scope'))
        bundle = {**metadata, 'h_sem_full': snapshots['full']}
        validate_full_semantic_bundle(bundle, n_contracts, phase1.HIDDEN_DIM,
                                      index['train_cutoff'], index['val_cutoff'], full_history_cutoff,
                                      expected_input_bindings=input_bindings,
                                      semantic_amount_mode=phase1.SEMANTIC_AMOUNT_MODE,
                                      amount_experiment=expected_amount)
    else:
        bundle = {'embeddings': snapshots, **metadata}
        validate_semantic_bundle(bundle, n_contracts, phase1.HIDDEN_DIM,
                                 index['train_cutoff'], index['val_cutoff'],
                                 semantic_amount_mode=phase1.SEMANTIC_AMOUNT_MODE,
                                 amount_experiment=expected_amount)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against another exporter racing this one.
    bundle_path = output_dir / bundle_name
    with bundle_path.open('xb') as handle:
        torch.save(bundle, handle)
    if not full_history:
        with (output_dir / 'h_sem_all_lmae_strict.pt').open('xb') as handle:
            torch.save(snapshots['train'], handle)
    manifest = {key: metadata[key] for key in manifest_fields}
    manifest['output'] = {'path': str(bundle_path), 'sha256': file_sha256(bundle_path)}
    with (output_dir / manifest_name).open('x', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    log('[done] re-exported frozen Phase1 history: %s' % bundle_path)
    return bundle_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Trained phase1_sem_lmae_strict.pt')
    parser.add_argument('--output-dir', required=True, help='Separate directory with no previous semantic exports')
    parser.add_argument('--window-size', type=int, default=16, help='Complete events per window')
    parser.add_argument('--window-batch-size', type=int, default=4, help='Windows per encoder call; does not limit history')
    parser.add_argument('--device', default=None, help='Optional torch device, e.g. cpu or cuda:0')
    parser.add_argument('--full-history-cutoff', default=None,
                        help='Export only the full-history snapshot at timestamp T, or use auto for the full graph')
    args = parser.parse_args(argv)
    options = {} if args.full_history_cutoff is None else {'full_history_cutoff': args.full_history_cutoff}
    export_checkpoint(args.checkpoint, args.output_dir, window_size=args.window_size,
                      window_batch_size=args.window_batch_size, device=args.device, **options)


if __name__ == '__main__':
    main()
