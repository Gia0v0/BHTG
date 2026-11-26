"""Phase1 pretraining and train/validation semantic snapshots.

Train BERT and the exported projection only on events before train_cutoff.
Encode frozen weights at train/val observation times from exact CSV timestamps.
Legacy calls mask saved amounts. Explicit continuous experiments consume saved
z-scores: event selection is strict, while amount preprocessing is retrospective.
"""
import argparse
import copy
import math
import os
import pickle
import time

import dgl
import numpy as np
import pandas as pd
import torch

from bhtg.config import (
    ADDR_TO_ID_CSV, BHTG_DATA_DIR, CHECKPOINT_DIR, DEVICE, GRAPH_BIN,
    HIDDEN_DIM, SEED, SEM_BERT_HIDDEN, SEM_BERT_LAYERS, SEM_BERT_HEADS,
    SEM_BERT_FFN, SEM_MAX_LEN, SEM_CTL_TEMP, SEM_CTL_MOMENTUM,
    SEM_MASK_RATE, SEM_NSP_PROB, SEM_NSP_WINDOW, SEM_EPOCHS, TOKEN_LIST_CSV,
    LABELED_EDGES_CSV, UNLABELED_EDGES_CSV,
)
from bhtg.action_csv import model_edges, model_action_frames, require_graph_binding
from bhtg.config import ACTION_CSV
from bhtg.config import (SEMANTIC_AMOUNT_MODE, AMOUNT_EXPERIMENT, AMOUNT_MASK_RATE,
                         AMOUNT_LOSS_WEIGHT, AMOUNT_MLP_HIDDEN)
from bhtg.amount_protocol import SEMANTIC_PROTOCOL, experiment_metadata
from bhtg.encoder.semantic_lmae import (
    FieldVocab, build_nsp_instances, make_mlm_input, build_semantic_lmae,
    collate_structured, semantic_checkpoint,
)


PROTOCOL = SEMANTIC_PROTOCOL if AMOUNT_EXPERIMENT else 'causal_semantic_v1'
AMOUNT_POLICY = SEMANTIC_AMOUNT_MODE if AMOUNT_EXPERIMENT else 'masked_global_zscore'


def build_causal_histories(edge_frames, cid_to_addr, cutoff, creators=None, *, amount_mode='legacy',
                           require_complete=False):
    """Build label-blind, exact-time histories from CSV frames/chunks.

    Token identities are a fixed registry. Labels and pool statistics are never
    model inputs. New mode copies only the audited amount side channels.
    """
    addr_to_cid = {str(address).lower(): int(cid) for cid, address in cid_to_addr.items()}
    histories = {int(cid): [] for cid in cid_to_addr}
    if not math.isfinite(float(cutoff)):
        raise ValueError('cutoff must be finite')
    columns = ('edge_type', 'src_addr', 'dst_addr', 'token_addr', 'timestamp')
    if amount_mode not in ('legacy', 'disabled', 'continuous_zscore'):
        raise ValueError('invalid history amount mode')
    extra_columns = ('amount_z', 'amount_known', 'source_row_id') if amount_mode != 'legacy' else ()
    identities = set()
    for frame in edge_frames:
        if set(extra_columns) - set(frame.columns):
            raise ValueError('continuous semantic history requires complete amount protocol')
        for values in frame.loc[:, list(columns + extra_columns)].itertuples(index=False, name=None):
            operation, source, destination, token, timestamp = values[:5]
            timestamp = float(timestamp)
            if not math.isfinite(timestamp) or timestamp >= cutoff:
                if require_complete:
                    raise ValueError('Full-history cutoff must be later than all finite action timestamps')
                continue
            if creators is not None and str(operation).lower() == 'create':
                continue
            cid = addr_to_cid.get(str(token).lower())
            if cid is None:
                continue
            record = {'edge_type': str(operation).lower(),
                                   'src_addr': str(source).lower(),
                                   'dst_addr': str(destination).lower(),
                                   'timestamp': timestamp, 'value': 0.0}
            if extra_columns:
                amount, known, identity = values[5:]
                if (not isinstance(known, (bool, np.bool_)) or not math.isfinite(float(amount))
                        or not np.isfinite(np.float32(amount)) or (not known and amount != 0)):
                    raise ValueError('invalid amount fields in semantic event ' + str(identity))
                if not str(identity) or identity in identities:
                    raise ValueError('repeated semantic source_row_id ' + str(identity))
                identities.add(identity)
                record.update(amount_z=float(amount), amount_known=bool(known), source_row_id=str(identity))
            histories[cid].append(record)
    if creators is not None:
        for token, creator, timestamp in creators.loc[:, ['token_address', 'creator_address', 'timestamp']].itertuples(index=False, name=None):
            cid = addr_to_cid.get(str(token).lower())
            timestamp = float(timestamp)
            if require_complete and (not math.isfinite(timestamp) or timestamp >= cutoff):
                raise ValueError('Full-history cutoff must be later than all finite creation timestamps')
            if cid is not None and math.isfinite(timestamp) and timestamp < cutoff:
                record = {'edge_type': 'create', 'src_addr': str(creator).lower(),
                                       'dst_addr': str(token).lower(),
                                       'timestamp': timestamp, 'value': 0.0}
                if extra_columns:
                    raise ValueError('continuous histories require canonical create event identities')
                histories[cid].append(record)
    for edges in histories.values():
        edges.sort(key=lambda edge: edge['timestamp'])
    return histories


def load_causal_index(*, full_history_cutoff=None):
    """Rebuild rather than trusting the legacy float32/Z-score event cache."""
    require_graph_binding()
    timestamps = model_edges(usecols=['timestamp'], legacy_path=LABELED_EDGES_CSV)['timestamp'].to_numpy(dtype=np.float64)
    if not len(timestamps) or not np.isfinite(timestamps).all():
        raise RuntimeError('Labeled CSV timestamps must be nonempty and finite')
    train_cutoff, val_cutoff = [float(value) for value in np.quantile(timestamps, [0.70, 0.85])]
    history_cutoff = val_cutoff if full_history_cutoff is None else float(full_history_cutoff)
    if full_history_cutoff is not None and (not math.isfinite(history_cutoff) or history_cutoff <= val_cutoff):
        raise ValueError('Full-history cutoff must be finite and later than the Phase1 validation cutoff')
    addresses = pd.read_csv(ADDR_TO_ID_CSV)
    id_to_addr = dict(zip(addresses['id'], addresses['address'].str.lower()))
    tokens = set(pd.read_csv(TOKEN_LIST_CSV)['token_address'].str.lower())
    gids = torch.load(os.path.join(BHTG_DATA_DIR, 'contract_gids.pt'), map_location='cpu')
    cid_to_addr = {lid: id_to_addr[int(gid)] for lid, gid in enumerate(gids)
                   if int(gid) in id_to_addr and id_to_addr[int(gid)] in tokens}
    # Canonical CSV histories use each action once, including verified create.
    creators = None if ACTION_CSV else pd.read_csv(
        os.path.join(BHTG_DATA_DIR, 'chain_creators.csv'),
        usecols=['token_address', 'creator_address', 'timestamp'])
    histories = build_causal_histories(
        model_action_frames(legacy_paths=(LABELED_EDGES_CSV, UNLABELED_EDGES_CSV)),
        cid_to_addr, history_cutoff, creators, amount_mode=SEMANTIC_AMOUNT_MODE,
        require_complete=full_history_cutoff is not None)
    registry = pd.read_csv(os.path.join(BHTG_DATA_DIR, 'chain_node_types.csv'),
                           usecols=['address', 'node_type'])
    node_types = dict(zip(registry['address'].str.lower(), registry['node_type'].str.lower()))
    return {'cid_edges': histories, 'cid_to_addr': cid_to_addr, 'node_types': node_types,
            'strict_temporal': True, 'train_cutoff': train_cutoff, 'val_cutoff': val_cutoff,
            'protocol': PROTOCOL, 'amount_policy': AMOUNT_POLICY,
            'source': 'exact_csv_timestamps'}


def legacy_vocab(node_types):
    def resolve_type(address):
        name = node_types.get(str(address).lower(), 'eoa')
        return {'eoa': 'EOA', 'contract': 'CONTRACT', 'pool': 'POOL'}.get(name, 'EOA')
    return FieldVocab(node_type_fn=resolve_type, amount_policy=AMOUNT_POLICY)


def collate(batch, vocab, max_len=SEM_MAX_LEN):
    if vocab.structured_amounts:
        return collate_structured(batch, vocab, max_len)
    seqs, segments = [], []
    for _, a_ids, b_ids, _ in batch:
        a, b = list(a_ids), list(b_ids)
        room = max_len - 3
        if len(b) > room - len(a):
            b = b[:max(0, room - len(a))]
        if len(a) > room - len(b):
            a = a[:max(0, room - len(b))]
        ids = [vocab.CLS] + a + [vocab.SEP] + b + [vocab.SEP]
        padding = max_len - len(ids)
        seqs.append(ids + [vocab.PAD] * padding)
        segments.append([0] * (len(a) + 2) + [1] * (len(b) + 1) + [0] * padding)
    return torch.tensor(seqs, dtype=torch.long), torch.tensor(segments, dtype=torch.long)


@torch.no_grad()
def export_semantic_snapshots(sem, vocab, index, n_contracts, *, window_size=16,
                              window_batch_size=4, cids=None, log=print, snapshot_cutoffs=None):
    """Export vectors at the selected cutoffs without changing model weights."""
    config = sem.history_encoding_config(window_size)
    selected = None if cids is None else set(cids)
    snapshots, coverage = {}, {}
    if snapshot_cutoffs is None:
        snapshot_cutoffs = {phase: index[phase + '_cutoff'] for phase in ('train', 'val')}
    for phase, cutoff in snapshot_cutoffs.items():
        cutoff = float(cutoff)
        if not math.isfinite(cutoff):
            raise ValueError('cutoff must be finite')
        output = torch.zeros(n_contracts, sem.out_proj.out_features)
        stats = {'visible_events': 0, 'encoded_events': 0, 'windows': 0,
                 'active': 0, 'missing': 0, 'considered': 0}
        for cid, edges in index['cid_edges'].items():
            if selected is not None and cid not in selected:
                continue
            if not 0 <= cid < n_contracts:
                raise ValueError('semantic contract index outside graph')
            stats['considered'] += 1
            count = sum(float(edge['timestamp']) < cutoff for edge in edges)
            stats['visible_events'] += count
            if not count:
                stats['missing'] += 1
                continue
            row = sem.encode_history(edges, vocab, window_batch_size=window_batch_size,
                                     cutoff=cutoff, window_size=window_size).cpu()
            if not torch.isfinite(row).all():
                raise RuntimeError('Nonfinite semantic feature for contract %d at %s' % (cid, phase))
            output[cid] = row
            stats['active'] += 1
            stats['encoded_events'] += count
            stats['windows'] += (count + config['window_size'] - 1) // config['window_size']
            if stats['active'] % 250 == 0:
                log('[export] %s active=%d windows=%d events=%d' % (
                    phase, stats['active'], stats['windows'], stats['encoded_events']))
        snapshots[phase], coverage[phase] = output, stats
        log('[snapshot] %s cutoff=%.0f active=%d missing=%d events=%d/%d windows=%d' % (
            phase, cutoff, stats['active'], stats['missing'], stats['encoded_events'],
            stats['visible_events'], stats['windows']))
    return snapshots, coverage


def main():
    if 'BHTG_BUNDLE_DIR' in os.environ:
        from bhtg.v3_training import run_stage
        return run_stage('phase1')

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=None)
    # The actual historical log says batch=32; the old config default was 8.
    parser.add_argument('--batch', type=int, default=None)
    parser.add_argument('--limit', type=int, default=0, help='Optional smoke subset, not the main experiment')
    parser.add_argument('--smoke', action='store_true', help='Tiny model, bounded examples/export; never usable as formal snapshots')
    for name in ('hidden-size', 'layers', 'heads', 'ffn', 'max-len'):
        parser.add_argument('--' + name, type=int, default=None, help='Smoke-only architecture override')
    parser.add_argument('--index-check-only', action='store_true', help='Read and validate input, no training or checkpoint')
    parser.add_argument('--export-window-size', type=int, default=16,
                        help='Complete events per nonoverlapping history window (bounded by model capacity)')
    parser.add_argument('--export-window-batch-size', type=int, default=4,
                        help='Windows encoded simultaneously; never limits history coverage')
    args = parser.parse_args()
    if not args.smoke and any(getattr(args, name) is not None for name in ('hidden_size', 'layers', 'heads', 'ffn', 'max_len')):
        parser.error('architecture overrides require --smoke')
    args.epochs = args.epochs if args.epochs is not None else 1 if args.smoke else SEM_EPOCHS
    args.batch = args.batch if args.batch is not None else 2 if args.smoke else 32
    if args.smoke and not args.limit:
        args.limit = 4
    is_smoke = bool(args.limit or args.smoke)
    semantic_config = {
        'd_model': args.hidden_size if args.hidden_size is not None else 16 if args.smoke else SEM_BERT_HIDDEN,
        'num_layers': args.layers if args.layers is not None else 1 if args.smoke else SEM_BERT_LAYERS,
        'num_heads': args.heads if args.heads is not None else 2 if args.smoke else SEM_BERT_HEADS,
        'ffn': args.ffn if args.ffn is not None else 32 if args.smoke else SEM_BERT_FFN,
        'max_seq_len': args.max_len if args.max_len is not None else 24 if args.smoke else SEM_MAX_LEN,
    }
    if (any(value < 1 for value in semantic_config.values()) or semantic_config['max_seq_len'] < 9
            or semantic_config['d_model'] % semantic_config['num_heads']):
        parser.error('positive model dimensions, hidden-size divisible by heads, and max-len >= 9 required')
    if args.epochs < 1 or args.batch < 1 or args.limit < 0:
        parser.error('epochs/batch must be positive and limit nonnegative')
    if args.export_window_size < 1 or args.export_window_batch_size < 1:
        parser.error('export window size and batch size must be positive')
    require_graph_binding()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    index = load_causal_index()
    cid_edges, node_types = index['cid_edges'], index['node_types']
    vocab = legacy_vocab(node_types)
    train_edges = {cid: [edge for edge in edges if edge['timestamp'] < index['train_cutoff']]
                   for cid, edges in cid_edges.items()}
    instances = build_nsp_instances(train_edges, sorted(train_edges), vocab,
                                    window=SEM_NSP_WINDOW, nsp_prob=SEM_NSP_PROB)
    if args.limit:
        instances = instances[:args.limit]
    n_pos = sum(row['flag'] == 0 for row in instances) if AMOUNT_EXPERIMENT else sum(flag == 0 for _, _, _, flag in instances)
    print('Event-time input: tokens=%d vocab=%d instances=%d pos=%d neg=%d' % (
        len(cid_edges), vocab.vocab_size, len(instances), n_pos, len(instances) - n_pos))
    print('Observation cutoffs: train=%.0f val=%.0f; amounts=%s' % (
        index['train_cutoff'], index['val_cutoff'], AMOUNT_POLICY))
    if args.index_check_only:
        print('Input validation completed; no training or artifacts created.')
        return
    if len(instances) < (1 if AMOUNT_EXPERIMENT else 2):
        raise RuntimeError('Too few NSP instances')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    output_suffix = '_smoke' if is_smoke else ''
    log_path = os.path.join(CHECKPOINT_DIR, 'phase1_lmae_log_strict%s.txt' % output_suffix)
    with open(log_path, 'w', encoding='utf-8') as handle:
        handle.write('%s; amount policy=%s; exact CSV timestamps; amount preprocessing causal=%s\n'
                     % (PROTOCOL, AMOUNT_POLICY, 'false' if AMOUNT_EXPERIMENT else 'not_used'))

    def log(message):
        print(message)
        with open(log_path, 'a', encoding='utf-8') as handle:
            handle.write(message + '\n')

    sem = build_semantic_lmae(
        vocab.vocab_size, **semantic_config,
        temperature=SEM_CTL_TEMP, momentum=SEM_CTL_MOMENTUM,
        dropout=0.1, out_dim=HIDDEN_DIM,
        semantic_amount_mode=SEMANTIC_AMOUNT_MODE,
        amount_mlp_hidden=AMOUNT_MLP_HIDDEN, amount_mask_rate=AMOUNT_MASK_RATE,
        amount_loss_weight=AMOUNT_LOSS_WEIGHT, vocab_mapping=vocab.token_to_id,
    ).to(DEVICE)
    trainable_parameters = [parameter for parameter in sem.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters,
                                  lr=1e-4, weight_decay=0.01)
    log('Epoch,TrainLoss,MLM,NSP,CT,MLMAcc,Steps')
    for epoch in range(args.epochs):
        sem.train()
        order = np.random.RandomState(SEED + epoch).permutation(len(instances))
        totals = np.zeros(5, dtype=np.float64)
        steps = 0
        for start in range(0, len(order), args.batch):
            batch = [instances[j] for j in order[start:start + args.batch]]
            if AMOUNT_EXPERIMENT:
                tensors = collate(batch, vocab, max_len=semantic_config['max_seq_len'])
                tensors = {name: value.to(DEVICE) if torch.is_tensor(value) else value for name, value in tensors.items()}
                losses = sem.forward_continuous(tensors, vocab, SEM_MASK_RATE)
                mlm, nsp = losses['mlm'], losses['nsp']
                contrastive = losses['contrastive'] + sem.projection_loss_weight * losses['projection']
                correct, count = losses['mlm_correct'], losses['mlm_count']
                loss = losses['loss']
            else:
                truth, segments = collate(batch, vocab, max_len=semantic_config['max_seq_len'])
                truth, masked, selected = make_mlm_input(truth, vocab, SEM_MASK_RATE)
                attention = (truth != vocab.PAD).float()
                flags = torch.tensor([row[3] for row in batch], dtype=torch.long)
                tensors = [truth, masked, segments, selected, attention, selected.float(), flags]
                mlm, nsp, contrastive, correct, count = sem(*[t.to(DEVICE) for t in tensors])
                loss = mlm + nsp + contrastive
            if not torch.isfinite(loss):
                log('[warn] nonfinite loss skipped')
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 3.0)
            optimizer.step()
            sem._ema_update()
            totals += [loss.item(), mlm.item(), nsp.item(), contrastive.item(), correct / max(count, 1)]
            steps += 1
            if steps == 1 or steps % 100 == 0:
                log('[progress] epoch=%d step=%d/%d loss=%.4f' % (
                    epoch + 1, steps, (len(instances) + args.batch - 1) // args.batch,
                    loss.item()))
                if AMOUNT_EXPERIMENT:
                    log('[amount] count=%d loss=%.6f MAE=%s' % (losses['amount_count'], losses['amount'].item(),
                        'unavailable' if losses['amount_mae'] is None else '%.6f' % losses['amount_mae'].item()))
        if not steps:
            raise RuntimeError('No finite Phase1 optimizer update')
        values = totals / steps
        log('%d,%.4f,%.4f,%.4f,%.4f,%.4f,%d' % (epoch + 1, *values, steps))

    graph = dgl.load_graphs(GRAPH_BIN)[0][0]
    sem.eval()
    smoke_cids = {row['cid'] if AMOUNT_EXPERIMENT else row[0] for row in instances}
    snapshots, coverage = export_semantic_snapshots(
        sem, vocab, index, graph.num_nodes('contract'),
        window_size=args.export_window_size, window_batch_size=args.export_window_batch_size,
        cids=smoke_cids if AMOUNT_EXPERIMENT and is_smoke else None, log=log)
    metadata = {'protocol': PROTOCOL, 'strict_temporal': True,
                'cutoffs': {phase: index[phase + '_cutoff'] for phase in ('train', 'val')},
                'training_cutoff': index['train_cutoff'], 'projection_trained': True,
                'amount_policy': AMOUNT_POLICY, 'source': index['source'],
                'smoke_subset': is_smoke, 'instance_limit': args.limit,
                'model_config': semantic_config,
                'history_encoding': sem.history_encoding_config(args.export_window_size),
                'history_coverage': coverage, 'export_window_batch_size': args.export_window_batch_size}
    if AMOUNT_EXPERIMENT:
        metadata.update(amount_experiment=experiment_metadata(), semantic_amount_mode=SEMANTIC_AMOUNT_MODE,
                        vocab=dict(vocab.token_to_id), vocab_version='structured_amount_v2',
                        config={'amount_mask_rate': sem.amount_mask_rate,
                                'amount_loss_weight': sem.amount_loss_weight,
                                'amount_mlp_hidden': sem.amount_mlp_hidden,
                                'hidden_size': sem.embed_dim, 'output_dim': HIDDEN_DIM},
                        event_time_policy='strict_before_cutoff', amount_transform='identity',
                        amount_preprocessing_causal=False,
                        evaluation_scope='retrospective_saved_preprocessed_amounts')
    torch.save({'embeddings': snapshots, **metadata},
               os.path.join(CHECKPOINT_DIR, 'h_sem_temporal_lmae_strict%s.pt' % output_suffix))
    # Keep the old tensor filename as the train snapshot for downstream tools.
    torch.save(snapshots['train'], os.path.join(CHECKPOINT_DIR, 'h_sem_all_lmae_strict%s.pt' % output_suffix))
    checkpoint = {
        'model_state': copy.deepcopy(sem.model.state_dict()),
        'out_proj': copy.deepcopy(sem.out_proj.state_dict()),
        'anchor_bert': copy.deepcopy(sem.anchor_bert.state_dict()),
        'anchor_out_proj': copy.deepcopy(sem.anchor_out_proj.state_dict()),
        'cfg': sem.cfg.to_dict(), 'vocab': vocab.token_to_id,
        'epoch': args.epochs, 'strict_temporal': True,
        'train_cutoff': index['train_cutoff'],
        'projection_loss_weight': sem.projection_loss_weight,
        'projection_temperature': sem.projection_temperature,
        **metadata,
    }
    if AMOUNT_EXPERIMENT:
        checkpoint = semantic_checkpoint(sem, vocab, epoch=args.epochs,
                                         **{key: value for key, value in metadata.items() if key != 'vocab'})
    torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, 'phase1_sem_lmae_strict%s.pt' % output_suffix))
    with open(os.path.join(CHECKPOINT_DIR, 'phase1_builder_lmae_strict%s.pkl' % output_suffix), 'wb') as handle:
        pickle.dump({'cid_to_addr': index['cid_to_addr'], 'vocab': vocab.token_to_id,
                     'train_cutoff': index['train_cutoff'], **metadata}, handle)
    log('[done] trained projection and frozen train/val snapshots exported')


if __name__ == '__main__':
    main()
