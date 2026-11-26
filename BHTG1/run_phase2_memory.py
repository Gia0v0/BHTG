"""Time-ordered memory pretraining: predict before update, select by validation BCE.

Each TBPTT window contains consecutive complete timestamp groups. Predictions
read only earlier groups; current events are rolled afterwards. A later loss
can therefore train the GRU that encoded its historical events. No token/rug
labels are loaded for probes or checkpoint selection.
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from bhtg.config import (MEM_EPOCHS, MEM_EDGE_BATCH, MEM_DIM, TEMPORAL_DIM,
                         MEM_LR, MEM_ROLL_BATCH, CHECKPOINT_DIR, DEVICE, SEED,
                         MEMORY_AMOUNT_MODE)
from bhtg.edge_stream import (load_edge_stream, compute_phase_cutoffs,
                              split_masks, build_neg_index, sample_negatives,
                              temporal_batches, iter_time_batches)
from bhtg.memory import HeteroTemporalMemory, memory_amount_inputs
from bhtg.encoder.link_predictor import EdgePredictor


def gather(tables, node_types, ids):
    result = torch.empty(len(ids), tables[0][0].shape[1], device=ids.device)
    for ti, values in tables.items():
        mask = node_types == ti
        result[mask] = values[0][ids[mask]]
    return result


def training_windows(timestamps, indices, batch_size):
    """Keep ties intact and include a later group when the first group is huge."""
    times = timestamps[indices]
    start = 0
    for _, end in iter_time_batches(times, batch_size):
        if end <= start:
            continue
        if end < len(times) and bool(times[start] == times[end - 1]):
            end = int(torch.searchsorted(times, times[end], right=True))
        yield indices[start:end]
        start = end


def causal_window(mem, predictor, stream, indices, neg_index, generator=None):
    """Return loss and a compact differentiable post-window state, without commit."""
    device = next(mem.parameters()).device
    fields = ('src_ntype', 'dst_ntype', 'src_lid', 'dst_lid', 'etype_idx', 'ts', 'value')
    src_nt, dst_nt, src, dst, et, ts, value = [stream[key][indices].to(device) for key in fields]
    amounts = memory_amount_inputs(stream, indices, device, mem.amount_mode)
    neg, valid = sample_negatives(
        neg_index, dst_nt.cpu(), dst.cpu(), ts.cpu(),
        src_ntype=src_nt.cpu(), src_lid=src.cpu(), return_mask=True, generator=generator)
    neg, valid = neg.to(device), valid.to(device)
    global_ids, tables = {}, {}
    local_src, local_dst = torch.empty_like(src), torch.empty_like(dst)
    local_neg = torch.full_like(neg, -1)
    for ti, state in mem.state_tables().items():
        sm, dm, nm = src_nt == ti, dst_nt == ti, (dst_nt == ti) & valid
        ids = torch.unique(torch.cat([src[sm], dst[dm], neg[nm]]), sorted=True)
        global_ids[ti] = ids
        tables[ti] = tuple(tensor[ids].detach().clone() for tensor in state)
        local_src[sm] = torch.searchsorted(ids, src[sm])
        local_dst[dm] = torch.searchsorted(ids, dst[dm])
        local_neg[nm] = torch.searchsorted(ids, neg[nm])
    total, count = None, 0
    _, group_counts = torch.unique_consecutive(ts, return_counts=True)
    start = 0
    for size in group_counts.tolist():
        rows = torch.arange(start, start + size, device=device)
        selected = rows[valid[rows]]
        if len(selected):
            # Both labels use precisely the same pre-event state. An edge with
            # no admissible negative is omitted from the loss but still rolled.
            h_src = gather(tables, src_nt[selected], local_src[selected])
            h_dst = gather(tables, dst_nt[selected], local_dst[selected])
            h_neg = gather(tables, dst_nt[selected], local_neg[selected])
            scores = torch.cat([predictor(h_src, h_dst), predictor(h_src, h_neg)])
            labels = torch.cat([torch.ones(len(selected), 1, device=device),
                                torch.zeros(len(selected), 1, device=device)])
            part = F.binary_cross_entropy_with_logits(scores, labels, reduction='sum')
            total = part if total is None else total + part
            count += len(scores)
        tables = mem.roll_diff(tables, src_nt[rows], dst_nt[rows],
                               local_src[rows], local_dst[rows], et[rows], ts[rows], value[rows],
                               **{key: tensor[rows] for key, tensor in amounts.items()})
        start += size
    return (None if total is None else total / count), count, tables, global_ids


def replay_train(mem, stream, indices):
    mem.reset()
    device = next(mem.parameters()).device
    with torch.no_grad():
        for rows in temporal_batches(stream['ts'], indices, MEM_ROLL_BATCH):
            values = [stream[key][rows].to(device) for key in (
                'src_ntype', 'dst_ntype', 'src_lid', 'dst_lid', 'etype_idx', 'ts', 'value')]
            mem.roll_batch(*values, **memory_amount_inputs(stream, rows, device, mem.amount_mode))


def validation_loss(mem, predictor, stream, indices, neg_index, batch_size):
    """Prequential validation; only past validation events update future history."""
    generator = torch.Generator().manual_seed(SEED + 11)
    mem.eval()
    predictor.eval()
    weighted, count = 0., 0
    with torch.no_grad():
        for rows in temporal_batches(stream['ts'], indices, batch_size):
            loss, n, state, ids = causal_window(mem, predictor, stream, rows, neg_index, generator)
            mem.commit_state(state, ids)
            if loss is not None:
                weighted += float(loss) * n
                count += n
    if count == 0:
        raise RuntimeError('no admissible historical negatives in validation; cannot select a checkpoint')
    return weighted / count


from bhtg.action_csv import require_graph_binding


def main(argv=None):
    if 'BHTG_BUNDLE_DIR' in os.environ:
        from bhtg.v3_training import run_stage
        return run_stage('memory', argv=argv)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=MEM_EPOCHS)
    parser.add_argument('--batch', type=int, default=MEM_EDGE_BATCH)
    parser.add_argument('--smoke', action='store_true',
                        help='cap history to small complete timestamp groups; marks checkpoint as incomplete')
    parser.add_argument('--smoke-train-events', type=int, default=4096)
    parser.add_argument('--smoke-val-events', type=int, default=1024)
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch, args.smoke_train_events, args.smoke_val_events) < 1:
        parser.error('epochs, batch and smoke event limits must be positive')
    if args.smoke and not os.environ.get('BHTG_OUTPUT_DIR'):
        parser.error('--smoke requires an explicit independent BHTG_OUTPUT_DIR')
    require_graph_binding()
    from bhtg.amount_protocol import experiment_metadata
    amount_experiment = experiment_metadata()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'phase2_memory.pt')
    log_path = os.path.join(CHECKPOINT_DIR, 'phase2_memory_log.txt')
    with open(log_path, 'w', encoding='utf-8') as handle:
        handle.write('')

    def log(message):
        print(message)
        with open(log_path, 'a', encoding='utf-8') as handle:
            handle.write(message + '\n')

    stream = load_edge_stream()
    train_cutoff, val_cutoff = compute_phase_cutoffs()
    masks = split_masks(stream, train_cutoff, val_cutoff)
    train_indices = masks['train'].nonzero(as_tuple=True)[0]
    val_indices = masks['val_only'].nonzero(as_tuple=True)[0]
    if args.smoke:
        def prefix_with_ties(indices, limit):
            if len(indices) <= limit:
                return indices
            times = stream['ts'][indices]
            return indices[:int(torch.searchsorted(times, times[limit - 1], right=True))]
        train_indices = prefix_with_ties(train_indices, args.smoke_train_events)
        val_indices = prefix_with_ties(val_indices, args.smoke_val_events)
    neg_index = build_neg_index(stream)
    mem = HeteroTemporalMemory(stream['num_nodes'], mem_dim=MEM_DIM,
                               temporal_dim=TEMPORAL_DIM, amount_mode=MEMORY_AMOUNT_MODE).to(DEVICE)
    predictor = EdgePredictor(MEM_DIM).to(DEVICE)
    parameters = [parameter for model in (mem, predictor) for parameter in model.parameters()
                  if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=MEM_LR)
    log('Phase 2-memory: predict-before-update; checkpoint criterion=validation link BCE')
    log('[protocol] memory=%s value=%s timestamps=%s' % (
        mem.memory_semantics, mem.value_policy, stream['_meta'].get('timestamp_semantics')))
    if amount_experiment is not None:
        log('[amount] event selection is strict before cutoff; saved z-score preprocessing is retrospective, not causal')
        memory_amount_inputs(stream, amount_mode=mem.amount_mode)
    log('[data] train=%d validation=%d train_cutoff=%.6f val_cutoff=%.6f' % (
        len(train_indices), len(val_indices), train_cutoff, val_cutoff))
    best_loss = float('inf')
    for epoch in range(args.epochs):
        mem.train()
        predictor.train()
        mem.reset()
        weighted, count, batches, gru_gradient_steps = 0., 0, 0, 0
        start_time = time.time()
        for rows in training_windows(stream['ts'], train_indices, args.batch):
            optimizer.zero_grad()
            loss, n, state, ids = causal_window(mem, predictor, stream, rows, neg_index)
            if loss is not None:
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('nonfinite causal memory training loss')
                loss.backward()
                gru_gradient_steps += int(any(
                    parameter.grad is not None and bool((parameter.grad != 0).any())
                    for parameter in mem.updaters.parameters()))
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                optimizer.step()
                weighted += float(loss.detach()) * n
                count += n
                batches += 1
            mem.commit_state(state, ids)
        if not count or not gru_gradient_steps:
            raise RuntimeError('training produced no GRU gradient; use recurrent events spanning multiple timestamps')
        # Rebuild train history using the current weights before evaluating.
        replay_train(mem, stream, train_indices)
        val_loss = validation_loss(mem, predictor, stream, val_indices, neg_index, args.batch)
        log('%d train_BCE=%.6f val_BCE=%.6f batches=%d GRU_steps=%d sec=%.1f' % (
            epoch + 1, weighted / count, val_loss, batches, gru_gradient_steps, time.time() - start_time))
        if val_loss < best_loss:
            best_loss = val_loss
            runtime_prefixes = ('memory_tables.', 'last_update_t.', 'seen.')
            torch.save({
                'memory_state': {key: value.detach().cpu().clone()
                                 for key, value in mem.state_dict().items()
                                 if not key.startswith(runtime_prefixes)},
                'predictor_state': {key: value.detach().cpu().clone()
                                    for key, value in predictor.state_dict().items()},
                'epoch': epoch + 1, 'val_link_loss': val_loss,
                'selection_metric': 'validation_link_bce',
                **mem.protocol_metadata(),
                **({'amount_experiment': amount_experiment} if amount_experiment is not None else {}),
                'smoke': args.smoke, 'smoke_subset': args.smoke, 'seed': SEED,
                'graph_fingerprint': stream['_meta'].get('graph_fingerprint'),
                'timestamp_semantics': stream['_meta'].get('timestamp_semantics'),
                'train_cutoff': train_cutoff, 'val_cutoff': val_cutoff,
            }, checkpoint_path)
            log('[save] epoch=%d validation BCE=%.6f %s' % (epoch + 1, val_loss, checkpoint_path))
    log('[done] best validation BCE=%.6f saved=%s' % (best_loss, checkpoint_path))


if __name__ == '__main__':
    main()
