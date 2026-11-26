"""
Phase 2: Edge-Prediction Pre-training (RugScreener-style).

Inputs are sampled strictly before each temporal batch, with point-in-time features.
The unlabeled protocol restricts the input history and link targets to unlabeled events.

Usage: python run_phase2.py
"""
import dgl, torch
import sys, os, copy, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))

import torch.nn as nn
from bhtg.config import *
from bhtg.action_csv import model_edges, require_graph_binding
from bhtg.encoder.graph_encoder import GraphEncoder, ENCODER_PROTOCOL
from bhtg.encoder.link_predictor import EdgePredictor
from bhtg.sampler import sample_token_subgraph, SAMPLER_PROTOCOL
from bhtg.node_feature_provider import feature_spec, make_feature_provider
from bhtg.retrospective_protocol import EXPERIMENT_PROTOCOL, unlabeled_phase2_inputs
from bhtg.temporal_time import conservative_timestamps
from bhtg.training_protocol import PROTOCOL_VERSION
from bhtg.losses import GraphContrastiveLoss
from bhtg.amount_protocol import experiment_metadata
from bhtg.amount_edge_features import graph_amount_fields

# ---- Fix random seeds for reproducibility ----
import numpy as np
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---- 冒烟/消融覆盖 (env): 短跑验证用, 默认全量 ----
use_ssl = USE_SSL and os.environ.get('PHASE2_SSL', '1') != '0'
max_batches = int(os.environ.get('PHASE2_MAX_BATCHES', 600))
EPOCHS = int(os.environ.get('PHASE2_EPOCHS', EPOCHS_PRETRAIN_GRAPH))
strict_temporal = (STRICT_TEMPORAL and
                   os.environ.get('PHASE2_STRICT_TEMPORAL', '1') != '0')
if not strict_temporal:
    raise RuntimeError('The corrected protocol requires strict temporal pretraining')

# ---- model-level homogenization 消融 (env HOMO_ABLATION=1 或 config) ----
homo_ablation = HOMO_ABLATION or os.environ.get('HOMO_ABLATION', '0') == '1'

# ---- 逐目标节点关系平衡采样 (env 可覆盖 config) ----
relation_balanced = (RELATION_BALANCED_SAMPLING or
                     os.environ.get('RELATION_BALANCED_SAMPLING', '0') == '1')
relation_quotas = RELATION_SAMPLE_QUOTAS if relation_balanced else None

# ===== P1: 节点级对比自监督 (论文 PhishEye §3.3) =====
class ProjectionHead(nn.Module):
    """InfoNCE 前的 2 层 FNN 投影头 (论文 Eq.7)。仅 phase2 使用, 不进入 phase3。"""
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim),
        )

    def forward(self, x):
        return self.net(x)


def build_view(sub_g, drop_rate, mask_rate, rng):
    """
    从子图构造一个增强视图: 边 Bernoulli drop + 特征维度置零 (论文 20%/20%)。

    节点 ID 保序: num_nodes_dict 与 sub_g 完全一致 → 视图与 sub_g 节点对齐,
    seed_in_subg / _gid_maps / _all_gids 直接复用。
    """
    dgl_edge_dict, edge_feat = {}, {}
    for etype in sub_g.canonical_etypes:
        ne = sub_g.num_edges(etype)
        if ne == 0:
            dgl_edge_dict[etype] = sub_g.edges(etype=etype)
            edge_feat[etype] = dict(sub_g.edges[etype].data)
            continue
        s, d = sub_g.edges(etype=etype)
        keep = torch.as_tensor(rng.rand(ne) > drop_rate, dtype=torch.bool)
        if not bool(keep.any()):
            keep[0] = True  # 至少保留 1 条边, 避免关系整型消失
        dgl_edge_dict[etype] = (s[keep], d[keep])
        edge_feat[etype] = {fk: sub_g.edges[etype].data[fk][keep]
                            for fk in sub_g.edges[etype].data}

    view = dgl.heterograph(
        dgl_edge_dict,
        num_nodes_dict={nt: sub_g.num_nodes(nt) for nt in sub_g.ntypes},
    )
    for nt in sub_g.ntypes:
        for fk in sub_g.nodes[nt].data:
            view.nodes[nt].data[fk] = sub_g.nodes[nt].data[fk].clone()
    for etype, feats in edge_feat.items():
        for fk, val in feats.items():
            view.edges[etype].data[fk] = val

    # 特征 mask: 每节点 (N, D) 独立 Bernoulli, 20% 维度置零
    for nt in view.ntypes:
        feat = view.nodes[nt].data.get('feat', None)
        if feat is None or feat.numel() == 0:
            continue
        m = (rng.rand(*feat.shape) > mask_rate).astype('float32')
        view.nodes[nt].data['feat'] = feat * torch.from_numpy(m)

    view._gid_maps = getattr(sub_g, '_gid_maps', {})
    view._all_gids = getattr(sub_g, '_all_gids', {})
    view._cid_map = getattr(sub_g, '_cid_map', {})
    view._sampling_hops = sub_g._sampling_hops
    return view


def temporal_blocks(start, stop, batch_size, max_blocks=None, seed=None):
    count = (stop - start + batch_size - 1) // batch_size
    block_ids = range(count)
    if max_blocks is not None:
        rng = np.random.RandomState(seed)
        block_ids = np.sort(rng.choice(count, min(max_blocks, count), replace=False))
    for block in block_ids:
        first = start + int(block) * batch_size
        yield torch.arange(first, min(first + batch_size, stop), dtype=torch.long)


def main():
    if 'BHTG_BUNDLE_DIR' in os.environ:
        from bhtg.v3_training import run_stage
        return run_stage('phase2')

    require_graph_binding()
    unlabeled_only = os.environ.get('BHTG_EXPERIMENT_PROTOCOL') == EXPERIMENT_PROTOCOL
    node_feature_protocol, node_feature_names = feature_spec()
    amount_meta = experiment_metadata()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    BATCH_SIZE = 32
    PHASE2_RUN_TAG = os.environ.get('PHASE2_TAG', '')
    if not PHASE2_RUN_TAG:
        _phase2_tags = ['causal_v2']
        if relation_balanced:
            _phase2_tags.append('relbal')
        if strict_temporal:
            _phase2_tags.append('strict')
        PHASE2_RUN_TAG = '_'.join(_phase2_tags)
    elif not (PHASE2_RUN_TAG.endswith('_v2') or 'causal_v2' in PHASE2_RUN_TAG):
        PHASE2_RUN_TAG += '_causal_v2'
    _phase2_suffix = ('_%s' % PHASE2_RUN_TAG) if PHASE2_RUN_TAG else ''
    LOG_PATH   = os.path.join(CHECKPOINT_DIR, 'phase2_log%s.txt' % _phase2_suffix)
    CKPT_GRAPH = os.path.join(CHECKPOINT_DIR, 'phase2_graph%s.pt' % _phase2_suffix)
    if NODE_FEATURE_MODE == 'behavior9' and (os.path.exists(CKPT_GRAPH) or os.path.exists(LOG_PATH)):
        raise RuntimeError('Existing behavior9 Phase2 outputs require a fresh BHTG_OUTPUT_DIR; '
                           'Phase2 does not support resume')

    def log(msg):
        print(msg)
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('')

    log('=' * 60)
    log('[node_features] protocol=%s input_dim=%d names=%s' % (
        node_feature_protocol, len(node_feature_names), ','.join(node_feature_names)))
    if amount_meta:
        import json
        log('[amount] %s' % json.dumps(amount_meta, ensure_ascii=False, sort_keys=True))
    source_description = 'unlabeled only' if unlabeled_only else 'labeled + unlabeled'
    log('Phase 2: Edge-Prediction Pre-training (%s)%s' % (
        source_description, ' + node-level InfoNCE (P1)' if use_ssl else ''))
    if unlabeled_only:
        log('[protocol] %s; checkpoint selection=validation_link_bce' % EXPERIMENT_PROTOCOL)
    log('  BATCH=%d  NEIGHBORS=%d  Epochs=%d  SSL=%s' % (BATCH_SIZE, N_NEIGHBORS, EPOCHS, use_ssl))
    log('  HOMO_ABLATION=%s (all node/edge types share proj/conv/λ/norm)' % homo_ablation)
    log('  RELATION_BALANCED=%s quotas=%s tag=%s' % (
        relation_balanced, str(relation_quotas), PHASE2_RUN_TAG or '<default>'))
    if unlabeled_only:
        log('  STRICT_TEMPORAL=True (chronological splits and batches; cutoff=batch min)')
    else:
        log('  STRICT_TEMPORAL=%s (train-only edges; chronological batches; cutoff=batch min)' %
            strict_temporal)
    log('=' * 60)

    # ---- Load graph ----
    gs, _ = dgl.load_graphs(GRAPH_BIN)
    g = gs[0]
    del gs

    # ---- Build global→local node ID mappings (all types) ----
    gid_to_lid = {}
    total_per_type = {}
    for ntype in g.ntypes:
        path = os.path.join(BHTG_DATA_DIR, '%s_gids.pt' % ntype)
        if os.path.exists(path):
            gids = torch.load(path)
            gid_to_lid[ntype] = {gid: lid for lid, gid in enumerate(gids)}
            total_per_type[ntype] = len(gids)
            log('  %s_gids.pt: %d entries' % (ntype, len(gids)))
        else:
            log('  [WARN] %s_gids.pt not found — graph needs rebuild' % ntype)

    if len(gid_to_lid) < 2:
        log('[ERROR] Missing node ID mappings — rebuild graph first')
        sys.exit(1)

    val_cutoff = None
    if unlabeled_only:
        g, feature_provider, train_cutoff, val_cutoff = unlabeled_phase2_inputs(g)

    # Collect contract-target graph edges as link examples. The opt-in graph
    # already contains only unlabeled events, including its context edges.
    all_src_l, all_dst_l, all_ts_l = [], [], []
    all_src_ntype = []

    for etype in g.canonical_etypes:
        if etype[2] != 'contract':
            continue
        n = g.num_edges(etype)
        if n < (1 if unlabeled_only else 2):
            continue
        s, d = g.edges(etype=etype)
        t = conservative_timestamps(g.edges[etype].data['timestamp'])
        all_src_l.append(s.tolist())
        all_dst_l.append(d.tolist())
        all_ts_l.append(t.tolist())
        all_src_ntype += [etype[0]] * len(s)
        log('  %s: %d edges' % (str(etype), len(s)))

    # Flatten
    all_src_l = sum(all_src_l, [])
    all_dst_l = sum(all_dst_l, [])
    all_ts_l  = sum(all_ts_l, [])
    all_src = torch.tensor(all_src_l, dtype=torch.long)
    all_dst = torch.tensor(all_dst_l, dtype=torch.long)
    all_ts  = torch.tensor(all_ts_l, dtype=torch.float64)

    # The original protocol scopes training to the classification cutoff. The
    # unlabeled protocol retains the entire positive pool for chronological evaluation.
    # In both protocols each batch is a contiguous time block.
    if unlabeled_only:
        order = torch.from_numpy(np.argsort(all_ts.numpy(), kind='stable'))
    elif strict_temporal:
        _ts_df = model_edges(usecols=['timestamp'], legacy_path=LABELED_EDGES_CSV)
        train_cutoff = float(np.quantile(_ts_df['timestamp'].values, 0.70))
        del _ts_df
        keep = all_ts < train_cutoff
        keep_idx = keep.nonzero(as_tuple=True)[0]
        all_src = all_src[keep_idx]
        all_dst = all_dst[keep_idx]
        all_ts = all_ts[keep_idx]
        all_src_ntype = [all_src_ntype[int(i)] for i in keep_idx.tolist()]
        order = torch.argsort(all_ts)
    else:
        train_cutoff = None
    if unlabeled_only or strict_temporal:
        all_src = all_src[order]
        all_dst = all_dst[order]
        all_ts = all_ts[order]
        all_src_ntype = [all_src_ntype[int(i)] for i in order.tolist()]
    total_edges = len(all_src)
    train_end = total_edges
    if unlabeled_only:
        cutoffs = torch.tensor([train_cutoff, val_cutoff], dtype=all_ts.dtype)
        train_end, val_end = torch.searchsorted(all_ts, cutoffs).tolist()
        split_counts = {'train': train_end, 'validation': val_end - train_end,
                        'test': total_edges - val_end}
        if min(split_counts.values()) == 0:
            raise RuntimeError('Empty chronological Phase2 split: %s' % split_counts)
        log('[link split] train < %.6f; %.6f <= validation < %.6f; test >= %.6f' % (
            train_cutoff, train_cutoff, val_cutoff, val_cutoff))
        log('[link split] train=%d validation=%d test=%d eligible unlabeled contract-target edges' % (
            split_counts['train'], split_counts['validation'], split_counts['test']))
    else:
        if total_edges == 0:
            raise RuntimeError('No visible Phase2 training events')
        feature_provider = make_feature_provider(g, train_cutoff)
        log('Total seed pool: %d graph events (labeled + unlabeled)' % total_edges)
        if strict_temporal:
            log('  strict train_cutoff=%.0f; timestamp range=[%.0f, %.0f]' % (
                train_cutoff, all_ts.min().item(), all_ts.max().item()))

    # ---- 释放读图期中间大对象 (本机 RAM 紧张: 稳态 RSS 越低, 训练期撞 alloc 干旱概率越小) ----
    # 保留: all_src/all_dst/all_ts/all_src_ntype (训练循环要用), g
    import gc
    del gid_to_lid, total_per_type, all_src_l, all_dst_l, all_ts_l
    gc.collect()

    # ---- Model ----
    node_feat_dims = {nt: len(node_feature_names) for nt in g.ntypes}

    graph_encoder = GraphEncoder(
        node_feat_dims=node_feat_dims, hidden_dim=HIDDEN_DIM,
        out_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
        etypes=g.canonical_etypes, lambda_init=LAMBDA_INIT, dropout=DROPOUT,
        homo_ablation=homo_ablation,
        edge_feature_fields=graph_amount_fields(MEMORY_AMOUNT_MODE) or None,
    ).to(DEVICE)

    predictor = EdgePredictor(HIDDEN_DIM).to(DEVICE)

    lambda_params = graph_encoder.get_lambda_params()
    other_params  = [p for p in graph_encoder.parameters() if p not in set(lambda_params)]

    # ---- P1: 投影头 + 对比损失 (仅 phase2, 不进 phase3) ----
    proj_head = ProjectionHead(HIDDEN_DIM).to(DEVICE) if use_ssl else None
    contrastive = GraphContrastiveLoss(temperature=SSL_TEMP).to(DEVICE) if use_ssl else None

    param_groups = [
        {'params': other_params,                 'lr': LEARNING_RATE},
        {'params': list(predictor.parameters()), 'lr': LEARNING_RATE},
        {'params': lambda_params,                'lr': LAMBDA_LR},
    ]
    if use_ssl:
        param_groups.append({'params': list(proj_head.parameters()), 'lr': LEARNING_RATE})
    optimizer = torch.optim.Adam(param_groups)

    bce = nn.BCEWithLogitsLoss()

    rng_views = np.random.RandomState(SEED * 7919)
    fallback_rows = []
    if unlabeled_only:
        # Finding the first two distinct destinations once is sufficient to
        # supply a prior different target to any one-contract evaluation block.
        different = (all_dst != all_dst[0]).nonzero(as_tuple=True)[0]
        fallback_rows = [0] + ([int(different[0])] if len(different) else [])

    def link_losses(idx, with_ssl, minimum_positives=4, negative_generator=None):
        if len(idx) < minimum_positives:
            return None

        # Every dst in the seed pool is a contract local ID.  Include src only
        # when its actual node type is contract.  The old numeric '< num_contract'
        # check incorrectly treated low EOA/pool local IDs as contract IDs.
        batch_src_types = [all_src_ntype[int(edge_i)] for edge_i in idx.tolist()]
        contract_src = [
            int(all_src[idx[j]]) for j, ntype in enumerate(batch_src_types)
            if ntype == 'contract'
        ]
        contract_src = torch.tensor(contract_src, dtype=torch.long)
        seed_ids = torch.cat([all_dst[idx], contract_src]).unique()

        if len(seed_ids) < 2:
            if negative_generator is None:
                return None
            for row in fallback_rows:
                if all_ts[row] < all_ts[idx[0]] and all_dst[row] != seed_ids[0]:
                    seed_ids = torch.cat([seed_ids, all_dst[row:row + 1]])
                    break
            else:
                return None
        # Strict: every positive edge in this batch has ts >= batch_t, while the
        # sampler uses ts < batch_t. Therefore no target edge or later context
        # can enter the input graph. Legacy mode retains the old max rule.
        batch_t = (all_ts[idx].min().item() if strict_temporal
                   else all_ts[idx].max().item())
        seed_nodes = {}
        for edge_index, ntype in zip(idx.tolist(), batch_src_types):
            seed_nodes.setdefault(ntype, []).append(int(all_src[edge_index]))
        sub_g, seed_in_subg = sample_token_subgraph(
            g, seed_ids, current_time=batch_t, k=N_NEIGHBORS,
            n_hop=graph_encoder.required_sampling_hops,
            relation_quotas=relation_quotas,
            feature_provider=feature_provider,
            seed_nodes=seed_nodes,
        )
        if sub_g is None or seed_in_subg is None or len(seed_in_subg) < 2:
            return None

        # ---- P1: 两个增强视图 (CPU 建, 节点 ID 保序对齐 sub_g) ----
        v1 = v2 = None
        if with_ssl:
            v1 = build_view(sub_g, SSL_DROP_RATE, SSL_MASK_RATE, rng_views)
            v2 = build_view(sub_g, SSL_DROP_RATE, SSL_MASK_RATE, rng_views)

        sub_g = sub_g.to(DEVICE)
        seed_in_subg = seed_in_subg.to(DEVICE)

        h_all, token_emb = graph_encoder(sub_g, current_time=batch_t, return_all=True)
        if token_emb is None or token_emb.shape[0] == 0:
            return None

        # ---- P1: 节点级 InfoNCE (视图间对齐 seed token 表示) ----
        ssl_loss = None
        if with_ssl:
            h1 = graph_encoder(v1.to(DEVICE), current_time=batch_t)
            h2 = graph_encoder(v2.to(DEVICE), current_time=batch_t)
            if h1 is not None and h2 is not None and h1.shape[0] >= 2 and h2.shape[0] >= 2:
                z1 = proj_head(h1[seed_in_subg])
                z2 = proj_head(h2[seed_in_subg])
                ssl_loss = contrastive(z1, z2)

        gid_maps = getattr(sub_g, '_gid_maps', {})
        cid_map = gid_maps.get('contract', {})
        if not cid_map:
            return None

        pos_src_emb, pos_dst = [], []
        for j in range(len(idx)):
            sc = int(all_src[idx[j]])
            dc = int(all_dst[idx[j]])
            sntype = batch_src_types[j]
            src_map = gid_maps.get(sntype, {})
            if sc in src_map and dc in cid_map and sntype in h_all:
                pos_src_emb.append(h_all[sntype][src_map[sc]])
                pos_dst.append(cid_map[dc])

        if len(pos_src_emb) < minimum_positives:
            return None

        pos_dst = torch.tensor(pos_dst, device=DEVICE)
        h_src = torch.stack(pos_src_emb, dim=0)
        h_dst = token_emb[pos_dst]

        B = len(pos_src_emb)
        # Only query seeds have complete encoder-depth receptive fields.
        # Other retained contracts can be boundary neighbours whose embeddings
        # change when unrelated seeds expand additional incoming history.
        candidates = seed_in_subg.unique(sorted=True)
        positive_positions = torch.searchsorted(candidates, pos_dst)
        if not torch.equal(candidates[positive_positions], pos_dst):
            raise RuntimeError('Positive endpoints must be complete query seeds')
        draw_device = DEVICE if negative_generator is None else 'cpu'
        draw = torch.randint(0, len(candidates) - 1, (B,), device=draw_device,
                             generator=negative_generator).to(DEVICE)
        draw += (draw >= positive_positions).long()
        neg_idx = candidates[draw]
        pos_score = predictor(h_src, h_dst)
        neg_score = predictor(h_src, token_emb[neg_idx])

        loss_bce = (bce(pos_score, torch.ones_like(pos_score)) +
                    bce(neg_score, torch.zeros_like(neg_score)))
        return loss_bce, ssl_loss, B

    @torch.no_grad()
    def evaluate_links(start, stop, split_name, seed):
        graph_encoder.eval()
        predictor.eval()
        negative_generator = torch.Generator().manual_seed(seed)
        combined_bce, count = 0.0, 0
        for idx in temporal_blocks(start, stop, BATCH_SIZE):
            result = link_losses(idx, False, minimum_positives=1,
                                 negative_generator=negative_generator)
            if result is None or result[2] != len(idx):
                raise RuntimeError('Cannot score every Phase2 %s edge in block beginning at %d' %
                                   (split_name, int(idx[0])))
            bce_sum, _, n_edges = result
            if not bool(torch.isfinite(bce_sum)):
                raise RuntimeError('Nonfinite Phase2 %s edge loss' % split_name)
            combined_bce += 0.5 * bce_sum.item() * n_edges
            count += n_edges
        return combined_bce / count

    log('Epoch,TrainBCE,TrainSSL,TrainTotal,Batches,lam_burn,lam_create,lam_mint,lam_transfer')
    best_loss = float('inf')
    best_epoch = 0
    best_graph_state = None
    best_proj_state = None
    best_predictor_state = None

    for epoch in range(EPOCHS):
        graph_encoder.train()
        predictor.train()
        if proj_head is not None:
            proj_head.train()
        bce_total, ssl_total, n_batches = 0.0, 0.0, 0

        if strict_temporal:
            batch_indices = temporal_blocks(0, train_end, BATCH_SIZE, max_batches, SEED + epoch)
        else:
            perm = torch.randperm(total_edges)
            batch_indices = [perm[start:start + BATCH_SIZE]
                             for start in range(0, total_edges, BATCH_SIZE)][:max_batches]

        for idx in batch_indices:
            result = link_losses(idx, use_ssl)
            if result is None:
                continue
            loss_bce, ssl_loss, _ = result
            loss = loss_bce
            if ssl_loss is not None:
                loss = loss + ALPHA_SSL * ssl_loss
                ssl_total += ssl_loss.item()
            if unlabeled_only and not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite Phase2 training loss')

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(graph_encoder.parameters()) + (list(proj_head.parameters()) if proj_head else []),
                max_norm=1.0)
            optimizer.step()
            for lp in lambda_params:
                lp.data.clamp_(min=0.001, max=1.0)

            bce_total += loss_bce.item()
            n_batches += 1
            if n_batches == 1:
                log('[train] epoch %d first optimizer update complete' % (epoch + 1))

        avg_bce = bce_total / max(n_batches, 1)
        avg_ssl = ssl_total / max(n_batches, 1)
        avg_total = avg_bce + ALPHA_SSL * avg_ssl  # 与每 batch 实际优化目标一致
        lams = graph_encoder.get_lambda_values()
        log('%d,%.4f,%.4f,%.4f,%d,%.4f,%.4f,%.4f,%.4f' % (
            epoch + 1, avg_bce, avg_ssl, avg_total, n_batches,
            lams['burn'], lams['create'], lams['mint'], lams['transfer'],
        ))

        selection_loss = avg_total
        if unlabeled_only:
            if n_batches == 0:
                raise RuntimeError('No valid Phase2 training links in epoch %d' % (epoch + 1))
            selection_loss = evaluate_links(train_end, val_end, 'validation', SEED + 101)
            log('[validation link BCE] epoch=%d loss=%.6f edges=%d' % (
                epoch + 1, selection_loss, split_counts['validation']))

        if n_batches > 0 and selection_loss < best_loss:
            best_loss = selection_loss
            best_epoch = epoch
            best_graph_state = copy.deepcopy(graph_encoder.state_dict())
            if proj_head is not None:
                best_proj_state = copy.deepcopy(proj_head.state_dict())
            if unlabeled_only:
                best_predictor_state = copy.deepcopy(predictor.state_dict())

    if best_graph_state is None:
        raise RuntimeError('No finite Phase2 training update; refusing to save empty weights')
    save_dict = {
        'protocol': PROTOCOL_VERSION,
        'amount_experiment': amount_meta,
        'smoke_subset': '--smoke' in sys.argv,
        'encoder_protocol': ENCODER_PROTOCOL,
        'graph_architecture_version': graph_encoder.architecture_version,
        'graph_edge_feature_fields': list(graph_encoder.edge_feature_fields),
        'sampling_hops': graph_encoder.required_sampling_hops,
        'sampler_protocol': SAMPLER_PROTOCOL,
        'node_feature_protocol': node_feature_protocol,
        'node_feature_names': list(node_feature_names),
        'node_feat_dims': node_feat_dims,
        'feature_state': feature_provider.state_dict(),
        'model_state': best_graph_state,
        'epoch': best_epoch + 1,
        'loss': best_loss,
            'relation_balanced': relation_balanced,
            'relation_quotas': relation_quotas,
            'strict_temporal': strict_temporal,
            'train_cutoff': train_cutoff,
            'batch_cutoff_rule': 'min' if strict_temporal else 'max',
    }
    if proj_head is not None:
        save_dict['proj_state'] = best_proj_state
    if unlabeled_only:
        graph_encoder.load_state_dict(best_graph_state)
        predictor.load_state_dict(best_predictor_state)
        test_link_loss = evaluate_links(val_end, total_edges, 'test', SEED + 102)
        log('[test link BCE] selected_epoch=%d loss=%.6f edges=%d' % (
            best_epoch + 1, test_link_loss, split_counts['test']))
        save_dict.update(
            experiment_protocol=EXPERIMENT_PROTOCOL,
            selection_metric='validation_link_bce',
            val_cutoff=val_cutoff,
            val_link_loss=best_loss,
            test_link_loss=test_link_loss,
            predictor_state=best_predictor_state,
            link_split=split_counts,
        )
    torch.save(save_dict, CKPT_GRAPH)
    log('\n[Save] %s  (epoch %d, loss %.4f)' % (CKPT_GRAPH, best_epoch + 1, best_loss))
    log('Phase 2 complete.')


if __name__ == "__main__":
    main()
