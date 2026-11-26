"""
Phase 3: Joint Fine-tuning — standalone script.

Loads versioned train/val semantic snapshots and the Phase2 graph/scaler,
then runs causal mini-batch fine-tuning. Saves weights by validation BAC.
Validation + test results written to checkpoints/phase3_results.txt.

Usage:
  python run_phase3.py
  python run_phase3.py --use-memory

Evaluation protocols:
  token_stratified  BHTG default: mutually exclusive, class-stratified token split.
The corrected protocol rejects overlapping RugScreener row-index splits.
"""
import dgl, torch, pandas as pd
import sys, os, copy, pickle, json, hashlib
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch.nn as nn
from sklearn.metrics import (average_precision_score, roc_auc_score,
                              balanced_accuracy_score, f1_score, recall_score,
                              confusion_matrix)

from bhtg.config import *
from bhtg.action_csv import model_edges, read_actions, creator_times_from_actions, require_graph_binding
# ---- Fix random seeds for reproducibility ----
torch.manual_seed(SEED)
np.random.seed(SEED)

from bhtg.encoder.graph_encoder import GraphEncoder, ENCODER_PROTOCOL
from bhtg.fusion import CrossModalFusion
from bhtg.losses import FocalLoss
from bhtg.sampler import sample_token_subgraph, SAMPLER_PROTOCOL
from bhtg.node_feature_provider import feature_spec, restore_feature_provider
from bhtg.training_protocol import (
    PROTOCOL_VERSION, model_weights, load_model_weights, validate_semantic_bundle,
    validate_full_semantic_bundle, semantic_history_input_bindings,
)
from bhtg.retrospective_protocol import (
    EXPERIMENT_PROTOCOL, full_history_cutoff, restore_full_history_feature_provider,
)
from bhtg.observation_protocol import ObservationIndex, load_creator_times
from bhtg.phase3_protocol import (
    OBSERVATION_PROTOCOL, FIXED_THRESHOLD, FIXED_THRESHOLD_POLICY, VALIDATION_THRESHOLD_POLICY,
    partition_observed_splits, classification_metrics, select_validation_threshold,
)
from bhtg.amount_protocol import experiment_metadata, validate_amount_metadata, upstream_identity
from bhtg.amount_edge_features import graph_amount_fields
from bhtg.phase3_checkpoint import (
    atomic_torch_save, epoch_checkpoint, restore_epoch, write_validation_history,
    set_replay_phase, begin_memory_replay,
)

def main():
    if 'BHTG_BUNDLE_DIR' in os.environ:
        from bhtg.v3_training import run_stage
        return run_stage('phase3')

    full_history = os.environ.get('BHTG_EXPERIMENT_PROTOCOL') == EXPERIMENT_PROTOCOL
    require_graph_binding()
    node_feature_protocol, node_feature_names = feature_spec()
    amount_meta = experiment_metadata()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    RESUME_PATH = (sys.argv[sys.argv.index('--resume-phase3') + 1]
                   if '--resume-phase3' in sys.argv else None)
    threshold_policy = (VALIDATION_THRESHOLD_POLICY if '--validation-threshold' in sys.argv
                        else FIXED_THRESHOLD_POLICY)

    BATCH_SIZE = 32
    CKPT_SEM   = os.path.join(CHECKPOINT_DIR, 'phase1_sem.pt')
    CKPT_H_SEM = os.path.join(CHECKPOINT_DIR, 'h_sem_full_history.pt' if full_history
                              else 'h_sem_temporal_lmae_strict.pt')
    CKPT_GRAPH = os.path.join(CHECKPOINT_DIR, 'phase2_graph.pt')
    CKPT_BEST  = os.path.join(CHECKPOINT_DIR, 'phase3_best.pt')
    LOG_PATH   = os.path.join(CHECKPOINT_DIR, 'phase3_log.txt')
    RESULT_PATH = os.path.join(CHECKPOINT_DIR, 'phase3_results.txt')

    # --eval-protocol P / env PHASE3_EVAL_PROTOCOL:
    #   token_stratified (默认): token 级分层互斥划分；PHASE3_ALL_TOKENS=1 使用全部标注 token。
    #   rugscreener_code: 复刻 classification.py 的边序号窗口 + 10% test-new-node 删除规则。
    # 注意：RugScreener 原始 CSV 并未按 timestamp 排序，且由边窗口收集出的 token 集可重叠；
    # 该模式用于代码口径对照，日志会显式报告时序单调性、类别数和集合重叠。
    EVAL_PROTOCOL = (sys.argv[sys.argv.index('--eval-protocol') + 1]
                     if '--eval-protocol' in sys.argv
                     else os.environ.get('PHASE3_EVAL_PROTOCOL', 'token_stratified'))
    EVAL_PROTOCOL = EVAL_PROTOCOL.strip().lower()
    if EVAL_PROTOCOL not in ('token_stratified', 'rugscreener_code'):
        raise ValueError('Unknown eval protocol %r; choose token_stratified or rugscreener_code' % EVAL_PROTOCOL)

    # Per-target relation-balanced sampler.  Config defaults off so historical
    # checkpoints remain reproducible; CLI/env enable it for the isolated ablation.
    RELATION_BALANCED = (
        '--relation-balanced' in sys.argv
        or RELATION_BALANCED_SAMPLING
        or os.environ.get('RELATION_BALANCED_SAMPLING', '0') == '1'
    )
    RELATION_QUOTAS = RELATION_SAMPLE_QUOTAS if RELATION_BALANCED else None

    # Strict temporal evaluation requires a Phase1 embedding and Phase2 graph
    # checkpoint trained without post-cutoff/within-batch future context.
    STRICT_RUN = ('--strict-temporal' in sys.argv or
                  os.environ.get('PHASE3_STRICT_TEMPORAL', '1') == '1')
    if not STRICT_RUN:
        raise RuntimeError('The corrected protocol requires strict temporal evaluation')
    if EVAL_PROTOCOL != 'token_stratified':
        raise RuntimeError('The corrected protocol requires disjoint token splits; '
                           'rugscreener_code permits overlapping supervised tokens')

    # --实验 tag (env PHASE3_TAG): 输出文件名加后缀, 避免覆盖生产 checkpoint/log。
    #   用法: PHASE3_TAG=all2673 python run_phase3.py ... → phase3_mem_best_all2673.pt 等
    RUN_TAG = os.environ.get('PHASE3_TAG', '')
    if not RUN_TAG:
        _auto_tags = ['causal_v2']
        if EVAL_PROTOCOL == 'rugscreener_code':
            _auto_tags.append('rugscreener_code')
        if RELATION_BALANCED:
            _auto_tags.append('relbal')
        if STRICT_RUN:
            _auto_tags.append('strict')
        RUN_TAG = '_'.join(_auto_tags)
    elif not (RUN_TAG.endswith('_v2') or 'causal_v2' in RUN_TAG):
        RUN_TAG += '_causal_v2'
    if RUN_TAG:
        LOG_PATH   = os.path.join(CHECKPOINT_DIR, 'phase3_log_%s.txt' % RUN_TAG)
        RESULT_PATH = os.path.join(CHECKPOINT_DIR, 'phase3_results_%s.txt' % RUN_TAG)
        CKPT_BEST  = os.path.join(CHECKPOINT_DIR, 'phase3_best_%s.pt' % RUN_TAG)

    # Relation-balanced Phase2 is a separately trained encoder.  Allow an explicit
    # path, otherwise use its isolated default checkpoint.
    if '--graph-path' in sys.argv:
        CKPT_GRAPH = sys.argv[sys.argv.index('--graph-path') + 1]
    elif os.environ.get('PHASE3_GRAPH_CKPT'):
        CKPT_GRAPH = os.environ['PHASE3_GRAPH_CKPT']
    elif RELATION_BALANCED:
        CKPT_GRAPH = os.path.join(
            CHECKPOINT_DIR,
            'phase2_graph_causal_v2_%s.pt' % ('relbal_strict' if STRICT_RUN else 'relbal'))
    elif STRICT_RUN:
        CKPT_GRAPH = os.path.join(CHECKPOINT_DIR, 'phase2_graph_causal_v2_strict.pt')

    if NODE_FEATURE_MODE == 'behavior9' and not RESUME_PATH:
        stage_prefix = 'phase3_mem' if ('--use-memory' in sys.argv or '--joint-memory' in sys.argv) else 'phase3'
        stage_suffix = ('_%s' % RUN_TAG) if RUN_TAG else ''
        existing = [os.path.join(CHECKPOINT_DIR, '%s_%s%s.%s' % (
            stage_prefix, name, stage_suffix, extension))
            for name, extension in (('best', 'pt'), ('latest', 'pt'), ('log', 'txt'),
                                    ('results', 'txt'), ('results', 'json'), ('validation', 'jsonl'))]
        existing.extend((LOG_PATH, RESULT_PATH,
                         os.path.join(CHECKPOINT_DIR, 'phase3_split%s.pt' % stage_suffix)))
        if any(os.path.exists(path) for path in existing):
            raise RuntimeError('Existing behavior9 Phase3 outputs require --resume-phase3 '
                               'or a fresh BHTG_OUTPUT_DIR')

    def log(msg):
        print(msg)
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    with open(LOG_PATH, 'a' if RESUME_PATH else 'w', encoding='utf-8') as f:
        f.write('')

    # --epochs N 覆盖 (冒烟测试用，默认 EPOCHS_FINETUNE)
    EPOCHS_RUN = int(sys.argv[sys.argv.index('--epochs') + 1]) if '--epochs' in sys.argv else EPOCHS_FINETUNE
    if EPOCHS_RUN < 1:
        raise ValueError('epochs must be positive')

    # --sem-path P 覆盖 h_sem 路径 (对比新旧语义编码器，默认 checkpoints/h_sem_all.pt)
    H_SEM_PATH = sys.argv[sys.argv.index('--sem-path') + 1] if '--sem-path' in sys.argv else CKPT_H_SEM

    log('=' * 60)
    log('Phase 3: Mini-Batch Joint Fine-tuning')
    log('  BATCH=%d HIDDEN=%d Epochs=%d' % (BATCH_SIZE, HIDDEN_DIM, EPOCHS_RUN))
    log('[node_features] protocol=%s input_dim=%d names=%s' % (
        node_feature_protocol, len(node_feature_names), ','.join(node_feature_names)))
    log('=' * 60)

    # Ablation switch: --no-semantic → graph-only (no semantic stream)
    USE_SEMANTIC = not ('--no-semantic' in sys.argv)
    log('[ablation] USE_SEMANTIC=%s' % USE_SEMANTIC)

    # model-level homogenization 消融 (env HOMO_ABLATION=1 或 config): 全类型共享参数。
    # ⚠️ 必须与 phase2 同一设置 — phase2_graph.pt 结构需匹配 (homo state_dict 加载)。
    homo_ablation = HOMO_ABLATION or os.environ.get('HOMO_ABLATION', '0') == '1'
    log('[ablation] HOMO_ABLATION=%s' % homo_ablation)

    # Memory switch: --use-memory → 注入 TGN GRU 记忆 (frozen GRU + 快照)
    # Joint-memory: --joint-memory → 每 epoch 重滚 + GRU BPTT 联合训练 (隐含 --use-memory)
    USE_JOINT = '--joint-memory' in sys.argv
    JOINT_SAMPLE = int(sys.argv[sys.argv.index('--joint-sample') + 1]) if '--joint-sample' in sys.argv else MEM_JOINT_SAMPLE
    JOINT_TAIL_CHUNKS = int(sys.argv[sys.argv.index('--joint-tail-chunks') + 1]) if '--joint-tail-chunks' in sys.argv else MEM_JOINT_TAIL_CHUNKS
    JOINT_STEPS = int(sys.argv[sys.argv.index('--joint-steps') + 1]) if '--joint-steps' in sys.argv else MEM_JOINT_STEPS
    USE_MEMORY = '--use-memory' in sys.argv or USE_JOINT
    if USE_JOINT and (JOINT_STEPS < 1 or JOINT_TAIL_CHUNKS < 1 or JOINT_SAMPLE < 2):
        raise ValueError('joint steps/tail must be positive and joint sample >= 2')
    log('[memory] USE_MEMORY=%s USE_JOINT=%s' % (USE_MEMORY, USE_JOINT))
    log('[sampler] RELATION_BALANCED=%s quotas=%s graph_ckpt=%s' % (
        RELATION_BALANCED, str(RELATION_QUOTAS), CKPT_GRAPH))
    log('[temporal] STRICT_RUN=%s' % STRICT_RUN)
    if amount_meta:
        log('[amount] %s' % json.dumps(amount_meta, ensure_ascii=False, sort_keys=True))
    if USE_MEMORY:
        _suffix = ('_%s' % RUN_TAG) if RUN_TAG else ''
        CKPT_BEST = os.path.join(CHECKPOINT_DIR, 'phase3_mem_best%s.pt' % _suffix)
        LOG_PATH = os.path.join(CHECKPOINT_DIR, 'phase3_mem_log%s.txt' % _suffix)
        RESULT_PATH = os.path.join(CHECKPOINT_DIR, 'phase3_mem_results%s.txt' % _suffix)

    CKPT_LATEST = CKPT_BEST.replace('_best', '_latest')
    VALIDATION_PATH = os.path.splitext(CKPT_BEST)[0].replace('_best', '_validation') + '.jsonl'
    if not RESUME_PATH and (os.path.exists(CKPT_BEST) or os.path.exists(CKPT_LATEST)):
        raise RuntimeError('Existing Phase3 weights require explicit --resume-phase3 or a fresh output directory')

    SPLIT_PATH = os.path.join(
        CHECKPOINT_DIR,
        'phase3_split%s.pt' % (('_%s' % RUN_TAG) if RUN_TAG else ''),
    )
    log('[eval] protocol=%s split_manifest=%s' % (EVAL_PROTOCOL, SPLIT_PATH))

    # ---- 标签 CSV 提前解析 (内存优化, 本机仅 ~2.3GB 可用) ----
    # 80MB labeled_edges.csv 若在图/模型/CUDA 常驻后 parse → pandas "C error: out of memory"。
    # 此处先读完并 del，只保留小产物 (tx_cnt/labels_map/cutoffs)，下游 Plan A 直接复用。
    import pandas as _pd
    _df = model_edges(usecols=['timestamp', 'token_addr', 'label'], legacy_path=LABELED_EDGES_CSV)
    reference_train_cutoff = float(np.quantile(_df['timestamp'].values, 0.70))
    reference_val_cutoff = float(np.quantile(_df['timestamp'].values, 0.85))
    train_cutoff, val_cutoff = reference_train_cutoff, reference_val_cutoff
    if ACTION_CSV:
        # Transaction counts are canonical saved actions, never graph legs / 2.
        _actions = read_actions(ACTION_CSV, source_roles=['labeled'], columns=['token_addr', 'timestamp'])
        if not full_history:
            _actions = _actions[_actions['timestamp'] < train_cutoff]
        tx_cnt = _actions.groupby('token_addr').size()
        del _actions
    else:
        count_rows = _df if full_history else _df[_df['timestamp'] < train_cutoff]
        tx_cnt = count_rows.groupby('token_addr').size() / 2.0
        del count_rows
    labels_map = _df.groupby('token_addr')['label'].first().to_dict()
    del _df
    import gc as _gc
    _gc.collect()
    if not full_history:
        log('[data] labeled edges parsed early; train_cutoff=%.0f val_cutoff=%.0f' % (train_cutoff, val_cutoff))


    def build_rugscreener_address_windows(seed):
        """Reproduce classification.py's edge-index split and return token addresses.

        RugScreener removes every edge incident to a random 10% of nodes occurring in
        the original test tail, then takes train/val/test edge-ID ranges from the
        remaining graph.  Classification converts each range to a set of labeled
        endpoint nodes.  DGL remove_edges preserves the relative order of survivors,
        so the same operation can be reproduced with NumPy without cloning the graph.
        """
        edge_path = os.path.join(DATA_DIR, 'data_std_label.csv')
        token_path = os.path.join(DATA_DIR, 'transfer_std_label_new.csv')
        map_path = os.path.join(DATA_DIR, 'address_to_id_std_label.csv')

        edges = _pd.read_csv(edge_path, usecols=['src', 'dst', 'timestamp'])
        src = edges['src'].to_numpy(dtype=np.int64)
        dst = edges['dst'].to_numpy(dtype=np.int64)
        ts = edges['timestamp'].to_numpy(dtype=np.float64)
        del edges

        n_edges = len(src)
        trainval_div = int(VALID_SPLIT * n_edges)
        test_split_ts = ts[trainval_div]
        test_nodes = np.unique(np.concatenate([src[trainval_div:], dst[trainval_div:]]))
        rs_rng = np.random.RandomState(seed)
        n_new = int(0.1 * len(test_nodes))
        test_new_nodes = rs_rng.choice(test_nodes, n_new, replace=False)

        incident = np.isin(src, test_new_nodes) | np.isin(dst, test_new_nodes)
        early_removed = int(np.sum(incident & (ts < test_split_ts)))
        src_keep = src[~incident]
        dst_keep = dst[~incident]
        train_end = int(TRAIN_SPLIT * len(src_keep))
        test_start = trainval_div - early_removed
        if not (0 <= train_end <= test_start <= len(src_keep)):
            raise RuntimeError('Invalid RugScreener split bounds: train_end=%d test_start=%d remaining=%d' % (
                train_end, test_start, len(src_keep)))

        token_df = _pd.read_csv(token_path, usecols=['token_address', 'label'])
        token_df = token_df[token_df['label'].isin([0, 1])].drop_duplicates('token_address')
        labeled_ids = set(token_df['token_address'].astype(np.int64).tolist())
        del token_df

        node_windows = [
            set(np.concatenate([src_keep[:train_end], dst_keep[:train_end]]).tolist()) & labeled_ids,
            set(np.concatenate([src_keep[train_end:test_start], dst_keep[train_end:test_start]]).tolist()) & labeled_ids,
            set(np.concatenate([src_keep[test_start:], dst_keep[test_start:]]).tolist()) & labeled_ids,
        ]
        addr_df = _pd.read_csv(map_path, usecols=['address', 'id'])
        id_to_address = dict(zip(addr_df['id'].astype(np.int64), addr_df['address'].str.lower()))
        del addr_df
        address_windows = [sorted(id_to_address[n] for n in nodes if n in id_to_address)
                           for nodes in node_windows]

        meta = {
            'source_edges': int(n_edges),
            'remaining_edges': int(len(src_keep)),
            'train_edge_end': int(train_end),
            'test_edge_start': int(test_start),
            'candidate_test_nodes': int(len(test_nodes)),
            'heldout_new_nodes': int(n_new),
            'early_removed_edges': int(early_removed),
            'source_timestamp_monotonic': bool(np.all(ts[1:] >= ts[:-1])),
            'source_timestamp_decrease_fraction': float(np.mean(ts[1:] < ts[:-1])),
            'source_position_70_timestamp': float(ts[int(TRAIN_SPLIT * n_edges)]),
            'source_position_85_timestamp': float(ts[trainval_div]),
        }
        del src, dst, ts, src_keep, dst_keep, incident, test_new_nodes
        _gc.collect()
        return address_windows, meta


    rs_address_windows = None
    rs_source_meta = None
    if EVAL_PROTOCOL == 'rugscreener_code':
        rs_address_windows, rs_source_meta = build_rugscreener_address_windows(SEED)
        log('[eval][RugScreener-code] source edges=%d remaining=%d bounds=%d/%d' % (
            rs_source_meta['source_edges'], rs_source_meta['remaining_edges'],
            rs_source_meta['train_edge_end'], rs_source_meta['test_edge_start']))
        log('[eval][RugScreener-code] source timestamp monotonic=%s decrease_fraction=%.4f' % (
            rs_source_meta['source_timestamp_monotonic'],
            rs_source_meta['source_timestamp_decrease_fraction']))
        if not rs_source_meta['source_timestamp_monotonic']:
            log('[WARN][eval] RugScreener source rows are not chronological; this reproduces row-index windows, not a strict temporal split.')

    # ---- Load graph + data ----
    gs, _ = dgl.load_graphs(GRAPH_BIN)
    g = gs[0]
    if full_history:
        train_cutoff = val_cutoff = full_history_cutoff(g)
        log('[data] retrospective full graph history for train/val/test; snapshot_end=%s' % train_cutoff)

    # ---- Plan A: filtered tokens (transaction count >20) + stratified random split ----
    # Only tokens with >20 transactions are used (verified: 983 rug + 673 benign)
    # Stratified split keeps rug:benign ratio consistent across train/val/test.
    # (CSV 的 tx_cnt/labels_map/cutoffs 已在图加载前提前解析并释放 df — 见上方"标签 CSV 提前解析")
    all_labels = g.nodes['contract'].data['label']
    valid_mask = all_labels >= 0

    # Keep only tokens with valid labels (0 or 1); optionally require >20 transactions.
    # CRITICAL: exclude label=-1 (unlabeled) — they break Focal Loss (bce negative)
    # env PHASE3_ALL_TOKENS=1 → 用全部标注 token (含 <=20 tx 的 rug)，与 RugScreener 0.928 的
    # 2,673-token 全量口径对比 (方案 A 的 >20tx 只覆盖 rug 983/benign 673)。
    USE_ALL_TOKENS = os.environ.get('PHASE3_ALL_TOKENS', '1') == '1'
    if EVAL_PROTOCOL == 'rugscreener_code':
        USE_ALL_TOKENS = True
        valid_tokens = sorted(set().union(*[set(x) for x in rs_address_windows]))
    elif USE_ALL_TOKENS:
        valid_tokens = [t for t, label in labels_map.items() if label in (0, 1)]
    else:
        valid_tokens = [t for t, cnt in tx_cnt.items()
                        if cnt > 20 and t in labels_map and labels_map[t] in (0, 1)]
    print('[data] %s valid tokens: %d' % ('ALL-labeled' if USE_ALL_TOKENS else '>20-tx', len(valid_tokens)))

    # Map token addresses → contract local IDs via contract_gids.pt
    import torch as _T
    cgids = _T.load(os.path.join(BHTG_DATA_DIR, 'contract_gids.pt'))
    df_addr_map = _pd.read_csv(ADDR_TO_ID_CSV)
    addr_to_id = dict(zip(df_addr_map['address'].str.lower(), df_addr_map['id']))
    id_to_addr = {v: k for k, v in addr_to_id.items()}
    # contract local ID i ↔ global ID cgids[i] ↔ address
    gid_to_lid = {gid: lid for lid, gid in enumerate(cgids)}
    tok_lid = {}
    mapping_omissions = []
    for t in valid_tokens:
        gid = addr_to_id.get(t)
        if gid is not None and gid in gid_to_lid:
            tok_lid[gid_to_lid[gid]] = labels_map[t]
        else:
            mapping_omissions.append({
                'address': t, 'label': int(labels_map[t]),
                'global_id': int(gid) if gid is not None else None,
                'reason': 'not_mapped_as_contract',
            })
    log('[data] mapping omissions=%d' % len(mapping_omissions))

    # The 14:28 backup predates this known fix. The actual legacy_strict_retrain
    # log confirms synchronized CSV supervision for 2671 tokens / 1 mismatch.
    # This small reconstruction is necessary to restore that run, not a modern fix.
    label_mismatches = sum(
        int(all_labels[lid].item()) != int(label) for lid, label in tok_lid.items())
    all_labels = all_labels.clone()
    for lid, label in tok_lid.items():
        all_labels[lid] = int(label)
    g.nodes['contract'].data['label'] = all_labels
    log('[data] synchronized %d supervised labels from token CSV (%d graph mismatches)' %
        (len(tok_lid), label_mismatches))

    # Split protocol.  The default is mutually exclusive and stratified.  The
    # RugScreener-code branch intentionally preserves node overlap across edge windows.
    if EVAL_PROTOCOL == 'rugscreener_code':
        address_to_lid = {}
        for address in valid_tokens:
            gid = addr_to_id.get(address)
            if gid is not None and gid in gid_to_lid:
                address_to_lid[address] = gid_to_lid[gid]
        split_lists = []
        for addresses in rs_address_windows:
            # Original and BHTG address maps differ for a handful of records.  The
            # original source label is necessary but not sufficient: require the
            # mapped BHTG contract node itself to carry a valid binary label too.
            split_lists.append(sorted({
                address_to_lid[a] for a in addresses
                if a in address_to_lid
                and int(all_labels[address_to_lid[a]].item()) in (0, 1)
            }))
        tr_lids, va_lids, te_lids = split_lists
        split = {
            'train': torch.tensor(tr_lids, dtype=torch.long),
            'val': torch.tensor(va_lids, dtype=torch.long),
            'test': torch.tensor(te_lids, dtype=torch.long),
        }
        split_sets = [set(tr_lids), set(va_lids), set(te_lids)]
        overlaps = {
            'train_val': len(split_sets[0] & split_sets[1]),
            'train_test': len(split_sets[0] & split_sets[2]),
            'val_test': len(split_sets[1] & split_sets[2]),
            'all_three': len(split_sets[0] & split_sets[1] & split_sets[2]),
        }
        def _class_count(ids, label):
            return sum(int(all_labels[i].item()) == label for i in ids)
        r_tr, b_tr = [None] * _class_count(tr_lids, 1), [None] * _class_count(tr_lids, 0)
        r_va, b_va = [None] * _class_count(va_lids, 1), [None] * _class_count(va_lids, 0)
        r_te, b_te = [None] * _class_count(te_lids, 1), [None] * _class_count(te_lids, 0)
        log('[eval][RugScreener-code] overlap train-val=%d train-test=%d val-test=%d all=%d' % (
            overlaps['train_val'], overlaps['train_test'], overlaps['val_test'], overlaps['all_three']))
        if min(len(b_va), len(b_te)) < 10:
            log('[WARN][eval] RugScreener-code val/test contain very few benign tokens; BAC is high-variance and must be reported with this caveat.')
    else:
        rug_lids = [lid for lid, lab in tok_lid.items() if lab == 1]
        ben_lids = [lid for lid, lab in tok_lid.items() if lab == 0]
        log('[data] rug=%d benign=%d (ratio=%.2f:1)' % (len(rug_lids), len(ben_lids), len(rug_lids)/max(len(ben_lids),1)))

        rng = np.random.RandomState(SEED)
        def stratify(lids):
            rng.shuffle(lids)
            n = len(lids)
            return (lids[:int(0.7*n)], lids[int(0.7*n):int(0.85*n)], lids[int(0.85*n):])

        r_tr, r_va, r_te = stratify(rug_lids)
        b_tr, b_va, b_te = stratify(ben_lids)
        split = {
            'train': torch.tensor(r_tr + b_tr, dtype=torch.long),
            'val': torch.tensor(r_va + b_va, dtype=torch.long),
            'test': torch.tensor(r_te + b_te, dtype=torch.long),
        }
        overlaps = {'train_val': 0, 'train_test': 0, 'val_test': 0, 'all_three': 0}

    # Freeze token assignments before observing availability. Excluded IDs are
    # never recycled into another split. Existence checks do not inspect labels.
    creator_times, creator_source = (creator_times_from_actions(ACTION_CSV) if ACTION_CSV else
        load_creator_times(os.path.join(BHTG_DATA_DIR, 'chain_creators.csv')))
    contract_addresses = {lid: id_to_addr[int(gid)] for lid, gid in enumerate(cgids)}
    observation_index = ObservationIndex(
        g, node_addresses={'contract': contract_addresses}, creator_times=creator_times)
    cohorts = partition_observed_splits(
        split, observation_index,
        {'train': train_cutoff, 'val': val_cutoff, 'test': val_cutoff}, SEED)
    split = cohorts['split']
    cold_start = cohorts['cold_start']
    for name in ('train', 'val', 'test'):
        if set(all_labels[split[name]].tolist()) != {0, 1}:
            raise RuntimeError('Primary %s cohort must contain both classes after observation filtering' % name)
        summary = cohorts['observation_meta']['summaries'][name]
        log('[observation] %s cutoff=%.0f active=%d cold=%d excluded=%d reasons=%s' % (
            name, cohorts['observation_meta']['cutoffs'][name],
            summary['active_history'], summary['cold_start'], summary['excluded'],
            summary['excluded_reasons']))
        log('[observation] %s cold classes: benign=%d rug=%d' % (
            name, int((all_labels[cold_start[name]] == 0).sum()),
            int((all_labels[cold_start[name]] == 1).sum())))
    r_tr, b_tr = (split['train'][all_labels[split['train']] == label].tolist()
                  for label in (1, 0))
    r_va, b_va = (split['val'][all_labels[split['val']] == label].tolist()
                  for label in (1, 0))
    r_te, b_te = (split['test'][all_labels[split['test']] == label].tolist()
                  for label in (1, 0))

    log('[data] Primary Train=%d(r=%d,b=%d) Val=%d(r=%d,b=%d) Test=%d(r=%d,b=%d)' % (
        len(split['train']), len(r_tr), len(b_tr),
        len(split['val']), len(r_va), len(b_va),
        len(split['test']), len(r_te), len(b_te)))

    split_meta = {
        'protocol': EVAL_PROTOCOL,
        'seed': int(SEED),
        'train_cutoff': float(train_cutoff),
        'val_cutoff': float(val_cutoff),
        'use_all_tokens': bool(USE_ALL_TOKENS),
        'overlaps': overlaps,
        'rugscreener_source': rs_source_meta,
        'training_protocol': PROTOCOL_VERSION,
        'observation_protocol': OBSERVATION_PROTOCOL,
        'observation': cohorts['observation_meta'],
        'creator_source': creator_source,
        'mapping_omissions': mapping_omissions,
    }
    def input_sha256(path):
        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()
    split_meta['input_sha256'] = {
        'graph': input_sha256(GRAPH_BIN),
        'contract_gids': input_sha256(os.path.join(BHTG_DATA_DIR, 'contract_gids.pt')),
        'address_map': input_sha256(ADDR_TO_ID_CSV),
        'labels': input_sha256(LABELED_EDGES_CSV),
    }
    if amount_meta:
        split_meta['amount_experiment'] = amount_meta
    if full_history:
        split_meta.update(experiment_protocol=EXPERIMENT_PROTOCOL,
                          full_history_cutoff=train_cutoff)
    split_meta['smoke_subset'] = '--smoke' in sys.argv
    if ACTION_CSV:
        split_meta['input_sha256']['actions_csv'] = input_sha256(ACTION_CSV)
    torch.save(dict(cohorts, meta=split_meta), SPLIT_PATH)
    OBSERVATION_PATH = os.path.splitext(SPLIT_PATH)[0] + '_observation.json'
    with open(OBSERVATION_PATH, 'w', encoding='utf-8') as handle:
        json.dump({'meta': split_meta, 'records': cohorts['observation_records']},
                  handle, ensure_ascii=False, indent=2, allow_nan=False)
    del observation_index, creator_times, cohorts
    log('[eval] saved deterministic split manifest: %s' % SPLIT_PATH)
    log('[eval] primary=active_history; cold starts reported separately; observation=%s' % OBSERVATION_PATH)
    if '--split-only' in sys.argv:
        log('[eval] --split-only complete; training skipped.')
        sys.exit(0)

    # ---- Load Phase 1 h_sem ----
    if USE_SEMANTIC and os.path.exists(H_SEM_PATH):
        semantic_bundle = torch.load(H_SEM_PATH, map_location='cpu')
        if full_history:
            h_sem_full = validate_full_semantic_bundle(
                semantic_bundle, g.num_nodes('contract'), HIDDEN_DIM,
                reference_train_cutoff, reference_val_cutoff, train_cutoff,
                semantic_amount_mode=SEMANTIC_AMOUNT_MODE, amount_experiment=amount_meta,
                expected_input_bindings=semantic_history_input_bindings())
            h_sem_snapshots = {'train': h_sem_full, 'val': h_sem_full}
        else:
            h_sem_snapshots = validate_semantic_bundle(
                semantic_bundle, g.num_nodes('contract'), HIDDEN_DIM, train_cutoff, val_cutoff,
                semantic_amount_mode=SEMANTIC_AMOUNT_MODE, amount_experiment=amount_meta,
                allow_smoke=('--smoke' in sys.argv))
        h_sem_all = h_sem_snapshots['train']
        del semantic_bundle
        log('Loaded Phase1 %s from %s' % ('full-history snapshot' if full_history
                                          else 'train/val snapshots', H_SEM_PATH))
    else:
        if USE_SEMANTIC:
            raise RuntimeError('Missing Phase1 semantic bundle: %s' % H_SEM_PATH)
        h_sem_snapshots = {}
        h_sem_all = None

    # ---- Build graph encoder ----
    node_feat_dims = {nt: len(node_feature_names) for nt in g.ntypes}

    graph_encoder = GraphEncoder(
        node_feat_dims=node_feat_dims, hidden_dim=HIDDEN_DIM,
        out_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
        etypes=g.canonical_etypes, lambda_init=LAMBDA_INIT, dropout=DROPOUT,
        homo_ablation=homo_ablation,
        edge_feature_fields=graph_amount_fields(MEMORY_AMOUNT_MODE) or None,
    ).to(DEVICE)

    if os.path.exists(CKPT_GRAPH):
        try:
            ckpt = torch.load(CKPT_GRAPH, map_location='cpu')
            validate_amount_metadata(ckpt.get('amount_experiment'), amount_meta)
            if ckpt.get('smoke_subset') and '--smoke' not in sys.argv:
                raise RuntimeError('Smoke graph weights require explicit --smoke')
            if ckpt.get('protocol') != PROTOCOL_VERSION or ckpt.get('sampler_protocol') != SAMPLER_PROTOCOL:
                raise RuntimeError('Phase2 checkpoint uses an old protocol; retrain Phase2')
            if (ckpt.get('encoder_protocol') != ENCODER_PROTOCOL or
                    ckpt.get('sampling_hops') != graph_encoder.required_sampling_hops):
                raise RuntimeError('Phase2 encoder or sampling depth predates batch-independent encoding; retrain Phase2')
            if ckpt.get('experiment_protocol') != (EXPERIMENT_PROTOCOL if full_history else None):
                raise RuntimeError('Phase2 checkpoint experiment protocol mismatch')
            phase2_train_cutoff = float(ckpt.get('train_cutoff', -1))
            if full_history:
                phase2_val_cutoff = float(ckpt.get('val_cutoff', -1))
                if not phase2_train_cutoff < phase2_val_cutoff < train_cutoff:
                    raise RuntimeError('Phase2 unlabeled edge cutoffs are inconsistent with the graph snapshot')
            elif phase2_train_cutoff != train_cutoff:
                raise RuntimeError('Phase2 pretraining cutoff mismatch')
            # Older legacy15 checkpoints have no top-level feature metadata;
            # their scaler state still receives the original strict validation.
            if ('node_feature_protocol' in ckpt and
                    (ckpt['node_feature_protocol'] != node_feature_protocol or
                     ckpt.get('node_feature_names') != list(node_feature_names) or
                     ckpt.get('node_feat_dims') != node_feat_dims)):
                raise RuntimeError('Phase2 node-feature protocol/names/dimensions mismatch')
            restore_features = (restore_full_history_feature_provider if full_history
                                else restore_feature_provider)
            feature_provider = restore_features(g, ckpt['feature_state'])
            if feature_provider.train_cutoff != phase2_train_cutoff:
                raise RuntimeError('Feature scaler training cutoff mismatch')
            if full_history and feature_provider.binding['phase2_val_cutoff'] != phase2_val_cutoff:
                raise RuntimeError('Phase2 validation cutoff differs from the input binding')
            ckpt_relbal = bool(ckpt.get('relation_balanced', False))
            ckpt_strict = bool(ckpt.get('strict_temporal', False))
            if ckpt_relbal != RELATION_BALANCED:
                raise RuntimeError(
                    'sampler/checkpoint mismatch: checkpoint relation_balanced=%s, run=%s'
                    % (ckpt_relbal, RELATION_BALANCED))
            if ckpt_strict != STRICT_RUN:
                raise RuntimeError(
                    'temporal/checkpoint mismatch: checkpoint strict_temporal=%s, run=%s'
                    % (ckpt_strict, STRICT_RUN))
            graph_encoder.load_state_dict(ckpt['model_state'])
            graph_encoder.hgt.reset_lambda(LAMBDA_INIT)
            log('Loaded Phase2 graph encoder (epoch %d) homo=%s' % (ckpt.get('epoch', -1), homo_ablation))
            del ckpt   # 释放 (~40MB), state_dict 已拷入 graph_encoder
        except Exception as exc:
            # The checkpoint owns the fitted feature provider as well as the
            # encoder. Continuing here would use missing or unvalidated input
            # state, so a failed load cannot fall back to random weights.
            raise RuntimeError('Phase2 checkpoint validation/load failed: %s' % exc) from exc
    else:
        raise RuntimeError('Missing Phase2 checkpoint: %s; train Phase2 first' % CKPT_GRAPH)

    fusion = CrossModalFusion(
        graph_dim=HIDDEN_DIM, sem_dim=HIDDEN_DIM,
        hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in list(graph_encoder.parameters()) + list(fusion.parameters()))
    log('[model] Trainable: %d' % n_params)

    # Negative-control audit: shuffle binary labels independently inside each split.
    # The graph, semantic vectors, memory and sampler remain unchanged.  We also
    # replace graph node labels so an accidental label read inside the encoder would
    # be exposed by an implausibly high validation/test score.
    PERMUTE_LABELS = os.environ.get('PHASE3_PERMUTE_LABELS', '0') == '1'
    if PERMUTE_LABELS:
        all_labels = all_labels.clone()
        perm_rng = np.random.RandomState(SEED + 424242)
        for split_name in ('train', 'val', 'test'):
            ids = split[split_name]
            valid_ids = ids[all_labels[ids] >= 0]
            values = all_labels[valid_ids].clone()
            order = torch.from_numpy(perm_rng.permutation(len(valid_ids))).long()
            all_labels[valid_ids] = values[order]
        g.nodes['contract'].data['label'] = all_labels
        log('[audit] PHASE3_PERMUTE_LABELS=True; labels shuffled within train/val/test')
    # ---- TGN GRU Memory 注入 (--use-memory) ----
    # 冻结 GRU 从零滚成 M_train@train_cutoff / M_val@val_cutoff 快照 (CPU)，
    # phase3 只微调 HGT+fusion+memory_proj (memory_proj 在 inject_memory 里懒建)。
    # 必须在 load_state_dict(phase2_graph.pt) 之后 inject (strict 加载不冲突)。
    if USE_MEMORY:
        from bhtg.memory import HeteroTemporalMemory, MEMORY_SEMANTICS, memory_amount_inputs
        from bhtg.compact_memory_tail import CompactMemoryTail
        from bhtg.replay_checkpoint import checkpoint_memory_block
        from functools import partial
        from bhtg.edge_stream import load_edge_stream, temporal_batches
        CKPT_MEM = os.path.join(CHECKPOINT_DIR, 'phase2_memory.pt')
        if not os.path.exists(CKPT_MEM):
            log('[fatal] --use-memory requires %s (run run_phase2_memory.py first)' % CKPT_MEM)
            sys.exit(1)
        stream = load_edge_stream()
        mem = HeteroTemporalMemory(stream['num_nodes'], mem_dim=MEM_DIM, temporal_dim=TEMPORAL_DIM,
                                  amount_mode=MEMORY_AMOUNT_MODE)
        mc = torch.load(CKPT_MEM, map_location='cpu')
        validate_amount_metadata(mc.get('amount_experiment'), amount_meta)
        if mc.get('smoke_subset') and '--smoke' not in sys.argv:
            raise RuntimeError('Smoke Memory weights require explicit --smoke')
        if mc.get('memory_semantics') != mem.memory_semantics:
            raise RuntimeError('Memory checkpoint uses an old aggregation protocol; retrain Memory')
        if float(mc.get('train_cutoff', -1)) != reference_train_cutoff:
            raise RuntimeError('Memory pretraining cutoff mismatch')
        _mem_epoch = mc.get('epoch', -1)
        _mem_val_loss = mc.get('val_link_loss', float('nan'))
        mem.load_weights(mc)
        del mc   # 释放 ~150MB (memory_state 已拷入 mem 表), 给首轮 roll 腾物理内存
        mem.eval()
        if USE_JOINT:
            # 联合训练: GRU/time_enc 保持可训练（表在 __init__ 就是 requires_grad=False）
            log('[joint] GRU trainable (opt_mem); memory tables keep requires_grad=False')
        else:
            mem.requires_grad_(False)   # frozen: GRU 冻结，表是只读查找
        # 子图 _all_gids 即 full-graph local id (见 graph_encoder._subgraph_local_ids 注释)，
        # memory 表按 local id 索引，无需 gid_to_lid。memory_proj 在 inject_memory 里懒建 (可训练)。
        graph_encoder.inject_memory(mem)

        def roll_snapshot(end_ts):
            idx = (stream['ts'] < end_ts).nonzero(as_tuple=True)[0]
            begin_memory_replay(end_ts, len(idx))
            with torch.no_grad():
                mem.reset()
                for b in temporal_batches(stream['ts'], idx, MEM_ROLL_BATCH):
                    mem.roll_batch(stream['src_ntype'][b], stream['dst_ntype'][b],
                                   stream['src_lid'][b], stream['dst_lid'][b],
                                   stream['etype_idx'][b], stream['ts'][b],
                                   stream['value'][b],
                                   **memory_amount_inputs(stream, b, amount_mode=mem.amount_mode))
                return mem.backup()

        def roll_diff_stream(end_ts, tail_chunks=JOINT_TAIL_CHUNKS, batch=MEM_ROLL_BATCH):
            """Replay the detached prefix and track gradients only for tail endpoints."""
            idx = (stream['ts'] < end_ts).nonzero(as_tuple=True)[0]
            prefix = mem.empty_state()
            chunks = list(temporal_batches(stream['ts'], idx, batch))
            begin_memory_replay(end_ts, len(idx), len(chunks))
            split_at = max(0, len(chunks) - tail_chunks)
            with torch.no_grad():
                for b in chunks[:split_at]:
                    prefix = mem.roll_diff(prefix, stream['src_ntype'][b], stream['dst_ntype'][b],
                                           stream['src_lid'][b], stream['dst_lid'][b],
                                           stream['etype_idx'][b], stream['ts'][b], stream['value'][b],
                                           inplace=True,
                                           **memory_amount_inputs(stream, b, amount_mode=mem.amount_mode))
                tail = chunks[split_at:]
                tail_rows = torch.cat(tail) if tail else idx
                src_types, dst_types = stream['src_ntype'][tail_rows], stream['dst_ntype'][tail_rows]
                src_ids, dst_ids = stream['src_lid'][tail_rows], stream['dst_lid'][tail_rows]
                node_ids = {
                    ti: torch.unique(torch.cat([src_ids[src_types == ti], dst_ids[dst_types == ti]]),
                                     sorted=True)
                    for ti in prefix
                }
                compact = CompactMemoryTail.compact_state(prefix, node_ids)
                local_src, local_dst = torch.empty_like(src_ids), torch.empty_like(dst_ids)
                for ti, ids in node_ids.items():
                    sm, dm = src_types == ti, dst_types == ti
                    local_src[sm] = torch.searchsorted(ids, src_ids[sm])
                    local_dst[dm] = torch.searchsorted(ids, dst_ids[dm])
            offset = 0
            for b in tail:
                rows = slice(offset, offset + len(b))
                roll_block = partial(
                    mem.roll_diff, src_nt=src_types[rows], dst_nt=dst_types[rows],
                    src=local_src[rows], dst=local_dst[rows],
                    et=stream['etype_idx'][b], ts=stream['ts'][b], value=stream['value'][b],
                    **memory_amount_inputs(stream, b, amount_mode=mem.amount_mode))
                compact = checkpoint_memory_block(roll_block, compact)
                offset += len(b)
            return CompactMemoryTail(prefix, node_ids, compact)

        def load_snapshot(snap):
            mem.clear_override()
            mem.restore(snap)

        if USE_JOINT:
            # joint: M_train/M_val 每 epoch 用当前 GRU 重滚（启动快照作废），此处不预滚
            log('[joint] M_train/M_val 每 epoch 重滚（无启动快照）')
            if '--joint-check' in sys.argv:
                set_replay_phase('setup_memory_replay')
                # 冒烟验证: roll_diff（前段 no_grad + 尾段 grad，含边界切换）与 roll_snapshot 逐位一致。
                # 不用 tail=全量: 全 grad 链 ~3-7GB，14GB RAM 下与常驻模型叠加有 OOM 风险。
                with torch.no_grad():
                    T1 = roll_diff_stream(train_cutoff, tail_chunks=JOINT_TAIL_CHUNKS)
                    overrides = T1.overrides()
                    T2 = roll_snapshot(train_cutoff)
                    for ti in range(len(NODE_TYPES)):
                        expected = T2[0][ti]
                        ok = all(torch.allclose(overrides[ti][torch.arange(start, min(start + 4096, len(expected)))],
                                                expected[start:start + 4096], rtol=1e-5, atol=1e-5)
                                 for start in range(0, len(expected), 4096))
                        log('[joint-check] type %d allclose=%s' % (ti, ok))
                        assert ok, 'roll_diff != roll_snapshot (type %d)' % ti
                    del overrides
                log('[joint-check] OK: roll_diff 与 roll_snapshot 逐位一致')
                del T1, T2   # 释放校验图，避免与 epoch0 的 M_diff 叠加吃 RAM
        else:
            set_replay_phase('setup_memory_replay')
            M_train = roll_snapshot(train_cutoff)
            M_val = M_train if train_cutoff == val_cutoff else roll_snapshot(val_cutoff)
            load_snapshot(M_train)   # 训练用 M_train
            stream = None  # frozen: roll 只在启动做一次；置空释放边流数组（注意 roll_snapshot 闭包引用此 cell）
        log('[memory] injected GRU (epoch %d, val link-loss=%.4f); M_train@%.0f M_val@%.0f' % (
            _mem_epoch, _mem_val_loss, train_cutoff, val_cutoff))

    all_ts = []
    for et in g.canonical_etypes:
        if 'timestamp' in g.edges[et].data:
            all_ts.append(g.edges[et].data['timestamp'])
    max_ts = torch.cat(all_ts).max().item()

    # (in-degree is now part of the 8-dim node features computed in graph_builder)

    # ---- Optimizer ----
    # Split λ: transfer learns, rare types (burn/mint/create) stay near initial
    rare_lam_names = {'burn', 'mint', 'create'}
    lambda_params = graph_encoder.get_lambda_params()
    rare_lam = []
    transfer_lam = []
    # Get names from hgt layer params
    all_names = []
    for layer in graph_encoder.hgt.layers:
        all_names.extend(layer.lambda_decay.keys())
    for p, name in zip(lambda_params, all_names):
        if name in rare_lam_names:
            rare_lam.append(p)
        else:
            transfer_lam.append(p)

    other_graph = [p for p in graph_encoder.parameters() if p not in set(lambda_params)]
    if USE_MEMORY:
        # 排除 memory 模块全部参数 (记忆表 146MB + 冻结 GRU) — 只微调 HGT+fusion+memory_proj
        mem_param_ids = {id(p) for p in graph_encoder.memory.parameters()}
        other_graph = [p for p in other_graph if id(p) not in mem_param_ids]
    optimizer = torch.optim.Adam([
        {'params': other_graph, 'lr': LEARNING_RATE},
        {'params': list(fusion.parameters()), 'lr': LEARNING_RATE},
        {'params': transfer_lam, 'lr': LAMBDA_LR},
        {'params': rare_lam, 'lr': LAMBDA_LR * 0.01},   # 100× slower
    ])
    if USE_JOINT:
        # GRU 联合训练优化器: 只含 time_enc + 4×GRUCell（表 requires_grad=False 天然被滤掉）。
        # 必须在跳过 requires_grad_(False) 之后构造，否则过滤掉全部参数。
        opt_mem = torch.optim.Adam([p for p in mem.parameters() if p.requires_grad], lr=MEM_LR)
        log('[joint] opt_mem: %d trainable GRU params (lr=%.1e)' %
            (len(opt_mem.param_groups[0]['params']), MEM_LR))
    else:
        opt_mem = None
    focal_fn = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    ALPHA_ALIGN = 0.0  # TEMP diagnostic: disable align loss
    BATCH_SIZE = 32  # override: more batches/epoch = more diverse gradient signal

    def predict_ids(ids, cutoff, semantic_snapshot):
        """Every requested token must produce exactly one finite prediction."""
        predictions = []
        with torch.no_grad():
            for start in range(0, len(ids), BATCH_SIZE):
                bid = ids[start:start + BATCH_SIZE]
                sub_g, sid = sample_token_subgraph(
                    g, bid, current_time=cutoff, k=N_NEIGHBORS,
                    n_hop=graph_encoder.required_sampling_hops,
                    relation_quotas=RELATION_QUOTAS, feature_provider=feature_provider)
                if sub_g is None or sid is None or len(sid) != len(bid):
                    raise RuntimeError('Evaluation sampler dropped requested tokens')
                sub_g = sub_g.to(DEVICE)
                emb = graph_encoder(sub_g, current_time=cutoff)
                if emb is None:
                    raise RuntimeError('Evaluation encoder returned no token embeddings')
                hg = emb[sid.to(DEVICE)]
                hs = (h_sem_snapshots[semantic_snapshot][bid.long()].to(DEVICE)
                      if USE_SEMANTIC else None)
                if hs is not None:
                    hs = torch.nan_to_num(hs, nan=0.0)
                probabilities = torch.sigmoid(fusion(hg, hs)).cpu().numpy().reshape(-1)
                if len(probabilities) != len(bid) or not np.isfinite(probabilities).all():
                    raise RuntimeError('Evaluation predictions are incomplete or nonfinite')
                predictions.extend(probabilities.tolist())
        return predictions

    def run_joint_phase_b(M_diff, epoch):
        graph_encoder.eval(); fusion.eval()
        pb_loss, pb_norm, pb_done = 0.0, 0.0, 0
        for _st in range(JOINT_STEPS):
            # Each optimizer step consumes its own autograd graph and current
            # GRU parameters; a second backward through M_diff is invalid.
            if _st > 0:
                M_diff = roll_diff_stream(train_cutoff, tail_chunks=JOINT_TAIL_CHUNKS)
            # Stage gradients at the replay output so each classifier batch can
            # release its graph before we recompute the recurrent blocks.
            tail_outputs = {ti: state[0] for ti, state in M_diff.tables.items()
                            if state[0].requires_grad and state[0].numel()}
            tail_leaves = {ti: value.detach().requires_grad_() for ti, value in tail_outputs.items()}
            head_tables = {
                ti: (tail_leaves.get(ti, state[0].detach()), state[1], state[2])
                for ti, state in M_diff.tables.items()
            }
            head_memory = CompactMemoryTail(M_diff.prefix, M_diff.node_ids, head_tables)
            mem.set_override(head_memory.overrides())
            rs = np.random.RandomState(SEED * 1000 + epoch * 10 + _st)
            jr = rs.choice(r_tr, size=min(JOINT_SAMPLE // 2, len(r_tr)), replace=False) if len(r_tr) else np.array([], dtype=int)
            jb = rs.choice(b_tr, size=min(JOINT_SAMPLE - len(jr), len(b_tr)), replace=False) if len(b_tr) else np.array([], dtype=int)
            jids = np.concatenate([jr, jb])
            jloss_terms = []
            tail_gradients = {}
            for st0 in range(0, len(jids), BATCH_SIZE):
                bid = torch.tensor(jids[st0:st0 + BATCH_SIZE], dtype=torch.long)
                if len(bid) < 2:
                    continue
                sub_g, sid = sample_token_subgraph(
                    g, bid, current_time=train_cutoff, k=N_NEIGHBORS,
                    n_hop=graph_encoder.required_sampling_hops,
                    relation_quotas=RELATION_QUOTAS, feature_provider=feature_provider)
                if sub_g is None or sid is None or len(sid) < 2:
                    continue
                sub_g = sub_g.to(DEVICE); sid = sid.to(DEVICE)
                emb = graph_encoder(sub_g, current_time=train_cutoff)
                if emb is None or emb.shape[0] == 0:
                    continue
                hg = emb[sid]
                hg = torch.nan_to_num(hg, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-50, 50)
                hs = None
                if USE_SEMANTIC:
                    hs = h_sem_snapshots['train'][bid.long()][:len(sid)].to(DEVICE)
                    hs = torch.nan_to_num(hs, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-50, 50)
                lg = fusion(hg, hs)
                lb = all_labels[bid.long()][:len(sid)].float().unsqueeze(1).to(DEVICE)
                batch_loss = focal_fn(lg, lb)
                jloss_terms.append(batch_loss.detach())
                if tail_leaves and batch_loss.requires_grad and torch.isfinite(batch_loss):
                    gradients = torch.autograd.grad(batch_loss, tuple(tail_leaves.values()),
                                                    allow_unused=True)
                    for ti, gradient in zip(tail_leaves, gradients):
                        if gradient is None:
                            continue
                        if ti in tail_gradients:
                            tail_gradients[ti].add_(gradient)
                        else:
                            tail_gradients[ti] = gradient
                    del gradients, gradient
                del sub_g, sid, emb, hg, hs, lg, lb, batch_loss
            mem.clear_override()
            n_j = len(jloss_terms)
            if n_j == 0:
                log('[joint] epoch %d step %d: no valid batches, skip' % (epoch + 1, _st + 1))
                continue
            jloss = sum(jloss_terms) / n_j
            if not torch.isfinite(jloss):
                log('[joint] epoch %d step %d: non-finite loss, skip' % (epoch + 1, _st + 1))
                continue
            opt_mem.zero_grad(set_to_none=True)
            if tail_gradients:
                log('[joint] epoch %d step %d: classifier gradients ready; replay backward' %
                    (epoch + 1, _st + 1))
                torch.autograd.backward(
                    tuple(tail_outputs[ti] for ti in tail_gradients),
                    tuple(gradient.div_(n_j) for gradient in tail_gradients.values()))
            pb_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in opt_mem.param_groups[0]['params']], 1.0).item()
            opt_mem.step()
            optimizer.zero_grad(set_to_none=True)
            pb_loss += jloss.item(); pb_done += 1
        mem.clear_override()
        opt_mem.zero_grad(set_to_none=True)
        return pb_loss, pb_norm, pb_done

    upstream_identities = {'semantic': upstream_identity(H_SEM_PATH) if USE_SEMANTIC else None,
                           'graph': upstream_identity(CKPT_GRAPH),
                           'memory': upstream_identity(CKPT_MEM) if USE_MEMORY else None}
    training_contract = {
        'training_protocol': PROTOCOL_VERSION, 'split_meta': split_meta,
        'split_ids': {name: ids.tolist() for name, ids in split.items()},
        'cold_start_ids': {name: ids.tolist() for name, ids in cold_start.items()},
        'upstream_identities': upstream_identities, 'amount_experiment': amount_meta,
        'joint': USE_JOINT, 'use_memory': USE_MEMORY, 'use_semantic': USE_SEMANTIC,
        'homo_ablation': homo_ablation, 'permute_labels': PERMUTE_LABELS,
        'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE, 'lambda_lr': LAMBDA_LR,
        'memory_lr': MEM_LR, 'joint_sample': JOINT_SAMPLE,
        'joint_tail_chunks': JOINT_TAIL_CHUNKS, 'joint_steps': JOINT_STEPS,
        'memory_roll_batch': MEM_ROLL_BATCH, 'focal_alpha': FOCAL_ALPHA,
        'focal_gamma': FOCAL_GAMMA, 'align_weight': ALPHA_ALIGN,
        'relation_balanced': RELATION_BALANCED, 'relation_quotas': RELATION_QUOTAS,
        'encoder_protocol': ENCODER_PROTOCOL, 'sampler_protocol': SAMPLER_PROTOCOL,
    }
    if NODE_FEATURE_MODE == 'behavior9':
        # Preserve old legacy15 epoch contracts so their checkpoints can still
        # resume. New feature-mode resumes must bind their full input schema.
        training_contract.update(node_feature_protocol=node_feature_protocol,
                                 node_feature_names=list(node_feature_names),
                                 node_feat_dims=node_feat_dims)
    if threshold_policy == VALIDATION_THRESHOLD_POLICY:
        training_contract['threshold_policy'] = threshold_policy

    def selected_payload(selected_epoch, selected_bac, selected_metrics):
        return {
        'protocol': PROTOCOL_VERSION,
        'amount_experiment': amount_meta,
        'smoke_subset': '--smoke' in sys.argv,
        'upstream_identities': upstream_identities,
        'encoder_protocol': ENCODER_PROTOCOL,
        'graph_architecture_version': graph_encoder.architecture_version,
        'graph_edge_feature_fields': list(graph_encoder.edge_feature_fields),
        'sampling_hops': graph_encoder.required_sampling_hops,
        'sampler_protocol': SAMPLER_PROTOCOL,
        'node_feature_protocol': node_feature_protocol,
        'node_feature_names': list(node_feature_names),
        'node_feat_dims': node_feat_dims,
        'feature_state': feature_provider.state_dict(),
        'semantic_bundle': H_SEM_PATH if USE_SEMANTIC else None,
        'selection_metric': 'validation_active_history_bac',
        'observation_protocol': OBSERVATION_PROTOCOL,
        'validation_metrics': selected_metrics,
        'graph_encoder': model_weights(graph_encoder),
        'fusion': model_weights(fusion),
        'epoch': selected_epoch,
        'val_bac': selected_bac,
        'eval_protocol': EVAL_PROTOCOL,
        'split_meta': split_meta,
        'relation_balanced': RELATION_BALANCED,
        'relation_quotas': RELATION_QUOTAS,
        'phase2_graph_checkpoint': CKPT_GRAPH,
        'strict_temporal': STRICT_RUN,
        }

    # ---- Training ----
    best_val_bac, best_epoch = float('-inf'), 0
    best_checkpoint, best_validation_metrics = None, None
    validation_history = []
    start_epoch = 0
    phase3_epoch_loop_finished = False
    if RESUME_PATH:
        resumed = torch.load(RESUME_PATH, map_location='cpu')
        start_epoch = restore_epoch(resumed, training_contract, graph_encoder, fusion,
                                    optimizer, opt_mem)
        if start_epoch > EPOCHS_RUN:
            raise RuntimeError('Checkpoint epoch exceeds requested Phase3 epoch count')
        best_checkpoint = resumed['best']
        best_val_bac, best_epoch = best_checkpoint['val_bac'], best_checkpoint['epoch']
        best_validation_metrics = best_checkpoint['validation_metrics']
        validation_history = resumed['validation_history']
        # A crash may leave a newer best file than the last committed epoch.
        # The resume checkpoint is the sole authority for the committed boundary.
        atomic_torch_save(best_checkpoint, CKPT_BEST)
        write_validation_history(validation_history, VALIDATION_PATH)
        del resumed
        log('[resume] restored completed epoch %d; next epoch %d' % (start_epoch, start_epoch + 1))
    else:
        write_validation_history(validation_history, VALIDATION_PATH)
    log('Epoch,TrainLoss,Batches,ValBAC,ValF1,ValAP,ValAUC,lam_burn,lam_create,lam_mint,lam_transfer')

    for epoch in range(start_epoch, EPOCHS_RUN):
        set_replay_phase('training_memory_replay_with_tail_bptt', epoch + 1)
        # ---- 阶段 0: 每 epoch 可微重滚（joint 专用）----
        if USE_JOINT:
            M_diff = roll_diff_stream(train_cutoff, tail_chunks=JOINT_TAIL_CHUNKS)
            M_diff.commit(mem)  # Phase A reads the detached whole-graph snapshot.
            log('[joint] epoch %d: rolled (compact tail %d chunks with activation checkpointing)' %
                (epoch + 1, JOINT_TAIL_CHUNKS))
        elif USE_MEMORY:
            load_snapshot(M_train)

        graph_encoder.train(); fusion.train()
        total_loss, n_batches = 0.0, 0

        train_ids = split['train'][torch.randperm(len(split['train']))]

        for start in range(0, len(train_ids), BATCH_SIZE):
            batch_ids = train_ids[start:start + BATCH_SIZE]
            if len(batch_ids) < 2: continue

            sub_g, seed_in_subg = sample_token_subgraph(
                g, batch_ids, current_time=train_cutoff, k=N_NEIGHBORS,
                n_hop=graph_encoder.required_sampling_hops,
                relation_quotas=RELATION_QUOTAS, feature_provider=feature_provider,
            )
            if sub_g is None or seed_in_subg is None or len(seed_in_subg) < 2: continue

            sub_g = sub_g.to(DEVICE); seed_in_subg = seed_in_subg.to(DEVICE)
            token_emb = graph_encoder(sub_g, current_time=train_cutoff)
            if token_emb is None or token_emb.shape[0] == 0: continue
            h_graph = token_emb[seed_in_subg]

            B_local = len(seed_in_subg)
            batch_labels = all_labels[batch_ids.long()][:B_local]
            if USE_SEMANTIC:
                h_sem_batch = h_sem_all[batch_ids.long()][:B_local].to(DEVICE)
                h_sem_batch = torch.nan_to_num(h_sem_batch, nan=0.0)
            else:
                h_sem_batch = None

            # Clip extreme values to prevent gradient explosion
            h_graph = torch.nan_to_num(h_graph, nan=0.0, posinf=10.0, neginf=-10.0)
            h_graph = h_graph.clamp(-50, 50)
            if h_sem_batch is not None:
                h_sem_batch = torch.nan_to_num(h_sem_batch, nan=0.0, posinf=10.0, neginf=-10.0)
                h_sem_batch = h_sem_batch.clamp(-50, 50)

            logits, gate = fusion(h_graph, h_sem_batch, return_gate=True)
            labels = batch_labels.float().unsqueeze(1).to(DEVICE)

            L_focal = focal_fn(logits, labels)
            if ALPHA_ALIGN > 0 and h_sem_batch is not None and torch.isfinite(L_focal):
                L_align = fusion.align_loss(h_graph, h_sem_batch)
                # Guard against Inf/NaN in contrastive loss (can explode when features blow up)
                if not torch.isfinite(L_align):
                    L_align = torch.tensor(0.0, device=DEVICE)
            else:
                L_align = torch.tensor(0.0, device=DEVICE)
            loss = L_focal + ALPHA_ALIGN * L_align

            # Debug first batch loss value
            if epoch == 0 and n_batches == 0 and start == 0:
                recompute = focal_fn(logits.detach(), labels)
                # Manual focal loss trace
                bce_manual = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.detach(), labels, reduction='none')
                pt_manual = torch.exp(-bce_manual)
                print('[dbg] L_focal=%.4f recompute=%.4f | logits[:8]=%s | labels[:8]=%s | bce[:8]=%s | pt[:8]=%s' % (
                    L_focal.item(), recompute.item(),
                    logits.detach().flatten()[:8].tolist(),
                    labels.flatten()[:8].tolist(),
                    bce_manual.flatten()[:8].tolist(),
                    pt_manual.flatten()[:8].tolist()))

            if not torch.isfinite(loss):
                if n_batches == 0:
                    print('[dbg] NaN/Inf loss at first batch: L_focal=%s logits_nan=%s logits_inf=%s logits_min=%.2f logits_max=%.2f hg_min=%.2f hg_max=%.2f' % (
                        L_focal.item(), torch.isnan(logits).any().item(), torch.isinf(logits).any().item(),
                        logits.min().item() if torch.isfinite(logits).all() else float('nan'),
                        logits.max().item() if torch.isfinite(logits).all() else float('nan'),
                        h_graph.min().item(), h_graph.max().item()))
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in optimizer.param_groups for parameter in group['params']],
                max_norm=1.0)
            optimizer.step()
            for lp in lambda_params:
                lp.data.clamp_(min=0.001, max=1.0)
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # ---- Phase B: GRU BPTT ----
        if USE_JOINT:
            pb_loss, pb_norm, pb_done = run_joint_phase_b(M_diff, epoch)
            del M_diff
            log('[joint] epoch %d PhaseB: loss=%.4f grad_norm=%.4f steps=%d' %
                (epoch + 1, pb_loss / max(pb_done, 1), pb_norm, pb_done))

        # ---- Validation ----
        set_replay_phase('validation_memory_replay', epoch + 1)
        if USE_MEMORY:
            if USE_JOINT:
                M_val = roll_snapshot(val_cutoff)   # 每 epoch 用当前（Phase B 后）GRU 重滚
            load_snapshot(M_val)   # val/test 用 M_val (ts<val_cutoff)
            if USE_JOINT:
                del M_val   # restore already owns its copy; do not retain another full table
        graph_encoder.eval(); fusion.eval()
        val_predictions = predict_ids(split['val'], val_cutoff, 'val')
        validation_metrics = classification_metrics(
            all_labels[split['val']].numpy(), val_predictions,
            bac_fn=balanced_accuracy_score)
        cold_val_predictions = predict_ids(cold_start['val'], val_cutoff, 'val')
        cold_validation_metrics = classification_metrics(
            all_labels[cold_start['val']].numpy(), cold_val_predictions)
        val_bac, val_f1, val_ap, val_auc = (
            validation_metrics[key] for key in ('bac', 'f1', 'ap', 'auc'))
        log('[cold-start][val] epoch=%d %s' % (
            epoch + 1, json.dumps(cold_validation_metrics, sort_keys=True)))

        lams = graph_encoder.get_lambda_values()
        log('%d,%.4f,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f' % (
            epoch + 1, avg_loss, n_batches,
            val_bac, val_f1, val_ap, val_auc,
            lams['burn'], lams['create'], lams['mint'], lams['transfer'],
        ))

        epoch_validation = {'active_history': validation_metrics, 'cold_start': cold_validation_metrics}
        record = {
            'epoch': epoch + 1, 'train_loss': avg_loss, 'batches': n_batches,
            'validation': epoch_validation, 'lambda': lams,
            'predictions': {
                'active_history': {'contract_ids': split['val'].tolist(),
                                   'labels': all_labels[split['val']].tolist(),
                                   'probabilities': val_predictions},
                'cold_start': {'contract_ids': cold_start['val'].tolist(),
                               'labels': all_labels[cold_start['val']].tolist(),
                               'probabilities': cold_val_predictions},
            },
        }
        validation_history.append(record)
        write_validation_history(validation_history, VALIDATION_PATH)
        if np.isfinite(val_bac) and val_bac > best_val_bac:
            best_val_bac, best_epoch = val_bac, epoch
            best_validation_metrics = epoch_validation
            best_checkpoint = selected_payload(epoch, val_bac, epoch_validation)
            best_checkpoint['validation_predictions'] = record['predictions']
            decision_threshold = FIXED_THRESHOLD
            if threshold_policy == VALIDATION_THRESHOLD_POLICY:
                decision_threshold = select_validation_threshold(
                    all_labels[split['val']].numpy(), val_predictions)
                best_checkpoint['validation_metrics_at_decision_threshold'] = {
                    'active_history': classification_metrics(
                        all_labels[split['val']].numpy(), val_predictions, threshold=decision_threshold),
                    'cold_start': classification_metrics(
                        all_labels[cold_start['val']].numpy(), cold_val_predictions,
                        threshold=decision_threshold),
                }
            best_checkpoint.update(threshold_policy=threshold_policy,
                                   decision_threshold=decision_threshold,
                                   epoch_selection_threshold=FIXED_THRESHOLD)
            atomic_torch_save(best_checkpoint, CKPT_BEST)
            log('[checkpoint] selected epoch %d val BAC=%.8f' % (epoch + 1, val_bac))
        if best_checkpoint is None:
            raise RuntimeError('No finite validation checkpoint was selected')
        latest = epoch_checkpoint(epoch + 1, training_contract, graph_encoder, fusion,
                                  optimizer, opt_mem, best_checkpoint, validation_history)
        atomic_torch_save(latest, CKPT_LATEST)
        del latest
        optimizer.zero_grad(set_to_none=True)
        _gc.collect()
        log('[checkpoint] committed epoch %d; latest=%s' % (epoch + 1, CKPT_LATEST))

    # ---- Test the exact selected encoder/fusion/Joint GRU ----
    phase3_epoch_loop_finished = True
    if best_checkpoint is None:
        raise RuntimeError('No finite validation checkpoint was selected')
    decision_threshold = best_checkpoint.get('decision_threshold', FIXED_THRESHOLD)
    if threshold_policy == VALIDATION_THRESHOLD_POLICY and (
            best_checkpoint.get('threshold_policy') != threshold_policy
            or not isinstance(best_checkpoint.get('decision_threshold'), float)
            or not 0 <= decision_threshold <= 1):
        raise RuntimeError('Selected Phase3 checkpoint has an invalid validation threshold')
    log('[decision threshold] %.17g policy=%s' % (decision_threshold, threshold_policy))
    load_model_weights(graph_encoder, best_checkpoint['graph_encoder'])
    load_model_weights(fusion, best_checkpoint['fusion'])
    graph_encoder.eval(); fusion.eval()
    set_replay_phase('final_test_memory_replay', best_epoch + 1)

    # ---- Test ----
    if USE_MEMORY:
        if USE_JOINT:
            M_val = roll_snapshot(val_cutoff)   # best GRU 已由 load_state_dict(best) 恢复，用其重滚
        load_snapshot(M_val)
    test_metrics = classification_metrics(
        all_labels[split['test']].numpy(),
        predict_ids(split['test'], val_cutoff, 'val'),
        bac_fn=balanced_accuracy_score, threshold=decision_threshold)
    cold_test_metrics = classification_metrics(
        all_labels[cold_start['test']].numpy(),
        predict_ids(cold_start['test'], val_cutoff, 'val'), threshold=decision_threshold)
    test_bac, test_f1, test_ap, test_auc, test_rec = (
        test_metrics[key] for key in ('bac', 'f1', 'ap', 'auc', 'recall'))
    cm = np.array([[test_metrics['tn'], test_metrics['fp']],
                   [test_metrics['fn'], test_metrics['tp']]])
    metrics_path = os.path.splitext(RESULT_PATH)[0] + '.json'
    with open(metrics_path, 'w', encoding='utf-8') as handle:
        json.dump({'training_protocol': PROTOCOL_VERSION,
                   'node_feature_protocol': node_feature_protocol,
                   'node_feature_names': list(node_feature_names),
                   'node_feat_dims': node_feat_dims,
                   'smoke_subset': '--smoke' in sys.argv,
                   'amount_experiment': amount_meta,
                   'observation_protocol': OBSERVATION_PROTOCOL,
                   'selection_metric': 'validation_active_history_bac',
                   'epoch_selection_threshold': FIXED_THRESHOLD,
                   'threshold_policy': threshold_policy,
                   'decision_threshold': decision_threshold,
                   'best_epoch': best_epoch + 1,
                   'validation': best_validation_metrics,
                   'validation_at_decision_threshold': best_checkpoint.get(
                       'validation_metrics_at_decision_threshold', best_validation_metrics),
                   'test': {'active_history': test_metrics, 'cold_start': cold_test_metrics},
                   'split_meta': split_meta}, handle, ensure_ascii=False,
                  indent=2, allow_nan=False)

    # ---- Save results to txt (RugScreener-style) ----
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        f.write('=' * 60 + '\n')
        f.write('BHTG Phase 3 — Test Results\n')
        f.write('Smoke subset: %s\n' % ('--smoke' in sys.argv))
        if amount_meta:
            f.write('Amount evaluation: retrospective_saved_preprocessed_amounts; preprocessing causal=False\n')
        f.write('=' * 60 + '\n')
        f.write('Evaluation protocol: %s\n' % EVAL_PROTOCOL)
        f.write('Node features: %s; input_dim=%d; names=%s\n' % (
            node_feature_protocol, len(node_feature_names), ','.join(node_feature_names)))
        f.write('Relation-balanced sampler: %s quotas=%s\n' % (
            RELATION_BALANCED, str(RELATION_QUOTAS)))
        f.write('Strict temporal training: %s\n' % STRICT_RUN)
        f.write('Phase2 graph checkpoint: %s\n' % CKPT_GRAPH)
        f.write('Split manifest: %s\n' % SPLIT_PATH)
        f.write('Split overlap: train-val=%d train-test=%d val-test=%d all=%d\n' % (
            overlaps['train_val'], overlaps['train_test'], overlaps['val_test'], overlaps['all_three']))
        if EVAL_PROTOCOL == 'rugscreener_code':
            f.write('Protocol warning: source rows are not chronological and val/test benign counts are very small.\n')
        f.write('Best epoch: %d (val BAC=%.4f)\n' % (best_epoch + 1, best_val_bac))
        f.write('Observation protocol: %s\n' % OBSERVATION_PROTOCOL)
        f.write('Selection metric: validation_active_history_bac\n')
        f.write('Epoch selection threshold: %.17g\n' % FIXED_THRESHOLD)
        f.write('Decision threshold: %.17g (%s)\n' % (decision_threshold, threshold_policy))
        f.write('Primary cohort: active_history (n=%d, benign=%d, rug=%d)\n' % (
            test_metrics['count'], test_metrics['benign'], test_metrics['rug']))
        f.write('Cold-start test (separate): %s\n' % json.dumps(cold_test_metrics, sort_keys=True))
        f.write('Cold-start BAC/AUC are unavailable for empty or single-class cohorts.\n')
        f.write('Observation audit: %s\n' % OBSERVATION_PATH)
        f.write('Metrics JSON: %s\n' % metrics_path)
        f.write('\nTest Metrics (active history):\n')
        f.write('  BAC:                %.4f\n' % test_bac)
        f.write('  F1:                 %.4f\n' % test_f1)
        f.write('  AP:                 %.4f\n' % test_ap)
        f.write('  AUC:                %.4f\n' % test_auc)
        f.write('  Recall:             %.4f\n' % test_rec)
        f.write('\nConfusion Matrix:\n')
        f.write('  TN=%d  FP=%d\n' % (cm[0,0], cm[0,1]))
        f.write('  FN=%d  TP=%d\n' % (cm[1,0], cm[1,1]))
        f.write('  FPR=%.4f  FNR=%.4f\n' % (
            cm[0,1] / (cm[0,1] + cm[0,0]) if (cm[0,1] + cm[0,0]) > 0 else 0,
            cm[1,0] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0,
        ))
        lams = graph_encoder.get_lambda_values()
        f.write('\nLearned lambda:\n')
        for k, v in lams.items():
            f.write('  lam_%s = %.4f\n' % (k, v))
        f.write('\nConfig:\n')
        f.write('  HIDDEN_DIM=%d\n' % HIDDEN_DIM)
        f.write('  NUM_HEADS=%d\n' % NUM_HEADS)
        f.write('  NUM_LAYERS=%d\n' % NUM_LAYERS)
        f.write('  BATCH_SIZE=%d\n' % BATCH_SIZE)
        f.write('  LEARNING_RATE=%.5f\n' % LEARNING_RATE)
        f.write('  LAMBDA_LR=%.5f\n' % LAMBDA_LR)

    log('\n[Save] %s (best epoch %d)' % (CKPT_BEST, best_epoch + 1))
    log('[Results] %s' % RESULT_PATH)
    log('  BAC: %.4f | F1: %.4f | AP: %.4f | AUC: %.4f' % (test_bac, test_f1, test_ap, test_auc))
    log('Phase 3 complete.')


if __name__ == "__main__":
    main()
