"""
BHTG 统一配置中心
"""
import os
import torch


# Node-feature experiments keep the historical 15-column route as the default.
NODE_FEATURE_MODE = os.environ.get('BHTG_NODE_FEATURE_MODE', 'legacy15').strip().lower()
if NODE_FEATURE_MODE not in ('legacy15', 'behavior9'):
    raise ValueError('BHTG_NODE_FEATURE_MODE must be legacy15 or behavior9')

# Explicit, matched architecture switches. Old callers retain the old protocol.
SEMANTIC_AMOUNT_MODE = os.environ.get('BHTG_SEMANTIC_AMOUNT_MODE', 'legacy')
MEMORY_AMOUNT_MODE = os.environ.get('BHTG_MEMORY_AMOUNT_MODE', 'legacy')
_amount_modes = ('legacy', 'disabled', 'continuous_zscore')
if SEMANTIC_AMOUNT_MODE not in _amount_modes or MEMORY_AMOUNT_MODE not in _amount_modes:
    raise ValueError('amount modes must be legacy, disabled or continuous_zscore')
AMOUNT_EXPERIMENT = SEMANTIC_AMOUNT_MODE != 'legacy' or MEMORY_AMOUNT_MODE != 'legacy'
if AMOUNT_EXPERIMENT and 'legacy' in (SEMANTIC_AMOUNT_MODE, MEMORY_AMOUNT_MODE):
    raise ValueError('new amount experiments require both explicit new architecture modes')
AMOUNT_SOURCE_POLICY = os.environ.get('BHTG_AMOUNT_SOURCE_POLICY', 'rugscreener_saved_zscore_v1')
AMOUNT_MASK_RATE = float(os.environ.get('BHTG_AMOUNT_MASK_RATE', '.15'))
AMOUNT_LOSS_WEIGHT = float(os.environ.get('BHTG_AMOUNT_LOSS_WEIGHT', '.1'))
AMOUNT_MLP_HIDDEN = int(os.environ.get('BHTG_AMOUNT_MLP_HIDDEN', '64'))
import math
if (not 0 <= AMOUNT_MASK_RATE <= 1 or not math.isfinite(AMOUNT_LOSS_WEIGHT)
        or AMOUNT_LOSS_WEIGHT < 0 or AMOUNT_MLP_HIDDEN < 1):
    raise ValueError('invalid amount masking/loss/MLP configuration')
if AMOUNT_EXPERIMENT and AMOUNT_SOURCE_POLICY != 'rugscreener_saved_zscore_v1':
    raise ValueError('unsupported amount source policy')

# ========== 路径配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
BHTG_DATA_DIR = os.path.join(BASE_DIR, 'data')
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, 'checkpoints')

LABELED_CSV = os.path.join(DATA_DIR, 'transfer_std_label.csv')
UNLABELED_CSV = os.path.join(DATA_DIR, 'token_transfer.csv')
ADDR_MAP_CSV = os.path.join(DATA_DIR, 'address_to_id_std_label.csv')

# 中间格式输出 (CSV)
LABELED_EDGES_CSV = os.path.join(BHTG_DATA_DIR, 'labeled_edges.csv')
UNLABELED_EDGES_CSV = os.path.join(BHTG_DATA_DIR, 'unlabeled_edges.csv')
ADDR_TO_ID_CSV = os.path.join(BHTG_DATA_DIR, 'addr_to_id.csv')
TOKEN_LIST_CSV = os.path.join(BHTG_DATA_DIR, 'token_list.csv')
GRAPH_BIN = os.path.join(BHTG_DATA_DIR, 'bhtg_graph.bin')

# A new bundle is selected explicitly; inspection does not create directories.
BHTG_BUNDLE_DIR = os.environ.get('BHTG_BUNDLE_DIR')
V3_BUNDLE_SELECTED = 'BHTG_BUNDLE_DIR' in os.environ
if AMOUNT_EXPERIMENT and V3_BUNDLE_SELECTED:
    raise ValueError('continuous amount architecture is only supported for four-actions inputs')
GRAPH_MANIFEST_PATH = None
TRAINING_BUNDLE_INFO = None
ACTION_CSV = None
FOUR_ACTIONS_SELECTED = False
ACTIONS_SELECTION = None
if V3_BUNDLE_SELECTED:
    if not str(BHTG_BUNDLE_DIR or '').strip():
        raise ValueError('empty BHTG_BUNDLE_DIR cannot select legacy inputs')
    from bhtg.training_data_v3 import inspect_training_bundle
    TRAINING_BUNDLE_INFO = inspect_training_bundle(BHTG_BUNDLE_DIR)
    BHTG_BUNDLE_DIR = TRAINING_BUNDLE_INFO['root']
    BHTG_DATA_DIR = os.path.join(BHTG_BUNDLE_DIR, 'graph')
    DATA_DIR = os.path.join(BHTG_BUNDLE_DIR, 'assets')
    CHECKPOINT_DIR = os.path.join(BHTG_BUNDLE_DIR, 'outputs')
    LABELED_EDGES_CSV = os.path.join(DATA_DIR, 'labeled_edges.csv')
    UNLABELED_EDGES_CSV = os.path.join(DATA_DIR, 'unlabeled_edges.csv')
    ADDR_TO_ID_CSV = os.path.join(BHTG_DATA_DIR, 'addr_to_id.csv')
    TOKEN_LIST_CSV = os.path.join(BHTG_DATA_DIR, 'token_list.csv')
    GRAPH_BIN = os.path.join(BHTG_DATA_DIR, 'bhtg_graph.bin')
    GRAPH_MANIFEST_PATH = os.path.join(BHTG_DATA_DIR, 'manifest.json')
    # These old normalized-CSV routes are intentionally unavailable in v3.
    LABELED_CSV = UNLABELED_CSV = ADDR_MAP_CSV = None
else:
    from bhtg.action_csv import active_actions_selection
    ACTIONS_SELECTION = active_actions_selection()
    if ACTIONS_SELECTION is not None:
        FOUR_ACTIONS_SELECTED = True
        ACTION_CSV = ACTIONS_SELECTION['actions_csv']['path']
        # Compatibility aliases share one canonical file; readers filter roles.
        LABELED_EDGES_CSV = UNLABELED_EDGES_CSV = ACTION_CSV
        DATA_DIR = os.path.dirname(ACTION_CSV)
        BHTG_DATA_DIR = os.path.join(DATA_DIR, 'model')
        CHECKPOINT_DIR = os.path.join(DATA_DIR, 'outputs')
        ADDR_TO_ID_CSV = os.path.join(BHTG_DATA_DIR, 'addr_to_id.csv')
        TOKEN_LIST_CSV = os.path.join(BHTG_DATA_DIR, 'token_list.csv')
        GRAPH_BIN = os.path.join(BHTG_DATA_DIR, 'bhtg_graph.bin')
        GRAPH_MANIFEST_PATH = os.path.join(BHTG_DATA_DIR, 'manifest.json')
        LABELED_CSV = UNLABELED_CSV = ADDR_MAP_CSV = None
    else:
        os.makedirs(BHTG_DATA_DIR, exist_ok=True)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)


if AMOUNT_EXPERIMENT:
    if not FOUR_ACTIONS_SELECTED:
        raise ValueError('new amount experiments require a hash-bound four-actions selection')
    graph_dir = os.environ.get('BHTG_FOUR_ACTIONS_GRAPH_DIR', '').strip()
    output_dir = os.environ.get('BHTG_OUTPUT_DIR', '').strip()
    if not graph_dir or not output_dir:
        raise ValueError('new amount experiments require BHTG_FOUR_ACTIONS_GRAPH_DIR and BHTG_OUTPUT_DIR')
    _old_dirs = (BHTG_DATA_DIR, CHECKPOINT_DIR, os.path.join(BASE_DIR, 'data'),
                 os.path.join(PROJECT_DIR, 'checkpoints'))
    def _inside(path, root):
        path, root = os.path.normcase(os.path.abspath(path)), os.path.normcase(os.path.abspath(root))
        return os.path.commonpath([path, root]) == root if os.path.splitdrive(path)[0] == os.path.splitdrive(root)[0] else False
    if any(_inside(path, old) for path in (graph_dir, output_dir) for old in _old_dirs):
        raise ValueError('amount experiment directories must be isolated from historical model/outputs')
    if _inside(graph_dir, output_dir) or _inside(output_dir, graph_dir):
        raise ValueError('amount graph and output directories must be separate')
    BHTG_DATA_DIR = os.path.abspath(graph_dir)
    CHECKPOINT_DIR = os.path.abspath(output_dir)
    ADDR_TO_ID_CSV = os.path.join(BHTG_DATA_DIR, 'addr_to_id.csv')
    TOKEN_LIST_CSV = os.path.join(BHTG_DATA_DIR, 'token_list.csv')
    GRAPH_BIN = os.path.join(BHTG_DATA_DIR, 'bhtg_graph.bin')
    GRAPH_MANIFEST_PATH = os.path.join(BHTG_DATA_DIR, 'manifest.json')


if NODE_FEATURE_MODE == 'behavior9':
    if not AMOUNT_EXPERIMENT or not FOUR_ACTIONS_SELECTED or V3_BUNDLE_SELECTED:
        raise ValueError('behavior9 requires a hash-bound four-actions amount experiment')
    # Reuse the read-only amount graph, but never write new weights/logs into
    # the existing 15-feature experiment's output directory.
    if os.path.basename(os.path.normpath(CHECKPOINT_DIR)).lower() not in (
            'behavior9', 'behavior9_unlabeled_full_history'):
        CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'behavior9')


# ========== 图参数 ==========
NODE_TYPES = ['eoa', 'contract', 'pool']
LEGACY_EDGE_TYPES = ['transfer', 'mint', 'burn', 'create']
from bhtg.model_schema_v3 import V3_EDGE_TYPES
EDGE_TYPES = list(V3_EDGE_TYPES) if V3_BUNDLE_SELECTED else list(LEGACY_EDGE_TYPES)

# 零地址
ZERO_ADDR = '0x0000000000000000000000000000000000000000'


# ========== 模型参数 ==========
HIDDEN_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 2
EDGE_ATTR_DIM = 4  # 边属性统一投影维度
DROPOUT = 0.3

# 时间衰减初始值 (可学习) — λ·Δt/86400 (Δt in seconds → days)
# λ_transfer=0.50: half-life ≈ 1.4 days (transfers quickly become noise)
# λ_burn=0.01:    half-life ≈ 69 days  (burn events remembered for months)
# λ_mint=0.05:    half-life ≈ 14 days
# λ_create=0.05:  half-life ≈ 14 days
LAMBDA_INIT = {
    'transfer': 0.50,
    'mint': 0.05,
    'burn': 0.01,
    'create': 0.05,
}
if V3_BUNDLE_SELECTED:
    LAMBDA_INIT.update(lp_mint=0.05, lp_burn=0.01, lp_swap=0.50, lp_sync=0.50)


# ========== 语义编码器参数 ==========
# 由于网络不可用，默认使用 LSTM 替代 BERT
# 设置 USE_BERT = True 需要有网络且安装了 transformers
USE_BERT = False
BERT_MODEL_NAME = 'bert-base-uncased'
LSTM_HIDDEN_DIM = 128
LSTM_NUM_LAYERS = 2
MAX_SEQ_LEN = 128
VOCAB_SIZE = 5000  # BHTG 专属词表大小

# ========== LMAE4Eth 语义阶段（run_phase1_lmae.py, semantic_lmae.py）==========
# 无网络 → from-scratch BERT（架构+MLM/NSP/Token对比 目标逐字移植 LMAE4Eth contrastive.py）
SEM_BERT_VOCAB = 64          # 结构化字段词表（实际 ~43，余量给 [UNK]）
SEM_BERT_HIDDEN = 256        # BERT hidden（phase1 独立，与 phase3 4GB 解耦）
SEM_BERT_LAYERS = 6
SEM_BERT_HEADS = 8
SEM_BERT_FFN = 1024
SEM_MAX_LEN = 128            # max_position_embeddings + padding 长度
SEM_CTL_TEMP = 0.01          # token 对比 temperature（LMAE4Eth 默认）
SEM_CTL_MOMENTUM = 0.996     # anchor BERT EMA momentum
SEM_MASK_RATE = 0.15         # MLM 掩码率
SEM_NSP_PROB = 0.5           # NSP 负例概率（随机另一 token 窗口）
SEM_NSP_WINDOW = 8           # 每条"句子"含多少条边
SEM_BATCH = 8                # BERT 训练批大小
SEM_EPOCHS = 10


# ========== 训练参数 ==========
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
EPOCHS_PRETRAIN_SEM = 10
EPOCHS_PRETRAIN_GRAPH = 10
EPOCHS_FINETUNE = 50

TRAIN_SPLIT = 0.7
VALID_SPLIT = 0.85

# 损失权重
ALPHA_ALIGN = 0.1
BETA_GRAPH_CON = 0.1
GAMMA_MLM = 0.05

# Focal Loss 参数
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# 邻居采样数
N_NEIGHBORS = 10
NEG_SAMPLES = 1

# ========== 关系平衡时间采样 ==========
# 默认逐目标采样；False 表示不预留关系配额（仍每个目标独享 k 预算）。
# 运行时可用 --relation-balanced
# (Phase3) 或 env RELATION_BALANCED_SAMPLING=1 (Phase2/3) 开启。
# 每个目标每跳总预算 N_NEIGHBORS=10：按配额保留稀有关系，
# 未使用的配额由所有关系中时间最近的事件补齐；不诱导恢复未采样边。
RELATION_BALANCED_SAMPLING = False
RELATION_SAMPLE_QUOTAS = {
    'create': 1,
    'burn': 2,
    'mint': 2,
    'transfer': 5,
}
if V3_BUNDLE_SELECTED:
    RELATION_SAMPLE_QUOTAS = {'create': 1, 'burn': 1, 'mint': 1, 'transfer': 2,
                             'lp_mint': 1, 'lp_burn': 2, 'lp_swap': 1, 'lp_sync': 1}

# ========== 严格时间训练协议 ==========
# True: Phase1 语义只看 train_cutoff 前交易；Phase2 只用 train_cutoff 前正边，
#       按时间连续组 batch，并以 batch 最小时间作为历史 cutoff，杜绝目标边/未来边泄漏。
# 训练入口强制严格时序；旧完整历史和批内 max cutoff 已不再支持。
STRICT_TEMPORAL = True

# ========== 节点特征扩展 (P0, 论文 PhishEye 启发) ==========
# True  → 8 维基础上追加 7 维 (in_val/out_val/dtd/cir/ncd/pntd/andc) → 15 维
# False → 只用旧 8 维 (重建图可复现 0.9095 基线; 消融用)
NODE_FEAT_EXTRA = True


# ========== 节点级对比自监督 (P1, 论文 PhishEye §3.3) ==========
# phase2 边预测 BCE 之上叠加 InfoNCE: 双视图增强 (20% 边 Bernoulli drop + 20% 特征维度置零),
# 对齐 batch 内 seed token 节点表示 → 缓解 phase2 BCE 饱和/过拟合, 学视图不变表示。
# True → 启用 (需重训 phase2 + phase3);  False → 纯 BCE (消融/复现旧行为)
USE_SSL = True
SSL_DROP_RATE = 0.2   # 每视图边丢弃率
SSL_MASK_RATE = 0.2   # 每视图特征维度置零率
SSL_TEMP = 0.5        # InfoNCE temperature (z 归一化后 /τ)
ALPHA_SSL = 0.5       # ssl 损失权重 (叠加在 BCE 上; SSL 原始 loss ~2.5-3 > BCE ~1, 0.5 使两者贡献可比)

# λ 参数的独立学习率（比主学习率大，因为跨所有边平均后梯度很小）
LAMBDA_LR = 0.01  # 100x main LR


# ========== model-level homogenization 消融 (BHTG 方向 2: w/o Hetero) ==========
# True → HGT encoder 内所有 node/edge types 共享同一 projection / conv / λ / LayerNorm
#   (图数据/边流/memory GRU/节点特征全部 byte-identical, 只在模型层同质化,
#    等价 single-type directed GAT + 共享时间衰减; 检验异构类型专用参数是否有正向效果)
# 运行脚本还支持 env HOMO_ABLATION=1 覆盖 (消融跑不用改 config)。
# ⚠️ 与 SSL-hetero checkpoint (BAC 0.9228) 结构不兼容 → 需独立重训 phase2+phase3。
HOMO_ABLATION = False


# ========== TGN Memory 机制 (方向 1) ==========
MEM_DIM = 128            # 每节点记忆向量维度 (= HIDDEN_DIM, additive 注入不改 HGT 输入)
TEMPORAL_DIM = 32        # TimeEncode 傅里叶时间编码维度
MEM_VALUE_CLIP = 5.0     # value 是逐 token Z-score (-5.4~31.6 重尾)，clip 到 [-5,5]
MEM_EDGE_BATCH = 2048    # phase2_memory 训练 GRU 的边批大小
MEM_ROLL_BATCH = 8192    # phase3 冻结 roll 的边批大小 (纯 GRU 更新，大批省开销)
MEM_EPOCHS = 8           # phase2_memory 预训练 epoch 数
MEM_NEG_CANDIDATES = 32  # 每正边负采样的候选数 (时间相关过滤后取第一个)
MEM_LR = 1e-3            # GRU 学习率 (比主模型高，memory 学得快)

# ========== Phase3 joint-memory (每 epoch 重滚 + 联合训练) ==========
MEM_JOINT_SAMPLE = 256        # Phase B 每步 BPTT 用的训练 token 子样本（rug/benign 分层各半）
MEM_JOINT_TAIL_CHUNKS = 8     # 截断 BPTT: 只让最后 N 个 roll 块带梯度（8 块 ≈ 65k 边 ≈ 0.6GB grad 图）。
                              # 本机仅 ~2.4GB 可用 RAM，tail=16 的 ~1.2GB 图叠基座会 OOM，8 是安全上限。
                              # 内存富余的机器可调回 16（~131k 边，信号更密）。
MEM_JOINT_STEPS = 1           # 每 epoch Phase B 的 BPTT 更新步数（>1 复用 M_diff，加信号）


# ========== 设备 ==========
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ========== 随机种子 ==========
SEED = 2021
