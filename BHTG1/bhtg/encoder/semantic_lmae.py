"""
BHTG semantic encoder with legacy masked or explicit continuous amount inputs.

基于 LMAE4Eth `model/contrastive.py` 的语义阶段，权重 from-scratch 随机初始化，
不使用 bert-base-uncased。保留27项词表、MLM/NSP/逐token对比，新增输出投影
的 pooled InfoNCE。旧 checkpoint 的 out_proj 未训练，必须重新运行 Phase1。

架构：
  1. BertForPreTraining（from-scratch BertConfig）：bert encoder + cls head（MLM + NSP）
  2. MLM loss：15% 掩码，80%→[MASK] 10%→random 10%→keep，预测被掩 token
  3. NSP loss：sentence A/B（token 交易历史切窗口），50% 随机负例
  4. Token-aware 对比 loss L_Ta：冻结 EMA anchor BERT 编码未掩码输入 → 逐位置目标；
     在线 masked rep 算 (B,L,L) 余弦相似度矩阵（temperature=0.01），gold=对角线，
     label_smoothed_nll_loss（跨位置 softmax，contrastive_labels 只加权被掩位置）

输出 encode_sequence → pooled(256) → Linear → 128（= HIDDEN_DIM，phase3 融合层不动）。
推理：全部可见历史按完整事件分窗，共享 BERT 编码后按事件数加权平均。
"""
import math
import copy
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertForPreTraining, BertModel

# ============================================================
# 1. Structured Field Vocabulary + Tokenizer
# ============================================================

class FieldVocab:
    """
    BHTG 交易结构化字段词表（~30 token）。

    每条边 → token 序列：[OP] SRC_T → DST_T AMT:x DT:y
    默认屏蔽旧全历史 Z-score 金额；只有显式 raw_log 模式才读取原始金额。
    身份（真实地址）不进文本——由图流覆盖，文本只表达操作/类型/金额/时序。
    """

    # Historical checkpoint vocabulary has exactly 27 entries (no NO_HISTORY).
    SPECIAL = ['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
    OPS = ['TRANSFER', 'MINT', 'BURN', 'CREATE', 'CALL']
    NODE_TYPES = ['EOA', 'CONTRACT', 'POOL']
    ARROW = ['->']
    AMT_BUCKETS = ['AMT:NEG_BIG', 'AMT:NEG', 'AMT:ZERO', 'AMT:POS', 'AMT:POS_BIG']
    DT_BUCKETS = ['DT:START', 'DT:SEC', 'DT:MIN', 'DT:HOUR', 'DT:DAY']
    VIEWS = ['[DEPLOYER]', '[TOKEN]', '[POOL]']

    PAD, CLS, SEP, MASK, UNK = 0, 1, 2, 3, 4

    def __init__(self, node_type_fn=None, amount_policy='masked_global_zscore', relation_schema=None):
        """
        Args:
            node_type_fn: str(addr) → 'EOA'|'CONTRACT'|'POOL'（默认全 'EOA'）
        """
        self.node_type_fn = node_type_fn
        if amount_policy not in ('masked_global_zscore', 'raw_log', 'continuous_zscore', 'disabled'):
            raise ValueError('unsupported semantic amount policy')
        self.amount_policy = amount_policy
        self.structured_amounts = amount_policy in ('continuous_zscore', 'disabled')
        from bhtg.model_schema_v3 import V3_EDGE_TYPES, LEGACY_EDGE_TYPES
        self.relation_schema = tuple(relation_schema or LEGACY_EDGE_TYPES)
        if self.relation_schema not in (LEGACY_EDGE_TYPES, V3_EDGE_TYPES):
            raise ValueError('unsupported semantic relation schema')
        if self.structured_amounts and self.relation_schema != LEGACY_EDGE_TYPES:
            raise ValueError('continuous amount semantics only support four actions')
        operations = self.OPS + (['LP_MINT', 'LP_BURN', 'LP_SWAP', 'LP_SYNC']
                                 if self.relation_schema == V3_EDGE_TYPES else [])
        tokens = self.SPECIAL + operations + self.NODE_TYPES + self.ARROW + \
            self.AMT_BUCKETS + self.DT_BUCKETS + self.VIEWS
        if self.structured_amounts:
            tokens += ['[AMT_CONT]', '[AMT_MISSING]', '[AMT_DISABLED]']
        self.token_to_id = {t: i for i, t in enumerate(tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.vocab_size = len(tokens)
        assert self.token_to_id['[PAD]'] == 0

        self._op_map = {'transfer': 'TRANSFER', 'mint': 'MINT',
                        'burn': 'BURN', 'create': 'CREATE'}
        if self.relation_schema == V3_EDGE_TYPES:
            self._op_map.update(lp_mint='LP_MINT', lp_burn='LP_BURN',
                                lp_swap='LP_SWAP', lp_sync='LP_SYNC')
        self._type_id = {'EOA': self.token_to_id['EOA'],
                         'CONTRACT': self.token_to_id['CONTRACT'],
                         'POOL': self.token_to_id['POOL']}
        # 元 token（[MASK] 等）直接给 id 方便 masked 输入
        self._arrow_id = self.token_to_id['->']
        self.TOKEN_VIEW = self.token_to_id['[TOKEN]']
        self._op_ids = {self._op_map[k]: self.token_to_id[self._op_map[k]] for k in self._op_map}
        self._op_ids['CALL'] = self.token_to_id['CALL']
        self.amount_token_ids = {self.token_to_id[t] for t in
                                 tuple(self.AMT_BUCKETS) + ('[AMT_CONT]', '[AMT_MISSING]', '[AMT_DISABLED]')
                                 if t in self.token_to_id}

    # ---- field → token id 辅助 ----
    def _resolve_type_id(self, addr_str):
        t = self.node_type_fn(str(addr_str)) if self.node_type_fn else 'EOA'
        return self._type_id.get(t, self._type_id['EOA'])

    def _amt_bucket(self, v):
        if v == 0:
            return self.token_to_id['AMT:ZERO']
        if v <= -1.5:
            return self.token_to_id['AMT:NEG_BIG']
        if v < -0.5:
            return self.token_to_id['AMT:NEG']
        if v >= 1.5:
            return self.token_to_id['AMT:POS_BIG']
        if v > 0.5:
            return self.token_to_id['AMT:POS']
        return self.token_to_id['AMT:ZERO']

    def _dt_bucket(self, dt):
        if dt <= 0:
            return self.token_to_id['DT:START']
        if dt < 60:
            return self.token_to_id['DT:SEC']
        if dt < 3600:
            return self.token_to_id['DT:MIN']
        if dt < 86400:
            return self.token_to_id['DT:HOUR']
        return self.token_to_id['DT:DAY']

    def edge_to_token_ids(self, edge, prev_ts=None):
        """
        edge: dict（seq_builder.token_edges 里的原始边记录，含
              edge_type/src_addr/dst_addr/value/timestamp）→ list[int]（6 个 token id）
        """
        etype = edge.get('edge_type', 'transfer')
        if len(getattr(self, 'relation_schema', ('transfer', 'mint', 'burn', 'create'))) > 4 and etype not in self._op_map and etype != 'call':
            raise ValueError('unknown relation in v3 semantic sequence')
        src_t = self._resolve_type_id(edge.get('src_addr', ''))
        dst_t = self._resolve_type_id(edge.get('dst_addr', ''))
        if self.amount_policy == 'raw_log':
            if 'raw_value' not in edge:
                raise ValueError('raw_log requires raw_value; normalized value is not a raw amount')
            raw = float(edge['raw_value'])
            if not math.isfinite(raw):
                raise ValueError('raw_value must be finite')
            val = math.copysign(math.log1p(abs(raw)), raw)
        else:
            # The supplied CSV/cache values used full-history statistics.
            # No causal raw amount source is available, so omit that field.
            val = 0.0
        ts = float(edge.get('timestamp', 0))
        dt = max(0.0, ts - prev_ts) if prev_ts is not None else 0.0
        amount_token = self._amt_bucket(val)
        if self.structured_amounts:
            if not {'amount_z', 'amount_known', 'source_row_id'} <= set(edge):
                raise ValueError('continuous sequence requires amount_z/amount_known/source_row_id')
            if not isinstance(edge['amount_known'], bool):
                # numpy bool is common in CSV preprocessing; reject numeric truthiness.
                if type(edge['amount_known']).__name__ != 'bool_':
                    raise ValueError('amount_known must be bool')
            amount_token = self.token_to_id['[AMT_DISABLED]' if self.amount_policy == 'disabled'
                                             else '[AMT_CONT]' if bool(edge['amount_known'])
                                             else '[AMT_MISSING]']
        return [
            self._op_ids.get(self._op_map.get(etype, 'CALL'), self._op_ids['CALL']),
            src_t, self._arrow_id, dst_t,
            amount_token, self._dt_bucket(dt),
        ]

    def edges_to_ids(self, edges, max_edges=32, keep='recent', *, prev_ts=None):
        """
        边序列 → 扁平 token id 列表；prev_ts 可保留上一窗口末事件的时间。

        Args:
            edges: 时间排序的原始边记录
            max_edges: 最多取多少条边（超限截断）
            keep: 'recent'（留最近，靠标签时间的边更有信息量）/ 'first'（留最早）
            prev_ts: 已可见的前一事件时间；None 表示历史起点
        """
        if len(edges) > max_edges:
            edges = edges[-max_edges:] if keep == 'recent' else edges[:max_edges]
        ids = []
        for e in edges:
            ids.extend(self.edge_to_token_ids(e, prev_ts))
            prev_ts = float(e['timestamp'])
        return ids

    def edges_to_sequence(self, edges, max_edges=32, keep='recent', *, prev_ts=None):
        if not self.structured_amounts:
            raise ValueError('structured sequence requires a new amount policy')
        edges = list(edges)
        if len(edges) > max_edges:
            edges = edges[-max_edges:] if keep == 'recent' else edges[:max_edges]
        result = StructuredSequence([], [], [], [], [])
        previous = prev_ts
        seen = set()
        for edge in edges:
            identity = str(edge['source_row_id'])
            if not identity or identity in seen:
                raise ValueError('missing or repeated source_row_id in semantic history: ' + identity)
            seen.add(identity)
            known = bool(edge['amount_known'])
            value = float(edge['amount_z'])
            if not math.isfinite(value) or not torch.isfinite(torch.tensor(value, dtype=torch.float32)):
                raise ValueError('nonfinite semantic amount: ' + identity)
            if not known and value != 0:
                raise ValueError('unknown amount must have zero placeholder: ' + identity)
            result.input_ids.extend(self.edge_to_token_ids(edge, previous))
            result.amount_values.extend([0., 0., 0., 0., value, 0.])
            result.amount_position_mask.extend([False, False, False, False, True, False])
            result.amount_known.extend([False, False, False, False, known, False])
            result.event_identity.extend([identity] * 6)
            previous = float(edge['timestamp'])
        return result


@dataclass
class StructuredSequence:
    """Aligned event fields. Identities are audit data and never BERT features."""
    input_ids: list
    amount_values: list
    amount_position_mask: list
    amount_known: list
    event_identity: list

    def __post_init__(self):
        lengths = {len(getattr(self, name)) for name in self.__dataclass_fields__}
        if len(lengths) != 1:
            raise ValueError('unaligned structured semantic sequence')

    def __len__(self):
        return len(self.input_ids)

    def slice(self, start, stop):
        return StructuredSequence(**{name: getattr(self, name)[start:stop]
                                     for name in self.__dataclass_fields__})


def collate_structured(batch, vocab, max_len):
    """Truncate by complete six-position events and pad every side channel."""
    if max_len < 9:
        raise ValueError('max_len must fit a complete event and separators')
    rows = []
    for instance in batch:
        a, b = instance['a'], instance['b']
        capacity = (max_len - 3) // 6
        na, nb = len(a) // 6, len(b) // 6
        while na + nb > capacity:
            if na > nb:
                na -= 1
            else:
                nb -= 1
        a, b = a.slice(0, 6 * na), b.slice(0, 6 * nb)
        special = lambda token: StructuredSequence([token], [0.], [False], [False], [None])
        pieces = [special(vocab.CLS), a, special(vocab.SEP), b, special(vocab.SEP)]
        row = {name: sum((getattr(piece, name) for piece in pieces), [])
               for name in StructuredSequence.__dataclass_fields__}
        padding = max_len - len(row['input_ids'])
        for name, value in [('input_ids', vocab.PAD), ('amount_values', 0.),
                            ('amount_position_mask', False), ('amount_known', False),
                            ('event_identity', None)]:
            row[name] += [value] * padding
        row['token_type_ids'] = [0] * (len(a) + 2) + [1] * (len(b) + 1) + [0] * padding
        row['nsp_flags'] = instance['flag']
        row['nsp_valid'] = instance['nsp_valid'] and bool(na and nb)
        identities = [identity for identity, pos in zip(row['event_identity'], row['amount_position_mask']) if pos]
        if len(identities) != len(set(identities)):
            raise ValueError('repeated source_row_id in online semantic sample')
        rows.append(row)
    output = {}
    for name in rows[0]:
        values = [row[name] for row in rows]
        if name == 'event_identity':
            output[name] = values
        else:
            dtype = (torch.float32 if name == 'amount_values' else torch.bool
                     if name in ('amount_position_mask', 'amount_known', 'nsp_valid') else torch.long)
            output[name] = torch.tensor(values, dtype=dtype)
    output['attention_mask'] = output['input_ids'].ne(vocab.PAD)
    return output


# ============================================================
# 2. MLM 掩码 + NSP 实例构建（对照 google_bert.create_instances_from_document）
# ============================================================

def make_mlm_input(ids, vocab, mask_rate=0.15):
    """
    ids: (B, L) long tensor（含 [CLS]/[SEP]/[PAD]）。
    → (truth, inp, msk)
        truth: 原始 ids
        inp:   掩码后 ids（80% [MASK] / 10% random / 10% keep）
        msk:   (B, L) bool，True = 被掩位置（用于 MLM gold + 对比 contrastive_labels）
    [CLS]/[SEP]/[PAD] 不参与掩码。
    """
    if vocab.structured_amounts:
        raise ValueError('structured amounts require make_continuous_mlm_input with all aligned fields')
    B, L = ids.shape
    maskable = ((ids != vocab.PAD) & (ids != vocab.CLS) & (ids != vocab.SEP))
    chosen = (torch.rand(B, L, device=ids.device) < mask_rate) & maskable
    inp = ids.clone()
    rand = torch.rand(B, L, device=ids.device)
    m80 = chosen & (rand < 0.8)
    inp[m80] = vocab.MASK
    m90 = chosen & (rand >= 0.8) & (rand < 0.9)
    inp[m90] = torch.randint(5, vocab.vocab_size, (int(m90.sum()),), device=ids.device)
    # 其余 10% keep 原 token
    return ids, inp, chosen


def make_continuous_mlm_input(batch, vocab, mask_rate=0.15, amount_mask_rate=0.15,
                              amount_selected=None):
    """Separate discrete 80/10/10 masking from 100% numeric hiding."""
    if not 0 <= mask_rate <= 1 or not 0 <= amount_mask_rate <= 1:
        raise ValueError('mask rates must be in [0, 1]')
    ids = batch['input_ids']
    position, known = batch['amount_position_mask'], batch['amount_known']
    candidates = batch['attention_mask'].bool() & ~position
    for special in (vocab.PAD, vocab.CLS, vocab.SEP, vocab.MASK, vocab.UNK):
        candidates &= ids.ne(special)
    discrete = (torch.rand(ids.shape, device=ids.device) < mask_rate) & candidates
    inp = ids.clone()
    random = torch.rand(ids.shape, device=ids.device)
    inp[discrete & (random < .8)] = vocab.MASK
    replace = discrete & (random >= .8) & (random < .9)
    allowed = torch.tensor([i for i in range(5, vocab.vocab_size)
                            if i not in vocab.amount_token_ids], device=ids.device)
    inp[replace] = allowed[torch.randint(len(allowed), (int(replace.sum()),), device=ids.device)]
    eligible = position & known & batch['attention_mask'].bool()
    if amount_selected is None:
        amount_selected = (torch.rand(ids.shape, device=ids.device) < amount_mask_rate) & eligible
    elif amount_selected.shape != ids.shape or amount_selected.dtype != torch.bool:
        raise ValueError('invalid amount selection mask')
    else:
        amount_selected = amount_selected & eligible
    if vocab.amount_policy == 'disabled':
        amount_selected = torch.zeros_like(eligible)
        inp[position] = vocab.token_to_id['[AMT_DISABLED]']
    inp[amount_selected] = vocab.MASK
    # Reject duplicates even when an external caller bypasses collate.
    if 'event_identity' in batch:
        for identities, positions in zip(batch['event_identity'], position):
            seen = set()
            for identity, pos in zip(identities, positions.tolist()):
                if pos:
                    if identity is None or identity in seen:
                        raise ValueError('repeated or missing event identity in online sample')
                    seen.add(identity)
    online = dict(batch)
    online['input_ids'] = inp
    online['amount_values'] = torch.where(eligible & ~amount_selected,
                                           batch['amount_values'], torch.zeros_like(batch['amount_values']))
    online['amount_selected'] = amount_selected
    return online, discrete, amount_selected


def build_nsp_instances(cid_edges, cids, vocab, window=8, nsp_prob=0.5, keep='recent'):
    """
    cid_edges: {contract_local_id → 时间排序的原始边记录列表}（由 seq_builder.token_edges + cid_to_addr 组装）
    每个 token 的历史切窗口（每窗口 window 条边 = 一条"sentence"）：
      - 连续窗口对 (j, j+1) → 正例候选；nsp_prob 概率用另一 token 的随机窗口替换 B → 负例
      - 单窗口 token → A=前半 B=后半（正例，保证 len(A),len(B)>=1）
      - 无可见历史 token 跳过（旧missing-history行为）
    返回 list of (cid, A_ids, B_ids, nxt_flag)：
      A_ids/B_ids 为 token id list（不含 [CLS]/[SEP]，训练时拼接）。
    """
    import random
    rng = random.Random(0)
    instances = []
    if vocab.structured_amounts:
        if window < 1:
            raise ValueError('window must be positive')
        windows = {cid: [vocab.edges_to_sequence(edges[i:i + window], max_edges=window, keep=keep)
                         for i in range(0, len(edges), window)]
                   for cid in cids for edges in [cid_edges.get(cid, [])]}
        for cid, sequences in windows.items():
            others = [other for other in windows if other != cid and windows[other]]
            for j in range(max(0, len(sequences) - 1)):
                a, b, flag = sequences[j], sequences[j + 1], 0
                if others and rng.random() < nsp_prob:
                    b, flag = rng.choice(windows[rng.choice(others)]), 1
                instances.append({'cid': cid, 'a': a, 'b': b, 'flag': flag, 'nsp_valid': True})
            if len(sequences) == 1:
                sequence = sequences[0]
                # Split whole events; a singleton can still train all non-NSP objectives.
                split = max(1, len(sequence) // 12) * 6
                instances.append({'cid': cid, 'a': sequence.slice(0, split),
                                  'b': sequence.slice(split, len(sequence)), 'flag': 0,
                                  'nsp_valid': len(sequence) >= 12})
        return instances

    # 预取所有窗口，供负例采样
    token_windows = {}  # cid → [window_token_ids]
    cid_list = []
    for cid in cids:
        edges = cid_edges.get(cid, [])
        if len(edges) == 0:
            token_windows[cid] = []
            cid_list.append(cid)
            continue
        windows = [edges[i:i + window] for i in range(0, len(edges), window)]
        token_windows[cid] = [vocab.edges_to_ids(w, max_edges=window, keep=keep) for w in windows]
        cid_list.append(cid)

    negative_cids = [cid for cid in cid_list if token_windows[cid]]

    for cid in cid_list:
        win_ids = token_windows[cid]
        if len(win_ids) == 0:
            continue
        elif len(win_ids) >= 2:
            for j in range(len(win_ids) - 1):
                A = win_ids[j]
                B = win_ids[j + 1]
                flag = 0
                if rng.random() < nsp_prob and negative_cids:
                    other = rng.choice(negative_cids)
                    while other == cid and len(negative_cids) > 1:
                        other = rng.choice(negative_cids)
                    B = rng.choice(token_windows[other])
                    flag = 1
                instances.append((cid, A, B, flag))
        else:
            # 单窗口：A=前半 B=后半（正例）
            A_ids = win_ids[0]
            k = len(A_ids) // 2
            if k < 1:
                continue
            instances.append((cid, A_ids[:k], A_ids[k:], 0))
    return instances


# ============================================================
# 3. 对比损失（逐字移植 contrastive.py::label_smoothed_nll_loss）
# ============================================================

def label_smoothed_nll_loss(contrastive_scores, contrastive_labels, eps=0.0):
    bsz, seqlen, _ = contrastive_scores.size()
    logprobs = F.log_softmax(contrastive_scores.view(-1, seqlen), dim=-1)
    gold = torch.arange(seqlen).view(-1,)
    gold = gold.expand(bsz, seqlen).contiguous().view(-1)
    if contrastive_scores.is_cuda:
        gold = gold.cuda(contrastive_scores.get_device())
    loss = -logprobs.gather(dim=-1, index=gold.unsqueeze(1)).squeeze(1)
    loss = loss.view(bsz, seqlen) * contrastive_labels
    loss = torch.sum(loss) / contrastive_labels.sum()
    _, pred = torch.max(logprobs, -1)
    correct_num = torch.eq(gold, pred).float().view(bsz, seqlen)
    correct_num = torch.sum(correct_num * contrastive_labels)
    total_num = contrastive_labels.sum()
    return loss, correct_num, total_num


# ============================================================
# 4. LMAE4Eth Contrastive Pretraining（from-scratch BERT）
# ============================================================

class LMAE4EthContrastivePretraining(nn.Module):
    """
    LMAE4Eth `BERTContrastivePretraining` 的结构移植：
      model=BertForPreTraining（bert + cls）、anchor_bert=BertModel（冻结 EMA）、
      MLM + NSP + Token-aware 对比。权重 from-scratch（无预训练 BERT 可用）。
    """

    def __init__(self, vocab_size, d_model=256, num_layers=6, num_heads=8,
                 ffn=1024, max_seq_len=128, sim='cosine', temperature=0.01,
                 momentum=0.996, dropout=0.1, out_dim=128,
                 projection_loss_weight=0.1, projection_temperature=0.1,
                 semantic_amount_mode='legacy', amount_mlp_hidden=64,
                 amount_loss_weight=0.1, amount_mask_rate=0.15,
                 vocab_mapping=None):
        super().__init__()
        if semantic_amount_mode not in ('legacy', 'disabled', 'continuous_zscore'):
            raise ValueError('unsupported semantic amount mode')
        self.semantic_amount_mode = semantic_amount_mode
        self.amount_loss_weight = amount_loss_weight if semantic_amount_mode == 'continuous_zscore' else 0.
        self.amount_mask_rate = amount_mask_rate
        self.amount_mlp_hidden = amount_mlp_hidden
        self.vocab_mapping = dict(vocab_mapping or {})
        if not 0 <= amount_mask_rate <= 1 or amount_loss_weight < 0 or amount_mlp_hidden < 1:
            raise ValueError('invalid amount training configuration')
        if semantic_amount_mode != 'legacy' and not all(
                marker in self.vocab_mapping for marker in ('[AMT_CONT]', '[AMT_MISSING]', '[AMT_DISABLED]')):
            raise ValueError('new semantic architecture requires explicit vocabulary mapping')
        self.embed_dim = d_model
        self.max_seq_len = max_seq_len
        self.sim = sim
        self.temperature = temperature
        self.momentum = momentum
        if projection_loss_weight <= 0 or projection_temperature <= 0:
            raise ValueError('projection loss weight and temperature must be positive')
        self.projection_loss_weight = projection_loss_weight
        self.projection_temperature = projection_temperature

        self.cfg = BertConfig(
            vocab_size=vocab_size, hidden_size=d_model,
            num_hidden_layers=num_layers, num_attention_heads=num_heads,
            intermediate_size=ffn, max_position_embeddings=max_seq_len,
            hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout,
        )
        self.model = BertForPreTraining(self.cfg)
        self.bert = self.model.bert
        self.cls = self.model.cls

        # 冻结 EMA anchor（LMAE4Eth 用预训练 anchor；从零训用 EMA 拷贝，语义同 edcoder encoder_ema）
        self.anchor_bert = BertModel(self.cfg)
        self.anchor_bert.load_state_dict(self.bert.state_dict())
        for p in self.anchor_bert.parameters():
            p.requires_grad = False
            p.detach_()

        # 输出投影：pooled(256) → 128（= HIDDEN_DIM，phase3 融合层不动）
        self.out_proj = nn.Linear(d_model, out_dim)
        self.anchor_out_proj = nn.Linear(d_model, out_dim)
        self.anchor_out_proj.load_state_dict(self.out_proj.state_dict())
        self.anchor_out_proj.requires_grad_(False)
        self.anchor_bert.eval()
        self.anchor_out_proj.eval()
        if semantic_amount_mode != 'legacy':
            self.amount_encoder = nn.Sequential(nn.Linear(1, amount_mlp_hidden), nn.GELU(),
                                                nn.Linear(amount_mlp_hidden, d_model))
            self.anchor_amount_encoder = copy.deepcopy(self.amount_encoder)
            self.anchor_amount_encoder.requires_grad_(False)
            self.anchor_amount_encoder.eval()
            self.amount_regressor = nn.Linear(d_model, 1)

    def train(self, mode=True):
        """EMA teachers stay deterministic while the online network trains."""
        super().train(mode)
        self.anchor_bert.eval()
        self.anchor_out_proj.eval()
        if hasattr(self, 'anchor_amount_encoder'):
            self.anchor_amount_encoder.eval()
        return self

    @torch.no_grad()
    def _ema_update(self):
        m = self.momentum
        for p_s, p_t in zip(self.bert.parameters(), self.anchor_bert.parameters()):
            p_t.data.mul_(m).add_((1 - m) * p_s.data)
        for p_s, p_t in zip(self.out_proj.parameters(), self.anchor_out_proj.parameters()):
            p_t.data.mul_(m).add_((1 - m) * p_s.data)
        if hasattr(self, 'anchor_amount_encoder'):
            for source, target in zip(self.amount_encoder.parameters(), self.anchor_amount_encoder.parameters()):
                target.mul_(m).add_((1 - m) * source)

    def amount_inputs_embeds(self, input_ids, amount_values, amount_position_mask,
                            amount_known, amount_selected=None, *, anchor=False):
        """Only word embeddings here: BERT adds position/segment exactly once."""
        if self.semantic_amount_mode == 'legacy':
            raise ValueError('legacy architecture has no continuous amount inputs')
        fields = (amount_values, amount_position_mask, amount_known)
        if input_ids.dtype != torch.long or input_ids.ndim != 2:
            raise ValueError('input_ids must be [B,L] long')
        if any(value is None or value.shape != input_ids.shape or value.device != input_ids.device
               for value in fields):
            raise ValueError('amount fields must share input shape and device')
        if not amount_values.is_floating_point() or amount_position_mask.dtype != torch.bool or amount_known.dtype != torch.bool:
            raise ValueError('invalid amount field dtypes')
        if not torch.isfinite(amount_values).all() or (amount_known & ~amount_position_mask).any():
            raise ValueError('nonfinite or invalid semantic amount fields')
        if ((input_ids.eq(FieldVocab.PAD) | input_ids.eq(FieldVocab.CLS) | input_ids.eq(FieldVocab.SEP))
                & amount_position_mask).any():
            raise ValueError('special/padding tokens cannot be amount positions')
        if (amount_values[~amount_position_mask] != 0).any():
            raise ValueError('non-amount positions must have zero values')
        if amount_selected is None:
            amount_selected = torch.zeros_like(amount_position_mask)
        if (amount_selected.dtype != torch.bool or amount_selected.shape != input_ids.shape
                or amount_selected.device != input_ids.device
                or (amount_selected & ~(amount_position_mask & amount_known)).any()):
            raise ValueError('invalid amount_selected')
        ids = input_ids.clone()
        if self.semantic_amount_mode == 'disabled':
            ids[amount_position_mask] = self.vocab_mapping['[AMT_DISABLED]']
        else:
            ids[amount_position_mask & amount_known] = self.vocab_mapping['[AMT_CONT]']
            ids[amount_position_mask & ~amount_known] = self.vocab_mapping['[AMT_MISSING]']
            ids[amount_selected] = FieldVocab.MASK
        bert = self.anchor_bert if anchor else self.bert
        base = bert.embeddings.word_embeddings(ids)
        if self.semantic_amount_mode == 'disabled':
            return base
        visible = amount_position_mask & amount_known & ~amount_selected
        safe = torch.where(visible, amount_values, torch.zeros_like(amount_values)).to(base.dtype)
        encoder = self.anchor_amount_encoder if anchor else self.amount_encoder
        numeric = encoder(safe.unsqueeze(-1)) * visible.unsqueeze(-1).to(base.dtype)
        if not torch.isfinite(numeric).all():
            raise ValueError('nonfinite projected semantic amount')
        return base + numeric

    def _bert_inputs(self, input_ids, amount_values=None, amount_position_mask=None,
                     amount_known=None, amount_selected=None, *, anchor=False):
        if self.semantic_amount_mode == 'legacy':
            if any(value is not None for value in (amount_values, amount_position_mask, amount_known, amount_selected)):
                raise ValueError('continuous fields cannot be passed to legacy model')
            return {'input_ids': input_ids}
        return {'inputs_embeds': self.amount_inputs_embeds(
            input_ids, amount_values, amount_position_mask, amount_known, amount_selected, anchor=anchor)}

    def compute_projection_loss(self, masked_pooled, truth_pooled):
        """Train the exact 128-D feature exported to Phase3.

        Distinct sequence views in a batch provide negatives. A singleton batch
        uses teacher alignment because its one-class InfoNCE has zero gradient.
        """
        online = F.normalize(self.out_proj(masked_pooled), dim=-1)
        with torch.no_grad():
            target = F.normalize(self.anchor_out_proj(truth_pooled), dim=-1)
        alignment = F.mse_loss(online, target)
        if len(online) == 1:
            return alignment
        scores = online @ target.t() / self.projection_temperature
        labels = torch.arange(len(online), device=online.device)
        return 0.5 * (F.cross_entropy(scores, labels) +
                      F.cross_entropy(scores.t(), labels)) + alignment

    # ---- representations（对照 contrastive.py compute_representations / anchor）----
    def compute_representations(self, input_ids, token_type_ids, attention_mask, **amount_fields):
        bsz, seqlen = input_ids.size()
        outputs = self.bert(**self._bert_inputs(input_ids, **amount_fields), token_type_ids=token_type_ids,
                            attention_mask=attention_mask)
        rep, pooled_output = outputs[0], outputs[1]
        rep = rep.view(bsz, seqlen, self.embed_dim)
        logits, sen_relation_scores = self.cls(rep, pooled_output)  # (B,L,V), (B,2)
        return rep, logits, sen_relation_scores, pooled_output

    @torch.no_grad()
    def compute_anchor_representations(self, input_ids, token_type_ids, attention_mask, **amount_fields):
        bsz, seqlen = input_ids.size()
        outputs = self.anchor_bert(**self._bert_inputs(input_ids, anchor=True, **amount_fields), token_type_ids=token_type_ids,
                                   attention_mask=attention_mask)
        rep, pooled_output = outputs[0], outputs[1]
        rep = rep.view(bsz, seqlen, self.embed_dim)
        return rep, pooled_output

    # ---- MLM loss（对照 contrastive.py compute_mlm_loss）----
    def compute_mlm_loss(self, truth, msk, logits):
        msk_token_num = torch.sum(msk).float().item()
        if msk_token_num == 0:
            return torch.tensor(0.0, device=logits.device), 0.0
        y_mlm = logits.masked_select(msk.unsqueeze(-1)).view(-1, self.cfg.vocab_size)
        gold = truth.masked_select(msk)
        log_probs_mlm = torch.log_softmax(y_mlm, -1)
        mlm_loss = F.nll_loss(log_probs_mlm, gold, reduction='mean')
        _, pred_mlm = log_probs_mlm.max(-1)
        mlm_correct_num = torch.eq(pred_mlm, gold).float().sum().item()
        return mlm_loss, mlm_correct_num

    def forward(self, truth, inp, seg, msk, attn_msk, contrastive_labels, nxt_snt_flag):
        """
        对照 contrastive.py::forward：
          masked_rep = bert(inp) → rep + logits(MLM) + srs(NSP)
          mlm_loss    = nll(truth, msk, logits)
          truth_rep   = anchor_bert(truth)   （冻结，未掩码）
          ct_scores   = cosine_sim(masked_rep, truth_rep)/T   (B,L,L)
          ct_loss     = label_smoothed_nll_loss(ct_scores, contrastive_labels)
          nsp_loss    = CE(srs, nxt_snt_flag)
        Returns:
          (mlm_loss, nsp_loss, contrastive_loss, mlm_correct, tot_tokens)
        """
        bsz, seqlen = truth.size()
        masked_rep, logits, sen_relation_scores, masked_pooled = \
            self.compute_representations(inp, seg, attn_msk)

        mlm_loss, mlm_correct_num = self.compute_mlm_loss(truth, msk, logits)

        truth_rep, truth_pooled = self.compute_anchor_representations(truth, seg, attn_msk)
        if self.sim == 'dot_product':
            contrastive_scores = torch.matmul(masked_rep, truth_rep.transpose(1, 2))
        else:  # cosine
            masked_rep = F.normalize(masked_rep, dim=2)
            truth_rep = F.normalize(truth_rep, dim=2)
            contrastive_scores = torch.matmul(masked_rep, truth_rep.transpose(1, 2)) / self.temperature
        # Padding must not act as a negative key in the token objective.
        contrastive_scores = contrastive_scores.masked_fill(
            ~attn_msk.bool().unsqueeze(1), torch.finfo(contrastive_scores.dtype).min)

        n_contrastive = contrastive_labels.sum()
        if n_contrastive > 0:
            contrastive_loss, _, _ = label_smoothed_nll_loss(contrastive_scores, contrastive_labels)
        else:
            contrastive_loss = torch.tensor(0.0, device=contrastive_scores.device)

        projected_loss = self.compute_projection_loss(masked_pooled, truth_pooled)
        contrastive_loss = contrastive_loss + self.projection_loss_weight * projected_loss

        nsp_loss = F.cross_entropy(
            sen_relation_scores.view(-1, 2), nxt_snt_flag.view(-1).long())

        return mlm_loss, nsp_loss, contrastive_loss, mlm_correct_num, int(msk.sum().item())

    def forward_continuous(self, batch, vocab, mask_rate=0.15, *, amount_selected=None):
        if self.semantic_amount_mode == 'legacy' or vocab.amount_policy != self.semantic_amount_mode:
            raise ValueError('continuous forward mode/vocabulary mismatch')
        online, discrete, selected = make_continuous_mlm_input(
            batch, vocab, mask_rate, self.amount_mask_rate, amount_selected)
        fields = lambda row: {name: row[name] for name in
                             ('amount_values', 'amount_position_mask', 'amount_known')}
        hidden, logits, scores, pooled = self.compute_representations(
            online['input_ids'], online['token_type_ids'], online['attention_mask'],
            **fields(online), amount_selected=selected)
        teacher, truth_pooled = self.compute_anchor_representations(
            batch['input_ids'], batch['token_type_ids'], batch['attention_mask'], **fields(batch))
        zero = hidden.sum() * 0.
        mlm, correct = self.compute_mlm_loss(batch['input_ids'], discrete, logits)
        valid = batch['nsp_valid']
        nsp = F.cross_entropy(scores[valid], batch['nsp_flags'][valid]) if valid.any() else zero
        token_scores = (hidden @ teacher.transpose(1, 2) if self.sim == 'dot_product' else
                        F.normalize(hidden, dim=-1) @ F.normalize(teacher, dim=-1).transpose(1, 2) / self.temperature)
        token_scores = token_scores.masked_fill(~batch['attention_mask'].bool().unsqueeze(1),
                                                 torch.finfo(token_scores.dtype).min)
        queries = (discrete | selected) & batch['attention_mask'].bool()
        contrastive = label_smoothed_nll_loss(token_scores, queries.float())[0] if queries.any() else zero
        projection = self.compute_projection_loss(pooled, truth_pooled)
        prediction = self.amount_regressor(hidden).squeeze(-1)
        count = int(selected.sum())
        if count:
            amount = F.smooth_l1_loss(prediction[selected], batch['amount_values'][selected], beta=1.)
            mae = (prediction[selected] - batch['amount_values'][selected]).abs().mean().detach()
        else:
            amount, mae = zero, None
        total = mlm + nsp + contrastive + self.projection_loss_weight * projection + self.amount_loss_weight * amount
        return {'loss': total, 'mlm': mlm, 'nsp': nsp, 'contrastive': contrastive,
                'projection': projection, 'amount': amount, 'amount_count': count,
                'amount_mae': mae, 'mlm_correct': correct, 'mlm_count': int(discrete.sum()),
                'amount_predictions': prediction, 'amount_selected': selected,
                'online_hidden': hidden}

    @torch.no_grad()
    def encode_sequence(self, input_ids, token_type_ids, attention_mask, **amount_fields):
        """Inference：未掩码全序列 → pooled → Linear → (B, out_dim=128)。"""
        modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            outputs = self.bert(**self._bert_inputs(input_ids, **amount_fields), token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
            return self.out_proj(outputs[1])
        finally:
            # Restore every submodule, including deliberately frozen teachers.
            for module, training in modes:
                module.training = training

    def history_encoding_config(self, window_size=16):
        """Describe the export algorithm independently of inference batch size."""
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1:
            raise ValueError('window_size must be a positive integer')
        # Structured collation uses CLS/SEP/SEP; legacy inference uses CLS/SEP.
        special_tokens = 2 if self.semantic_amount_mode == 'legacy' else 3
        capacity = (self.max_seq_len - special_tokens) // 6
        if capacity < 1:
            raise ValueError('max_seq_len must fit one complete event')
        effective_size = min(window_size, capacity)
        return {'version': 'all_visible_windows_v1', 'mode': 'all_visible_windows',
                'window_size': effective_size, 'requested_window_size': window_size,
                'stride': effective_size, 'aggregation': 'event_count_weighted_mean',
                'time_policy': 'strict_before_cutoff',
                'window_boundary_dt': 'previous_visible_event', 'empty_history': 'zeros'}

    @torch.no_grad()
    def encode_history(self, edges, vocab, window_batch_size=4, *, cutoff=None, window_size=16):
        """Encode every event before cutoff, with bounded BERT inference memory.

        Nonoverlapping chronological windows are pooled with their actual event
        counts as weights. A short tail is included once. Batch size limits only
        simultaneously encoded windows, never the amount of visible history.
        """
        config = self.history_encoding_config(window_size)
        if (isinstance(window_batch_size, bool) or not isinstance(window_batch_size, int)
                or window_batch_size < 1):
            raise ValueError('window_batch_size must be a positive integer')
        device = self.out_proj.weight.device
        if cutoff is not None:
            if not math.isfinite(float(cutoff)):
                raise ValueError('cutoff must be finite')
            cutoff = float(cutoff)
        visible = []
        for edge in edges:
            timestamp = float(edge['timestamp'])
            if not math.isfinite(timestamp):
                raise ValueError('history timestamp must be finite')
            if cutoff is None or timestamp < cutoff:
                visible.append(edge)
        edges = visible
        edges = sorted(edges, key=lambda edge: float(edge['timestamp']))
        if not edges:
            return self.out_proj.weight.new_zeros(self.out_proj.out_features)
        if self.semantic_amount_mode != 'legacy':
            if vocab.amount_policy != self.semantic_amount_mode:
                raise ValueError('inference amount mode/vocabulary mismatch')
            identities = [str(edge['source_row_id']) for edge in edges]
            if any(not identity for identity in identities) or len(set(identities)) != len(identities):
                raise ValueError('missing or repeated source_row_id in semantic history')
        size = config['window_size']
        # Accumulate in float32 even if the encoder later uses reduced precision.
        total = torch.zeros(self.out_proj.out_features, device=device, dtype=torch.float32)
        for batch_start in range(0, len(edges), size * window_batch_size):
            starts = range(batch_start, min(len(edges), batch_start + size * window_batch_size), size)
            rows, counts = [], []
            for start in starts:
                window = edges[start:start + size]
                previous = float(edges[start - 1]['timestamp']) if start else None
                counts.append(len(window))
                if self.semantic_amount_mode == 'legacy':
                    tokens = [vocab.CLS] + vocab.edges_to_ids(
                        window, max_edges=size, keep='first', prev_ts=previous) + [vocab.SEP]
                    rows.append(tokens + [vocab.PAD] * (self.max_seq_len - len(tokens)))
                else:
                    sequence = vocab.edges_to_sequence(window, max_edges=size, keep='first', prev_ts=previous)
                    rows.append({'a': sequence, 'b': StructuredSequence([], [], [], [], []),
                                 'flag': 0, 'nsp_valid': False})
            if self.semantic_amount_mode == 'legacy':
                ids = torch.tensor(rows, dtype=torch.long, device=device)
                vectors = self.encode_sequence(ids, torch.zeros_like(ids), ids.ne(vocab.PAD).float())
            else:
                batch = collate_structured(rows, vocab, self.max_seq_len)
                batch = {name: value.to(device) if torch.is_tensor(value) else value for name, value in batch.items()}
                vectors = self.encode_sequence(
                    batch['input_ids'], batch['token_type_ids'], batch['attention_mask'],
                    **{name: batch[name] for name in ('amount_values', 'amount_position_mask', 'amount_known')})
            # Sequential accumulation keeps the reduction order independent of batching.
            for vector, count in zip(vectors, counts):
                total.add_(vector.float(), alpha=count)
        return total / len(edges)


# ============================================================
# Factory
# ============================================================

def build_semantic_lmae(vocab_size, **kw):
    """构建 LMAE4Eth 风格语义编码器（from-scratch BERT）。"""
    print('[semantic_lmae] LMAE4Eth-style BERT (from-scratch, no pretrained weights) '
          'vocab=%d hidden=%d layers=%d' % (
              vocab_size, kw.get('d_model', 256), kw.get('num_layers', 6)))
    return LMAE4EthContrastivePretraining(
        vocab_size=vocab_size,
        d_model=kw.get('d_model', 256),
        num_layers=kw.get('num_layers', 6),
        num_heads=kw.get('num_heads', 8),
        ffn=kw.get('ffn', 1024),
        max_seq_len=kw.get('max_seq_len', 128),
        temperature=kw.get('temperature', 0.01),
        momentum=kw.get('momentum', 0.996),
        dropout=kw.get('dropout', 0.1),
        out_dim=kw.get('out_dim', 128),
        projection_loss_weight=kw.get('projection_loss_weight', 0.1),
        projection_temperature=kw.get('projection_temperature', 0.1),
        semantic_amount_mode=kw.get('semantic_amount_mode', 'legacy'),
        amount_mlp_hidden=kw.get('amount_mlp_hidden', 64),
        amount_loss_weight=kw.get('amount_loss_weight', 0.1),
        amount_mask_rate=kw.get('amount_mask_rate', 0.15),
        vocab_mapping=kw.get('vocab_mapping'),
    )


def semantic_checkpoint(sem, vocab, **metadata):
    """Complete trainable and EMA state, with explicit input protocol/config."""
    if sem.semantic_amount_mode == 'legacy' or vocab.amount_policy != sem.semantic_amount_mode:
        raise ValueError('continuous checkpoint requires matching new model and vocabulary')
    return {'schema_version': 'continuous_amount_semantic_v2',
            'semantic_amount_mode': sem.semantic_amount_mode,
            'state_dict': copy.deepcopy(sem.state_dict()), 'cfg': sem.cfg.to_dict(),
            'vocab': dict(vocab.token_to_id), 'vocab_version': 'structured_amount_v2',
            'model_options': {'vocab_size': sem.cfg.vocab_size, 'd_model': sem.cfg.hidden_size,
                              'num_layers': sem.cfg.num_hidden_layers, 'num_heads': sem.cfg.num_attention_heads,
                              'ffn': sem.cfg.intermediate_size, 'max_seq_len': sem.max_seq_len,
                              'dropout': sem.cfg.hidden_dropout_prob, 'out_dim': sem.out_proj.out_features,
                              'sim': sem.sim, 'temperature': sem.temperature, 'momentum': sem.momentum,
                              'projection_loss_weight': sem.projection_loss_weight,
                              'projection_temperature': sem.projection_temperature,
                              'semantic_amount_mode': sem.semantic_amount_mode,
                              'amount_mlp_hidden': sem.amount_mlp_hidden,
                              'amount_loss_weight': sem.amount_loss_weight,
                              'amount_mask_rate': sem.amount_mask_rate, 'vocab_mapping': dict(vocab.token_to_id)},
            **metadata}


def load_semantic_checkpoint(checkpoint, *, expected_mode, vocab, expected_amount_metadata):
    from bhtg.amount_protocol import validate_amount_metadata
    if checkpoint.get('schema_version') != 'continuous_amount_semantic_v2':
        raise RuntimeError('expected complete continuous amount semantic checkpoint')
    if not isinstance(expected_amount_metadata, dict) or not expected_amount_metadata:
        raise RuntimeError('explicit amount source/mapping metadata is required')
    if checkpoint.get('semantic_amount_mode') != expected_mode or vocab.amount_policy != expected_mode:
        raise RuntimeError('semantic checkpoint amount mode mismatch')
    if checkpoint.get('vocab') != vocab.token_to_id:
        raise RuntimeError('semantic checkpoint vocabulary mismatch')
    validate_amount_metadata(checkpoint.get('amount_experiment'), expected_amount_metadata)
    options = checkpoint['model_options']
    if options.get('semantic_amount_mode') != expected_mode or options.get('vocab_mapping') != vocab.token_to_id:
        raise RuntimeError('semantic checkpoint model configuration mismatch')
    for key in ('amount_mask_rate', 'amount_loss_weight', 'amount_mlp_hidden'):
        if key in expected_amount_metadata and options.get(key) != expected_amount_metadata[key]:
            raise RuntimeError('semantic checkpoint configuration mismatch: ' + key)
    model = LMAE4EthContrastivePretraining(**options)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model
