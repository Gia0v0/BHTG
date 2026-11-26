"""
BHTG 图编码器

封装 TemporalHGT → 提取各类型节点的最终 embedding
特别是从 contract 节点中提取 token-level embedding 用于下游分类
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hgt_layer import TemporalHGT


GRAPH_ENCODER_PROTOCOL = 'per_node_residual_complete_receptive_field_v2'
ENCODER_PROTOCOL = GRAPH_ENCODER_PROTOCOL


class GraphEncoder(nn.Module):
    """
    图编码器：将异构图映射到节点 embedding

    Args:
        node_feat_dims: {ntype: input_feature_dim} 各类节点的初始特征维度
        hidden_dim:     隐藏层维度
        out_dim:        最终输出维度
        num_heads:      注意力头数
        num_layers:     HGT 层数
        etypes:         meta-relation 列表
        lambda_init:    时间衰减初始值
        dropout:        dropout rate
    """

    def __init__(self, node_feat_dims, hidden_dim=128, out_dim=128,
                 num_heads=8, num_layers=2, etypes=None,
                 lambda_init=None, dropout=0.3, homo_ablation=False, edge_feature_fields=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.homo_ablation = homo_ablation
        if int(num_layers) != num_layers or num_layers < 1:
            raise ValueError('num_layers must be a positive integer')
        self.num_layers = int(num_layers)
        # Strict loading also rejects old graph weights whose tensor shapes
        # coincide but whose per-relation residual semantics were different.
        from bhtg.model_schema_v3 import POOL_EDGE_FEATURE_FIELDS
        from bhtg.amount_edge_features import GRAPH_AMOUNT_EDGE_FIELDS
        if edge_feature_fields is None:
            edge_feature_fields = (POOL_EDGE_FEATURE_FIELDS
                                   if any(et[1].startswith('lp_') for et in (etypes or ())) else ())
        self.edge_feature_fields = tuple(edge_feature_fields)
        if self.edge_feature_fields not in ((), POOL_EDGE_FEATURE_FIELDS, GRAPH_AMOUNT_EDGE_FIELDS):
            raise ValueError('unsupported graph edge feature contract')
        if (self.edge_feature_fields == GRAPH_AMOUNT_EDGE_FIELDS and
                any(canonical[1] not in ('transfer', 'mint', 'burn', 'create') for canonical in (etypes or ()))):
            raise ValueError('continuous graph amounts support only the four-actions relations')
        self.architecture_version = (4 if self.edge_feature_fields == GRAPH_AMOUNT_EDGE_FIELDS else
                                     3 if self.edge_feature_fields else 2)
        self.register_buffer('_graph_architecture_version', torch.tensor(self.architecture_version, dtype=torch.long))

        # 节点特征投影: 各类型输入维度 → hidden_dim
        # homo_ablation=True → 全类型共享同一投影 (model-level homogenization 消融,
        #   等价 single-type GAT; 只注册 node_proj['all'] 一个模块)
        self.node_proj = nn.ModuleDict()
        if homo_ablation:
            _in_dim = next((d for d in node_feat_dims.values() if d > 0), hidden_dim)
            self.node_proj['all'] = nn.Sequential(
                nn.Linear(_in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            for ntype, in_dim in node_feat_dims.items():
                if in_dim > 0:
                    self.node_proj[ntype] = nn.Sequential(
                        nn.Linear(in_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                else:
                    self.node_proj[ntype] = None

        # 时序 HGT
        self.hgt = TemporalHGT(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            etypes=etypes,
            lambda_init=lambda_init,
            dropout=dropout,
            homo_ablation=homo_ablation,
            node_types=list(node_feat_dims), edge_feature_fields=self.edge_feature_fields,
        )

        # Output normalization to prevent gradient explosion
        # HGT outputs can reach ±400; LayerNorm keeps fusion training stable
        self.output_norm = nn.ModuleDict()
        for ntype in (['all'] if homo_ablation else list(node_feat_dims)):
            self.output_norm[ntype] = nn.LayerNorm(out_dim)

        # 可选: Memory 机制 (继承自 RugScreener)
        self.use_memory = False

    def _key(self, ntype):
        """homo_ablation=True → 全部节点类型共享同一模块 (统一映射到 'all' key)"""
        return 'all' if self.homo_ablation else ntype

    @property
    def required_sampling_hops(self):
        """Sample the complete incoming receptive field of all HGT layers."""
        return self.num_layers

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # ``strict=False`` only relaxes missing/unexpected keys; it must never
        # make old residual semantics compatible. Validate at this recursive
        # hook so loading an encoder nested in another module is equally safe.
        version_key = prefix + '_graph_architecture_version'
        version = state_dict.get(version_key)
        valid_version = (
            isinstance(version, torch.Tensor) and version.shape == torch.Size([])
            and not version.is_floating_point() and not version.is_complex()
            and version.dtype != torch.bool and int(version.detach().cpu().item()) == self.architecture_version
        )
        if not valid_version:
            error_msgs.append(
                '%s requires graph architecture version %s (%s); '
                'old or unversioned graph weights must be retrained' %
                (version_key, self.architecture_version, GRAPH_ENCODER_PROTOCOL))
            # Avoid overwriting the version buffer before the eventual error,
            # which matters if the caller catches the load failure.
            state_dict = dict(state_dict)
            state_dict[version_key] = self._graph_architecture_version
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def inject_memory(self, memory_module, gid_to_lid=None):
        """
        注入 Memory 模块 (必须在 load_state_dict 之后调用，避免 strict 加载冲突)

        Args:
            memory_module: bhtg.memory.HeteroTemporalMemory — 含 memory 表 (按 full-graph local id 索引)
            gid_to_lid: {ntype: {global_id: local_id}} — 由 {ntype}_gids.pt 构建，
                        用于把子图节点的 global id 转成 full-graph local id
        """
        self.memory = memory_module
        self.use_memory = True
        # memory 投影 (phase3 可训练，让 HGT 自适应常量记忆状态)
        # 小方差初始化: 初始接近 baseline (记忆通道近乎无贡献)，随训练增强
        # 新建 nn.Linear 默认在 CPU，需显式建到本编码器所在 device (memory 表保持 CPU)
        dev = self.output_norm[next(iter(self.output_norm))].weight.device
        self.memory_proj = nn.ModuleDict()
        for ntype in self.node_proj.keys():
            self.memory_proj[ntype] = nn.Linear(self.hidden_dim, self.hidden_dim).to(dev)
            nn.init.normal_(self.memory_proj[ntype].weight, std=0.01)
            nn.init.zeros_(self.memory_proj[ntype].bias)
        # global→local 映射 (子图节点 memory 查询用)
        self.gid_to_lid = gid_to_lid or {}
        # 节点类型名 → memory 表整数索引 (memory.node_types = NODE_TYPES 顺序)
        self.ntype_to_idx = {nt: i for i, nt in enumerate(self.memory.node_types)}

    def _subgraph_local_ids(self, g, ntype):
        """
        返回子图/图中 ntype 节点对应的 full-graph local id (memory 表按它索引)

        - 子图 (sample_token_subgraph): _all_gids[ntype] 就是 full-graph LOCAL id
          (sampler 的 active 从 seed 的 local id 出发，边 sub.edges() 也是 local id；
          注释里的 "global" 是误导性命名，实测 _gid_maps 值域为 local→subgraph-local)
          → 直接返回，无需 gid_to_lid 转换
        - 全图 forward (evaluate 等): g.nodes(ntype) 即 full-graph local id
        """
        all_gids = getattr(g, '_all_gids', None)
        if all_gids is not None:
            gids = all_gids.get(ntype)
            if gids is None or len(gids) == 0:
                return torch.empty(0, dtype=torch.long)
            return torch.as_tensor(gids, dtype=torch.long)
        return g.nodes(ntype)

    def forward(self, g, current_time=None, return_all=False):
        """
        Args:
            g:            DGLHeteroGraph
            current_time: float — 观测截断时间
            return_all:   bool — 是否返回所有节点类型的 embedding

        Returns:
            h_dict:         {ntype: embeddings}
            token_emb:      (num_contract, out_dim) token (代币) 节点 embedding
        """
        if hasattr(g, '_sampler_protocol') or hasattr(g, '_sampling_hops'):
            sampled_hops = getattr(g, '_sampling_hops', None)
            if sampled_hops is None or sampled_hops < self.required_sampling_hops:
                raise ValueError(
                    'Incomplete sampled receptive field: n_hop=%s, encoder layers=%d; '
                    'sample with n_hop=encoder.required_sampling_hops' %
                    (sampled_hops, self.required_sampling_hops))
        # 1. 投影各类节点特征到统一维度
        h_dict = {}
        for ntype in g.ntypes:
            feat = g.nodes[ntype].data.get('feat', None)
            if feat is not None and feat.numel() > 0:
                proj = self.node_proj[self._key(ntype)]
                if proj is not None:
                    h_dict[ntype] = proj(feat)
                else:
                    h_dict[ntype] = feat
            else:
                h_dict[ntype] = torch.empty(0, self.hidden_dim, device=feat.device if feat is not None else 'cpu')

        # 2. 可选: 注入 Memory
        if self.use_memory and hasattr(self, 'memory'):
            for ntype in h_dict:
                locs = self._subgraph_local_ids(g, ntype)
                if locs.numel() == 0:
                    continue
                # Legacy experiment: query the shared full-graph Memory table.
                # Do not import reviewfix_v3 per-endpoint memory rows.
                mem = self.memory.get_memory(self.ntype_to_idx[ntype], locs)
                if mem is None:
                    continue
                mem = mem.to(h_dict[ntype].device)
                h_dict[ntype] = h_dict[ntype] + self.memory_proj[self._key(ntype)](mem)

        # 3. HGT 消息传递
        h_dict = self.hgt(g, h_dict, current_time)

        # 3.5 Normalize outputs to prevent gradient explosion
        # HGT outputs can reach ±400; LayerNorm stabilizes for fusion training
        for ntype in h_dict:
            if h_dict[ntype].numel() > 0:
                h_dict[ntype] = self.output_norm[self._key(ntype)](h_dict[ntype])

        # 4. 输出
        # token embedding = contract 节点的最终表示
        token_emb = h_dict.get('contract', torch.empty(0, self.out_dim))

        if return_all:
            return h_dict, token_emb
        return token_emb

    def get_lambda_values(self):
        """获取时间衰减参数值"""
        return self.hgt.get_lambda_values()

    def get_lambda_params(self):
        """返回所有可学习 λ 参数的列表（用于独立优化器组）"""
        params = []
        for layer in self.hgt.layers:
            for name, param in layer.lambda_decay.items():
                if param.requires_grad:
                    params.append(param)
        return params
