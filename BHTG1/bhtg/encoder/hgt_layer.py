"""
TemporalHGTLayer — 时间感知的异构图层

用 DGL 的 HeteroGraphConv 包装 TimeDecayGATConv，
为每种 meta-relation 分配独立的卷积 + 可学习的时间衰减参数 λ_r。

实现 BHTG 方案 D.1 节的完整公式:
    h_t^(l+1) = Aggregate_{∀s∈N(t)} softmax(QK^T/√d × exp(-λ_r·Δt)) · V(s)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import math

from .gat_conv import TimeDecayGATConv


class TemporalHGTLayer(nn.Module):
    """
    一层时间感知的 HGT

    Args:
        in_dim:      输入特征维度（所有节点类型统一）
        out_dim:     每头输出维度
        num_heads:   注意力头数
        etypes:      实际出现的边类型列表 (src_t, e_t, dst_t)
        lambda_init: 时间衰减参数初始值 {etype: float}
        dropout:     dropout rate
    """

    def __init__(self, in_dim, out_dim, num_heads, etypes,
                 lambda_init=None, dropout=0.3, homo_ablation=False,
                 node_types=None, edge_feature_fields=()):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.homo_ablation = homo_ablation
        self.edge_feature_fields = tuple(edge_feature_fields)

        # Store conv modules directly (bypass HeteroGraphConv for DGL 0.8 compat)
        # homo_ablation=True → 所有 meta-relation 共享同一 conv (model-level
        #   homogenization 消融; 时间注意力仍按 relation 分组 softmax, 只共享权重)
        self.conv_dict = nn.ModuleDict()
        if homo_ablation:
            self.conv_dict['all'] = TimeDecayGATConv(
                in_feats=in_dim,
                out_feats=out_dim,
                num_heads=num_heads,
                feat_drop=dropout,
                attn_drop=dropout,
                residual=False, edge_feature_fields=edge_feature_fields,
            )
        else:
            for etype in etypes:
                key = str(etype)  # convert tuple to string key
                self.conv_dict[key] = TimeDecayGATConv(
                    in_feats=in_dim,
                    out_feats=out_dim,
                    num_heads=num_heads,
                    feat_drop=dropout,
                    attn_drop=dropout,
                    residual=False, edge_feature_fields=edge_feature_fields,
                )

        # Manual dispatch (no HeteroGraphConv for DGL 0.8 compat)
        self.aggregate = 'sum'

        # A node keeps its state exactly once per layer. Adding this inside
        # every relation makes its state depend on which relations happen to
        # occur elsewhere in the batch, including ones with no edge to it.
        node_types = sorted(set(node_types or ()) | {
            ntype for canonical in etypes for ntype in (canonical[0], canonical[2])})
        self.node_residual = nn.ModuleDict()
        for key in (['all'] if homo_ablation else node_types):
            if in_dim == out_dim * num_heads:
                self.node_residual[key] = nn.Identity()
            else:
                projection = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                nn.init.xavier_normal_(projection.weight, gain=nn.init.calculate_gain('relu'))
                self.node_residual[key] = projection

        # 可学习的时间衰减参数 λ_r (每种边类型一个标量)
        # homo_ablation=True → 单一共享 λ (ParameterDict 里注册 '_shared' 一个,
        #   日志/结果仍按边名索引, 由 TemporalHGT.get_lambda_values 广播)
        self.lambda_decay = nn.ParameterDict()
        if lambda_init is None:
            lambda_init = {
                'transfer': 0.50, 'mint': 0.05,
                'burn': 0.01, 'create': 0.05,
            }
        if homo_ablation:
            self._homo_default = lambda_init.get(etypes[0][1], 0.5) if etypes else 0.5
            self.lambda_decay['_shared'] = nn.Parameter(
                torch.tensor(self._homo_default, dtype=torch.float)
            )
        else:
            for etype in etypes:
                etype_name = etype[1]  # 取边类型的名称
                init_val = lambda_init.get(etype_name, 0.5)
                self.lambda_decay[etype_name] = nn.Parameter(
                    torch.tensor(init_val, dtype=torch.float)
                )

    def compute_time_weights(self, g, current_time):
        """
        为每条边计算时间衰减权重 exp(-λ_r · Δt)

        Args:
            g: DGLHeteroGraph
            current_time: float — 当前观测时间

        Returns:
            time_weights: {etype: (E, 1) tensor}
        """
        time_weights = {}
        for srctype, etype, dsttype in g.canonical_etypes:
            if g.num_edges((srctype, etype, dsttype)) == 0:
                continue
            edge_data = g.edges[(srctype, etype, dsttype)].data
            timestamps = edge_data['timestamp']
            lam = self.lambda_decay['_shared' if self.homo_ablation else etype].clamp(min=0.001, max=1.0)
            # Legacy experiment used one shared observation cutoff, not
            # per-edge endpoint query times.
            delta_t_days = ((current_time - timestamps).clamp(min=0) / 86400.0).to(lam.dtype)
            time_weights[(srctype, etype, dsttype)] = torch.exp(-lam * delta_t_days)
        return time_weights

    def forward(self, g, h_dict, current_time=None):
        """
        Args:
            g:            DGLHeteroGraph
            h_dict:       {ntype: (N, in_dim)} 各类节点特征
            current_time: float — 当前观测截断时间

        Returns:
            h_dict: {ntype: (N, num_heads * out_dim)} 更新后的节点特征
        """
        # Explicit LP/amount edge contracts accept only historical graphs. The sampler enforces
        # this too, but direct calls must not assign future edges weight 1.
        if self.edge_feature_fields:
            if current_time is None or not math.isfinite(float(current_time)):
                raise ValueError('edge-feature graph requires an explicit finite observation cutoff')
            for canonical in g.canonical_etypes:
                if not g.num_edges(canonical):
                    continue
                timestamps = g.edges[canonical].data.get('timestamp')
                if (timestamps is None or timestamps.ndim != 1
                        or len(timestamps) != g.num_edges(canonical)
                        or not bool(torch.isfinite(timestamps).all())
                        or bool((timestamps >= float(current_time)).any())):
                    raise ValueError('edge-feature graph contains a nonhistorical edge at observation cutoff')
        # 计算时间衰减权重
        time_weights = None
        if current_time is not None:
            time_weights = self.compute_time_weights(g, current_time)

        # 为每个 meta-relation 计算特征投影
        # HeteroGraphConv 自动按类型调度对应 conv
        # 需要传入每个关系的时间衰减权重
        final_dict = {
            ntype: self.node_residual['all' if self.homo_ablation else ntype](features)
            for ntype, features in h_dict.items()
        }
        for srctype, etype, dsttype in g.canonical_etypes:
            if srctype not in h_dict or dsttype not in h_dict:
                continue
            if g.num_edges((srctype, etype, dsttype)) == 0:
                continue

            sub_g = g[(srctype, etype, dsttype)]
            feat_src = h_dict[srctype]
            feat_dst = h_dict[dsttype]
            tw = time_weights.get((srctype, etype, dsttype)) if time_weights else None

            # 直接用 conv 字典中的对应卷积（绕过 HeteroGraphConv 的自动类型查找）
            # homo_ablation → 全部关系共享 'all' 那一个 conv
            conv_module = self.conv_dict['all' if self.homo_ablation else str((srctype, etype, dsttype))]
            out = conv_module(sub_g, feat_src, feat_dst, tw)
            has_incoming = sub_g.in_degrees().to(out.device) > 0
            messages = torch.where(has_incoming.unsqueeze(-1), out,
                                   torch.zeros_like(out))
            final_dict[dsttype] = final_dict[dsttype] + messages

        return final_dict


class TemporalHGT(nn.Module):
    """
    完整的多层时间感知 HGT 编码器

    Args:
        in_dim:       输入特征维度
        hidden_dim:   隐藏维度
        out_dim:      输出维度
        num_heads:    注意力头数
        num_layers:   层数
        etypes:       meta-relation 列表
        lambda_init:  时间衰减初始值
        dropout:      dropout rate
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_heads,
                 num_layers, etypes, lambda_init=None, dropout=0.3,
                 homo_ablation=False, node_types=None, edge_feature_fields=()):
        super().__init__()
        self.num_layers = num_layers
        self.homo_ablation = homo_ablation
        if homo_ablation:
            # 日志/结果文件仍按边类型名索引 λ → 共享值需广播回全部出现的边名
            # (并保证 run_phase2/3 硬编码的 burn/create/mint/transfer 四个键存在)
            canon = {e[1] for e in (etypes or [])}
            self.homo_edge_names = sorted(canon | {'transfer', 'mint', 'burn', 'create'})

        # 第一层
        self.layers = nn.ModuleList()
        self.layers.append(
            TemporalHGTLayer(in_dim, (hidden_dim if num_layers > 1 else out_dim) // num_heads, num_heads,
                             etypes, lambda_init, dropout, homo_ablation, node_types, edge_feature_fields)
        )

        # 中间层
        for _ in range(num_layers - 2):
            self.layers.append(
                TemporalHGTLayer(hidden_dim, hidden_dim // num_heads, num_heads,
                                 etypes, lambda_init, dropout, homo_ablation, node_types, edge_feature_fields)
            )

        # 最后一层
        if num_layers > 1:
            self.layers.append(
                TemporalHGTLayer(hidden_dim, out_dim // num_heads, num_heads,
                                 etypes, lambda_init, dropout, homo_ablation, node_types, edge_feature_fields)
            )

    def forward(self, g, h_dict, current_time=None):
        """多层前向传播"""
        for i, layer in enumerate(self.layers):
            h_dict = layer(g, h_dict, current_time)
            # 层间 ReLU 激活（最后一层不加）
            if i < len(self.layers) - 1:
                h_dict = {k: F.relu(v) for k, v in h_dict.items()}
        return h_dict

    def reset_lambda(self, init_dict):
        for layer in self.layers:
            for ename, p in layer.lambda_decay.items():
                # homo: 共享 λ 重置回首条边类型初值 (_homo_default), 与 hetero
                # 每条 λ 重置回各自 LAMBDA_INIT 的协议保持一致
                default = getattr(layer, '_homo_default', 0.5)
                p.data.fill_(init_dict.get(ename, default))

    def get_lambda_values(self):
        """返回当前所有 λ_r 的值（用于分析和可视化）"""
        lambdas = {}
        if self.homo_ablation:
            # 共享 λ → 广播回全部边名, 保持 run_phase2/3 日志格式不变
            shared = self.layers[0].lambda_decay['_shared'].item()
            for ename in self.homo_edge_names:
                lambdas[ename] = shared
            return lambdas
        for layer in self.layers:
            for etype, param in layer.lambda_decay.items():
                if etype not in lambdas:
                    lambdas[etype] = param.item()
        return lambdas
