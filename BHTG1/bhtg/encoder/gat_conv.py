"""
TimeDecayGATConv — 带时间衰减的图注意力卷积

核心创新：在 DGL apply_edges 计算注意力分数时注入 exp(-λ·Δt)
实现 BHTG 方案中 D.1 节的公式:
    A = softmax( Q·K^T / √d  ×  exp(-λ_r · Δt) )

每类边有独立的可学习 λ_r 参数。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
import math


class TimeDecayGATConv(nn.Module):
    """
    带时间衰减的单头/多头图注意力卷积层

    处理一种 meta-relation 的消息传递。

    Args:
        in_feats:  输入特征维度
        out_feats: 每头输出维度
        num_heads: 注意力头数
        feat_drop: 特征 dropout
        attn_drop: 注意力 dropout
        residual: standalone-node residual; HGT uses False and adds it once
                  after aggregating all incoming relations.
    """

    def __init__(self, in_feats, out_feats, num_heads,
                 feat_drop=0.1, attn_drop=0.1, negative_slope=0.2,
                 residual=False, edge_feature_fields=()):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.residual = residual
        self.edge_feature_fields = tuple(edge_feature_fields)
        self.edge_fc = None
        if self.edge_feature_fields:
            self.edge_fc = nn.Linear(len(self.edge_feature_fields), out_feats * num_heads, bias=False)
            self.attn_edge = nn.Parameter(torch.empty(1, num_heads, out_feats))

        # 线性投影
        self.fc_src = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.fc_dst = nn.Linear(in_feats, out_feats * num_heads, bias=False)

        # 注意力参数: a_l (用于源节点) 和 a_r (用于目标节点)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))

        # 残差连接
        if residual:
            if in_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.res_fc = None

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.edge_fc is not None:
            nn.init.xavier_normal_(self.edge_fc.weight, gain=gain)
            nn.init.xavier_normal_(self.attn_edge, gain=gain)
        if self.res_fc is not None and isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, g, feat_src, feat_dst, time_weight):
        """
        Args:
            g:           DGLGraph (一种 meta-relation 的子图)
            feat_src:    (N_src, in_feats)  源节点特征
            feat_dst:    (N_dst, in_feats)  目标节点特征
            time_weight: (E, 1) 或 (E,)     时间衰减权重 exp(-λ·Δt)

        Returns:
            rst: (N_dst, num_heads * out_feats) 目标节点更新后的特征
        """
        with g.local_scope():
            # 特征 dropout + 线性投影 -> 多头
            h_src = self.fc_src(self.feat_drop(feat_src)).view(-1, self.num_heads, self.out_feats)
            h_dst = self.fc_dst(self.feat_drop(feat_dst)).view(-1, self.num_heads, self.out_feats)

            # 存入图中
            g.srcdata.update({'h': h_src, 'el': (h_src * self.attn_l).sum(dim=-1, keepdim=True)})
            g.dstdata.update({'er': (h_dst * self.attn_r).sum(dim=-1, keepdim=True)})

            # 计算每条边的原始注意力分数。
            g.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = g.edata['e']
            if self.edge_fc is not None:
                from bhtg.model_schema_v3 import POOL_EDGE_FEATURE_FIELDS
                from bhtg.amount_edge_features import GRAPH_AMOUNT_EDGE_FIELDS, amount_edge_features
                if self.edge_feature_fields == GRAPH_AMOUNT_EDGE_FIELDS:
                    edge_values = amount_edge_features(g.edata, g.num_edges(), feat_src.device).to(h_src.dtype)
                elif self.edge_feature_fields == POOL_EDGE_FEATURE_FIELDS:
                    edge_values = []
                    for field in self.edge_feature_fields:
                        values = g.edata.get(field, torch.zeros(g.num_edges(), device=feat_src.device))
                        if values.ndim != 1 or len(values) != g.num_edges():
                            raise ValueError('invalid causal pool feature shape')
                        edge_values.append(values.to(device=feat_src.device, dtype=h_src.dtype))
                    edge_values = torch.stack(edge_values, dim=1)
                    if not bool(torch.isfinite(edge_values).all()) or bool(((edge_values < 0) | (edge_values > 1)).any()):
                        raise ValueError('invalid causal pool feature value')
                    known = edge_values[:, 3]
                    reserved = edge_values[:, 4]
                    if (bool(((known != 0) & (known != 1)).any())
                            or bool(((reserved != 0) & (reserved != 1)).any())
                            or bool((known > reserved).any())):
                        raise ValueError('inconsistent causal pool evidence masks')
                    edge_values = torch.cat([edge_values[:, :3] * known[:, None], edge_values[:, 3:]], dim=1)
                else:
                    raise ValueError('unsupported causal pool feature contract')
                edge_h = self.edge_fc(edge_values).view(-1, self.num_heads, self.out_feats)
                g.edata['edge_h'] = edge_h
                e = e + (edge_h * self.attn_edge).sum(dim=-1, keepdim=True)
            e = self.leaky_relu(e)  # (E, num_heads, 1)

            # ===== 时间衰减注入 (BHTG 核心创新) =====
            if time_weight is not None:
                if time_weight.dim() == 1:
                    time_weight = time_weight.unsqueeze(-1).unsqueeze(-1)
                elif time_weight.dim() == 2:
                    time_weight = time_weight.unsqueeze(-1)
                e = e * time_weight

            # 边级 softmax (按目标节点分组)
            g.edata['a'] = self.attn_drop(edge_softmax(g, e))

            # Legacy: messages depend only on source representation and attention.
            # The timestamp enters through time_weight; edge amount is unused.
            if self.edge_fc is None:
                g.update_all(fn.u_mul_e('h', 'a', 'm'), fn.sum('m', 'h_out'))
            else:
                g.apply_edges(lambda edges: {'m': (edges.src['h'] + edges.data['edge_h']) * edges.data['a']})
                g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'h_out'))

            rst = g.dstdata['h_out'].flatten(1)  # (N_dst, H * D)

            # 残差连接
            if self.res_fc is not None:
                res = self.res_fc(feat_dst)
                rst = rst + res

            return rst
