"""
BHTG 跨模态融合模块

方案 F 节完整实现:
    1. 跨模态投影 → 共享隐空间
    2. 对比对齐 L_align
    3. 逐元素门控融合
    4. 风险分类 MLP
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalFusion(nn.Module):
    """
    跨模态融合: 将图表征和语义表征对齐后通过门控机制融合

    Args:
        graph_dim:  图编码器输出维度
        sem_dim:    语义编码器输出维度
        hidden_dim: 融合隐空间维度
        num_classes: 输出类别数 (1 = 二分类)
        dropout:    dropout rate
    """

    def __init__(self, graph_dim=128, sem_dim=128, hidden_dim=128,
                 num_classes=1, dropout=0.3):
        super().__init__()

        # 投影到共享隐空间
        self.proj_graph = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.proj_sem = nn.Sequential(
            nn.Linear(sem_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 逐元素门控: g = σ(W_g·[h'_graph || h'_sem] + b_g)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        # 风险分类器: 两层 MLP
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, h_graph, h_sem=None, return_gate=False):
        """
        Args:
            h_graph: (B, graph_dim)
            h_sem:   (B, sem_dim) or None — None = graph-only mode
        Returns:
            logit: (B, 1)
            gate:  (B, hidden_dim) if return_gate
        """
        hg = self.proj_graph(h_graph)
        if h_sem is None:
            # Graph-only mode: no semantic stream
            logit = self.classifier(hg)
            if return_gate:
                # No semantic gate in graph-only mode
                return logit, torch.zeros_like(hg)
            return logit

        hs = self.proj_sem(h_sem)

        # 门控融合
        g = self.gate(torch.cat([hg, hs], dim=-1))
        h_fusion = g * hg + (1.0 - g) * hs

        # 分类
        logit = self.classifier(h_fusion)

        if return_gate:
            return logit, g
        return logit

    def align_loss(self, h_graph, h_sem, temperature=0.07):
        """
        跨模态对比对齐损失 L_align (方案 F.1)

        将同一 token 的图表征和语义表征拉近，
        将不同 token 的表征推远。

        Args:
            h_graph: (B, D) — 图编码器输出
            h_sem:   (B, D) — 语义编码器输出

        Returns:
            loss: scalar
        """
        # 归一化
        h_graph = F.normalize(self.proj_graph(h_graph), dim=-1)
        h_sem = F.normalize(self.proj_sem(h_sem), dim=-1)

        # 相似度矩阵
        sim = h_graph @ h_sem.T / temperature   # (B, B)
        labels = torch.arange(sim.shape[0], device=sim.device)

        # 双向对比: 图→语义 + 语义→图
        loss_g2s = F.cross_entropy(sim, labels)
        loss_s2g = F.cross_entropy(sim.T, labels)

        return (loss_g2s + loss_s2g) / 2.0
