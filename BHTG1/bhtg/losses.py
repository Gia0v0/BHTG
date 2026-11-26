"""
BHTG 损失函数集合

- FocalLoss:  处理 rugpull/benign 类别不平衡
- GraphContrastiveLoss: 图级时序对比损失 L_graph_con
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss 处理类别不平衡 (方案 F.4)

    L_focal = -(1/N) Σ [w_y · (1 - p_t)^γ · (y·log(ŷ) + (1-y)·log(1-ŷ))]

    Args:
        alpha:  类别权重因子 (rugpull 类权重)
        gamma:  聚焦参数 (default=2.0)
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        Args:
            logits:  (B, 1) — 未归一化的 logit
            targets: (B, 1) 或 (B,) — 0/1 标签

        Returns:
            loss: scalar
        """
        targets = targets.float().view(-1, 1)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # p_t = 正确类别的预测置信度
        pt = torch.exp(-bce)

        # 类别权重: rugpull (y=1) 用 alpha, benign (y=0) 用 1-alpha
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Focal weight
        focal_weight = alpha_t * (1 - pt) ** self.gamma

        return (focal_weight * bce).mean()


class GraphContrastiveLoss(nn.Module):
    """
    图级时序对比损失 L_graph_con (方案 D.3)

    同一 token 在两个时间窗口的 embedding 应相似 (正样本对)，
    不同 token 的 embedding 应被推离 (负样本对)。

    L = -log[ exp(sim(h_t1, h_t2)/τ) / Σ_{j≠i} exp(sim(h_t1, h_j)/τ) ]
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, h_t1, h_t2):
        """
        Args:
            h_t1: (B, D) — 第一个时间窗口的 token embeddings
            h_t2: (B, D) — 第二个时间窗口的 token embeddings

        Returns:
            loss: scalar
        """
        h_t1 = F.normalize(h_t1, dim=-1)
        h_t2 = F.normalize(h_t2, dim=-1)

        # 相似度: (B, B)
        sim = h_t1 @ h_t2.T / self.temperature
        labels = torch.arange(sim.shape[0], device=sim.device)

        # 双向 InfoNCE
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2.0
        return loss
