"""
Mini-batch MLP Link Predictor — simple & fast.
Predicts whether an edge exists between two node embeddings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgePredictor(nn.Module):
    """MLP link predictor: score = MLP([h_src, h_dst])"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_src, h_dst):
        return self.mlp(torch.cat([h_src, h_dst], dim=-1))
