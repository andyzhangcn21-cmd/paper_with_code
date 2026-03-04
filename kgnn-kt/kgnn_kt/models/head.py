from __future__ import annotations
import torch
import torch.nn as nn

class PredHead(nn.Module):
    """MLP head (BN + GeLU + Dropout + Sigmoid)."""
    def __init__(self, in_dim: int = 256, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.bn = nn.BatchNorm1d(hidden)  # approximation; paper says normalize along feature dim
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc1(z)
        h = self.bn(h)
        h = self.act(h)
        h = self.drop(h)
        y = torch.sigmoid(self.fc2(h)).squeeze(-1)
        return y
