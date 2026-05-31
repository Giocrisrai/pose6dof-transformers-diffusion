"""Value network for DPPO — small MLP over cond vector."""
from __future__ import annotations

import torch
import torch.nn as nn


class ValueNet(nn.Module):
    """MLP small sobre cond 64-D → V(s) scalar."""

    def __init__(self, cond_dim: int = 64, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond).squeeze(-1)
