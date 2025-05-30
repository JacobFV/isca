from __future__ import annotations
import torch, torch.nn as nn


class OperatorFlow(nn.Module):
    """
    family of learnable vector‑field operators; closure regularized.
    """

    def __init__(self, dim: int, depth: int = 2):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers += [nn.Linear(dim, dim), nn.GELU()]
        layers.append(nn.Linear(dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, h):  # (b,n,d)
        # Ensure the network weights are converted to the same dtype as input
        if next(self.parameters()).dtype != h.dtype:
            self.to(h.dtype)
        return h + self.net(h)
