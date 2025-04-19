from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


class AttractorSymbolLayer(nn.Module):
    """
    learnable set of centroids (k‑means style) + differentiable assignment.
    EMA updates ensure centroids drift toward stable attractors.
    """

    def __init__(self, dim: int, k: int, ema: float = 0.05):
        super().__init__()
        self.dim, self.k = dim, k
        self.register_buffer("centroids", torch.randn(k, dim))
        self.register_buffer("cluster_size", torch.zeros(k))
        self.ema = ema

    @torch.no_grad()
    def _ema_update(self, h, assign):
        # h : (b,n,d) ; assign : (b,n,k)
        counts = assign.sum(1).sum(0)  # (k,)
        summed = (assign.transpose(1, 2) @ h).sum(0)  # (k,d)
        self.cluster_size.mul_(1 - self.ema).add_(self.ema * counts)
        self.centroids.mul_(1 - self.ema).add_(
            self.ema * summed / (counts.unsqueeze(-1) + 1e-4)
        )

    def forward(self, h: torch.Tensor):
        """
        returns:
            attractors : projected embeddings (b,n,d)
            assign     : assignment weights   (b,n,k)
        """
        # cosine distance
        h_norm = F.normalize(h, dim=-1)
        c_norm = F.normalize(self.centroids, dim=-1)
        logits = h_norm @ c_norm.T  # (b,n,k)
        assign = F.softmax(logits, dim=-1)
        attract = assign @ self.centroids  # (b,n,d)

        if self.training:  # EMA centroid update
            self._ema_update(h, assign.detach())

        return attract, assign
