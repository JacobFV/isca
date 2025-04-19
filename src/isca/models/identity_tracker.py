from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
import networkx as nx


class IdentityTracker(nn.Module):
    """
    tracks persistence of the 'self' subgraph; exposes spectral‑persistence loss.
    """

    def __init__(self):
        super().__init__()
        self.prev_adj: torch.Tensor | None = None

    @staticmethod
    def subgraph_centrality(adj: torch.Tensor, top_k: int = 10):
        """
        centrality‑based node mask; picks candidate self nodes.
        adj : (n,n) dense 0/1
        """
        n = adj.size(0)
        g = nx.from_numpy_array(adj.cpu().numpy())
        central = nx.eigenvector_centrality_numpy(g)
        scores = torch.tensor([central[i] for i in range(n)], device=adj.device)
        mask = torch.topk(scores, top_k, largest=True).indices
        return adj[mask][:, mask]

    def forward(self, adj: torch.Tensor):
        """
        adj : (b,n,n) binary adjacency
        returns self‑loss scalar
        """
        B = adj.size(0)
        loss = 0.0
        for b in range(B):
            sub = self.subgraph_centrality(adj[b])
            if self.prev_adj is not None:
                prev_sub = self.subgraph_centrality(self.prev_adj[b])
                loss += (sub - prev_sub).pow(2).mean()
        if self.training:
            self.prev_adj = adj.detach()
        return loss / max(B, 1)
