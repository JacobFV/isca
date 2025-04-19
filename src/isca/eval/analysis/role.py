"""
Role similarity analysis for ISCA.
"""
from __future__ import annotations
import torch
import numpy as np
import matplotlib.pyplot as plt

def analyze_role_similarity(model, sample_batch, device, save_path=None):
    """
    Analyze emergent attention roles.
    
    Args:
        model: The ISCA model to analyze
        sample_batch: A batch of data to analyze
        device: Device to run analysis on
        save_path: Optional path to save visualizations
        
    Returns:
        Dictionary of role similarity metrics
    """
    model.eval()
    batch = {k: v.to(device) for k, v in sample_batch.items()}
    metrics = {}
    
    with torch.no_grad():
        # Get embeddings
        h = model.backbone.embeddings(batch["input_ids"])
        
        # Compute role attention patterns
        q = torch.nn.functional.normalize(model.role_proj_q(h), dim=-1)
        k = torch.nn.functional.normalize(model.role_proj_k(h), dim=-1)
        sim = torch.einsum("bnh,bmh->bhnm", q, k)
        role_attn = (sim / model.tau_role).softmax(-1)
        
        # Calculate attention entropy to measure how focused the attention is
        attn_entropy = -torch.sum(role_attn * torch.log(role_attn + 1e-10), dim=-1)
        metrics["avg_attn_entropy"] = attn_entropy.mean().item()
        
        # Calculate attention sparsity
        attn_max = role_attn.max(dim=-1)[0].mean().item()
        metrics["avg_attn_max"] = attn_max
        
        # Visualize attention patterns for a few heads
        num_heads_to_show = min(4, role_attn.size(1))
        fig, axes = plt.subplots(1, num_heads_to_show, figsize=(16, 4))
        
        for i in range(num_heads_to_show):
            ax = axes[i] if num_heads_to_show > 1 else axes
            im = ax.imshow(role_attn[0, i].cpu(), cmap='viridis')
            ax.set_title(f"Head {i}")
            
        fig.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/role_attention.png")
            plt.close()
        else:
            plt.show()
    
    return metrics 