"""
Symbolic representation analysis for ISCA.
"""

from __future__ import annotations
import torch
import numpy as np
import matplotlib.pyplot as plt


def analyze_symbolic_representations(
    model, sample_batch, device, cfg=None, save_path=None
):
    """
    Analyze the quality of symbolic representations.

    Args:
        model: The ISCA model to analyze
        sample_batch: A batch of data to analyze
        device: Device to run analysis on
        cfg: Model configuration weights
        save_path: Optional path to save visualizations

    Returns:
        Dictionary of symbolic representation metrics
    """
    model.eval()
    batch = {k: v.to(device) for k, v in sample_batch.items()}
    metrics = {}

    # Forward pass to get representations
    with torch.no_grad():
        out = model(**batch, cfg=cfg, step=0)

        # Extract and analyze symbol assignments
        if hasattr(model, "assign"):
            assign = model.assign.detach().cpu().numpy()

            # 1. Analyze assignment sparsity/entropy
            entropy = -np.sum(assign * np.log(assign + 1e-10), axis=-1)
            metrics["avg_assignment_entropy"] = entropy.mean()

            # 2. Evaluate assignment sparsity
            # Measure how concentrated the assignments are
            max_probs = np.max(assign, axis=-1)
            metrics["avg_max_probability"] = max_probs.mean()

            # 3. Visualize symbol assignments for the first example
            plt.figure(figsize=(12, 6))
            plt.imshow(assign[0], aspect="auto", cmap="viridis")
            plt.colorbar()
            plt.title("Symbol Assignments for Sample")
            plt.xlabel("Centroids")
            plt.ylabel("Sequence Position")

            if save_path:
                plt.savefig(f"{save_path}/symbol_assignments.png")
                plt.close()
            else:
                plt.show()

    return metrics
