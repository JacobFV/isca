"""
Operator flow analysis for ISCA.
"""

from __future__ import annotations
import torch
import numpy as np
import matplotlib.pyplot as plt


def analyze_operator_flows(model, sample_batch, device, save_path=None):
    """
    Analyze the operator flows.

    Args:
        model: The ISCA model to analyze
        sample_batch: A batch of data to analyze
        device: Device to run analysis on
        save_path: Optional path to save visualizations

    Returns:
        Dictionary of operator flow metrics
    """
    model.eval()
    batch = {k: v.to(device) for k, v in sample_batch.items()}
    metrics = {}

    # Get hidden states before and after operator flows
    with torch.no_grad():
        # Initial hidden states (before flows)
        h_init = model.backbone.embeddings(batch["input_ids"])

        # Apply each flow operator and analyze transformations
        flow_displacements = []
        flow_outputs = []

        for i, flow in enumerate(model.flows):
            flow_out = flow(h_init)

            # Measure displacement magnitude
            displacement = (flow_out - h_init).pow(2).sum(-1).sqrt().mean().item()
            flow_displacements.append(displacement)
            metrics[f"flow_{i}_displacement"] = displacement

            # Analyze flow direction consistency
            # Calculate cosine similarity between displacement vectors
            # This could indicate if flows operate in coherent directions
            if i > 0:
                prev_displacement = flow_outputs[-1] - h_init
                curr_displacement = flow_out - h_init

                # Normalize vectors
                norm_prev = torch.nn.functional.normalize(prev_displacement, dim=-1)
                norm_curr = torch.nn.functional.normalize(curr_displacement, dim=-1)

                # Calculate cosine similarity
                cos_sim = (norm_prev * norm_curr).sum(-1).mean().item()
                metrics[f"flow_{i-1}_{i}_cosine_sim"] = cos_sim

            flow_outputs.append(flow_out)

        # Overall metrics
        metrics["avg_flow_displacement"] = np.mean(flow_displacements)
        metrics["max_flow_displacement"] = np.max(flow_displacements)

        # Visualize flow displacements
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(flow_displacements)), flow_displacements)
        plt.title("Operator Flow Displacements")
        plt.xlabel("Flow Operator")
        plt.ylabel("Average Displacement Magnitude")

        if save_path:
            plt.savefig(f"{save_path}/flow_displacements.png")
            plt.close()
        else:
            plt.show()

    return metrics
