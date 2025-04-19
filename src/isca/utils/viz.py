from __future__ import annotations
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_cluster_assignments(assign, save_path=None):
    """
    Visualize cluster assignments across sequence positions.

    Args:
        assign: Tensor of shape (batch_size, seq_len, num_centroids)
        save_path: Optional path to save the plot
    """
    # Take the first sample in the batch
    sample = assign[0].detach().cpu().numpy()

    plt.figure(figsize=(12, 8))
    plt.imshow(sample, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title("Symbol Assignments")
    plt.xlabel("Centroids")
    plt.ylabel("Sequence Position")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_attention_patterns(role_attn, num_heads=4, save_path=None):
    """
    Visualize attention patterns for multiple heads.

    Args:
        role_attn: Tensor of shape (batch_size, num_heads, seq_len, seq_len)
        num_heads: Number of heads to visualize
        save_path: Optional path to save the plot
    """
    # Take the first sample in the batch
    sample = role_attn[0].detach().cpu().numpy()

    # Determine number of heads to show
    num_to_show = min(num_heads, sample.shape[0])

    fig, axes = plt.subplots(1, num_to_show, figsize=(16, 4))

    for i in range(num_to_show):
        ax = axes[i] if num_to_show > 1 else axes
        im = ax.imshow(sample[i], cmap="viridis")
        ax.set_title(f"Head {i}")
        fig.colorbar(im, ax=ax)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_flow_displacement(model, inputs, save_path=None):
    """
    Visualize how different operator flows transform embeddings.

    Args:
        model: ISCA model
        inputs: Input tensor
        save_path: Optional path to save the plot
    """
    model.eval()
    with torch.no_grad():
        # Get initial embeddings
        h_init = model.backbone.embeddings(inputs)

        # Apply each flow
        displacements = []
        for i, flow in enumerate(model.flows):
            h_out = flow(h_init)
            # Calculate L2 displacement
            disp = torch.norm(h_out - h_init, dim=-1)
            displacements.append(disp.mean().item())

        # Plot displacements
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(displacements)), displacements)
        plt.xlabel("Flow Index")
        plt.ylabel("Average Displacement")
        plt.title("Operator Flow Displacement Magnitude")

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


def compare_graphs(A1, A2, save_path=None):
    """
    Compare two graph adjacency matrices.

    Args:
        A1, A2: Adjacency matrices to compare
        save_path: Optional path to save the plot
    """
    A1_np = A1.detach().cpu().numpy()
    A2_np = A2.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot first graph
    im1 = axes[0].imshow(A1_np, cmap="viridis")
    axes[0].set_title("Graph 1")
    fig.colorbar(im1, ax=axes[0])

    # Plot second graph
    im2 = axes[1].imshow(A2_np, cmap="viridis")
    axes[1].set_title("Graph 2")
    fig.colorbar(im2, ax=axes[1])

    # Plot difference
    diff = A1_np - A2_np
    im3 = axes[2].imshow(diff, cmap="RdBu", vmin=-1, vmax=1)
    axes[2].set_title("Difference")
    fig.colorbar(im3, ax=axes[2])

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def generate_evaluation_report(metrics, run_name="latest_run", save_dir="reports"):
    """
    Generate a simple markdown report from evaluation metrics.

    Args:
        metrics: Dictionary of evaluation metrics
        run_name: Name for this evaluation run
        save_dir: Directory to save the report
    """
    report_dir = Path(save_dir)
    report_dir.mkdir(exist_ok=True)

    report_path = report_dir / f"{run_name}_report.md"

    with open(report_path, "w") as f:
        f.write(f"# ISCA Evaluation Report: {run_name}\n\n")

        f.write("## Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")

        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                f.write(f"| {k} | {v:.4f} |\n")
            else:
                f.write(f"| {k} | {v} |\n")

    return report_path
