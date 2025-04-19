"""
Visualization functions for ISCA evaluation.
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def visualize_centroids(model, save_path=None):
    """
    Visualize the attractor centroids using PCA.
    
    Args:
        model: The ISCA model to visualize
        save_path: Optional path to save the visualization
    """
    centroids = model.attractor.centroids.detach().cpu().numpy()
    pca = PCA(n_components=2)
    points = pca.fit_transform(centroids)
    
    plt.figure(figsize=(10, 10))
    plt.scatter(points[:, 0], points[:, 1], alpha=0.7)
    plt.title("Attractor Centroid Visualization (PCA)")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_metrics(metrics, save_path=None):
    """
    Visualize evaluation metrics as a bar chart.
    
    Args:
        metrics: Dictionary of metrics to visualize
        save_path: Optional path to save the visualization
    """
    # Filter out perplexity which is usually on a different scale
    plot_metrics = {k: v for k, v in metrics.items() if k != "perplexity"}
    
    plt.figure(figsize=(12, 6))
    plt.bar(plot_metrics.keys(), plot_metrics.values())
    plt.title("Evaluation Metrics")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
    # Separately plot perplexity if present
    if "perplexity" in metrics:
        plt.figure(figsize=(6, 4))
        plt.bar(["perplexity"], [metrics["perplexity"]])
        plt.title("Model Perplexity")
        plt.tight_layout()
        
        if save_path:
            perplexity_path = save_path.replace(".png", "_perplexity.png")
            plt.savefig(perplexity_path)
            plt.close()
        else:
            plt.show()

def visualize_metric_comparison(metric_sets, labels, metric_name, title=None, save_path=None):
    """
    Compare a specific metric across multiple evaluation runs.
    
    Args:
        metric_sets: List of metric dictionaries from different runs
        labels: Labels for each set of metrics
        metric_name: The specific metric to compare
        title: Optional title for the plot
        save_path: Optional path to save the visualization
    """
    values = [metrics.get(metric_name, 0) for metrics in metric_sets]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.title(title or f"Comparison of {metric_name}")
    plt.ylabel(metric_name)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show() 