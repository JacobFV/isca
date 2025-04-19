"""
Graph structure analysis for ISCA.
"""
from __future__ import annotations
import torch
import numpy as np
import matplotlib.pyplot as plt

def analyze_graph_structure(model, sample_batch, device, save_path=None):
    """
    Analyze the emergent graph structure.
    
    Args:
        model: The ISCA model to analyze
        sample_batch: A batch of data to analyze
        device: Device to run analysis on
        save_path: Optional path to save visualizations
        
    Returns:
        Dictionary of graph structure metrics
    """
    model.eval()
    batch = {k: v.to(device) for k, v in sample_batch.items()}
    
    metrics = {}
    
    with torch.no_grad():
        _ = model(**batch, cfg=None, step=0)
        
        # Access the graph adjacency
        A_soft = model.graph_adj.detach().cpu().numpy()
        
        # Add basic graph metrics
        metrics["avg_connectivity"] = np.mean(A_soft)
        metrics["max_connectivity"] = np.max(A_soft)
        
        # Visualize first sample's adjacency matrix
        plt.figure(figsize=(10, 10))
        plt.imshow(A_soft[0], cmap='viridis')
        plt.colorbar()
        plt.title("Soft Graph Adjacency Matrix")
        
        if save_path:
            plt.savefig(f"{save_path}/graph_adjacency.png")
            plt.close()
        else:
            plt.show()
        
        # More advanced metrics could be added:
        # - Spectral properties
        # - Clustering coefficient
        # - Path lengths
    
    return metrics

def analyze_graph_memory(model, dataloader, device, num_batches=5, save_path=None):
    """
    Analyze graph memory over a sequence of inputs.
    
    Args:
        model: The ISCA model to analyze
        dataloader: DataLoader with evaluation data
        device: Device to run analysis on
        num_batches: Number of batches to process
        save_path: Optional path to save visualizations
        
    Returns:
        Dictionary of memory stability metrics
    """
    model.eval()
    memory_evolution = []
    
    # Process several batches to build up memory
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
            
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():
            _ = model(**batch, cfg=None, step=i)
            
            # Capture graph memory state
            if model.graph_memory is not None:
                memory_snapshot = model.graph_memory.detach().cpu().numpy()
                memory_evolution.append(memory_snapshot)
    
    metrics = {}
    
    # Analyze memory evolution (stability, drift, etc.)
    if memory_evolution:
        # Calculate stability metrics
        stability_metrics = []
        for i in range(1, len(memory_evolution)):
            # Measure change between consecutive memory states
            diff = np.mean((memory_evolution[i] - memory_evolution[i-1])**2)
            stability_metrics.append(diff)
            
        metrics["avg_memory_stability"] = np.mean(stability_metrics)
        metrics["memory_drift"] = stability_metrics[-1] if stability_metrics else 0
        
        # Plot memory stability over time
        plt.figure(figsize=(10, 6))
        plt.plot(stability_metrics)
        plt.title("Graph Memory Stability Over Time")
        plt.xlabel("Steps")
        plt.ylabel("Mean Squared Change")
        
        if save_path:
            plt.savefig(f"{save_path}/memory_stability.png")
            plt.close()
        else:
            plt.show()
    
    return metrics 