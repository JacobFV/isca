from __future__ import annotations
import yaml, torch, os, argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.text_dataset import TextDataset
from .models.isca import ISCA
from .utils.graph_utils import infer_soft_graph

def load_cfg(path): return yaml.safe_load(Path(path).read_text())

def visualize_centroids(model, save_path=None):
    """Visualize the attractor centroids using PCA or t-SNE."""
    from sklearn.decomposition import PCA
    
    centroids = model.attractor.centroids.detach().cpu().numpy()
    pca = PCA(n_components=2)
    points = pca.fit_transform(centroids)
    
    plt.figure(figsize=(10, 10))
    plt.scatter(points[:, 0], points[:, 1], alpha=0.7)
    plt.title("Attractor Centroid Visualization (PCA)")
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def evaluate(model, dataloader, device, loss_weights=None):
    """Evaluate model on dataloader and return metrics."""
    model.eval()
    total_loss = 0
    total_ce = 0
    total_sym = 0
    total_flow = 0
    total_self = 0
    total_role = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Must match the main model's forward params
            out = model(**batch, cfg=loss_weights, step=0)
            
            batch_size = batch["input_ids"].size(0)
            total_samples += batch_size
            
            total_loss += out["loss"].item() * batch_size
            total_ce += out["ce"].item() * batch_size
            total_sym += out["sym"].item() * batch_size
            total_flow += out["flow"].item() * batch_size
            total_self += out["self"].item() * batch_size
            total_role += out["role"].item() * batch_size
    
    metrics = {
        "loss": total_loss / total_samples,
        "ce": total_ce / total_samples,
        "sym": total_sym / total_samples,
        "flow": total_flow / total_samples,
        "self": total_self / total_samples,
        "role": total_role / total_samples,
        "perplexity": torch.exp(torch.tensor(total_ce / total_samples)).item()
    }
    
    return metrics

def analyze_graph_structure(model, sample_batch, device):
    """Analyze the emergent graph structure."""
    model.eval()
    batch = {k: v.to(device) for k, v in sample_batch.items()}
    
    with torch.no_grad():
        _ = model(**batch, cfg=None, step=0)
        
        # Access the graph adjacency
        A_soft = model.graph_adj.detach().cpu().numpy()
        
        # Example: visualize first sample's adjacency matrix
        plt.figure(figsize=(10, 10))
        plt.imshow(A_soft[0], cmap='viridis')
        plt.colorbar()
        plt.title("Soft Graph Adjacency Matrix")
        plt.show()
        
        # Here we could compute graph metrics:
        # - Spectral properties
        # - Clustering coefficient
        # - Path lengths
        # etc.

def main(args):
    cfg_all = load_cfg(args.config)
    cfg_m, cfg_t, cfg_l = cfg_all["model"], cfg_all["train"], cfg_all["loss"]
    
    # Create eval dataset - update this section with your eval data
    ds = TextDataset(args.eval_data or cfg_t["dataset"], cfg_m["backbone"], cfg_t["max_seq"])
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    
    # Load model
    model = ISCA({**cfg_m, **cfg_l}).to(args.device)
    
    if args.checkpoint:
        print(f"Loading model from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    
    # ======================================================================
    # BASIC EVALUATION
    # ======================================================================
    metrics = evaluate(model, dl, args.device, {**cfg_m, **cfg_l})
    
    print("\n===== Evaluation Results =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # ======================================================================
    # ATTRACTOR VISUALIZATION
    # ======================================================================
    if args.visualize:
        visualize_centroids(model, save_path=args.save_plots)
    
    # ======================================================================
    # CUSTOM EVALUATION SECTION 1: SYMBOLIC REPRESENTATION QUALITY
    # ======================================================================
    # Uncomment and modify this section for custom symbolic representation evaluation
    """
    # Get a sample batch
    sample_batch = next(iter(dl))
    sample_batch = {k: v.to(args.device) for k, v in sample_batch.items()}
    
    # Forward pass to get representations
    with torch.no_grad():
        out = model(**sample_batch, cfg={**cfg_m, **cfg_l}, step=0)
        
        # Extract and analyze symbol assignments (model.assign)
        assign = model.assign.detach().cpu().numpy()
        
        # 1. Analyze assignment sparsity/entropy
        entropy = -np.sum(assign * np.log(assign + 1e-10), axis=-1)
        print(f"Average assignment entropy: {entropy.mean():.4f}")
        
        # 2. Visualize symbol assignments
        plt.figure(figsize=(12, 6))
        plt.imshow(assign[0], aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title("Symbol Assignments for Sample")
        plt.xlabel("Centroids")
        plt.ylabel("Sequence Position")
        plt.savefig("symbol_assignments.png")
        
        # 3. Add your custom symbolic metrics here
        # ...
    """
    
    # ======================================================================
    # CUSTOM EVALUATION SECTION 2: OPERATOR FLOW ANALYSIS
    # ======================================================================
    # Uncomment and modify this section for operator flow analysis
    """
    # Sample batch to analyze operator flows
    sample_batch = next(iter(dl))
    sample_batch = {k: v.to(args.device) for k, v in sample_batch.items()}
    
    # Get hidden states before and after operator flows
    with torch.no_grad():
        # Initial hidden states (before flows)
        h_init = model.backbone.embeddings(sample_batch["input_ids"])
        
        # Apply each flow operator and analyze transformations
        flow_outputs = []
        for i, flow in enumerate(model.flows):
            flow_out = flow(h_init)
            
            # Measure displacement magnitude
            displacement = (flow_out - h_init).pow(2).sum(-1).sqrt().mean().item()
            print(f"Flow {i} average displacement: {displacement:.4f}")
            
            # Analyze flow direction consistency
            # ...
            
            flow_outputs.append(flow_out)
            
        # Add your custom operator flow metrics here
        # ...
    """
    
    # ======================================================================
    # CUSTOM EVALUATION SECTION 3: GRAPH MEMORY ANALYSIS
    # ======================================================================
    # Uncomment and modify this section for graph memory analysis
    """
    # Analyze graph memory over a sequence of inputs
    memory_evolution = []
    
    # Process several batches to build up memory
    for i, batch in enumerate(dl):
        if i >= 5:  # Limit to a few batches for analysis
            break
            
        batch = {k: v.to(args.device) for k, v in batch.items()}
        
        with torch.no_grad():
            _ = model(**batch, cfg={**cfg_m, **cfg_l}, step=i)
            
            # Capture graph memory state
            if model.graph_memory is not None:
                memory_snapshot = model.graph_memory.detach().cpu().numpy()
                memory_evolution.append(memory_snapshot)
    
    # Analyze memory evolution (stability, drift, etc.)
    if memory_evolution:
        # Plot memory stability over time
        stability_metrics = []
        for i in range(1, len(memory_evolution)):
            # Measure change between consecutive memory states
            diff = np.mean((memory_evolution[i] - memory_evolution[i-1])**2)
            stability_metrics.append(diff)
            
        plt.figure(figsize=(10, 6))
        plt.plot(stability_metrics)
        plt.title("Graph Memory Stability Over Time")
        plt.xlabel("Steps")
        plt.ylabel("Mean Squared Change")
        plt.savefig("memory_stability.png")
        
        # Add your custom memory analysis metrics here
        # ...
    """
    
    # ======================================================================
    # CUSTOM EVALUATION SECTION 4: ROLE SIMILARITY ANALYSIS
    # ======================================================================
    # Uncomment and modify this section for role similarity analysis
    """
    # Analyze emergent attention roles
    sample_batch = next(iter(dl))
    sample_batch = {k: v.to(args.device) for k, v in sample_batch.items()}
    
    with torch.no_grad():
        # Get embeddings
        h = model.backbone.embeddings(sample_batch["input_ids"])
        
        # Compute role attention patterns
        q = torch.nn.functional.normalize(model.role_proj_q(h), dim=-1)
        k = torch.nn.functional.normalize(model.role_proj_k(h), dim=-1)
        sim = torch.einsum("bnh,bmh->bhnm", q, k)
        role_attn = (sim / model.tau_role).softmax(-1)
        
        # Visualize attention patterns for a few heads
        num_heads_to_show = min(4, role_attn.size(1))
        fig, axes = plt.subplots(1, num_heads_to_show, figsize=(16, 4))
        
        for i in range(num_heads_to_show):
            ax = axes[i] if num_heads_to_show > 1 else axes
            im = ax.imshow(role_attn[0, i].cpu(), cmap='viridis')
            ax.set_title(f"Head {i}")
            
        fig.tight_layout()
        plt.savefig("role_attention.png")
        
        # Add your custom role analysis metrics here
        # ...
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval_data", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--save_plots", type=str, default=None)
    
    args = parser.parse_args()
    main(args) 