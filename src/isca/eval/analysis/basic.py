"""
Basic evaluation functions for ISCA.
"""
from __future__ import annotations
import torch
from tqdm import tqdm

def evaluate(model, dataloader, device, loss_weights=None):
    """
    Evaluate model on dataloader and return metrics.
    
    Args:
        model: The ISCA model to evaluate
        dataloader: DataLoader with evaluation data
        device: Device to run evaluation on
        loss_weights: Loss configuration weights
        
    Returns:
        Dictionary of evaluation metrics
    """
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