#!/usr/bin/env python
from __future__ import annotations
import argparse
import torch
import yaml
from pathlib import Path
import os
import sys

# Add parent directory to path to allow importing from parent package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from isca.models.isca_hf import ISCAConfig, ISCAModelForCausalLM
from transformers import AutoTokenizer

def convert_checkpoint(checkpoint_path, config_path, output_dir, push_to_hub=None):
    """
    Convert an ISCA checkpoint to HuggingFace format.
    
    Args:
        checkpoint_path: Path to the ISCA checkpoint file
        config_path: Path to the ISCA config file
        output_dir: Directory to save the HuggingFace model
        push_to_hub: (Optional) Repository ID to push to HuggingFace Hub
    """
    # Load configuration
    print(f"Loading configuration from {config_path}")
    config_dict = yaml.safe_load(Path(config_path).read_text())
    model_cfg = config_dict["model"]
    loss_cfg = config_dict["loss"]
    
    # Create HuggingFace config
    config = ISCAConfig(
        backbone=model_cfg["backbone"],
        freeze_layers=model_cfg["freeze_layers"],
        hidden_dim=model_cfg["hidden_dim"],
        num_centroids=model_cfg["num_centroids"],
        num_operator_flows=model_cfg["num_operator_flows"],
        flow_depth=model_cfg["flow_depth"],
        tau_role=model_cfg["tau_role"],
        gamma_mem=model_cfg["gamma_mem"],
        lambda_sym=loss_cfg["lambda_sym"],
        lambda_flow=loss_cfg["lambda_flow"],
        lambda_self=loss_cfg["lambda_self"],
    )
    
    # Create ISCA model
    print(f"Loading ISCA model from {checkpoint_path}")
    model = ISCAModelForCausalLM.from_pretrained(checkpoint_path, config=config)
    
    # Get tokenizer from base model
    print(f"Loading tokenizer from {model_cfg['backbone']}")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["backbone"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save model and tokenizer
    print(f"Saving HuggingFace model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Push to hub if requested
    if push_to_hub:
        print(f"Pushing model to HuggingFace Hub as {push_to_hub}")
        model.push_to_hub(push_to_hub, commit_message="Convert ISCA checkpoint to HuggingFace format")
        tokenizer.push_to_hub(push_to_hub, commit_message="Add tokenizer for ISCA model")
    
    print("Conversion complete!")

def main():
    parser = argparse.ArgumentParser(description="Convert ISCA checkpoint to HuggingFace format")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to ISCA checkpoint")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to ISCA config file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for HuggingFace model")
    parser.add_argument("--push_to_hub", type=str, default=None, help="Push model to HuggingFace Hub with this repo ID")
    
    args = parser.parse_args()
    convert_checkpoint(args.checkpoint, args.config, args.output_dir, args.push_to_hub)

if __name__ == "__main__":
    main() 