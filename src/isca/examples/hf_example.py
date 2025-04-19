from __future__ import annotations
import torch
from transformers import AutoTokenizer
import argparse
import os
from pathlib import Path

from isca.utils.isca_hf import ISCAConfig, ISCAModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Demo the ISCA HuggingFace wrapper")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to an ISCA checkpoint"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="The quick brown fox jumps over the lazy dog.",
        help="Text to process",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--save_hf_model",
        type=str,
        default=None,
        help="Save the model in HF format to this directory",
    )
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="Push model to HF Hub with this repo ID",
    )
    args = parser.parse_args()

    # Load the model configuration
    config = ISCAConfig(
        backbone="meta-llama/Llama-2-7b-hf",  # Default value, will be overridden by model
        freeze_layers=6,
        hidden_dim=4096,
        num_centroids=256,
        num_operator_flows=32,
        flow_depth=2,
        tau_role=0.07,
        gamma_mem=0.95,
        lambda_sym=0.5,
        lambda_flow=1.0,
        lambda_self=0.5,
    )

    # Create the model
    print(f"Loading ISCA model from checkpoint: {args.checkpoint}")
    model = ISCAModelForCausalLM.from_pretrained(args.checkpoint, config=config)
    model.to(args.device)
    model.eval()

    # Get the tokenizer from the base model
    tokenizer = AutoTokenizer.from_pretrained(config.backbone)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize input text
    inputs = tokenizer(args.text, return_tensors="pt").to(args.device)

    # Sample forward pass
    print(f"Processing text: '{args.text}'")
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=inputs.input_ids,  # Using input as labels for demo purposes
        )

    # Print metrics
    print("\nISCA Model Metrics:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.4f}")
        else:
            print(f"  {key}: {value}")

    # Save model in HuggingFace format if requested
    if args.save_hf_model:
        save_dir = Path(args.save_hf_model)
        save_dir.mkdir(exist_ok=True, parents=True)
        print(f"Saving model in HuggingFace format to: {save_dir}")
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    # Push to HuggingFace Hub if requested
    if args.push_to_hub:
        print(f"Pushing model to HuggingFace Hub as: {args.push_to_hub}")
        model.push_to_hub(args.push_to_hub, commit_message="Upload ISCA model")
        tokenizer.push_to_hub(
            args.push_to_hub, commit_message="Upload tokenizer for ISCA model"
        )


if __name__ == "__main__":
    main()
