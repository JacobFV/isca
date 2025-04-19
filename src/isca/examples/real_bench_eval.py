from __future__ import annotations
import torch
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import json
from transformers import AutoTokenizer
from agisdk import real

from ..models.isca_hf import ISCAConfig, ISCAModelForCausalLM

def setup_isca_for_real_bench(checkpoint_path, device="cuda"):
    """Set up the ISCA model for REAL Bench evaluation."""
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
    print(f"Loading ISCA model from checkpoint: {checkpoint_path}")
    model = ISCAModelForCausalLM.from_pretrained(checkpoint_path, config=config)
    model.to(device)
    model.eval()
    
    # Get the tokenizer from the base model
    tokenizer = AutoTokenizer.from_pretrained(config.backbone)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

class ISCAREALBenchAdapter:
    """Adapter to use ISCA with REAL Bench evaluation harness."""
    
    def __init__(self, model, tokenizer, device="cuda", max_new_tokens=1024):
        """Initialize with an ISCA model."""
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
    
    def __call__(self, prompt):
        """Process a prompt and return the response.
        
        This function is called by the REAL Bench harness when evaluating tasks.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # Generate response
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0
            )
            
            # Decode the response
            response = self.tokenizer.decode(generated_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        return response

def evaluate_real_bench(model, tokenizer, device="cuda", output_dir=None, websites=None, task_types=None):
    """Evaluate the ISCA model on REAL Bench.
    
    Args:
        model: The ISCA model
        tokenizer: The tokenizer for the model
        device: Device to use for evaluation
        output_dir: Directory to save results
        websites: List of specific websites to evaluate on (e.g., ["Staynb", "Omnizon"])
        task_types: List of specific task types to evaluate on
    """
    # Create the ISCA model adapter
    model_adapter = ISCAREALBenchAdapter(model, tokenizer, device)
    
    # Set up the REAL Bench harness
    print("Initializing REAL Bench harness...")
    harness = real.Harness(model=model_adapter)
    
    # Configure harness options based on provided parameters
    options = {}
    if websites:
        options["websites"] = websites
    if task_types:
        options["task_types"] = task_types
    
    # Run the evaluation
    print("Running REAL Bench evaluation...")
    results = harness.run(**options)
    
    # Print summary results
    print("\n===== REAL Bench Evaluation Results =====")
    print(f"Overall REAL Score: {results['real_score']:.4f}")
    
    # Print website-specific results
    if "website_scores" in results:
        print("\nWebsite Scores:")
        for website, score in results["website_scores"].items():
            print(f"{website}: {score:.4f}")
    
    # Print task type results
    if "task_type_scores" in results:
        print("\nTask Type Scores:")
        for task_type, score in results["task_type_scores"].items():
            print(f"{task_type}: {score:.4f}")
    
    # Save results if output directory is specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save the full results
        with open(output_dir / "real_bench_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_dir / 'real_bench_results.json'}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate ISCA model on REAL Bench")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to an ISCA checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--output_dir", type=str, default="results/real_bench", help="Directory to save results")
    parser.add_argument("--websites", type=str, nargs="+", default=None, 
                        help="Specific websites to evaluate on (e.g., Staynb Omnizon)")
    parser.add_argument("--task_types", type=str, nargs="+", default=None,
                        help="Specific task types to evaluate on")
    parser.add_argument("--max_new_tokens", type=int, default=1024, 
                        help="Maximum number of tokens to generate per response")
    args = parser.parse_args()
    
    # Setup the ISCA model
    model, tokenizer = setup_isca_for_real_bench(args.checkpoint, args.device)
    
    # Run the evaluation
    evaluate_real_bench(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        output_dir=args.output_dir,
        websites=args.websites,
        task_types=args.task_types
    )

if __name__ == "__main__":
    main() 