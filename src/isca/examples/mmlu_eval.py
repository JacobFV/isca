from __future__ import annotations
import torch
import numpy as np
import argparse
import os
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer

from ..models.isca_hf import ISCAConfig, ISCAModelForCausalLM

def format_example(example):
    """Format an MMLU example into a prompt for the model."""
    question = example["question"]
    choices = example["choices"]
    formatted_choices = ""
    for i, choice in enumerate(choices):
        formatted_choices += f"{chr(65+i)}. {choice}\n"
    
    return f"Question: {question}\n\nChoices:\n{formatted_choices}\nAnswer:"

def evaluate_mmlu(model, tokenizer, dataset, device, batch_size=1, max_seq_len=2048):
    """Evaluate model on MMLU dataset."""
    model.eval()
    results = defaultdict(list)
    
    subjects = set(dataset["test"]["subject"])
    
    for subject in subjects:
        subject_data = [example for example in dataset["test"] if example["subject"] == subject]
        if not subject_data:
            continue
            
        correct = 0
        total = len(subject_data)
        
        for example in tqdm(subject_data, desc=f"Evaluating {subject}"):
            prompt = format_example(example)
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", max_length=max_seq_len, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate predictions for each choice
            with torch.no_grad():
                # Get the logits for the next token
                outputs = model(**inputs)
                logits = outputs.get("logits", model.isca.lm_head(outputs.get("hidden_states", outputs.get("last_hidden_state"))))
                
                # Get the logits for A, B, C, D answer tokens
                next_token_logits = logits[0, -1]
                option_tokens = [tokenizer(" A", add_special_tokens=False).input_ids[0],
                                tokenizer(" B", add_special_tokens=False).input_ids[0],
                                tokenizer(" C", add_special_tokens=False).input_ids[0],
                                tokenizer(" D", add_special_tokens=False).input_ids[0]]
                
                option_logits = [next_token_logits[token].item() for token in option_tokens]
                predicted_option = chr(65 + np.argmax(option_logits))
                
                # Check if prediction is correct
                correct_option = example["answer"][0]  # Answer format is like "A" or "B"
                is_correct = predicted_option == correct_option
                
                if is_correct:
                    correct += 1
        
        # Record results for this subject
        accuracy = correct / total
        results[subject] = accuracy
        print(f"{subject}: {accuracy:.4f} ({correct}/{total})")
    
    # Calculate overall accuracy
    all_subjects = list(results.keys())
    avg_accuracy = sum(results.values()) / len(all_subjects)
    
    # Group subjects by category
    categories = {
        "STEM": ["abstract_algebra", "astronomy", "college_biology", "college_chemistry", 
                "college_computer_science", "college_mathematics", "college_physics",
                "computer_security", "conceptual_physics", "electrical_engineering",
                "elementary_mathematics", "high_school_biology", "high_school_chemistry",
                "high_school_computer_science", "high_school_mathematics", "high_school_physics",
                "high_school_statistics", "machine_learning"],
        "Humanities": ["formal_logic", "high_school_european_history", "high_school_us_history",
                      "high_school_world_history", "international_law", "jurisprudence", 
                      "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
                      "prehistory", "world_religions"],
        "Social Sciences": ["econometrics", "high_school_geography", "high_school_government_and_politics",
                           "high_school_macroeconomics", "high_school_microeconomics", 
                           "high_school_psychology", "human_sexuality", "sociology"],
        "Other": ["anatomy", "business_ethics", "clinical_knowledge", "college_medicine", 
                 "global_facts", "human_aging", "management", "marketing", "medical_genetics",
                 "miscellaneous", "nutrition", "professional_accounting", "professional_law",
                 "professional_medicine", "professional_psychology", "public_relations",
                 "security_studies", "us_foreign_policy", "virology"]
    }
    
    # Calculate category accuracies
    category_results = {}
    for category, subjects in categories.items():
        category_subjects = [s for s in subjects if s in results]
        if category_subjects:
            category_acc = sum(results[s] for s in category_subjects) / len(category_subjects)
            category_results[category] = category_acc
    
    return {
        "subject_results": dict(results),
        "category_results": category_results,
        "average_accuracy": avg_accuracy
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate ISCA model on MMLU benchmark")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to an ISCA checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--subjects", type=str, nargs="+", default=None, help="Specific subjects to evaluate on")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save results as JSON")
    args = parser.parse_args()
    
    # Load the MMLU dataset
    print("Loading MMLU dataset...")
    dataset = load_dataset("cais/mmlu", "all")
    
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
    
    # Filter subjects if specified
    if args.subjects:
        for split in dataset:
            dataset[split] = dataset[split].filter(lambda x: x["subject"] in args.subjects)
    
    # Evaluate on MMLU
    results = evaluate_mmlu(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len
    )
    
    # Print summary of results
    print("\n===== MMLU Evaluation Results =====")
    print(f"Average accuracy: {results['average_accuracy']:.4f}")
    
    print("\nCategory Results:")
    for category, acc in results['category_results'].items():
        print(f"{category}: {acc:.4f}")
    
    # Save results if output file is specified
    if args.output_file:
        import json
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main() 