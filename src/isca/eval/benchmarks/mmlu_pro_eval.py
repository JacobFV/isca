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

# LangChain imports
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# ISCA model imports
from ..models.isca_hf import ISCAConfig, ISCAModelForCausalLM

def format_example(example, use_cot=True):
    """Format an MMLU-Pro example into a prompt for the model.
    
    Args:
        example: The MMLU-Pro example
        use_cot: Whether to use chain-of-thought prompting
    """
    question = example["question"]
    options = example["options"]
    formatted_options = ""
    for i, option in enumerate(options):
        # MMLU-Pro has up to 10 options (A-J)
        formatted_options += f"{chr(65+i)}. {option}\n"
    
    if use_cot:
        # Chain-of-thought prompt
        return f"Question: {question}\n\nChoices:\n{formatted_options}\nLet's think step by step to find the answer. After analyzing the problem carefully, I'll choose one of the options A-J."
    else:
        # Direct answer prompt
        return f"Question: {question}\n\nChoices:\n{formatted_options}\nAnswer:"

def extract_answer_from_generation(text, num_options=10):
    """Extract the answer letter (A-J) from a generated text."""
    # First check if the answer is directly stated in the format "the answer is X" or similar
    lower_text = text.lower()
    
    # Check for common answer patterns
    patterns = [
        "answer is ([a-j])",
        "answer: ([a-j])",
        "choose ([a-j])",
        "option ([a-j])",
        "([a-j]) is correct",
        "select ([a-j])"
    ]
    
    for pattern in patterns:
        import re
        match = re.search(pattern, lower_text)
        if match:
            return match.group(1).upper()
    
    # If no direct statement is found, extract the final answer after chain-of-thought
    # Look for the last occurrence of a standalone letter A-J
    for line in reversed(text.split('\n')):
        line = line.strip()
        if line in [chr(65+i) for i in range(num_options)]:  # A, B, C, etc.
            return line
        
        # Also check for patterns like "Answer: A" at the end
        if line.startswith("Answer:") and len(line) >= 9:
            ans = line[8].strip()
            if ans in [chr(65+i) for i in range(num_options)]:
                return ans
    
    # If still no answer found, default to the first option
    return "A"

def setup_language_model(args):
    """Set up the language model based on the specified type and configuration."""
    model_type = args.model_type.lower()
    
    if model_type == "isca":
        # Load ISCA model directly (not through LangChain)
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
        
        print(f"Loading ISCA model from checkpoint: {args.checkpoint}")
        model = ISCAModelForCausalLM.from_pretrained(args.checkpoint, config=config)
        model.to(args.device)
        model.eval()
        
        # Get the tokenizer from the base model
        tokenizer = AutoTokenizer.from_pretrained(config.backbone)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return {"model": model, "tokenizer": tokenizer, "type": "isca"}
    
    elif model_type == "huggingface":
        # Load HuggingFace model
        from transformers import AutoModelForCausalLM, pipeline
        
        print(f"Loading HuggingFace model: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, 
            torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
            device_map=args.device
        )
        
        # Create a pipeline and wrap it with LangChain
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        
        return {"model": llm, "type": "langchain"}
    
    elif model_type == "openai":
        # Use OpenAI models via LangChain
        print(f"Using OpenAI model: {args.model_name}")
        llm = ChatOpenAI(
            model_name=args.model_name,
            temperature=0.0
        )
        return {"model": llm, "type": "langchain"}
    
    elif model_type == "anthropic":
        # Use Anthropic models via LangChain
        print(f"Using Anthropic model: {args.model_name}")
        llm = ChatAnthropic(
            model_name=args.model_name,
            temperature=0.0
        )
        return {"model": llm, "type": "langchain"}
    
    elif model_type == "gemini":
        # Use Google Gemini models via LangChain
        print(f"Using Google Gemini model: {args.model_name}")
        llm = ChatGoogleGenerativeAI(
            model=args.model_name,
            temperature=0.0
        )
        return {"model": llm, "type": "langchain"}
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def evaluate_mmlu_pro_langchain(model_info, dataset, use_cot=True, max_seq_len=2048):
    """Evaluate LangChain model on MMLU-Pro dataset."""
    results = defaultdict(list)
    
    # Get all categories from the test set
    categories = set(dataset["test"]["category"])
    
    # Setup LangChain prompt template
    if use_cot:
        prompt_template = PromptTemplate.from_template(
            "Question: {question}\n\nChoices:\n{options}\n"
            "Let's think step by step to find the answer. After analyzing the problem carefully, I'll choose one of the options A-J."
        )
    else:
        prompt_template = PromptTemplate.from_template(
            "Question: {question}\n\nChoices:\n{options}\nAnswer:"
        )
    
    # Create chain
    llm = model_info["model"]
    chain = prompt_template | llm | StrOutputParser()
    
    for category in categories:
        category_data = [example for example in dataset["test"] if example["category"] == category]
        if not category_data:
            continue
            
        correct = 0
        total = len(category_data)
        
        for example in tqdm(category_data, desc=f"Evaluating {category}"):
            # Format options
            options_text = ""
            for i, option in enumerate(example["options"]):
                options_text += f"{chr(65+i)}. {option}\n"
            
            # Run through the chain
            generated_text = chain.invoke({
                "question": example["question"],
                "options": options_text
            })
            
            # Extract the answer from the generation
            predicted_option = extract_answer_from_generation(generated_text, len(example["options"]))
            
            # Check if prediction is correct
            correct_option = example["answer"]
            correct_index = example["answer_index"]
            
            # Some answers might be indices, others might be letters
            if isinstance(correct_option, str) and len(correct_option) == 1:
                is_correct = predicted_option == correct_option
            else:
                # If the answer is not in letter format, use the index
                is_correct = predicted_option == chr(65 + correct_index)
            
            if is_correct:
                correct += 1
        
        # Record results for this category
        accuracy = correct / total
        results[category] = accuracy
        print(f"{category}: {accuracy:.4f} ({correct}/{total})")
    
    # Calculate overall accuracy
    all_categories = list(results.keys())
    avg_accuracy = sum(results.values()) / len(all_categories)
    
    # MMLU-Pro categories based on the dataset
    discipline_groups = {
        "STEM": ["math", "physics", "chemistry", "engineering", "computer science", "biology"],
        "Social Sciences": ["economics", "psychology"],
        "Humanities": ["philosophy", "history"],
        "Professional": ["business", "health", "law"],
        "Other": ["other"]
    }
    
    # Group results by discipline
    group_results = {}
    for group, categories in discipline_groups.items():
        group_categories = [cat for cat in categories if cat.lower() in [c.lower() for c in results.keys()]]
        if group_categories:
            matching_categories = []
            for result_category in results.keys():
                if any(gc.lower() == result_category.lower() for gc in group_categories):
                    matching_categories.append(result_category)
            
            if matching_categories:
                group_acc = sum(results[c] for c in matching_categories) / len(matching_categories)
                group_results[group] = group_acc
    
    return {
        "category_results": dict(results),
        "group_results": group_results,
        "average_accuracy": avg_accuracy
    }

def evaluate_mmlu_pro_isca(model_info, dataset, device, use_cot=True, max_new_tokens=200, max_seq_len=2048):
    """Evaluate ISCA model on MMLU-Pro dataset."""
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    model.eval()
    results = defaultdict(list)
    
    # Get all categories from the test set
    categories = set(dataset["test"]["category"])
    
    for category in categories:
        category_data = [example for example in dataset["test"] if example["category"] == category]
        if not category_data:
            continue
            
        correct = 0
        total = len(category_data)
        
        for example in tqdm(category_data, desc=f"Evaluating {category}"):
            prompt = format_example(example, use_cot=use_cot)
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", max_length=max_seq_len, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                if use_cot:
                    # For CoT, we need to generate a response and extract the answer
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=0.0
                    )
                    
                    # Decode the generation
                    generation = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    
                    # Extract the answer from the generation
                    predicted_option = extract_answer_from_generation(generation)
                else:
                    # Direct prediction approach - get logits for option letters
                    outputs = model(**inputs)
                    logits = outputs.get("logits", model.isca.lm_head(outputs.get("hidden_states", outputs.get("last_hidden_state"))))
                    
                    # MMLU-Pro has up to 10 options (A-J)
                    next_token_logits = logits[0, -1]
                    
                    # Create option tokens for all possible options (A-J)
                    num_options = len(example["options"])
                    option_tokens = []
                    for i in range(min(num_options, 10)):  # Up to 10 options
                        letter = chr(65 + i)  # A, B, C, ...
                        token_id = tokenizer(f" {letter}", add_special_tokens=False).input_ids[0]
                        option_tokens.append(token_id)
                    
                    # Get logits for the available options
                    option_logits = [next_token_logits[token].item() for token in option_tokens]
                    predicted_option = chr(65 + np.argmax(option_logits[:num_options]))
                
                # Check if prediction is correct
                # MMLU-Pro provides both answer (letter) and answer_index (number)
                correct_option = example["answer"]
                correct_index = example["answer_index"]
                
                # Some answers might be indices, others might be letters
                if isinstance(correct_option, str) and len(correct_option) == 1:
                    is_correct = predicted_option == correct_option
                else:
                    # If the answer is not in letter format, use the index
                    is_correct = predicted_option == chr(65 + correct_index)
                
                if is_correct:
                    correct += 1
        
        # Record results for this category
        accuracy = correct / total
        results[category] = accuracy
        print(f"{category}: {accuracy:.4f} ({correct}/{total})")
    
    # Calculate overall accuracy
    all_categories = list(results.keys())
    avg_accuracy = sum(results.values()) / len(all_categories)
    
    # MMLU-Pro categories based on the dataset
    discipline_groups = {
        "STEM": ["math", "physics", "chemistry", "engineering", "computer science", "biology"],
        "Social Sciences": ["economics", "psychology"],
        "Humanities": ["philosophy", "history"],
        "Professional": ["business", "health", "law"],
        "Other": ["other"]
    }
    
    # Group results by discipline
    group_results = {}
    for group, categories in discipline_groups.items():
        group_categories = [cat for cat in categories if cat.lower() in [c.lower() for c in results.keys()]]
        if group_categories:
            matching_categories = []
            for result_category in results.keys():
                if any(gc.lower() == result_category.lower() for gc in group_categories):
                    matching_categories.append(result_category)
            
            if matching_categories:
                group_acc = sum(results[c] for c in matching_categories) / len(matching_categories)
                group_results[group] = group_acc
    
    return {
        "category_results": dict(results),
        "group_results": group_results,
        "average_accuracy": avg_accuracy
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate models on MMLU-Pro benchmark")
    
    # Model configuration
    parser.add_argument("--model_type", type=str, required=True, 
                        choices=["isca", "huggingface", "openai", "anthropic", "gemini"],
                        help="Type of model to evaluate")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name/path (required for non-ISCA models)")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to an ISCA checkpoint (required for ISCA models)")
    
    # Evaluation settings
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use (for ISCA and HuggingFace models)")
    parser.add_argument("--max_seq_len", type=int, default=2048, 
                        help="Maximum sequence length")
    parser.add_argument("--use_cot", action="store_true", default=True, 
                        help="Use chain-of-thought reasoning")
    parser.add_argument("--categories", type=str, nargs="+", default=None, 
                        help="Specific categories to evaluate on")
    parser.add_argument("--max_new_tokens", type=int, default=200, 
                        help="Maximum new tokens to generate for CoT")
    parser.add_argument("--output_file", type=str, default=None, 
                        help="Path to save results as JSON")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.model_type == "isca" and args.checkpoint is None:
        parser.error("--checkpoint is required for ISCA models")
    elif args.model_type != "isca" and args.model_name is None:
        parser.error("--model_name is required for non-ISCA models")
    
    # Load the MMLU-Pro dataset
    print("Loading MMLU-Pro dataset...")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    
    # Filter categories if specified
    if args.categories:
        for split in dataset:
            dataset[split] = dataset[split].filter(lambda x: x["category"].lower() in [c.lower() for c in args.categories])
    
    # Set up the language model
    model_info = setup_language_model(args)
    
    # Evaluate on MMLU-Pro
    print(f"Evaluating with {'chain-of-thought' if args.use_cot else 'direct answering'} approach")
    if model_info["type"] == "isca":
        results = evaluate_mmlu_pro_isca(
            model_info=model_info,
            dataset=dataset,
            device=args.device,
            use_cot=args.use_cot,
            max_new_tokens=args.max_new_tokens,
            max_seq_len=args.max_seq_len
        )
    else:  # langchain models
        results = evaluate_mmlu_pro_langchain(
            model_info=model_info,
            dataset=dataset,
            use_cot=args.use_cot,
            max_seq_len=args.max_seq_len
        )
    
    # Print summary of results
    print("\n===== MMLU-Pro Evaluation Results =====")
    print(f"Average accuracy: {results['average_accuracy']:.4f}")
    
    print("\nGroup Results:")
    for group, acc in results['group_results'].items():
        print(f"{group}: {acc:.4f}")
    
    # Save results if output file is specified
    if args.output_file:
        import json
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main() 