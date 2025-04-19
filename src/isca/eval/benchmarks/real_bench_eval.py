from __future__ import annotations
import torch
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import json
from transformers import AutoTokenizer
from agisdk import real

# LangChain imports
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# ISCA model imports
from isca.eval.utils.isca_hf import ISCAConfig, ISCAModelForCausalLM


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


def setup_huggingface_for_real_bench(model_name, device="cuda", max_new_tokens=1024):
    """Set up a HuggingFace model for REAL Bench evaluation."""
    from transformers import AutoModelForCausalLM, pipeline

    print(f"Loading HuggingFace model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )

    # Create a pipeline and wrap it with LangChain
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )

    return pipe, tokenizer


def setup_language_model(args):
    """Set up the language model based on the specified type and configuration."""
    model_type = args.model_type.lower()

    if model_type == "isca":
        # Handle ISCA model
        model, tokenizer = setup_isca_for_real_bench(args.checkpoint, args.device)
        return {
            "model": model,
            "tokenizer": tokenizer,
            "type": "isca",
            "max_new_tokens": args.max_new_tokens,
        }

    elif model_type == "huggingface":
        # Handle HuggingFace model
        pipe, tokenizer = setup_huggingface_for_real_bench(
            args.model_name, args.device, args.max_new_tokens
        )
        return {
            "model": pipe,
            "tokenizer": tokenizer,
            "type": "huggingface",
            "max_new_tokens": args.max_new_tokens,
        }

    elif model_type == "openai":
        # Use OpenAI models via LangChain
        print(f"Using OpenAI model: {args.model_name}")
        model = ChatOpenAI(
            model_name=args.model_name, temperature=0.0, max_tokens=args.max_new_tokens
        )
        return {
            "model": model,
            "type": "langchain",
            "max_new_tokens": args.max_new_tokens,
        }

    elif model_type == "anthropic":
        # Use Anthropic models via LangChain
        print(f"Using Anthropic model: {args.model_name}")
        model = ChatAnthropic(
            model_name=args.model_name, temperature=0.0, max_tokens=args.max_new_tokens
        )
        return {
            "model": model,
            "type": "langchain",
            "max_new_tokens": args.max_new_tokens,
        }

    elif model_type == "gemini":
        # Use Google Gemini models via LangChain
        print(f"Using Google Gemini model: {args.model_name}")
        model = ChatGoogleGenerativeAI(
            model=args.model_name,
            temperature=0.0,
            max_output_tokens=args.max_new_tokens,
        )
        return {
            "model": model,
            "type": "langchain",
            "max_new_tokens": args.max_new_tokens,
        }

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


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
                temperature=0.0,
            )

            # Decode the response
            response = self.tokenizer.decode(
                generated_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )

        return response


class HuggingFaceREALBenchAdapter:
    """Adapter to use HuggingFace Pipeline with REAL Bench evaluation harness."""

    def __init__(self, pipeline, max_new_tokens=1024):
        """Initialize with a HuggingFace pipeline."""
        self.pipeline = pipeline
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt):
        """Process a prompt and return the response."""
        outputs = self.pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            return_full_text=False,  # Only return the newly generated tokens
        )

        # Extract the generated text from the pipeline output
        if isinstance(outputs, list) and len(outputs) > 0:
            if "generated_text" in outputs[0]:
                return outputs[0]["generated_text"]

        # Fallback handling
        return str(outputs)


class LangChainREALBenchAdapter:
    """Adapter to use LangChain models with REAL Bench evaluation harness."""

    def __init__(self, model):
        """Initialize with a LangChain model."""
        self.model = model

    def __call__(self, prompt):
        """Process a prompt and return the response."""
        # Handle different LangChain model types
        if hasattr(self.model, "invoke"):
            # Modern LangChain interface
            response = self.model.invoke(prompt)
            if hasattr(response, "content"):
                return response.content
            return str(response)
        else:
            # Older interface or direct models
            return self.model(prompt)


def create_model_adapter(model_info):
    """Create an appropriate adapter based on the model type."""
    model_type = model_info["type"]

    if model_type == "isca":
        return ISCAREALBenchAdapter(
            model=model_info["model"],
            tokenizer=model_info["tokenizer"],
            max_new_tokens=model_info["max_new_tokens"],
        )
    elif model_type == "huggingface":
        return HuggingFaceREALBenchAdapter(
            pipeline=model_info["model"], max_new_tokens=model_info["max_new_tokens"]
        )
    elif model_type == "langchain":
        return LangChainREALBenchAdapter(model=model_info["model"])
    else:
        raise ValueError(f"Unsupported model type for adapter: {model_type}")


def evaluate_real_bench(model_info, output_dir=None, websites=None, task_types=None):
    """Evaluate a model on REAL Bench.

    Args:
        model_info: Dictionary with model configuration
        output_dir: Directory to save results
        websites: List of specific websites to evaluate on (e.g., ["Staynb", "Omnizon"])
        task_types: List of specific task types to evaluate on
    """
    # Create the model adapter
    model_adapter = create_model_adapter(model_info)

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
    parser = argparse.ArgumentParser(description="Evaluate models on REAL Bench")

    # Model configuration
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["isca", "huggingface", "openai", "anthropic", "gemini"],
        help="Type of model to evaluate",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name/path (required for non-ISCA models)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to an ISCA checkpoint (required for ISCA models)",
    )

    # Evaluation settings
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (for ISCA and HuggingFace models)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/real_bench",
        help="Directory to save results",
    )
    parser.add_argument(
        "--websites",
        type=str,
        nargs="+",
        default=None,
        help="Specific websites to evaluate on (e.g., Staynb Omnizon)",
    )
    parser.add_argument(
        "--task_types",
        type=str,
        nargs="+",
        default=None,
        help="Specific task types to evaluate on",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate per response",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.model_type == "isca" and args.checkpoint is None:
        parser.error("--checkpoint is required for ISCA models")
    elif args.model_type != "isca" and args.model_name is None:
        parser.error("--model_name is required for non-ISCA models")

    # Set up the model
    model_info = setup_language_model(args)

    # Run the evaluation
    evaluate_real_bench(
        model_info=model_info,
        output_dir=args.output_dir,
        websites=args.websites,
        task_types=args.task_types,
    )


if __name__ == "__main__":
    main()
