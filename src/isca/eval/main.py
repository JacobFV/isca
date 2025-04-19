"""
Main entry point for ISCA evaluation.
"""

from __future__ import annotations
import yaml, torch, os, argparse
from pathlib import Path
import importlib
from torch.utils.data import DataLoader

from isca.data.text_dataset import TextDataset
from isca.models.isca import ISCA

# Import evaluation analysis functions
from isca.eval.analysis.basic import evaluate
from isca.eval.visualizations.plots import visualize_centroids, visualize_metrics


def load_cfg(path):
    return yaml.safe_load(Path(path).read_text())


def main(args):
    """
    Main entry point for ISCA evaluation.

    Args:
        args: Command line arguments
    """
    cfg_all = load_cfg(args.config)
    cfg_m, cfg_t, cfg_l = (
        cfg_all.get("model", {}),
        cfg_all.get("train", {}),
        cfg_all.get("loss", {}),
    )

    # Create eval dataset
    ds = TextDataset(
        args.eval_data or cfg_t.get("dataset"),
        cfg_m.get("backbone"),
        cfg_t.get("max_seq"),
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # Load model
    model = ISCA({**cfg_m, **cfg_l}).to(args.device)

    if args.checkpoint:
        print(f"Loading model from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

    # Create output directory for visualizations if needed
    save_path = None
    if args.save_plots:
        save_path = Path(args.save_plots)
        save_path.mkdir(parents=True, exist_ok=True)

    # Run the specified evaluation
    if args.eval_type == "basic":
        # Run basic evaluation
        metrics = evaluate(model, dl, args.device, {**cfg_m, **cfg_l})

        print("\n===== Basic Evaluation Results =====")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        # Visualize metrics if requested
        if args.visualize:
            visualize_metrics(
                metrics, save_path=save_path / "metrics.png" if save_path else None
            )

    elif args.eval_type == "centroids":
        # Run centroid visualization
        visualize_centroids(
            model, save_path=save_path / "centroids.png" if save_path else None
        )

    elif args.eval_type == "graph":
        # Dynamically import and run graph structure analysis
        from isca.eval.analysis.graph import analyze_graph_structure

        sample_batch = next(iter(dl))
        metrics = analyze_graph_structure(
            model, sample_batch, args.device, save_path=save_path
        )

        print("\n===== Graph Structure Analysis =====")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    elif args.eval_type == "memory":
        # Dynamically import and run graph memory analysis
        from isca.eval.analysis.graph import analyze_graph_memory

        metrics = analyze_graph_memory(
            model, dl, args.device, num_batches=5, save_path=save_path
        )

        print("\n===== Graph Memory Analysis =====")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    elif args.eval_type == "symbolic":
        # Dynamically import and run symbolic representation analysis
        from isca.eval.analysis.symbolic import analyze_symbolic_representations

        sample_batch = next(iter(dl))
        metrics = analyze_symbolic_representations(
            model,
            sample_batch,
            args.device,
            cfg={**cfg_m, **cfg_l},
            save_path=save_path,
        )

        print("\n===== Symbolic Representation Analysis =====")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    elif args.eval_type == "operators":
        # Dynamically import and run operator flow analysis
        from isca.eval.analysis.operators import analyze_operator_flows

        sample_batch = next(iter(dl))
        metrics = analyze_operator_flows(
            model, sample_batch, args.device, save_path=save_path
        )

        print("\n===== Operator Flow Analysis =====")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    elif args.eval_type == "roles":
        # Dynamically import and run role similarity analysis
        from isca.eval.analysis.role import analyze_role_similarity

        sample_batch = next(iter(dl))
        metrics = analyze_role_similarity(
            model, sample_batch, args.device, save_path=save_path
        )

        print("\n===== Role Similarity Analysis =====")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    elif args.eval_type == "mmlu":
        # Run MMLU benchmark
        from isca.eval.benchmarks import evaluate_mmlu

        evaluate_mmlu()

    elif args.eval_type == "mmlu_pro":
        # Run MMLU-Pro benchmark
        from isca.eval.benchmarks import evaluate_mmlu_pro

        evaluate_mmlu_pro()

    elif args.eval_type == "real_bench":
        # Run REAL Bench benchmark
        from isca.eval.benchmarks import evaluate_real_bench_cli

        evaluate_real_bench_cli()

    elif args.eval_type == "custom" and args.custom_module:
        # Run a custom evaluation module
        try:
            # Import the custom module
            module_path = args.custom_module
            if module_path.endswith(".py"):
                module_path = module_path[:-3]
            module_path = module_path.replace("/", ".")

            custom_module = importlib.import_module(module_path)

            # Call the main function from the custom module
            if hasattr(custom_module, "main"):
                custom_module.main(args)
            else:
                print(
                    f"Error: Custom module {args.custom_module} does not have a 'main' function."
                )
        except ImportError as e:
            print(f"Error importing custom module: {e}")

    else:
        print(f"Unknown evaluation type: {args.eval_type}")
        print(
            "Available evaluation types: basic, centroids, graph, memory, symbolic, operators, roles, mmlu, mmlu_pro, real_bench, custom"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISCA Evaluation")
    parser.add_argument(
        "--config", type=str, default="config/default.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--eval_data", type=str, default=None, help="Path to evaluation data"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument(
        "--save_plots", type=str, default=None, help="Directory to save plots"
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        default="basic",
        help="Type of evaluation to run (basic, centroids, graph, memory, symbolic, operators, roles, mmlu, mmlu_pro, real_bench, custom)",
    )
    parser.add_argument(
        "--custom_module",
        type=str,
        default=None,
        help="Path to custom evaluation module (for eval_type=custom)",
    )

    args = parser.parse_args()
    main(args)
