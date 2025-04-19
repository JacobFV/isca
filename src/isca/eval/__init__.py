"""
ISCA evaluation utilities.
"""

from isca.eval.main import main

# Import evaluation analysis functions
from isca.eval.analysis.basic import evaluate
from isca.eval.analysis.graph import analyze_graph_structure, analyze_graph_memory
from isca.eval.analysis.symbolic import analyze_symbolic_representations
from isca.eval.analysis.operators import analyze_operator_flows
from isca.eval.analysis.role import analyze_role_similarity

# Import visualization functions
from isca.eval.visualizations.plots import (
    visualize_centroids,
    visualize_metrics,
    visualize_metric_comparison,
)

# Import benchmark evaluations
from isca.eval.benchmarks import evaluate_mmlu, evaluate_mmlu_pro, evaluate_real_bench

__all__ = [
    # Main entry point
    "main",
    # Analysis functions
    "evaluate",
    "analyze_graph_structure",
    "analyze_graph_memory",
    "analyze_symbolic_representations",
    "analyze_operator_flows",
    "analyze_role_similarity",
    # Visualization functions
    "visualize_centroids",
    "visualize_metrics",
    "visualize_metric_comparison",
    # Benchmark evaluations
    "evaluate_mmlu",
    "evaluate_mmlu_pro",
    "evaluate_real_bench",
]
