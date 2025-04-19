"""
Benchmark evaluation modules for ISCA.
"""

from .mmlu_eval import main as evaluate_mmlu
from .mmlu_pro_eval import main as evaluate_mmlu_pro
from .real_bench_eval import evaluate_real_bench, main as evaluate_real_bench_cli

__all__ = ["evaluate_mmlu", "evaluate_mmlu_pro", "evaluate_real_bench", "evaluate_real_bench_cli"] 