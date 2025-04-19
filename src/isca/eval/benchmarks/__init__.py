"""
Benchmark evaluation modules for ISCA.
"""

from isca.eval.benchmarks.mmlu_eval import main as evaluate_mmlu
from isca.eval.benchmarks.mmlu_pro_eval import main as evaluate_mmlu_pro
from isca.eval.benchmarks.real_bench_eval import (
    evaluate_real_bench,
    main as evaluate_real_bench_cli,
)

__all__ = [
    "evaluate_mmlu",
    "evaluate_mmlu_pro",
    "evaluate_real_bench",
    "evaluate_real_bench_cli",
]
