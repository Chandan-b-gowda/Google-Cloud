"""Benchmark package — runs preset prompts across models and measures metrics."""

from benchmark.suite import BenchmarkRow, PRESET_PROMPTS, run_benchmark

__all__ = ["BenchmarkRow", "PRESET_PROMPTS", "run_benchmark"]
