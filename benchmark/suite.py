"""Benchmark suite — sends a fixed set of prompts to each model and records
latency, token usage, and estimated cost so the results can be compared
in a table.

The prompts cover a spread of task types (factual, reasoning, coding,
creative, multilingual, summarization) so the comparison reflects more
than one kind of work.
"""

import time
from dataclasses import dataclass
from typing import Callable, Optional

from chatbots import ask_claude, ask_gemini

# (category, prompt) pairs. Kept short so a full run stays cheap and fast.
PRESET_PROMPTS: list[tuple[str, str]] = [
    ("Factual", "What is the capital of Australia? Answer in one sentence."),
    ("Reasoning", "If a train travels 60 km in 45 minutes, what is its speed "
                  "in km/h? Show the calculation."),
    ("Coding", "Write a Python function that returns True if a string is a "
               "palindrome. Code only."),
    ("Creative", "Write a two-line poem about the ocean."),
    ("Multilingual", "Translate 'Good morning, how are you?' into German, "
                     "French, and Spanish."),
    ("Summarization", "Summarize in one sentence: Large language models are AI "
                      "systems trained on large amounts of text to generate "
                      "human-like responses."),
]

# Approximate list price in USD per 1,000,000 tokens, as (input, output).
# These are estimates for cost comparison only — verify against each
# provider's official pricing page before quoting them in the report.
PRICING_USD_PER_1M: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash": (0.30, 2.50),
    "claude-opus-4-8": (5.00, 25.00),
}


@dataclass
class BenchmarkRow:
    """One prompt run against one model."""

    category: str
    provider: str          # "Gemini" or "Claude"
    model: str
    latency_s: Optional[float]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cost_usd: Optional[float]
    answer: str
    ok: bool


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """Estimate USD cost of one call from token counts. None if price unknown."""
    if model not in PRICING_USD_PER_1M:
        return None
    price_in, price_out = PRICING_USD_PER_1M[model]
    return round((input_tokens * price_in + output_tokens * price_out) / 1_000_000, 6)


def run_benchmark(
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> list[BenchmarkRow]:
    """Run every preset prompt against both models.

    `on_progress(done, total)` is called after each call so the UI can
    show a progress bar. Returns one BenchmarkRow per (prompt, model).
    """
    runners = [("Gemini", ask_gemini), ("Claude", ask_claude)]
    total = len(PRESET_PROMPTS) * len(runners)
    done = 0
    rows: list[BenchmarkRow] = []

    for category, prompt in PRESET_PROMPTS:
        for provider, ask in runners:
            result = ask(prompt)
            cost = estimate_cost(
                result.model,
                result.input_tokens or 0,
                result.output_tokens or 0,
            )
            rows.append(
                BenchmarkRow(
                    category=category,
                    provider=provider,
                    model=result.model,
                    latency_s=result.latency_s,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    cost_usd=cost,
                    answer=result.text if result.ok else (result.error or ""),
                    ok=result.ok,
                )
            )
            done += 1
            if on_progress:
                on_progress(done, total)

    return rows
