"""Shared response type for all chatbot wrappers.

Both providers (and any added later, e.g. ChatGPT) return this same
dataclass, so the UI and the future benchmark code never need
provider-specific handling.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """Normalized result of one prompt sent to one LLM."""

    text: str                          # the model's answer ("" on error)
    model: str                         # e.g. "gemini-3.5-flash"
    provider: str                      # e.g. "Google Vertex AI"
    latency_s: Optional[float] = None  # wall-clock seconds for the API call
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    error: Optional[str] = None        # set only when the call failed

    @property
    def ok(self) -> bool:
        return self.error is None
