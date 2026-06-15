"""Google Gemini wrapper — served through Vertex AI on Google Cloud.

Authentication: Application Default Credentials (ADC), not an API key.
One-time setup per machine:

    gcloud auth application-default login
    gcloud auth application-default set-quota-project <project-id>

Billing goes to the GCP project (covered by the education credit).
"""

import time
from functools import lru_cache

from google import genai
from google.genai import types

import config
from chatbots.base import LLMResponse

PROVIDER = "Google Vertex AI"


@lru_cache(maxsize=1)
def _client() -> genai.Client:
    """Create the Vertex AI client once, on first use (lazy singleton).

    Lazy creation means importing this module never fails — auth or
    project problems surface as a readable LLMResponse.error instead
    of crashing the whole app at startup.
    """
    return genai.Client(
        vertexai=True,
        project=config.GCP_PROJECT,
        location=config.GCP_LOCATION,
    )


def _to_gemini_contents(messages: list[dict]) -> list[types.Content]:
    """Convert our common message format to Gemini's Content list.

    Common format : {"role": "user" | "assistant", "content": str}
    Gemini format : Content(role="user" | "model", parts=[Part(text=...)])
    Note Gemini calls the assistant role "model".
    """
    contents: list[types.Content] = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append(
            types.Content(role=role, parts=[types.Part(text=msg["content"])])
        )
    return contents


def chat_gemini(messages: list[dict]) -> LLMResponse:
    """Send a full conversation history to Gemini (multi-turn).

    `messages` is a list of {"role": "user"|"assistant", "content": str},
    oldest first. Returns a normalized LLMResponse.
    """
    try:
        start = time.perf_counter()
        response = _client().models.generate_content(
            model=config.GEMINI_MODEL,
            contents=_to_gemini_contents(messages),
        )
        elapsed = round(time.perf_counter() - start, 3)

        usage = response.usage_metadata
        return LLMResponse(
            text=response.text or "",
            model=config.GEMINI_MODEL,
            provider=PROVIDER,
            latency_s=elapsed,
            input_tokens=usage.prompt_token_count if usage else None,
            output_tokens=usage.candidates_token_count if usage else None,
        )

    except Exception as e:  # noqa: BLE001 — surface any failure to the UI
        return LLMResponse(
            text="",
            model=config.GEMINI_MODEL,
            provider=PROVIDER,
            error=f"{type(e).__name__}: {e}",
        )


def ask_gemini(prompt: str) -> LLMResponse:
    """Single-turn convenience wrapper (no history)."""
    return chat_gemini([{"role": "user", "content": prompt}])
