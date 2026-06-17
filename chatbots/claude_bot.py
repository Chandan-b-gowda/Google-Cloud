"""Anthropic Claude wrapper.

Authentication: API key from .env (ANTHROPIC_API_KEY).
Model and token budget come from config.py.
"""

import base64
import time
from functools import lru_cache

import anthropic

import config
from chatbots.base import LLMResponse

PROVIDER = "Anthropic API"


@lru_cache(maxsize=1)
def _client() -> anthropic.Anthropic:
    """Create the Anthropic client once, on first use (lazy singleton)."""
    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


def chat_claude(messages: list[dict]) -> LLMResponse:
    """Send a full conversation history to Claude (multi-turn).

    `messages` is a list of {"role": "user"|"assistant", "content": str},
    oldest first — already the exact shape the Anthropic API expects.
    Returns a normalized LLMResponse.
    """
    try:
        start = time.perf_counter()
        response = _client().messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=config.MAX_TOKENS,
            messages=messages,
        )
        elapsed = round(time.perf_counter() - start, 3)

        # The response is a list of typed content blocks; join the text ones.
        text = "".join(
            block.text for block in response.content if block.type == "text"
        )

        return LLMResponse(
            text=text,
            model=config.CLAUDE_MODEL,
            provider=PROVIDER,
            latency_s=elapsed,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    except anthropic.APIError as e:
        return LLMResponse(
            text="",
            model=config.CLAUDE_MODEL,
            provider=PROVIDER,
            error=f"{type(e).__name__}: {e.message}",
        )
    except Exception as e:  # noqa: BLE001 — surface any failure to the UI
        return LLMResponse(
            text="",
            model=config.CLAUDE_MODEL,
            provider=PROVIDER,
            error=f"{type(e).__name__}: {e}",
        )


def ask_claude(prompt: str) -> LLMResponse:
    """Single-turn convenience wrapper (no history)."""
    return chat_claude([{"role": "user", "content": prompt}])


def describe_image(
    image_bytes: bytes,
    mime_type: str = "image/png",
    question: str = "Describe this image in detail.",
) -> LLMResponse:
    """Analyze an image with Claude's vision.

    Claude cannot *generate* images, but it can *see* them — this sends an
    image (base64) plus a question and returns Claude's textual analysis.
    Used to pair against Gemini's image generation in the Playground.
    """
    try:
        data = base64.standard_b64encode(image_bytes).decode("utf-8")
        start = time.perf_counter()
        response = _client().messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=config.MAX_TOKENS,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": data,
                        },
                    },
                    {"type": "text", "text": question},
                ],
            }],
        )
        elapsed = round(time.perf_counter() - start, 3)
        text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        return LLMResponse(
            text=text,
            model=config.CLAUDE_MODEL,
            provider=PROVIDER,
            latency_s=elapsed,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    except anthropic.APIError as e:
        return LLMResponse(
            text="", model=config.CLAUDE_MODEL, provider=PROVIDER,
            error=f"{type(e).__name__}: {e.message}",
        )
    except Exception as e:  # noqa: BLE001
        return LLMResponse(
            text="", model=config.CLAUDE_MODEL, provider=PROVIDER,
            error=f"{type(e).__name__}: {e}",
        )
