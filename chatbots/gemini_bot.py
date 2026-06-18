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

def generate_code(prompt: str, language: str = "") -> LLMResponse:
    """Generate code from a natural-language task description.

    `language` is optional free text (e.g. "Python", "JavaScript").
    If left blank, the model infers the most sensible language from
    the prompt itself. Returns a normal text LLMResponse — the UI is
    responsible for rendering it as a code block.
    """
    language_hint = f" in {language}" if language.strip() else ""
    instruction = (
        f"Write code{language_hint} for the following task:\n\n"
        f"{prompt}\n\n"
        "Return the code in a single fenced code block with the correct "
        "language tag, followed by a brief explanation (2-4 sentences) of "
        "how it works. Do not omit the explanation."
    )

    try:
        start = time.perf_counter()
        response = _client().models.generate_content(
            model=config.GEMINI_MODEL,
            contents=[types.Content(role="user", parts=[types.Part(text=instruction)])],
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

def generate_image(prompt: str) -> LLMResponse:
    """Generate an image from a text prompt using Gemini's image model.

    Returns an LLMResponse with `image_bytes`/`image_mime_type` set on
    success. `text` carries any accompanying caption text the model
    returns alongside the image (Gemini image models can emit both).
    """
    try:
        start = time.perf_counter()
        response = _client().models.generate_content(
            model=config.GEMINI_IMAGE_MODEL,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        )
        elapsed = round(time.perf_counter() - start, 3)

        image_bytes = None
        image_mime_type = None
        caption_text = ""
        candidates = response.candidates or []
        if candidates:
            for part in candidates[0].content.parts or []:
                if getattr(part, "inline_data", None) is not None:
                    image_bytes = part.inline_data.data
                    image_mime_type = part.inline_data.mime_type
                elif getattr(part, "text", None):
                    caption_text += part.text

        if image_bytes is None:
            return LLMResponse(
                text=caption_text,
                model=config.GEMINI_IMAGE_MODEL,
                provider=PROVIDER,
                latency_s=elapsed,
                error="No image was returned by the model.",
            )

        usage = response.usage_metadata
        return LLMResponse(
            text=caption_text,
            model=config.GEMINI_IMAGE_MODEL,
            provider=PROVIDER,
            latency_s=elapsed,
            input_tokens=usage.prompt_token_count if usage else None,
            output_tokens=usage.candidates_token_count if usage else None,
            image_bytes=image_bytes,
            image_mime_type=image_mime_type,
        )

    except Exception as e:  # noqa: BLE001 — surface any failure to the UI
        return LLMResponse(
            text="",
            model=config.GEMINI_IMAGE_MODEL,
            provider=PROVIDER,
            error=f"{type(e).__name__}: {e}",
        )


def transcribe_audio(audio_bytes: bytes, mime_type: str = "audio/wav") -> LLMResponse:
    """Transcribe spoken audio to text using a Gemini multimodal model.

    `audio_bytes` is raw audio data (e.g. straight from
    st.audio_input, which returns WAV bytes). Gemini accepts audio
    as inline data alongside a text instruction.
    """
    try:
        start = time.perf_counter()
        response = _client().models.generate_content(
            model=config.GEMINI_AUDIO_MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(text=(
                            "Transcribe the following audio exactly as "
                            "spoken. Return only the transcript text, with "
                            "no commentary."
                        )),
                        types.Part(
                            inline_data=types.Blob(
                                data=audio_bytes, mime_type=mime_type
                            )
                        ),
                    ],
                )
            ],
        )
        elapsed = round(time.perf_counter() - start, 3)

        usage = response.usage_metadata
        return LLMResponse(
            text=response.text or "",
            model=config.GEMINI_AUDIO_MODEL,
            provider=PROVIDER,
            latency_s=elapsed,
            input_tokens=usage.prompt_token_count if usage else None,
            output_tokens=usage.candidates_token_count if usage else None,
        )

    except Exception as e:  # noqa: BLE001 — surface any failure to the UI
        return LLMResponse(
            text="",
            model=config.GEMINI_AUDIO_MODEL,
            provider=PROVIDER,
            error=f"{type(e).__name__}: {e}",
        )
