"""Central configuration for DoubleChat.

Loads the .env file exactly once and exposes typed settings.
Every other module imports from here instead of calling os.getenv()
directly — one place to see (and change) all knobs.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load the .env that sits next to this file, regardless of the
# working directory the app was launched from.
load_dotenv(Path(__file__).parent / ".env")

# --- Anthropic (Claude) -----------------------------------------------------
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-opus-4-8")

# --- Google Cloud (Gemini via Vertex AI) ------------------------------------
# Auth uses Application Default Credentials (ADC), not an API key.
# One-time machine setup: gcloud auth application-default login
GCP_PROJECT: str = os.getenv("GCP_PROJECT", "")
GCP_LOCATION: str = os.getenv("GCP_LOCATION", "global")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_IMAGE_MODEL: str = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
GEMINI_AUDIO_MODEL: str = os.getenv("GEMINI_AUDIO_MODEL", "gemini-2.5-flash")

# --- Generation settings ----------------------------------------------------
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1024"))

# --- Firestore (chat persistence) -------------------------------------------
# Uses the same GCP_PROJECT and ADC credentials as Vertex AI.
FIRESTORE_COLLECTION: str = os.getenv("FIRESTORE_COLLECTION", "conversations")

# --- Fine-tuned headline model ----------------------------------------------
# The base model the tuning job is derived from (compared against the tuned one).
HEADLINE_BASE_MODEL: str = os.getenv("HEADLINE_BASE_MODEL", "gemini-2.5-flash")
# Paste the tuned model's resource name here AFTER the tuning job completes,
# e.g. projects/<num>/locations/us-central1/models/<id>. Empty = not tuned yet.
TUNED_GEMINI_MODEL: str = os.getenv("TUNED_GEMINI_MODEL", "")
# Tuned models are served from the region they were trained in.
TUNED_GEMINI_LOCATION: str = os.getenv("TUNED_GEMINI_LOCATION", "us-central1")
# The same instruction used during training — both models get it, so the only
# difference in the comparison is the fine-tuning itself.
HEADLINE_SYSTEM_INSTRUCTION: str = (
    "You are a news headline writer. Given a short description of a news "
    "story, write a single concise, catchy headline in news style. "
    "Return only the headline, with no extra text."
)


def validate() -> list[str]:
    """Return human-readable configuration problems. Empty list = all good."""
    problems: list[str] = []
    if not ANTHROPIC_API_KEY:
        problems.append("ANTHROPIC_API_KEY is missing — add it to .env")
    if not GCP_PROJECT:
        problems.append("GCP_PROJECT is missing — add it to .env")
    return problems
