"""Chatbot wrappers — one module per LLM provider, one shared response type.

Public API:
    ask_gemini / ask_claude   — single-turn (one prompt)
    chat_gemini / chat_claude — multi-turn (full message history)
    LLMResponse               — normalized response shape
"""

from chatbots.base import LLMResponse
from chatbots.claude_bot import ask_claude, chat_claude
from chatbots.gemini_bot import ask_gemini, chat_gemini

__all__ = [
    "LLMResponse",
    "ask_claude",
    "ask_gemini",
    "chat_claude",
    "chat_gemini",
]

# Maps a friendly provider label to its multi-turn function — used by the
# chat UI to dispatch without provider-specific if/else.
CHAT_FUNCTIONS = {
    "Gemini": chat_gemini,
    "Claude": chat_claude,
}
