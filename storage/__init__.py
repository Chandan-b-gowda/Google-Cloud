"""Storage layer — persistence backends for DoubleChat.

Currently: Google Cloud Firestore for conversation history.
"""

from storage.firestore_db import (
    Conversation,
    delete_conversation,
    list_conversations,
    load_conversation,
    save_conversation,
)

__all__ = [
    "Conversation",
    "delete_conversation",
    "list_conversations",
    "load_conversation",
    "save_conversation",
]
