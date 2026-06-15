"""Google Cloud Firestore persistence for chat conversations (DoubleChat).

Data model
----------
Collection: "conversations"
  Document: <auto-id>
    title:      str   — short label (first user message, truncated)
    model:      str   — e.g. "gemini-3.5-flash"
    provider:   str   — e.g. "Google Vertex AI"
    created_at: timestamp
    updated_at: timestamp
    messages:   list[ {role, content} ]   — the full transcript

Auth: Application Default Credentials (same as Vertex AI) + GCP_PROJECT.

Every public function returns a `(value, latency_ms)` tuple so the UI can
display read/write timings — matching the assignment's sample screenshot
("Load Time for Cloud Firestore", "Search Time for Cloud Firestore").
"""

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from google.cloud import firestore

import config


@dataclass
class Conversation:
    """One chat session, mirroring a Firestore document."""

    id: Optional[str]                       # Firestore document id (None = unsaved)
    title: str
    model: str
    provider: str
    messages: list[dict] = field(default_factory=list)  # [{role, content}, ...]


@lru_cache(maxsize=1)
def _client() -> firestore.Client:
    """Create the Firestore client once, on first use (lazy singleton)."""
    return firestore.Client(project=config.GCP_PROJECT)


def _timed(func):
    """Run func, return (result, elapsed_ms)."""
    start = time.perf_counter()
    result = func()
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    return result, elapsed_ms


def save_conversation(conv: Conversation) -> tuple[str, float]:
    """Create or update a conversation. Returns (document_id, write_ms)."""
    collection = _client().collection(config.FIRESTORE_COLLECTION)

    payload = {
        "title": conv.title,
        "model": conv.model,
        "provider": conv.provider,
        "messages": conv.messages,
        "updated_at": firestore.SERVER_TIMESTAMP,
    }

    def _write() -> str:
        if conv.id:  # update existing
            doc_ref = collection.document(conv.id)
            doc_ref.set(payload, merge=True)
            return conv.id
        # create new — stamp created_at only on first write
        payload["created_at"] = firestore.SERVER_TIMESTAMP
        _, doc_ref = collection.add(payload)
        return doc_ref.id

    return _timed(_write)


def load_conversation(conv_id: str) -> tuple[Optional[Conversation], float]:
    """Load one conversation by id. Returns (Conversation|None, read_ms)."""
    def _read() -> Optional[Conversation]:
        snap = _client().collection(config.FIRESTORE_COLLECTION).document(conv_id).get()
        if not snap.exists:
            return None
        data = snap.to_dict()
        return Conversation(
            id=snap.id,
            title=data.get("title", "(untitled)"),
            model=data.get("model", ""),
            provider=data.get("provider", ""),
            messages=data.get("messages", []),
        )

    return _timed(_read)


def list_conversations(limit: int = 25) -> tuple[list[Conversation], float]:
    """List recent conversations, newest first. Returns (list, query_ms)."""
    def _query() -> list[Conversation]:
        docs = (
            _client()
            .collection(config.FIRESTORE_COLLECTION)
            .order_by("updated_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        result = []
        for snap in docs:
            data = snap.to_dict()
            result.append(
                Conversation(
                    id=snap.id,
                    title=data.get("title", "(untitled)"),
                    model=data.get("model", ""),
                    provider=data.get("provider", ""),
                    messages=data.get("messages", []),
                )
            )
        return result

    return _timed(_query)


def delete_conversation(conv_id: str) -> tuple[None, float]:
    """Delete one conversation. Returns (None, delete_ms)."""
    def _delete() -> None:
        _client().collection(config.FIRESTORE_COLLECTION).document(conv_id).delete()

    return _timed(_delete)
