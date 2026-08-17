"""Low-level ChromaDB access: the embedded client, the embedding call, and collection handles.

Two collections (spec §5):
  * `places`            — shared catalog; no user filter, by design.
  * `preference_memory` — per-user; EVERY query must carry `where={"user_id": ...}`. Nothing here
                          queries it directly — `services/memory.py` owns that collection and is
                          constructed with a user_id, so an unfiltered query is unwritable.

Embeddings are computed explicitly and passed to Chroma rather than configuring an embedding
function on the collection: it keeps the OpenAI dependency in one place and makes the no-key
failure mode obvious instead of deferred.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import TYPE_CHECKING

from ..config import settings
from .tracing import traced, wrap_openai

if TYPE_CHECKING:  # pragma: no cover
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection

log = logging.getLogger(__name__)

PLACES_COLLECTION = "places"
PREFERENCE_COLLECTION = "preference_memory"

# text-embedding-3-small; batched because the seed pushes ~160 documents at once.
EMBED_BATCH = 96


class EmbeddingUnavailable(RuntimeError):
    """No OPENAI_API_KEY, or the embeddings call failed. Callers fall back to SQL scoring."""


@lru_cache
def get_client() -> "ClientAPI":
    # chromadb 0.6 still constructs its telemetry client before honouring the settings flag, and
    # then logs a traceback per event. Set the env var it reads at import time and mute the logger.
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

    import chromadb
    from chromadb.config import Settings as ChromaSettings

    return chromadb.PersistentClient(
        path=settings.chroma_path,
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )


def get_collection(name: str) -> "Collection":
    return get_client().get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def embeddings_available() -> bool:
    return bool(settings.openai_api_key)


@traced("embed", run_type="llm", model=settings.openai_embedding_model)
def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts with text-embedding-3-small. Raises EmbeddingUnavailable rather than returning
    degenerate vectors — a silent zero-vector would poison similarity scoring invisibly."""
    if not texts:
        return []
    if not settings.openai_api_key:
        raise EmbeddingUnavailable("OPENAI_API_KEY is not set")

    from openai import OpenAI

    client = wrap_openai(OpenAI(api_key=settings.openai_api_key))
    out: list[list[float]] = []
    try:
        for start in range(0, len(texts), EMBED_BATCH):
            batch = texts[start : start + EMBED_BATCH]
            response = client.embeddings.create(model=settings.openai_embedding_model, input=batch)
            out.extend(item.embedding for item in response.data)
    except Exception as exc:  # noqa: BLE001 — any transport/auth error degrades the same way
        raise EmbeddingUnavailable(str(exc)) from exc
    return out


def place_document(place) -> str:
    """The text embedded per place: name + description + tags (spec §5)."""
    tags = " ".join(place.tags or [])
    return f"{place.name}. {place.description} Tags: {tags}. Category: {place.category}. Located in {place.emirate}."
