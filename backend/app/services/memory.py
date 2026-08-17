"""Per-user preference memory over Chroma.

This class is the ONLY thing that touches the `preference_memory` collection. It is constructed
with a user_id and injects `where={"user_id": ...}` into every query, so an unfiltered
cross-user read is not expressible from a route handler (spec §5).

Chroma is an accelerator, never the source of truth: every method degrades to a SQL fallback or a
no-op if embeddings are unavailable, so the app keeps working with the OpenAI key removed.
"""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from ..models import Preference
from . import vectors

log = logging.getLogger(__name__)


class MemoryService:
    def __init__(self, db: Session, user_id: int) -> None:
        self.db = db
        self.user_id = user_id

    # --- writes ------------------------------------------------------------------------------

    def remember_preference(self, pref: Preference) -> bool:
        """File a preference into vector memory. Returns False if it degraded to SQL-only."""
        document = self._document(pref)
        try:
            embedding = vectors.embed([document])[0]
        except vectors.EmbeddingUnavailable as exc:
            log.info("preference memory not embedded (%s); SQL row still authoritative", exc)
            return False

        try:
            self._collection().upsert(
                ids=[self._doc_id(pref.id)],
                documents=[document],
                embeddings=[embedding],
                metadatas=[
                    {
                        "user_id": self.user_id,
                        "kind": pref.kind,
                        "category": pref.category or "",
                        "strength": pref.strength,
                        "preference_id": pref.id,
                    }
                ],
            )
        except Exception:  # noqa: BLE001 — vector memory must never fail a user-facing write
            log.exception("could not write preference %s to Chroma", pref.id)
            return False
        return True

    def forget_preference(self, preference_id: int) -> None:
        try:
            self._collection().delete(ids=[self._doc_id(preference_id)])
        except Exception:  # noqa: BLE001
            log.exception("could not delete preference %s from Chroma", preference_id)

    def reindex_all(self) -> int:
        """Re-embed every stored preference for this user. Used after a bulk import."""
        prefs = self.db.query(Preference).filter(Preference.user_id == self.user_id).all()
        return sum(1 for pref in prefs if self.remember_preference(pref))

    # --- reads -------------------------------------------------------------------------------

    def recall(self, query: str, limit: int = 5) -> list[dict]:
        """Semantically recall this user's preferences. Falls back to their most recent SQL rows.

        The `where` clause below is the isolation guarantee — it is applied here, once, rather
        than at each call site.
        """
        try:
            embedding = vectors.embed([query])[0]
        except vectors.EmbeddingUnavailable:
            return self._sql_fallback(limit)

        try:
            result = self._collection().query(
                query_embeddings=[embedding],
                n_results=limit,
                where={"user_id": self.user_id},
            )
        except Exception:  # noqa: BLE001
            log.exception("preference recall failed; falling back to SQL")
            return self._sql_fallback(limit)

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        recalled: list[dict] = []
        for i, doc in enumerate(documents):
            meta = metadatas[i] if i < len(metadatas) else {}
            # Belt and braces: the `where` filter should make this impossible, but a leak here
            # would be a privacy bug, so re-check rather than trust.
            if meta.get("user_id") != self.user_id:
                log.error("dropped a preference document belonging to another user")
                continue
            recalled.append(
                {
                    "text": doc,
                    "kind": meta.get("kind"),
                    "category": meta.get("category") or None,
                    "strength": meta.get("strength", 0.6),
                    "similarity": 1.0 - float(distances[i]) if i < len(distances) else None,
                }
            )
        return recalled

    def _sql_fallback(self, limit: int) -> list[dict]:
        prefs = (
            self.db.query(Preference)
            .filter(Preference.user_id == self.user_id)
            .order_by(Preference.strength.desc(), Preference.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "text": self._document(p),
                "kind": p.kind,
                "category": p.category,
                "strength": p.strength,
                "similarity": None,
            }
            for p in prefs
        ]

    # --- helpers -----------------------------------------------------------------------------

    def _collection(self):
        return vectors.get_collection(vectors.PREFERENCE_COLLECTION)

    def _doc_id(self, preference_id: int) -> str:
        # Namespaced by user so two users' rows can never collide on an id.
        return f"u{self.user_id}-pref-{preference_id}"

    @staticmethod
    def _document(pref: Preference) -> str:
        verb = "likes" if pref.kind == "like" else "dislikes"
        category = f" (category: {pref.category})" if pref.category else ""
        return f"The family {verb} {pref.subject}{category}."
