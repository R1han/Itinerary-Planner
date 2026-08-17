"""The SSE chat endpoint.

Streams typed events — `token`, `tool`, `itinerary_updated`, `budget_updated`, `notice`, `done` —
so the right pane re-renders live while the assistant is still talking (spec §8).

Consumed in the browser with fetch + ReadableStream rather than EventSource, because EventSource
cannot send an Authorization header.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..db import get_db
from ..models import Conversation, User
from ..repo import get_conversation_or_404
from ..schemas import ChatRequest
from ..services.orchestrator import ChatOrchestrator, sse

log = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def _resolve_conversation(db: Session, user: User, conversation_id: int | None) -> Conversation:
    if conversation_id is not None:
        return get_conversation_or_404(db, conversation_id, user.id)

    conversation = Conversation(user_id=user.id, title="New plan")
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


@router.post("/chat")
def chat(
    payload: ChatRequest,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    conversation = _resolve_conversation(db, current, payload.conversation_id)
    orchestrator = ChatOrchestrator(db, current, conversation)

    def events() -> Iterator[str]:
        # The thread id goes first so the client can attach before any token arrives.
        yield sse("conversation", {"conversation_id": conversation.id, "title": conversation.title})
        try:
            yield from orchestrator.stream(payload.message)
        except Exception as exc:  # noqa: BLE001 — a mid-stream 500 would look like a hang
            log.exception("chat stream aborted")
            db.rollback()
            yield sse("error", {"message": str(exc)})
            yield sse("done", {"conversation_id": conversation.id, "failed": True})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Without this, nginx and friends buffer the whole stream and the UI sees nothing
            # until the assistant has finished talking.
            "X-Accel-Buffering": "no",
        },
    )
