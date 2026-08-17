"""Chat threads — one per plan, powering the workspace thread rail and its unread dots."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..db import get_db
from ..models import Conversation, Message, User, utcnow
from ..repo import get_conversation_or_404, list_messages, owned_query
from ..schemas import ConversationCreate, ConversationOut, MessageOut

router = APIRouter(prefix="/conversations", tags=["chat"])


def _decorate(conversation: Conversation) -> ConversationOut:
    out = ConversationOut.model_validate(conversation)
    # A thread is unread when it changed after the user last looked at it.
    out.unread = conversation.updated_at > conversation.last_seen_at
    return out


@router.get("", response_model=list[ConversationOut])
def list_conversations(
    current: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> list[ConversationOut]:
    rows = (
        owned_query(db, Conversation, current.id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )
    return [_decorate(row) for row in rows]


@router.post("", response_model=ConversationOut, status_code=status.HTTP_201_CREATED)
def create_conversation(
    payload: ConversationCreate,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ConversationOut:
    conversation = Conversation(
        user_id=current.id, title=payload.title, event_id=payload.event_id
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return _decorate(conversation)


@router.get("/{conversation_id}/messages", response_model=list[MessageOut])
def conversation_messages(
    conversation_id: int,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[Message]:
    return [m for m in list_messages(db, conversation_id, current.id) if m.role in ("user", "assistant")]


@router.post("/{conversation_id}/seen", response_model=ConversationOut)
def mark_seen(
    conversation_id: int,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ConversationOut:
    conversation = get_conversation_or_404(db, conversation_id, current.id)
    conversation.last_seen_at = utcnow()
    db.commit()
    db.refresh(conversation)
    return _decorate(conversation)


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    response_model=None,
)
def delete_conversation(
    conversation_id: int,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    # The plan outlives its chat thread — Conversation.itinerary_id is ON DELETE SET NULL, and
    # nothing cascades from here into itineraries.
    conversation = get_conversation_or_404(db, conversation_id, current.id)
    db.delete(conversation)
    db.commit()
