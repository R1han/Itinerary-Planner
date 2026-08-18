"""The single scoping choke point for user-owned data.

Rule: no route handler ever queries a user-owned table directly. Everything goes through here, and
every function takes `user_id` from the auth dependency. Cross-user access raises 404 (not 403) so
we never leak the existence of another user's row.

`Place` and `TravelCache` are deliberately absent — they are shared/global by design (spec §4).
"""

from __future__ import annotations

from typing import TypeVar

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Query, Session

from .models import (
    Conversation,
    Event,
    FamilyMember,
    Itinerary,
    Message,
    Preference,
    Slot,
)

# Only these may be reached through the generic helpers; anything else is a programming error.
USER_OWNED = (FamilyMember, Preference, Event, Itinerary, Conversation)

T = TypeVar("T")

NOT_FOUND = HTTPException(status_code=404, detail="Not found")


def owned_query(db: Session, model: type[T], user_id: int) -> Query[T]:
    """A query over `model` pre-filtered to `user_id`. Refuses non-user-owned models."""
    if model not in USER_OWNED:
        raise TypeError(f"{model.__name__} is not a user-owned table; query it directly")
    return db.query(model).filter(model.user_id == user_id)  # type: ignore[attr-defined]


def get_owned_or_404(db: Session, model: type[T], obj_id: int, user_id: int) -> T:
    obj = owned_query(db, model, user_id).filter(model.id == obj_id).one_or_none()  # type: ignore[attr-defined]
    if obj is None:
        raise NOT_FOUND
    return obj


# --- ownership inherited through a parent ---------------------------------------------------


def get_itinerary_or_404(db: Session, itinerary_id: int, user_id: int) -> Itinerary:
    return get_owned_or_404(db, Itinerary, itinerary_id, user_id)


def get_slot_or_404(db: Session, itinerary_id: int, slot_id: int, user_id: int) -> Slot:
    """Slots inherit ownership from their itinerary — join rather than trusting slot_id."""
    slot = (
        db.query(Slot)
        .join(Itinerary, Slot.itinerary_id == Itinerary.id)
        .filter(Slot.id == slot_id, Slot.itinerary_id == itinerary_id, Itinerary.user_id == user_id)
        .one_or_none()
    )
    if slot is None:
        raise NOT_FOUND
    return slot


def get_conversation_or_404(db: Session, conversation_id: int, user_id: int) -> Conversation:
    return get_owned_or_404(db, Conversation, conversation_id, user_id)


def list_messages(db: Session, conversation_id: int, user_id: int) -> list[Message]:
    get_conversation_or_404(db, conversation_id, user_id)  # ownership gate
    return list(
        db.scalars(
            select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id)
        )
    )
