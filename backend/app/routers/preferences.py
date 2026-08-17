"""Preferences, including the slot-edit confirmations raised by the itinerary strip.

Writing a preference also files it into the user's Chroma preference memory, so it can be recalled
semantically in later sessions (spec §5). Chroma being unavailable never fails the write — SQLite
is the source of truth.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Response, Depends, status
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..db import get_db
from ..models import Preference, User
from ..repo import get_owned_or_404, owned_query
from ..schemas import PreferenceIn, PreferenceOut
from ..services.memory import MemoryService

log = logging.getLogger(__name__)

router = APIRouter(prefix="/preferences", tags=["preferences"])


@router.get("", response_model=list[PreferenceOut])
def list_preferences(
    current: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> list[Preference]:
    return owned_query(db, Preference, current.id).order_by(Preference.created_at.desc()).all()


@router.post("", response_model=PreferenceOut, status_code=status.HTTP_201_CREATED)
def create_preference(
    payload: PreferenceIn,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Preference:
    existing = (
        owned_query(db, Preference, current.id)
        .filter(Preference.kind == payload.kind, Preference.subject == payload.subject)
        .first()
    )
    if existing is not None:
        # Re-stating a preference strengthens it rather than duplicating the row.
        existing.strength = min(1.0, max(existing.strength, payload.strength) + 0.1)
        if payload.category and not existing.category:
            existing.category = payload.category
        db.commit()
        db.refresh(existing)
        return existing

    pref = Preference(user_id=current.id, **payload.model_dump())
    db.add(pref)
    db.commit()
    db.refresh(pref)

    MemoryService(db, current.id).remember_preference(pref)
    return pref


@router.delete(
    "/{preference_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    response_model=None,
)
def delete_preference(
    preference_id: int,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    pref = get_owned_or_404(db, Preference, preference_id, current.id)
    MemoryService(db, current.id).forget_preference(pref.id)
    db.delete(pref)
    db.commit()
