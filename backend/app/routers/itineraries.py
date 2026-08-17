"""Itinerary generation, retrieval and single-slot editing.

Every response is assembled server-side from persisted rows — the client never computes a total
and never patches a day locally (spec §6.5, §6 single-slot edit).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..db import get_db
from ..models import Itinerary, User
from ..repo import get_itinerary_or_404, get_slot_or_404, owned_query
from ..schemas import (
    AlternativeOut,
    DayPatchResponse,
    GenerateRequest,
    ItineraryOut,
    ItinerarySummary,
    SlotPatch,
)
from ..services import itinerary as service

router = APIRouter(prefix="/itineraries", tags=["itineraries"])


@router.get("", response_model=list[ItinerarySummary])
def list_itineraries(
    current: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> list[Itinerary]:
    return owned_query(db, Itinerary, current.id).order_by(Itinerary.updated_at.desc()).all()


@router.post("/generate", response_model=ItineraryOut, status_code=status.HTTP_201_CREATED)
def generate_itinerary(
    payload: GenerateRequest,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    try:
        itinerary = service.generate(
            db,
            current,
            start_date=payload.start_date,
            num_days=payload.num_days,
            total_budget=payload.total_budget,
            start_lat=payload.start_lat,
            start_lng=payload.start_lng,
            event_id=payload.event_id,
            title=payload.title,
            currency=payload.currency,
            prayer_breaks=payload.prayer_breaks,
        )
    except service.IntakeIncomplete as exc:
        # 422 with the exact list, so the chat can render its numbered intake checklist.
        raise HTTPException(
            status_code=422,
            detail={"error": "intake_incomplete", "missing_fields": exc.missing},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return service.itinerary_payload(db, itinerary)


@router.get("/{itinerary_id}", response_model=ItineraryOut)
def get_itinerary(
    itinerary_id: int,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    itinerary = get_itinerary_or_404(db, itinerary_id, current.id)
    return service.itinerary_payload(db, itinerary)


@router.delete(
    "/{itinerary_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    response_model=None,
)
def delete_itinerary(
    itinerary_id: int,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    itinerary = get_itinerary_or_404(db, itinerary_id, current.id)
    db.delete(itinerary)
    db.commit()


@router.get("/{itinerary_id}/slots/{slot_id}/alternatives", response_model=list[AlternativeOut])
def slot_alternatives(
    itinerary_id: int,
    slot_id: int,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[dict]:
    itinerary = get_itinerary_or_404(db, itinerary_id, current.id)
    slot = get_slot_or_404(db, itinerary_id, slot_id, current.id)
    return service.alternatives_for_slot(db, itinerary, current, slot)


@router.patch("/{itinerary_id}/slots/{slot_id}", response_model=DayPatchResponse)
def patch_slot(
    itinerary_id: int,
    slot_id: int,
    payload: SlotPatch,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """Replace / adjust / remove one slot; returns the whole updated day and budget."""
    itinerary = get_itinerary_or_404(db, itinerary_id, current.id)
    slot = get_slot_or_404(db, itinerary_id, slot_id, current.id)

    try:
        _, _, day_index = service.patch_slot(
            db,
            itinerary,
            current,
            slot,
            action=payload.action,
            place_id=payload.place_id,
            start_time=payload.start_time,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return service.day_payload(db, itinerary, day_index)


@router.post("/{itinerary_id}/days/{day_index}/cheaper", response_model=ItineraryOut)
def make_day_cheaper(
    itinerary_id: int,
    day_index: int,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    itinerary = get_itinerary_or_404(db, itinerary_id, current.id)
    try:
        service.cheaper_day(db, itinerary, current, day_index)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return service.itinerary_payload(db, itinerary)


@router.post("/{itinerary_id}/prayer-breaks", response_model=ItineraryOut)
def add_prayer_breaks(
    itinerary_id: int,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    itinerary = get_itinerary_or_404(db, itinerary_id, current.id)
    service.add_prayer_breaks(db, itinerary, current)
    return service.itinerary_payload(db, itinerary)
