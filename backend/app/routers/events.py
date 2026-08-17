"""Events CRUD and the upcoming-events feed. Every query is scoped through `repo`."""

from __future__ import annotations

from datetime import date, timedelta

from fastapi import APIRouter, Response, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..db import get_db
from ..models import Event, Itinerary, User
from ..repo import get_owned_or_404, owned_query
from ..schemas import EventIn, EventOut, EventUpdate

router = APIRouter(prefix="/events", tags=["events"])


@router.get("", response_model=list[EventOut])
def list_events(
    current: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> list[Event]:
    return owned_query(db, Event, current.id).order_by(Event.date).all()


@router.get("/upcoming", response_model=list[EventOut])
def upcoming_events(
    horizon_days: int = Query(default=30, ge=1, le=730),
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[Event]:
    today = date.today()
    return (
        owned_query(db, Event, current.id)
        .filter(Event.date >= today, Event.date <= today + timedelta(days=horizon_days))
        .order_by(Event.date)
        .all()
    )


@router.post("", response_model=EventOut, status_code=status.HTTP_201_CREATED)
def create_event(
    payload: EventIn,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Event:
    duplicate = (
        owned_query(db, Event, current.id)
        .filter(Event.title == payload.title, Event.date == payload.date)
        .first()
    )
    if duplicate is not None:
        raise HTTPException(status_code=409, detail="You already have that event on that date")

    event = Event(user_id=current.id, planned=False, **payload.model_dump())
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


@router.patch("/{event_id}", response_model=EventOut)
def update_event(
    event_id: int,
    payload: EventUpdate,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Event:
    event = get_owned_or_404(db, Event, event_id, current.id)
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(event, key, value)
    db.commit()
    db.refresh(event)
    return event


@router.delete(
    "/{event_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    response_model=None,
)
def delete_event(
    event_id: int,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    event = get_owned_or_404(db, Event, event_id, current.id)
    # Detach any itinerary rather than cascading it away — a plan outlives the calendar entry.
    db.query(Itinerary).filter(
        Itinerary.event_id == event.id, Itinerary.user_id == current.id
    ).update({"event_id": None})
    db.delete(event)
    db.commit()
