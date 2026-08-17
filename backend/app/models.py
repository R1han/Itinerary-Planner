"""SQLAlchemy models.

Ownership model (spec §4):
  per-user  — family_members, preferences, events, itineraries (+ slots, travel_segments),
              conversations (+ messages)
  shared    — places, travel_cache  (place-to-place travel is user-agnostic)

`slots` and `travel_segments` inherit ownership through their itinerary; `messages` through their
conversation. Nothing reads these tables without going through `repo.py`.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base

EVENT_TYPES = (
    "birthday",
    "anniversary",
    "family_visit",
    "graduation",
    "eid",
    "holiday",
    "other",
)


def utcnow() -> datetime:
    """Naive UTC. The DateTime columns are naive, and SQLite hands them back naive — returning an
    aware value here would make `a > b` raise as soon as one side came from the database."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    home_base_lat: Mapped[float] = mapped_column(Float, default=25.2048)
    home_base_lng: Mapped[float] = mapped_column(Float, default=55.2708)
    default_currency: Mapped[str] = mapped_column(String(8), default="AED")
    default_budget: Mapped[float] = mapped_column(Float, default=3500.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    family_members: Mapped[list["FamilyMember"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    preferences: Mapped[list["Preference"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    events: Mapped[list["Event"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class FamilyMember(Base):
    __tablename__ = "family_members"
    __table_args__ = (CheckConstraint("role IN ('adult','child')", name="ck_family_role"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str | None] = mapped_column(String(120))

    user: Mapped[User] = relationship(back_populates="family_members")


class Preference(Base):
    __tablename__ = "preferences"
    __table_args__ = (
        CheckConstraint("kind IN ('like','dislike')", name="ck_pref_kind"),
        CheckConstraint("source IN ('stated','slot_edit')", name="ck_pref_source"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    subject: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str | None] = mapped_column(String(64))
    source: Mapped[str] = mapped_column(String(16), default="stated", nullable=False)
    strength: Mapped[float] = mapped_column(Float, default=0.6)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    user: Mapped[User] = relationship(back_populates="preferences")


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (
        UniqueConstraint("user_id", "title", "date", name="uq_event_user_title_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)
    planned: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # Set when an event happens at a known venue — a concert, a festival. The planner pins that
    # place into the matching day, which is what makes a live event schedulable rather than just
    # a calendar note.
    place_id: Mapped[int | None] = mapped_column(ForeignKey("places.id", ondelete="SET NULL"))

    user: Mapped[User] = relationship(back_populates="events")
    place: Mapped["Place | None"] = relationship()


class Place(Base):
    """Shared catalog — deliberately has no user_id."""

    __tablename__ = "places"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    emirate: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lng: Mapped[float] = mapped_column(Float, nullable=False)
    category: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    price_adult: Mapped[float] = mapped_column(Float, default=0.0)
    price_child: Mapped[float] = mapped_column(Float, default=0.0)
    # Age-tier pricing: [{"max_age": 2, "price": 0}, {"max_age": 12, "price": 155},
    #                   {"max_age": null, "price": 199}]  — bands are checked in order.
    price_bands: Mapped[list | None] = mapped_column(JSON)
    min_age: Mapped[int] = mapped_column(Integer, default=0)
    open_time: Mapped[str] = mapped_column(String(5), default="09:00")
    close_time: Mapped[str] = mapped_column(String(5), default="22:00")
    avg_duration_min: Mapped[int] = mapped_column(Integer, default=90)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    # Facets the planner filters on, promoted out of `tags` because free text cannot be queried.
    indoor: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    booking_required: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # Months (1-12) the venue is shut. Empty means year-round. A great many UAE outdoor
    # attractions close for high summer, and a boolean "seasonal" flag could only warn about it —
    # a month list lets the planner avoid scheduling a trip into a closed venue at all.
    closed_months: Mapped[list] = mapped_column(JSON, default=list)
    kid_score: Mapped[float] = mapped_column(Float, default=0.5)
    teen_score: Mapped[float] = mapped_column(Float, default=0.5)
    romance_score: Mapped[float] = mapped_column(Float, default=0.5)
    image_url: Mapped[str | None] = mapped_column(String(500))
    category_icon: Mapped[str | None] = mapped_column(String(64))
    description: Mapped[str] = mapped_column(Text, default="")


class Itinerary(Base):
    __tablename__ = "itineraries"
    __table_args__ = (CheckConstraint("num_days <= 5", name="ck_itinerary_max_days"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    event_id: Mapped[int | None] = mapped_column(ForeignKey("events.id", ondelete="SET NULL"))
    title: Mapped[str] = mapped_column(String(200), default="Trip")
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    num_days: Mapped[int] = mapped_column(Integer, nullable=False)
    total_budget: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(String(8), default="AED")
    status: Mapped[str] = mapped_column(String(24), default="draft")
    start_lat: Mapped[float] = mapped_column(Float, default=25.2048)
    start_lng: Mapped[float] = mapped_column(Float, default=55.2708)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    event: Mapped[Event | None] = relationship()
    slots: Mapped[list["Slot"]] = relationship(
        back_populates="itinerary", cascade="all, delete-orphan"
    )
    travel_segments: Mapped[list["TravelSegment"]] = relationship(
        back_populates="itinerary", cascade="all, delete-orphan"
    )


class Slot(Base):
    __tablename__ = "slots"

    id: Mapped[int] = mapped_column(primary_key=True)
    itinerary_id: Mapped[int] = mapped_column(
        ForeignKey("itineraries.id", ondelete="CASCADE"), index=True, nullable=False
    )
    day_index: Mapped[int] = mapped_column(Integer, nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    place_id: Mapped[int] = mapped_column(ForeignKey("places.id"), nullable=False)
    start_time: Mapped[str] = mapped_column(String(5), nullable=False)
    end_time: Mapped[str] = mapped_column(String(5), nullable=False)
    cost_breakdown_json: Mapped[dict] = mapped_column(JSON, default=dict)
    locked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    itinerary: Mapped[Itinerary] = relationship(back_populates="slots")
    place: Mapped[Place] = relationship()


class TravelSegment(Base):
    __tablename__ = "travel_segments"

    id: Mapped[int] = mapped_column(primary_key=True)
    itinerary_id: Mapped[int] = mapped_column(
        ForeignKey("itineraries.id", ondelete="CASCADE"), index=True, nullable=False
    )
    day_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    from_slot_id: Mapped[int | None] = mapped_column(ForeignKey("slots.id", ondelete="CASCADE"))
    to_slot_id: Mapped[int | None] = mapped_column(ForeignKey("slots.id", ondelete="CASCADE"))
    distance_km: Mapped[float] = mapped_column(Float, default=0.0)
    duration_min: Mapped[int] = mapped_column(Integer, default=0)
    mode: Mapped[str] = mapped_column(String(24), default="driving-car")
    est_cost: Mapped[float] = mapped_column(Float, default=0.0)
    estimated: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    geometry_json: Mapped[list | None] = mapped_column(JSON)

    itinerary: Mapped[Itinerary] = relationship(back_populates="travel_segments")


class TravelCache(Base):
    """Shared cache — place-to-place travel is user-agnostic, so everyone benefits."""

    __tablename__ = "travel_cache"

    from_place_id: Mapped[int] = mapped_column(ForeignKey("places.id"), primary_key=True)
    to_place_id: Mapped[int] = mapped_column(ForeignKey("places.id"), primary_key=True)
    mode: Mapped[str] = mapped_column(String(24), primary_key=True, default="driving-car")
    distance_km: Mapped[float] = mapped_column(Float, nullable=False)
    duration_min: Mapped[int] = mapped_column(Integer, nullable=False)
    est_cost: Mapped[float] = mapped_column(Float, default=0.0)
    geometry_json: Mapped[list | None] = mapped_column(JSON)
    provider: Mapped[str] = mapped_column(String(32), default="ors")
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Conversation(Base):
    """One chat thread per plan — powers the workspace thread rail and its unread dots."""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    itinerary_id: Mapped[int | None] = mapped_column(
        ForeignKey("itineraries.id", ondelete="SET NULL")
    )
    event_id: Mapped[int | None] = mapped_column(ForeignKey("events.id", ondelete="SET NULL"))
    title: Mapped[str] = mapped_column(String(200), default="New plan")
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    # Deliberately no onupdate: this tracks the last MESSAGE, not the last row touch. With
    # onupdate, marking a thread seen would bump it too and it would read as unread again.
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Message.id"
    )


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint("role IN ('user','assistant','system','tool')", name="ck_message_role"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True, nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, default="")
    tool_calls_json: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
