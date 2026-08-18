"""Pydantic v2 boundary schemas.

Validation here is the outer wall: >5 days, past dates, coordinates outside the UAE bounding box
and negative budgets are rejected before any service sees them (spec §4). The LLM is never the
validator.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator

# UAE bounding box, generous enough to include all seven emirates and their islands.
UAE_LAT = (22.5, 26.5)
UAE_LNG = (51.0, 56.6)

EventType = Literal["birthday", "anniversary", "family_visit", "graduation", "eid", "holiday", "other"]
HHMM = Annotated[str, Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")]


def validate_uae_coords(lat: float, lng: float) -> None:
    if not (UAE_LAT[0] <= lat <= UAE_LAT[1] and UAE_LNG[0] <= lng <= UAE_LNG[1]):
        raise ValueError("Coordinates must fall inside the UAE")


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --- auth ------------------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=72)
    name: str = Field(min_length=1, max_length=120)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserOut"


class UserOut(ORMModel):
    id: int
    email: str
    name: str
    home_base_lat: float
    home_base_lng: float
    default_currency: str
    default_budget: float


# --- family / preferences --------------------------------------------------------------------


class FamilyMemberIn(BaseModel):
    role: Literal["adult", "child"]
    age: int = Field(ge=0, le=120)
    name: str | None = Field(default=None, max_length=120)

    @model_validator(mode="after")
    def _role_matches_age(self) -> Self:
        if self.role == "child" and self.age >= 18:
            raise ValueError("A family member aged 18+ must have role 'adult'")
        if self.role == "adult" and self.age < 16:
            raise ValueError("A family member under 16 must have role 'child'")
        return self


class FamilyMemberOut(FamilyMemberIn, ORMModel):
    id: int


class FamilyUpdate(BaseModel):
    members: list[FamilyMemberIn] = Field(min_length=1, max_length=20)

    @field_validator("members")
    @classmethod
    def _needs_an_adult(cls, v: list[FamilyMemberIn]) -> list[FamilyMemberIn]:
        if not any(m.role == "adult" for m in v):
            raise ValueError("A family needs at least one adult")
        return v


class PreferenceIn(BaseModel):
    kind: Literal["like", "dislike"]
    subject: str = Field(min_length=1, max_length=255)
    category: str | None = Field(default=None, max_length=64)
    source: Literal["stated", "slot_edit"] = "stated"
    strength: float = Field(default=0.6, ge=0.0, le=1.0)


class PreferenceOut(PreferenceIn, ORMModel):
    id: int
    created_at: dt.datetime


# --- events ----------------------------------------------------------------------------------


class EventIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    event_type: EventType
    date: dt.date
    notes: str | None = Field(default=None, max_length=2000)

    @field_validator("date")
    @classmethod
    def _not_in_the_past(cls, v: dt.date) -> dt.date:
        if v < dt.date.today():
            raise ValueError("Event date cannot be in the past")
        return v


class EventOut(ORMModel):
    id: int
    title: str
    event_type: str
    date: dt.date
    notes: str | None
    planned: bool
    place_id: int | None = None


# --- places ----------------------------------------------------------------------------------


class PlaceOut(ORMModel):
    id: int
    name: str
    emirate: str
    lat: float
    lng: float
    category: str
    price_adult: float
    price_child: float
    min_age: int
    open_time: str
    close_time: str
    avg_duration_min: int
    tags: list[str]
    indoor: bool
    booking_required: bool
    closed_months: list[int]
    image_url: str | None
    category_icon: str | None
    description: str


# --- itineraries -----------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """The intake checklist. Every field here is required before generation (spec §1.2)."""

    event_id: int | None = None
    start_date: dt.date
    num_days: int = Field(ge=1, le=5)
    total_budget: float = Field(gt=0)
    start_lat: float
    start_lng: float
    currency: str = Field(default="AED", max_length=8)
    title: str | None = Field(default=None, max_length=200)
    prayer_breaks: bool = False

    @field_validator("start_date")
    @classmethod
    def _not_in_the_past(cls, v: dt.date) -> dt.date:
        if v < dt.date.today():
            raise ValueError("Start date cannot be in the past")
        return v

    @model_validator(mode="after")
    def _coords(self) -> Self:
        validate_uae_coords(self.start_lat, self.start_lng)
        return self


class CostChip(BaseModel):
    """One rendered cost chip: 'N adults · AED X' / 'N children · AED Y' / 'N child free (under Z)'."""

    label: str
    count: int
    amount: float
    tone: Literal["adult", "child", "free"]


class CostBreakdown(BaseModel):
    adults: list[float] = []
    children: list[float] = []
    free_children: int = 0
    free_under_age: int | None = None
    travel_in: float = 0.0
    total: float = 0.0
    chips: list[CostChip] = []


class SlotOut(ORMModel):
    id: int
    day_index: int
    position: int
    place_id: int
    start_time: str
    end_time: str
    locked: bool
    cost_breakdown: CostBreakdown
    place: PlaceOut


class TravelSegmentOut(ORMModel):
    id: int
    day_index: int
    from_slot_id: int | None
    to_slot_id: int | None
    distance_km: float
    duration_min: int
    mode: str
    est_cost: float
    estimated: bool
    geometry_json: list | None


class DayOut(BaseModel):
    day_index: int
    date: dt.date
    theme: str
    subtotal: float
    driving_total_min: int
    slots: list[SlotOut]
    segments: list[TravelSegmentOut]


class BudgetCategorySplit(BaseModel):
    activities: float = 0.0
    food: float = 0.0
    travel: float = 0.0


class BudgetOut(BaseModel):
    total: float
    cap: float
    remaining: float
    currency: str
    over_budget: bool
    per_day: list[float]
    categories: BudgetCategorySplit


class Suggestion(BaseModel):
    """Drives the chat's action-chip row; server-decided so chips vary with state."""

    id: str
    label: str
    action: Literal["cheaper_day", "prayer_breaks"]
    day_index: int | None = None


class ItinerarySummary(ORMModel):
    id: int
    title: str
    event_id: int | None
    start_date: dt.date
    num_days: int
    total_budget: float
    currency: str
    status: str
    updated_at: dt.datetime


class ItineraryOut(BaseModel):
    id: int
    title: str
    event_id: int | None
    event_title: str | None = None
    start_date: dt.date
    num_days: int
    currency: str
    status: str
    transport_mode: Literal["taxi", "own_car"] = "taxi"
    # What this party has to travel in — derived from the family size, not stored.
    vehicle: str = "standard"
    days: list[DayOut]
    budget: BudgetOut
    suggestions: list[Suggestion]
    warnings: list[str] = []


class TransportPatch(BaseModel):
    mode: Literal["taxi", "own_car"]


class DayPatchResponse(BaseModel):
    """A slot edit returns the whole day + budget — the client never patches locally."""

    day: DayOut
    budget: BudgetOut
    suggestions: list[Suggestion]
    warnings: list[str] = []


class SlotPatch(BaseModel):
    action: Literal["replace", "adjust", "remove"]
    place_id: int | None = None
    start_time: HHMM | None = None

    @model_validator(mode="after")
    def _action_args(self) -> Self:
        if self.action == "replace" and self.place_id is None:
            raise ValueError("replace requires place_id")
        if self.action == "adjust" and self.start_time is None:
            raise ValueError("adjust requires start_time")
        return self


class AlternativeOut(BaseModel):
    place: PlaceOut
    start_time: str
    end_time: str
    cost_breakdown: CostBreakdown
    score: float


# --- conversations ---------------------------------------------------------------------------


class MessageOut(ORMModel):
    id: int
    role: str
    content: str
    created_at: dt.datetime


class ConversationOut(ORMModel):
    id: int
    title: str
    itinerary_id: int | None
    event_id: int | None
    updated_at: dt.datetime
    last_seen_at: dt.datetime
    unread: bool = False


class ConversationCreate(BaseModel):
    title: str = Field(default="New plan", max_length=200)
    event_id: int | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: int | None = None


TokenResponse.model_rebuild()
