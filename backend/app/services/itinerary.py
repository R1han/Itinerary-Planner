"""Everything between the database and the pure planner.

Responsibilities: assemble the party and preferences for a user, retrieve candidates, run the
planner, persist the result, and load persisted rows back into planner dataclasses so that edits
re-run the very same validator that generation did.

Nothing here decides scheduling — that lives in planner.py and validator.py.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..models import (
    Event,
    FamilyMember,
    Guest,
    Itinerary,
    Place,
    Preference,
    Slot,
    TravelSegment,
    User,
)
from .budget import (
    Attendee,
    CostBreakdown,
    category_bucket,
    slot_cost_breakdown,
)
from .planner import (
    DayPlan,
    PartyProfile,
    Plan,
    PlaceCandidate,
    PlannedSegment,
    PlannedSlot,
    PreferenceSignal,
    TravelInfo,
    DINING_CATEGORIES,
    DINNER_ONLY,
    FULL_DAY,
    MAX_DAYS,
    MAX_HOP_KM,
    PLAN_FOCUS,
    build_profile,
    dinner_only,
    generate_plan,
    geographic_penalty,
    haversine_km,
    rebuild_segments,
    reflow_day,
    score_place,
    theme_from_categories,
    to_minutes,
)
from .prayer import insert_prayer_breaks
from .retrieval import query_for, retrieve_candidates, to_candidate
from .travel import TAXI, TRANSPORT_MODES, TravelService, fare, vehicle_for
from .tracing import traced
from .validator import repair_plan

log = logging.getLogger(__name__)

ALTERNATIVES_COUNT = 3
CHEAPER_FACTOR = 0.65


class IntakeIncomplete(Exception):
    """Raised when generation is attempted before the intake checklist is satisfied (spec §1.2)."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        super().__init__(f"Missing intake fields: {', '.join(missing)}")


@dataclass
class PlanContext:
    """Everything the planner needs about one user, gathered once."""

    user: User
    profile: PartyProfile
    preferences: list[PreferenceSignal]
    origin: tuple[float, float]
    # None means "anywhere". Lives here rather than being passed around because every retrieval
    # in a plan's lifetime — generation, gap-filling, alternatives — must apply the same one.
    emirates: tuple[str, ...] | None = None


# --- gathering ---------------------------------------------------------------------------------


def family_attendees(db: Session, user_id: int) -> list[Attendee]:
    members = db.scalars(
        select(FamilyMember).where(FamilyMember.user_id == user_id).order_by(FamilyMember.id)
    )
    return [Attendee(role=m.role, age=m.age, name=m.name) for m in members]


def guest_attendees(db: Session, itinerary_id: int) -> list[Attendee]:
    """The non-household people on one trip. Empty for the ordinary family outing."""
    rows = db.scalars(
        select(Guest).where(Guest.itinerary_id == itinerary_id).order_by(Guest.id)
    )
    return [Attendee(role=g.role, age=g.age, name=g.name) for g in rows]


def cap_party(attendees: list[Attendee], party_size: int | None) -> list[Attendee]:
    """Trim the party to a stated total that is smaller than the household.

    `guests` handles a party BIGGER than the household; this is the other direction, which had no
    representation at all — the household was a floor, so "just the two of us" in a house of four
    was priced for four.

    ponytail: whoever comes first in the household is kept, which is insertion order. Fine while
    the trim is a headcount for pricing — four adults cost the same in any order. If it ever has
    to name WHICH two, persist the resolved roster instead of a number.
    """
    if not party_size or party_size >= len(attendees):
        return attendees
    return attendees[:party_size]


def trip_party(db: Session, itinerary: Itinerary) -> list[Attendee]:
    """Everyone this itinerary is priced for: the household, plus its guests, minus any trim.

    The single source of truth for party size. Every consumer — the vehicle tier, taxi fares,
    per-head admission — must go through here, or a plan ends up costed for the wrong number of
    people while still looking plausible.
    """
    everyone = family_attendees(db, itinerary.user_id) + guest_attendees(db, itinerary.id)
    return cap_party(everyone, itinerary.party_size)


def preference_signals(db: Session, user_id: int) -> list[PreferenceSignal]:
    prefs = db.scalars(select(Preference).where(Preference.user_id == user_id))
    return [
        PreferenceSignal(kind=p.kind, subject=p.subject, category=p.category, strength=p.strength)
        for p in prefs
    ]


def missing_intake_fields(db: Session, user: User) -> list[str]:
    """What still has to be known before an itinerary may be generated."""
    missing: list[str] = []
    attendees = family_attendees(db, user.id)
    if not any(a.role == "adult" for a in attendees):
        missing.append("adults")
    children = [a for a in attendees if a.role == "child"]
    if children and any(a.age is None for a in children):
        missing.append("children_ages")
    if not user.home_base_lat or not user.home_base_lng:
        missing.append("start_location")
    return missing


def build_context(
    db: Session, user: User, event: Event | None, *, start_lat: float, start_lng: float,
    adults_only: bool = False, focus: str = FULL_DAY,
    guests: Sequence[Attendee] = (),
    emirates: Sequence[str] | None = None,
    party_size: int | None = None,
) -> PlanContext:
    # Guests join before the adults_only filter, so "just the grown-ups" drops a guest child too.
    attendees = family_attendees(db, user.id) + list(guests)
    if adults_only:
        # An anniversary "just the two of us" is still recorded against a household with kids in
        # it. Leaving them in the party means `romantic` never fires, and the whole evening gets
        # scored for a seven-year-old.
        attendees = [a for a in attendees if a.role == "adult"] or attendees
    # Last, so a stated total counts the people who are actually coming — trimming before the
    # adults_only filter would spend the headcount on children about to be dropped.
    attendees = cap_party(attendees, party_size)
    event_type = event.event_type if event else "other"

    profile = build_profile(attendees, event_type)
    if focus == DINNER_ONLY:
        profile = dinner_only(profile)

    return PlanContext(
        user=user,
        profile=profile,
        preferences=preference_signals(db, user.id),
        origin=(start_lat, start_lng),
        emirates=tuple(emirates) if emirates else None,
    )


# --- generation --------------------------------------------------------------------------------


@traced("itinerary.generate", run_type="chain")
def generate(
    db: Session,
    user: User,
    *,
    start_date: date,
    num_days: int,
    total_budget: float,
    start_lat: float,
    start_lng: float,
    event_id: int | None = None,
    title: str | None = None,
    currency: str = "AED",
    prayer_breaks: bool = False,
    transport_mode: str = TAXI,
    adults_only: bool = False,
    focus: str = FULL_DAY,
    guests: Sequence[Attendee] = (),
    emirates: Sequence[str] | None = None,
    party_size: int | None = None,
    into: Itinerary | None = None,
) -> Itinerary:
    """Plan a trip and persist it. Raises IntakeIncomplete before doing any work.

    `into` re-solves onto an existing itinerary row instead of creating one. The stops are
    replaced — that is what the caller asked for — but the row's identity survives, so the
    conversation pointing at it and the event it was planned for stay attached. Building a fresh
    row and abandoning the old one is what left a rebuilt plan belonging to no thread.
    """
    missing = missing_intake_fields(db, user)
    if missing:
        raise IntakeIncomplete(missing)

    event = None
    if event_id is not None:
        event = db.scalars(
            select(Event).where(Event.id == event_id, Event.user_id == user.id)
        ).one_or_none()
        if event is None:
            raise ValueError("Unknown event")

    # A dinner is one evening whatever the request said.
    if focus == DINNER_ONLY:
        num_days = 1

    context = build_context(
        db, user, event, start_lat=start_lat, start_lng=start_lng,
        adults_only=adults_only, focus=focus, guests=guests, emirates=emirates,
        party_size=party_size,
    )
    candidates = retrieve_candidates(
        db,
        context.profile,
        query_for(context.profile, event.title if event else "", event.notes if event else ""),
        origin=context.origin,
        emirates=context.emirates,
    )

    travel_service = TravelService(
        db, mode=transport_mode, party_size=len(context.profile.attendees)
    )
    travel_fn = travel_service.travel_fn(list(candidates))

    plan = generate_plan(
        candidates,
        context.profile,
        # Candidates are ranked on cheap estimates; the chosen legs are routed for real below.
        travel_service.estimate_fn(),
        start_date=start_date,
        num_days=num_days,
        total_budget=total_budget,
        origin=context.origin,
        preferences=context.preferences,
        currency=currency,
    )

    plan = pin_event_venue(db, plan, event, context, travel_fn)
    plan = route_for_real(plan, context, travel_fn)

    if prayer_breaks:
        for day in plan.days:
            insert_prayer_breaks(day, travel_fn, context.origin)
        plan = repair_plan(plan, context.profile, travel_fn, context.origin)

    if into is None:
        itinerary = Itinerary(
            user_id=user.id,
            event_id=event.id if event else None,
            title=title or (event.title if event else "UAE trip"),
        )
        db.add(itinerary)
    else:
        itinerary = into
        # Silence keeps what the row already has: re-solving a plan into Abu Dhabi says nothing
        # about which event it is for or what it is called, and inventing "UAE trip" over the
        # user's own title would lose more than the stops did.
        if event is not None:
            itinerary.event_id = event.id
        if title:
            itinerary.title = title
        elif event is not None:
            itinerary.title = event.title
        # The party is restated on every solve, so leaving the old rows would double it.
        db.query(Guest).filter(Guest.itinerary_id == itinerary.id).delete()

    itinerary.start_date = start_date
    itinerary.num_days = num_days
    itinerary.total_budget = total_budget
    itinerary.currency = currency
    itinerary.status = "ready"
    itinerary.start_lat = start_lat
    itinerary.start_lng = start_lng
    itinerary.transport_mode = transport_mode
    itinerary.emirates_json = list(context.emirates) if context.emirates else None
    # Only a trim is worth storing. Recording the full headcount would freeze the plan against
    # the household it was solved for, so adding a child would stop repricing the trip.
    itinerary.party_size = party_size if party_size and party_size < len(
        family_attendees(db, user.id) + list(guests)
    ) else None
    db.flush()

    # Stored so that everything reached later — a transport switch, a cheaper day, the payload's
    # vehicle tier — re-prices for the same party this plan was solved for.
    for guest in guests:
        db.add(
            Guest(
                itinerary_id=itinerary.id, role=guest.role, age=guest.age, name=guest.name
            )
        )

    for day in plan.days:
        for slot in day.slots:
            slot.locked = False

    persist_plan(db, itinerary, plan)
    if event is not None:
        event.planned = True
    db.commit()
    db.refresh(itinerary)
    return itinerary


def pin_event_venue(
    db: Session, plan: Plan, event: Event | None, context: PlanContext, travel_fn
) -> Plan:
    """Put the event's own venue into the day the event falls on.

    A concert on the 20th is the reason for the trip, so the planner should not be free to score
    it away. The slot is locked, which means the repair pass trims around it rather than dropping
    it — the opposite of how every other slot is treated.
    """
    if event is None or event.place_id is None or not plan.days:
        return plan

    day_index = (event.date - plan.days[0].day_date).days
    if not 0 <= day_index < len(plan.days):
        return plan

    day = plan.days[day_index]
    if any(slot.place.id == event.place_id for slot in day.slots):
        return plan

    row = db.get(Place, event.place_id)
    if row is None:
        return plan

    venue = to_candidate(row)
    if not all(person.age >= venue.min_age for person in context.profile.attendees):
        log.info("event venue %s excluded: the party does not clear min_age", venue.name)
        return plan

    start = max(venue.opens_at, context.profile.day_start)
    duration = min(venue.avg_duration_min, context.profile.max_slot_min)
    if start + duration > venue.closes_at:
        return plan

    day.slots.append(
        PlannedSlot(
            place=venue,
            day_index=day_index,
            position=len(day.slots),
            start_min=start,
            end_min=start + duration,
            score=99.0, 
            cost=slot_cost_breakdown(venue, context.profile.attendees),
            locked=True,
        )
    )
    reflow_day(day, context.profile, travel_fn, context.origin)
    return plan


def route_for_real(plan: Plan, context: PlanContext, travel_fn) -> Plan:
    """Replace the planner's estimates with real routed legs, then re-validate.

    A real route is usually a little slower than the estimate, so the day can end up infeasible —
    which is exactly what the validator and repair pass are for.
    """
    for day in plan.days:
        rebuild_segments(day, travel_fn, context.origin)
    return repair_plan(plan, context.profile, travel_fn, context.origin)


def persist_plan(db: Session, itinerary: Itinerary, plan: Plan) -> None:
    """Write a plan to the database, updating slots in place rather than recreating them.

    Slot ids must survive an edit: the client holds them for hover and selection linking the strip
    to the map, and a delete-and-reinsert would silently hand a surviving slot a neighbour's id.
    """
    existing = {
        row.id: row
        for row in db.scalars(select(Slot).where(Slot.itinerary_id == itinerary.id))
    }

    # Segments are fully derived from the slots, so they are always rebuilt. They also hold FKs to
    # slots, so they must go before any slot is deleted.
    db.query(TravelSegment).filter(TravelSegment.itinerary_id == itinerary.id).delete()
    db.flush()

    surviving: set[int] = set()
    for day in plan.days:
        position_to_row: dict[int, Slot] = {}
        for slot in sorted(day.slots, key=lambda s: s.start_min):
            row = existing.get(slot.row_id) if slot.row_id else None
            if row is None:
                row = Slot(itinerary_id=itinerary.id, place_id=slot.place.id)
                db.add(row)

            row.day_index = day.day_index
            row.position = slot.position
            row.place_id = slot.place.id
            row.start_time = slot.start_time
            row.end_time = slot.end_time
            row.cost_breakdown_json = slot.cost.as_dict()
            row.locked = slot.locked
            db.flush()

            slot.row_id = row.id
            surviving.add(row.id)
            position_to_row[slot.position] = row

        db.flush()

        for segment in day.segments:
            db.add(
                TravelSegment(
                    itinerary_id=itinerary.id,
                    day_index=day.day_index,
                    from_slot_id=(
                        position_to_row[segment.from_position].id
                        if segment.from_position is not None
                        and segment.from_position in position_to_row
                        else None
                    ),
                    to_slot_id=(
                        position_to_row[segment.to_position].id
                        if segment.to_position in position_to_row
                        else None
                    ),
                    distance_km=segment.info.distance_km,
                    duration_min=segment.info.duration_min,
                    mode="driving-car",
                    est_cost=segment.info.est_cost,
                    estimated=segment.info.estimated,
                    geometry_json=segment.info.geometry,
                )
            )

    for row_id, row in existing.items():
        if row_id not in surviving:
            db.delete(row)
    db.flush()


# --- loading persisted rows back into planner dataclasses --------------------------------------


def load_plan(db: Session, itinerary: Itinerary) -> Plan:
    """Rebuild a Plan from persisted rows, so edits re-run the same validator as generation."""
    slots = list(
        db.scalars(
            select(Slot)
            .where(Slot.itinerary_id == itinerary.id)
            .order_by(Slot.day_index, Slot.position)
        )
    )
    segments = list(
        db.scalars(select(TravelSegment).where(TravelSegment.itinerary_id == itinerary.id))
    )

    slot_by_id = {row.id: row for row in slots}
    plan = Plan(
        total_budget=itinerary.total_budget,
        currency=itinerary.currency,
    )

    days: dict[int, DayPlan] = {}
    for index in range(itinerary.num_days):
        days[index] = DayPlan(
            day_index=index, day_date=itinerary.start_date + timedelta(days=index)
        )

    for row in slots:
        day = days.setdefault(
            row.day_index, DayPlan(day_index=row.day_index, day_date=itinerary.start_date)
        )
        cost = CostBreakdown(**{k: v for k, v in (row.cost_breakdown_json or {}).items()})
        day.slots.append(
            PlannedSlot(
                place=to_candidate(row.place),
                day_index=row.day_index,
                position=row.position,
                start_min=to_minutes(row.start_time),
                end_min=to_minutes(row.end_time),
                score=0.0,
                cost=cost,
                locked=row.locked,
                row_id=row.id,
            )
        )

    for row in segments:
        day = days.get(row.day_index)
        if day is None:
            continue
        to_slot = slot_by_id.get(row.to_slot_id) if row.to_slot_id else None
        from_slot = slot_by_id.get(row.from_slot_id) if row.from_slot_id else None
        if to_slot is None:
            continue
        day.segments.append(
            PlannedSegment(
                day_index=row.day_index,
                from_position=from_slot.position if from_slot else None,
                to_position=to_slot.position,
                info=TravelInfo(
                    distance_km=row.distance_km,
                    duration_min=row.duration_min,
                    est_cost=row.est_cost,
                    estimated=row.estimated,
                    geometry=row.geometry_json,
                ),
            )
        )

    plan.days = [days[index] for index in sorted(days)]
    return plan


def context_for(db: Session, itinerary: Itinerary, user: User) -> PlanContext:
    event = db.get(Event, itinerary.event_id) if itinerary.event_id else None
    if event is not None and event.user_id != user.id:  # defence in depth
        event = None
    return build_context(
        db, user, event, start_lat=itinerary.start_lat, start_lng=itinerary.start_lng,
        guests=guest_attendees(db, itinerary.id),
        emirates=itinerary.emirates_json,
        party_size=itinerary.party_size,
    )


# --- payload assembly --------------------------------------------------------------------------


def _travel_service(db: Session, itinerary: Itinerary, context: PlanContext) -> TravelService:
    """A travel service that prices legs for this plan's transport mode and this party's size."""
    return TravelService(
        db, mode=itinerary.transport_mode, party_size=len(context.profile.attendees)
    )


@traced("itinerary.recost_travel", run_type="chain")
def recost_travel(db: Session, itinerary: Itinerary) -> None:
    """Re-price the stored legs after the transport mode changed.

    Deliberately not a re-plan: the route, the times and the places are all still right, and
    re-solving them would move the trip when the user only said how they intend to get around.
    Only `est_cost` changes, recomputed from the distance already on the row.
    """
    party = len(trip_party(db, itinerary)) or 1
    segments = db.scalars(
        select(TravelSegment).where(TravelSegment.itinerary_id == itinerary.id)
    ).all()
    for segment in segments:
        segment.est_cost = fare(
            segment.distance_km,
            mode=itinerary.transport_mode,
            party_size=party,
            # Every leg but a day's last one arrives somewhere that has to be parked at.
            arriving_stops=1 if segment.to_slot_id is not None else 0,
        )
    db.commit()


def itinerary_payload(db: Session, itinerary: Itinerary) -> dict:
    """The full GET /itineraries/{id} body, built from persisted rows.

    Everything the workspace needs to render in one response: geometry per segment and image_url
    per slot's place, so the map and strip need no extra round trips (spec §9).
    """
    slot_rows = list(
        db.scalars(
            select(Slot)
            .where(Slot.itinerary_id == itinerary.id)
            .order_by(Slot.day_index, Slot.position)
        )
    )
    segment_rows = list(
        db.scalars(
            select(TravelSegment)
            .where(TravelSegment.itinerary_id == itinerary.id)
            .order_by(TravelSegment.day_index, TravelSegment.id)
        )
    )

    by_day: dict[int, list[Slot]] = {}
    for row in slot_rows:
        by_day.setdefault(row.day_index, []).append(row)
    segments_by_day: dict[int, list[TravelSegment]] = {}
    for row in segment_rows:
        segments_by_day.setdefault(row.day_index, []).append(row)

    days: list[dict] = []
    per_day_totals: list[float] = []
    activities = food = travel = 0.0

    for index in range(itinerary.num_days):
        day_slots = by_day.get(index, [])
        day_segments = segments_by_day.get(index, [])

        subtotal = 0.0
        for row in day_slots:
            breakdown = row.cost_breakdown_json or {}
            admission = sum(breakdown.get("adults", [])) + sum(breakdown.get("children", []))
            subtotal += admission
            if category_bucket(row.place.category) == "food":
                food += admission
            else:
                activities += admission
        for row in day_segments:
            subtotal += row.est_cost
            travel += row.est_cost

        per_day_totals.append(round(subtotal, 2))
        days.append(
            {
                "day_index": index,
                "date": itinerary.start_date + timedelta(days=index),
                "theme": theme_from_categories(
                    [
                        (row.place.category, to_minutes(row.end_time) - to_minutes(row.start_time))
                        for row in day_slots
                    ]
                ),
                "subtotal": round(subtotal, 2),
                "driving_total_min": sum(row.duration_min for row in day_segments),
                "slots": [
                    {
                        "id": row.id,
                        "day_index": row.day_index,
                        "position": row.position,
                        "place_id": row.place_id,
                        "start_time": row.start_time,
                        "end_time": row.end_time,
                        "locked": row.locked,
                        "cost_breakdown": row.cost_breakdown_json or {},
                        "place": row.place,
                    }
                    for row in day_slots
                ],
                "segments": day_segments,
            }
        )

    total = round(sum(per_day_totals), 2)
    event = db.get(Event, itinerary.event_id) if itinerary.event_id else None

    party = trip_party(db, itinerary)
    return {
        "id": itinerary.id,
        "title": itinerary.title,
        "event_id": itinerary.event_id,
        "event_title": event.title if event else None,
        "start_date": itinerary.start_date,
        "num_days": itinerary.num_days,
        "currency": itinerary.currency,
        "status": itinerary.status,
        "transport_mode": itinerary.transport_mode,
        "vehicle": vehicle_for(len(party) or 1)[0],
        "emirates": itinerary.emirates_json,
        "party_size": len(party),
        "days": days,
        "budget": {
            "total": total,
            "cap": round(itinerary.total_budget, 2),
            "remaining": round(itinerary.total_budget - total, 2),
            "currency": itinerary.currency,
            "over_budget": total > itinerary.total_budget + 0.01,
            "per_day": per_day_totals,
            "categories": {
                "activities": round(activities, 2),
                "food": round(food, 2),
                "travel": round(travel, 2),
            },
        },
        "suggestions": suggestions_from_rows(per_day_totals),
        "warnings": [],
    }


def day_payload(db: Session, itinerary: Itinerary, day_index: int) -> dict:
    """One day of the payload above — what a slot edit returns."""
    full = itinerary_payload(db, itinerary)
    day = next((d for d in full["days"] if d["day_index"] == day_index), None)
    return {
        "day": day,
        "budget": full["budget"],
        "suggestions": full["suggestions"],
        "warnings": full["warnings"],
    }


def suggestions_from_rows(per_day_totals: list[float]) -> list[dict]:
    """Server-decided action chips, so they vary with the plan's actual state (design, chat pane)."""
    chips: list[dict] = []
    spending_days = [(index, total) for index, total in enumerate(per_day_totals) if total > 0]
    if not spending_days:
        return chips

    priciest = max(spending_days, key=lambda pair: pair[1])[0]
    return chips


# --- editing -----------------------------------------------------------------------------------


def _candidates_for_gap(
    db: Session,
    context: PlanContext,
    exclude_place_ids: set[int],
) -> list[PlaceCandidate]:
    candidates = retrieve_candidates(
        db, context.profile, query_for(context.profile), origin=context.origin,
        emirates=context.emirates,
    )
    return [c for c in candidates if c.id not in exclude_place_ids]


def _meals_eaten(profile: PartyProfile, day: DayPlan) -> str:
    """"a lunch and a dinner" — what a day has already booked, for a refusal that explains."""
    eaten = [
        f"a {role}"
        for slot in sorted(day.slots, key=lambda s: s.start_min)
        if slot.place.category in DINING_CATEGORIES
        and (role := meal_role(profile, slot.start_min, slot.end_min))
    ]
    if not eaten:
        return "no meals"
    return " and ".join([", ".join(eaten[:-1]), eaten[-1]] if len(eaten) > 1 else eaten)


def meal_role(profile: PartyProfile, start_min: int, end_min: int) -> str | None:
    """Which meal this time range would be — "lunch", "dinner", or None for neither.

    `profile.meal_windows` is what the generator plans around, so reading it here is what stops
    an edit from quietly dismantling the structure generation built.
    """
    for label, opens, closes in profile.meal_windows:
        if start_min < closes and end_min > opens:
            return label
    return None


def free_meal_windows(
    profile: PartyProfile, day: DayPlan, *, ignore_position: int | None = None
) -> set[str]:
    """The meals this day has not eaten yet, ignoring one slot that is about to change."""
    taken = {
        role
        for slot in day.slots
        if slot.position != ignore_position
        and slot.place.category in DINING_CATEGORIES
        and (role := meal_role(profile, slot.start_min, slot.end_min))
    }
    return {label for label, _, _ in profile.meal_windows} - taken


def _placements_in_day(
    context: PlanContext,
    day: DayPlan,
    candidates: list[PlaceCandidate],
    *,
    travel_fn,
    spare: float,
    free_meals: set[str],
    allow_shift: bool = False,
) -> list[tuple[float, int, PlaceCandidate, Placement, CostBreakdown]]:
    """Every way a candidate could sit in this day, ranked by what it costs the day.

    Every insertion point: before the first stop, between each pair, and after the last. Shared
    by "add a stop" and by a replace that has to re-time the day, so the two can never disagree
    about what fits where.

    Dining is the one category that is not free to land anywhere. The generator treats it as a
    role — `assemble_day` builds around meal windows and refuses dining when filling an activity
    slot — and edits used to treat it as a plain category, which is how a day ended up with three
    restaurants in a row. A restaurant goes in a meal window that is still free, or nowhere.
    """
    ordered = sorted(day.slots, key=lambda s: s.start_min)
    considered: list[tuple[float, int, PlaceCandidate, Placement, CostBreakdown]] = []

    for index in range(len(ordered) + 1):
        previous = ordered[index - 1] if index > 0 else None
        following = ordered[index] if index < len(ordered) else None
        from_point = (previous.place.lat, previous.place.lng) if previous else context.origin
        earliest = previous.end_min if previous else context.profile.day_start
        latest = following.start_min if following else context.profile.day_end

        for candidate in candidates:
            # `allow_shift` stops the NEXT stop being a wall: it may move later, which is what
            # "the day re-times around it" means. The day's own end still binds, so re-timing
            # cannot quietly turn a day out into a night out.
            fit = placement_for(
                candidate, context, travel_fn,
                earliest=earliest,
                latest=context.profile.day_end if allow_shift else latest,
                from_point=from_point,
                following=None if allow_shift else following,
                day_month=day.day_date.month,
                from_origin=previous is None,
            )
            if fit is None:
                continue
            if candidate.category in DINING_CATEGORIES:
                role = meal_role(context.profile, fit.start_min, fit.end_min)
                if role is None or role not in free_meals:
                    continue
            cost = slot_cost_breakdown(candidate, context.profile.attendees, fit.inbound.est_cost)
            if cost.total > spare:
                continue
            # Prefer the stop that adds the least travel, then the better-scoring place.
            detour = geographic_penalty(candidate, from_point, context.origin)
            rank = detour - 20.0 * score_place(candidate, context.profile, context.preferences)
            considered.append((rank, index, candidate, fit, cost))

    return considered


# The words people use for a category where they differ from the category itself. Not a synonym
# engine — just the handful the transcripts actually produced. "shopping" for a `mall` is the one
# that went wrong three times running.
_CATEGORY_WORDS = {"shopping": "mall", "restaurant": "dining", "food": "dining", "leisure": "park"}

# Which sitting a meal word names, resolved against the clock.
#
# Deliberately separate from `PartyProfile.meal_windows`, which is what the generator schedules
# around: those are narrow, per-party, and know only lunch and dinner. These are wide contiguous
# buckets whose only job is to tell one day's sittings apart from each other once the user has
# named one, so they cover the whole day and include breakfast — which the prompt has always told
# the model to use and which nothing here could resolve.
MEAL_SITTINGS: dict[str, tuple[int, int]] = {
    "breakfast": (0, 11 * 60),
    "lunch": (11 * 60, 16 * 60),
    "dinner": (16 * 60, 24 * 60),
}


def find_stop(db: Session, itinerary: Itinerary, description: str, *, day: int | None = None) -> Slot:
    """Which stop the user meant, from the words they used for it.

    The chat has no slot ids to offer — they exist only in the database, so a model that has not
    just read the plan can only invent one, and did. It has the user's own phrasing instead
    ("the shopping stop", "Shakespeare and Co", "the park at the end"), which is the thing the
    server can actually resolve, because the conversation already says which plan is meant.

    `day` (1-based) scopes the search to one day, for when the same place name shows up more than
    once in the plan — "day 4 dinner" is unambiguous even when "dinner" alone is not.

    Raises with the plan's real contents listed, so a miss is answerable without another lookup.
    """
    slots = (
        db.query(Slot)
        .filter(Slot.itinerary_id == itinerary.id)
        .order_by(Slot.day_index, Slot.position)
        .all()
    )
    if not slots:
        raise ValueError("This plan has no stops in it yet.")

    if day is not None:
        scoped = [s for s in slots if s.day_index == day - 1]
        if not scoped:
            # A wrong guess here used to read as "the plan has no stops at all" and send the
            # model straight for a rebuild. Naming the days that DO have stops lets it retry
            # with the right one (or drop `day` outright) instead of concluding there's nothing
            # to edit.
            real_days = sorted({s.day_index + 1 for s in slots})
            raise ValueError(
                f"Day {day} has no stops. This plan's stops are on day(s) {real_days} — "
                "retry with one of those, or drop `day` and let the stop's name resolve it."
            )
        slots = scoped

    text = " ".join(description.lower().split())
    if not text:
        # Empty matched every stop through the substring test below, which reads as ambiguity
        # when it is really an omission.
        raise ValueError("Say which stop to change — its name, or what kind of place it is.")
    for word, category in _CATEGORY_WORDS.items():
        text = text.replace(word, category)

    def listing() -> str:
        return "; ".join(
            f"{slot.place.name} ({slot.place.category.replace('_', ' ')}, day {slot.day_index + 1})"
            for slot in slots
        )

    # A name beats a category: "Shakespeare and Co" is a specific ask, "dining" is a description
    # of two of them, and someone who names a place has told you which one they mean.
    for match in (
        [s for s in slots if text and (text in s.place.name.lower() or s.place.name.lower() in text)],
        [s for s in slots
         if (label := s.place.category.replace("_", " ")) in text or text in label],
    ):
        if len(match) == 1:
            return match[0]
        if match:
            # The same place can sit twice on one day (a lunch and a dinner at the same
            # restaurant) — a name or category match alone can't tell those apart, but a meal word
            # in the user's own words can. Matched against the clock rather than by position: with
            # three sittings "lunch" is the middle one, and taking the earliest handed back
            # breakfast instead.
            meal = next((word for word in MEAL_SITTINGS if word in text), None)
            if meal is not None:
                opens, closes = MEAL_SITTINGS[meal]
                sitting = [s for s in match if opens <= to_minutes(s.start_time) < closes]
                if len(sitting) == 1:
                    return sitting[0]
                if not sitting:
                    raise ValueError(
                        f"None of these is a {meal} sitting — "
                        + "; ".join(
                            f"{s.place.name} at {s.start_time} on day {s.day_index + 1}"
                            for s in match
                        )
                        + ". Say which one by its time, or which day."
                    )
                # Narrowed but still ambiguous: report the sittings that survived, not all of them.
                match = sitting
            raise ValueError(
                f"{description!r} matches more than one stop — "
                + "; ".join(
                    f"{s.place.name} at {s.start_time} on day {s.day_index + 1}" for s in match
                )
                + ". Say which one, or which day."
            )

    raise ValueError(f"This plan has no stop like {description!r}. It has: {listing()}.")


def find_catalog_place(db: Session, description: str, *, category: str | None = None) -> Place:
    """Which catalog place the user meant, from the words they used for it.

    Mirrors find_stop's matching, but over the whole catalog rather than one plan's slots — for
    add_stop/edit_stop callers that named an exact place ("add the UAQ Mangrove Kayak") instead of
    a category. Without this, naming a specific place is indistinguishable from naming its
    category, and a category only ever gets the best FIT, never the place actually asked for.
    """
    text = " ".join(description.lower().split())
    if not text:
        raise ValueError("Say which place — its name, the way find_places or the plan showed it.")

    statement = select(Place)
    if category:
        statement = statement.where(Place.category == category)
    matches = [
        place for place in db.scalars(statement)
        if text in place.name.lower() or place.name.lower() in text
    ]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ValueError(
            f"{description!r} matches more than one place — "
            + "; ".join(f"{p.name} ({p.emirate})" for p in matches)
            + ". Say which one."
        )
    raise ValueError(f"No place in the catalog matches {description!r}. Try find_places to look it up.")


def gap_window(day: DayPlan, position: int) -> tuple[int, int]:
    """The time window a replacement slot must fit into: between its two neighbours."""
    ordered = sorted(day.slots, key=lambda s: s.start_min)
    index = next((i for i, s in enumerate(ordered) if s.position == position), None)
    if index is None:
        return (0, 24 * 60)
    earliest = ordered[index - 1].end_min if index > 0 else 0
    latest = ordered[index + 1].start_min if index + 1 < len(ordered) else 24 * 60 - 1
    return (earliest, latest)


@dataclass(frozen=True)
class Placement:
    """Where a candidate would sit between two neighbours, and what getting there costs."""

    start_min: int
    end_min: int
    inbound: TravelInfo


def placement_for(
    candidate: PlaceCandidate,
    context: PlanContext,
    travel_fn,
    *,
    earliest: int,
    latest: int,
    from_point: tuple[float, float],
    following: PlannedSlot | None,
    day_month: int,
    ignore_window: bool = False,
    from_origin: bool = False,
) -> Placement | None:
    """Fit one candidate into one gap, or None if it cannot go there.

    Shared by "what else could this slot be" and "put something new in this day" — the two only
    differ in what they do with the answer, and having one implementation of *fits* means a rule
    added here (the hop cap, an age gate) can never apply to one and not the other.

    `ignore_window` drops only the day's own boundaries — the gap between the neighbours and
    `day_end`. It is safe because `validate_plan` never checks `day_end`: a long stop pushes the
    rest of the day later via `reflow_day` instead of being rejected. Everything that would make
    the plan *wrong* rather than *late* — opening hours, the age gate, the hop cap, and the
    budget the caller applies — still holds.
    """
    if not all(person.age >= candidate.min_age for person in context.profile.attendees):
        return None
    if not candidate.open_in_month(day_month):
        return None
    # The cap is on the hop BETWEEN stops — "this far from the last one is a different trip, not
    # the next thing to do". The drive from home is not that hop, and `assemble_day` has always
    # exempted it (`previous_position is not None`). Applying it here anyway meant no edit could
    # ever put a first stop where generation had happily put one: an Abu Dhabi day built 130 km
    # from home could not have its opening stop swapped, added to, or re-timed into.
    if not from_origin and haversine_km(
        from_point[0], from_point[1], candidate.lat, candidate.lng
    ) > MAX_HOP_KM:
        return None

    inbound = travel_fn(from_point[0], from_point[1], candidate.lat, candidate.lng)
    start = max(earliest + inbound.duration_min, candidate.opens_at)
    end = start + min(candidate.avg_duration_min, context.profile.max_slot_min)

    if end > candidate.closes_at:
        return None
    if not ignore_window:
        if end > context.profile.day_end:
            return None
        if following is not None:
            outbound = travel_fn(
                candidate.lat, candidate.lng, following.place.lat, following.place.lng
            )
            if end + outbound.duration_min > following.start_min:
                return None
        elif end > latest:
            return None

    return Placement(start_min=start, end_min=end, inbound=inbound)


@traced("itinerary.alternatives", run_type="chain")
def alternatives_for_slot(
    db: Session,
    itinerary: Itinerary,
    user: User,
    slot_row: Slot,
    limit: int = ALTERNATIVES_COUNT,
    ignore_window: bool = False,
    ignore_budget: bool = False,
) -> list[dict]:
    """Three options that fit the exact window and the remaining budget (spec §6).

    The two `ignore_*` flags exist for diagnosis, not for planning: relaxing one at a time is how
    a caller finds out *which* constraint emptied the list, instead of guessing at the answer.
    """
    context = context_for(db, itinerary, user)
    plan = load_plan(db, itinerary)
    day = plan.days[slot_row.day_index]

    travel_service = _travel_service(db, itinerary, context)
    booked = {s.place.id for d in plan.days for s in d.slots}
    candidates = _candidates_for_gap(db, context, booked)
    # Offering three options must not cost dozens of route lookups; the real leg is routed when
    # one is actually chosen, in patch_slot.
    travel_fn = travel_service.estimate_fn()

    earliest, latest = gap_window(day, slot_row.position)
    ordered = sorted(day.slots, key=lambda s: s.start_min)
    index = next((i for i, s in enumerate(ordered) if s.position == slot_row.position), None)
    previous = ordered[index - 1] if index and index > 0 else None
    following = ordered[index + 1] if index is not None and index + 1 < len(ordered) else None

    current_cost = next(
        (s.cost.total for s in day.slots if s.position == slot_row.position), 0.0
    )
    spare = plan.total_budget - plan.total_cost + current_cost

    origin = (previous.place.lat, previous.place.lng) if previous else context.origin
    scored = sorted(
        candidates, key=lambda c: score_place(c, context.profile, context.preferences), reverse=True
    )

    options: list[dict] = []
    for candidate in scored:
        fit = placement_for(
            candidate, context, travel_fn,
            earliest=earliest, latest=latest, from_point=origin, following=following,
            day_month=day.day_date.month, ignore_window=ignore_window,
            from_origin=previous is None,
        )
        if fit is None:
            continue
        start, end = fit.start_min, fit.end_min

        cost = slot_cost_breakdown(candidate, context.profile.attendees, fit.inbound.est_cost)
        if cost.total > spare and not ignore_budget:
            continue

        options.append(
            {
                # The DB row, not the planner's candidate — the response schema needs the full
                # place record (image_url, description) that the popover and card render.
                "place": db.get(Place, candidate.id),
                "start_time": f"{start // 60:02d}:{start % 60:02d}",
                "end_time": f"{end // 60:02d}:{end % 60:02d}",
                "cost_breakdown": cost.as_dict(),
                "score": score_place(candidate, context.profile, context.preferences),
            }
        )
        if len(options) >= limit:
            break

    return options


class WindowOverrunRequired(ValueError):
    """Nothing of this kind fits the slot, but something does if the day is allowed to run late.

    A ValueError so every `except ValueError` already wrapped around `patch_slot` still catches
    it; the subclass exists so a caller that can ask the user gets the choice — "the only kayak
    tour ends at 18:30, an hour past this slot, shall I take it?" — instead of a flat refusal.
    """

    def __init__(self, category: str, place_name: str, ends_at: str) -> None:
        super().__init__(
            f"{place_name} is the only {category} available here, but it runs to {ends_at}, "
            f"past this slot's window. Ask the user whether the day may run later, then retry "
            f"with allow_overrun=true."
        )
        self.category = category
        self.place_name = place_name
        self.ends_at = ends_at


def _retimed_placement(
    db: Session,
    context: PlanContext,
    plan: Plan,
    day: DayPlan,
    target: PlannedSlot,
    category: str,
    travel_service,
    free_meals: set[str],
) -> tuple[PlaceCandidate, Placement, CostBreakdown] | None:
    """The best place of this kind anywhere in the day, with the target slot free to move.

    The target is lifted out for the search — it is the slot being replaced, so its hours are not
    a constraint on its own replacement — and put straight back. Nothing is persisted from here;
    the caller either applies the answer or raises with the plan untouched.
    """
    # The target's own place stays in `booked`: a replace that reselects what is already there
    # is not a replace.
    booked = {slot.place.id for d in plan.days for slot in d.slots}
    candidates = [
        c for c in _candidates_for_gap(db, context, booked) if c.category == category
    ]
    if not candidates:
        return None

    day.slots.remove(target)
    try:
        considered = _placements_in_day(
            context, day, candidates,
            travel_fn=travel_service.estimate_fn(),
            spare=plan.total_budget - plan.total_cost,
            free_meals=free_meals,
            allow_shift=True,
        )
    finally:
        day.slots.append(target)
        day.slots.sort(key=lambda slot: slot.start_min)

    if not considered:
        return None
    _, _, candidate, fit, cost = min(considered, key=lambda row: row[0])
    return candidate, fit, cost


class DayReorderRequired(ValueError):
    """Nothing of this kind can occupy that slot's hours, but something can if the day re-times.

    A ValueError like its sibling, so handlers already wrapped around `patch_slot` still catch
    it. What it carries is the distinction the old refusal could not make: a stop's place in the
    clock is not what the user asked about. "Swap the shopping for an adventure" is a request
    about the day, and every adventure in range being shut at 20:35 is a fact about the hour.
    """

    def __init__(self, category: str, place_name: str, duration_min: int) -> None:
        hours = duration_min / 60
        length = f"{hours:.0f} hours" if hours >= 1.5 else f"{duration_min} minutes"
        super().__init__(
            f"Nothing of that kind is open at this stop's hour. {place_name} works, but it runs "
            f"{length} and has to sit earlier in the day, so the stops after it shift later. Ask "
            f"the user whether the schedule may move, then retry with allow_reorder=true."
        )
        self.category = category
        self.place_name = place_name
        self.duration_min = duration_min


def _best_alternative(
    db: Session,
    itinerary: Itinerary,
    user: User,
    slot_row: Slot,
    category: str,
    ignore_window: bool = False,
    ignore_budget: bool = False,
) -> dict | None:
    """The highest-scoring swap of a given kind that fits this slot's window and budget.

    Lets a request stay in the user's own words — "replace the park with shopping" — instead of
    making the assistant fetch a list and pick an id out of it.
    """
    options = [
        option
        for option in alternatives_for_slot(
            db, itinerary, user, slot_row,
            limit=ALTERNATIVES_COUNT * 4,
            ignore_window=ignore_window, ignore_budget=ignore_budget,
        )
        if option["place"].category == category
    ]
    return options[0] if options else None


@traced("itinerary.patch_slot", run_type="chain")
def patch_slot(
    db: Session,
    itinerary: Itinerary,
    user: User,
    slot_row: Slot,
    *,
    action: str,
    place_id: int | None = None,
    start_time: str | None = None,
    category: str | None = None,
    allow_overrun: bool = False,
    allow_reorder: bool = False,
) -> tuple[Plan, PlanContext, int]:
    """Replace / adjust / remove one slot, re-solving only that gap.

    Every other slot is locked first, so the repair pass can trim the edited day without
    cascading into slots the user did not touch. Returns the whole re-validated plan.
    """
    context = context_for(db, itinerary, user)
    plan = load_plan(db, itinerary)
    day_index = slot_row.day_index
    day = plan.days[day_index]
    position = slot_row.position

    # Lock everything the user did not touch, so the repair pass can only ever resolve a problem
    # by moving or dropping the edited slot itself — an edit must not cascade into other days.
    for other_day in plan.days:
        for slot in other_day.slots:
            slot.locked = True

    target = next((s for s in day.slots if s.position == position), None)
    if target is None:
        raise ValueError("Slot not found in the loaded plan")
    stops_before = len(day.slots)
    # Raised only once the reflow has proved the re-time actually works. Asking first and
    # refusing after the user says yes is worse than not asking.
    pending_reorder: DayReorderRequired | None = None

    travel_service = _travel_service(db, itinerary, context)

    if action == "remove":
        day.slots.remove(target)

    elif action == "adjust":
        if start_time is None:
            raise ValueError("adjust requires start_time")
        duration = target.end_min - target.start_min
        target.start_min = to_minutes(start_time)
        target.end_min = target.start_min + duration

    elif action == "replace":
        # Set when the day was re-timed around the new stop, which places it itself — the common
        # tail below would otherwise drag it back to the outgoing stop's hour.
        retimed_in_place = False

        if place_id is None:
            if not category:
                raise ValueError("replace requires place_id or category")
            label = category.replace("_", " ")

            # Dining is a role, not a slot filler: a restaurant belongs in a meal window this day
            # has not used yet. Checked before searching, so an 8pm show is not swapped for a
            # second dinner merely because a restaurant happens to be open at 8pm.
            free_meals = free_meal_windows(context.profile, day, ignore_position=position)
            fits_a_meal = (
                category not in DINING_CATEGORIES
                or meal_role(context.profile, target.start_min, target.end_min) in free_meals
            )

            def here(**relaxed):
                return (
                    _best_alternative(db, itinerary, user, slot_row, category, **relaxed)
                    if fits_a_meal
                    else None
                )

            chosen = here()
            if chosen is not None:
                place_id = chosen["place"].id
            elif (by_window := here(ignore_window=True)) is not None:
                # "Nothing fits" is five different answers wearing one coat — the window, the
                # budget, the age gate, the distance cap, or a catalog with none of that kind.
                # Naming the wrong one sends the user off adjusting something that was never the
                # problem, so relax one constraint at a time and let whichever unblocks the
                # search name itself.
                if not allow_overrun:
                    raise WindowOverrunRequired(
                        label, by_window["place"].name, by_window["end_time"]
                    )
                place_id = by_window["place"].id
            elif (by_budget := here(ignore_budget=True)) is not None:
                # Same arithmetic as `alternatives_for_slot`: what is left once this slot's own
                # cost is handed back.
                spare = plan.total_budget - plan.total_cost + target.cost.total
                raise ValueError(
                    f"The cheapest {label} that fits this slot is {by_budget['place'].name} at "
                    f"{by_budget['cost_breakdown']['total']:.0f} {itinerary.currency}, and only "
                    f"{spare:.0f} {itinerary.currency} of the budget is unspent. Time is not the "
                    f"problem here."
                )
            else:
                # Everything above asked what fits HERE — in the hours this stop happens to
                # occupy. That is not what was asked. Every other slot is locked so an edit
                # cannot cascade, which is right for "move this half an hour later" and wrong for
                # "swap this kind for that kind": a category carries opening hours, and opening
                # hours decide when it can happen at all. So ask the day, not the slot.
                retimed = _retimed_placement(
                    db, context, plan, day, target, category, travel_service, free_meals
                )
                if retimed is None:
                    if not fits_a_meal:
                        raise ValueError(
                            f"This day already has {_meals_eaten(context.profile, day)}, and this "
                            f"stop is not at a mealtime — swapping it for a {label} would make it "
                            f"the day's third sit-down meal. Replace one of the meals instead."
                        )
                    raise ValueError(
                        f"No {label} can go anywhere in this day — nothing of that kind in range "
                        f"is open, suitable for the party, and within budget. Neither a later "
                        f"finish nor a bigger budget would change it."
                    )
                candidate, fit, cost = retimed
                if not allow_reorder:
                    # Applied anyway, then raised after the repair pass below has had its say —
                    # the probe only knows the stop fits, not what shifting the day does to the
                    # stops behind it.
                    pending_reorder = DayReorderRequired(
                        label, candidate.name, fit.end_min - fit.start_min
                    )
                # Position is derived from start_min by `rebuild_segments`, so moving the slot in
                # the clock is the whole of moving it in the day.
                target.place = candidate
                target.start_min, target.end_min = fit.start_min, fit.end_min
                target.cost = cost
                retimed_in_place = True

        if not retimed_in_place:
            place_row = db.get(Place, place_id)
            if place_row is None:
                raise ValueError("Unknown place")

            candidate = to_candidate(place_row)
            if not all(person.age >= candidate.min_age for person in context.profile.attendees):
                raise ValueError(f"{candidate.name} requires age {candidate.min_age}+")

            # Named swaps skip `_best_alternative`'s budget filter, so without this check a swap
            # that is simply too expensive falls through to repair_plan silently dropping the
            # slot, which then surfaces downstream as "would cost the day a stop" — true, but not
            # why, and the model has been seen inventing a schedule reason to fill that gap.
            new_cost = slot_cost_breakdown(candidate, context.profile.attendees)
            spare = plan.total_budget - plan.total_cost + target.cost.total
            if new_cost.total > spare + 0.01:
                raise ValueError(
                    f"{candidate.name} costs {new_cost.total:.0f} {itinerary.currency}, and only "
                    f"{spare:.0f} {itinerary.currency} of the budget is unspent. Time is not the "
                    f"problem here."
                )

            target.place = candidate
            duration = min(candidate.avg_duration_min, context.profile.max_slot_min)
            target.start_min = max(target.start_min, candidate.opens_at)
            target.end_min = target.start_min + duration
            target.cost = new_cost
    else:
        raise ValueError(f"Unknown action {action!r}")

    if target in day.slots:
        target.locked = False

    all_places = [s.place for d in plan.days for s in d.slots]
    travel_fn = travel_service.travel_fn(all_places)

    # Cascade the day forward around the edit before validating, so a moved slot pushes its
    # neighbours rather than overlapping them.
    reflow_day(day, context.profile, travel_fn, context.origin)
    plan = repair_plan(plan, context.profile, travel_fn, context.origin)

    # `repair_plan` resolves a violation by dropping a slot, and the edited one is the only
    # unlocked candidate — so a replacement that breaks the day gets deleted instead of applied.
    # Nothing is persisted yet, so raising here leaves the plan exactly as the user last saw it.
    if action == "replace":
        placed = [slot.place.name for slot in plan.days[day_index].slots]
        if place_id is not None and not any(
            slot.place.id == place_id for d in plan.days for slot in d.slots
        ):
            raise ValueError(
                f"{db.get(Place, place_id).name} could not be fitted here without breaking "
                f"the day."
            )
        if len(placed) < stops_before:
            raise ValueError(
                f"Making room for that would cost the day a stop — it ends up with "
                f"{len(placed)} instead of {stops_before}. Remove something first if that is "
                f"what you want."
            )
        if pending_reorder is not None:
            raise pending_reorder

    for other_day in plan.days:
        for slot in other_day.slots:
            slot.locked = False

    persist_plan(db, itinerary, plan)
    db.commit()
    return plan, context, day_index


@traced("itinerary.add_stop", run_type="chain")
def add_stop(
    db: Session,
    itinerary: Itinerary,
    user: User,
    *,
    day_index: int,
    category: str | None = None,
    place_id: int | None = None,
) -> tuple[Plan, PlanContext, dict]:
    """Put one new stop into a day, in whichever gap costs the day least.

    Removing a stop used to be one-way: nothing could put anything back. Every gap between two
    existing stops is tried, plus the ends of the day; the placement that adds the fewest
    kilometres wins. Returns the plan, the context, and what was chosen alongside the runners-up
    so the caller can say what else was available.

    A `place_id` asks for one exact place rather than the best of a category, so it is fetched
    directly rather than filtered out of `_candidates_for_gap` — that pool is narrowed to the
    plan's emirates and the profile's usual scoring, which is right for "add an adventure" and
    wrong for "add the one I named": the user chose it already, the only question left is where
    in the day it goes.
    """
    context = context_for(db, itinerary, user)
    plan = load_plan(db, itinerary)
    if not 0 <= day_index < len(plan.days):
        raise ValueError("Unknown day")
    day = plan.days[day_index]

    booked = {s.place.id for d in plan.days for s in d.slots}
    if place_id is not None:
        if place_id in booked:
            raise ValueError("That place is already in the plan.")
        place_row = db.get(Place, place_id)
        if place_row is None:
            raise ValueError("Unknown place")
        candidates = [to_candidate(place_row)]
    else:
        candidates = _candidates_for_gap(db, context, booked)
        if category:
            candidates = [c for c in candidates if c.category == category]
    if not candidates:
        raise ValueError(f"Nothing available in {category!r}." if category else "Nothing available.")

    travel_service = _travel_service(db, itinerary, context)
    considered = _placements_in_day(
        context, day, candidates,
        travel_fn=travel_service.estimate_fn(),
        spare=plan.total_budget - plan.total_cost,
        free_meals=free_meal_windows(context.profile, day),
    )

    if not considered:
        if category in DINING_CATEGORIES and not free_meal_windows(context.profile, day):
            raise ValueError(
                f"This day already has {_meals_eaten(context.profile, day)}. A third sit-down "
                f"meal is not a gap in the schedule, it is a third meal — swap one of those "
                f"instead, or say which one to replace."
            )
        if place_id is not None:
            raise ValueError(f"{candidates[0].name} does not fit this day's schedule or budget.")
        raise ValueError("Nothing in that category fits this day's schedule or budget.")

    _, position, chosen, fit, cost = min(considered, key=lambda row: row[0])
    runners_up = [c.name for _, _, c, _, _ in sorted(considered, key=lambda r: r[0])
                  if c.id != chosen.id][:2]

    for other_day in plan.days:
        for slot in other_day.slots:
            slot.locked = True

    day.slots.insert(position, PlannedSlot(
        place=chosen,
        day_index=day_index,
        position=position,
        start_min=fit.start_min,
        end_min=fit.end_min,
        score=score_place(chosen, context.profile, context.preferences),
        cost=cost,
    ))
    for order, slot in enumerate(sorted(day.slots, key=lambda s: s.start_min)):
        slot.position = order

    real_travel = travel_service.travel_fn([s.place for d in plan.days for s in d.slots])
    reflow_day(day, context.profile, real_travel, context.origin)
    plan = repair_plan(plan, context.profile, real_travel, context.origin)

    for other_day in plan.days:
        for slot in other_day.slots:
            slot.locked = False

    survived = any(s.place.id == chosen.id for s in plan.days[day_index].slots)
    if not survived:
        raise ValueError(f"{chosen.name} could not be fitted into that day.")

    persist_plan(db, itinerary, plan)
    db.commit()
    return plan, context, {"chosen": chosen.name, "alternatives": runners_up}


@traced("itinerary.cheaper_day", run_type="chain")
def cheaper_day(db: Session, itinerary: Itinerary, user: User, day_index: int) -> Plan:
    """Re-solve one day against a reduced envelope, substituting budget alternatives."""
    context = context_for(db, itinerary, user)
    plan = load_plan(db, itinerary)
    if not 0 <= day_index < len(plan.days):
        raise ValueError("Unknown day")

    day = plan.days[day_index]
    target_spend = day.subtotal * CHEAPER_FACTOR
    booked_elsewhere = {
        s.place.id for d in plan.days for s in d.slots if d.day_index != day_index
    }

    candidates = _candidates_for_gap(db, context, booked_elsewhere)
    travel_service = _travel_service(db, itinerary, context)
    travel_fn = travel_service.travel_fn(candidates + [s.place for d in plan.days for s in d.slots])

    from .planner import assemble_day

    scores = {c.id: score_place(c, context.profile, context.preferences) for c in candidates}
    # Nudge scoring toward cheaper venues for this one pass, without touching the shared scorer.
    for candidate in candidates:
        scores[candidate.id] -= candidate.price_adult / 400.0

    replacement = assemble_day(
        day_index=day_index,
        day_date=day.day_date,
        candidates=candidates,
        profile=context.profile,
        travel_fn=travel_service.estimate_fn(),
        origin=context.origin,
        day_envelope=target_spend,
        remaining_total=plan.total_budget,
        scores=scores,
        used=set(booked_elsewhere),
    )

    if replacement.slots and replacement.subtotal < day.subtotal:
        plan.days[day_index] = replacement
    else:
        log.info("cheaper_day found nothing better for day %s", day_index)

    plan = route_for_real(plan, context, travel_fn)
    persist_plan(db, itinerary, plan)
    db.commit()
    return plan


@traced("itinerary.reschedule", run_type="chain")
def reschedule(db: Session, itinerary: Itinerary, user: User, new_start_date: date) -> Plan:
    """Move an existing plan to a new start date, keeping every stop and edit.

    A day's slots depend on the calendar only through the month (seasonal closures, prayer
    times), so this is a metadata change plus the same repair pass every edit already runs —
    never a rebuild. Routing is untouched: the places and their order have not moved, so the
    persisted travel segments are still correct. A slot that turns out to be seasonally closed in
    the new month is dropped by repair_plan, same as it would be on generation.
    """
    context = context_for(db, itinerary, user)
    itinerary.start_date = new_start_date
    plan = load_plan(db, itinerary)

    travel_service = _travel_service(db, itinerary, context)
    travel_fn = travel_service.travel_fn([s.place for d in plan.days for s in d.slots])

    plan = repair_plan(plan, context.profile, travel_fn, context.origin)
    persist_plan(db, itinerary, plan)
    db.commit()
    return plan


def emirate_centroid(db: Session, emirates: Sequence[str]) -> tuple[float, float] | None:
    """The middle of the catalog across these emirates, or None if it holds nothing there."""
    if not emirates:
        return None
    lat, lng = db.execute(
        select(func.avg(Place.lat), func.avg(Place.lng)).where(Place.emirate.in_(list(emirates)))
    ).one()
    return (lat, lng) if lat is not None else None


@traced("itinerary.set_origin", run_type="chain")
def set_origin(db: Session, itinerary: Itinerary, user: User, lat: float, lng: float) -> Plan:
    """Move where the trip sets off from, keeping every stop.

    Deliberately not a relocation: "we live in Abu Dhabi" says where the car starts, not what the
    trip is. The places and their order are untouched — only the leg from the origin into each
    day's first stop moves. That is why the segments are rebuilt rather than re-priced:
    `recost_travel` recomputes the fare from the distance already on the row, and here the
    distance is the thing that changed.
    """
    itinerary.start_lat, itinerary.start_lng = lat, lng
    # Read the context after the move, so `context.origin` is the new one.
    context = context_for(db, itinerary, user)
    plan = load_plan(db, itinerary)

    travel_service = _travel_service(db, itinerary, context)
    travel_fn = travel_service.travel_fn([s.place for d in plan.days for s in d.slots])

    for day in plan.days:
        rebuild_segments(day, travel_fn, context.origin)

    plan = repair_plan(plan, context.profile, travel_fn, context.origin)
    persist_plan(db, itinerary, plan)
    db.commit()
    return plan


class DayShiftChoiceRequired(ValueError):
    """Dropping a day that is not the last one leaves a choice only the user can make.

    The days after it can slide up — the trip ends a day sooner — or hold their dates and leave
    the dropped day free. An event anchored to one of those later dates is what makes the
    difference matter, and neither answer is safe to assume, so this asks instead of picking.
    """

    def __init__(self, day: int, num_days: int) -> None:
        super().__init__(
            f"Day {day} is not the last of {num_days}. Ask the user whether the days after it "
            f"should shift earlier, ending the trip a day sooner, or keep their dates and leave "
            f"day {day} free — then retry with shift_later_days set."
        )
        self.day = day
        self.num_days = num_days


@traced("itinerary.drop_day", run_type="chain")
def drop_day(
    db: Session,
    itinerary: Itinerary,
    user: User,
    day: int,
    *,
    shift_later_days: bool | None = None,
) -> Plan:
    """Remove one whole day. `day` is 1-based, the way the plan is numbered to the user.

    The budget cap is left alone. It is a ceiling the user set for the trip, not a per-day
    allowance to hand back, and lowering it here would quietly push `repair_plan` into dropping
    stops from the days they kept.
    """
    index = day - 1
    if itinerary.num_days <= 1:
        raise ValueError("This plan is a single day — dropping it would leave nothing.")
    if not 0 <= index < itinerary.num_days:
        raise ValueError(f"This plan runs {itinerary.num_days} days, so there is no day {day}.")

    is_last = index == itinerary.num_days - 1
    if not is_last and shift_later_days is None:
        raise DayShiftChoiceRequired(day, itinerary.num_days)

    context = context_for(db, itinerary, user)
    plan = load_plan(db, itinerary)
    plan.days = [d for d in plan.days if d.day_index != index]

    if is_last or shift_later_days:
        for later in plan.days:
            if later.day_index > index:
                later.day_index -= 1
                later.day_date -= timedelta(days=1)
                for slot in later.slots:
                    slot.day_index = later.day_index
        itinerary.num_days -= 1
    # Otherwise the day keeps its place in the calendar with nothing in it, which is what leaving
    # it free means: every date after it stays where the user already has it.

    travel_service = _travel_service(db, itinerary, context)
    travel_fn = travel_service.travel_fn([s.place for d in plan.days for s in d.slots])

    plan = repair_plan(plan, context.profile, travel_fn, context.origin)
    persist_plan(db, itinerary, plan)
    db.commit()
    return plan


class DayBudgetRequired(ValueError):
    """What is left of the trip's cap will not fill another day, so the user has to top it up.

    Raised only after trying: "enough for a day" is not a number anyone can name in advance — it
    depends on the party, the emirate and what is still open — so the remainder is handed to the
    planner and this fires when the planner comes back with nothing.
    """

    def __init__(self, remaining: float, currency: str) -> None:
        super().__init__(
            f"Only {remaining:,.0f} {currency} is left of this trip's budget, and nothing fits a "
            f"day into it. Ask the user what they want to spend on the extra day, then retry with "
            f"extra_budget set to the figure they give."
        )
        self.remaining = round(remaining, 2)


@traced("itinerary.add_day", run_type="chain")
def add_day(
    db: Session,
    itinerary: Itinerary,
    user: User,
    extra_budget: float | None = None,
    emirates: Sequence[str] | None = None,
) -> Plan:
    """Append one more day to the end of a plan, leaving every existing day untouched.

    The gap this fills: there was a `drop_day` and no way back. Asked to add a day, the model
    called drop_day — the only day-shaped tool it had — and then offered a rebuild, which is the
    one thing that would have thrown away every stop the user had approved.

    So the new day is SOLVED ON ITS OWN, at num_days=1, and appended. The existing days are never
    handed to the planner, which is what makes this an edit rather than a rebuild: nothing already
    on the plan can be re-scored, re-timed or dropped by adding to it.

    `extra_budget` raises the cap rather than sharing it. A trip that has spent 2,961 of 3,000 has
    39 left, and solving a day into 39 either returns something empty or sends `repair_plan` off
    to strip stops from the days the user already has. The cap is a ceiling they set, so only they
    can raise it — which is why the tool takes the figure instead of inventing one.
    """
    # Zero is not a budget, it is the absence of one — and it is what arrives, because strict mode
    # gives the model a slot it must fill and "leave it out" is advice. Rejecting it asked the user
    # to fund a day their own remaining 2,859 already covered.
    if extra_budget is not None and extra_budget <= 0:
        extra_budget = None
    if itinerary.num_days >= MAX_DAYS:
        raise ValueError(f"A trip runs at most {MAX_DAYS} days, and this one already does.")

    context = context_for(db, itinerary, user)
    plan = load_plan(db, itinerary)
    event = db.get(Event, itinerary.event_id) if itinerary.event_id else None

    # Spend what the trip has not spent before asking for more. A plan that came in well under its
    # cap already has the day paid for, and asking anyway is asking the user to fund something
    # they funded — while a plan that spent nearly all of it cannot fit a day into the difference.
    # Which of the two this is nobody can say from a number, so it is settled by trying.
    remaining = max(itinerary.total_budget - plan.total_cost, 0.0)
    budget_for_day = extra_budget if extra_budget is not None else remaining
    new_index = itinerary.num_days
    new_date = itinerary.start_date + timedelta(days=new_index)

    # A day of its own can go somewhere of its own. Narrowing only the retrieval leaves the trip's
    # own `emirates_json` alone, which is the point: this is where ONE day happens, not a decision
    # that the whole trip has moved.
    #
    # ponytail: the day still sets off from the trip's origin, so an Abu Dhabi day on a Dubai trip
    # is priced with the drive there — which is the truth. Move the origin per day if that ever
    # needs to model an overnight instead.
    candidates = retrieve_candidates(
        db,
        context.profile,
        query_for(context.profile, event.title if event else "", event.notes if event else ""),
        origin=context.origin,
        emirates=list(emirates) if emirates else context.emirates,
    )
    travel_service = _travel_service(db, itinerary, context)

    # A day that repeats what the trip has already seen is not another day out. The places
    # already on the plan are dropped from the candidate pool, so the new one is somewhere new.
    #
    # `or candidates` because an exclusion that empties the pool is not an exclusion, it is a
    # dead end — a trip confined to one emirate can genuinely run out. A repeat beats no day.
    # It is a filter on the POOL, not a rule inside the day: a restaurant appearing twice within
    # a single day is the planner's own doing and stays that way.
    used = {slot.place.id for day in plan.days for slot in day.slots}
    candidates = [c for c in candidates if c.id not in used] or candidates

    solved = generate_plan(
        candidates,
        context.profile,
        travel_service.estimate_fn(),
        start_date=new_date,
        num_days=1,
        total_budget=budget_for_day,
        origin=context.origin,
        preferences=context.preferences,
        currency=plan.currency,
    )
    if not solved.days or not solved.days[0].slots:
        if emirates:
            raise ValueError(
                f"Nothing fits a day in {', '.join(emirates)} at "
                f"{budget_for_day:,.0f} {plan.currency}. Say so, and ask whether to spend more or "
                f"look somewhere else."
            )
        if extra_budget is None:
            raise DayBudgetRequired(remaining, plan.currency)
        raise ValueError(
            f"Nothing fits a day at {extra_budget:,.0f} {plan.currency}. Ask for a higher figure."
        )

    fresh = solved.days[0]
    fresh.day_index = new_index
    fresh.day_date = new_date
    for slot in fresh.slots:
        slot.day_index = new_index
    for segment in fresh.segments:
        segment.day_index = new_index

    plan.days.append(fresh)
    itinerary.num_days += 1
    # Only a figure the user added raises the cap. Spending the remainder is spending what the cap
    # already allowed, and raising it here would hand the trip a budget nobody agreed to.
    itinerary.total_budget += extra_budget or 0.0
    plan.total_budget = itinerary.total_budget

    travel_fn = travel_service.travel_fn([s.place for d in plan.days for s in d.slots])
    plan = route_for_real(plan, context, travel_fn)
    persist_plan(db, itinerary, plan)
    db.commit()
    return plan


@traced("itinerary.prayer_breaks", run_type="chain")
def add_prayer_breaks(db: Session, itinerary: Itinerary, user: User) -> Plan:
    context = context_for(db, itinerary, user)
    plan = load_plan(db, itinerary)

    travel_service = _travel_service(db, itinerary, context)
    travel_fn = travel_service.travel_fn([s.place for d in plan.days for s in d.slots])

    for day in plan.days:
        insert_prayer_breaks(day, travel_fn, context.origin)

    plan = repair_plan(plan, context.profile, travel_fn, context.origin)
    persist_plan(db, itinerary, plan)
    db.commit()
    return plan
