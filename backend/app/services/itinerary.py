"""Everything between the database and the pure planner.

Responsibilities: assemble the party and preferences for a user, retrieve candidates, run the
planner, persist the result, and load persisted rows back into planner dataclasses so that edits
re-run the very same validator that generation did.

Nothing here decides scheduling — that lives in planner.py and validator.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Event, FamilyMember, Itinerary, Place, Preference, Slot, TravelSegment, User
from ..repo import get_itinerary_or_404
from .budget import (
    Attendee,
    CostBreakdown,
    category_bucket,
    slot_cost_breakdown,
    summarise,
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
    build_profile,
    day_theme,
    generate_plan,
    rebuild_segments,
    reflow_day,
    score_place,
    theme_from_categories,
    to_minutes,
)
from .prayer import insert_prayer_breaks
from .retrieval import query_for, retrieve_candidates, to_candidate
from .travel import TravelService
from .tracing import traced
from .validator import repair_plan, validate_plan

log = logging.getLogger(__name__)

ALTERNATIVES_COUNT = 3
# "Cheaper Day N" re-solves the day against this fraction of what it currently costs.
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


# --- gathering ---------------------------------------------------------------------------------


def family_attendees(db: Session, user_id: int) -> list[Attendee]:
    members = db.scalars(
        select(FamilyMember).where(FamilyMember.user_id == user_id).order_by(FamilyMember.id)
    )
    return [Attendee(role=m.role, age=m.age, name=m.name) for m in members]


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
    db: Session, user: User, event: Event | None, *, start_lat: float, start_lng: float
) -> PlanContext:
    attendees = family_attendees(db, user.id)
    event_type = event.event_type if event else "other"
    return PlanContext(
        user=user,
        profile=build_profile(attendees, event_type),
        preferences=preference_signals(db, user.id),
        origin=(start_lat, start_lng),
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
) -> Itinerary:
    """Plan a trip and persist it. Raises IntakeIncomplete before doing any work."""
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

    context = build_context(db, user, event, start_lat=start_lat, start_lng=start_lng)
    candidates = retrieve_candidates(
        db,
        context.profile,
        query_for(context.profile, event.title if event else "", event.notes if event else ""),
    )

    travel_service = TravelService(db)
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

    plan = route_for_real(plan, context, travel_fn)

    if prayer_breaks:
        for day in plan.days:
            insert_prayer_breaks(day, travel_fn, context.origin)
        plan = repair_plan(plan, context.profile, travel_fn, context.origin)

    itinerary = Itinerary(
        user_id=user.id,
        event_id=event.id if event else None,
        title=title or (event.title if event else "UAE trip"),
        start_date=start_date,
        num_days=num_days,
        total_budget=total_budget,
        currency=currency,
        status="ready",
        start_lat=start_lat,
        start_lng=start_lng,
    )
    db.add(itinerary)
    db.flush()

    persist_plan(db, itinerary, plan)
    if event is not None:
        event.planned = True
    db.commit()
    db.refresh(itinerary)
    return itinerary


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
        db, user, event, start_lat=itinerary.start_lat, start_lng=itinerary.start_lng
    )


# --- payload assembly --------------------------------------------------------------------------


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

    return {
        "id": itinerary.id,
        "title": itinerary.title,
        "event_id": itinerary.event_id,
        "event_title": event.title if event else None,
        "start_date": itinerary.start_date,
        "num_days": itinerary.num_days,
        "currency": itinerary.currency,
        "status": itinerary.status,
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
    chips.append(
        {
            "id": f"cheaper-day-{priciest}",
            "label": f"Cheaper Day {priciest + 1}",
            "action": "cheaper_day",
            "day_index": priciest,
        }
    )
    chips.append(
        {
            "id": "prayer-breaks",
            "label": "Add prayer breaks",
            "action": "prayer_breaks",
            "day_index": None,
        }
    )
    return chips


def budget_payload(plan: Plan) -> dict:
    return summarise(plan.days, plan.total_budget, plan.currency)


# --- editing -----------------------------------------------------------------------------------


def _candidates_for_gap(
    db: Session,
    context: PlanContext,
    exclude_place_ids: set[int],
) -> list[PlaceCandidate]:
    candidates = retrieve_candidates(db, context.profile, query_for(context.profile))
    return [c for c in candidates if c.id not in exclude_place_ids]


def gap_window(day: DayPlan, position: int) -> tuple[int, int]:
    """The time window a replacement slot must fit into: between its two neighbours."""
    ordered = sorted(day.slots, key=lambda s: s.start_min)
    index = next((i for i, s in enumerate(ordered) if s.position == position), None)
    if index is None:
        return (0, 24 * 60)
    earliest = ordered[index - 1].end_min if index > 0 else 0
    latest = ordered[index + 1].start_min if index + 1 < len(ordered) else 24 * 60 - 1
    return (earliest, latest)


@traced("itinerary.alternatives", run_type="chain")
def alternatives_for_slot(
    db: Session, itinerary: Itinerary, user: User, slot_row: Slot, limit: int = ALTERNATIVES_COUNT
) -> list[dict]:
    """Three options that fit the exact window and the remaining budget (spec §6)."""
    context = context_for(db, itinerary, user)
    plan = load_plan(db, itinerary)
    day = plan.days[slot_row.day_index]

    travel_service = TravelService(db)
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
        if not all(person.age >= candidate.min_age for person in context.profile.attendees):
            continue

        inbound = travel_fn(origin[0], origin[1], candidate.lat, candidate.lng)
        start = max(earliest + inbound.duration_min, candidate.opens_at)
        duration = min(candidate.avg_duration_min, context.profile.max_slot_min)
        end = start + duration

        if end > candidate.closes_at:
            continue
        if following is not None:
            outbound = travel_fn(
                candidate.lat, candidate.lng, following.place.lat, following.place.lng
            )
            if end + outbound.duration_min > following.start_min:
                continue
        elif end > latest:
            continue

        cost = slot_cost_breakdown(candidate, context.profile.attendees, inbound.est_cost)
        if cost.total > spare:
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

    travel_service = TravelService(db)

    if action == "remove":
        day.slots.remove(target)

    elif action == "adjust":
        if start_time is None:
            raise ValueError("adjust requires start_time")
        duration = target.end_min - target.start_min
        target.start_min = to_minutes(start_time)
        target.end_min = target.start_min + duration

    elif action == "replace":
        if place_id is None:
            raise ValueError("replace requires place_id")
        place_row = db.get(Place, place_id)
        if place_row is None:
            raise ValueError("Unknown place")

        candidate = to_candidate(place_row)
        if not all(person.age >= candidate.min_age for person in context.profile.attendees):
            raise ValueError(f"{candidate.name} requires age {candidate.min_age}+")

        target.place = candidate
        duration = min(candidate.avg_duration_min, context.profile.max_slot_min)
        target.start_min = max(target.start_min, candidate.opens_at)
        target.end_min = target.start_min + duration
        target.cost = slot_cost_breakdown(candidate, context.profile.attendees)
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

    for other_day in plan.days:
        for slot in other_day.slots:
            slot.locked = False

    persist_plan(db, itinerary, plan)
    db.commit()
    return plan, context, day_index


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
    travel_service = TravelService(db)
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


@traced("itinerary.prayer_breaks", run_type="chain")
def add_prayer_breaks(db: Session, itinerary: Itinerary, user: User) -> Plan:
    context = context_for(db, itinerary, user)
    plan = load_plan(db, itinerary)

    travel_service = TravelService(db)
    travel_fn = travel_service.travel_fn([s.place for d in plan.days for s in d.slots])

    for day in plan.days:
        insert_prayer_breaks(day, travel_fn, context.origin)

    plan = repair_plan(plan, context.profile, travel_fn, context.origin)
    persist_plan(db, itinerary, plan)
    db.commit()
    return plan


def reload_and_validate(db: Session, itinerary_id: int, user: User) -> tuple[Plan, PlanContext]:
    itinerary = get_itinerary_or_404(db, itinerary_id, user.id)
    context = context_for(db, itinerary, user)
    plan = load_plan(db, itinerary)
    violations = validate_plan(plan, context.profile)
    if violations:
        log.warning("persisted itinerary %s has %s violations", itinerary_id, len(violations))
    return plan, context
