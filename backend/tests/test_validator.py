"""Validator and repair loop. These tests deliberately hand-build BROKEN plans — the planner is
not supposed to produce them, and the validator is the reason a broken one never reaches a user."""

from __future__ import annotations

from datetime import date, timedelta

from app.services.budget import slot_cost_breakdown
from app.services.planner import (
    DayPlan,
    Plan,
    PlannedSegment,
    PlannedSlot,
    TravelInfo,
    build_profile,
    to_minutes,
)
from app.services.validator import (
    BUDGET_EXCEEDED,
    MIN_AGE_NOT_MET,
    SLOT_OVERLAP,
    TOO_MANY_DAYS,
    TRAVEL_TIME_VIOLATED,
    VENUE_CLOSED,
    repair_plan,
    validate_day,
    validate_plan,
)

from .factories import ORIGIN, family, fixed_travel, place

TOMORROW = date.today() + timedelta(days=1)


def slot(place_obj, start: str, end: str, position: int, attendees, score: float = 1.0) -> PlannedSlot:
    return PlannedSlot(
        place=place_obj,
        day_index=0,
        position=position,
        start_min=to_minutes(start),
        end_min=to_minutes(end),
        score=score,
        cost=slot_cost_breakdown(place_obj, attendees),
    )


def day_with(slots, travel_min: int = 0) -> DayPlan:
    day = DayPlan(day_index=0, day_date=TOMORROW, slots=list(slots))
    day.segments = [
        PlannedSegment(
            day_index=0,
            from_position=None if i == 0 else i - 1,
            to_position=i,
            info=TravelInfo(distance_km=5.0, duration_min=travel_min, est_cost=0.0),
        )
        for i in range(len(slots))
    ]
    return day


def codes(violations) -> set[str]:
    return {v.code for v in violations}


# --- individual constraints --------------------------------------------------------------------


def test_overlapping_slots_are_caught():
    attendees = family(2)
    profile = build_profile(attendees)
    a = place(1, "A", open_time="08:00", close_time="22:00")
    b = place(2, "B", open_time="08:00", close_time="22:00")
    day = day_with([slot(a, "10:00", "12:00", 0, attendees), slot(b, "11:30", "13:00", 1, attendees)])
    assert SLOT_OVERLAP in codes(validate_day(day, profile))


def test_back_to_back_slots_without_travel_time_are_caught():
    attendees = family(2)
    profile = build_profile(attendees)
    a = place(1, "A", open_time="08:00", close_time="22:00")
    b = place(2, "B", open_time="08:00", close_time="22:00")
    # 10 minutes apart, but the drive takes 35.
    day = day_with(
        [slot(a, "10:00", "12:00", 0, attendees), slot(b, "12:10", "13:00", 1, attendees)],
        travel_min=35,
    )
    assert TRAVEL_TIME_VIOLATED in codes(validate_day(day, profile))


def test_exactly_enough_travel_time_is_accepted():
    attendees = family(2)
    profile = build_profile(attendees)
    a = place(1, "A", open_time="08:00", close_time="22:00")
    b = place(2, "B", open_time="08:00", close_time="22:00")
    day = day_with(
        [slot(a, "10:00", "12:00", 0, attendees), slot(b, "12:35", "13:30", 1, attendees)],
        travel_min=35,
    )
    assert validate_day(day, profile) == []


def test_a_slot_starting_before_opening_is_caught():
    attendees = family(2)
    profile = build_profile(attendees)
    late_opener = place(1, "Aquarium", open_time="10:00", close_time="22:00")
    day = day_with([slot(late_opener, "09:00", "10:30", 0, attendees)])
    assert VENUE_CLOSED in codes(validate_day(day, profile))


def test_a_slot_running_past_closing_is_caught():
    attendees = family(2)
    profile = build_profile(attendees)
    early_closer = place(1, "Museum", open_time="10:00", close_time="18:00")
    day = day_with([slot(early_closer, "17:00", "19:00", 0, attendees)])
    assert VENUE_CLOSED in codes(validate_day(day, profile))


def test_a_venue_open_past_midnight_is_not_flagged():
    attendees = family(2)
    profile = build_profile(attendees)
    late = place(1, "Night Cafe", open_time="19:00", close_time="02:00")
    day = day_with([slot(late, "22:00", "23:30", 0, attendees)])
    assert validate_day(day, profile) == []


def test_min_age_is_checked_against_the_youngest_attendee():
    attendees = family(2, (7,))
    profile = build_profile(attendees)
    adults_only = place(1, "Zipline", min_age=12, open_time="08:00", close_time="18:00")
    day = day_with([slot(adults_only, "10:00", "12:00", 0, attendees)])
    assert MIN_AGE_NOT_MET in codes(validate_day(day, profile))


def test_min_age_passes_when_every_attendee_clears_it():
    attendees = family(2, (14,))
    profile = build_profile(attendees)
    twelve_plus = place(1, "Zipline", min_age=12, open_time="08:00", close_time="18:00")
    day = day_with([slot(twelve_plus, "10:00", "12:00", 0, attendees)])
    assert validate_day(day, profile) == []


def test_budget_overrun_is_caught_at_trip_level():
    attendees = family(2)
    profile = build_profile(attendees)
    pricey = place(1, "Pricey", price_adult=900, open_time="08:00", close_time="22:00")
    plan = Plan(days=[day_with([slot(pricey, "10:00", "12:00", 0, attendees)])], total_budget=500.0)
    assert BUDGET_EXCEEDED in codes(validate_plan(plan, profile))


def test_more_than_five_days_is_caught():
    profile = build_profile(family(2))
    plan = Plan(days=[DayPlan(i, TOMORROW + timedelta(days=i)) for i in range(6)], total_budget=1e6)
    assert TOO_MANY_DAYS in codes(validate_plan(plan, profile))


# --- repair ------------------------------------------------------------------------------------


def test_repair_drops_the_lowest_scored_slot_to_fix_an_overlap():
    attendees = family(2)
    profile = build_profile(attendees)
    keeper = place(1, "Keeper", open_time="08:00", close_time="22:00")
    loser = place(2, "Loser", open_time="08:00", close_time="22:00")

    day = day_with(
        [
            slot(keeper, "10:00", "12:00", 0, attendees, score=5.0),
            slot(loser, "11:00", "13:00", 1, attendees, score=0.2),
        ]
    )
    plan = Plan(days=[day], total_budget=10_000.0)

    repaired = repair_plan(plan, profile, fixed_travel(0), ORIGIN)
    assert validate_plan(repaired, profile) == []
    assert [s.place.name for s in repaired.days[0].slots] == ["Keeper"]


def test_repair_trims_the_most_expensive_day_to_meet_the_cap():
    attendees = family(2)
    profile = build_profile(attendees)
    cheap = place(1, "Cheap", price_adult=20, open_time="08:00", close_time="22:00")
    dear = place(2, "Dear", price_adult=800, open_time="08:00", close_time="22:00")

    cheap_day = day_with([slot(cheap, "10:00", "11:00", 0, attendees, score=1.0)])
    dear_day = DayPlan(day_index=1, day_date=TOMORROW + timedelta(days=1))
    dear_day.slots = [slot(dear, "10:00", "12:00", 0, attendees, score=0.5)]
    dear_day.slots[0].day_index = 1

    plan = Plan(days=[cheap_day, dear_day], total_budget=100.0)
    repaired = repair_plan(plan, profile, fixed_travel(0), ORIGIN)

    assert validate_plan(repaired, profile) == []
    assert repaired.total_cost <= 100.0
    assert [s.place.name for s in repaired.days[0].slots] == ["Cheap"]


def test_repair_recomputes_travel_segments_for_the_day_it_touched():
    attendees = family(2)
    profile = build_profile(attendees)
    a = place(1, "A", lat=25.20, lng=55.27, open_time="08:00", close_time="22:00")
    b = place(2, "B", lat=25.21, lng=55.28, open_time="08:00", close_time="22:00")
    c = place(3, "C", lat=25.22, lng=55.29, open_time="08:00", close_time="22:00")

    day = day_with(
        [
            slot(a, "09:00", "10:00", 0, attendees, score=5.0),
            slot(b, "09:30", "10:30", 1, attendees, score=0.1),  # overlaps A
            slot(c, "12:00", "13:00", 2, attendees, score=4.0),
        ]
    )
    plan = Plan(days=[day], total_budget=10_000.0)
    repaired = repair_plan(plan, profile, fixed_travel(15), ORIGIN)

    survivors = repaired.days[0]
    assert len(survivors.segments) == len(survivors.slots)
    assert {s.to_position for s in survivors.segments} == set(range(len(survivors.slots)))
    assert survivors.segments[0].from_position is None  # first leg comes from the start location


def test_repair_will_not_remove_a_locked_slot():
    """Slot editing locks every other slot; repair must respect that or edits would cascade."""
    attendees = family(2)
    profile = build_profile(attendees)
    pinned = place(1, "Pinned", open_time="08:00", close_time="22:00")
    other = place(2, "Other", open_time="08:00", close_time="22:00")

    day = day_with(
        [
            slot(pinned, "10:00", "12:00", 0, attendees, score=0.1),
            slot(other, "11:00", "13:00", 1, attendees, score=5.0),
        ]
    )
    day.slots[0].locked = True
    plan = Plan(days=[day], total_budget=10_000.0)

    repaired = repair_plan(plan, profile, fixed_travel(0), ORIGIN)
    assert validate_plan(repaired, profile) == []
    assert [s.place.name for s in repaired.days[0].slots] == ["Pinned"]


def test_an_unrepairable_plan_ends_up_empty_with_a_warning_never_subtly_wrong():
    attendees = family(2, (5,))
    profile = build_profile(attendees)
    forbidden = place(1, "Adults Only", min_age=18, open_time="08:00", close_time="22:00")
    day = day_with([slot(forbidden, "10:00", "12:00", 0, attendees)])
    day.slots[0].locked = True
    plan = Plan(days=[day], total_budget=10_000.0)

    repaired = repair_plan(plan, profile, fixed_travel(0), ORIGIN)
    assert repaired.warnings, "an unrepairable plan must say so"


def test_repair_is_a_no_op_on_an_already_valid_plan():
    attendees = family(2)
    profile = build_profile(attendees)
    a = place(1, "A", open_time="08:00", close_time="22:00")
    day = day_with([slot(a, "10:00", "12:00", 0, attendees)])
    plan = Plan(days=[day], total_budget=10_000.0)

    repaired = repair_plan(plan, profile, fixed_travel(0), ORIGIN)
    assert len(repaired.days[0].slots) == 1
    assert repaired.warnings == []
