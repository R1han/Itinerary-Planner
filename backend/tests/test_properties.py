"""Property-based tests (spec §13): random party profiles + budgets → the validator always passes
on the planner's output.

This is the strongest statement of the "foolproof" guarantee. Unit tests check the cases we
thought of; these check the ones we did not. The planner being pure — travel injected, no DB, no
clock — is what makes running hundreds of these cheap.
"""

from __future__ import annotations

from datetime import date, timedelta

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from app.services.budget import Attendee
from app.services.planner import (
    DINING_CATEGORIES,
    PreferenceSignal,
    build_profile,
    generate_plan,
)
from app.services.validator import validate_plan

from .factories import ORIGIN, distance_travel, place

TOMORROW = date.today() + timedelta(days=1)

CATEGORIES = [
    "park", "waterpark", "theme_park", "museum", "aquarium", "beach",
    "adventure", "casual_dining", "fine_dining", "mall", "show", "cruise",
]
EVENT_TYPES = ["birthday", "anniversary", "family_visit", "holiday", "eid", "other"]

SLOW = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)


@st.composite
def a_place(draw, place_id: int):
    open_hour = draw(st.integers(min_value=6, max_value=18))
    # Always at least two hours of opening, so a feasible slot can exist at all.
    close_hour = draw(st.integers(min_value=open_hour + 2, max_value=24))
    return place(
        place_id,
        name=f"Place {place_id}",
        category=draw(st.sampled_from(CATEGORIES)),
        lat=draw(st.floats(min_value=24.0, max_value=26.0, allow_nan=False)),
        lng=draw(st.floats(min_value=54.0, max_value=56.3, allow_nan=False)),
        price_adult=draw(st.floats(min_value=0, max_value=900, allow_nan=False)),
        price_child=draw(st.floats(min_value=0, max_value=600, allow_nan=False)),
        min_age=draw(st.sampled_from([0, 0, 0, 3, 5, 8, 12, 16, 18])),
        open_time=f"{open_hour:02d}:00",
        close_time=f"{close_hour % 24:02d}:00",
        avg_duration_min=draw(st.integers(min_value=30, max_value=330)),
        kid_score=draw(st.floats(min_value=0, max_value=1, allow_nan=False)),
        teen_score=draw(st.floats(min_value=0, max_value=1, allow_nan=False)),
        romance_score=draw(st.floats(min_value=0, max_value=1, allow_nan=False)),
        similarity=draw(st.floats(min_value=0, max_value=1, allow_nan=False)),
    )


@st.composite
def a_catalog(draw):
    size = draw(st.integers(min_value=0, max_value=45))
    return [draw(a_place(i)) for i in range(1, size + 1)]


@st.composite
def a_party(draw):
    adults = draw(st.integers(min_value=1, max_value=4))
    child_ages = draw(st.lists(st.integers(min_value=0, max_value=17), max_size=4))
    people = [Attendee("adult", draw(st.integers(min_value=18, max_value=70))) for _ in range(adults)]
    people += [Attendee("child", age) for age in child_ages]
    return people


@st.composite
def a_preference_set(draw):
    return [
        PreferenceSignal(
            kind=draw(st.sampled_from(["like", "dislike"])),
            subject=draw(st.sampled_from(["animals", "thrill rides", "quiet beaches", "queues"])),
            category=draw(st.sampled_from([None, *CATEGORIES])),
            strength=draw(st.floats(min_value=0.1, max_value=1.0, allow_nan=False)),
        )
        for _ in range(draw(st.integers(min_value=0, max_value=4)))
    ]


@SLOW
@given(
    catalog=a_catalog(),
    party=a_party(),
    event_type=st.sampled_from(EVENT_TYPES),
    num_days=st.integers(min_value=1, max_value=5),
    budget=st.floats(min_value=50, max_value=20_000, allow_nan=False),
    preferences=a_preference_set(),
)
def test_planner_output_always_validates(catalog, party, event_type, num_days, budget, preferences):
    profile = build_profile(party, event_type)
    plan = generate_plan(
        catalog,
        profile,
        distance_travel(),
        start_date=TOMORROW,
        num_days=num_days,
        total_budget=budget,
        origin=ORIGIN,
        preferences=preferences,
    )
    violations = validate_plan(plan, profile)
    assert violations == [], "\n".join(str(v) for v in violations)


@SLOW
@given(
    catalog=a_catalog(),
    party=a_party(),
    num_days=st.integers(min_value=1, max_value=5),
    budget=st.floats(min_value=50, max_value=20_000, allow_nan=False),
)
def test_planner_never_exceeds_the_budget_cap(catalog, party, num_days, budget):
    profile = build_profile(party, "holiday")
    plan = generate_plan(
        catalog, profile, distance_travel(), start_date=TOMORROW, num_days=num_days,
        total_budget=budget, origin=ORIGIN,
    )
    assert plan.total_cost <= budget + 0.01


@SLOW
@given(
    catalog=a_catalog(),
    party=a_party(),
    num_days=st.integers(min_value=1, max_value=5),
)
def test_no_place_is_ever_booked_twice(catalog, party, num_days):
    profile = build_profile(party, "family_visit")
    plan = generate_plan(
        catalog, profile, distance_travel(), start_date=TOMORROW, num_days=num_days,
        total_budget=50_000.0, origin=ORIGIN,
    )
    booked = [slot.place.id for day in plan.days for slot in day.slots]
    assert len(booked) == len(set(booked))


@SLOW
@given(catalog=a_catalog(), party=a_party(), num_days=st.integers(min_value=1, max_value=5))
def test_every_attendee_always_clears_min_age(catalog, party, num_days):
    profile = build_profile(party, "birthday")
    plan = generate_plan(
        catalog, profile, distance_travel(), start_date=TOMORROW, num_days=num_days,
        total_budget=50_000.0, origin=ORIGIN,
    )
    youngest = min(person.age for person in party)
    for day in plan.days:
        for slot in day.slots:
            assert youngest >= slot.place.min_age


@SLOW
@given(catalog=a_catalog(), party=a_party(), num_days=st.integers(min_value=1, max_value=5))
def test_the_day_count_always_matches_the_request(catalog, party, num_days):
    profile = build_profile(party, "holiday")
    plan = generate_plan(
        catalog, profile, distance_travel(), start_date=TOMORROW, num_days=num_days,
        total_budget=8_000.0, origin=ORIGIN,
    )
    assert len(plan.days) == num_days
    assert [day.day_index for day in plan.days] == list(range(num_days))
    assert [day.day_date for day in plan.days] == [
        TOMORROW + timedelta(days=i) for i in range(num_days)
    ]


@SLOW
@given(catalog=a_catalog(), party=a_party(), num_days=st.integers(min_value=1, max_value=5))
def test_every_slot_has_exactly_one_inbound_travel_segment(catalog, party, num_days):
    """The strip renders one TravelConnector per slot; a missing or duplicated segment would
    silently misreport how long the day's driving takes."""
    profile = build_profile(party, "holiday")
    plan = generate_plan(
        catalog, profile, distance_travel(), start_date=TOMORROW, num_days=num_days,
        total_budget=8_000.0, origin=ORIGIN,
    )
    for day in plan.days:
        inbound = [segment.to_position for segment in day.segments]
        assert sorted(inbound) == sorted(slot.position for slot in day.slots)
        assert len(inbound) == len(set(inbound))


@SLOW
@given(catalog=a_catalog(), party=a_party())
def test_meal_slots_only_ever_appear_in_meal_windows(catalog, party):
    profile = build_profile(party, "holiday")
    plan = generate_plan(
        catalog, profile, distance_travel(), start_date=TOMORROW, num_days=2,
        total_budget=8_000.0, origin=ORIGIN,
    )
    windows = [(start, end) for _, start, end in profile.meal_windows]
    for day in plan.days:
        for slot in day.slots:
            if slot.place.category in DINING_CATEGORIES:
                assert any(start <= slot.start_min <= end + 90 for start, end in windows), (
                    f"{slot.place.name} at {slot.start_time} is outside every meal window"
                )
