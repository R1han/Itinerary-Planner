"""Planner engine: profiles, scoring, clustering, day assembly and pricing."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.services.budget import Attendee, price_for_age, slot_cost_breakdown, summarise
from app.services.planner import (
    DINING_CATEGORIES,
    PreferenceSignal,
    build_profile,
    cluster_by_proximity,
    day_theme,
    generate_plan,
    haversine_km,
    score_place,
    to_hhmm,
    to_minutes,
)
from app.services.validator import validate_plan

from .factories import ORIGIN, distance_travel, family, fixed_travel, place

TOMORROW = date.today() + timedelta(days=1)


# --- time helpers ------------------------------------------------------------------------------


@pytest.mark.parametrize(("text", "minutes"), [("00:00", 0), ("09:30", 570), ("23:59", 1439)])
def test_time_round_trips(text, minutes):
    assert to_minutes(text) == minutes
    assert to_hhmm(minutes) == text


def test_after_midnight_closing_is_treated_as_next_day():
    """A restaurant open 19:00–02:00 must not be read as closing before it opens."""
    late = place(1, open_time="19:00", close_time="02:00")
    assert late.closes_at > late.opens_at
    assert late.closes_at == 26 * 60


# --- party profile -----------------------------------------------------------------------------


def test_young_children_shorten_slots_and_add_a_midday_rest():
    profile = build_profile(family(2, (5, 7)), "birthday")
    assert profile.w_kid > profile.w_teen
    assert profile.max_slot_min <= 120
    assert profile.needs_midday_rest is True
    assert profile.day_end <= 20 * 60


def test_teens_weight_teen_score_and_allow_long_slots():
    profile = build_profile(family(2, (14, 16)), "family_visit")
    assert profile.w_teen > profile.w_kid
    assert profile.needs_midday_rest is False
    assert profile.max_slot_min >= 300


def test_adults_only_anniversary_is_romantic_and_evening_heavy():
    profile = build_profile(family(2), "anniversary")
    assert profile.w_romance == 1.0
    assert profile.w_kid == 0.0
    assert profile.evening_bias is True
    assert profile.day_end >= 23 * 60


def test_anniversary_with_children_present_is_not_treated_as_romantic():
    profile = build_profile(family(2, (6,)), "anniversary")
    assert profile.evening_bias is False
    assert profile.w_kid > profile.w_romance


# --- scoring -----------------------------------------------------------------------------------


def test_profile_weights_decide_ranking_between_two_places():
    zoo = place(1, "Zoo", "aquarium", kid_score=0.95, teen_score=0.3, romance_score=0.1)
    zipline = place(2, "Zipline", "adventure", kid_score=0.1, teen_score=0.98, romance_score=0.3)

    kids = build_profile(family(2, (6,)), "birthday")
    teens = build_profile(family(2, (15,)), "family_visit")

    assert score_place(zoo, kids) > score_place(zipline, kids)
    assert score_place(zipline, teens) > score_place(zoo, teens)


def test_a_dislike_outweighs_an_equal_strength_like():
    """Dislikes are penalised harder than likes are boosted — avoiding a bad slot matters more."""
    coaster = place(1, "Coaster", "adventure", tags=("thrill",))
    disliked = [PreferenceSignal("dislike", "thrill rides", "adventure", 0.7)]
    liked = [PreferenceSignal("like", "thrill rides", "adventure", 0.7)]
    profile = build_profile(family(2, (10,)))

    baseline = score_place(coaster, profile)
    assert score_place(coaster, disliked and profile, disliked) < baseline
    assert score_place(coaster, profile, liked) > baseline
    assert baseline - score_place(coaster, profile, disliked) > score_place(coaster, profile, liked) - baseline


def test_preference_matches_on_tags_not_only_category():
    romantic = place(1, "Sunset Terrace", "beach", tags=("sunset", "romantic"))
    profile = build_profile(family(2), "anniversary")
    prefs = [PreferenceSignal("like", "romantic sunset spots", None, 0.9)]
    assert score_place(romantic, profile, prefs) > score_place(romantic, profile)


# --- clustering --------------------------------------------------------------------------------


def test_clustering_keeps_distant_emirates_on_different_days():
    dubai = [place(i, f"Dubai {i}", lat=25.20 + i * 0.01, lng=55.27) for i in range(1, 5)]
    fujairah = [place(10 + i, f"Fujairah {i}", lat=25.50 + i * 0.01, lng=56.36) for i in range(1, 5)]

    buckets = cluster_by_proximity(dubai + fujairah, 2, ORIGIN)
    assert len(buckets) == 2
    for bucket in buckets:
        emirate_spread = max(haversine_km(a.lat, a.lng, b.lat, b.lng) for a in bucket for b in bucket)
        assert emirate_spread < 50, "a cluster spans two emirates"


def test_first_cluster_is_the_one_nearest_the_start_location():
    near = [place(i, f"Near {i}", lat=25.20, lng=55.27 + i * 0.01) for i in range(1, 5)]
    far = [place(10 + i, f"Far {i}", lat=25.50, lng=56.36 + i * 0.01) for i in range(1, 5)]
    buckets = cluster_by_proximity(near + far, 2, ORIGIN)
    assert all(p.name.startswith("Near") for p in buckets[0])


# --- pricing -----------------------------------------------------------------------------------


def test_price_bands_are_checked_in_order():
    banded = place(
        1,
        price_bands=({"max_age": 2, "price": 0}, {"max_age": 12, "price": 155}, {"max_age": None, "price": 199}),
    )
    assert price_for_age(banded, 1) == 0
    assert price_for_age(banded, 2) == 0
    assert price_for_age(banded, 3) == 155
    assert price_for_age(banded, 12) == 155
    assert price_for_age(banded, 13) == 199


def test_without_bands_a_thirteen_year_old_pays_the_adult_rate():
    flat = place(1, price_adult=199, price_child=155, price_bands=None)
    assert price_for_age(flat, 12) == 155
    assert price_for_age(flat, 13) == 199


def test_infant_produces_a_free_chip_and_older_children_do_not():
    banded = place(
        1,
        price_bands=({"max_age": 2, "price": 0}, {"max_age": 12, "price": 155}, {"max_age": None, "price": 199}),
    )
    with_infant = slot_cost_breakdown(banded, family(2, (1, 7)), travel_in=38.0)
    assert with_infant.free_children == 1
    assert with_infant.free_under_age == 3
    assert [c["tone"] for c in with_infant.chips] == ["adult", "child", "free"]
    assert with_infant.chips[2]["label"] == "1 child free (under 3)"
    assert with_infant.total == pytest.approx(199 * 2 + 155 + 38.0)

    without_infant = slot_cost_breakdown(banded, family(2, (7, 9)))
    assert without_infant.free_children == 0
    assert [c["tone"] for c in without_infant.chips] == ["adult", "child"]


def test_chip_labels_are_singular_or_plural_correctly():
    flat = place(1, price_adult=100, price_child=50)
    one_each = slot_cost_breakdown(flat, [Attendee("adult", 30), Attendee("child", 8)])
    assert one_each.chips[0]["label"] == "1 adult · AED 100"
    assert one_each.chips[1]["label"] == "1 child · AED 50"


# --- day assembly ------------------------------------------------------------------------------


def _catalog() -> list:
    """A small but realistic pool: activities, budget relief and both meal categories."""
    return [
        place(1, "Aquarium", "aquarium", lat=25.197, lng=55.279, price_adult=199, price_child=155,
              open_time="10:00", close_time="22:00", avg_duration_min=90, kid_score=0.9),
        place(2, "Park", "park", lat=25.21, lng=55.28, price_adult=10, price_child=5,
              open_time="08:00", close_time="22:00", avg_duration_min=120, kid_score=0.95),
        place(3, "Beach", "beach", lat=25.19, lng=55.26, price_adult=0, price_child=0,
              open_time="06:00", close_time="22:00", avg_duration_min=90, kid_score=0.85),
        place(4, "Museum", "museum", lat=25.22, lng=55.29, price_adult=50, price_child=20,
              open_time="10:00", close_time="20:00", avg_duration_min=90, kid_score=0.6),
        place(5, "Cafe", "casual_dining", lat=25.20, lng=55.275, price_adult=85, price_child=45,
              open_time="08:00", close_time="23:00", avg_duration_min=60, kid_score=0.7),
        place(6, "Grill", "casual_dining", lat=25.205, lng=55.272, price_adult=60, price_child=30,
              open_time="11:00", close_time="23:30", avg_duration_min=60, kid_score=0.7),
        place(7, "Bistro", "fine_dining", lat=25.203, lng=55.271, price_adult=300, price_child=0,
              min_age=12, open_time="18:00", close_time="23:30", avg_duration_min=120,
              romance_score=0.95, kid_score=0.0),
    ]


def _plan(attendees, event_type="birthday", *, days=2, budget=3000.0, travel=None):
    profile = build_profile(attendees, event_type)
    return (
        generate_plan(
            _catalog(),
            profile,
            travel or fixed_travel(20),
            start_date=TOMORROW,
            num_days=days,
            total_budget=budget,
            origin=ORIGIN,
        ),
        profile,
    )


def test_generated_plan_validates_cleanly():
    plan, profile = _plan(family(2, (7, 13)))
    assert validate_plan(plan, profile) == []
    assert any(day.slots for day in plan.days)


def test_slots_never_overlap_and_leave_room_for_the_drive():
    plan, _ = _plan(family(2, (7, 13)), travel=fixed_travel(35))
    for day in plan.days:
        required = {s.to_position: s.info.duration_min for s in day.segments}
        ordered = sorted(day.slots, key=lambda s: s.start_min)
        for previous, current in zip(ordered, ordered[1:]):
            gap = current.start_min - previous.end_min
            assert gap >= 0, "slots overlap"
            assert gap >= required[current.position], "not enough time for the drive"


def test_every_venue_is_open_for_the_whole_slot():
    plan, _ = _plan(family(2, (7, 13)))
    for day in plan.days:
        for slot in day.slots:
            assert slot.start_min >= slot.place.opens_at
            assert slot.end_min <= slot.place.closes_at


def test_a_seven_year_old_never_lands_in_a_twelve_plus_venue():
    plan, _ = _plan(family(2, (7,)))
    booked = {slot.place.name for day in plan.days for slot in day.slots}
    assert "Bistro" not in booked, "min_age 12 venue booked for a 7-year-old"


def test_adults_only_anniversary_reaches_the_fine_dining_slot():
    plan, _ = _plan(family(2), "anniversary", days=1, budget=2000.0)
    booked = {slot.place.name for day in plan.days for slot in day.slots}
    assert "Bistro" in booked


def test_fine_dining_is_never_scheduled_in_the_morning():
    plan, _ = _plan(family(2), "anniversary", days=2, budget=4000.0)
    for day in plan.days:
        for slot in day.slots:
            if slot.place.category == "fine_dining":
                assert slot.start_min >= 17 * 60


def test_meal_slots_are_placed_inside_their_windows():
    plan, profile = _plan(family(2, (7, 13)), days=1, budget=3000.0)
    windows = {label: (start, end) for label, start, end in profile.meal_windows}
    meals = [s for day in plan.days for s in day.slots if s.place.category in DINING_CATEGORIES]
    assert meals, "expected at least one meal slot"
    for meal in meals:
        assert any(start <= meal.start_min <= end + 60 for start, end in windows.values())


def test_young_children_get_a_protected_midday_gap():
    plan, _ = _plan(family(2, (5,)), days=1, budget=3000.0)
    day = plan.days[0]
    ordered = sorted(day.slots, key=lambda s: s.start_min)
    crossing = [
        (a, b) for a, b in zip(ordered, ordered[1:]) if a.end_min <= 13 * 60 <= b.start_min
    ]
    if crossing:
        previous, following = crossing[0]
        assert following.start_min - previous.end_min >= 60


def test_no_place_is_used_twice_across_the_trip():
    plan, _ = _plan(family(2, (7, 13)), days=3, budget=6000.0)
    booked = [slot.place.id for day in plan.days for slot in day.slots]
    assert len(booked) == len(set(booked))


def test_the_trip_stays_within_its_cap():
    plan, _ = _plan(family(2, (7, 13)), days=3, budget=800.0)
    assert plan.total_cost <= 800.0


def test_a_tight_budget_substitutes_cheaper_places_rather_than_failing():
    """The catalog carries budget options in every category precisely so this can succeed."""
    plan, _ = _plan(family(2, (7, 13)), days=1, budget=400.0)
    assert any(day.slots for day in plan.days), "planner gave up instead of substituting"
    assert plan.total_cost <= 400.0


def test_planning_is_deterministic():
    first, _ = _plan(family(2, (7, 13)), days=3)
    second, _ = _plan(family(2, (7, 13)), days=3)
    as_tuple = lambda p: [  # noqa: E731
        [(s.place.id, s.start_min, s.end_min) for s in d.slots] for d in p.days
    ]
    assert as_tuple(first) == as_tuple(second)


def test_real_distances_still_produce_a_valid_plan():
    plan, profile = _plan(family(2, (7, 13)), days=2, travel=distance_travel())
    assert validate_plan(plan, profile) == []


def test_rejects_more_than_five_days():
    with pytest.raises(ValueError, match="between 1 and 5"):
        _plan(family(2), days=6)


def test_empty_catalog_yields_an_empty_plan_with_a_warning():
    profile = build_profile(family(2, (7,)), "birthday")
    plan = generate_plan(
        [], profile, fixed_travel(), start_date=TOMORROW, num_days=2,
        total_budget=2000.0, origin=ORIGIN,
    )
    assert all(day.slots == [] for day in plan.days)
    assert plan.warnings
    assert validate_plan(plan, profile) == []


# --- derived labels ----------------------------------------------------------------------------


def test_day_theme_names_the_two_dominant_categories_ignoring_meals():
    plan, _ = _plan(family(2, (7, 13)), days=1, budget=3000.0)
    theme = day_theme(plan.days[0])
    assert theme != "Open day"
    assert "Food" not in theme


def test_day_theme_of_an_empty_day():
    from app.services.planner import DayPlan

    assert day_theme(DayPlan(day_index=0, day_date=TOMORROW)) == "Open day"


# --- budget summary ----------------------------------------------------------------------------


def test_summary_splits_activities_food_and_travel():
    plan, _ = _plan(family(2, (7, 13)), days=2, budget=3000.0)
    summary = summarise(plan.days, 3000.0)

    categories = summary["categories"]
    assert summary["total"] == pytest.approx(
        categories["activities"] + categories["food"] + categories["travel"], abs=0.05
    )
    assert summary["remaining"] == pytest.approx(3000.0 - summary["total"], abs=0.01)
    assert summary["over_budget"] is False
    assert len(summary["per_day"]) == 2
    assert sum(summary["per_day"]) == pytest.approx(summary["total"], abs=0.05)


# --- seasonal closure, heat and rest ------------------------------------------------------------


def test_a_venue_closed_that_month_is_never_scheduled():
    """A correctness filter, not a preference: the venue is shut."""
    from datetime import date as _date

    summer_shut = place(1, "Global Village", "mall", price_adult=25, open_time="16:00",
                        close_time="23:00", closed_months=(5, 6, 7, 8, 9), kid_score=0.99)
    open_all_year = place(2, "Aquarium", "aquarium", price_adult=100, open_time="10:00",
                          close_time="22:00", kid_score=0.5)
    profile = build_profile(family(2, (7,)), "birthday")

    august = generate_plan(
        [summer_shut, open_all_year], profile, fixed_travel(10), start_date=_date(2026, 8, 10),
        num_days=1, total_budget=5000.0, origin=ORIGIN,
    )
    december = generate_plan(
        [summer_shut, open_all_year], profile, fixed_travel(10), start_date=_date(2026, 12, 10),
        num_days=1, total_budget=5000.0, origin=ORIGIN,
    )

    assert "Global Village" not in {s.place.name for d in august.days for s in d.slots}
    assert "Global Village" in {s.place.name for d in december.days for s in d.slots}


def test_the_validator_flags_a_seasonally_closed_slot():
    from datetime import date as _date

    from app.services.validator import VENUE_CLOSED_SEASONALLY, validate_day
    from app.services.budget import slot_cost_breakdown
    from app.services.planner import DayPlan, PlannedSlot

    shut = place(1, "Miracle Garden", "park", closed_months=(6, 7, 8, 9),
                 open_time="09:00", close_time="21:00")
    attendees = family(2)
    day = DayPlan(day_index=0, day_date=_date(2026, 7, 15))
    day.slots = [
        PlannedSlot(place=shut, day_index=0, position=0, start_min=600, end_min=720,
                    score=1.0, cost=slot_cost_breakdown(shut, attendees))
    ]
    codes = {v.code for v in validate_day(day, build_profile(attendees))}
    assert VENUE_CLOSED_SEASONALLY in codes


def test_a_summer_midday_prefers_somewhere_air_conditioned():
    """Spec §10 calls malls and indoor attractions the midday heat fallback."""
    from datetime import date as _date

    outdoor = place(1, "Open Park", "park", price_adult=10, open_time="08:00",
                    close_time="20:00", kid_score=0.6)
    indoor = place(2, "Cool Mall", "mall", price_adult=10, open_time="08:00",
                   close_time="20:00", kid_score=0.6, indoor=True)
    profile = build_profile(family(2, (7,)), "birthday")

    july = generate_plan([outdoor, indoor], profile, fixed_travel(10),
                         start_date=_date(2026, 7, 10), num_days=1,
                         total_budget=5000.0, origin=ORIGIN)
    booked = [s.place.name for d in july.days for s in d.slots]
    midday = [s for d in july.days for s in d.slots if 11 * 60 <= s.start_min <= 16 * 60]
    assert booked, "expected a plan"
    if midday:
        assert any(s.place.indoor for s in midday), "no air-conditioned option taken at midday"


def test_a_due_meal_beats_the_midday_rest():
    """The rest used to fire at 13:00 and push the cursor past the lunch window, so a family with
    a small child got a nap instead of lunch every single day."""
    catalog = _catalog()
    profile = build_profile(family(2, (5,)), "birthday")
    assert profile.needs_midday_rest is True

    plan = generate_plan(catalog, profile, fixed_travel(10), start_date=TOMORROW,
                         num_days=1, total_budget=4000.0, origin=ORIGIN)
    meals = [s for d in plan.days for s in d.slots if s.place.category in DINING_CATEGORIES]
    assert meals, "the midday rest swallowed the lunch window"


def test_a_cluster_without_restaurants_is_topped_up():
    """Clustering is geographic and blind to category, so a day can come out with no dining."""
    from app.services.planner import ensure_dining

    cluster = [place(i, f"Sight {i}", "park", lat=25.2 + i * 0.01, lng=55.2) for i in range(1, 6)]
    dining = [place(50 + i, f"Cafe {i}", "casual_dining", lat=25.9, lng=55.9) for i in range(3)]

    assert not [p for p in cluster if p.category in DINING_CATEGORIES]
    topped = ensure_dining(cluster, dining, minimum=3)
    assert len([p for p in topped if p.category in DINING_CATEGORIES]) == 3
    assert all(p in topped for p in cluster), "the original cluster must be preserved"


def test_ensure_dining_leaves_a_cluster_that_already_has_enough_alone():
    from app.services.planner import ensure_dining

    cluster = [place(i, f"Cafe {i}", "casual_dining") for i in range(1, 5)]
    assert ensure_dining(cluster, [], minimum=3) is cluster
