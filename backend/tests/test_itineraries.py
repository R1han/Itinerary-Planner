"""Itinerary API: intake gating, generation, single-slot editing and the action chips.

These run against the real seeded catalog so the assertions are about the actual product, not a
toy fixture.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from app.models import Place
from app.seed import default_price_bands

TOMORROW = (date.today() + timedelta(days=1)).isoformat()
ABU_DHABI = {"start_lat": 24.4539, "start_lng": 54.3773}

DATA = Path(__file__).resolve().parent.parent / "app" / "data" / "places.json"


@pytest.fixture
def catalog(db) -> int:
    """Load the real 164-place catalog (no embeddings — retrieval falls back to keywords)."""
    rows = json.loads(DATA.read_text(encoding="utf-8"))
    for row in rows:
        db.add(
            Place(
                name=row["name"], emirate=row["emirate"], lat=row["lat"], lng=row["lng"],
                category=row["category"], price_adult=row.get("price_adult", 0),
                price_child=row.get("price_child", 0),
                price_bands=row.get("price_bands") or default_price_bands(row),
                min_age=row.get("min_age", 0), open_time=row.get("open_time", "09:00"),
                close_time=row.get("close_time", "22:00"),
                avg_duration_min=row.get("avg_duration_min", 90), tags=row.get("tags", []),
                kid_score=row.get("kid_score", 0.5), teen_score=row.get("teen_score", 0.5),
                romance_score=row.get("romance_score", 0.5),
                category_icon=row["category"], description=row.get("description", ""),
            )
        )
    db.commit()
    return len(rows)


@pytest.fixture
def mixed_family(client, make_user, catalog):
    headers, user = make_user("family@rihla.app", "Family")
    client.put(
        "/family",
        headers=headers,
        json={
            "members": [
                {"role": "adult", "age": 34, "name": "Dad"},
                {"role": "adult", "age": 31, "name": "Mom"},
                {"role": "child", "age": 7, "name": "Aisha"},
                {"role": "child", "age": 13, "name": "Omar"},
            ]
        },
    )
    for pref in (
        {"kind": "like", "subject": "animals and zoos", "category": "aquarium", "strength": 0.9},
        {"kind": "dislike", "subject": "very loud thrill rides", "category": "adventure",
         "strength": 0.7},
    ):
        client.post("/preferences", headers=headers, json=pref)
    return headers, user


@pytest.fixture
def couple(client, make_user, catalog):
    headers, user = make_user("couple@rihla.app", "Couple")
    client.put(
        "/family",
        headers=headers,
        json={"members": [{"role": "adult", "age": 36}, {"role": "adult", "age": 38}]},
    )
    client.post(
        "/preferences",
        headers=headers,
        json={"kind": "like", "subject": "fine dining", "category": "fine_dining", "strength": 0.9},
    )
    return headers, user


def generate(client, headers, **overrides) -> dict:
    body = {"start_date": TOMORROW, "num_days": 3, "total_budget": 3500.0, **ABU_DHABI, **overrides}
    response = client.post("/itineraries/generate", headers=headers, json=body)
    assert response.status_code == 201, response.text
    return response.json()


# --- intake gating -----------------------------------------------------------------------------


def test_generation_is_refused_until_the_family_is_known(client, make_user, catalog):
    headers, _ = make_user("nofamily@rihla.app")
    response = client.post(
        "/itineraries/generate",
        headers=headers,
        json={"start_date": TOMORROW, "num_days": 2, "total_budget": 2000.0, **ABU_DHABI},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "intake_incomplete"
    assert "adults" in response.json()["detail"]["missing_fields"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_days", 6),
        ("num_days", 0),
        ("total_budget", -100),
        ("start_date", (date.today() - timedelta(days=1)).isoformat()),
        ("start_lat", 48.85),  # Paris — outside the UAE bounding box
    ],
)
def test_boundary_validation_rejects_impossible_requests(client, mixed_family, field, value):
    headers, _ = mixed_family
    body = {"start_date": TOMORROW, "num_days": 2, "total_budget": 2000.0, **ABU_DHABI, field: value}
    assert client.post("/itineraries/generate", headers=headers, json=body).status_code == 422


# --- generation --------------------------------------------------------------------------------


def test_a_generated_plan_satisfies_every_hard_constraint(client, mixed_family):
    headers, _ = mixed_family
    plan = generate(client, headers)

    assert len(plan["days"]) == 3
    assert plan["budget"]["total"] <= plan["budget"]["cap"]
    assert plan["budget"]["over_budget"] is False

    for day in plan["days"]:
        travel_into = {s["to_slot_id"]: s["duration_min"] for s in day["segments"]}
        slots = sorted(day["slots"], key=lambda s: s["start_time"])
        for previous, current in zip(slots, slots[1:]):
            assert current["start_time"] >= previous["end_time"], "slots overlap"
            required = travel_into.get(current["id"], 0)
            gap = _minutes(current["start_time"]) - _minutes(previous["end_time"])
            assert gap >= required, "not enough time for the drive"

        for slot in day["slots"]:
            place = slot["place"]
            assert slot["start_time"] >= place["open_time"]
            assert place["min_age"] <= 7, "a 7-year-old was booked into an age-restricted venue"


def _minutes(hhmm: str) -> int:
    hours, _, mins = hhmm.partition(":")
    return int(hours) * 60 + int(mins)


def test_the_response_carries_everything_the_workspace_renders(client, mixed_family):
    """Spec §9: geometry per segment and image_url per place, so there are no extra round trips."""
    headers, _ = mixed_family
    plan = generate(client, headers)
    day = next(d for d in plan["days"] if d["slots"])

    assert day["theme"] and day["theme"] != "Open day"
    assert day["subtotal"] > 0
    assert day["driving_total_min"] > 0
    assert set(plan["budget"]["categories"]) == {"activities", "food", "travel"}

    # The budget block must add up three ways, or the BudgetPanel's bar, legend and per-day
    # mini-bars disagree with each other on screen.
    budget = plan["budget"]
    categories = budget["categories"]
    assert budget["total"] == pytest.approx(
        categories["activities"] + categories["food"] + categories["travel"], abs=0.05
    )
    assert sum(budget["per_day"]) == pytest.approx(budget["total"], abs=0.05)
    assert budget["remaining"] == pytest.approx(budget["cap"] - budget["total"], abs=0.01)
    assert len(budget["per_day"]) == plan["num_days"]

    slot = day["slots"][0]
    assert {"id", "start_time", "end_time", "cost_breakdown", "place"} <= set(slot)
    assert "image_url" in slot["place"]
    assert slot["cost_breakdown"]["chips"], "cost chips drive the slot card"

    for segment in day["segments"]:
        assert segment["geometry_json"] is not None
        assert "estimated" in segment


def test_a_birthday_plan_for_young_children_skews_to_parks_and_wildlife(client, mixed_family):
    headers, _ = mixed_family
    event = client.post(
        "/events",
        headers=headers,
        json={"title": "Aisha's 7th birthday", "event_type": "birthday", "date": TOMORROW,
              "notes": "loves animals, afraid of loud rides"},
    ).json()

    plan = generate(client, headers, event_id=event["id"])
    categories = [s["place"]["category"] for d in plan["days"] for s in d["slots"]]

    assert any(c in {"park", "aquarium", "beach"} for c in categories)
    assert "adventure" not in categories, "a disliked category was still booked"
    assert client.get("/events", headers=headers).json()[0]["planned"] is True


def test_an_adults_only_anniversary_is_romantic_and_evening_heavy(client, couple):
    headers, _ = couple
    event = client.post(
        "/events",
        headers=headers,
        json={"title": "Anniversary weekend", "event_type": "anniversary", "date": TOMORROW},
    ).json()

    plan = generate(client, headers, event_id=event["id"], num_days=2, total_budget=4000.0)
    slots = [s for d in plan["days"] for s in d["slots"]]
    categories = {s["place"]["category"] for s in slots}

    assert "fine_dining" in categories
    for slot in slots:
        if slot["place"]["category"] == "fine_dining":
            assert slot["start_time"] >= "17:00"
        assert slot["cost_breakdown"]["free_children"] == 0
        assert slot["cost_breakdown"]["children"] == []


def test_a_teen_visit_reaches_waterparks_or_adventure(client, make_user, catalog):
    headers, _ = make_user("teens@rihla.app")
    client.put(
        "/family",
        headers=headers,
        json={
            "members": [
                {"role": "adult", "age": 40},
                {"role": "child", "age": 14},
                {"role": "child", "age": 16},
            ]
        },
    )
    event = client.post(
        "/events",
        headers=headers,
        json={"title": "Cousins visiting", "event_type": "family_visit", "date": TOMORROW},
    ).json()

    plan = generate(client, headers, event_id=event["id"], num_days=4, total_budget=6000.0)
    categories = {s["place"]["category"] for d in plan["days"] for s in d["slots"]}
    assert categories & {"waterpark", "theme_park", "adventure"}


def test_a_tight_budget_substitutes_rather_than_failing(client, mixed_family):
    headers, _ = mixed_family
    plan = generate(client, headers, num_days=2, total_budget=350.0)
    assert any(d["slots"] for d in plan["days"]), "planner gave up instead of substituting"
    assert plan["budget"]["total"] <= 350.0


# --- action chips ------------------------------------------------------------------------------


def test_no_action_chips_are_offered(client, mixed_family):
    """The "Cheaper Day N" and "Add prayer breaks" chips were deliberately withdrawn.

    Both actions are still reachable — `/days/{i}/cheaper` and the chat's make_day_cheaper and
    add_prayer_breaks — so this is about what the plan volunteers, not what it can do. The key
    stays in the payload because the client types it as a list and reads it on every day patch.
    """
    headers, _ = mixed_family
    plan = generate(client, headers)

    assert plan["suggestions"] == []


def test_cheaper_day_reduces_that_day_without_breaking_the_plan(client, mixed_family):
    headers, _ = mixed_family
    plan = generate(client, headers)
    target = max(range(len(plan["budget"]["per_day"])), key=lambda i: plan["budget"]["per_day"][i])
    before = plan["budget"]["per_day"][target]

    response = client.post(
        f"/itineraries/{plan['id']}/days/{target}/cheaper", headers=headers
    )
    assert response.status_code == 200
    after = response.json()

    assert after["budget"]["per_day"][target] <= before
    assert after["budget"]["over_budget"] is False
    assert after["days"][target]["slots"], "the day was emptied rather than made cheaper"


def test_prayer_breaks_reflow_the_day_and_keep_it_valid(client, mixed_family):
    headers, _ = mixed_family
    plan = generate(client, headers)

    response = client.post(f"/itineraries/{plan['id']}/prayer-breaks", headers=headers)
    assert response.status_code == 200
    after = response.json()

    assert after["budget"]["over_budget"] is False
    for day in after["days"]:
        slots = sorted(day["slots"], key=lambda s: s["start_time"])
        for previous, current in zip(slots, slots[1:]):
            assert current["start_time"] >= previous["end_time"]


# --- single-slot editing -----------------------------------------------------------------------


def _first_editable(plan: dict) -> tuple[int, dict]:
    for day in plan["days"]:
        if len(day["slots"]) >= 2:
            return day["day_index"], day["slots"][1]
    day = next(d for d in plan["days"] if d["slots"])
    return day["day_index"], day["slots"][0]


def test_alternatives_fit_the_window_and_the_remaining_budget(client, mixed_family):
    headers, _ = mixed_family
    plan = generate(client, headers)
    day_index, slot = _first_editable(plan)

    response = client.get(
        f"/itineraries/{plan['id']}/slots/{slot['id']}/alternatives", headers=headers
    )
    assert response.status_code == 200
    options = response.json()
    assert len(options) <= 3

    booked = {s["place_id"] for d in plan["days"] for s in d["slots"]}
    for option in options:
        assert option["place"]["id"] not in booked, "offered a place already in the plan"
        assert option["place"]["min_age"] <= 7
        assert option["start_time"] >= option["place"]["open_time"]


def test_removing_a_slot_returns_the_whole_day_and_a_fresh_budget(client, mixed_family):
    headers, _ = mixed_family
    plan = generate(client, headers)
    day_index, slot = _first_editable(plan)
    before_total = plan["budget"]["total"]

    response = client.patch(
        f"/itineraries/{plan['id']}/slots/{slot['id']}",
        headers=headers,
        json={"action": "remove"},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["day"]["day_index"] == day_index
    assert slot["id"] not in [s["id"] for s in body["day"]["slots"]]
    assert body["budget"]["total"] < before_total, "budget did not update server-side"
    assert "suggestions" in body


def test_an_edit_touches_only_its_own_day(client, mixed_family):
    headers, _ = mixed_family
    plan = generate(client, headers)
    day_index, slot = _first_editable(plan)

    untouched_before = {
        d["day_index"]: [(s["place_id"], s["start_time"]) for s in d["slots"]]
        for d in plan["days"]
        if d["day_index"] != day_index
    }

    client.patch(
        f"/itineraries/{plan['id']}/slots/{slot['id']}", headers=headers, json={"action": "remove"}
    )
    after = client.get(f"/itineraries/{plan['id']}", headers=headers).json()

    untouched_after = {
        d["day_index"]: [(s["place_id"], s["start_time"]) for s in d["slots"]]
        for d in after["days"]
        if d["day_index"] != day_index
    }
    assert untouched_before == untouched_after, "an edit cascaded into another day"


def test_replacing_a_slot_swaps_the_place_and_recomputes_its_travel(client, mixed_family):
    headers, _ = mixed_family
    plan = generate(client, headers)
    day_index, slot = _first_editable(plan)

    options = client.get(
        f"/itineraries/{plan['id']}/slots/{slot['id']}/alternatives", headers=headers
    ).json()
    if not options:
        pytest.skip("no alternative fits this window")

    replacement = options[0]["place"]["id"]
    response = client.patch(
        f"/itineraries/{plan['id']}/slots/{slot['id']}",
        headers=headers,
        json={"action": "replace", "place_id": replacement},
    )
    assert response.status_code == 200
    day = response.json()["day"]

    assert replacement in [s["place_id"] for s in day["slots"]]
    assert len(day["segments"]) == len(day["slots"]), "a slot lost its travel segment"


def test_replacing_with_an_age_restricted_place_is_rejected(client, mixed_family, db):
    headers, _ = mixed_family
    plan = generate(client, headers)
    _, slot = _first_editable(plan)

    adults_only = db.query(Place).filter(Place.min_age >= 12).first()
    response = client.patch(
        f"/itineraries/{plan['id']}/slots/{slot['id']}",
        headers=headers,
        json={"action": "replace", "place_id": adults_only.id},
    )
    assert response.status_code == 400
    assert "requires age" in response.json()["detail"]


def test_adjusting_a_time_keeps_the_day_valid(client, mixed_family):
    headers, _ = mixed_family
    plan = generate(client, headers)
    day_index, slot = _first_editable(plan)

    response = client.patch(
        f"/itineraries/{plan['id']}/slots/{slot['id']}",
        headers=headers,
        json={"action": "adjust", "start_time": "15:00"},
    )
    assert response.status_code == 200
    day = response.json()["day"]

    slots = sorted(day["slots"], key=lambda s: s["start_time"])
    for previous, current in zip(slots, slots[1:]):
        assert current["start_time"] >= previous["end_time"]
    for entry in day["slots"]:
        assert entry["start_time"] >= entry["place"]["open_time"]


def test_a_malformed_patch_is_rejected(client, mixed_family):
    headers, _ = mixed_family
    plan = generate(client, headers)
    _, slot = _first_editable(plan)

    assert client.patch(
        f"/itineraries/{plan['id']}/slots/{slot['id']}",
        headers=headers,
        json={"action": "replace"},  # no place_id
    ).status_code == 422


# --- ownership ---------------------------------------------------------------------------------


def test_another_user_cannot_read_or_edit_the_plan(client, mixed_family, make_user):
    headers, _ = mixed_family
    plan = generate(client, headers)
    _, slot = _first_editable(plan)
    intruder, _ = make_user("intruder@rihla.app")

    assert client.get(f"/itineraries/{plan['id']}", headers=intruder).status_code == 404
    assert client.patch(
        f"/itineraries/{plan['id']}/slots/{slot['id']}",
        headers=intruder,
        json={"action": "remove"},
    ).status_code == 404
    assert client.get(
        f"/itineraries/{plan['id']}/slots/{slot['id']}/alternatives", headers=intruder
    ).status_code == 404
    assert client.post(
        f"/itineraries/{plan['id']}/days/0/cheaper", headers=intruder
    ).status_code == 404
    assert client.get("/itineraries", headers=intruder).json() == []


# --- transport mode: own car vs taxi ------------------------------------------------------------


def test_a_plan_starts_out_priced_as_taxi_fares(client, mixed_family):
    headers, _ = mixed_family
    plan = generate(client, headers)
    assert plan["transport_mode"] == "taxi"
    assert plan["budget"]["categories"]["travel"] > 0


def test_switching_to_your_own_car_reprices_the_plan_in_place(client, mixed_family):
    """The reported bug: the assistant said "Noted!" and the AED 419 of taxi fare stayed put."""
    headers, _ = mixed_family
    plan = generate(client, headers)
    before = plan["budget"]["categories"]["travel"]

    response = client.post(
        f"/itineraries/{plan['id']}/transport", headers=headers, json={"mode": "own_car"}
    )
    assert response.status_code == 200, response.text
    after = response.json()

    assert after["transport_mode"] == "own_car"
    assert after["budget"]["categories"]["travel"] < before, "travel was not repriced"
    assert after["budget"]["total"] < plan["budget"]["total"]


def test_repricing_moves_the_money_without_moving_the_trip(client, mixed_family):
    """Same route, same times, same places — only the arithmetic changes."""
    headers, _ = mixed_family
    plan = generate(client, headers)
    client.post(
        f"/itineraries/{plan['id']}/transport", headers=headers, json={"mode": "own_car"}
    )
    after = client.get(f"/itineraries/{plan['id']}", headers=headers).json()

    def shape(payload):
        return [
            (slot["place"]["id"], slot["start_time"], slot["end_time"])
            for day in payload["days"]
            for slot in day["slots"]
        ]

    assert shape(after) == shape(plan)
    assert [seg["duration_min"] for day in after["days"] for seg in day["segments"]] == [
        seg["duration_min"] for day in plan["days"] for seg in day["segments"]
    ]


def test_switching_back_to_a_taxi_restores_the_fares(client, mixed_family):
    headers, _ = mixed_family
    plan = generate(client, headers)
    for mode in ("own_car", "taxi"):
        back = client.post(
            f"/itineraries/{plan['id']}/transport", headers=headers, json={"mode": mode}
        ).json()
    assert back["budget"]["categories"]["travel"] == pytest.approx(
        plan["budget"]["categories"]["travel"], abs=0.01
    )


def test_an_unknown_transport_mode_is_rejected(client, mixed_family):
    headers, _ = mixed_family
    plan = generate(client, headers)
    response = client.post(
        f"/itineraries/{plan['id']}/transport", headers=headers, json={"mode": "camel"}
    )
    assert response.status_code == 422


def test_transport_mode_cannot_be_changed_on_someone_elses_plan(client, mixed_family, make_user):
    headers, _ = mixed_family
    plan = generate(client, headers)
    intruder, _ = make_user("intruder@rihla.app")
    response = client.post(
        f"/itineraries/{plan['id']}/transport", headers=intruder, json={"mode": "own_car"}
    )
    assert response.status_code == 404


def test_a_bigger_family_pays_for_a_bigger_vehicle(client, make_user, catalog):
    """Six people do not fit in a saloon, and the fare should say so."""
    small, _ = make_user("four@rihla.app")
    client.put("/family", headers=small, json={"members": [
        {"role": "adult", "age": 34}, {"role": "adult", "age": 31},
        {"role": "child", "age": 7}, {"role": "child", "age": 13},
    ]})
    big, _ = make_user("six@rihla.app")
    client.put("/family", headers=big, json={"members": [
        {"role": "adult", "age": 34}, {"role": "adult", "age": 31},
        {"role": "child", "age": 7}, {"role": "child", "age": 13},
        {"role": "child", "age": 10}, {"role": "child", "age": 4},
    ]})

    four = generate(client, small, num_days=1, total_budget=6000.0)
    six = generate(client, big, num_days=1, total_budget=6000.0)

    assert four["vehicle"] == "standard"
    assert six["vehicle"] == "6-seater"


# --- putting something new into a day -------------------------------------------------------------


def _day_with_room(plan: dict) -> int:
    return min(range(len(plan["days"])), key=lambda i: len(plan["days"][i]["slots"]))


def _remove_one(client, headers, plan) -> tuple[int, dict]:
    """Free up a slot, which is the situation add_stop exists for."""
    day_index, slot = _first_editable(plan)
    client.patch(
        f"/itineraries/{plan['id']}/slots/{slot['id']}", headers=headers, json={"action": "remove"}
    )
    return day_index, slot


def test_a_removed_stop_can_be_put_back(client, mixed_family):
    """Removing used to be one-way from chat: nothing could add anything to a day."""
    headers, _ = mixed_family
    plan = generate(client, headers, num_days=2, total_budget=6000.0)
    day_index, removed = _remove_one(client, headers, plan)

    after_removal = client.get(f"/itineraries/{plan['id']}", headers=headers).json()
    count_before = len(after_removal["days"][day_index]["slots"])

    response = client.post(
        f"/itineraries/{plan['id']}/days/{day_index}/stops", headers=headers, json={}
    )
    assert response.status_code == 200, response.text
    assert len(response.json()["days"][day_index]["slots"]) == count_before + 1


def test_a_stop_can_be_added_by_category(client, mixed_family):
    """"Replace the park with shopping" in the shape the chat actually needs it."""
    headers, _ = mixed_family
    plan = generate(client, headers, num_days=2, total_budget=6000.0)
    day_index, removed = _remove_one(client, headers, plan)
    wanted = removed["place"]["category"]

    response = client.post(
        f"/itineraries/{plan['id']}/days/{day_index}/stops",
        headers=headers,
        json={"category": wanted},
    )
    assert response.status_code == 200, response.text
    categories = [s["place"]["category"] for s in response.json()["days"][day_index]["slots"]]
    assert wanted in categories, categories


def test_an_added_stop_keeps_the_day_valid(client, mixed_family):
    """It has to be scheduled, not merely appended — times in order, no overlaps."""
    headers, _ = mixed_family
    plan = generate(client, headers, num_days=2, total_budget=6000.0)
    day_index, _ = _remove_one(client, headers, plan)

    after = client.post(
        f"/itineraries/{plan['id']}/days/{day_index}/stops", headers=headers, json={}
    ).json()

    slots = sorted(after["days"][day_index]["slots"], key=lambda s: s["start_time"])
    for a, b in zip(slots, slots[1:]):
        assert a["end_time"] <= b["start_time"], (a["place"]["name"], b["place"]["name"])


def test_a_full_day_refuses_a_new_stop_rather_than_forcing_one_in(client, mixed_family):
    """A freshly generated day is packed. Saying so beats overlapping two stops."""
    headers, _ = mixed_family
    plan = generate(client, headers, num_days=2, total_budget=6000.0)
    response = client.post(
        f"/itineraries/{plan['id']}/days/0/stops", headers=headers, json={"category": "mall"}
    )
    assert response.status_code == 422
    assert "fits" in response.json()["detail"]


def test_adding_a_stop_that_cannot_fit_is_an_error_not_a_silent_no_op(client, mixed_family):
    headers, _ = mixed_family
    plan = generate(client, headers, num_days=2, total_budget=6000.0)
    response = client.post(
        f"/itineraries/{plan['id']}/days/0/stops", headers=headers, json={"category": "nonsense"}
    )
    assert response.status_code == 422


def test_a_stop_cannot_be_added_to_someone_elses_plan(client, mixed_family, make_user):
    headers, _ = mixed_family
    plan = generate(client, headers, num_days=2, total_budget=6000.0)
    intruder, _ = make_user("intruder2@rihla.app")
    response = client.post(
        f"/itineraries/{plan['id']}/days/0/stops", headers=intruder, json={"category": "mall"}
    )
    assert response.status_code == 404


def test_replacing_a_slot_by_category_swaps_it_for_that_kind_of_place(client, mixed_family):
    """"Replace the park with shopping" — one call, no place ids to juggle."""
    headers, _ = mixed_family
    plan = generate(client, headers, num_days=2, total_budget=6000.0)
    day_index, slot = _first_editable(plan)

    response = client.patch(
        f"/itineraries/{plan['id']}/slots/{slot['id']}",
        headers=headers,
        json={"action": "replace", "category": "mall"},
    )
    assert response.status_code == 200, response.text

    names = {s["place"]["id"]: s["place"] for s in response.json()["day"]["slots"]}
    assert slot["place"]["id"] not in names
    assert any(p["category"] == "mall" for p in names.values()), [
        p["name"] for p in names.values()
    ]


def test_replace_still_accepts_an_explicit_place_id(client, mixed_family):
    """The strip UI picks from a list and sends an id; that path must not change."""
    headers, _ = mixed_family
    plan = generate(client, headers, num_days=2, total_budget=6000.0)
    day_index, slot = _first_editable(plan)
    options = client.get(
        f"/itineraries/{plan['id']}/slots/{slot['id']}/alternatives", headers=headers
    ).json()
    assert options, "no alternatives to test with"

    response = client.patch(
        f"/itineraries/{plan['id']}/slots/{slot['id']}",
        headers=headers,
        json={"action": "replace", "place_id": options[0]["place"]["id"]},
    )
    assert response.status_code == 200, response.text
    assert options[0]["place"]["id"] in [
        s["place"]["id"] for s in response.json()["day"]["slots"]
    ]


def test_replace_needs_either_a_place_or_a_category(client, mixed_family):
    headers, _ = mixed_family
    plan = generate(client, headers, num_days=2, total_budget=6000.0)
    _, slot = _first_editable(plan)
    response = client.patch(
        f"/itineraries/{plan['id']}/slots/{slot['id']}", headers=headers,
        json={"action": "replace"},
    )
    assert response.status_code == 422


# --- guests ------------------------------------------------------------------------------------


GUESTS = [
    {"role": "adult", "age": 30},
    {"role": "adult", "age": 28},
    {"role": "child", "age": 9},
]


def test_guests_join_the_party_for_pricing_and_the_vehicle(client, mixed_family):
    """People coming on the trip but not in the household still get counted and charged.

    `mixed_family` is four; three guests make seven, which no single taxi seats.
    """
    headers, _ = mixed_family
    plan = generate(client, headers, total_budget=7000.0, guests=GUESTS)

    assert plan["vehicle"] == "two vehicles"

    priced = 0
    for day in plan["days"]:
        for slot in day["slots"]:
            cost = slot["cost_breakdown"]
            heads = len(cost["adults"]) + len(cost["children"]) + cost["free_children"]
            assert heads == 7, f"{slot['place']['name']} charged {heads} people, not 7"
            priced += 1
    assert priced, "the plan had no slots to check"


def test_a_transport_round_trip_does_not_forget_the_guests(client, mixed_family):
    """recost_travel must re-price for the same party generation used, guests included."""
    headers, _ = mixed_family
    plan = generate(client, headers, total_budget=7000.0, guests=GUESTS)
    taxi_travel = plan["budget"]["categories"]["travel"]

    for mode in ("own_car", "taxi"):
        response = client.post(
            f"/itineraries/{plan['id']}/transport", headers=headers, json={"mode": mode}
        )
        assert response.status_code == 200, response.text
        back = response.json()

    assert back["vehicle"] == "two vehicles"
    assert back["budget"]["categories"]["travel"] == pytest.approx(taxi_travel)


# --- region ------------------------------------------------------------------------------------


def test_a_trip_confined_to_one_emirate_stays_there(client, mixed_family):
    """The reported bug: "in and around Abu Dhabi or Al Ain" returned City Walk and La Mer.

    Al Ain is a city inside the Abu Dhabi emirate, so that request is one emirate, not two.
    """
    headers, _ = mixed_family
    plan = generate(client, headers, num_days=2, total_budget=6000.0, emirates=["Abu Dhabi"])

    stops = [(s["place"].get("name"), s["place"].get("emirate")) for d in plan["days"] for s in d["slots"]]
    assert stops, "the plan had no stops"
    assert all(emirate == "Abu Dhabi" for _, emirate in stops), stops


def test_a_later_edit_still_respects_the_region(client, mixed_family):
    """Persisted, not just applied once — otherwise the next edit quietly reintroduces Dubai.

    Swapping a stop goes through the same gap retrieval that add_stop and cheaper-day use, so
    this covers every edit path without depending on one category happening to fit the day.
    """
    headers, _ = mixed_family
    plan = generate(client, headers, num_days=2, total_budget=6000.0, emirates=["Abu Dhabi"])
    slot = plan["days"][0]["slots"][0]

    options = client.get(
        f"/itineraries/{plan['id']}/slots/{slot['id']}/alternatives", headers=headers
    ).json()
    assert options, "no alternatives were offered"
    assert all(o["place"]["emirate"] == "Abu Dhabi" for o in options), [
        (o["place"]["name"], o["place"]["emirate"]) for o in options
    ]


# --- moving a plan: the origin, a dropped day, and a re-solve in place --------------------------


def _orm(db, user_email: str, plan_id: int):
    """The ORM pair the service functions take, for a plan built through the API above."""
    from app.models import Itinerary, User

    return db.get(Itinerary, plan_id), db.query(User).filter(User.email == user_email).one()


def test_moving_the_origin_keeps_every_stop_and_re_costs_the_driving(client, mixed_family, db):
    """"We live in Abu Dhabi" moves where the car sets off, not what the trip is.

    The reported bug answered it by claiming the whole plan had moved and changing nothing at all.
    Nothing about the places changes here; the legs into each day do.
    """
    from app.services.itinerary import set_origin

    headers, _ = mixed_family
    plan = generate(client, headers, num_days=2, total_budget=6000.0, emirates=["Dubai"])
    itinerary, user = _orm(db, "family@rihla.app", plan["id"])

    before = [(s["place"]["name"], s["start_time"]) for d in plan["days"] for s in d["slots"]]
    travel_before = sum(seg["est_cost"] for d in plan["days"] for seg in d["segments"])

    # Al Ain, a long way from any Dubai stop, so the first leg of each day certainly changes.
    after = set_origin(db, itinerary, user, 24.2075, 55.7447)

    assert [(s.place.name, s.start_time) for d in after.days for s in d.slots] == before
    assert itinerary.start_lat == 24.2075
    travel_after = sum(seg.info.est_cost for d in after.days for seg in d.segments)
    assert travel_after != travel_before, "the driving was not re-costed"


def test_dropping_a_middle_day_asks_before_moving_anyone_else(client, mixed_family, db):
    """Shift the later days up, or leave the day free? An event on a later date decides it."""
    from app.services.itinerary import DayShiftChoiceRequired, drop_day

    headers, _ = mixed_family
    plan = generate(client, headers, num_days=3, total_budget=9000.0, emirates=["Abu Dhabi"])
    itinerary, user = _orm(db, "family@rihla.app", plan["id"])

    with pytest.raises(DayShiftChoiceRequired):
        drop_day(db, itinerary, user, 2)

    db.refresh(itinerary)
    assert itinerary.num_days == 3, "the question must not have moved anything"


def test_dropping_a_day_with_a_shift_pulls_the_later_days_up(client, mixed_family, db):
    from app.services.itinerary import drop_day

    headers, _ = mixed_family
    plan = generate(client, headers, num_days=3, total_budget=9000.0, emirates=["Abu Dhabi"])
    itinerary, user = _orm(db, "family@rihla.app", plan["id"])
    start = itinerary.start_date
    day3 = [s["place"]["name"] for s in plan["days"][2]["slots"]]

    after = drop_day(db, itinerary, user, 2, shift_later_days=True)

    assert itinerary.num_days == 2
    assert [d.day_index for d in after.days] == [0, 1]
    assert after.days[1].day_date == start + timedelta(days=1), "day 3 kept its old date"
    assert [s.place.name for s in after.days[1].slots] == day3, "day 3's stops did not come along"


def test_leaving_a_dropped_day_free_moves_nobody(client, mixed_family, db):
    from app.services.itinerary import drop_day

    headers, _ = mixed_family
    plan = generate(client, headers, num_days=3, total_budget=9000.0, emirates=["Abu Dhabi"])
    itinerary, user = _orm(db, "family@rihla.app", plan["id"])
    day3 = [s["place"]["name"] for s in plan["days"][2]["slots"]]

    after = drop_day(db, itinerary, user, 2, shift_later_days=False)

    assert itinerary.num_days == 3, "the trip still runs the same dates"
    assert not [d for d in after.days if d.day_index == 1], "day 2 still has stops"
    kept = next(d for d in after.days if d.day_index == 2)
    assert [s.place.name for s in kept.slots] == day3


def test_dropping_the_last_day_needs_no_question(client, mixed_family, db):
    """There is nothing after it to move, so there is no choice to put to the user."""
    from app.services.itinerary import drop_day

    headers, _ = mixed_family
    plan = generate(client, headers, num_days=3, total_budget=9000.0, emirates=["Abu Dhabi"])
    itinerary, user = _orm(db, "family@rihla.app", plan["id"])

    drop_day(db, itinerary, user, 3)
    assert itinerary.num_days == 2


def test_re_solving_in_place_moves_the_region_and_keeps_the_row(client, mixed_family, db):
    """The reported bug: "change the location of the plan to Abu Dhabi" was unanswerable.

    emirates_json was written once, at creation, and nothing could ever change it — so the model
    had no legal move and said it had done it anyway. Re-solving keeps the row, because the
    conversation and the event both point at it.
    """
    from app.services.itinerary import generate as solve

    headers, _ = mixed_family
    plan = generate(client, headers, num_days=2, total_budget=6000.0, emirates=["Dubai"])
    itinerary, user = _orm(db, "family@rihla.app", plan["id"])
    assert {s["place"]["emirate"] for d in plan["days"] for s in d["slots"]} == {"Dubai"}

    moved = solve(
        db, user,
        start_date=itinerary.start_date, num_days=itinerary.num_days,
        total_budget=itinerary.total_budget,
        start_lat=24.4539, start_lng=54.3773,
        emirates=["Abu Dhabi"], into=itinerary,
    )

    assert moved.id == itinerary.id, "a new row would orphan the conversation"
    assert moved.emirates_json == ["Abu Dhabi"]
    assert moved.title == itinerary.title, "the user's own title survived"
    from app.services.itinerary import load_plan

    stops = [s.place for d in load_plan(db, moved).days for s in d.slots]
    assert stops and all(p.emirate == "Abu Dhabi" for p in stops), [p.name for p in stops]
