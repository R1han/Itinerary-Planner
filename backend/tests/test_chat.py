"""Chat orchestration: SSE framing, tool isolation and thread persistence.

The model itself is stubbed — the suite must never make a billable call — so what is under test is
everything around it. Tool behaviour is checked by calling the tools directly, so the assertions
are about what a tool *does to the database*, not about what a model chose to say.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest
from sqlalchemy import select

from app.models import Conversation, Event, FamilyMember, Preference, User
from app.services.budget import Attendee
from app.services.orchestrator import TOOLS, ChatOrchestrator

FUTURE = (date.today() + timedelta(days=20)).isoformat()


def frames(response) -> list[dict]:
    """Parse an SSE body into its typed events."""
    return [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


@pytest.fixture
def stub_llm(monkeypatch):
    """Replace the model call with deterministic frames.

    The chat endpoint needs a working assistant now that there is no fallback, and the suite must
    never make a billable call — so the one network-bound method is swapped out and everything
    around it (framing, persistence, unread state, ownership) is exercised for real.
    """
    from app.services.orchestrator import ChatOrchestrator, sse

    reply = "Here is a reply."

    def fake_llm(self, user_message: str):
        del user_message
        for word in reply.split(" "):
            yield sse("token", word + " ")
        self.record("assistant", reply)
        self.db.commit()
        yield sse("done", {"conversation_id": self.conversation.id})

    monkeypatch.setattr(ChatOrchestrator, "_llm", fake_llm)
    return reply


@pytest.fixture
def orchestrator(db):
    def _make(email: str = "chat@rihla.app") -> ChatOrchestrator:
        user = User(
            email=email, password_hash="x", name="Chat User",
            home_base_lat=24.4539, home_base_lng=54.3773,
        )
        db.add(user)
        db.commit()
        conversation = Conversation(user_id=user.id, title="New plan")
        db.add(conversation)
        db.commit()
        return ChatOrchestrator(db, user, conversation)

    return _make


# --- tool schemas ------------------------------------------------------------------------------


def test_the_spec_tools_are_all_exposed():
    names = {tool["function"]["name"] for tool in TOOLS}
    spec_tools = {
        "save_family_details",
        "create_event",
        "get_upcoming_events",
        "generate_itinerary",
        "record_preference",
    }
    assert spec_tools <= names, f"missing: {spec_tools - names}"
    # Beyond spec §8: get_itinerary, because without a way to READ a plan the assistant could
    # only describe one from stale context and would contradict the budget bar next to it;
    # find_live_events, spec §1.10's optional secondary path, exposed so the model can reach it
    # rather than being a library nobody calls; and the edit tools, without which the model
    # is asked to change plans it has no way to change — and answers by describing edits it never
    # made; and find_places, without which the catalog is unreachable and "what is there to see
    # in the UAE" is answered from the model's own knowledge — places the planner cannot book.
    #
    # And three that exist because every field an Itinerary is built with was write-once except
    # start_date and transport_mode. Asked to change any of the others the model had no legal move,
    # so it said it had done it anyway: "change the location of the plan to Abu Dhabi" was answered
    # twice with the same unchanged Dubai plan. set_origin moves where the trip sets off from and
    # keeps every stop; replace_plan is the only thing that can move the REGION, because no Dubai
    # place survives the trip becoming an Abu Dhabi one; drop_day removes a day.
    assert names == spec_tools | {
        "get_itinerary",
        "find_live_events",
        "find_places",
        "make_day_cheaper",
        "add_prayer_breaks",
        "edit_stop",
        "set_transport",
        "add_stop",
        "reschedule_itinerary",
        "set_origin",
        "drop_day",
        "replace_plan",
    }


def test_no_tool_schema_exposes_a_user_id():
    """The model must not be able to address another user, however it is prompted."""
    for tool in TOOLS:
        properties = tool["function"]["parameters"].get("properties", {})
        assert "user_id" not in properties, tool["function"]["name"]
        assert "user_id" not in json.dumps(tool)


def test_no_tool_schema_exposes_an_itinerary_id():
    """An argument the model must supply is an argument the model can get wrong.

    It did: strict mode makes every property required, so this had to be sent on every call, and
    a model with no id to give sends 0. `get_itinerary` then answered "no plan yet" on a thread
    that had one, and the model set about building a replacement. Which plan is meant was never
    ambiguous — the conversation knows — so the parameter bought nothing and cost that.
    """
    for tool in TOOLS:
        properties = tool["function"]["parameters"].get("properties", {})
        assert "itinerary_id" not in properties, tool["function"]["name"]


# --- tools write only the current user's rows --------------------------------------------------


def test_save_family_details_replaces_the_family_and_records_preferences(db, orchestrator):
    chat = orchestrator()
    result = chat.call_tool(
        "save_family_details",
        {"adults": 2, "children_ages": [7, 13], "likes": ["animals"], "dislikes": ["loud rides"]},
    )
    db.commit()

    assert result["saved"] is True
    members = db.query(FamilyMember).filter(FamilyMember.user_id == chat.user.id).all()
    assert sorted((m.role, m.age) for m in members) == [
        ("adult", 35), ("adult", 35), ("child", 7), ("child", 13),
    ]
    subjects = {p.subject: p.kind for p in db.query(Preference).all()}
    assert subjects == {"animals": "like", "loud rides": "dislike"}


def test_create_event_is_idempotent(db, orchestrator):
    chat = orchestrator()
    args = {"title": "Aisha's birthday", "event_type": "birthday", "date": FUTURE}

    first = chat.call_tool("create_event", args)
    db.commit()
    second = chat.call_tool("create_event", args)
    db.commit()

    assert first["created"] is True
    assert second["created"] is False
    assert db.query(Event).count() == 1


def test_create_event_rejects_a_bad_type_and_a_bad_date(db, orchestrator):
    chat = orchestrator()
    assert "error" in chat.call_tool(
        "create_event", {"title": "X", "event_type": "wedding", "date": FUTURE}
    )
    assert "error" in chat.call_tool(
        "create_event", {"title": "X", "event_type": "birthday", "date": "29-08-2026"}
    )
    assert db.query(Event).count() == 0


def test_get_upcoming_events_returns_only_the_callers_events(db, orchestrator):
    alice = orchestrator("alice@rihla.app")
    bob = orchestrator("bob@rihla.app")

    alice.call_tool("create_event", {"title": "Alice only", "event_type": "birthday", "date": FUTURE})
    bob.call_tool("create_event", {"title": "Bob only", "event_type": "eid", "date": FUTURE})
    db.commit()

    assert [e["title"] for e in alice.call_tool("get_upcoming_events", {})["events"]] == ["Alice only"]
    assert [e["title"] for e in bob.call_tool("get_upcoming_events", {})["events"]] == ["Bob only"]


def test_a_web_search_never_writes_to_the_calendar(db, orchestrator, monkeypatch):
    """A search is a search. Eight scraped listings appearing in someone's calendar unasked is
    not a feature — the user has to say yes, and saying yes goes through create_event."""
    rows = [
        {"title": "Desert Jazz Night", "event_type": "other", "date": date.fromisoformat(FUTURE),
         "notes": "https://example.test/list", "planned": False},
        {"title": "Pitch Hunt | B2B Networking", "event_type": "other",
         "date": date.fromisoformat(FUTURE), "notes": None, "planned": False},
    ]
    monkeypatch.setattr("app.services.orchestrator.find_live_events", lambda *a, **k: rows)

    alice = orchestrator("alice@rihla.app")
    result = alice.call_tool("find_live_events", {"query": "concerts in Dubai"})
    db.commit()

    assert result["found"] == 2
    assert [e["title"] for e in result["events"]] == [r["title"] for r in rows]
    assert db.query(Event).filter(Event.user_id == alice.user.id).count() == 0, (
        "the search wrote to the calendar"
    )


def test_a_search_result_says_which_ones_are_already_on_the_calendar(db, orchestrator, monkeypatch):
    """So the reply can say "you already have this" instead of offering it again."""
    rows = [
        {"title": "Desert Jazz Night", "event_type": "other", "date": date.fromisoformat(FUTURE),
         "notes": None, "planned": False},
        {"title": "Aisha's birthday", "event_type": "birthday",
         "date": date.fromisoformat(FUTURE), "notes": None, "planned": False},
    ]
    monkeypatch.setattr("app.services.orchestrator.find_live_events", lambda *a, **k: rows)

    chat = orchestrator("alice@rihla.app")
    chat.call_tool(
        "create_event", {"title": "Aisha's birthday", "event_type": "birthday", "date": FUTURE}
    )
    db.commit()

    found = {e["title"]: e["already_saved"] for e in chat.call_tool(
        "find_live_events", {"query": "what is on"}
    )["events"]}
    assert found == {"Desert Jazz Night": False, "Aisha's birthday": True}


def test_find_live_events_is_silent_when_the_search_finds_nothing(db, orchestrator, monkeypatch):
    """No key, a timeout, a rate limit — the adapter returns []; the tool must not be an error."""
    monkeypatch.setattr("app.services.orchestrator.find_live_events", lambda *a, **k: [])
    result = orchestrator().call_tool("find_live_events", {"query": "concerts"})
    assert result == {"found": 0, "events": []}
    assert db.query(Event).count() == 0


def test_generate_itinerary_is_refused_until_intake_is_complete(db, orchestrator):
    chat = orchestrator()
    result = chat.call_tool("generate_itinerary", {"days": 3, "budget": 3000, "start_date": FUTURE})
    assert result["error"] == "intake_incomplete"
    assert "adults" in result["missing_fields"]


@pytest.mark.parametrize(
    ("args", "fragment"),
    [
        ({"days": 9, "budget": 3000, "start_date": FUTURE}, "at most 5 days"),
        ({"days": 3, "budget": 0, "start_date": FUTURE}, "budget"),
        (
            {"days": 3, "budget": 3000, "start_date": (date.today() - timedelta(days=2)).isoformat()},
            "in the past",
        ),
        ({"days": 3, "budget": 3000, "event_id": 999, "start_date": FUTURE}, "does not exist"),
        (
            {"days": 1, "budget": 3000, "budget_per_day": 5000, "start_date": FUTURE},
            "Pass one budget, not both",
        ),
    ],
)
def test_generate_itinerary_validates_server_side(db, orchestrator, args, fragment):
    """The LLM is never the validator — bad tool arguments are rejected by the server."""
    chat = orchestrator()
    chat.call_tool("save_family_details", {"adults": 2, "children_ages": [7]})
    db.commit()

    result = chat.call_tool("generate_itinerary", args)
    assert "error" in result
    assert fragment in result["error"]


def test_a_one_day_budget_may_be_passed_as_both_fields_when_they_agree(db, orchestrator):
    """budget and budget_per_day land equal for a one-day trip — that is not a conflict to
    reject, just the same figure said two ways, so it must fall through to other validation
    instead of the "pass one, not both" error."""
    result = orchestrator().call_tool(
        "generate_itinerary",
        {"days": 1, "budget": 750, "budget_per_day": 750, "start_date": FUTURE},
    )
    assert result.get("error") != "Pass one budget, not both: 750 for the whole trip or 750 for each day. Which did the user mean?"
    assert result["error"] == "intake_incomplete"


def test_record_preference_writes_it_for_the_calling_user_only(db, orchestrator):
    alice = orchestrator("alice@rihla.app")
    bob = orchestrator("bob@rihla.app")

    alice.call_tool("record_preference", {"kind": "dislike", "subject": "queues"})
    db.commit()

    assert db.query(Preference).filter(Preference.user_id == alice.user.id).count() == 1
    assert db.query(Preference).filter(Preference.user_id == bob.user.id).count() == 0


def test_an_unknown_tool_is_an_error_message_not_a_crash(db, orchestrator):
    assert "error" in orchestrator().call_tool("drop_all_tables", {})


# --- system prompt -----------------------------------------------------------------------------


def test_the_system_prompt_carries_only_the_current_users_context(db, orchestrator):
    alice = orchestrator("alice@rihla.app")
    bob = orchestrator("bob@rihla.app")

    alice.call_tool("save_family_details", {"adults": 2, "likes": ["quiet beaches at sunset"]})
    bob.call_tool("save_family_details", {"adults": 1, "likes": ["loud theme parks"]})
    db.commit()

    assert "quiet beaches at sunset" in alice.system_prompt()
    assert "loud theme parks" not in alice.system_prompt()
    assert "quiet beaches at sunset" not in bob.system_prompt()


def test_the_system_prompt_forbids_the_model_from_scheduling(db, orchestrator):
    prompt = orchestrator().system_prompt()
    assert "do NOT build itineraries" in prompt
    assert "never invent times, prices or places" in prompt


# --- SSE endpoint, with the LLM down -----------------------------------------------------------


def test_chat_streams_sse_frames_and_persists_the_thread(client, make_user, stub_llm):
    headers, _ = make_user("sse@rihla.app")
    response = client.post("/chat", headers=headers, json={"message": "hello"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = frames(response)
    types = [event["type"] for event in events]
    assert types[0] == "conversation"
    assert "token" in types
    assert types[-1] == "done"

    conversation_id = events[0]["data"]["conversation_id"]
    history = client.get(f"/conversations/{conversation_id}/messages", headers=headers).json()
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[0]["content"] == "hello"



def test_messages_accumulate_in_the_same_thread(client, make_user, stub_llm):
    headers, _ = make_user("thread@rihla.app")
    first = client.post("/chat", headers=headers, json={"message": "hello"})
    conversation_id = frames(first)[0]["data"]["conversation_id"]

    client.post(
        "/chat", headers=headers, json={"message": "and again", "conversation_id": conversation_id}
    )
    history = client.get(f"/conversations/{conversation_id}/messages", headers=headers).json()
    assert [m["content"] for m in history if m["role"] == "user"] == ["hello", "and again"]


# --- threads and unread ------------------------------------------------------------------------


def test_a_new_message_marks_the_thread_unread_until_it_is_seen(client, make_user, stub_llm):
    headers, _ = make_user("unread@rihla.app")
    conversation_id = frames(client.post("/chat", headers=headers, json={"message": "hi"}))[0][
        "data"
    ]["conversation_id"]

    listed = client.get("/conversations", headers=headers).json()
    assert listed[0]["unread"] is True

    assert client.post(f"/conversations/{conversation_id}/seen", headers=headers).json()["unread"] is False
    assert client.get("/conversations", headers=headers).json()[0]["unread"] is False


def test_threads_are_private_to_their_owner(client, make_user, stub_llm):
    headers, _ = make_user("owner@rihla.app")
    intruder, _ = make_user("intruder@rihla.app")
    conversation_id = frames(client.post("/chat", headers=headers, json={"message": "secret"}))[0][
        "data"
    ]["conversation_id"]

    assert client.get(f"/conversations/{conversation_id}/messages", headers=intruder).status_code == 404
    assert client.post(f"/conversations/{conversation_id}/seen", headers=intruder).status_code == 404
    assert client.get("/conversations", headers=intruder).json() == []

    # Posting into someone else's thread is a 404, not a silent write.
    assert client.post(
        "/chat", headers=intruder, json={"message": "hi", "conversation_id": conversation_id}
    ).status_code == 404


def test_chat_requires_authentication(client):
    assert client.post("/chat", json={"message": "hi"}).status_code == 401


# --- reading the current plan ------------------------------------------------------------------


@pytest.fixture
def planned(client, make_user, db):
    """A user with a real generated plan, and their auth headers."""
    from pathlib import Path

    from app.models import Place
    from app.seed import default_price_bands

    rows = json.loads(
        (Path(__file__).resolve().parent.parent / "app" / "data" / "places.json").read_text()
    )
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
                romance_score=row.get("romance_score", 0.5), description="",
            )
        )
    db.commit()

    headers, user = make_user("planner@rihla.app")
    client.put(
        "/family",
        headers=headers,
        json={"members": [{"role": "adult", "age": 34}, {"role": "child", "age": 8}]},
    )
    plan = client.post(
        "/itineraries/generate",
        headers=headers,
        json={
            "start_date": FUTURE, "num_days": 2, "total_budget": 2500.0,
            "start_lat": 25.2048, "start_lng": 55.2708,
        },
    ).json()
    return headers, user, plan


def test_get_itinerary_is_exposed_as_a_tool():
    assert "get_itinerary" in {tool["function"]["name"] for tool in TOOLS}


def test_get_itinerary_reports_no_plan_before_one_exists(db, orchestrator):
    assert orchestrator().call_tool("get_itinerary", {})["itinerary"] is None


def test_get_itinerary_matches_what_the_budget_bar_shows(client, planned, db):
    """The whole point: the assistant must read current state, not recall it."""
    headers, user, plan = planned
    from app.models import Conversation, User

    row = db.query(User).filter(User.email == "planner@rihla.app").one()
    conversation = Conversation(user_id=row.id, itinerary_id=plan["id"])
    db.add(conversation)
    db.commit()

    read = ChatOrchestrator(db, row, conversation).call_tool("get_itinerary", {})

    assert read["itinerary_id"] == plan["id"]
    assert read["budget"]["total"] == plan["budget"]["total"]
    assert read["budget"]["cap"] == plan["budget"]["cap"]
    assert read["budget"]["remaining"] == plan["budget"]["remaining"]
    assert [d["subtotal"] for d in read["days"]] == [d["subtotal"] for d in plan["days"]]
    assert [s["name"] for d in read["days"] for s in d["stops"]] == [
        s["place"]["name"] for d in plan["days"] for s in d["slots"]
    ]


def test_get_itinerary_reflects_an_edit_rather_than_the_original(client, planned, db):
    """A stale recap is exactly the bug this tool exists to prevent."""
    from app.models import Conversation, User

    headers, _, plan = planned
    day = next(d for d in plan["days"] if d["slots"])
    slot = day["slots"][0]

    client.patch(
        f"/itineraries/{plan['id']}/slots/{slot['id']}", headers=headers, json={"action": "remove"}
    )

    row = db.query(User).filter(User.email == "planner@rihla.app").one()
    conversation = Conversation(user_id=row.id, itinerary_id=plan["id"])
    db.add(conversation)
    db.commit()

    read = ChatOrchestrator(db, row, conversation).call_tool("get_itinerary", {})
    fresh = client.get(f"/itineraries/{plan['id']}", headers=headers).json()

    assert read["budget"]["total"] == fresh["budget"]["total"]
    assert read["budget"]["total"] != plan["budget"]["total"], "the tool returned the stale total"
    assert slot["place"]["name"] not in [s["name"] for d in read["days"] for s in d["stops"]]


def test_get_itinerary_cannot_read_another_users_plan(client, planned, db, orchestrator):
    """There is no argument left to try, and naming one anyway changes nothing."""
    _, _, plan = planned
    intruder = orchestrator("intruder@rihla.app")
    assert intruder.call_tool("get_itinerary", {})["itinerary"] is None
    assert intruder.call_tool("get_itinerary", {"itinerary_id": plan["id"]})["itinerary"] is None


def test_the_system_prompt_forbids_quoting_figures_from_memory(db, orchestrator):
    prompt = orchestrator().system_prompt()
    assert "Never quote a time, a price or a total from memory" in prompt
    assert "get_itinerary" in prompt


def test_a_model_failure_surfaces_as_an_error_frame(client, make_user, monkeypatch):
    """With no fallback, a failed assistant must be reported — never silently swallowed."""
    from app.services.orchestrator import ChatOrchestrator

    def boom(self, user_message: str):
        raise RuntimeError("upstream is down")
        yield  # pragma: no cover — makes this a generator

    monkeypatch.setattr(ChatOrchestrator, "_llm", boom)

    headers, _ = make_user("broken@rihla.app")
    response = client.post("/chat", headers=headers, json={"message": "hello"})
    events = frames(response)
    types = [event["type"] for event in events]

    assert response.status_code == 200, "the stream must not 500 mid-flight"
    assert "error" in types
    assert types[-1] == "done"
    assert events[-1]["data"]["failed"] is True
    assert "upstream is down" in next(e["data"]["message"] for e in events if e["type"] == "error")


def test_the_users_message_is_kept_even_when_the_assistant_fails(client, make_user, monkeypatch):
    """Losing what the user typed because the model fell over would be its own bug."""
    from app.services.orchestrator import ChatOrchestrator

    def boom(self, user_message: str):
        raise RuntimeError("upstream is down")
        yield  # pragma: no cover

    monkeypatch.setattr(ChatOrchestrator, "_llm", boom)

    headers, _ = make_user("kept@rihla.app")
    response = client.post("/chat", headers=headers, json={"message": "remember this"})
    conversation_id = frames(response)[0]["data"]["conversation_id"]

    history = client.get(f"/conversations/{conversation_id}/messages", headers=headers).json()
    assert [m["content"] for m in history] == ["remember this"]


def test_a_blocked_intake_is_surfaced_to_the_client(client, make_user, monkeypatch):
    """Chat is the only way to plan now, so the user must be told what is still missing."""
    from app.services.orchestrator import ChatOrchestrator, sse

    def fake_llm(self, user_message: str):
        del user_message
        result = self.call_tool("generate_itinerary", {"days": 3, "budget": 3000, "start_date": FUTURE})
        if result.get("error") == "intake_incomplete":
            yield sse("intake_required", {"missing_fields": result["missing_fields"]})
        self.record("assistant", "I need a little more first.")
        self.db.commit()
        yield sse("done", {"conversation_id": self.conversation.id})

    monkeypatch.setattr(ChatOrchestrator, "_llm", fake_llm)

    headers, _ = make_user("intake@rihla.app")  # registered, but no family recorded
    events = frames(client.post("/chat", headers=headers, json={"message": "plan my trip"}))

    checklist = next(e for e in events if e["type"] == "intake_required")
    assert "adults" in checklist["data"]["missing_fields"]


# --- the activity trace ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "args", "label", "detail"),
    [
        (
            "save_family_details",
            {"adults": 2, "children_ages": [6, 13], "dislikes": ["loud rides"]},
            "Saving your family details",
            "2 adults · 2 children aged 6, 13 · dislikes loud rides",
        ),
        ("save_family_details", {"adults": 1}, "Saving your family details", "1 adult"),
        ("get_upcoming_events", {"horizon_days": 30}, "Checking your calendar", "next 30 days"),
        ("get_upcoming_events", {}, "Checking your calendar", "next 60 days"),
        (
            "generate_itinerary",
            {"days": 3, "budget": 2500, "prayer_breaks": True},
            "Building your itinerary",
            "3 days · AED 2,500 · with prayer breaks",
        ),
        ("get_itinerary", {}, "Reading the current plan", None),
        (
            "record_preference",
            {"kind": "dislike", "subject": "long queues"},
            "Noting a preference",
            "dislikes long queues",
        ),
        (
            "create_event",
            {"title": "Aisha's birthday", "date": "2026-08-29"},
            "Adding an event",
            "Aisha's birthday · 2026-08-29",
        ),
    ],
)
def test_a_tool_call_is_described_in_plain_language(name, args, label, detail):
    from app.services.orchestrator import describe_tool_call

    assert describe_tool_call(name, args) == (label, detail)


def test_an_unknown_tool_still_gets_a_readable_label():
    from app.services.orchestrator import describe_tool_call

    assert describe_tool_call("some_new_tool", {}) == ("Some new tool", None)


def test_describing_a_call_never_raises_on_odd_arguments():
    """The trace is cosmetic; malformed arguments must not take the stream down with them."""
    from app.services.orchestrator import describe_tool_call

    for args in ({}, {"adults": None}, {"children_ages": [None]}, {"budget": "lots"}):
        label, _ = describe_tool_call("generate_itinerary", args)
        assert label
        describe_tool_call("save_family_details", args)


@pytest.mark.parametrize(
    ("name", "result", "outcome"),
    [
        ("get_upcoming_events", {"events": [1, 2, 3]}, "3 events"),
        ("get_upcoming_events", {"events": [1]}, "1 event"),
        ("generate_itinerary", {"total": 1574.41, "cap": 2500}, "AED 1,574 of AED 2,500"),
        (
            "generate_itinerary",
            {"error": "intake_incomplete", "missing_fields": ["adults", "start_location"]},
            "needs adults, start location",
        ),
        ("get_itinerary", {"itinerary": None}, "no plan yet"),
        # The populated shape carries itinerary_id and no "itinerary" key at all — the reason the
        # first version of this reported "no plan yet" for every successful read.
        (
            "get_itinerary",
            {"itinerary_id": 4, "days": [1, 2, 3], "budget": {"total": 1559.98}},
            "3 days · AED 1,560",
        ),
        ("create_event", {"created": True}, "added"),
        ("create_event", {"created": False}, "already on your calendar"),
        ("record_preference", {"recorded": True}, "noted"),
    ],
)
def test_a_tool_result_is_summarised_for_the_trace(name, result, outcome):
    from app.services.orchestrator import summarise_tool_result

    assert summarise_tool_result(name, result) == outcome


def test_the_stream_pairs_every_tool_frame_with_a_result(client, make_user, monkeypatch):
    """A row that starts and never resolves reads as a hang."""
    from app.services.orchestrator import ChatOrchestrator, describe_tool_call, sse
    from app.services.orchestrator import summarise_tool_result as summarise

    def fake_llm(self, user_message: str):
        del user_message
        args = {"horizon_days": 30}
        label, detail = describe_tool_call("get_upcoming_events", args)
        yield sse("tool", {"id": "c1", "name": "get_upcoming_events", "label": label, "detail": detail})
        result = self.call_tool("get_upcoming_events", args)
        yield sse("tool_done", {"id": "c1", "outcome": summarise("get_upcoming_events", result), "failed": False})
        self.record("assistant", "Nothing coming up.")
        self.db.commit()
        yield sse("done", {"conversation_id": self.conversation.id})

    monkeypatch.setattr(ChatOrchestrator, "_llm", fake_llm)

    headers, _ = make_user("trace@rihla.app")
    events = frames(client.post("/chat", headers=headers, json={"message": "what's on?"}))

    started = [e for e in events if e["type"] == "tool"]
    finished = [e for e in events if e["type"] == "tool_done"]

    assert len(started) == len(finished) == 1
    assert started[0]["data"]["label"] == "Checking your calendar"
    assert started[0]["data"]["detail"] == "next 30 days"
    assert finished[0]["data"]["id"] == started[0]["data"]["id"]
    assert finished[0]["data"]["outcome"] == "0 events"


# --- editing an existing plan from chat ----------------------------------------------------------


def _warned_last_turn(chat):
    """Put the thread where a real one is when the user says "yes, start over".

    The warning was given in a previous turn and the user has since replied, which is the only
    state in which replace_existing counts for anything.
    """
    chat.conversation.rebuild_warned = True
    chat.warned_at_turn_start = True
    return chat


def _chat_for(db, plan):
    """An orchestrator whose conversation is attached to `plan`, as the real one would be."""
    from app.models import Conversation, User

    row = db.query(User).filter(User.email == "planner@rihla.app").one()
    conversation = Conversation(user_id=row.id, itinerary_id=plan["id"])
    db.add(conversation)
    db.commit()
    return ChatOrchestrator(db, row, conversation)


def test_removing_a_stop_from_chat_changes_the_plan_and_flags_the_pane(client, planned, db):
    """The bug this exists to kill: the model says it changed the plan, and nothing changed.

    Two halves, and both matter. The database must actually lose the stop, and
    `touched_itinerary` must be set — that flag is the only thing that makes the stream emit
    `itinerary_updated`, so without it the right pane keeps rendering the pre-edit plan.
    """
    _, _, plan = planned
    chat = _chat_for(db, plan)

    slot = next(d for d in plan["days"] if d["slots"])["slots"][0]
    result = chat.call_tool("edit_stop", {"stop": slot["place"]["name"], "action": "remove"})

    assert "error" not in result, result
    assert chat.touched_itinerary is not None, "the right pane will not refresh"
    assert chat.touched_itinerary.id == plan["id"]

    names = [s["name"] for d in chat.call_tool("get_itinerary", {})["days"] for s in d["stops"]]
    assert slot["place"]["name"] not in names


def test_a_stop_can_be_edited_without_reading_the_plan_first(client, planned, db):
    """The point of naming stops: no lookup round, so nothing to remember or get stale.

    edit_stop used to take a slot_id, which exists only in the database — a model that had not
    just called get_itinerary could only invent one, and did.
    """
    _, _, plan = planned
    chat = _chat_for(db, plan)
    named = next(d for d in plan["days"] if d["slots"])["slots"][0]["place"]["name"]

    result = chat.call_tool("edit_stop", {"stop": named, "action": "remove"})

    assert "error" not in result, result
    names = [s["name"] for d in chat.call_tool("get_itinerary", {})["days"] for s in d["stops"]]
    assert named not in names


def test_an_edit_cannot_reach_another_users_plan(client, planned, db, orchestrator):
    """The stop is real and its name is guessable; the caller is not its owner.

    Naming stops instead of numbering them changes nothing here — resolution runs against the
    plan the CONVERSATION owns, and this conversation owns none.
    """
    _, _, plan = planned
    slot = next(d for d in plan["days"] if d["slots"])["slots"][0]

    intruder = orchestrator("intruder@rihla.app")
    result = intruder.call_tool(
        "edit_stop", {"stop": slot["place"]["name"], "action": "remove"}
    )

    assert "error" in result
    assert intruder.touched_itinerary is None
    fresh = client.get(f"/itineraries/{plan['id']}", headers=planned[0]).json()
    assert [s["id"] for d in fresh["days"] for s in d["slots"]] == [
        s["id"] for d in plan["days"] for s in d["slots"]
    ]


def test_adjust_without_a_time_is_an_error_not_a_silent_no_op(client, planned, db):
    _, _, plan = planned
    chat = _chat_for(db, plan)
    slot = next(d for d in plan["days"] if d["slots"])["slots"][0]
    assert "error" in chat.call_tool(
        "edit_stop", {"stop": slot["place"]["name"], "action": "adjust"}
    )


def test_making_a_day_cheaper_reports_what_it_actually_saved(client, planned, db):
    """The planner often finds nothing better. `saved` is what stops the reply inventing a win."""
    _, _, plan = planned
    chat = _chat_for(db, plan)

    result = chat.call_tool("make_day_cheaper", {"day": 1})

    assert "error" not in result
    assert result["saved"] == pytest.approx(plan["budget"]["total"] - result["total"], abs=0.01)
    assert result["saved"] >= 0


def test_make_day_cheaper_rejects_a_day_that_does_not_exist(client, planned, db):
    result = _chat_for(db, planned[2]).call_tool("make_day_cheaper", {"day": 99})
    assert "error" in result


def test_reschedule_moves_the_start_date_and_keeps_every_stop(client, planned, db):
    """The whole point: same trip, different calendar — nothing about it should be rebuilt."""
    _, _, plan = planned
    chat = _chat_for(db, plan)
    new_start = (date.today() + timedelta(days=45)).isoformat()
    before_names = sorted(s["place"]["name"] for d in plan["days"] for s in d["slots"])

    result = chat.call_tool("reschedule_itinerary", {"start_date": new_start})

    assert "error" not in result, result
    assert result["start_date"].isoformat() == new_start
    assert chat.touched_itinerary is not None
    after = client.get(f"/itineraries/{plan['id']}", headers=planned[0]).json()
    assert after["start_date"] == new_start
    assert sorted(s["place"]["name"] for d in after["days"] for s in d["slots"]) == before_names


def test_reschedule_rejects_a_past_date(client, planned, db):
    _, _, plan = planned
    chat = _chat_for(db, plan)
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    result = chat.call_tool("reschedule_itinerary", {"start_date": yesterday})

    assert "error" in result
    assert chat.touched_itinerary is None


def test_rescheduling_needs_a_plan_to_reschedule(db, orchestrator):
    result = orchestrator().call_tool("reschedule_itinerary", {"start_date": FUTURE})
    assert "error" in result


def test_editing_needs_a_plan_to_edit(db, orchestrator):
    chat = orchestrator()
    assert "error" in chat.call_tool("add_prayer_breaks", {})
    assert chat.touched_itinerary is None


def test_the_system_prompt_forbids_claiming_an_edit_that_did_not_happen(db, orchestrator):
    """The honesty rule. The "no tool exists for that" clause it used to carry is gone —
    add_stop and edit_stop/replace now cover the asks it was written for — but the rule that a
    change must have actually happened before it is described stands on its own."""
    prompt = orchestrator().system_prompt()
    assert "Never say the plan changed unless a tool you called in THIS turn returned" in prompt
    assert "Listing a stop the plan does not contain is" in prompt


# --- a rebuild must not quietly orphan the plan --------------------------------------------------


def test_a_rebuild_keeps_the_event_the_conversation_is_planning(client, planned, db):
    """The model rebuilt a birthday plan without passing event_id, and the plan lost the event.

    That is not cosmetic. The retrieval query is built from the event's title and notes, so
    dropping the link changes which places are shortlisted at all — in the reported case it left
    the day with one nearby restaurant, already used at lunch, and the planner reaching into the
    next emirate for dinner.
    """
    from app.models import Conversation, Event, Itinerary, User

    headers, _, plan = planned
    row = db.query(User).filter(User.email == "planner@rihla.app").one()
    event = Event(
        user_id=row.id, title="Aisha's 7th birthday", event_type="birthday",
        date=date.fromisoformat(FUTURE), notes="loves animals, afraid of loud rides",
    )
    db.add(event)
    db.commit()

    conversation = Conversation(user_id=row.id, itinerary_id=plan["id"], event_id=event.id)
    db.add(conversation)
    db.commit()

    chat = _warned_last_turn(ChatOrchestrator(db, row, conversation))
    result = chat.call_tool(
        "generate_itinerary", {"replace_existing": True, "days": 1, "budget": 4500, "start_date": FUTURE}
    )

    assert "error" not in result, result
    rebuilt = db.get(Itinerary, result["itinerary_id"])
    assert rebuilt.event_id == event.id, "the rebuild orphaned the plan from its event"
    assert rebuilt.title == event.title, "the plan was retitled 'UAE trip'"


def test_an_explicit_event_id_still_wins_over_the_conversations(client, planned, db):
    """The fallback is a default, not an override — the model can still retarget deliberately."""
    from app.models import Conversation, Event, Itinerary, User

    _, _, plan = planned
    row = db.query(User).filter(User.email == "planner@rihla.app").one()
    attached = Event(user_id=row.id, title="Birthday", event_type="birthday",
                     date=date.fromisoformat(FUTURE))
    asked_for = Event(user_id=row.id, title="Cousins visiting", event_type="family_visit",
                      date=date.fromisoformat(FUTURE))
    db.add_all([attached, asked_for])
    db.commit()

    conversation = Conversation(user_id=row.id, itinerary_id=plan["id"], event_id=attached.id)
    db.add(conversation)
    db.commit()

    result = _warned_last_turn(ChatOrchestrator(db, row, conversation)).call_tool(
        "generate_itinerary",
        {"replace_existing": True, "days": 1, "budget": 4500, "start_date": FUTURE, "event_id": asked_for.id},
    )
    assert db.get(Itinerary, result["itinerary_id"]).event_id == asked_for.id


def test_the_system_prompt_forbids_rebuilding_as_a_workaround(db, orchestrator):
    prompt = orchestrator().system_prompt()
    assert "throws the current one away" in prompt
    assert "never a reason to start over" in prompt


def test_saying_you_have_your_own_car_actually_reprices_the_plan(client, planned, db):
    """The reported bug: the assistant replied "Noted!" and the taxi fares stayed on screen."""
    _, _, plan = planned
    chat = _chat_for(db, plan)
    before = chat.call_tool("get_itinerary", {})["budget"]["travel"]

    result = chat.call_tool("set_transport", {"mode": "own_car"})

    assert "error" not in result, result
    assert result["transport_mode"] == "own_car"
    assert result["travel"]["total"] < before
    assert chat.touched_itinerary is not None, "the right pane will not refresh"
    assert chat.call_tool("get_itinerary", {})["budget"]["travel"] == result["travel"]["total"]


def test_an_unknown_transport_mode_is_an_error_not_a_silent_write(client, planned, db):
    _, _, plan = planned
    chat = _chat_for(db, plan)
    assert "error" in chat.call_tool("set_transport", {"mode": "camel"})
    assert chat.call_tool("get_itinerary", {})["itinerary_id"] == plan["id"]


def test_a_rebuild_keeps_the_transport_mode_the_family_told_us_about(client, planned, db):
    """Same failure shape as the dropped event: say "own car", rebuild, taxi fares return."""
    from app.models import Itinerary

    _, _, plan = planned
    chat = _warned_last_turn(_chat_for(db, plan))
    chat.call_tool("set_transport", {"mode": "own_car"})

    result = chat.call_tool(
        "generate_itinerary", {"replace_existing": True, "days": 1, "budget": 4500, "start_date": FUTURE}
    )
    assert "error" not in result, result
    assert db.get(Itinerary, result["itinerary_id"]).transport_mode == "own_car"


# --- the assistant knows what is already on the calendar -----------------------------------------


def test_the_system_prompt_lists_the_events_already_on_the_calendar(db, orchestrator):
    """The reported bug: asked to plan a wedding anniversary that was already in the calendar,
    with its date and its notes, the assistant asked for the date.

    Family and preferences are injected; events were the one thing the model had to go and fetch,
    and nothing told it to. So it asked instead.
    """
    chat = orchestrator()
    chat.call_tool(
        "create_event",
        {"title": "Wedding anniversary", "event_type": "anniversary", "date": FUTURE,
         "notes": "dinner, just the two of us"},
    )
    db.commit()

    prompt = chat.system_prompt("plan a romantic dinner for my wedding anniversary")

    assert "Wedding anniversary" in prompt
    assert FUTURE in prompt
    assert "dinner, just the two of us" in prompt
    assert "never ask for a date" in prompt.lower()


def test_the_prompt_says_when_the_calendar_is_empty_rather_than_omitting_it(db, orchestrator):
    """An absent section reads as "no information"; "nothing on the calendar" reads as a fact."""
    assert "nothing on the calendar yet" in orchestrator().system_prompt()


def test_a_planned_event_is_marked_as_such_in_the_prompt(db, orchestrator):
    """So the assistant offers to plan the unplanned ones, not the one it already did."""
    chat = orchestrator()
    chat.call_tool("create_event", {"title": "Eid trip", "event_type": "eid", "date": FUTURE})
    db.commit()
    db.query(Event).filter(Event.title == "Eid trip").update({"planned": True})
    db.commit()

    assert "Eid trip" in chat.system_prompt()
    assert "PLANNED already" in chat.system_prompt()

    # And the other state is marked too. Only marking the planned ones made "not planned" the
    # absence of a marker, which is what got read backwards.
    chat.call_tool("create_event", {"title": "Open day", "event_type": "other", "date": FUTURE})
    db.commit()
    line = next(l for l in chat.system_prompt().splitlines() if "Open day" in l)
    assert line.endswith("NO PLAN YET"), line


def test_the_prompt_tells_the_model_not_to_ask_for_a_date_it_already_has(db, orchestrator):
    prompt = orchestrator().system_prompt()
    assert "On their calendar:" in prompt
    assert "never ask for a date you have been given" in prompt


def test_a_search_followed_by_a_yes_is_what_writes_the_event(db, orchestrator, monkeypatch):
    """The whole flow: search suggests, the user picks, create_event saves. Two steps, on purpose."""
    monkeypatch.setattr(
        "app.services.orchestrator.find_live_events",
        lambda *a, **k: [
            {"title": "Desert Jazz Night", "event_type": "other",
             "date": date.fromisoformat(FUTURE), "notes": None, "planned": False}
        ],
    )
    chat = orchestrator()
    chat.call_tool("find_live_events", {"query": "what is on"})
    db.commit()
    assert db.query(Event).count() == 0

    chat.call_tool(
        "create_event", {"title": "Desert Jazz Night", "event_type": "other", "date": FUTURE}
    )
    db.commit()
    assert [e.title for e in db.query(Event).all()] == ["Desert Jazz Night"]


# --- strict tool schemas -------------------------------------------------------------------------


def test_every_tool_is_declared_strict():
    for tool in TOOLS:
        assert tool["function"].get("strict") is True, tool["function"]["name"]


def test_every_tool_schema_satisfies_strict_mode():
    """Strict mode has no optional properties: everything in `required`, every object closed.

    A schema that breaks these rules is rejected by the API, and a rejected schema fails the
    whole chat request — so this is checked here rather than discovered in production.

    Checked at EVERY depth, not just the top: the rule applies to an object nested inside an
    array's `items` too, and a top-level-only check let exactly that reach the API.
    """

    def closed(node, where: str):
        if node.get("type") == "object" or "properties" in node:
            assert node.get("additionalProperties") is False, f"{where} is not closed"
            assert set(node.get("required", [])) == set(node.get("properties", {})), (
                where,
                set(node.get("properties", {})) ^ set(node.get("required", [])),
            )
        for name, child in node.get("properties", {}).items():
            closed(child, f"{where}.{name}")
        if isinstance(node.get("items"), dict):
            closed(node["items"], f"{where}[]")

    for tool in TOOLS:
        closed(tool["function"]["parameters"], tool["function"]["name"])


def test_a_formerly_optional_argument_is_declared_nullable():
    """It has to still be omittable in spirit, and null is how strict mode spells that."""
    by_name = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}

    assert "null" in by_name["get_upcoming_events"]["properties"]["horizon_days"]["type"]
    assert "null" in by_name["edit_stop"]["properties"]["start_time"]["type"]
    assert "null" in by_name["save_family_details"]["properties"]["likes"]["type"]
    # A required one stays a plain scalar.
    assert by_name["edit_stop"]["properties"]["stop"]["type"] == "string"


def test_no_schema_carries_a_keyword_strict_mode_may_reject():
    """`default`, `minimum` and friends are dropped; the handlers still enforce the ranges."""
    banned = {"default", "minimum", "maximum", "minItems", "maxItems"}

    def walk(node):
        assert not (banned & set(node)), node
        for child in node.get("properties", {}).values():
            walk(child)
        if "items" in node:
            walk(node["items"])

    for tool in TOOLS:
        walk(tool["function"]["parameters"])


ALL_NULL_CALLS = [
    ("save_family_details", {"adults": 2, "children_ages": None, "likes": None, "dislikes": None}),
    ("create_event", {"title": "X", "event_type": "birthday", "date": FUTURE, "notes": None}),
    ("get_upcoming_events", {"horizon_days": None}),
    ("get_itinerary", {}),
    ("record_preference", {"kind": "like", "subject": "beaches", "category": None}),
    ("add_prayer_breaks", {}),
    ("set_transport", {"mode": "own_car"}),
    ("make_day_cheaper", {"day": 1}),
    ("edit_stop", {"stop": "anything", "action": "remove", "start_time": None, "category": None}),
    ("add_stop", {"day": 1, "category": None}),
]


@pytest.mark.parametrize(("name", "args"), ALL_NULL_CALLS, ids=[c[0] for c in ALL_NULL_CALLS])
def test_a_handler_survives_the_nulls_strict_mode_forces_it_to_send(db, orchestrator, name, args):
    """Strict mode sends every property, so optional ones arrive as None rather than missing.

    `args.get("horizon_days", 60)` returns None for a present-but-null key, and int(None) raises.
    A tool error is a message, not a crash — but "unsupported operand" is not a message.
    """
    result = orchestrator().call_tool(name, args)
    assert isinstance(result, dict)
    problem = str(result.get("error", ""))
    assert "NoneType" not in problem and "int()" not in problem, problem


def test_find_live_events_survives_a_null_horizon(db, orchestrator, monkeypatch):
    monkeypatch.setattr("app.services.orchestrator.find_live_events", lambda *a, **k: [])
    result = orchestrator().call_tool(
        "find_live_events", {"query": "concerts", "horizon_days": None}
    )
    assert result == {"found": 0, "events": []}


def test_generate_itinerary_survives_the_nulls(db, orchestrator):
    result = orchestrator().call_tool(
        "generate_itinerary",
        {"event_id": None, "start_date": None, "days": 2, "budget": 3000, "prayer_breaks": None},
    )
    assert "NoneType" not in str(result.get("error", "")), result


def test_describing_a_call_survives_nulls_too(db):
    """The activity row is built from the same arguments, and it runs before the handler does."""
    from app.services.orchestrator import describe_tool_call

    for name, args in ALL_NULL_CALLS:
        label, _ = describe_tool_call(name, args)
        assert label
    assert describe_tool_call("get_upcoming_events", {"horizon_days": None})[1]


# --- asking for a dinner should not produce a day out ---------------------------------------------


def test_a_dinner_only_request_plans_one_evening_stop(client, planned, db):
    """The reported bug: "plan a romantic dinner for my anniversary" returned an aquarium, a
    lunch, a theme park and a dinner."""
    from app.models import Itinerary, User

    _, _, plan = planned
    row = db.query(User).filter(User.email == "planner@rihla.app").one()
    conversation = Conversation(user_id=row.id)
    db.add(conversation)
    db.commit()

    result = ChatOrchestrator(db, row, conversation).call_tool(
        "generate_itinerary",
        {"days": 1, "budget": 5000, "start_date": FUTURE,
         "focus": "dinner_only", "adults_only": True},
    )

    assert "error" not in result, result
    built = db.get(Itinerary, result["itinerary_id"])
    slots = list(built.slots)
    assert len(slots) == 1, [s.place.name for s in slots]
    assert slots[0].place.category in ("casual_dining", "fine_dining")
    assert slots[0].start_time >= "17:00", slots[0].start_time


def test_adults_only_charges_for_the_adults_and_not_the_children(client, planned, db):
    """`planned` is a family of two; a couple's night out must not price the child."""
    from app.models import User

    _, _, plan = planned
    row = db.query(User).filter(User.email == "planner@rihla.app").one()
    conversation = Conversation(user_id=row.id)
    db.add(conversation)
    db.commit()
    chat = ChatOrchestrator(db, row, conversation)

    couple = chat.call_tool(
        "generate_itinerary",
        {"days": 1, "budget": 5000, "start_date": FUTURE,
         "focus": "dinner_only", "adults_only": True},
    )
    payload = chat.call_tool("get_itinerary", {})
    breakdown = payload["days"][0]["stops"][0]
    assert breakdown["cost"] > 0

    from app.services.itinerary import itinerary_payload
    from app.models import Itinerary

    full = itinerary_payload(db, db.get(Itinerary, couple["itinerary_id"]))
    costs = full["days"][0]["slots"][0]["cost_breakdown"]
    assert len(costs["adults"]) == 1, costs
    assert not costs["children"], costs


def test_a_full_day_request_is_unchanged(client, planned, db):
    """The default must not move — everything else depends on it."""
    from app.models import Itinerary, User

    _, _, plan = planned
    row = db.query(User).filter(User.email == "planner@rihla.app").one()
    conversation = Conversation(user_id=row.id)
    db.add(conversation)
    db.commit()

    result = ChatOrchestrator(db, row, conversation).call_tool(
        "generate_itinerary", {"days": 1, "budget": 5000, "start_date": FUTURE}
    )
    assert len(list(db.get(Itinerary, result["itinerary_id"]).slots)) > 1


def test_an_unknown_focus_is_rejected(db, orchestrator):
    result = orchestrator().call_tool(
        "generate_itinerary", {"days": 1, "budget": 5000, "start_date": FUTURE, "focus": "brunch"}
    )
    assert "Unknown focus" in result.get("error", "")


def test_the_trace_says_it_is_finding_a_restaurant_not_building_a_trip(db):
    from app.services.orchestrator import describe_tool_call

    label, detail = describe_tool_call(
        "generate_itinerary",
        {"days": 1, "budget": 5000, "focus": "dinner_only", "adults_only": True},
    )
    assert label == "Finding a restaurant"
    assert "dinner only" in detail and "adults only" in detail


def test_the_prompt_tells_the_model_to_match_the_scope_of_the_request(db, orchestrator):
    prompt = orchestrator().system_prompt()
    assert "focus='dinner_only'" in prompt
    assert "not a day out with a restaurant at the end of it" in prompt
    assert "adults_only" in prompt


# --- swapping a stop from chat --------------------------------------------------------------------


def test_replacing_a_stop_by_category_swaps_rather_than_removes(client, planned, db):
    """The reported bug: asked to swap a park for shopping, the assistant could only remove.

    edit_stop had no 'replace', so it removed the park, said "Day 2: Adventure & Shopping", and
    left the day one stop short with nothing shopping about it.
    """
    _, _, plan = planned
    chat = _chat_for(db, plan)
    before = chat.call_tool("get_itinerary", {})
    stop = before["days"][0]["stops"][0]

    swap = {"stop": stop["name"], "action": "replace", "category": "mall"}
    result = chat.call_tool("edit_stop", swap)

    if result.get("needs_confirmation"):
        # The mall fits only if the day runs later, so the server asks first. Answer yes — the
        # swap itself is what this test is about.
        result = chat.call_tool("edit_stop", {**swap, "allow_overrun": True})
    if "error" in result:  # a packed day may genuinely have no room for that category
        assert "place_id or category" not in result["error"]
        return
    after = chat.call_tool("get_itinerary", {})
    names = [s["name"] for d in after["days"] for s in d["stops"]]
    assert stop["name"] not in names
    assert len(after["days"][0]["stops"]) == len(before["days"][0]["stops"]), "a swap, not a removal"


def test_a_removed_stop_can_be_added_back_from_chat(client, planned, db):
    _, _, plan = planned
    chat = _chat_for(db, plan)
    stop = chat.call_tool("get_itinerary", {})["days"][0]["stops"][0]
    chat.call_tool("edit_stop", {"stop": stop["name"], "action": "remove"})
    gone = len(chat.call_tool("get_itinerary", {})["days"][0]["stops"])

    result = chat.call_tool("add_stop", {"day": 1})

    assert "error" not in result, result
    assert result["added"], result
    assert len(chat.call_tool("get_itinerary", {})["days"][0]["stops"]) == gone + 1


def test_add_stop_adds_the_exact_place_named_not_just_its_category(client, planned, db):
    """Same bug as edit_stop's replace (see test_edit_stop_swaps_in_the_exact_place_named...),
    for the plain add path: once the day has room, a named place must not become 'best fit for
    its category' either.

    Uses a runner-up the server already vouched fits this gap (from `alternatives`), rather than
    a hardcoded place name, so the test does not depend on this plan's geography or opening hours
    to make the point: asking BY NAME must not collapse into 'best fit of any kind'.
    """
    _, _, plan = planned
    chat = _chat_for(db, plan)
    stop = chat.call_tool("get_itinerary", {})["days"][0]["stops"][0]
    chat.call_tool("edit_stop", {"stop": stop["name"], "action": "remove"})

    preview = chat.call_tool("add_stop", {"day": 1})
    assert "error" not in preview, preview
    assert preview["alternatives"], "need a runner-up to prove `place` overrides the picker's own choice"
    target = preview["alternatives"][0]
    chat.call_tool("edit_stop", {"stop": preview["added"], "action": "remove"})

    result = chat.call_tool("add_stop", {"day": 1, "place": target})

    assert "error" not in result, result
    assert result["added"] == target, result
    names = [s["name"] for d in chat.call_tool("get_itinerary", {})["days"] for s in d["stops"]]
    assert target in names


def test_adding_a_stop_reports_what_else_was_available(db, planned):
    """So the reply can offer a different one instead of pretending there was no choice."""
    _, _, plan = planned
    chat = _chat_for(db, plan)
    stop = chat.call_tool("get_itinerary", {})["days"][0]["stops"][0]
    chat.call_tool("edit_stop", {"stop": stop["name"], "action": "remove"})

    result = chat.call_tool("add_stop", {"day": 1})
    assert "alternatives" in result
    assert result["added"] not in result["alternatives"]


def test_a_full_day_refuses_a_stop_from_chat_too(client, planned, db):
    _, _, plan = planned
    chat = _chat_for(db, plan)
    result = chat.call_tool("add_stop", {"day": 1, "category": "mall"})
    if "error" in result:
        assert "fits" in result["error"] or "Nothing available" in result["error"]


def test_edit_stop_swaps_in_the_exact_place_named_not_just_its_category(client, planned, db):
    """Naming a specific place must not be silently downgraded to 'best fit for its category'.

    Regression for the reported bug: asked to swap in "UAQ Mangrove Kayak" specifically, the
    server only ever accepted a category, so it picked whatever adventure place fit best — a
    different place entirely — and the reply narrated the one the user asked for anyway.
    """
    _, _, plan = planned
    chat = _chat_for(db, plan)
    stop = next(d for d in plan["days"] if d["slots"])["slots"][0]

    result = chat.call_tool("edit_stop", {
        "stop": stop["place"]["name"], "action": "replace",
        "place": "UAQ Mangrove Kayak", "category": "adventure",
    })

    assert "error" not in result, result
    assert result["replaced_with"] == "UAQ Mangrove Kayak", result
    names = [s["name"] for d in chat.call_tool("get_itinerary", {})["days"] for s in d["stops"]]
    assert "UAQ Mangrove Kayak" in names


def test_replace_without_a_target_is_an_error_not_a_removal(client, planned, db):
    """A malformed swap must never fall through to deleting the stop."""
    _, _, plan = planned
    chat = _chat_for(db, plan)
    before = chat.call_tool("get_itinerary", {})
    stop = before["days"][0]["stops"][0]

    result = chat.call_tool("edit_stop", {"stop": stop["name"], "action": "replace"})

    assert "error" in result
    after = chat.call_tool("get_itinerary", {})
    assert stop["name"] in [s["name"] for d in after["days"] for s in d["stops"]]


def test_the_trace_says_swapping_not_removing(db):
    from app.services.orchestrator import describe_tool_call

    label, detail = describe_tool_call(
        "edit_stop", {"stop": "the park", "action": "replace", "category": "mall"}
    )
    assert label == "Swapping a stop"
    # The row names the stop now, because the model does — worth showing, it is what changed.
    assert detail == "the park · for mall"


def test_the_prompt_tells_the_model_to_default_to_swap_but_allow_remove(db, orchestrator):
    prompt = orchestrator().system_prompt()
    assert "action='replace'" in prompt
    assert "action='remove'" in prompt
    assert "add_stop" in prompt


# --- the session is closed before the stream body runs ---------------------------------------------


def test_the_plan_still_links_to_its_thread_after_the_session_is_closed(client, planned, db,
                                                                        monkeypatch):
    """The reported bug: a plan built in chat, and a right pane that comes up empty on reload.

    FastAPI exits `yield` dependencies before the response body is sent, so by the time the SSE
    generator runs, get_db has already closed the request session and every instance the
    orchestrator was handed is detached. Writes to a detached instance are dropped silently:
    messages kept inserting, `conversation.itinerary_id` never updated, and `updated_at` stayed
    frozen at the moment the row was created.

    The suite shares one session between client and fixtures, so this is closed explicitly —
    otherwise the condition that broke production cannot arise here at all.
    """
    from app.models import Conversation, Itinerary, User
    from app.services.orchestrator import ChatOrchestrator, sse

    _, _, plan = planned
    row = db.query(User).filter(User.email == "planner@rihla.app").one()
    conversation = Conversation(user_id=row.id, title="New plan")
    db.add(conversation)
    db.commit()

    chat = ChatOrchestrator(db, row, conversation)

    def fake_llm(self, user_message: str):
        del user_message
        result = self.call_tool(
            "generate_itinerary", {"days": 1, "budget": 4500, "start_date": FUTURE}
        )
        assert "error" not in result, result
        self.record("assistant", "done")
        self.db.commit()
        yield sse("done", {"conversation_id": self.conversation.id})

    monkeypatch.setattr(ChatOrchestrator, "_llm", fake_llm)

    conversation_id = conversation.id
    db.close()  # exactly what the dependency's `finally` does, before the body streams
    list(chat.stream("plan it"))

    db.expire_all()
    linked = db.get(Conversation, conversation_id)
    newest = db.query(Itinerary).order_by(Itinerary.id.desc()).first()
    assert linked.itinerary_id == newest.id, "the plan belongs to no thread"


def test_a_closed_session_still_advances_the_threads_timestamp(client, planned, db, monkeypatch):
    """`updated_at` frozen at creation was the tell — every conversation UPDATE was being lost."""
    from app.models import Conversation, User
    from app.services.orchestrator import ChatOrchestrator, sse

    row = db.query(User).filter(User.email == "planner@rihla.app").one()
    conversation = Conversation(user_id=row.id, title="New plan")
    db.add(conversation)
    db.commit()
    conversation_id, before = conversation.id, conversation.updated_at

    monkeypatch.setattr(
        ChatOrchestrator,
        "_llm",
        lambda self, m: iter([sse("done", {"conversation_id": self.conversation.id})]),
    )
    chat = ChatOrchestrator(db, row, conversation)
    db.close()
    list(chat.stream("hello"))

    db.expire_all()
    assert db.get(Conversation, conversation_id).updated_at > before


# --- binding a plan to the right event ------------------------------------------------------------


def test_the_calendar_in_the_prompt_carries_the_event_ids(db, orchestrator):
    """Injecting the calendar without ids made event_id a guess.

    The model could read "Milad un Nabi, 2026-08-21" off the prompt, pass that date correctly,
    and still bind the plan to a different event because it had no id to quote.
    """
    chat = orchestrator()
    chat.call_tool(
        "create_event", {"title": "Milad un Nabi", "event_type": "eid", "date": FUTURE}
    )
    db.commit()
    event = db.query(Event).filter(Event.title == "Milad un Nabi").one()

    prompt = chat.system_prompt()
    assert f"event_id {event.id}" in prompt, prompt[prompt.index("On their calendar:"):][:300]


def test_an_event_id_that_contradicts_the_start_date_is_refused(db, orchestrator, planned):
    """Right date, wrong id — exactly what happened. Saying so beats mislabelling the plan."""
    from app.models import User

    _, _, plan = planned
    row = db.query(User).filter(User.email == "planner@rihla.app").one()
    conversation = Conversation(user_id=row.id)
    db.add(conversation)
    db.commit()
    chat = ChatOrchestrator(db, row, conversation)

    other = date.fromisoformat(FUTURE) + timedelta(days=1)
    chat.call_tool(
        {"title": "Graduation", "event_type": "graduation", "date": other.isoformat()}
        and "create_event",
        {"title": "Graduation", "event_type": "graduation", "date": other.isoformat()},
    )
    chat.call_tool("create_event", {"title": "Milad un Nabi", "event_type": "eid", "date": FUTURE})
    db.commit()

    graduation = db.query(Event).filter(Event.title == "Graduation").one()
    milad = db.query(Event).filter(Event.title == "Milad un Nabi").one()

    result = chat.call_tool(
        "generate_itinerary",
        {"days": 1, "budget": 4000, "start_date": FUTURE, "event_id": graduation.id},
    )

    assert "error" in result, result
    assert "Milad un Nabi" in result["error"], result["error"]
    assert str(milad.id) in result["error"], result["error"]


def test_an_event_id_matching_its_own_date_is_accepted(db, orchestrator, planned):
    from app.models import Itinerary, User

    _, _, plan = planned
    row = db.query(User).filter(User.email == "planner@rihla.app").one()
    conversation = Conversation(user_id=row.id)
    db.add(conversation)
    db.commit()
    chat = ChatOrchestrator(db, row, conversation)
    chat.call_tool("create_event", {"title": "Milad un Nabi", "event_type": "eid", "date": FUTURE})
    db.commit()
    milad = db.query(Event).filter(Event.title == "Milad un Nabi").one()

    result = chat.call_tool(
        "generate_itinerary",
        {"days": 1, "budget": 4000, "start_date": FUTURE, "event_id": milad.id},
    )
    assert "error" not in result, result
    assert db.get(Itinerary, result["itinerary_id"]).title == "Milad un Nabi"


def test_an_event_on_no_particular_date_is_left_alone(db, orchestrator, planned):
    """A trip that starts a day after its event is legitimate; only a *contradiction* is refused."""
    from app.models import User

    _, _, plan = planned
    row = db.query(User).filter(User.email == "planner@rihla.app").one()
    conversation = Conversation(user_id=row.id)
    db.add(conversation)
    db.commit()
    chat = ChatOrchestrator(db, row, conversation)

    later = (date.fromisoformat(FUTURE) + timedelta(days=3)).isoformat()
    chat.call_tool("create_event", {"title": "Cousins", "event_type": "family_visit", "date": FUTURE})
    db.commit()
    cousins = db.query(Event).filter(Event.title == "Cousins").one()

    result = chat.call_tool(
        "generate_itinerary",
        {"days": 1, "budget": 4000, "start_date": later, "event_id": cousins.id},
    )
    assert "error" not in result, result


# --- guests ---------------------------------------------------------------------------------


def test_the_chat_can_plan_for_more_people_than_the_household(client, planned, db):
    """The reported bug: "we will have 7 people" was acknowledged, then planned for the family.

    `planned` saves a household of two, so five guests make seven — more than one taxi seats.
    """
    _, _, plan = planned
    chat = _warned_last_turn(_chat_for(db, plan))
    result = chat.call_tool(
        "generate_itinerary",
        {"replace_existing": True, 
            "days": 1,
            "budget": 7000,
            "start_date": FUTURE,
            "guests": [{"role": "adult", "age": 30} for _ in range(5)],
        },
    )
    assert result.get("party_size") == 7, result
    assert result["vehicle"] == "two vehicles"


def test_a_rebuild_does_not_quietly_drop_the_guests(client, planned, db):
    """Same failure the transport mode had: replan, and the extra five stop being charged for."""
    _, _, plan = planned
    chat = _warned_last_turn(_chat_for(db, plan))
    first = chat.call_tool(
        "generate_itinerary",
        {"replace_existing": True, "days": 1, "budget": 7000, "start_date": FUTURE,
         "guests": [{"role": "adult", "age": 30} for _ in range(5)]},
    )
    assert first["party_size"] == 7

    again = chat.call_tool(
        "generate_itinerary", {"replace_existing": True, "days": 1, "budget": 7000, "start_date": FUTURE}
    )
    assert again["party_size"] == 7, "the rebuild fell back to the household"


def test_a_malformed_guest_is_dropped_rather_than_guessed(db):
    """The list is written by a model, so it arrives in whatever shape the model felt like."""
    from app.services.orchestrator import _guests

    people = _guests(
        {
            "guests": [
                {"role": "adult", "age": 30},
                {"role": "adult", "age": "not a number"},  # unreadable adult → default age
                {"role": "child", "age": None},  # unreadable child → dropped, never guessed
                {"role": "friend", "age": 22},  # unknown role → adult
                {"role": "child", "age": 900},  # impossible → dropped
                "just a string",
            ]
        }
    )
    assert [(p.role, p.age) for p in people] == [
        ("adult", 30),
        ("adult", 30),
        ("adult", 22),
    ]


def test_guests_are_capped(db):
    from app.services.orchestrator import MAX_GUESTS, _guests

    assert len(_guests({"guests": [{"role": "adult", "age": 30}] * 200})) == MAX_GUESTS


def test_guests_default_to_empty_when_absent_or_null(db):
    from app.services.orchestrator import _guests

    assert _guests({}) == []
    assert _guests({"guests": None}) == []


def test_a_stated_total_beats_the_models_arithmetic(client, planned, db):
    """The reported bug: "7 people" arrived as 6 guests on a household of 3 — a party of 9.

    `planned` saves a household of two, so a stated total of 7 must mean five guests however
    many the model happened to list.
    """
    _, _, plan = planned
    chat = _warned_last_turn(_chat_for(db, plan))
    result = chat.call_tool(
        "generate_itinerary",
        {"replace_existing": True, "days": 1, "budget": 7000, "start_date": FUTURE, "party_size": 7,
         "guests": [{"role": "adult", "age": 30} for _ in range(6)]},
    )
    assert result["party_size"] == 7, result


def test_a_stated_total_keeps_the_ages_the_model_did_give(db):
    """A guest child's age must survive the reconciliation — it drives their ticket band."""
    from app.services.orchestrator import _fit_party

    kid = Attendee(role="child", age=6)
    fitted = _fit_party(2, 5, [kid])
    assert len(fitted) == 3
    assert fitted[0] is kid
    assert [p.role for p in fitted[1:]] == ["adult", "adult"]


def test_a_total_no_larger_than_the_household_adds_nobody(db):
    from app.services.orchestrator import _fit_party

    assert _fit_party(4, 4, []) == []
    assert _fit_party(4, 2, []) == []


# --- a swap that only fits if the day runs late -----------------------------------------------


def test_a_tool_error_never_reaches_the_row_verbatim():
    """Tool errors are addressed to the model; the row is addressed to the user."""
    from app.services.orchestrator import summarise_tool_result

    assert summarise_tool_result(
        "edit_stop", {"error": "This plan has no stop like 'the casino'. It has: ..."}
    ) == "no change made"
    assert summarise_tool_result("edit_stop", {"error": "Unknown action 'foo'"}) == "no change made"
    # The one that is a question, not a failure, still reads as one.
    assert summarise_tool_result(
        "edit_stop", {"needs_confirmation": "window_overrun", "place": "X", "ends_at": "19:30"}
    ) == "needs your OK"


def test_a_swap_that_only_fits_past_the_window_asks_before_taking_it(client, planned, db,
                                                                     monkeypatch):
    """The reported case: the user names a category, and 'nothing fits' was the whole answer."""
    from app.services import itinerary as itinerary_service

    _, _, plan = planned
    chat = _chat_for(db, plan)
    before = chat.call_tool("get_itinerary", {})
    stop = before["days"][0]["stops"][0]

    # Force the shape the bug report describes: nothing fits the slot's own window, but something
    # does once the day is allowed to run later. Which category the seeded catalog happens to
    # offer is not what this test is about, so the relaxed search ignores it.
    def only_when_relaxed(db_, itinerary_, user_, slot_row_, category_, ignore_window=False):
        if not ignore_window:
            return None
        options = itinerary_service.alternatives_for_slot(
            db_, itinerary_, user_, slot_row_, limit=8, ignore_window=True
        )
        return options[0] if options else None

    monkeypatch.setattr(itinerary_service, "_best_alternative", only_when_relaxed)

    asked = chat.call_tool(
        "edit_stop", {"stop": stop["name"], "action": "replace", "category": "museum"}
    )
    assert asked.get("needs_confirmation") == "window_overrun"
    assert "error" not in asked  # a question, so the row must not render as a failure
    assert asked["proposed_place"] and asked["proposed_ends_at"]

    # Nothing may have changed while the question was outstanding.
    held = chat.call_tool("get_itinerary", {})
    assert stop["name"] in [s["name"] for d in held["days"] for s in d["stops"]]

    # And the user says yes.
    agreed = chat.call_tool(
        "edit_stop",
        {
            "stop": stop["name"],
            "action": "replace",
            "category": "museum",
            "allow_overrun": True,
        },
    )
    assert "needs_confirmation" not in agreed
    if "error" not in agreed:
        after = chat.call_tool("get_itinerary", {})
        assert asked["place"] in [s["name"] for d in after["days"] for s in d["stops"]]


def test_relaxing_the_window_still_respects_opening_hours(db, planned):
    """`ignore_window` may make the day late, never the plan wrong."""
    from app.models import Place
    from app.services.itinerary import context_for, placement_for
    from app.services.retrieval import to_candidate

    from app.models import Itinerary, User

    _, _, plan = planned
    itinerary = db.get(Itinerary, plan["id"])
    context = context_for(db, itinerary, db.get(User, itinerary.user_id))
    candidate = to_candidate(db.query(Place).filter(Place.min_age == 0).first())

    def no_travel(*_):
        from app.services.planner import TravelInfo

        return TravelInfo(distance_km=0.0, duration_min=0, est_cost=0.0)

    common = dict(
        travel_fn=no_travel, from_point=(candidate.lat, candidate.lng),
        following=None, day_month=6,
    )
    # Starting one minute before the venue shuts is refused either way.
    assert placement_for(
        candidate, context, earliest=candidate.closes_at - 1, latest=24 * 60,
        ignore_window=True, **common,
    ) is None
    # Ending past the nominal day end is refused only while the window is enforced.
    late = candidate.closes_at - min(candidate.avg_duration_min, context.profile.max_slot_min)
    assert placement_for(
        candidate, context, earliest=late, latest=late, ignore_window=False, **common
    ) is None
    assert placement_for(
        candidate, context, earliest=late, latest=late, ignore_window=True, **common
    ) is not None


# --- naming the constraint that actually bit ---------------------------------------------------


def test_a_swap_blocked_by_money_says_money_not_time(client, planned, db, monkeypatch):
    """The reported bug: 'not feasible within budget and time' when AED 1,850 was sitting unspent.

    One `None` stood for five different refusals, and the message guessed at both.
    """
    from app.services import itinerary as itinerary_service

    _, _, plan = planned
    chat = _chat_for(db, plan)
    stop = chat.call_tool("get_itinerary", {})["days"][0]["stops"][0]

    def only_without_the_budget_cap(db_, itinerary_, user_, slot_row_, category_,
                                    ignore_window=False, ignore_budget=False):
        if not ignore_budget:
            return None
        options = itinerary_service.alternatives_for_slot(
            db_, itinerary_, user_, slot_row_, limit=8, ignore_budget=True
        )
        return options[0] if options else None

    monkeypatch.setattr(itinerary_service, "_best_alternative", only_without_the_budget_cap)

    result = chat.call_tool(
        "edit_stop", {"stop": stop["name"], "action": "replace", "category": "museum"}
    )
    assert "Time is not the problem" in result["error"]
    assert "cheapest museum" in result["error"]


def test_a_swap_blocked_by_nothing_relaxable_says_so(client, planned, db, monkeypatch):
    from app.services import itinerary as itinerary_service

    _, _, plan = planned
    chat = _chat_for(db, plan)
    stop = chat.call_tool("get_itinerary", {})["days"][0]["stops"][0]
    monkeypatch.setattr(itinerary_service, "_best_alternative", lambda *a, **k: None)
    # Re-timing the day is the last thing tried, so a true "no" has to defeat that too.
    monkeypatch.setattr(itinerary_service, "_retimed_placement", lambda *a, **k: None)

    result = chat.call_tool(
        "edit_stop", {"stop": stop["name"], "action": "replace", "category": "museum"}
    )
    # The old wording blamed the window and the budget every time. Neither applies here.
    assert "can go anywhere in this day" in result["error"]
    assert "Neither a later finish nor a bigger budget" in result["error"]


# --- a rebuild may not quietly replace an approved plan ----------------------------------------


def test_generating_again_will_not_silently_discard_the_current_plan(client, planned, db):
    """The reported bug: 'Add an Adventure' rebuilt the trip, orphaning the approved itinerary.

    A stop the user had swapped out came back and one they never touched disappeared, because a
    whole new itinerary row was created and the conversation re-pointed at it.
    """
    from app.models import Itinerary

    _, _, plan = planned
    chat = _chat_for(db, plan)
    before = chat.call_tool("get_itinerary", {})
    count_before = db.query(Itinerary).count()

    result = chat.call_tool(
        "generate_itinerary", {"days": 1, "budget": 5000, "start_date": "2026-09-01"}
    )

    assert "Their ANSWER is what unlocks this" in result["error"]
    assert db.query(Itinerary).count() == count_before, "no orphan row"
    after = chat.call_tool("get_itinerary", {})
    assert after["itinerary_id"] == before["itinerary_id"]
    assert after["days"][0]["stops"] == before["days"][0]["stops"]


def test_an_explicit_start_over_still_works(client, planned, db):
    _, _, plan = planned
    chat = _warned_last_turn(_chat_for(db, plan))
    before = chat.call_tool("get_itinerary", {})

    result = chat.call_tool(
        "generate_itinerary",
        {"days": 1, "budget": 5000, "start_date": "2026-09-01", "replace_existing": True},
    )

    assert "error" not in result
    assert result["itinerary_id"] != before["itinerary_id"]


def test_a_fresh_thread_is_not_blocked_by_an_older_plan(client, planned, db):
    """The guard protects THIS conversation's plan, not the user's right to plan again."""
    from app.models import Conversation, User

    _, _, plan = planned
    row = db.query(User).filter(User.email == "planner@rihla.app").one()
    fresh = Conversation(user_id=row.id)
    db.add(fresh)
    db.commit()

    result = ChatOrchestrator(db, row, fresh).call_tool(
        "generate_itinerary", {"days": 1, "budget": 5000, "start_date": "2026-09-01"}
    )
    assert "error" not in result
    assert result["itinerary_id"] != plan["id"]


def test_the_prompt_no_longer_offers_start_over_as_a_way_around_a_failed_edit(db, orchestrator):
    prompt = orchestrator().system_prompt()
    assert "agreeing to an EDIT" in prompt
    assert "replace_existing" in prompt


# --- a filler itinerary_id must not hide the plan ----------------------------------------------


@pytest.mark.parametrize("junk", [0, 999])
def test_a_junk_itinerary_id_does_not_report_the_plan_as_missing(client, planned, db, junk):
    """The reported bug: 'no plan yet' on a thread that plainly had one.

    Strict schemas make itinerary_id a property the model must send on EVERY call, and 0 is what
    a model reaches for when it has no value. That answer sent it off building a replacement.
    """
    from app.services.orchestrator import summarise_tool_result

    _, _, plan = planned
    chat = _chat_for(db, plan)
    truth = chat.call_tool("get_itinerary", {})

    result = chat.call_tool("get_itinerary", {"itinerary_id": junk})

    assert result["itinerary_id"] == truth["itinerary_id"]
    assert summarise_tool_result("get_itinerary", result) != "no plan yet"
    # Editing tools resolve through the same path, and said "nothing to edit" for the same reason.
    stop = truth["days"][0]["stops"][0]
    edit = chat.call_tool(
        "edit_stop", {"itinerary_id": junk, "stop": stop["name"], "action": "remove"}
    )
    assert "nothing to edit" not in edit.get("error", "")


def test_an_unknown_id_in_the_arguments_is_simply_ignored(client, planned, db):
    """Nothing supplies one any more, but a stray key must not resurrect the old failure."""
    _, _, plan = planned
    chat = _chat_for(db, plan)
    for junk in (0, 999):
        assert chat.call_tool("get_itinerary", {"itinerary_id": junk})["itinerary_id"] == plan["id"]


# --- the model cannot grant itself a rebuild it was just refused -------------------------------


def test_replace_existing_cannot_grant_itself(client, planned, db):
    """The reported bug: the guard was a formality.

    "This plan is okay, let us finalize it" became a rebuild that discarded every edit. The flag
    was free to set, and the model set it on its first attempt, so the refusal it was meant to
    follow never happened. Permission arrives a turn after the warning, because that is how long
    the user takes to answer.
    """
    _, _, plan = planned
    chat = _chat_for(db, plan)
    args = {"days": 1, "budget": 5000, "start_date": FUTURE}

    straight_in = chat.call_tool("generate_itinerary", {**args, "replace_existing": True})
    assert "does not grant itself" in straight_in["error"]
    assert chat.call_tool("get_itinerary", {})["itinerary_id"] == plan["id"], "plan untouched"

    # Refused, then insisting, still inside the turn the user has not spoken in.
    assert "does not grant itself" in chat.call_tool(
        "generate_itinerary", {**args, "replace_existing": True}
    )["error"]
    assert chat.call_tool("get_itinerary", {})["itinerary_id"] == plan["id"]

    # The warning stuck, so once the user has actually answered it goes through.
    assert chat.conversation.rebuild_warned is True
    chat.warned_at_turn_start = True
    went_through = chat.call_tool("generate_itinerary", {**args, "replace_existing": True})
    assert "error" not in went_through
    assert went_through["itinerary_id"] != plan["id"]
    assert chat.conversation.rebuild_warned is False, "spent — the next rebuild asks again"


# --- a turn never ends without a reply ---------------------------------------------------------


class _FakeStream:
    """An OpenAI chat client that emits scripted rounds: a tool call, then prose."""

    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.calls: list[dict] = []
        chat = type("Chat", (), {})()
        chat.completions = self
        self.chat = chat

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.rounds.pop(0) if self.rounds else [])


def _chunk(content=None, tool=None):
    delta = type("Delta", (), {"content": content, "tool_calls": None})()
    if tool:
        fn = type("Fn", (), {"name": tool[0], "arguments": tool[1]})()
        delta.tool_calls = [type("TC", (), {"index": 0, "id": "c1", "function": fn})()]
    choice = type("Choice", (), {"delta": delta})()
    return type("Chunk", (), {"choices": [choice]})()


def test_a_turn_that_spends_every_round_on_tools_still_answers(client, planned, db, monkeypatch):
    """The reported bug: three assistant messages saved as an empty string.

    The tool rows scrolled past and nothing explained them — the loop hit MAX_TOOL_ROUNDS while
    the model was still calling tools, so `answer` was never assigned.
    """
    from app.services import orchestrator as orch

    _, _, plan = planned
    chat = _chat_for(db, plan)

    tool_round = [_chunk(tool=("get_itinerary", '{"itinerary_id": null}'))]
    fake = _FakeStream([list(tool_round) for _ in range(orch.MAX_TOOL_ROUNDS)]
                       + [[_chunk(content="Here is where things stand.")]])
    monkeypatch.setattr(orch, "wrap_openai", lambda c: c)
    monkeypatch.setitem(
        __import__("sys").modules, "openai", type("M", (), {"OpenAI": lambda **_: fake})
    )

    frames = "".join(chat.stream("what does the plan look like?"))

    assert "Here is where things stand." in frames
    saved = [m for m in _messages(db, chat) if m.role == "assistant"][-1]
    assert saved.content, "an empty bubble is what the user actually saw"
    # The rescue round must not hand the tools back, or it can loop forever.
    assert "tools" not in fake.calls[-1]


def _messages(db, chat):
    from app.models import Message

    return db.query(Message).filter(Message.conversation_id == chat.conversation.id).all()


# --- naming a stop instead of numbering it -----------------------------------------------------


def test_the_words_the_user_actually_used_find_the_stop(client, planned, db):
    """'Replace shopping' has to reach a stop whose category is `mall`.

    The transcripts say "shopping", "dining", "the park at the end". None of those is a slot id,
    and every one of them is unambiguous against a plan the server is already holding.
    """
    from app.models import Itinerary
    from app.services.itinerary import find_stop

    _, _, plan = planned
    itinerary = db.get(Itinerary, plan["id"])
    stops = {s["place"]["category"]: s["place"]["name"] for d in plan["days"] for s in d["slots"]}

    for phrase, category in (("shopping", "mall"), ("the shopping stop", "mall")):
        if category not in stops:
            continue
        assert find_stop(db, itinerary, phrase).place.name == stops[category]

    # A name always wins, however it is cased or padded.
    any_name = next(iter(stops.values()))
    assert find_stop(db, itinerary, f"  {any_name.upper()} ").place.name == any_name


def test_a_stop_that_is_not_there_is_told_what_is(client, planned, db):
    """The old answer was "call get_itinerary for current slot_ids" — a round trip for data the
    server already had in its hand."""
    from app.models import Itinerary
    from app.services.itinerary import find_stop

    _, _, plan = planned
    itinerary = db.get(Itinerary, plan["id"])

    with pytest.raises(ValueError) as caught:
        find_stop(db, itinerary, "the casino")

    message = str(caught.value)
    assert "no stop like 'the casino'" in message
    for slot in plan["days"][0]["slots"]:
        assert slot["place"]["name"] in message, "the reply can name what IS there"


def test_an_ambiguous_description_asks_rather_than_guesses(client, planned, db):
    from app.models import Itinerary
    from app.services.itinerary import find_stop

    _, _, plan = planned
    itinerary = db.get(Itinerary, plan["id"])
    categories = [s["place"]["category"] for d in plan["days"] for s in d["slots"]]
    doubled = next((c for c in categories if categories.count(c) > 1), None)
    if doubled is None:
        pytest.skip("this generated plan has no repeated category")

    with pytest.raises(ValueError, match="more than one stop"):
        find_stop(db, itinerary, doubled.replace("_", " "))


def test_a_repeated_name_on_one_day_is_told_apart_by_day_and_meal(client, planned, db):
    """The same place can sit twice on one day — a lunch and a dinner at the same restaurant.

    A bare name, even with the day given, is genuinely ambiguous between the two. But "day 4
    dinner" is what the user actually says, and it has to land on the later of the two rather
    than loop forever asking a question it already has the answer to.
    """
    from app.models import Itinerary, Slot
    from app.services.itinerary import find_stop

    _, _, plan = planned
    itinerary = db.get(Itinerary, plan["id"])
    day0 = (
        db.query(Slot)
        .filter(Slot.itinerary_id == itinerary.id, Slot.day_index == 0)
        .order_by(Slot.position)
        .all()
    )
    original = day0[0]

    duplicate = Slot(
        itinerary_id=itinerary.id, day_index=0, position=max(s.position for s in day0) + 1,
        place_id=original.place_id, start_time="23:59", end_time="23:59",
    )
    db.add(duplicate)
    db.commit()

    with pytest.raises(ValueError, match="more than one stop"):
        find_stop(db, itinerary, original.place.name)

    with pytest.raises(ValueError, match="more than one stop"):
        find_stop(db, itinerary, original.place.name, day=1)

    found = find_stop(db, itinerary, f"{original.place.name} dinner", day=1)
    assert found.id == duplicate.id


def test_three_sittings_at_one_place_are_told_apart_by_the_meal_word(client, planned, db):
    """Three sittings on one day, and the meal word has to land on the right one.

    Resolving by position could not do this: "lunch" was whichever match came earliest, which is
    breakfast once a day has three. And "breakfast" was not handled at all, though the prompt has
    always told the model to disambiguate with it — so the one word the user was invited to say
    dead-ended in the same ambiguity error it was meant to answer, which the prompt then forbade
    retrying.
    """
    from app.models import Itinerary, Slot, TravelSegment
    from app.services.itinerary import find_stop

    _, _, plan = planned
    itinerary = db.get(Itinerary, plan["id"])
    day0 = (
        db.query(Slot)
        .filter(Slot.itinerary_id == itinerary.id, Slot.day_index == 0)
        .order_by(Slot.position)
        .all()
    )
    place_id, name = day0[0].place_id, day0[0].place.name

    # Segments hold FKs to slots, so they go first — the same order persist_plan uses.
    db.query(TravelSegment).filter(TravelSegment.itinerary_id == itinerary.id).delete()
    db.flush()
    for row in day0:
        db.delete(row)
    db.flush()

    sittings = {}
    for position, (label, start, end) in enumerate(
        (("breakfast", "08:15", "09:00"), ("lunch", "12:41", "13:30"), ("dinner", "19:24", "20:30"))
    ):
        row = Slot(
            itinerary_id=itinerary.id, day_index=0, position=position,
            place_id=place_id, start_time=start, end_time=end,
        )
        db.add(row)
        sittings[label] = row
    db.commit()

    for label, row in sittings.items():
        assert find_stop(db, itinerary, f"{name} {label}", day=1).id == row.id, label

    # The bare name is still genuinely ambiguous — three sittings, nothing said about which.
    with pytest.raises(ValueError, match="more than one stop"):
        find_stop(db, itinerary, name, day=1)


def test_an_empty_description_is_an_omission_not_a_match(client, planned, db):
    """Every stop contains the empty string, so this read as ambiguity instead of a mistake."""
    from app.models import Itinerary
    from app.services.itinerary import find_stop

    _, _, plan = planned
    with pytest.raises(ValueError, match="Say which stop"):
        find_stop(db, db.get(Itinerary, plan["id"]), "   ")


def test_no_chat_tool_asks_the_model_for_a_database_id():
    """What the conversation knows, the conversation supplies. What it cannot know is `event_id`.

    An id is a value the model can only have by looking it up, so every one of them is a round
    trip it can skip and a number it can invent. `event_id` survives because which event the user
    means is their intent, not plumbing — and the system prompt lists the ids outright.
    """
    supplied = {
        tool["function"]["name"]: sorted(
            k for k in tool["function"]["parameters"].get("properties", {}) if k.endswith("_id")
        )
        for tool in TOOLS
    }
    assert {name: ids for name, ids in supplied.items() if ids} == {
        "generate_itinerary": ["event_id"]
    }


# --- a swap is a request about the day, not about the hour -------------------------------------


def _abu_dhabi_day(db, planned):
    """The reported scenario: seven people, one day, AED 7,000, confined to Abu Dhabi."""
    from app.models import Conversation, User

    row = db.query(User).filter(User.email == "planner@rihla.app").one()
    conversation = Conversation(user_id=row.id)
    db.add(conversation)
    db.commit()
    chat = ChatOrchestrator(db, row, conversation)
    built = chat.call_tool("generate_itinerary", {
        "days": 1, "budget": 7000, "start_date": FUTURE,
        "party_size": 7, "emirates": ["Abu Dhabi"],
    })
    assert "error" not in built, built
    return chat


def _thin_day_to(chat, limit: int) -> None:
    """Remove stops until day 1 holds at most `limit` of them.

    Every stop is addressed precisely, and a pass that removes nothing fails instead of going
    round again. Removing `stops[0]` blindly did not terminate: the planner puts the same
    restaurant on a day twice on purpose, a bare name cannot say which sitting is meant, and
    `call_tool` reports that as a result rather than raising — so the day never got smaller and
    the loop asked forever.
    """
    from app.services.itinerary import MEAL_SITTINGS

    def handle_for(stop, names) -> str | None:
        if names.count(stop["name"]) == 1:
            return stop["name"]
        start = int(stop["start"][:2]) * 60 + int(stop["start"][3:])
        for word, (opens, closes) in MEAL_SITTINGS.items():
            if opens <= start < closes:
                return f"{stop['name']} {word}"
        return None

    while True:
        stops = chat.call_tool("get_itinerary", {})["days"][0]["stops"]
        if len(stops) <= limit:
            return
        names = [s["name"] for s in stops]
        for stop in stops:
            handle = handle_for(stop, names)
            if handle is None:
                continue
            if "error" not in chat.call_tool("edit_stop", {"stop": handle, "action": "remove"}):
                break
        else:
            raise AssertionError(f"no stop on this day could be removed: {names}")


def test_a_swap_the_hour_forbids_offers_to_re_time_the_day(client, planned, db):
    """The reported bug: 'replace shopping with an adventure' refused with AED 1,168 unspent.

    Every adventure in range is shut by 20:35, and the replacement was pinned to the outgoing
    stop's place in the clock. Neither budget nor a later finish was ever the constraint, so
    waiving them — as the user did, twice — could not have helped.
    """
    chat = _abu_dhabi_day(db, planned)
    # A packed day has no room to re-time INTO — the honest answer there is "that would cost the
    # day a stop", tested separately. Thin it out so re-timing is the thing under test.
    _thin_day_to(chat, 2)

    plan = chat.call_tool("get_itinerary", {})
    last = plan["days"][0]["stops"][-1]

    # A museum, because museums shut in the evening: a later finish cannot help, which is what
    # separates this from the window-overrun question.
    asked = chat.call_tool(
        "edit_stop", {"stop": last["name"], "action": "replace", "category": "museum"}
    )
    if asked.get("needs_confirmation") != "day_reorder":
        pytest.skip(f"this day admits a museum without re-timing: {asked}")

    assert asked["needs_confirmation"] == "day_reorder"
    assert asked["proposed_place"] and asked["proposed_duration_min"] > 0
    # Nothing may move while the question is outstanding.
    held = chat.call_tool("get_itinerary", {})
    assert [s["name"] for s in held["days"][0]["stops"]] == [
        s["name"] for s in plan["days"][0]["stops"]
    ]

    agreed = chat.call_tool(
        "edit_stop",
        {"stop": last["name"], "action": "replace", "category": "museum",
         "allow_reorder": True},
    )
    assert "error" not in agreed, agreed
    after = chat.call_tool("get_itinerary", {})
    names = [s["name"] for s in after["days"][0]["stops"]]
    assert asked["proposed_place"] in names, "the place it named is the place it placed"
    assert last["name"] not in names
    assert len(names) == len(plan["days"][0]["stops"]), "a swap, not a removal"


def test_re_timing_never_costs_the_day_a_stop(client, planned, db):
    """Shifting the later stops can push one past its own closing time, and the repair pass
    deletes what it cannot fix. Losing a stop the user never mentioned is the failure this whole
    thread has been about."""
    chat = _abu_dhabi_day(db, planned)
    before = chat.call_tool("get_itinerary", {})["days"][0]["stops"]
    last = before[-1]

    chat.call_tool(
        "edit_stop",
        {"stop": last["name"], "action": "replace", "category": "adventure",
         "allow_reorder": True},
    )

    after = chat.call_tool("get_itinerary", {})["days"][0]["stops"]
    assert len(after) == len(before), [s["name"] for s in after]


# --- dining is a meal, not a category ----------------------------------------------------------


def test_a_day_will_not_take_a_third_sit_down_meal(client, planned, db):
    """The reported bug: three dining stops in a row at the end of the day.

    The generator treats dining as a role — `assemble_day` builds around meal windows and refuses
    dining when filling an activity slot — and the edit paths treated it as a plain category, so
    every edit eroded the structure generation had built.
    """
    from app.models import Itinerary
    from app.services.itinerary import free_meal_windows, context_for, load_plan

    chat = _abu_dhabi_day(db, planned)
    plan = chat.call_tool("get_itinerary", {})
    itinerary = db.get(Itinerary, plan["itinerary_id"])
    day = load_plan(db, itinerary).days[0]
    profile = context_for(db, itinerary, chat.user).profile

    # Fill both meal windows, so any further dining would be a third meal.
    for role in list(free_meal_windows(profile, day)):
        del role
        for stop in plan["days"][0]["stops"]:
            chat.call_tool(
                "edit_stop",
                {"stop": stop["name"], "action": "replace", "category": "casual_dining"},
            )
        plan = chat.call_tool("get_itinerary", {})

    final = chat.call_tool("get_itinerary", {})["days"][0]["stops"]
    dining = [s for s in final if s["category"] in ("casual_dining", "fine_dining")]
    assert len(dining) <= 2, [s["name"] for s in dining]


def test_meal_roles_come_from_the_profile_the_generator_plans_with(client, planned, db):
    from app.models import Itinerary
    from app.services.itinerary import context_for, free_meal_windows, load_plan, meal_role

    _, _, plan = planned
    itinerary = db.get(Itinerary, plan["id"])
    profile = context_for(db, itinerary, itinerary.user_id and db.get(
        __import__("app.models", fromlist=["User"]).User, itinerary.user_id)).profile

    lunch = next(w for w in profile.meal_windows if w[0] == "lunch")
    assert meal_role(profile, lunch[1] + 10, lunch[1] + 70) == "lunch"
    # 03:00 is nobody's mealtime.
    assert meal_role(profile, 3 * 60, 4 * 60) is None

    day = load_plan(db, itinerary).days[0]
    assert free_meal_windows(profile, day) <= {label for label, _, _ in profile.meal_windows}


def test_adding_a_third_meal_says_which_two_are_already_there(client, planned, db):
    from app.models import Itinerary, User
    from app.services import itinerary as isvc

    chat = _abu_dhabi_day(db, planned)
    itinerary = db.get(Itinerary, chat.call_tool("get_itinerary", {})["itinerary_id"])
    user = db.get(User, itinerary.user_id)
    monkey = isvc.free_meal_windows

    try:
        isvc.free_meal_windows = lambda *a, **k: set()   # every meal already eaten
        with pytest.raises(ValueError, match="third sit-down meal|already has"):
            isvc.add_stop(db, itinerary, user, day_index=0, category="casual_dining")
    finally:
        isvc.free_meal_windows = monkey


def test_the_hop_cap_applies_between_stops_not_to_the_drive_from_home(client, planned, db):
    """Generation exempts the leg from home; every edit path applied the 60 km cap to it.

    `assemble_day` guards its hop check with `previous_position is not None` — "this far from the
    LAST one is a different trip". A day generated 130 km away in Abu Dhabi could therefore never
    have its opening stop swapped, added to, or re-timed into, because the only reference point
    for the first slot is the family's home in Dubai.
    """
    from app.services.itinerary import placement_for
    from app.services.planner import MAX_HOP_KM, TravelInfo
    from app.services.itinerary import context_for
    from app.models import Itinerary, Place, User
    from app.services.retrieval import to_candidate

    _, _, plan = planned
    itinerary = db.get(Itinerary, plan["id"])
    context = context_for(db, itinerary, db.get(User, itinerary.user_id))
    candidate = to_candidate(db.query(Place).filter(Place.min_age == 0).first())

    def no_travel(*_):
        return TravelInfo(distance_km=0.0, duration_min=0, est_cost=0.0)

    # A point well beyond the cap from the candidate, standing in for home.
    faraway = (candidate.lat + 2.0, candidate.lng)
    common = dict(
        travel_fn=no_travel, from_point=faraway, following=None, day_month=6,
        earliest=candidate.opens_at, latest=24 * 60,
    )
    assert MAX_HOP_KM < 200, "this test assumes the cap is well under the 2-degree offset used"

    assert placement_for(candidate, context, from_origin=False, **common) is None
    assert placement_for(candidate, context, from_origin=True, **common) is not None


def test_the_prompt_asks_for_preferences_to_be_recorded_alongside_the_edit(db, orchestrator):
    """The reported bug: "I don't like Kayaking" made the swap and recorded nothing.

    record_preference existed, worked, and was never once mentioned in a system prompt that
    instructs the model about every other tool in detail. With an actionable edit in the same
    sentence, the model did the edit and dropped the preference.
    """
    prompt = orchestrator().system_prompt()
    assert "record_preference in the same turn as the edit" in prompt
    assert "two things at once" in prompt


def test_recording_a_dislike_actually_reaches_the_scorer(client, planned, db):
    """End to end: what the tool writes has to be what the planner reads."""
    from app.models import Itinerary, Place, User
    from app.services.itinerary import context_for
    from app.services.planner import preference_signal
    from app.services.retrieval import to_candidate

    _, _, plan = planned
    chat = _chat_for(db, plan)
    kayak = db.query(Place).filter(Place.name.like("%Kayak%")).first()
    if kayak is None:
        pytest.skip("no kayak in the catalog")

    chat.call_tool("record_preference", {"kind": "dislike", "subject": "kayaking"})
    db.commit()

    context = context_for(db, db.get(Itinerary, plan["id"]), db.get(User, chat.user.id))
    assert preference_signal(to_candidate(kayak), context.preferences) < 0


# --- a question is not an answer ---------------------------------------------------------------


def test_a_confirmation_carries_the_plan_as_it_really_stands(client, planned, db):
    """The reported bug: the chat described a swap that never happened.

    The result was `{"needs_confirmation": ..., "place": "Hudayriyat Adventure Park", ...}` — a
    place name and a plausible story — and the model reported it as done. The right pane was
    correct all along and the user was told otherwise, which is the worst way to be wrong.
    """
    chat = _abu_dhabi_day(db, planned)
    _thin_day_to(chat, 2)

    before = [s["name"] for s in chat.call_tool("get_itinerary", {})["days"][0]["stops"]]
    asked = chat.call_tool(
        "edit_stop", {"stop": before[-1], "action": "replace", "category": "museum"}
    )
    if not asked.get("needs_confirmation"):
        pytest.skip("this day needed no confirmation")

    assert asked["applied"] is False, "the model has to be told nothing happened"
    assert asked["plan_is_unchanged"] == before, "the truth travels with the question"
    # Nothing may be named in a way that reads like an outcome.
    assert "place" not in asked and "ends_at" not in asked and "duration_min" not in asked
    assert asked["proposed_place"] not in asked["plan_is_unchanged"]


def test_a_confirmation_does_not_nudge_the_right_pane(client, planned, db):
    """`touched_itinerary` is what makes the stream emit `itinerary_updated`. A question changed
    nothing, so a refresh would only redraw the same plan and imply otherwise."""
    chat = _abu_dhabi_day(db, planned)
    _thin_day_to(chat, 2)

    last = chat.call_tool("get_itinerary", {})["days"][0]["stops"][-1]["name"]
    chat.touched_itinerary = None
    asked = chat.call_tool("edit_stop", {"stop": last, "action": "replace", "category": "museum"})
    if not asked.get("needs_confirmation"):
        pytest.skip("this day needed no confirmation")
    assert chat.touched_itinerary is None


def test_finalising_a_plan_is_not_a_thing_the_model_has_to_do(db, orchestrator):
    """"Let us finalize it" found no finalise tool and reached for the most destructive one.

    There is nothing to call: the plan is already saved. The prompt now says so, and the refusal
    repeats it, because the refusal is what the model reads when it guesses wrong.
    """
    prompt = orchestrator().system_prompt()
    assert "there is no " in prompt and "finalising, confirming or committing" in prompt
    assert "A tool that comes back asking has changed NOTHING" in prompt


def test_the_first_refusal_does_not_name_the_flag_that_defeats_it(client, planned, db):
    """Naming `replace_existing` in the refusal is what taught the model to set it."""
    _, _, plan = planned
    chat = _chat_for(db, plan)
    refusal = chat.call_tool(
        "generate_itinerary", {"days": 1, "budget": 5000, "start_date": FUTURE}
    )["error"]
    assert "replace_existing" not in refusal
    assert "no finalising, confirming or committing" in refusal


def test_the_warning_survives_the_turn_it_was_given_in(client, planned, db):
    """Consent arrives a turn after the warning, so the warning has to outlive the turn."""
    from app.models import Conversation

    _, _, plan = planned
    chat = _chat_for(db, plan)
    chat.call_tool("generate_itinerary", {"days": 1, "budget": 5000, "start_date": FUTURE})
    db.commit()

    assert db.get(Conversation, chat.conversation.id).rebuild_warned is True


def test_the_calendar_answers_which_events_have_no_plan(client, planned, db, orchestrator):
    """The reported bug: two planned events reported as unplanned, and the unplanned one as done.

    The flags were right — `Event.planned` is set on generate and the database agreed. The model
    was handed three booleans and asked to negate them, and got the sentence backwards. So it is
    handed words, and the negation it kept fumbling is done for it.
    """
    chat = orchestrator()
    chat.call_tool("create_event", {"title": "Has a plan", "event_type": "eid", "date": FUTURE})
    chat.call_tool("create_event", {"title": "Has none", "event_type": "other", "date": FUTURE})
    db.commit()
    db.query(Event).filter(Event.title == "Has a plan").update({"planned": True})
    db.commit()

    result = chat.call_tool("get_upcoming_events", {"horizon_days": 365})
    by_title = {e["title"]: e for e in result["events"]}

    assert by_title["Has a plan"]["status"] == "planned"
    assert by_title["Has none"]["status"] == "no plan yet"
    # No bare boolean left to invert.
    assert "planned" not in by_title["Has none"]
    # And the question is answered rather than left to be derived.
    assert result["without_a_plan"] == ["Has none"]


# --- browsing the places catalog ---------------------------------------------------------------


@pytest.fixture
def catalog(db):
    """The real seeded catalog, descriptions included.

    Deliberately the shipped places.json rather than a handful of fakes: what is under test here
    is paging and capping over a catalog far larger than one reply, and 146 Dubai rows is the
    thing that makes page 2 mean something.
    """
    from pathlib import Path

    from app.models import Place
    from app.seed import default_price_bands

    rows = json.loads(
        (Path(__file__).resolve().parent.parent / "app" / "data" / "places.json").read_text()
    )
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
                description=row.get("description", ""),
            )
        )
    db.commit()
    return rows


def test_find_places_is_exposed_as_a_tool():
    assert "find_places" in {tool["function"]["name"] for tool in TOOLS}


def test_a_whole_emirate_comes_back_a_page_at_a_time(catalog, orchestrator):
    result = orchestrator().call_tool("find_places", {"emirate": "Dubai"})

    assert len(result["places"]) == 20
    assert result["total_matching"] == sum(1 for row in catalog if row["emirate"] == "Dubai")
    assert result["has_more"] is True
    assert {place["emirate"] for place in result["places"]} == {"Dubai"}


def test_full_details_come_back_ten_at_a_time(catalog, orchestrator):
    result = orchestrator().call_tool("find_places", {"emirate": "Dubai", "detail": "full"})

    assert len(result["places"]) == 10
    assert all(place["description"] for place in result["places"])


def test_a_brief_listing_carries_no_descriptions(catalog, orchestrator):
    """The point of two detail levels is that the brief one is actually brief."""
    result = orchestrator().call_tool("find_places", {"emirate": "Dubai"})

    assert "description" not in result["places"][0]


def test_page_two_continues_rather_than_repeating(catalog, orchestrator):
    tool = orchestrator()
    first = tool.call_tool("find_places", {"emirate": "Dubai"})
    second = tool.call_tool("find_places", {"emirate": "Dubai", "page": 2})

    names = {place["name"] for place in first["places"]}
    assert not names & {place["name"] for place in second["places"]}
    assert second["page"] == 2


def test_the_last_page_says_there_is_no_more(catalog, orchestrator):
    result = orchestrator().call_tool("find_places", {"emirate": "Umm Al Quwain"})

    assert result["has_more"] is False
    assert len(result["places"]) == result["total_matching"]


def test_places_can_be_narrowed_to_one_kind(catalog, orchestrator):
    result = orchestrator().call_tool("find_places", {"emirate": "Dubai", "category": "beach"})

    assert result["places"]
    assert {place["category"] for place in result["places"]} == {"beach"}


def test_a_budget_ceiling_excludes_the_pricier_places(catalog, orchestrator):
    result = orchestrator().call_tool("find_places", {"max_price_adult": 0})

    assert result["places"]
    assert all(place["price_adult"] == 0 for place in result["places"])
    assert result["total_matching"] == sum(
        1 for row in catalog if row.get("price_adult", 0) == 0
    )


def test_naming_one_place_returns_it_in_full(catalog, orchestrator):
    """A named place is a request for detail, whatever the detail argument says."""
    result = orchestrator().call_tool("find_places", {"name": "Ski Dubai"})

    assert result["total_matching"] == 1
    assert result["places"][0]["name"] == "Ski Dubai"
    assert result["places"][0]["description"]


def test_an_ambiguous_name_offers_the_candidates_rather_than_picking(catalog, orchestrator):
    result = orchestrator().call_tool("find_places", {"name": "beach"})

    assert result["total_matching"] > 1
    assert len(result["places"]) > 1
    # Brief, because the model's next move is to ask which one — not to describe all of them.
    assert "description" not in result["places"][0]


def test_a_name_that_matches_nothing_says_so(catalog, orchestrator):
    result = orchestrator().call_tool("find_places", {"name": "Eiffel Tower"})

    assert result["places"] == []
    assert result["no_match"] == "Eiffel Tower"


def test_an_unknown_emirate_is_refused_by_name(catalog, orchestrator):
    result = orchestrator().call_tool("find_places", {"emirate": "Al Ain"})

    assert "Abu Dhabi" in result["error"]
    assert "places" not in result


# --- searching the catalog by meaning ----------------------------------------------------------


def test_a_query_ranks_by_meaning_and_says_so(catalog, orchestrator, monkeypatch, db):
    """With embeddings available, order comes from Chroma rather than the alphabet."""
    from app.models import Place
    from app.services import orchestrator as module

    far = db.scalars(select(Place).where(Place.name == "Aquaventure Waterpark")).one()
    near = db.scalars(select(Place).where(Place.name == "Yas Waterworld")).one()
    monkeypatch.setattr(
        module, "semantic_similarities", lambda query, **kw: {near.id: 0.9, far.id: 0.4}
    )

    result = orchestrator().call_tool("find_places", {"query": "water rides"})

    assert [place["name"] for place in result["places"]] == [near.name, far.name]
    assert result["matched_by"] == "meaning"


def test_a_query_falls_back_to_keywords_without_embeddings(catalog, orchestrator):
    """The suite runs with embeddings disabled — the same path as a missing API key."""
    result = orchestrator().call_tool("find_places", {"query": "water rides"})

    assert result["matched_by"] == "keywords"
    assert result["places"]


def test_a_query_is_narrowed_by_the_other_filters(catalog, orchestrator):
    result = orchestrator().call_tool(
        "find_places", {"query": "water rides", "emirate": "Abu Dhabi"}
    )

    assert result["places"]
    assert {place["emirate"] for place in result["places"]} == {"Abu Dhabi"}


def test_a_narrow_filter_the_vector_pool_missed_still_answers(catalog, orchestrator, monkeypatch):
    """Chroma scores only the top slice of the whole catalog.

    A small emirate can have none of its places in that slice, and coming back empty would read
    as "there is nothing there" when the truth is "nothing there ranked in the global top 200".
    """
    from app.services import orchestrator as module

    monkeypatch.setattr(module, "semantic_similarities", lambda query, **kw: {-1: 0.9})

    result = orchestrator().call_tool(
        "find_places", {"query": "beach", "emirate": "Umm Al Quwain"}
    )

    assert result["places"]
    assert result["matched_by"] == "keywords"


def test_a_query_matching_nothing_says_so_rather_than_listing_everything(catalog, orchestrator):
    result = orchestrator().call_tool("find_places", {"query": "skiing in the alps"})

    assert result["places"] == []
    assert result["no_match"] == "skiing in the alps"


def test_an_age_excludes_places_that_child_cannot_enter(catalog, orchestrator):
    result = orchestrator().call_tool("find_places", {"suitable_for_age": 8, "detail": "full"})

    assert result["places"]
    assert all(place["min_age"] <= 8 for place in result["places"])
    assert result["total_matching"] == sum(1 for row in catalog if row.get("min_age", 0) <= 8)
    assert result["total_matching"] < len(catalog)


def test_the_whole_question_answers_in_one_call(catalog, orchestrator):
    """'What places can my child who enjoys water rides visit in Abu Dhabi?'"""
    result = orchestrator().call_tool(
        "find_places",
        {"query": "child who enjoys water rides", "emirate": "Abu Dhabi", "suitable_for_age": 7},
    )

    names = [place["name"] for place in result["places"]]
    assert names, "the catalog has Abu Dhabi water attractions a 7-year-old can enter"
    assert "Yas Waterworld" in names


def test_a_listing_without_a_query_stays_alphabetical(catalog, orchestrator):
    """The meaning path is opt-in; plain browsing must not start ranking."""
    result = orchestrator().call_tool("find_places", {"emirate": "Dubai"})

    names = [place["name"] for place in result["places"]]
    assert names == sorted(names)
    assert "matched_by" not in result


# --- the plan can be moved, shortened and re-solved ---------------------------------------------


def _dubai_day(db, planned, days: int = 2):
    """A plan that lands in Dubai, so moving it somewhere else is a real change."""
    from app.models import Conversation, User

    row = db.query(User).filter(User.email == "planner@rihla.app").one()
    conversation = Conversation(user_id=row.id)
    db.add(conversation)
    db.commit()
    chat = ChatOrchestrator(db, row, conversation)
    built = chat.call_tool("generate_itinerary", {
        "days": days, "budget": 6000, "start_date": FUTURE, "emirates": ["Dubai"],
    })
    assert "error" not in built, built
    return chat


def test_moving_the_starting_point_keeps_the_trip_where_it_is(client, planned, db):
    """The reported bug, first half: "we live in Abu Dhabi" was answered by claiming the whole
    plan had moved to Abu Dhabi, twice, while returning the identical Dubai itinerary.

    It says where the car sets off, not what the trip is — so the stops stay and only the driving
    is re-costed. The tool that changes what the trip IS is replace_plan, and it costs the stops.
    """
    chat = _dubai_day(db, planned)
    before = chat.call_tool("get_itinerary", {})
    names = [s["name"] for d in before["days"] for s in d["stops"]]
    origin_before = chat._resolve_itinerary().start_lat

    moved = chat.call_tool("set_origin", {"emirate": "Abu Dhabi"})

    assert "error" not in moved, moved
    assert moved["origin_emirate"] == "Abu Dhabi"
    # _plan_result lists a day's stops by name, so these are strings, not slot dicts.
    assert [name for d in moved["days"] for name in d["stops"]] == names, "a stop was lost"
    assert chat._resolve_itinerary().start_lat < origin_before, "the origin did not move south"


def test_only_replace_plan_can_move_the_region(client, planned, db):
    """The reported bug, second half: "change the location of the plan to Abu Dhabi".

    emirates_json was written once, at creation, and no tool could touch it — so the model had no
    legal move, and said it had done it anyway. Every Dubai stop has to go, because none of them
    exist in Abu Dhabi; what must NOT go is the plan's identity, which the conversation points at.
    """
    chat = _dubai_day(db, planned)
    before = chat.call_tool("get_itinerary", {})
    itinerary_id = before["itinerary_id"]

    moved = chat.call_tool("replace_plan", {"emirates": ["Abu Dhabi"]})

    assert "error" not in moved, moved
    assert moved["itinerary_id"] == itinerary_id, "a new row would orphan the conversation"
    assert chat.conversation.itinerary_id == itinerary_id
    assert moved["emirates"] == ["Abu Dhabi"]
    assert moved["replaced"], "the reply has to be able to say what was given up"

    from app.models import Place

    stops = [name for d in moved["days"] for name in d["stops"]]
    assert stops, "the re-solved plan has no stops"
    emirates = {
        row.emirate
        for name in stops
        if (row := db.query(Place).filter(Place.name == name).first()) is not None
    }
    assert emirates == {"Abu Dhabi"}, emirates


def test_dropping_a_middle_day_asks_before_it_moves_anything(client, planned, db):
    """Shift the later days up, or leave the day free? An event on a later date decides it, so
    there is no safe default — and the question travels with the plan as it really stands."""
    chat = _dubai_day(db, planned, days=3)
    before = chat.call_tool("get_itinerary", {})

    asked = chat.call_tool("drop_day", {"day": 2})

    assert asked["applied"] is False
    assert asked["needs_confirmation"] == "day_shift_choice"
    assert asked["plan_is_unchanged"] == [s["name"] for d in before["days"] for s in d["stops"]]
    assert chat.touched_itinerary is None, "a question must not nudge the right pane"
    assert chat.call_tool("get_itinerary", {})["num_days"] == 3

    answered = chat.call_tool("drop_day", {"day": 2, "shift_later_days": True})
    assert "error" not in answered, answered
    assert answered["remaining_days"] == 2, "the trip did not get shorter"


def test_the_last_day_goes_without_a_question(client, planned, db):
    chat = _dubai_day(db, planned, days=3)
    dropped = chat.call_tool("drop_day", {"day": 3})
    assert "error" not in dropped, dropped
    assert dropped["remaining_days"] == 2


def test_where_the_family_lives_is_not_where_the_current_trip_goes(client, planned, db):
    """Recording a home emirate sets the default origin for plans built later. It must not
    silently re-point a plan that already exists — that one moves through set_origin, or not at
    all."""
    chat = _dubai_day(db, planned)
    origin_before = chat._resolve_itinerary().start_lat
    home_before = chat.user.home_base_lat

    saved = chat.call_tool("save_family_details", {"adults": 2, "home_emirate": "Sharjah"})

    assert saved["home_emirate"] == "Sharjah"
    assert chat.user.home_base_lat != home_before, "the home base was not recorded"
    assert chat._resolve_itinerary().start_lat == origin_before, "the live plan was moved too"


# --- the deterministic layer, against a real plan ----------------------------------------------


def test_the_fingerprint_changes_exactly_when_the_plan_does(client, planned, db):
    """Ground truth for "did anything change", computed rather than believed.

    Cheap on purpose: itinerary_payload is a full re-render with places and geometry attached,
    and this runs twice a turn to answer a yes/no question.
    """
    from app.services.policy import plan_fingerprint

    chat = _dubai_day(db, planned, days=2)
    itinerary = chat._resolve_itinerary()

    first = plan_fingerprint(db, itinerary)
    assert plan_fingerprint(db, itinerary) == first, "reading the plan is not changing it"

    chat.call_tool("get_itinerary", {})
    assert plan_fingerprint(db, itinerary) == first, "a read moved the fingerprint"

    chat.call_tool("set_transport", {"mode": "own_car"})
    after_transport = plan_fingerprint(db, itinerary)
    assert after_transport != first

    chat.call_tool("set_origin", {"emirate": "Abu Dhabi"})
    assert plan_fingerprint(db, itinerary) != after_transport, "moving the origin did not show"


def test_a_refused_rebuild_never_reaches_the_handler(client, planned, db):
    """The gate runs in policy.intercept, before the handler — so a rebuild that must not happen
    cannot get as far as writing a row and then being tidied up."""
    from app.models import Itinerary
    from app.services import policy

    chat = _dubai_day(db, planned)
    rows_before = db.query(Itinerary).count()

    refusal = policy.intercept(chat, "generate_itinerary", {"days": 1, "budget": 5000})

    assert refusal is not None and "replace_plan" in refusal["error"]
    assert db.query(Itinerary).count() == rows_before
    # And the warning is now on record, which is what a later consent is measured against.
    assert chat.conversation.rebuild_warned is True


def test_consent_still_takes_two_turns_after_the_move(client, planned, db):
    """`replace_existing` set in the same turn as the warning is the model granting itself
    permission. Moving the gate out of the handler must not have loosened that."""
    from app.services import policy

    chat = _dubai_day(db, planned)
    chat.warned_at_turn_start = False

    same_turn = policy.intercept(chat, "generate_itinerary", {"days": 1, "replace_existing": True})
    assert same_turn is not None and "does not grant itself" in same_turn["error"]

    # A turn later, the user having spoken since, the same call is allowed through.
    chat.warned_at_turn_start = True
    assert policy.intercept(chat, "generate_itinerary", {"days": 1, "replace_existing": True}) is None


def test_no_other_tool_is_intercepted(client, planned, db):
    """Interception is a short list of refusals, not a gate every call has to argue with."""
    from app.services import policy

    chat = _dubai_day(db, planned)
    for name in ("get_itinerary", "find_places", "edit_stop", "set_origin", "drop_day",
                 "replace_plan", "reschedule_itinerary", "record_preference"):
        assert policy.intercept(chat, name, {}) is None, name
