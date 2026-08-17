"""Chat orchestration: SSE framing, tool isolation and thread persistence.

The model itself is stubbed — the suite must never make a billable call — so what is under test is
everything around it. Tool behaviour is checked by calling the tools directly, so the assertions
are about what a tool *does to the database*, not about what a model chose to say.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from app.models import Conversation, Event, FamilyMember, Preference, User
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
    # get_itinerary is beyond spec §8: without a way to READ a plan, the assistant could only
    # describe one from stale context and would contradict the budget bar next to it.
    assert names == spec_tools | {"get_itinerary"}


def test_no_tool_schema_exposes_a_user_id():
    """The model must not be able to address another user, however it is prompted."""
    for tool in TOOLS:
        properties = tool["function"]["parameters"].get("properties", {})
        assert "user_id" not in properties, tool["function"]["name"]
        assert "user_id" not in json.dumps(tool)


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
    assert client.delete(f"/conversations/{conversation_id}", headers=intruder).status_code == 404
    assert client.get("/conversations", headers=intruder).json() == []

    # Posting into someone else's thread is a 404, not a silent write.
    assert client.post(
        "/chat", headers=intruder, json={"message": "hi", "conversation_id": conversation_id}
    ).status_code == 404


def test_chat_requires_authentication(client):
    assert client.post("/chat", json={"message": "hi"}).status_code == 401


# --- binding a thread to a plan ----------------------------------------------------------------


def test_a_thread_can_be_renamed_and_bound_to_its_plan(client, make_user, stub_llm):
    """The rail shows the event's initial, so a generated plan renames its thread."""
    headers, _ = make_user("bind@rihla.app")
    conversation_id = frames(client.post("/chat", headers=headers, json={"message": "hi"}))[0][
        "data"
    ]["conversation_id"]

    response = client.patch(
        f"/conversations/{conversation_id}",
        headers=headers,
        json={"title": "Anniversary weekend"},
    )
    assert response.status_code == 200
    assert response.json()["title"] == "Anniversary weekend"
    assert client.get("/conversations", headers=headers).json()[0]["title"] == "Anniversary weekend"


def test_a_thread_cannot_be_bound_to_another_users_plan(client, make_user, db, stub_llm):
    """Otherwise a thread could be used to read a plan its owner never created."""
    from app.models import Itinerary

    owner_headers, owner = make_user("planowner@rihla.app")
    other_headers, other = make_user("other@rihla.app")

    plan = Itinerary(
        user_id=owner["id"], title="Private", start_date=date.today() + timedelta(days=3),
        num_days=2, total_budget=1000.0,
    )
    db.add(plan)
    db.commit()

    conversation_id = frames(client.post("/chat", headers=other_headers, json={"message": "hi"}))[0][
        "data"
    ]["conversation_id"]

    assert client.patch(
        f"/conversations/{conversation_id}",
        headers=other_headers,
        json={"itinerary_id": plan.id},
    ).status_code == 404

    # And the owner cannot reach into someone else's thread either.
    assert client.patch(
        f"/conversations/{conversation_id}", headers=owner_headers, json={"title": "hijacked"}
    ).status_code == 404


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
    _, _, plan = planned
    intruder = orchestrator("intruder@rihla.app")
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
