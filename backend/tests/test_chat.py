"""Chat orchestration: SSE framing, tool isolation, thread persistence and the LLM-down fallback.

These run with no OPENAI_API_KEY, which is the fallback path — and also exactly the state of
acceptance criterion 6. Tool behaviour is tested by calling the tools directly, so the assertions
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
    assert names == {
        "save_family_details",
        "create_event",
        "get_upcoming_events",
        "generate_itinerary",
        "record_preference",
    }


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


def test_chat_streams_sse_frames_and_persists_the_thread(client, make_user):
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


def test_with_the_llm_down_upcoming_events_still_answer(client, make_user):
    headers, _ = make_user("degraded@rihla.app")
    client.post(
        "/events",
        headers=headers,
        json={"title": "Aisha's birthday", "event_type": "birthday", "date": FUTURE},
    )

    response = client.post("/chat", headers=headers, json={"message": "What events are upcoming?"})
    reply = "".join(e["data"] for e in frames(response) if e["type"] == "token")

    assert "Aisha's birthday" in reply
    assert "Want me to plan an itinerary for Aisha's birthday?" in reply
    assert frames(response)[-1]["data"]["degraded"] is True


def test_with_the_llm_down_planning_names_the_missing_intake_fields(client, make_user):
    headers, _ = make_user("degraded2@rihla.app")
    response = client.post("/chat", headers=headers, json={"message": "plan a trip for me"})
    reply = "".join(e["data"] for e in frames(response) if e["type"] == "token")
    assert "still need" in reply
    assert "adults" in reply


def test_messages_accumulate_in_the_same_thread(client, make_user):
    headers, _ = make_user("thread@rihla.app")
    first = client.post("/chat", headers=headers, json={"message": "hello"})
    conversation_id = frames(first)[0]["data"]["conversation_id"]

    client.post(
        "/chat", headers=headers, json={"message": "and again", "conversation_id": conversation_id}
    )
    history = client.get(f"/conversations/{conversation_id}/messages", headers=headers).json()
    assert [m["content"] for m in history if m["role"] == "user"] == ["hello", "and again"]


# --- threads and unread ------------------------------------------------------------------------


def test_a_new_message_marks_the_thread_unread_until_it_is_seen(client, make_user):
    headers, _ = make_user("unread@rihla.app")
    conversation_id = frames(client.post("/chat", headers=headers, json={"message": "hi"}))[0][
        "data"
    ]["conversation_id"]

    listed = client.get("/conversations", headers=headers).json()
    assert listed[0]["unread"] is True

    assert client.post(f"/conversations/{conversation_id}/seen", headers=headers).json()["unread"] is False
    assert client.get("/conversations", headers=headers).json()[0]["unread"] is False


def test_threads_are_private_to_their_owner(client, make_user):
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
