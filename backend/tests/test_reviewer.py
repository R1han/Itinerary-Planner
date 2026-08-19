"""The reviewer, with the model stubbed.

What is worth testing here is not whether an LLM judges well — it is everything around the call:
that a refusal, a timeout or a malformed answer lets the draft through rather than taking the turn
down, and that what we hand the model is the raw trace rather than the lossy summary the activity
rows use.
"""

from __future__ import annotations

import json

import pytest

from app.services.reviewer import (
    NEEDS_TOOLS,
    OK,
    RESULT_CHARS,
    REVIEW_SCHEMA,
    REWRITE,
    Verdict,
    render_trace,
    review,
)


class _Message:
    def __init__(self, content=None, refusal=None):
        self.content = content
        self.refusal = refusal


class _FakeClient:
    """Stands in for the OpenAI client, recording what it was asked."""

    def __init__(self, message=None, raises=None):
        self._message = message or _Message(content=json.dumps(
            {"verdict": OK, "unsupported_claims": [], "missing_tools": [], "guidance": ""}
        ))
        self._raises = raises
        self.calls = []
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        return type("R", (), {"choices": [type("C", (), {"message": self._message})()]})()


TRACE_NOTHING_APPLIED = [
    {
        "name": "edit_stop",
        "args": {"stop": "Yas Beach", "action": "remove"},
        "applied": False,
        "result": {"error": "'Yas Marina Waterfront Cafe' matches more than one stop"},
    }
]


# --- failing open ------------------------------------------------------------------------------


def test_a_refusal_is_not_approval_but_does_not_fail_the_turn(caplog):
    """A refusal has no content. Left unlogged it reads exactly like a clean pass."""
    client = _FakeClient(message=_Message(refusal="I cannot help with that"))
    verdict = review("move it", TRACE_NOTHING_APPLIED, "Done.", client=client)

    assert verdict.is_ok
    assert "refused" in caplog.text.lower()


def test_an_exception_lets_the_draft_through():
    client = _FakeClient(raises=RuntimeError("upstream is down"))
    assert review("move it", TRACE_NOTHING_APPLIED, "Done.", client=client).is_ok


def test_malformed_json_lets_the_draft_through():
    client = _FakeClient(message=_Message(content="not json at all"))
    assert review("move it", TRACE_NOTHING_APPLIED, "Done.", client=client).is_ok


def test_an_unknown_verdict_is_treated_as_ok():
    client = _FakeClient(message=_Message(content=json.dumps({"verdict": "burn it down"})))
    assert review("move it", TRACE_NOTHING_APPLIED, "Done.", client=client).is_ok


def test_an_empty_draft_is_not_worth_a_call():
    client = _FakeClient()
    assert review("hi", [], "   ", client=client).is_ok
    assert client.calls == [], "an empty draft has nothing to check"


# --- what the model is actually shown ----------------------------------------------------------


def test_the_reviewer_sees_the_raw_error_not_the_activity_row_summary():
    """summarise_tool_result turns every error into "no change made" — deliberately, because
    those strings are addressed to the model in the chat. A reviewer given that cannot say WHICH
    claim is unsupported, only that something is off."""
    from app.services.orchestrator import summarise_tool_result

    assert summarise_tool_result("edit_stop", {"error": "anything at all"}) == "no change made"

    rendered = render_trace(TRACE_NOTHING_APPLIED)
    assert "matches more than one stop" in rendered
    assert "no change made" not in rendered


def test_the_applied_flag_travels_with_every_call():
    rendered = render_trace(TRACE_NOTHING_APPLIED)
    assert "applied: False" in rendered


def test_a_long_result_is_truncated_not_dropped():
    trace = [{"name": "find_places", "args": {}, "applied": False, "result": {"x": "y" * 5000}}]
    rendered = render_trace(trace)
    assert "truncated" in rendered
    assert len(rendered) < RESULT_CHARS * 3
    assert "find_places" in rendered


def test_a_turn_with_no_tools_says_so_rather_than_going_blank():
    assert "no tools" in render_trace([])


def test_the_history_is_not_sent():
    """The question is whether THIS reply is supported by THIS turn's evidence. The history is
    where an earlier turn's confident wrong answer lives."""
    client = _FakeClient()
    review("move it to Abu Dhabi", TRACE_NOTHING_APPLIED, "Done.", client=client)

    sent = json.dumps(client.calls[0]["messages"])
    assert "move it to Abu Dhabi" in sent
    assert "matches more than one stop" in sent
    assert client.calls[0]["temperature"] == 0


def test_the_call_asks_for_strict_structured_output():
    client = _FakeClient()
    review("x", [], "a draft", client=client)

    fmt = client.calls[0]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"] is REVIEW_SCHEMA


def test_the_schema_obeys_strict_modes_rules():
    """Every property required, every object closed — one violation fails the whole request."""
    assert REVIEW_SCHEMA["additionalProperties"] is False
    assert set(REVIEW_SCHEMA["required"]) == set(REVIEW_SCHEMA["properties"])


# --- verdicts ----------------------------------------------------------------------------------


def test_a_rewrite_carries_the_sentence_to_fix():
    client = _FakeClient(message=_Message(content=json.dumps({
        "verdict": REWRITE,
        "unsupported_claims": ["The location has now been set to Abu Dhabi."],
        "missing_tools": [],
        "guidance": "Nothing was applied; say the region did not change.",
    })))
    verdict = review("move it", TRACE_NOTHING_APPLIED, "The location has now been set...", client=client)

    assert not verdict.is_ok
    assert not verdict.wants_tools
    assert verdict.unsupported_claims == ["The location has now been set to Abu Dhabi."]
    assert "did not change" in verdict.guidance


def test_needs_tools_names_what_was_missed():
    client = _FakeClient(message=_Message(content=json.dumps({
        "verdict": NEEDS_TOOLS,
        "unsupported_claims": [],
        "missing_tools": ["get_itinerary"],
        "guidance": "Read the plan before quoting its total.",
    })))
    verdict = review("what's my total?", [], "It is 1,940 AED.", client=client)

    assert verdict.wants_tools
    assert verdict.missing_tools == ["get_itinerary"]


@pytest.mark.parametrize("verdict", [OK, NEEDS_TOOLS, REWRITE])
def test_every_verdict_the_schema_allows_round_trips(verdict):
    client = _FakeClient(message=_Message(content=json.dumps(
        {"verdict": verdict, "unsupported_claims": [], "missing_tools": [], "guidance": ""}
    )))
    assert review("x", [], "a draft", client=client).verdict == verdict


def test_the_default_verdict_is_ok():
    assert Verdict().is_ok
    assert Verdict().unsupported_claims == []


def test_a_whole_plan_result_reaches_the_reviewer_intact():
    """Live validation caught this: at 400 characters the trace stopped inside day two of a
    three-day plan, the reviewer reported day three as unsupported — correctly, from what it
    could see — and the assistant deleted a real day from its reply. A limit set too low does not
    make the reviewer blind, it makes it confidently wrong."""
    plan = {
        "itinerary_id": 1,
        "start_date": "2026-12-10",
        "days": [
            {
                "day": day,
                "date": f"2026-12-{9 + day}",
                "theme": "Adventure & Wildlife",
                "subtotal": 1688.74,
                "stops": ["Abu Dhabi Falcon Hospital", "Emirates Park Zoo", "Al Khatim Desert Safari"],
            }
            for day in (1, 2, 3)
        ],
        "total": 4377.97, "cap": 6000.0, "remaining": 1622.03,
        "transport_mode": "taxi", "vehicle": "one vehicle", "party_size": 3,
        "emirates": ["Abu Dhabi"], "travel": {"total": 288.7},
    }
    rendered = render_trace([
        {"name": "generate_itinerary", "args": {"days": 3}, "applied": True, "result": plan}
    ])

    assert "truncated" not in rendered, "a plan must not be cut off before the reviewer sees it"
    for day in (1, 2, 3):
        assert f'"day": {day}' in rendered, f"day {day} never reached the reviewer"


def test_a_five_day_plan_also_fits():
    plan = {
        "days": [
            {"day": d, "date": f"2026-12-{9 + d}", "theme": "Theme Park & Adventure",
             "subtotal": 1873.75,
             "stops": ["SeaWorld Abu Dhabi", "Yas Kartzone", "Yas Marina Waterfront Cafe"]}
            for d in range(1, 6)
        ],
        "total": 9000.0, "cap": 9000.0, "emirates": ["Abu Dhabi"],
    }
    rendered = render_trace([
        {"name": "replace_plan", "args": {}, "applied": True, "result": plan}
    ])
    assert "truncated" not in rendered
