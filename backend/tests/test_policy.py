"""The deterministic tool-call rules.

Every function here is pure, so these need no database, no fixtures and no API key — which is the
point of moving the rules out of the system prompt. A rule you can only check by running a
conversation is a rule nobody checks.
"""

from __future__ import annotations

import pytest

from app.services.policy import (
    CHANGE_VERBS,
    MUTATING_TOOLS,
    PLAN_TOOLS,
    applied,
    claim_check,
    is_small_talk,
)


# --- applied: did this call change anything ----------------------------------------------------


@pytest.mark.parametrize(
    ("name", "result", "expected"),
    [
        ("edit_stop", {"itinerary_id": 1}, True),
        ("replace_plan", {"itinerary_id": 1, "replaced": ["a"]}, True),
        # An error is not a change, however plausible the rest of the payload looks.
        ("edit_stop", {"error": "no such stop"}, False),
        ("replace_plan", {"error": "intake_incomplete", "missing_fields": ["adults"]}, False),
        # The confirmation protocol: a question that changed nothing, and was narrated as a swap.
        ("edit_stop", {"applied": False, "needs_confirmation": "day_reorder"}, False),
        ("drop_day", {"applied": False, "needs_confirmation": "day_shift_choice"}, False),
        # Read-only tools never "change" anything, so a turn that only searched is not a turn
        # that failed to change something.
        ("find_places", {"places": [1, 2, 3]}, False),
        ("get_itinerary", {"days": []}, False),
        ("find_places", {"places": []}, False),
    ],
)
def test_applied_is_a_fact_not_a_reading_of_one(name, result, expected):
    assert applied(name, result) is expected


def test_every_plan_tool_is_a_mutating_tool():
    """PLAN_TOOLS is the subset that writes to an itinerary; it cannot contain a read-only tool."""
    assert PLAN_TOOLS <= MUTATING_TOOLS


def test_reading_the_plan_is_not_writing_to_it():
    assert "get_itinerary" not in MUTATING_TOOLS
    assert "find_places" not in MUTATING_TOOLS


# --- claim_check: does the prose claim something the trace cannot support ----------------------


NOTHING_HAPPENED = [{"name": "edit_stop", "applied": False}]


def test_the_reported_lie_is_caught_without_an_llm():
    """The exact shape of the reported bug: nothing was applied, the reply says it was."""
    draft = (
        "The location of your brother's birthday plan has now been set to Abu Dhabi. "
        "However, I noticed that some stops are more suited to Dubai."
    )
    flagged = claim_check(draft, NOTHING_HAPPENED)
    assert flagged and "now been set to Abu Dhabi" in flagged[0]


def test_the_honest_answer_is_not_flagged():
    """The fix reads as the bug to a naive verb match — it uses the same words.

    Flagging this would train the correct answer out of existence, so it matters more than
    catching the lie does.
    """
    draft = (
        "The region was not changed — that request did not go through, so the plan is unchanged. "
        "Nothing was removed."
    )
    assert claim_check(draft, NOTHING_HAPPENED) == []


def test_an_offer_is_not_a_claim():
    draft = "I can move the plan to Abu Dhabi, but every stop would be replaced. Shall I?"
    assert claim_check(draft, NOTHING_HAPPENED) == []


def test_a_question_is_not_a_claim():
    assert claim_check("Would you like the later days shifted earlier?", NOTHING_HAPPENED) == []


def test_a_turn_that_really_changed_something_is_left_to_the_reviewer():
    """Whether the prose describes a real change *correctly* is a judgement, not a verb match."""
    draft = "I removed Yas Beach and the day now finishes at 18:00."
    assert claim_check(draft, [{"name": "edit_stop", "applied": True}]) == []


def test_only_the_offending_sentence_comes_back():
    draft = (
        "Here is what I found. I replaced the museum with a park. "
        "Would you like anything else?"
    )
    flagged = claim_check(draft, NOTHING_HAPPENED)
    assert len(flagged) == 1
    assert "replaced the museum" in flagged[0]


def test_an_empty_draft_claims_nothing():
    assert claim_check("", NOTHING_HAPPENED) == []
    assert claim_check("Here are ten places in Abu Dhabi.", NOTHING_HAPPENED) == []


def test_the_base_form_of_a_verb_is_not_a_claim():
    """"add" is an offer and "added" is a claim. Carrying the base forms would flag every
    sentence that describes what a tool *could* do, which is most of a good reply."""
    assert not CHANGE_VERBS & {
        "add", "create", "change", "update", "move", "remove", "replace", "swap",
        "reschedule", "drop", "apply", "adjust", "shorten", "save", "record", "rebuild",
    }


def test_a_verb_that_is_its_own_past_tense_still_needs_a_modal_to_be_an_offer():
    """`set`, `put` and `cut` read both ways, so they lean on the hypothetical filter alone."""
    assert claim_check("I can set the transport to taxi.", NOTHING_HAPPENED) == []
    assert claim_check("Shall I cut day 3?", NOTHING_HAPPENED) == []
    assert claim_check("The transport has been set to taxi.", NOTHING_HAPPENED) != []


# --- is_small_talk: whether a turn may be let off calling a tool -------------------------------


@pytest.mark.parametrize("message", ["hi", "Hello!", "thanks", "Thank you.", "bye"])
def test_greetings_need_no_tool(message):
    assert is_small_talk(message) is True


@pytest.mark.parametrize(
    "message",
    [
        # The answer to a confirmation question is exactly when a tool MUST run — reading these
        # as pleasantries is how "Ok" got a reply that changed nothing and said otherwise.
        "ok",
        "Yes",
        "sure",
        "go ahead",
        "Change the location of the plan to Abu Dhabi.",
        "what museums are in Sharjah?",
    ],
)
def test_an_answer_is_not_small_talk(message):
    assert is_small_talk(message) is False
