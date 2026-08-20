"""The rules about tool calls that used to live in the system prompt.

A rule written in English is advice: the model follows it most of the time, and the times it does
not are the bugs this file exists to close. Everything here is a plain function over the same
facts the model has, decided in code, and answerable in a test without an API key.

Four jobs, in the order a turn meets them:

  `intercept`        before a tool runs — refuse a call that must not happen, with a message that
                     names what the model should have called instead.
  `applied`          after it runs — did this call actually change anything. A fact, not the
                     model's opinion of one.
  `plan_fingerprint` a cheap cross-check on that, for the tools that touch an itinerary.
  `claim_check`      before the answer ships — does the prose claim a change the trace cannot
                     support.

Deliberately NOT here: choosing which tools to expose. Removing a tool from the request based on
state reads like the stronger move and is a trap — `Conversation.rebuild_warned` is set inside the
very handler such a rule would remove, so gating the tool on the flag makes the flag unreachable
and no restart is ever possible again. Interception gets the same guarantee (a refused call cannot
mutate anything) while leaving every tool reachable, and leaves the refusal text in place, which is
what teaches the model the right call.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.orm import Session

    from ..models import Itinerary


# --- which tools can change anything -----------------------------------------------------------

# Everything not listed is read-only. `applied` is False for those by definition: a search that
# found nothing has not failed, and a turn that only looked things up has changed nothing to
# report — telling a reviewer "nothing changed" about such a turn invents a problem.
MUTATING_TOOLS = frozenset(
    {
        "save_family_details",
        "create_event",
        "record_preference",
        "generate_itinerary",
        "replace_plan",
        "reschedule_itinerary",
        "set_origin",
        "drop_day",
        "make_day_cheaper",
        "add_prayer_breaks",
        "set_transport",
        "add_stop",
        "edit_stop",
    }
)

# The tools that write to an itinerary specifically, and so can be cross-checked against a
# fingerprint of one.
PLAN_TOOLS = frozenset(
    {
        "generate_itinerary",
        "replace_plan",
        "reschedule_itinerary",
        "set_origin",
        "drop_day",
        "make_day_cheaper",
        "add_prayer_breaks",
        "set_transport",
        "add_stop",
        "edit_stop",
    }
)


def applied(name: str, result: Any) -> bool:
    """Did this call change anything.

    Three ways a call changes nothing, and all of them have been narrated as success at least
    once: it errored, it came back asking (`applied: false`, the confirmation protocol), or it
    never writes in the first place.
    """
    if name not in MUTATING_TOOLS or not isinstance(result, dict):
        return False
    if result.get("error"):
        return False
    return result.get("applied") is not False


def plan_fingerprint(db: "Session", itinerary: "Itinerary | None") -> tuple | None:
    """A cheap value that changes exactly when the plan does.

    Deliberately not `itinerary_payload`: that is a full re-render with places and geometry
    attached, and it would run twice a turn purely to answer a yes/no question.
    """
    if itinerary is None:
        return None

    from sqlalchemy import select

    from ..models import Slot

    rows = db.execute(
        select(Slot.id, Slot.place_id, Slot.day_index, Slot.position, Slot.start_time, Slot.end_time)
        .where(Slot.itinerary_id == itinerary.id)
        .order_by(Slot.day_index, Slot.position)
    ).all()
    return (
        itinerary.id,
        itinerary.num_days,
        itinerary.start_date,
        itinerary.total_budget,
        itinerary.transport_mode,
        tuple(itinerary.emirates_json or ()),
        (itinerary.start_lat, itinerary.start_lng),
        tuple(tuple(row) for row in rows),
    )


# --- before the call ---------------------------------------------------------------------------


REBUILD_IS_NOT_AN_EDIT = (
    "This conversation already has a plan, and this tool builds a SEPARATE one — the current "
    "plan, and every edit the user approved, is abandoned along with the thread that points at "
    "it. There is a tool for whatever was actually meant. Adding, removing or swapping one stop "
    "is add_stop or edit_stop. Different dates, same stops, is reschedule_itinerary. Setting off "
    "from somewhere else — 'we live in Abu Dhabi', 'start us from Sharjah' — is set_origin, and "
    "it costs nothing. Removing a whole day is drop_day. Moving the trip's REGION, or genuinely "
    "starting over, is replace_plan: it re-solves in place, so the conversation and the event "
    "stay attached. Nothing else needs this tool: a plan is saved as it is built and edited, so "
    "there is no finalising, confirming or committing to do. If the user is choosing one specific "
    "place out of options you listed ('choose Beirut Restaurant') that is edit_stop with "
    "place='Beirut Restaurant', never this tool — this tool has no `place` argument, so a named "
    "choice made through it is silently dropped and the solver picks whatever it likes instead. "
    "If they truly want a separate plan rather than this one re-solved, tell them the current "
    "plan will be discarded, and ask. Their ANSWER is what unlocks this."
)

CONSENT_DOES_NOT_GRANT_ITSELF = (
    "replace_existing does not grant itself. The user has not been told this plan would be "
    "discarded and has not agreed to it — as of the start of this turn, nobody had raised it. "
    "Stop calling tools, tell them what would be lost, and ask. Their next message is when this "
    "works."
)


REPLACING_LOSES_THE_STOPS = (
    "Replacing this plan re-solves it from scratch. Every stop goes, and so does every edit the "
    "user has approved — only the dates, the budget and the party carry over. Tell them that in "
    "plain words, name what they are about to lose, and ask. Their answer is what unlocks this. "
    "If they only want the trip to set off from somewhere else, that is set_origin and it keeps "
    "the whole plan."
)


# Company the user has named but not counted. Deliberately only the unquantified words: the
# number itself is what the gate wants, so "10 friends" must sail through while "a bunch of
# friends" does not.
#
# ponytail: a word list, so "the whole crew" and "ten friends" both slip past — the first fires
# nothing, the second is already a count. Ask the model to classify instead if the misses matter.
_VAGUE_COMPANY = re.compile(
    r"(?<!\d )\b(?:friends?|mates|buddies|classmates|colleagues|cousins|a bunch|a group|"
    r"a crew|the guys|the girls|some people|a few people)\b",
    re.IGNORECASE,
)


HEADCOUNT_IS_NOT_OURS_TO_GUESS = (
    "The user named people coming along without saying how many, so the size of this party is "
    "not known — and it decides the vehicle, the fares and every ticket, so a guess is priced as "
    "fact and shown to them as a settled figure. The household on file is not the answer here: "
    "they have already said someone else is coming. Stop calling tools and ask how many people "
    "in total. Their answer is what unlocks this."
)


def _this_turns_message(orchestrator: Any) -> str:
    """What the user just said. Committed before the model runs, so it is already on the row."""
    from ..models import Message

    row = (
        orchestrator.db.query(Message)
        .filter(
            Message.conversation_id == orchestrator.conversation.id,
            Message.role == "user",
        )
        .order_by(Message.id.desc())
        .first()
    )
    return row.content if row else ""


def _headcount_is_not_ours_to_guess(orchestrator: Any, args: dict) -> dict | None:
    """Make the model ask for a headcount it was never told, instead of inventing one.

    Live validation, twice. "He has a bunch of friends" was first priced as party_size 11 — a
    number nobody said — and then, once the prompt told the model to ask, as party_size 4: the
    household, exactly, with the friends silently dropped. Prompt text asks for the question; only
    a refusal actually produces it, which is the difference between this field and `budget`, whose
    handler has rejected an unset figure all along.

    Deliberately narrow. Silence about the party still means the household, because that is the
    documented default and asking every time would interrogate a family whose members are on file.
    This fires only when the user has SAID others are coming and left them uncounted.
    """
    if asked_before(orchestrator, "party_size"):
        return None

    from .itinerary import family_attendees

    household = len(family_attendees(orchestrator.db, orchestrator.user.id))
    stated = args.get("party_size") or 0
    # The household count is not evidence of a decision: it is what the model falls back to when
    # it has nothing, and it is what arrived when the friends went missing.
    if stated and stated != household:
        return None
    if not _VAGUE_COMPANY.search(_this_turns_message(orchestrator)):
        return None

    # Shaped like `_unapplied`, minus the plan — there may not be one yet. `applied: false` and a
    # question rather than a proposal, so a reply cannot narrate this as a trip that got built.
    return {
        "applied": False,
        "needs_confirmation": "party_size",
        "question_for_the_user": HEADCOUNT_IS_NOT_OURS_TO_GUESS,
    }


def intercept(orchestrator: Any, name: str, args: dict) -> dict | None:
    """Refuse a call that must not happen, or None to let it through.

    Runs before the handler, so a refusal cannot have written anything on its way to being
    refused. The returned dict is what the model sees as the tool's result.
    """
    if name == "generate_itinerary":
        return (
            _rebuild_is_not_an_edit(orchestrator, args)
            or _headcount_is_not_ours_to_guess(orchestrator, args)
        )
    if name == "replace_plan":
        return _replacing_needs_the_users_word(orchestrator, args)
    if name == "drop_day":
        _shift_is_not_ours_to_choose(orchestrator, args)
    return None


def asked_before(orchestrator: Any, kind: str) -> bool:
    """Did the assistant already put this question to the user, in an earlier turn?

    Read from the persisted trace rather than taken on the model's word, for the same reason
    `replace_existing` does not grant itself: setting an argument is free, and a model with a
    blank to fill will fill it. The last assistant row is necessarily from a previous turn,
    because this turn's reply has not been recorded yet.
    """
    from ..models import Message

    row = (
        orchestrator.db.query(Message)
        .filter(
            Message.conversation_id == orchestrator.conversation.id,
            Message.role == "assistant",
        )
        .order_by(Message.id.desc())
        .first()
    )
    if row is None:
        return False
    for call in (row.tool_calls_json or {}).get("calls") or []:
        if (call.get("result") or {}).get("needs_confirmation") == kind:
            return True
    return False


def _replacing_needs_the_users_word(orchestrator: Any, args: dict) -> dict | None:
    """Ask once before throwing away every stop in the plan.

    Live validation is what put this here. "Change the location of the plan to Abu Dhabi" went
    straight through: the region moved, which is what the user asked for, and every edit they had
    made went with it without anyone mentioning that it would. The tool description said to ask
    first, and a description is advice.
    """
    current = orchestrator._resolve_itinerary()
    if current is None:
        return {
            "error": (
                "There is no plan in this conversation to replace. Build one with "
                "generate_itinerary."
            )
        }
    if asked_before(orchestrator, "plan_replacement"):
        return None

    emirates = [e for e in (args.get("emirates") or []) if e]
    return orchestrator._unapplied(
        current,
        "plan_replacement",
        ValueError(REPLACING_LOSES_THE_STOPS),
        proposed_emirates=emirates or None,
    )


def _shift_is_not_ours_to_choose(orchestrator: Any, args: dict) -> None:
    """Make `drop_day` put its question, rather than answer it on the user's behalf.

    Also from live validation. Told "Drop day 2", the model called
    `drop_day(day=2, shift_later_days=False)` — inventing an answer to the one question the tool
    exists to ask, and leaving a silent gap in the middle of the trip. The handler only asks when
    the argument is absent, so the argument is removed unless the question has actually been put.

    Mutates `args` rather than returning a refusal, deliberately: what should happen next is the
    handler raising DayShiftChoiceRequired, which is already shaped so a question cannot be read
    as an answer.
    """
    if args.get("shift_later_days") is None:
        return
    if asked_before(orchestrator, "day_shift_choice"):
        return
    args["shift_later_days"] = None


def _rebuild_is_not_an_edit(orchestrator: Any, args: dict) -> dict | None:
    """Protect the conversation's own plan from being replaced by a brand new one.

    The reported bug: asked to add one adventure, the model called generate_itinerary, a new
    itinerary row was created, the conversation was re-pointed at it, and the approved plan was
    orphaned — a stop the user never touched vanished and one they had swapped out came back.
    Prompt text was the only thing standing in the way, and the prompt's own carve-out ("or agreed
    to start over") is exactly what a reply like "sure, I can sacrifice some budget" reads as.

    Only the conversation's OWN plan is protected: a new thread still plans freely even though the
    user has older plans elsewhere.
    """
    current = orchestrator._resolve_itinerary()
    if current is None or current.id != orchestrator.conversation.itinerary_id:
        return None

    # `replace_existing` alone is not permission. Setting it is free, and the model set it on its
    # first attempt — "let us finalize it" became a rebuild that threw away every edit. Permission
    # arrives a TURN after the warning, because that is how long it takes the user to answer, so
    # the flag counts only once the warning has been given and the user has spoken since.
    warned_before_this_turn = orchestrator.warned_at_turn_start
    orchestrator.conversation.rebuild_warned = True
    orchestrator.rebuild_refused = True

    value = args.get("replace_existing")
    if not bool(value if value is not None else False):
        return {"error": REBUILD_IS_NOT_AN_EDIT}
    if not warned_before_this_turn:
        return {"error": CONSENT_DOES_NOT_GRANT_ITSELF}
    return None


# --- before the answer ships -------------------------------------------------------------------


# Past participles, on purpose. "I can add a museum" is an offer and "shall I remove it?" is a
# question; neither claims anything happened, and both use the base form — so leaving the base
# forms out is what keeps an offer from being read as a lie.
#
# `set`, `put` and `cut` are their own past participles, so they carry both readings. They are in
# the list because the reported bug was "the location ... has now been SET to Abu Dhabi", and the
# `_HYPOTHETICAL` filter below is what keeps "I can set the transport" out of it.
CHANGE_VERBS = frozenset(
    {
        "added", "created", "changed", "updated", "moved", "removed", "replaced", "swapped",
        "rescheduled", "dropped", "applied", "adjusted", "shortened", "saved", "recorded",
        "rebuilt", "relocated", "shifted", "made", "set", "put", "cut", "took", "switched",
    }
)

# A sentence that says a change did NOT happen contains the same verbs as one that says it did —
# it is the honest answer, and flagging it would train the fix out of existence.
_NEGATED = re.compile(
    r"\b(not|never|unable|cannot|can't|couldn't|didn't|wasn't|weren't|hasn't|haven't|failed|"
    r"unchanged|nothing)\b|\bno (change|stops?|plan)\b",
    re.IGNORECASE,
)

# Offers and intentions, which are about the future rather than the past.
_HYPOTHETICAL = re.compile(
    r"\b(can|could|shall|should|would|will|may|might|want|like me to|going to|if you)\b",
    re.IGNORECASE,
)

_SENTENCE = re.compile(r"[^.!?\n]+[.!?\n]?")


def claim_check(draft: str, trace: list[dict]) -> list[str]:
    """Sentences that claim a change the trace cannot support.

    A first filter, not the guarantee. It is a verb list, so a paraphrase gets past it — which
    costs nothing, because the LLM reviewer runs on every mutating turn regardless of what this
    returns. What it buys is the other direction: a turn that changed nothing and claims nothing
    needs no reviewer at all, and a turn that changed nothing and claims something is caught here
    without one.

    `trace` is this turn's calls, each `{"name": ..., "applied": bool}`.
    """
    if any(entry.get("applied") for entry in trace):
        return []  # something really did change; whether the prose describes it right is T3's job

    flagged = []
    for raw in _SENTENCE.findall(draft or ""):
        sentence = raw.strip()
        if not sentence or sentence.endswith("?"):
            continue
        if _NEGATED.search(sentence) or _HYPOTHETICAL.search(sentence):
            continue
        words = {word.lower() for word in re.findall(r"[a-zA-Z']+", sentence)}
        if words & CHANGE_VERBS:
            flagged.append(sentence)
    return flagged


# --- whether a turn should be made to call something -------------------------------------------


_SMALL_TALK = frozenset(
    {
        "hi", "hello", "hey", "yo", "salam", "assalamu alaikum",
        "thanks", "thank you", "thanks!", "cheers", "much appreciated",
        "bye", "goodbye", "good night", "see you",
    }
)


def is_small_talk(message: str) -> bool:
    """True only for a greeting or a thank-you, where forcing a tool call would be absurd.

    Used to decide `tool_choice` on the FIRST request of a turn, so the bar is deliberately high:
    "ok", "yes" and "sure" are not small talk — they are usually the answer to a confirmation
    question, and that answer is precisely when a tool must run.
    """
    text = " ".join((message or "").lower().split()).strip(" .!,")
    return text in _SMALL_TALK
