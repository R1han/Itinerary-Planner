"""Chat orchestration over OpenAI function calling (spec §8).

The orchestrator is constructed with the authenticated user. It loads that user's family,
preferences and preference memory into its system context, and every tool implementation reads and
writes only that user's rows. **No tool schema exposes a user_id parameter**, so the model cannot
address another user even if a prompt tries to make it.

The LLM never builds an itinerary. `generate_itinerary` calls the deterministic planner; the model
only decides when to call it and how to phrase the result.

The assistant is a hard dependency: OPENAI_API_KEY is required, and a failed call surfaces as an
error rather than degrading to a scripted responder.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import date, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import (
    Conversation,
    Event,
    FamilyMember,
    Itinerary,
    Message,
    Preference,
    Slot,
    User,
    utcnow,
)
from . import itinerary as itinerary_service
from .memory import MemoryService
from .websearch import find_live_events
from .tracing import traced, wrap_openai

log = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 4
HISTORY_LIMIT = 20

EVENT_TYPES = ["birthday", "anniversary", "family_visit", "graduation", "eid", "holiday", "other"]

# Note the absence of any user_id parameter — deliberate, and load-bearing.
#
# Written with the ordinary required/optional split and rewritten for strict mode below, because
# strict mode's shape (everything required, optional spelled as nullable) is unreadable to write
# by hand and easy to get subtly wrong across eleven tools.
_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "save_family_details",
            "description": "Record who is in the family and what they like or dislike.",
            "parameters": {
                "type": "object",
                "properties": {
                    "adults": {"type": "integer", "description": "1 to 12."},
                    "children_ages": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "One entry per child, their age in years, 0 to 17.",
                    },
                    "likes": {"type": "array", "items": {"type": "string"}},
                    "dislikes": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["adults"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_event",
            "description": "Add an upcoming event to the user's calendar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "event_type": {"type": "string", "enum": EVENT_TYPES},
                    "date": {"type": "string", "description": "ISO date, YYYY-MM-DD"},
                    "notes": {"type": "string"},
                },
                "required": ["title", "event_type", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_upcoming_events",
            "description": "List the user's upcoming events, and whether each is already planned.",
            "parameters": {
                "type": "object",
                "properties": {
                    "horizon_days": {
                        "type": "integer",
                        "description": "1 to 730. Null means the default of 60.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_live_events",
            "description": (
                "Search the web for dated one-off happenings — a concert, a festival weekend — "
                "that the seeded catalog cannot know about. READ-ONLY: it saves nothing. List "
                "what it returns and let the user pick; call create_event only for the ones they "
                "actually choose. Use only when the user asks what is on around a date. "
                "Everything else (places, attractions, restaurants) comes from seeded data, "
                "never from here."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for, e.g. 'concerts in Dubai in March'",
                    },
                    "horizon_days": {
                        "type": "integer",
                        "description": "1 to 365. Null means the default of 90.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_itinerary",
            "description": (
                "Build an itinerary for an event. The server rejects this if the intake "
                "checklist is incomplete; ask for whatever is missing and try again."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "integer"},
                    "start_date": {"type": "string", "description": "ISO date, YYYY-MM-DD"},
                    "days": {"type": "integer", "description": "1 to 5."},
                    "budget": {"type": "number", "description": "In AED, greater than zero."},
                    "prayer_breaks": {"type": "boolean"},
                    "focus": {
                        "type": "string",
                        "enum": list(itinerary_service.PLAN_FOCUS),
                        "description": (
                            "'full_day' plans attractions and every meal. Use 'dinner_only' when "
                            "the user asked for a meal and nothing else — it plans one evening "
                            "stop, not a day out. Defaults to 'full_day'."
                        ),
                    },
                    "adults_only": {
                        "type": "boolean",
                        "description": (
                            "True when the children are not coming — a couple's anniversary, a "
                            "night out. Changes what the planner optimises for and who is "
                            "charged, so set it whenever the user or the event notes say so."
                        ),
                    },
                },
                "required": ["days", "budget"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_itinerary",
            "description": (
                "Read the CURRENT state of a plan — its days, stops, times and budget. Call this "
                "before describing or summarising a plan. Figures change whenever the user edits "
                "a slot, asks for a cheaper day or adds prayer breaks, so anything remembered "
                "from earlier in the conversation may already be stale."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "itinerary_id": {
                        "type": "integer",
                        "description": "Omit for the plan currently open beside the chat.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_day_cheaper",
            "description": (
                "Re-solve one day of an existing plan against a smaller budget, swapping in "
                "cheaper places. The planner may find nothing better; the result says how much "
                "it actually saved, which may be nothing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "day": {"type": "integer", "description": "1-based day number."},
                    "itinerary_id": {"type": "integer"},
                },
                "required": ["day"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_prayer_breaks",
            "description": "Insert prayer breaks into every day of an existing plan and reflow it.",
            "parameters": {
                "type": "object",
                "properties": {"itinerary_id": {"type": "integer"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_stop",
            "description": (
                "Add one new stop to a day of an existing plan, placed wherever it costs the day "
                "least. Use it to fill a gap, or after removing something. The server refuses "
                "rather than overlapping two stops if the day has no room."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "day": {"type": "integer", "description": "1-based day number."},
                    "category": {
                        "type": "string",
                        "enum": ["adventure", "aquarium", "beach", "casual_dining", "cruise", "fine_dining", "mall", "museum", "park", "show", "theme_park", "waterpark"],
                        "description": "What kind of place. Omit to take the best of any kind.",
                    },
                    "itinerary_id": {"type": "integer"},
                },
                "required": ["day"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_transport",
            "description": (
                "Record how the family is getting around and re-price the plan. Call this "
                "whenever they mention having their own car, or going back to taxis — the travel "
                "figures on screen are wrong until you do."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["taxi", "own_car"]},
                    "itinerary_id": {"type": "integer"},
                },
                "required": ["mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_stop",
            "description": (
                "Remove one stop from a plan, or move it to a different start time. The rest of "
                "the day reflows around the edit. slot_id comes from get_itinerary — call that "
                "first, and never guess an id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "slot_id": {"type": "integer"},
                    "action": {"type": "string", "enum": ["remove", "adjust", "replace"]},
                    "category": {
                        "type": "string",
                        "enum": ["adventure", "aquarium", "beach", "casual_dining", "cruise", "fine_dining", "mall", "museum", "park", "show", "theme_park", "waterpark"],
                        "description": (
                            "For action='replace': the kind of place to swap in. The server "
                            "picks the best one that fits the slot's window and budget."
                        ),
                    },
                    "start_time": {
                        "type": "string",
                        "description": "24h HH:MM. Required when action is adjust.",
                    },
                    "itinerary_id": {"type": "integer"},
                },
                "required": ["slot_id", "action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_preference",
            "description": "Record a like or dislike the user mentions in passing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["like", "dislike"]},
                    "subject": {"type": "string"},
                    "category": {"type": "string"},
                },
                "required": ["kind", "subject"],
            },
        },
    },
]


# Strict mode accepts a subset of JSON Schema. These keywords are dropped rather than risked: an
# unsupported one is rejected by the API, and a rejected schema fails the whole chat request. Every
# bound they expressed is re-checked in the handler anyway, and restated in the description so the
# model still sees it.
_UNSUPPORTED_KEYWORDS = frozenset({"default", "minimum", "maximum", "minItems", "maxItems"})


def _nullable(spec: dict) -> dict:
    kind = spec.get("type")
    types = kind if isinstance(kind, list) else [kind]
    return spec if "null" in types else {**spec, "type": [*types, "null"]}


def _strict_parameters(parameters: dict) -> dict:
    """Rewrite one tool's parameters for strict mode.

    Strict mode has no optional properties: everything is listed in `required` and objects are
    closed. An argument that used to be left out is declared nullable instead, and therefore
    arrives as None rather than missing — which is what `_arg` exists to absorb.
    """
    properties = parameters.get("properties", {})
    optional = set(properties) - set(parameters.get("required", []))

    rewritten = {}
    for name, spec in properties.items():
        cleaned = {k: v for k, v in spec.items() if k not in _UNSUPPORTED_KEYWORDS}
        if isinstance(cleaned.get("items"), dict):
            cleaned["items"] = {
                k: v for k, v in cleaned["items"].items() if k not in _UNSUPPORTED_KEYWORDS
            }
        rewritten[name] = _nullable(cleaned) if name in optional else cleaned

    return {
        "type": "object",
        "properties": rewritten,
        "required": list(rewritten),
        "additionalProperties": False,
    }


TOOLS = [
    {
        "type": "function",
        "function": {
            **tool["function"],
            "strict": True,
            "parameters": _strict_parameters(tool["function"].get("parameters", {})),
        },
    }
    for tool in _TOOL_DEFINITIONS
]


def _arg(args: dict, name: str, default):
    """`args.get` that treats an explicit null the same as an absent key.

    Strict schemas require every property, so an optional argument arrives as None. Plain
    `args.get(name, default)` returns that None — and int(None) is a crash, not a tool error.
    """
    value = args.get(name)
    return default if value is None else value


def sse(event_type: str, data) -> str:
    """One Server-Sent Event frame."""
    return f"data: {json.dumps({'type': event_type, 'data': data}, default=str)}\n\n"


def _count(items, singular: str, plural: str | None = None) -> str:
    n = len(items)
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


def _money(amount, currency: str = "AED") -> str:
    try:
        return f"{currency} {float(amount):,.0f}"
    except (TypeError, ValueError):
        return f"{currency} —"


def describe_tool_call(name: str, args: dict) -> tuple[str, str | None]:
    """A human label and a short summary of the inputs, for the activity row in the chat.

    Written here rather than in the client: the server already knows what these arguments mean,
    and phrasing them in one place keeps the wording testable and the client free of a second
    copy of the domain vocabulary.
    """
    if name == "save_family_details":
        bits: list[str] = []
        adults = args.get("adults")
        if adults:
            bits.append(_count([None] * int(adults), "adult"))
        ages = [a for a in (args.get("children_ages") or []) if a is not None]
        if ages:
            bits.append(f"{_count(ages, 'child', 'children')} aged {', '.join(str(a) for a in ages)}")
        if args.get("likes"):
            bits.append("likes " + ", ".join(str(x) for x in args["likes"][:3]))
        if args.get("dislikes"):
            bits.append("dislikes " + ", ".join(str(x) for x in args["dislikes"][:3]))
        return "Saving your family details", " · ".join(bits) or None

    if name == "create_event":
        detail = " · ".join(str(x) for x in (args.get("title"), args.get("date")) if x)
        return "Adding an event", detail or None

    if name == "get_upcoming_events":
        return "Checking your calendar", f"next {int(_arg(args, 'horizon_days', 60))} days"

    if name == "find_live_events":
        query = str(_arg(args, "query", "")).strip()
        return "Searching for live events", query or None

    if name == "generate_itinerary":
        bits = []
        if args.get("days"):
            bits.append(f"{int(args['days'])} days")
        if args.get("budget"):
            bits.append(_money(args["budget"]))
        if args.get("start_date"):
            bits.append(f"from {args['start_date']}")
        if args.get("prayer_breaks"):
            bits.append("with prayer breaks")
        if args.get("adults_only"):
            bits.append("adults only")
        if args.get("focus") == "dinner_only":
            bits = ["dinner only", *bits]
            return "Finding a restaurant", " · ".join(bits) or None
        return "Building your itinerary", " · ".join(bits) or None

    if name == "get_itinerary":
        return "Reading the current plan", None

    if name == "make_day_cheaper":
        return "Finding cheaper options", f"day {int(_arg(args, 'day', 1))}"

    if name == "add_prayer_breaks":
        return "Adding prayer breaks", None

    if name == "set_transport":
        mode = str(_arg(args, "mode", ""))
        return "Switching transport", "own car" if mode == "own_car" else "taxi"

    if name == "add_stop":
        kind = str(args.get("category") or "").replace("_", " ")
        return "Adding a stop", f"{kind} · day {int(_arg(args, 'day', 1))}" if kind else None

    if name == "edit_stop":
        action = str(_arg(args, "action", ""))
        if action == "replace":
            kind = str(args.get("category") or "").replace("_", " ")
            return "Swapping a stop", f"for {kind}" if kind else None
        if action == "adjust":
            return "Moving a stop", f"to {args.get('start_time')}" if args.get("start_time") else None
        return "Removing a stop", None

    if name == "record_preference":
        kind = str(_arg(args, "kind", "like"))
        subject = str(_arg(args, "subject", "")).strip()
        verb = "likes" if kind == "like" else "dislikes"
        return "Noting a preference", f"{verb} {subject}" if subject else None

    return name.replace("_", " ").capitalize(), None


def summarise_tool_result(name: str, result: dict) -> str:
    """One phrase describing what the tool did, so the row resolves instead of hanging."""
    if not isinstance(result, dict):
        return "done"

    if result.get("error") == "intake_incomplete":
        missing = ", ".join(str(f).replace("_", " ") for f in result.get("missing_fields", []))
        return f"needs {missing}" if missing else "more detail needed"
    if result.get("error"):
        return str(result["error"])[:120]

    if name == "get_upcoming_events":
        return _count(result.get("events", []), "event")
    if name == "find_live_events":
        if not result.get("found"):
            return "nothing found"
        return _count(result.get("events", []), "event") + " found"
    if name == "generate_itinerary":
        return f"{_money(result.get('total'))} of {_money(result.get('cap'))}"
    if name == "get_itinerary":
        if result.get("itinerary_id") is None:
            return "no plan yet"
        budget = result.get("budget") or {}
        return f"{_count(result.get('days', []), 'day')} · {_money(budget.get('total'))}"
    if name == "make_day_cheaper":
        saved = result.get("saved") or 0
        return f"saved {_money(saved)}" if saved > 0 else "nothing cheaper available"
    if name == "set_transport":
        travel = (result.get("travel") or {}).get("total")
        return f"travel now {_money(travel)}" if travel is not None else "repriced"
    if name == "add_stop":
        chosen = result.get("added")
        return f"added {chosen}" if chosen else "added"
    if name in ("add_prayer_breaks", "edit_stop"):
        return f"{_money(result.get('total'))} of {_money(result.get('cap'))}"
    if name == "create_event":
        return "added" if result.get("created") else "already on your calendar"
    if name == "save_family_details":
        return "saved"
    if name == "record_preference":
        return "noted"
    return "done"


class ChatOrchestrator:
    def __init__(self, db: Session, user: User, conversation: Conversation) -> None:
        self.db = db
        self.user = user
        self.conversation = conversation
        # Captured now, while the instances are certainly still usable. See `_rebind`.
        self._user_id = user.id
        self._conversation_id = conversation.id
        self.memory = MemoryService(db, user.id)
        self.touched_itinerary: Itinerary | None = None

    def _rebind(self) -> None:
        """Re-load the user and conversation from the session by id.

        FastAPI exits `yield` dependencies *before* the response body is sent (>= 0.106), so by
        the time this generator runs, `get_db`'s `finally: db.close()` has already fired and every
        instance handed to __init__ is detached. The session itself still works — it simply opens
        a new transaction — which is why this failed so quietly: `db.add(Message(...))` inserted
        happily while `conversation.itinerary_id = x` was silently dropped on flush. The symptom
        was a plan that existed but belonged to no thread, so the right pane came up empty.
        """
        self.user = self.db.get(User, self._user_id) or self.user
        self.conversation = (
            self.db.get(Conversation, self._conversation_id) or self.conversation
        )

    # --- context -----------------------------------------------------------------------------

    def system_prompt(self, user_message: str = "") -> str:
        members = self.db.scalars(
            select(FamilyMember).where(FamilyMember.user_id == self.user.id)
        ).all()
        preferences = self.db.scalars(
            select(Preference).where(Preference.user_id == self.user.id)
        ).all()
        recalled = self.memory.recall(user_message or "family preferences", limit=5)
        # Events are injected, not fetched. They used to be reachable only through
        # get_upcoming_events, and nothing obliged the model to look — so asked to plan an
        # anniversary that was already on the calendar, with its date and its notes, it asked
        # for the date. The list is a handful of rows; carrying it costs less than the round trip.
        upcoming = self.db.scalars(
            select(Event)
            .where(Event.user_id == self.user.id, Event.date >= date.today())
            .order_by(Event.date)
            .limit(12)
        ).all()

        family_text = (
            ", ".join(
                f"{m.name or m.role} ({m.role}, {m.age})" for m in members
            )
            or "not recorded yet"
        )
        likes = [p.subject for p in preferences if p.kind == "like"] or ["none recorded"]
        dislikes = [p.subject for p in preferences if p.kind == "dislike"] or ["none recorded"]
        memory_text = "\n".join(f"- {item['text']}" for item in recalled) or "- nothing yet"
        calendar_text = "\n".join(
            f"- event_id {event.id}: {event.title} — {event.date.isoformat()}, {event.event_type}"
            + (f", notes: {event.notes}" if event.notes else "")
            + (" (already planned)" if event.planned else "")
            for event in upcoming
        ) or "- nothing on the calendar yet"

        return (
            "You are Rihla, a UAE trip planner. You help one family plan short trips (at most 5 "
            "days, inside the UAE) around their upcoming events.\n\n"
            "You do NOT build itineraries yourself. Call generate_itinerary and a deterministic "
            "planner does the scheduling; describe what it returns, never invent times, prices or "
            "places.\n\n"
            "Never quote a time, a price or a total from memory. Call get_itinerary and read the "
            "current figures first — the user edits slots, swaps stops and asks for cheaper days "
            "between messages, so anything you saw earlier in this conversation may already be "
            "wrong, and the real numbers are on screen next to you.\n\n"
            "Never say the plan changed unless a tool you called in THIS turn returned the "
            "change. You can edit an existing plan with make_day_cheaper, add_prayer_breaks, "
            "set_transport, add_stop and edit_stop. To swap one stop for a different kind of "
            "place use edit_stop with action='replace' and a category — never remove it and hope; "
            "removing leaves the day one stop short. Do NOT reach for generate_itinerary to work "
            "around an edit: it builds a replacement plan from scratch and throws the current one "
            "away, so call it a second time only when the user has asked for a new plan or agreed "
            "to start over. Listing a stop the plan does not contain is "
            "worse than admitting the limit — the real plan is on screen beside you, and the "
            "user can see that it did not change.\n\n"
            f"Today is {date.today().isoformat()}.\n"
            f"Signed in as: {self.user.name}\n"
            f"Family: {family_text}\n"
            f"Likes: {', '.join(likes)}\n"
            f"Dislikes: {', '.join(dislikes)}\n"
            f"Remembered from earlier sessions:\n{memory_text}\n"
            f"On their calendar:\n{calendar_text}\n\n"
            "Plan what was asked for and no more. A request for a dinner is generate_itinerary "
            "with focus='dinner_only' — one evening stop — not a day out with a restaurant at the "
            "end of it. Set adults_only when the children are not coming, which an anniversary "
            "usually implies and the event's notes often say outright; without it the evening is "
            "scored for the youngest child in the family.\n\n"
            "Those calendar entries are facts you already have. If the user mentions one of them "
            "— by name or by occasion — use its date and its notes and never ask for a date you "
            "have been given, and pass its event_id exactly as listed. Do not guess an id: the "
            "plan is titled after the event you name, so the wrong one mislabels the whole trip. "
            "get_upcoming_events is only for looking further ahead than the list above.\n\n"
            "Everything listed above is already on file — never ask the user to repeat it. Ask "
            "only for what is genuinely missing: usually just the budget and the dates, and an "
            "event's own date is a fine default start date. When you have enough, call "
            "generate_itinerary; the server validates the checklist and will tell you if something "
            "is still missing, so prefer trying over interrogating. When an event is coming up and "
            "unplanned, offer to plan it. Keep replies short and concrete — the itinerary itself "
            "is shown beside the chat."
        )

    def history(self) -> list[dict]:
        rows = (
            self.db.query(Message)
            .filter(Message.conversation_id == self.conversation.id)
            .order_by(Message.id.desc())
            .limit(HISTORY_LIMIT)
            .all()
        )
        return [
            {"role": row.role, "content": row.content}
            for row in reversed(rows)
            if row.role in ("user", "assistant") and row.content
        ]

    def record(self, role: str, content: str, tool_calls: dict | None = None) -> Message:
        message = Message(
            conversation_id=self.conversation.id,
            role=role,
            content=content,
            tool_calls_json=tool_calls,
        )
        self.db.add(message)
        self.conversation.updated_at = utcnow()
        self.db.flush()
        return message

    # --- tools -------------------------------------------------------------------------------

    def call_tool(self, name: str, arguments: dict) -> dict:
        handler = {
            "save_family_details": self._save_family_details,
            "create_event": self._create_event,
            "get_upcoming_events": self._get_upcoming_events,
            "find_live_events": self._find_live_events,
            "generate_itinerary": self._generate_itinerary,
            "get_itinerary": self._get_itinerary,
            "make_day_cheaper": self._make_day_cheaper,
            "add_prayer_breaks": self._add_prayer_breaks,
            "set_transport": self._set_transport,
            "add_stop": self._add_stop,
            "edit_stop": self._edit_stop,
            "record_preference": self._record_preference,
        }.get(name)
        if handler is None:
            return {"error": f"unknown tool {name}"}
        try:
            return handler(arguments)
        except Exception as exc:  # noqa: BLE001 — a tool failure is a message, not a 500
            log.exception("tool %s failed", name)
            return {"error": str(exc)}

    def _save_family_details(self, args: dict) -> dict:
        adults = max(1, int(_arg(args, "adults", 1)))
        ages = [int(age) for age in _arg(args, "children_ages", [])]

        self.db.query(FamilyMember).filter(FamilyMember.user_id == self.user.id).delete()
        for _ in range(adults):
            self.db.add(FamilyMember(user_id=self.user.id, role="adult", age=35))
        for age in ages:
            self.db.add(FamilyMember(user_id=self.user.id, role="child", age=age))

        for subject in _arg(args, "likes", []):
            self._remember("like", str(subject))
        for subject in _arg(args, "dislikes", []):
            self._remember("dislike", str(subject))

        self.db.flush()
        return {"saved": True, "adults": adults, "children_ages": ages}

    def _create_event(self, args: dict) -> dict:
        try:
            when = date.fromisoformat(str(args["date"]))
        except ValueError:
            return {"error": f"{args.get('date')!r} is not an ISO date"}
        if args.get("event_type") not in EVENT_TYPES:
            return {"error": f"event_type must be one of {', '.join(EVENT_TYPES)}"}

        existing = (
            self.db.query(Event)
            .filter(
                Event.user_id == self.user.id, Event.title == args["title"], Event.date == when
            )
            .first()
        )
        if existing:
            return {"created": False, "reason": "already exists", "event_id": existing.id}

        event = Event(
            user_id=self.user.id,
            title=str(args["title"]),
            event_type=str(args["event_type"]),
            date=when,
            notes=str(args.get("notes") or "") or None,
            planned=False,
        )
        self.db.add(event)
        self.db.flush()
        return {"created": True, "event_id": event.id, "title": event.title}

    def _get_upcoming_events(self, args: dict) -> dict:
        horizon = int(_arg(args, "horizon_days", 60))
        today = date.today()
        events = (
            self.db.query(Event)
            .filter(
                Event.user_id == self.user.id,
                Event.date >= today,
                Event.date <= today + timedelta(days=horizon),
            )
            .order_by(Event.date)
            .all()
        )
        return {
            "events": [
                {
                    "id": event.id,
                    "title": event.title,
                    "event_type": event.event_type,
                    "date": event.date.isoformat(),
                    "days_away": (event.date - today).days,
                    "planned": event.planned,
                }
                for event in events
            ]
        }

    def _find_live_events(self, args: dict) -> dict:
        """Web search → validated event rows → the model. **Writes nothing.**

        This used to save every hit straight to the calendar, which meant one question about
        what was on filled the user's calendar with scraped listings they never asked for. A
        search result is a suggestion; it becomes a calendar entry only when the user says yes,
        and that goes through create_event like anything else.
        """
        rows = find_live_events(
            str(args.get("query") or ""),
            horizon_days=int(_arg(args, "horizon_days", 90)),
        )
        existing = {
            (title.lower(), when)
            for title, when in self.db.query(Event.title, Event.date).filter(
                Event.user_id == self.user.id
            )
        }

        return {
            "found": len(rows),
            "events": [
                {
                    "title": row["title"],
                    "date": row["date"].isoformat(),
                    "event_type": row["event_type"],
                    # So the reply can say "you already have that one" rather than re-offering it.
                    "already_saved": (row["title"].lower(), row["date"]) in existing,
                }
                for row in rows
            ],
        }

    def _generate_itinerary(self, args: dict) -> dict:
        event_id = args.get("event_id")
        if event_id is None:
            event_id = self.conversation.event_id
        event = None
        if event_id is not None:
            event = (
                self.db.query(Event)
                .filter(Event.id == int(event_id), Event.user_id == self.user.id)
                .one_or_none()
            )
            if event is None:
                return {"error": "That event does not exist."}

        start = args.get("start_date")
        try:
            start_date = date.fromisoformat(str(start)) if start else (event.date if event else None)
        except ValueError:
            return {"error": f"{start!r} is not an ISO date"}
        if start_date is None:
            return {"error": "I still need the dates for the trip."}
        if start_date < date.today():
            return {"error": "That start date is in the past."}

        # The reported bug: the model read the date off the calendar correctly and passed a
        # different event's id, so a Milad un Nabi outing was built, titled and marked planned as
        # a Graduation Ceremony. The date is what the user actually said out loud; when it names
        # a different event, that is a contradiction rather than a preference.
        if event is not None and event.date != start_date:
            on_that_day = self.db.scalars(
                select(Event).where(
                    Event.user_id == self.user.id, Event.date == start_date
                )
            ).first()
            if on_that_day is not None and on_that_day.id != event.id:
                return {
                    "error": (
                        f"event_id {event.id} is {event.title!r} on {event.date.isoformat()}, but "
                        f"you asked to start on {start_date.isoformat()}, which is "
                        f"{on_that_day.title!r} (event_id {on_that_day.id}). Use that id, or a "
                        f"start date that matches the event you meant."
                    )
                }

        days = int(_arg(args, "days", 3))
        if not 1 <= days <= 5:
            return {"error": "Trips can be at most 5 days."}
        focus = str(_arg(args, "focus", itinerary_service.FULL_DAY))
        if focus not in itinerary_service.PLAN_FOCUS:
            return {"error": f"Unknown focus {focus!r}."}
        budget = float(_arg(args, "budget", 0))
        if budget <= 0:
            return {"error": "I still need a budget in AED."}

        # Carry the transport mode across a rebuild. Same failure as dropping the event: the
        # family says "we have our own car", the plan is rebuilt, and the taxi fares come back.
        current = self._resolve_itinerary(None)
        try:
            created = itinerary_service.generate(
                self.db,
                self.user,
                start_date=start_date,
                num_days=days,
                total_budget=budget,
                start_lat=self.user.home_base_lat,
                start_lng=self.user.home_base_lng,
                event_id=event.id if event else None,
                title=event.title if event else None,
                currency=self.user.default_currency,
                prayer_breaks=bool(_arg(args, "prayer_breaks", False)),
                transport_mode=current.transport_mode if current else itinerary_service.TAXI,
                adults_only=bool(_arg(args, "adults_only", False)),
                focus=focus,
            )
        except itinerary_service.IntakeIncomplete as exc:
            return {"error": "intake_incomplete", "missing_fields": exc.missing}

        self.conversation.itinerary_id = created.id
        if event is not None:
            self.conversation.event_id = event.id
            self.conversation.title = event.title
        self.db.flush()
        return self._plan_result(created)

    def _resolve_itinerary(self, itinerary_id: int | None = None) -> Itinerary | None:
        """Find a plan for THIS user. Every branch filters by user_id, including the explicit id."""
        owned = self.db.query(Itinerary).filter(Itinerary.user_id == self.user.id)

        if itinerary_id is not None:
            return owned.filter(Itinerary.id == int(itinerary_id)).one_or_none()

        if self.conversation.itinerary_id:
            attached = owned.filter(Itinerary.id == self.conversation.itinerary_id).one_or_none()
            if attached is not None:
                return attached

        return owned.order_by(Itinerary.updated_at.desc()).first()

    def _get_itinerary(self, args: dict) -> dict:
        """The plan as it stands right now, straight from the same payload the UI renders.

        Without this the model can only describe a plan from whatever a previous tool call left in
        its context — which goes stale the moment a slot is edited or a day is made cheaper, and it
        then reports totals that contradict the budget bar sitting next to it.
        """
        itinerary = self._resolve_itinerary(args.get("itinerary_id"))
        if itinerary is None:
            return {"itinerary": None, "note": "No plan has been generated yet."}

        payload = itinerary_service.itinerary_payload(self.db, itinerary)
        budget = payload["budget"]
        return {
            "itinerary_id": itinerary.id,
            "title": payload["title"],
            "start_date": payload["start_date"],
            "num_days": payload["num_days"],
            "currency": payload["currency"],
            "days": [
                {
                    "day": day["day_index"] + 1,
                    "date": day["date"],
                    "theme": day["theme"],
                    "subtotal": day["subtotal"],
                    "driving_min": day["driving_total_min"],
                    "stops": [
                        {
                            "slot_id": slot["id"],
                            "name": slot["place"].name,
                            "category": slot["place"].category,
                            "start": slot["start_time"],
                            "end": slot["end_time"],
                            "cost": slot["cost_breakdown"].get("total", 0),
                        }
                        for slot in day["slots"]
                    ],
                }
                for day in payload["days"]
            ],
            "budget": {
                "total": budget["total"],
                "cap": budget["cap"],
                "remaining": budget["remaining"],
                "over_budget": budget["over_budget"],
                "activities": budget["categories"]["activities"],
                "food": budget["categories"]["food"],
                "travel": budget["categories"]["travel"],
            },
        }

    def _plan_result(self, itinerary: Itinerary) -> dict:
        """What every plan edit returns: the figures as they stand *after* the edit.

        Also marks the itinerary as touched, which is what makes `_emit_updates` push the new
        state to the right pane. A mutating tool that forgets this leaves the pane showing the
        plan as it was before the edit.
        """
        self.touched_itinerary = itinerary
        payload = itinerary_service.itinerary_payload(self.db, itinerary)
        return {
            "itinerary_id": itinerary.id,
            "days": [
                {
                    "day": day["day_index"] + 1,
                    "theme": day["theme"],
                    "subtotal": day["subtotal"],
                    "stops": [slot["place"].name for slot in day["slots"]],
                }
                for day in payload["days"]
            ],
            "total": payload["budget"]["total"],
            "cap": payload["budget"]["cap"],
            "remaining": payload["budget"]["remaining"],
            "transport_mode": payload["transport_mode"],
            "vehicle": payload["vehicle"],
            "travel": {"total": payload["budget"]["categories"]["travel"]},
        }

    def _open_plan(self, args: dict) -> tuple[Itinerary | None, dict | None]:
        itinerary = self._resolve_itinerary(args.get("itinerary_id"))
        if itinerary is None:
            return None, {"error": "No plan has been generated yet, so there is nothing to edit."}
        return itinerary, None

    def _make_day_cheaper(self, args: dict) -> dict:
        itinerary, error = self._open_plan(args)
        if error:
            return error

        before = itinerary_service.itinerary_payload(self.db, itinerary)["budget"]["total"]
        try:
            itinerary_service.cheaper_day(
                self.db, itinerary, self.user, int(_arg(args, "day", 1)) - 1
            )
        except ValueError as exc:
            return {"error": str(exc)}

        result = self._plan_result(itinerary)
        # The planner substitutes only when it finds something genuinely cheaper, so this is
        # often zero. Reporting it keeps the reply from announcing a saving that did not happen.
        result["saved"] = round(before - result["total"], 2)
        return result

    def _add_prayer_breaks(self, args: dict) -> dict:
        itinerary, error = self._open_plan(args)
        if error:
            return error
        itinerary_service.add_prayer_breaks(self.db, itinerary, self.user)
        return self._plan_result(itinerary)

    def _set_transport(self, args: dict) -> dict:
        itinerary, error = self._open_plan(args)
        if error:
            return error

        mode = str(_arg(args, "mode", ""))
        if mode not in itinerary_service.TRANSPORT_MODES:
            return {"error": f"Unknown transport mode {mode!r}."}

        itinerary.transport_mode = mode
        itinerary_service.recost_travel(self.db, itinerary, self.user)
        return self._plan_result(itinerary)

    def _add_stop(self, args: dict) -> dict:
        itinerary, error = self._open_plan(args)
        if error:
            return error

        try:
            _, _, outcome = itinerary_service.add_stop(
                self.db, itinerary, self.user,
                day_index=int(_arg(args, "day", 1)) - 1,
                category=args.get("category"),
            )
        except ValueError as exc:
            return {"error": str(exc)}

        result = self._plan_result(itinerary)
        result["added"] = outcome["chosen"]
        # So the reply can offer a different one instead of pretending there was no choice.
        result["alternatives"] = outcome["alternatives"]
        return result

    def _edit_stop(self, args: dict) -> dict:
        itinerary, error = self._open_plan(args)
        if error:
            return error

        # Scoped to this itinerary, which _resolve_itinerary already scoped to this user — a
        # slot_id from someone else's plan finds nothing rather than editing it.
        slot = (
            self.db.query(Slot)
            .filter(Slot.id == int(_arg(args, "slot_id", 0)), Slot.itinerary_id == itinerary.id)
            .one_or_none()
        )
        if slot is None:
            return {"error": "No such stop in this plan. Call get_itinerary for current slot_ids."}

        try:
            itinerary_service.patch_slot(
                self.db,
                itinerary,
                self.user,
                slot,
                action=str(_arg(args, "action", "")),
                start_time=args.get("start_time"),
                category=args.get("category"),
            )
        except ValueError as exc:
            return {"error": str(exc)}
        return self._plan_result(itinerary)

    def _record_preference(self, args: dict) -> dict:
        self._remember(str(_arg(args, "kind", "like")), str(_arg(args, "subject", "")),
                       args.get("category"))
        return {"recorded": True}

    def _remember(self, kind: str, subject: str, category: str | None = None) -> None:
        if kind not in ("like", "dislike") or not subject.strip():
            return
        existing = (
            self.db.query(Preference)
            .filter(
                Preference.user_id == self.user.id,
                Preference.kind == kind,
                Preference.subject == subject,
            )
            .first()
        )
        if existing:
            existing.strength = min(1.0, existing.strength + 0.1)
            return
        preference = Preference(
            user_id=self.user.id, kind=kind, subject=subject, category=category, source="stated"
        )
        self.db.add(preference)
        self.db.flush()
        self.memory.remember_preference(preference)

    # --- streaming ---------------------------------------------------------------------------

    @traced("chat.stream", run_type="chain")
    def stream(self, user_message: str) -> Iterator[str]:
        self._rebind()
        self.record("user", user_message)
        self.db.commit()
        # A failure here propagates to the router, which turns it into an `error` frame. The
        # assistant is a hard dependency, so a failure is reported rather than worked around.
        yield from self._llm(user_message)

    def _llm(self, user_message: str) -> Iterator[str]:
        from openai import OpenAI

        client = wrap_openai(OpenAI(api_key=settings.openai_api_key))
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt(user_message)},
            *self.history(),
        ]

        answer = ""
        for _ in range(MAX_TOOL_ROUNDS):
            stream = client.chat.completions.create(
                model=settings.openai_chat_model,
                messages=messages,
                tools=TOOLS,
                stream=True,
            )

            content = ""
            pending: dict[int, dict] = {}
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if delta.content:
                    content += delta.content
                    yield sse("token", delta.content)

                for call in delta.tool_calls or []:
                    slot = pending.setdefault(
                        call.index, {"id": "", "name": "", "arguments": ""}
                    )
                    if call.id:
                        slot["id"] = call.id
                    if call.function and call.function.name:
                        slot["name"] = call.function.name
                    if call.function and call.function.arguments:
                        slot["arguments"] += call.function.arguments

            if not pending:
                answer = content
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {"name": call["name"], "arguments": call["arguments"] or "{}"},
                        }
                        for call in pending.values()
                    ],
                }
            )

            for call in pending.values():
                try:
                    arguments = json.loads(call["arguments"] or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                label, detail = describe_tool_call(call["name"], arguments)
                yield sse(
                    "tool",
                    {"id": call["id"], "name": call["name"], "label": label, "detail": detail},
                )

                result = self.call_tool(call["name"], arguments)
                self.db.commit()

                yield sse(
                    "tool_done",
                    {
                        "id": call["id"],
                        "outcome": summarise_tool_result(call["name"], result),
                        "failed": bool(isinstance(result, dict) and result.get("error")),
                    },
                )

                # Surface a blocked intake to the client as well as to the model, so the chat can
                # render the numbered checklist from the design. The model will also ask in prose;
                # the checklist makes what is missing scannable.
                if isinstance(result, dict) and result.get("error") == "intake_incomplete":
                    yield sse("intake_required", {"missing_fields": result.get("missing_fields", [])})

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(result, default=str),
                    }
                )
                yield from self._emit_updates()

        self.record("assistant", answer)
        self.db.commit()
        yield sse("done", {"conversation_id": self.conversation.id})

    def _emit_updates(self) -> Iterator[str]:
        """Push the right pane's new state as soon as a tool changed it."""
        if self.touched_itinerary is None:
            return
        payload = itinerary_service.itinerary_payload(self.db, self.touched_itinerary)
        yield sse("itinerary_updated", {"itinerary_id": self.touched_itinerary.id})
        yield sse("budget_updated", payload["budget"])

