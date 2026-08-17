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
import re
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
    User,
    utcnow,
)
from . import itinerary as itinerary_service
from .memory import MemoryService
from .tracing import traced, wrap_openai

log = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 4
HISTORY_LIMIT = 20

EVENT_TYPES = ["birthday", "anniversary", "family_visit", "graduation", "eid", "holiday", "other"]

# Note the absence of any user_id parameter — deliberate, and load-bearing.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "save_family_details",
            "description": "Record who is in the family and what they like or dislike.",
            "parameters": {
                "type": "object",
                "properties": {
                    "adults": {"type": "integer", "minimum": 1, "maximum": 12},
                    "children_ages": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0, "maximum": 17},
                        "description": "One entry per child, their age in years.",
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
                    "horizon_days": {"type": "integer", "minimum": 1, "maximum": 730, "default": 60}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_itinerary",
            "description": (
                "Build a complete itinerary for an event. The server rejects this if the intake "
                "checklist is incomplete; ask for whatever is missing and try again."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "integer"},
                    "start_date": {"type": "string", "description": "ISO date, YYYY-MM-DD"},
                    "days": {"type": "integer", "minimum": 1, "maximum": 5},
                    "budget": {"type": "number", "minimum": 1},
                    "prayer_breaks": {"type": "boolean", "default": False},
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


def sse(event_type: str, data) -> str:
    """One Server-Sent Event frame."""
    return f"data: {json.dumps({'type': event_type, 'data': data}, default=str)}\n\n"


class ChatOrchestrator:
    def __init__(self, db: Session, user: User, conversation: Conversation) -> None:
        self.db = db
        self.user = user
        self.conversation = conversation
        self.memory = MemoryService(db, user.id)
        self.touched_itinerary: Itinerary | None = None

    # --- context -----------------------------------------------------------------------------

    def system_prompt(self, user_message: str = "") -> str:
        members = self.db.scalars(
            select(FamilyMember).where(FamilyMember.user_id == self.user.id)
        ).all()
        preferences = self.db.scalars(
            select(Preference).where(Preference.user_id == self.user.id)
        ).all()
        recalled = self.memory.recall(user_message or "family preferences", limit=5)

        family_text = (
            ", ".join(
                f"{m.name or m.role} ({m.role}, {m.age})" for m in members
            )
            or "not recorded yet"
        )
        likes = [p.subject for p in preferences if p.kind == "like"] or ["none recorded"]
        dislikes = [p.subject for p in preferences if p.kind == "dislike"] or ["none recorded"]
        memory_text = "\n".join(f"- {item['text']}" for item in recalled) or "- nothing yet"

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
            f"Today is {date.today().isoformat()}.\n"
            f"Signed in as: {self.user.name}\n"
            f"Family: {family_text}\n"
            f"Likes: {', '.join(likes)}\n"
            f"Dislikes: {', '.join(dislikes)}\n"
            f"Remembered from earlier sessions:\n{memory_text}\n\n"
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
            "generate_itinerary": self._generate_itinerary,
            "get_itinerary": self._get_itinerary,
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
        adults = max(1, int(args.get("adults", 1)))
        ages = [int(age) for age in args.get("children_ages", [])]

        self.db.query(FamilyMember).filter(FamilyMember.user_id == self.user.id).delete()
        for _ in range(adults):
            self.db.add(FamilyMember(user_id=self.user.id, role="adult", age=35))
        for age in ages:
            self.db.add(FamilyMember(user_id=self.user.id, role="child", age=age))

        for subject in args.get("likes", []):
            self._remember("like", str(subject))
        for subject in args.get("dislikes", []):
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
        horizon = int(args.get("horizon_days", 60))
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

    def _generate_itinerary(self, args: dict) -> dict:
        event_id = args.get("event_id")
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

        days = int(args.get("days", 3))
        if not 1 <= days <= 5:
            return {"error": "Trips can be at most 5 days."}
        budget = float(args.get("budget", 0))
        if budget <= 0:
            return {"error": "I still need a budget in AED."}

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
                prayer_breaks=bool(args.get("prayer_breaks", False)),
            )
        except itinerary_service.IntakeIncomplete as exc:
            return {"error": "intake_incomplete", "missing_fields": exc.missing}

        self.touched_itinerary = created
        self.conversation.itinerary_id = created.id
        if event is not None:
            self.conversation.event_id = event.id
            self.conversation.title = event.title
        self.db.flush()

        payload = itinerary_service.itinerary_payload(self.db, created)
        return {
            "itinerary_id": created.id,
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
        }

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

    def _record_preference(self, args: dict) -> dict:
        self._remember(str(args.get("kind", "like")), str(args.get("subject", "")),
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
                yield sse("tool", {"name": call["name"]})
                result = self.call_tool(call["name"], arguments)
                self.db.commit()

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

