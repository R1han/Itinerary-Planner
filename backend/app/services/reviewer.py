"""The second opinion on a draft reply, before the user ever sees it.

`policy.claim_check` catches the blunt version of the reported bug — prose that claims a change
when nothing in the turn changed anything — with a verb list and no LLM. This catches what a verb
list cannot: a reply that names a stop the plan does not contain, quotes a total the tools never
returned, describes a proposal as though it had been applied, or answers a question about the
catalog without having looked. It also gets to say the turn is not finished, and send the loop
back for the call that was missed.

What it is shown is the point:

  * the raw tool results, truncated but NOT summarised. `summarise_tool_result` is deliberately
    lossy — every error becomes the string "no change made", and the confirmation protocol's
    `applied: false` / `question_for_the_user` / `plan_is_unchanged` payload becomes "needs your
    OK". A reviewer given that cannot tell a stale date from a missing stop from a budget
    overrun, so it cannot say WHICH claim is unsupported. Those strings are for the activity rows.
  * the per-call `applied` flag from policy, which is computed, not judged.
  * the user's message and the draft.

It is NOT shown the conversation history. The question is whether this reply is supported by this
turn's evidence, and the history is where an earlier turn's confident wrong answer lives.

Every failure path returns `ok`. A reviewer that takes down a turn is worse than the bug it
exists to catch.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ..config import settings
from .tracing import traced, wrap_openai

log = logging.getLogger(__name__)

# Must comfortably hold a WHOLE plan result, which is the largest thing a tool returns and the
# one the reviewer is most often asked about.
#
# This was 400, and live validation showed exactly why that is not a detail. A three-day plan
# serialises to ~760 characters, so the trace the reviewer saw stopped inside day two. It then
# reported — correctly, from what it could see — that day three was unsupported, and the assistant
# dutifully deleted a real day from its reply. A truncation limit set too low does not make the
# reviewer blind, it makes it confidently wrong, and the loop turns that into lost information.
#
# A five-day plan is around 1,200. This leaves room and still keeps a fourteen-call turn inside a
# few thousand tokens.
RESULT_CHARS = 2000

OK = "ok"
NEEDS_TOOLS = "needs_tools"
REWRITE = "rewrite"

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": [OK, NEEDS_TOOLS, REWRITE],
            "description": (
                "'ok' — every claim in the draft is supported by the trace. "
                "'rewrite' — the draft asserts something the trace does not support, and no "
                "further tool call is needed to fix it; the wording is what is wrong. "
                "'needs_tools' — the draft cannot be supported without a call that was not made, "
                "such as answering about the plan without reading it."
            ),
        },
        "unsupported_claims": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Quote each unsupported sentence from the draft, verbatim. Empty if ok.",
        },
        "missing_tools": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Tool names that should have been called. Empty unless needs_tools.",
        },
        "guidance": {
            "type": "string",
            "description": (
                "One or two sentences addressed to the assistant, saying what to do differently. "
                "Empty if ok."
            ),
        },
    },
    "required": ["verdict", "unsupported_claims", "missing_tools", "guidance"],
    "additionalProperties": False,
}

REVIEW_PROMPT = (
    "You are checking one draft reply from a UAE trip-planning assistant against the tools it "
    "actually called this turn. The trace below is the whole truth about what happened: if it is "
    "not in the trace, it did not happen.\n\n"
    "`applied` is computed by the server, not claimed by the assistant. `applied: false` means "
    "that call changed NOTHING — it either errored or came back asking the user a question. A "
    "draft that describes such a call as done is the single most important thing to catch: the "
    "user is reading one plan in the chat and looking at a different one on screen.\n\n"
    "Also catch: a stop, price, time or total that does not appear in any result; a question "
    "answered about the itinerary with no call that read it; a claim about what the catalog "
    "contains with no search behind it.\n\n"
    "Do NOT flag: an offer or a question about what could be done next; a reply that correctly "
    "reports a failure, a refusal, or that nothing changed; a reply that asks the user to choose. "
    "Those are the assistant behaving properly, and calling them unsupported would train the "
    "honest answer out of it.\n\n"
    "Prefer 'ok' when the draft is merely terse. Only 'rewrite' when a specific sentence asserts "
    "something the trace contradicts or does not contain."
)


@dataclass(frozen=True)
class Verdict:
    verdict: str = OK
    unsupported_claims: list[str] = field(default_factory=list)
    missing_tools: list[str] = field(default_factory=list)
    guidance: str = ""

    @property
    def is_ok(self) -> bool:
        return self.verdict == OK

    @property
    def wants_tools(self) -> bool:
        return self.verdict == NEEDS_TOOLS


def render_trace(trace: list[dict]) -> str:
    """The turn's calls, raw enough to argue with.

    Truncated rather than summarised: a shortened error still says which stop was not found, and
    that is exactly the detail the summary throws away.
    """
    if not trace:
        return "(no tools were called this turn)"

    lines = []
    for index, entry in enumerate(trace, start=1):
        result = entry.get("result")
        body = result if isinstance(result, str) else json.dumps(result, default=str)
        if len(body) > RESULT_CHARS:
            body = body[:RESULT_CHARS] + "…(truncated)"
        lines.append(
            f"{index}. {entry.get('name')}({json.dumps(entry.get('args') or {}, default=str)})\n"
            f"   applied: {bool(entry.get('applied'))}\n"
            f"   result: {body}"
        )
    return "\n".join(lines)


@traced("chat.review", run_type="llm")
def review(user_message: str, trace: list[dict], draft: str, *, client: Any = None) -> Verdict:
    """Check `draft` against `trace`. Never raises; an unusable answer is `ok`."""
    if not (draft or "").strip():
        return Verdict()  # nothing to check; the loop has its own guarantee of a non-empty reply

    try:
        if client is None:
            from openai import OpenAI

            client = wrap_openai(OpenAI(api_key=settings.openai_api_key))

        response = client.chat.completions.create(
            model=settings.openai_chat_model,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "review", "strict": True, "schema": REVIEW_SCHEMA},
            },
            messages=[
                {"role": "system", "content": REVIEW_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"USER SAID:\n{user_message}\n\n"
                        f"TOOLS CALLED THIS TURN:\n{render_trace(trace)}\n\n"
                        f"DRAFT REPLY:\n{draft}"
                    ),
                },
            ],
        )
        message = response.choices[0].message
        # Structured outputs can come back as a refusal rather than content. Unlogged it looks
        # exactly like approval, which is the failure mode this whole file exists to stop.
        if getattr(message, "refusal", None):
            log.warning("review refused: %s", message.refusal)
            return Verdict()

        payload = json.loads(message.content or "{}")
    except Exception:  # noqa: BLE001 — see the module docstring: review must never fail a turn
        log.exception("review failed; letting the draft through")
        return Verdict()

    verdict = payload.get("verdict")
    if verdict not in (OK, NEEDS_TOOLS, REWRITE):
        log.warning("review returned an unknown verdict %r", verdict)
        return Verdict()

    return Verdict(
        verdict=verdict,
        unsupported_claims=[str(x) for x in (payload.get("unsupported_claims") or [])],
        missing_tools=[str(x) for x in (payload.get("missing_tools") or [])],
        guidance=str(payload.get("guidance") or ""),
    )
