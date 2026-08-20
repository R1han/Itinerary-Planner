"""One chat turn: call the model, run its tools, check the answer, then send it.

A plain generator of SSE frames — the same shape `_llm` has always had, so the router, the
streaming response and every test that stubs `_llm` are untouched by it existing.

    agent ──tool_calls & rounds left?──► tools ──┐
      │                                          │
      └──no──► review ──not ok & retries left?───┘
                 │
                 └──► respond

No framework. The state machine is a while loop and two conditionals; a graph library would have
bought worker-thread node execution, a cross-thread SQLAlchemy session, contextvar copying and a
recursion limit to tune, none of which are the problem being solved.

What is new against the loop this replaces:

  * The tools offered are never taken away mid-turn. The old rescue round re-called the model with
    `tools=` removed when the rounds ran out, which left it able to describe an action and unable
    to take one — manufacturing the exact "claimed a change that didn't happen" bug the prompt
    spends a paragraph forbidding. The last round is pinned to `tool_choice="none"` with guidance
    saying to report the trace and claim nothing more, so the model can still see what exists.
  * The first request is pinned to `tool_choice="required"` unless the message is a greeting.
    Refusing a bad call is easy; nothing else makes the model call SOMETHING rather than answer
    from its own knowledge of the UAE.
  * The reply is held back until it has been checked, then streamed. Tool rows still go out live,
    so the pane and the activity list move while the model is working.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from typing import Any

from ..config import settings
from . import policy, reviewer
from .orchestrator import (
    MAX_TOOL_ROUNDS,
    describe_tool_call,
    persisted_trace,
    sse,
    summarise_tool_result,
)
from .tracing import wrap_openai

log = logging.getLogger(__name__)

# One correction, not two. Each retry is a whole generation plus another review, and a reviewer
# that disagrees twice is usually the one that is wrong.
MAX_REVIEW_ROUNDS = 1

# Roughly a word at a time, so a held-back reply still arrives as prose rather than one block.
STREAM_CHUNK = 24

OUT_OF_ROUNDS = (
    "You have used every tool round available for this turn. Do not describe anything as done "
    "that the results above do not already show. Report exactly what happened, say plainly what "
    "is still outstanding, and stop."
)


def run_turn(orchestrator: Any, user_message: str, *, client: Any = None) -> Iterator[str]:
    """Yield this turn's SSE frames. Assumes `stream()` has already recorded the user message."""
    if client is None:
        from openai import OpenAI

        client = wrap_openai(OpenAI(api_key=settings.openai_api_key))

    messages: list[dict] = [
        {"role": "system", "content": orchestrator.system_prompt(user_message)},
        *orchestrator.history(),
    ]
    trace: list[dict] = []
    tool_rounds = 0
    review_rounds = 0
    verdict = reviewer.Verdict()

    while True:
        spent = tool_rounds >= MAX_TOOL_ROUNDS
        if spent:
            # Pinned rather than stripped: the model keeps seeing what exists, so "I still need to
            # check the plan" stays available to it instead of being replaced by a guess.
            choice = "none"
            messages.append({"role": "system", "content": OUT_OF_ROUNDS})
        elif tool_rounds == 0 and review_rounds == 0 and not policy.is_small_talk(user_message):
            choice = "required"
        else:
            choice = "auto"

        content, pending = _generate(orchestrator, client, messages, tool_choice=choice)

        if pending and not spent:
            messages.append(
                {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": call["arguments"] or "{}",
                            },
                        }
                        for call in pending
                    ],
                }
            )
            for call in pending:
                yield from _run_tool(orchestrator, call, trace, messages)
            tool_rounds += 1
            continue

        draft = content
        verdict = _check(user_message, trace, draft)
        if not verdict.is_ok and review_rounds < MAX_REVIEW_ROUNDS:
            review_rounds += 1
            messages.append({"role": "assistant", "content": draft or "(no reply yet)"})
            messages.append({"role": "system", "content": _correction(verdict)})
            continue
        break

    yield from _respond(orchestrator, draft, verdict, trace)


# --- the model ---------------------------------------------------------------------------------


def _generate(
    orchestrator: Any, client: Any, messages: list[dict], *, tool_choice: str
) -> tuple[str, list[dict]]:
    """One completion. Returns the prose and the tool calls, neither of them streamed.

    Prose is buffered on purpose: a sentence already on its way to the browser cannot be checked,
    and an intermediate round's "let me look that up" is preamble the user never needed.
    """
    from .orchestrator import TOOLS

    stream = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=messages,
        tools=TOOLS,
        tool_choice=tool_choice,
        stream=True,
        stream_options={"include_usage": True},
    )

    content = ""
    pending: dict[int, dict] = {}
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            content += delta.content
        for call in delta.tool_calls or []:
            slot = pending.setdefault(call.index, {"id": "", "name": "", "arguments": ""})
            if call.id:
                slot["id"] = call.id
            if call.function and call.function.name:
                slot["name"] = call.function.name
            if call.function and call.function.arguments:
                slot["arguments"] += call.function.arguments

    return content, list(pending.values())


def _run_tool(orchestrator: Any, call: dict, trace: list[dict], messages: list[dict]) -> Iterator[str]:
    """Execute one call, telling the browser about it as it goes."""
    try:
        arguments = json.loads(call["arguments"] or "{}")
    except json.JSONDecodeError:
        arguments = {}

    label, detail = describe_tool_call(call["name"], arguments)
    yield sse("tool", {"id": call["id"], "name": call["name"], "label": label, "detail": detail})

    result = orchestrator.call_tool(call["name"], arguments)
    orchestrator.db.commit()

    # Computed here, once, from the result itself — never inferred later from the prose.
    changed = policy.applied(call["name"], result)
    trace.append({"name": call["name"], "args": arguments, "result": result, "applied": changed})

    yield sse(
        "tool_done",
        {
            "id": call["id"],
            "outcome": summarise_tool_result(call["name"], result),
            "failed": bool(isinstance(result, dict) and result.get("error")),
        },
    )
    if isinstance(result, dict) and result.get("error") == "intake_incomplete":
        yield sse("intake_required", {"missing_fields": result.get("missing_fields", [])})

    messages.append(
        {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, default=str)}
    )
    yield from orchestrator._emit_updates()


# --- checking ----------------------------------------------------------------------------------


def _check(user_message: str, trace: list[dict], draft: str) -> reviewer.Verdict:
    """Deterministic first, and an LLM only when there is something for it to weigh.

    A turn that called nothing but `find_places` and claims nothing is the common case, and paying
    a whole completion to be told so would tax every question the user asks.
    """
    flagged = policy.claim_check(draft, trace)
    mutated = any(entry["name"] in policy.MUTATING_TOOLS for entry in trace)
    if not flagged and not mutated:
        return reviewer.Verdict()

    # A turn that came back asking has exactly one right reply — the question — and there is
    # nothing left for a judgement to add once claim_check has passed. Live validation showed the
    # cost of asking anyway: told "change the location of the plan", the assistant drafted the
    # confirmation, the reviewer answered `needs_tools` ("read the plan before summarising it"),
    # and the rewrite came back as a tidy summary of the unchanged plan with the question gone.
    # Safe, and useless — the user is left waiting on an answer nobody asked them for.
    if not flagged and not any(entry["applied"] for entry in trace) and any(
        isinstance(entry.get("result"), dict) and entry["result"].get("needs_confirmation")
        for entry in trace
    ):
        return reviewer.Verdict()

    if flagged:
        log.info("claim_check flagged %d sentence(s) with nothing applied", len(flagged))
    return reviewer.review(user_message, trace, draft)


def _correction(verdict: reviewer.Verdict) -> str:
    parts = ["That draft was not sent. " + (verdict.guidance or "Rewrite it.")]
    if verdict.unsupported_claims:
        quoted = "; ".join(f"{claim!r}" for claim in verdict.unsupported_claims)
        parts.append(f"These are not supported by anything a tool returned: {quoted}.")
    if verdict.wants_tools:
        parts.append(
            "Call "
            + (", ".join(verdict.missing_tools) if verdict.missing_tools else "the tool you need")
            + " and answer from what it returns."
        )
    return " ".join(parts)


def _settle(draft: str, verdict: reviewer.Verdict, trace: list[dict]) -> str:
    """What actually ships when the reviewer still objects and the retries are gone.

    Shipping the draft anyway would be the bug this exists to catch, so the offending sentences
    come out. If that leaves nothing, the trace speaks for itself.
    """
    if verdict.is_ok:
        return draft

    settled = draft
    for claim in verdict.unsupported_claims:
        settled = settled.replace(claim, " ")

    # Tidy horizontally only. `" ".join(text.split())` reads like whitespace cleanup and is not:
    # split() with no argument splits on newlines too, so it flattened the whole reply onto one
    # line. The client renders markdown, where `###` and `-` mean nothing without a line break —
    # so a reply that had merely lost a sentence arrived as a wall of literal `###` and `-`.
    # Order matters: the removed sentence leaves its line holding a space, and a blank run only
    # collapses once that space is gone.
    settled = re.sub(r"[^\S\n]+", " ", settled)
    settled = "\n".join(line.rstrip() for line in settled.split("\n"))
    settled = re.sub(r"\n{3,}", "\n\n", settled).strip()
    if settled:
        return settled

    changed = [entry["name"] for entry in trace if entry["applied"]]
    if changed:
        return "That went through, but I could not describe it accurately — the plan on screen is correct."
    return "I could not complete that, and nothing in the plan has changed."


def _respond(
    orchestrator: Any, draft: str, verdict: reviewer.Verdict, trace: list[dict]
) -> Iterator[str]:
    answer = _settle(draft, verdict, trace)
    for index in range(0, len(answer), STREAM_CHUNK):
        yield sse("token", answer[index : index + STREAM_CHUNK])

    # The trace is stored with the reply so the next turn inherits what was proposed and refused.
    # Answering "yes" to a question whose evidence has been thrown away is how a confirmation
    # became a swap that never happened.
    orchestrator.record("assistant", answer, tool_calls=persisted_trace(trace))
    orchestrator.db.commit()
    yield sse("done", {"conversation_id": orchestrator.conversation.id})
