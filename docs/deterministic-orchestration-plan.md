# Deterministic tool calling + a reviewer pass

Status: approved with amendments, not started. Working tree unchanged.

Revision 2 — incorporates the review response: no agent framework (§3), conditional
reviewer (§5.4), `ChatPanel` fix pulled into scope (§6), `drop_day` confirmation (§4).

---

## 1. Context

Rihla's chat agent is a hand-rolled ReAct loop over the raw OpenAI SDK
(`orchestrator.py:1987-2101`): up to `MAX_TOOL_ROUNDS = 6` passes, all 14 tools offered every
time, no verification of the answer before it streams to the browser.

The deterministic half of the product is already sound. `planner.py` solves the schedule and
`validator.py` runs a bounded `validate → repair → re-validate` loop with `MAX_REPAIR_PASSES = 24`.
**The LLM never builds an itinerary.** The problem is entirely in the control plane: which tools
get called, whether they succeeded, and whether the prose that reaches the user matches what
actually happened.

Today that control plane is enforced by *prose*. `system_prompt()` (`orchestrator.py:1002-1197`)
is ~150 lines of English policy, and the git history shows it growing one bug at a time:

```
cae1d0f Make starting-emirate selection deterministic instead of prompt-only
40bc116 Stop the chat inventing ids, and stop it rebuilding plans it should edit
c503b23 Stop a question being reported as an answer, and a flag granting itself
d16612f Say which events have no plan instead of handing over booleans to negate
e344c9e Stop restaurant swaps from silently discarding the plan
```

Each is a real fix. Each is a rule the model may still ignore.

**Intended outcome:** move that policy out of English and into code, and add a bounded verification
pass that checks the draft answer against what the tools actually returned before the user sees it.

---

## 2. The reported failure, diagnosed

A plan for a Dec 10 birthday was built in Dubai. The user said "We live in Abu Dhabi", then
"Change the location of the plan to Abu Dhabi". Twice the assistant replied the location **had
been changed** — and both times returned the byte-identical Dubai itinerary: same three stops,
same `1940.05 AED`. Finally it called `find_places` three times and narrated a "Suggested
Itinerary" in prose that was never applied to anything.

Three bugs. Only two are architectural.

| # | Bug | Fixed by |
|---|---|---|
| 1 | Fabricated success (×2) | `claim_check` (T2b) + per-call `applied` ledger (T2) + reviewer (T3) |
| 2 | A plan narrated in chat that isn't the plan on screen | Same |
| 3 | **A capability gap no architecture fixes** | New mutators (§4) |

### The capability gap is systematic

`emirates_json` is written in exactly one place — `itinerary.py:263`, during plan creation.
**No code path anywhere changes an existing plan's region.** More generally:

> **Every field on `Itinerary` set at creation is write-once, except two:** `start_date`
> (via `reschedule`, `itinerary.py:1531`) and `transport_mode` (via `set_transport`).

That is the structural reason the model fabricates. For most "change X" requests there is
literally no tool, so the model has no legal move — and a model with no legal move narrates.

Compare `reschedule_itinerary`: someone hit "change the dates", found no tool, and added one
(commit `1a4e7c8`). "Change the region" never got that treatment.

Confirming detail: `starting_emirate_hint` is consumed **only** inside `_generate_itinerary`
(`orchestrator.py:1588`). Both the user's stated home emirate and the UI dropdown were parsed and
then discarded, because their only consumer was a tool that had already been refused.

**Without the missing capabilities, verification upgrades a confident lie into an honest "I can't
do that." Necessary, and not sufficient.** Both halves ship here.

### Two further root causes

**Tool results are never persisted.** `record()` is only ever called with `"user"` and
`"assistant"` (`orchestrator.py:1981`, `:2099`). `Message.tool_calls_json` already exists as a
column and `record()` already accepts and writes it (`:1213-1223`) — **no caller has ever passed
it.** `history()` (`:1199`) filters to prose roles. So the two-turn confirmation protocol —
`_unapplied()` (`:1921`) returns `{"applied": false, "needs_confirmation": …,
"plan_is_unchanged": […]}`, the user says "yes" — replays with the structured evidence gone.

**The rescue round manufactures bug 1.** At `orchestrator.py:2085-2096`, when all 6 rounds are
spent, the model is re-invoked with `tools=` **removed**. It cannot act, only describe — precisely
the state the prompt's "never announce an edit you only described in prose" is fighting.

---

## 3. Architecture — a plain sync generator, no framework

The spec fixes the stack at line 5 — "OpenAI API (chat orchestration + embeddings)" — and titles
§8 "Chat Orchestration (OpenAI function calling)". No agent framework is listed, and none is
needed: **the state machine below is a `while` loop, an `if`, and two function calls.**

An earlier revision of this plan proposed LangGraph. Its own risk section was the argument against:
worker-thread node execution, `check_same_thread=False` silently becoming load-bearing,
`max_concurrency=1`, `copy_context()` for contextvar nesting, `recursion_limit` backstops, a
streaming smoke test to prove custom writes even arrive, and a forced major `langsmith` bump.
**Every one of those risks is framework-induced. None is the problem being solved.**

`backend/app/services/turn.py` exposes:

```python
def run_turn(orchestrator, user_message) -> Iterator[str]:  # yields SSE frames
```

— the same shape `_llm` delegates to today, so `stream()`, `chat.py`, and all seven monkeypatch
sites keep working verbatim. Tool rows are `yield`ed directly from inside the loop. No stream
writer, no custom stream mode, no thread pool.

```
        ┌────────────────────────────────────────────┐
        ▼                                            │
      agent ──tool_calls & tool_rounds < 6?──► tools ┘
        │
        └──no──► review ──needs_tools | rewrite──► agent   (review_rounds < 1)
                   │
                   └──ok / exhausted──► respond
```

- **`agent`** — one OpenAI call. Prose accumulated, not streamed.
- **`tools`** — executes through the existing `call_tool`; emits `tool` / `tool_done` live.
- **`review`** — §5.4. Deterministic first; LLM only when warranted.
- **`respond`** — emits the approved draft, persists it.

**Semantics and bounds (unchanged from the graph design):**

- `tool_rounds` capped at the existing `MAX_TOOL_ROUNDS = 6`; `review_rounds` capped at **1**.
  Two independent counters, so a chatty reviewer cannot starve the tool budget.
- `rewrite` **loops back to agent** — never falls through to respond. "The draft claims a change
  that didn't happen, and no further tool call is needed" is precisely bug 1's shape.
- `needs_tools` loops back to agent with the reviewer's `guidance` appended.
- On review exhaustion: strip the flagged sentences or fall back to a trace-rendered template.
  **Never ship the unverified draft.**
- On reviewer error (refusal / bad JSON / exception): fail **open**, log, proceed.
- At `tool_rounds >= MAX`: route through review with synthetic guidance ("out of tool rounds —
  report exactly what the trace shows, claim nothing more"), preserving the `test_chat.py:1979`
  non-empty-reply guarantee.

**`_rebind()` and turn setup stay in `stream()`** (`:1975-1985`). That constraint is about FastAPI
closing `yield` dependencies before the streaming body runs — not about any framework — and it
stands. It caused the "plan belonging to no thread" bug once already.

If flows ever become genuinely graph-shaped (parallel branches, durable mid-turn checkpoints),
revisit — behind the same `run_turn` interface, so it is a contained swap.

---

## 4. The missing mutators

Four gaps, in ascending cost. **Distinguishing "I live in X" from "put the trip in X" is the crux
of the reported bug** — the user said both, the system did neither, and the cheap non-destructive
one must not be routed through the destructive one.

| User says | Field | Today | New | Destructive? |
|---|---|---|---|---|
| "I live in Abu Dhabi" | `User.home_base_lat/lng` — written only by `seed.py:194` | no path | fold into `save_family_details` | no |
| "start the trip from Abu Dhabi" | `Itinerary.start_lat/lng` | no path | `set_origin(emirate)` | **no** — keeps every stop, re-routes and re-prices the origin legs |
| "make the trip be in Abu Dhabi" | `Itinerary.emirates_json` (`itinerary.py:263`) | no path | `replace_plan(emirates=…)` | yes — Dubai stops cannot exist in an Abu Dhabi trip |
| "drop day 3" | `Itinerary.num_days` (`itinerary.py:236`) | no path | `drop_day(day)` | partial — one day |

All four follow the existing `reschedule` template exactly (`itinerary.py:1531-1550`): mutate the
field → `context_for` → `load_plan` → build `travel_fn` → `repair_plan` → `persist_plan` →
`commit`. ~20 lines each.

Verified prerequisites, all already present:

- `persist_plan` (`:350`) reconciles slots by `row_id` and **deletes non-surviving rows**
  (`:415-417`), so re-solving onto an existing itinerary needs no changes there.
- `rebuild_segments`, `reflow_day`, `repair_plan` are already imported into `itinerary.py`.
- `MAX_DAYS` and `CheckConstraint("num_days <= 5")` cap only the *top*; nothing prevents shrinking.
- `recost_travel` (`:517`) recomputes `est_cost` from `distance_km` **already on the row** — so it
  is sufficient for a transport-mode change but **not** for `set_origin`, which moves the
  origin→first-stop distance. `set_origin` must call `rebuild_segments` per day.

`replace_plan` re-solves onto the **same** itinerary row via a new `into=` parameter on `generate`,
so the conversation link and event link survive — unlike `generate_itinerary`, which builds a
replacement and orphans the old one. It also consumes `starting_emirate_hint`.

### `drop_day` must ask about dates

Shift-vs-gap has no safe default when the dropped day is not the last one — a later day may be
anchored to the event's own date. Requirement:

- **Non-final day, and the request did not specify** → return a `needs_confirmation` result through
  the existing `_unapplied` two-turn protocol (`:1921`), asking *"shift the later days earlier, or
  leave the day free?"* Nothing is mutated.
- **Final day** → apply directly.
- Budget handling: the trip total shrinks; other days are untouched.

---

## 5. Enforcement tiers

### 5.1 T1 — deterministic interception (bug 2)

`policy.intercept(orchestrator, name, args) -> dict | None`, wired into `call_tool` at
`orchestrator.py:1246`, *before* the handler runs. A non-`None` return is a synthesized refusal;
the handler never executes.

> **Why interception, not tool removal.** Removing tools from the request based on DB state is a
> **permanent trap**: `conversation.rebuild_warned` is set in exactly one place —
> `orchestrator.py:1557`, *inside* the very handler the scoping would remove. Gate the tool on the
> flag and the flag becomes unreachable forever; the user could ask to start over every turn for
> the rest of time with no legal move existing.
>
> Removal is net-negative elsewhere too: `_edit_target` (`:1754`) returning "No plan has been
> generated yet, so there is nothing to edit" is a *teaching signal* steering the model toward
> `generate_itinerary`. Delete the tool and the model gets silence — trading bug 2 for bug 3.

Interception gives the same guarantee — the wrong tool **provably cannot mutate**, and the decision
is code not prompt — with none of the trapping. `TOOLS` stays a static module constant, so the
shape tests at `test_chat.py:76,103,110` keep passing. The two-stage rebuild gate (`:1548-1585`)
moves into `policy.intercept()` near-verbatim, `rebuild_warned` side effect included.

### 5.2 T1b — `tool_choice="required"` on round 0 (bug 3)

Interception can only *refuse* a tool; it cannot *force* one. One parameter does. Unless the turn
is classifiable as pure chitchat by a cheap deterministic check, the first completion must call
something. This removes most of "answered from model knowledge" before review is consulted.

### 5.3 T2 — per-call `applied` ledger (bug 1)

`applied = not result.get("error") and result.get("applied") is not False`, computed at the call
site.

> **Not a turn-wide `plan_changed` boolean.** Seven of the fourteen tools never touch an itinerary,
> so on any read-only turn a turn-wide flag is legitimately `false`, and a reviewer told "ground
> truth: nothing changed" manufactures unsupported-claim flags against a correct answer. It also
> misses family/event/preference mutations entirely, and `itinerary_payload` is a full DB re-render
> that `_unapplied` already calls a third time (`:1936`).

A plan fingerprint is kept only as a cross-check on the mutating tools, and kept cheap:
`(itinerary.id, itinerary.updated_at, tuple of slot row_id/place_id)` — not a payload render.

### 5.4 T2b + T3 — deterministic claim-check first, LLM reviewer only when warranted

The headline regression (§11) is *deterministically checkable*. Buffering every turn behind an LLM
call spends latency on the ~80% of turns that are read-only.

**Every turn — `policy.claim_check(draft, trace) -> list[str]`.** A pure function. Scans the draft
for change-claims (action verbs: added / created / changed / updated / moved / removed / replaced
/ rescheduled / dropped / applied …) and matches each against the per-call `applied` ledger.
Unit-tested in `test_policy.py` with canned drafts + traces. No LLM.

**The LLM reviewer runs only if** (a) the trace contains at least one mutating tool call, **or**
(b) `claim_check` found a change-claim it cannot match to an `applied == true` entry. Otherwise the
verdict is `ok` with no LLM call.

Consequence: read-only turns ("what's my plan's total?") stream immediately, zero added latency,
and the §11 adversarial check is satisfied by construction. It also means **the headline regression
is covered by the deterministic path even if the reviewer is disabled entirely.**

When it does run, T3 is as specced: OpenAI strict `json_schema` output — the pattern is proven in
this repo at `websearch.py:115-187`, including the `message.refusal` path at `:178` — seeing the
user message, the raw result dicts (~400 chars each), the per-call `applied` flags, and the draft.
Once per turn. Fail-open on error.

```json
{"verdict": "ok" | "needs_tools" | "rewrite",
 "unsupported_claims": ["…"], "missing_tools": ["…"], "guidance": "…"}
```

> **Never feed it `summarise_tool_result`.** That function is *deliberately lossy*: for **any**
> error it returns the literal string `"no change made"` (the comment at `:911-914` says outright
> the real error is withheld because those strings are addressed to the model), and for
> `_unapplied` results it returns `"needs your OK"`. The `applied: false` /
> `question_for_the_user` / `plan_is_unchanged` payload that `_unapplied` was purpose-built to
> produce would never reach the reviewer at all. It stays for SSE rows only.

**Buffering applies only to reviewed turns.** Unreviewed turns stream as they do today. If reviewed
turns read as a stall in practice, the pre-approved fallback is to stream the draft and emit a
correction frame — but measure first.

---

## 6. Also in scope

- **Remove the fabricated plan bubble, `ChatPanel.tsx:373-406`.** It renders assistant-styled prose
  ("Here's a N-day plan, budget …") derived from *client state* whenever `itinerary && !streaming`
  — after every turn once a plan exists, including turns whose real reply said the change did not
  go through. This is bug 1 reproduced client-side and invisible to any server-side check by
  definition; shipping the server fix while the client fabricates success defeats the purpose.

  **Precise fix:** strip the fabricated prose and the `bubble--assistant` / `msg__avatar` framing.
  **Keep `<DayChips />` and `<SuggestionChips />`** — they are useful affordances, not claims;
  re-present them as plain UI outside the message list. Add an assertion that no assistant-styled
  text is rendered that did not arrive over the stream.

- **Write `Message.tool_calls_json`** so the trace survives to the next turn. **Render it into
  `history()` as prose, never as `assistant.tool_calls`** — the OpenAI API hard-requires a
  `role:"tool"` message per `tool_call_id` immediately following; those rows do not exist, and the
  request would 400 rather than degrade. A test asserts no history entry ever contains a
  `tool_calls` key.

- **Replace the tools-stripped rescue round** (`:2085-2096`). Not simply delete it:
  `test_chat.py:1979` documents a real reported bug (three assistant messages saved as empty
  strings) and asserts the reply is non-empty. Preserved per §3.

- Delete the system-prompt paragraphs now enforced in code.

---

## 7. Deferred

Filed as issues so they do not evaporate; not fixed in this changeset.

- `find_stop` (`itinerary.py:857-861`) handles `"lunch"` and `"dinner"` but **not `"breakfast"`** —
  while the prompt (`orchestrator.py:1078`) explicitly tells the model to disambiguate with
  "breakfast". That path dead-ends in the identical ambiguity error, which the prompt has already
  forbidden retrying. Also `"lunch"` resolves via `min(start_time)`, which is the *breakfast*
  sitting when three exist. (This is about *resolving which* repeat the user means — the repeats
  themselves are intentional and must not be "fixed".)

Also not taken: Pydantic tool-argument models (a real improvement, not one of the three failure
modes); `planner.py` / `validator.py` changes; any intent-classifier.

---

## 8. Files

| File | Change |
|---|---|
| `backend/requirements.txt` | **untouched** |
| `backend/app/services/policy.py` | **new** — `intercept()`, `applied()`, `claim_check()`, `plan_fingerprint()`, chitchat check |
| `backend/app/services/reviewer.py` | **new** — strict-schema critic, fail-open |
| `backend/app/services/turn.py` | **new** — `run_turn()` sync generator, the §3 state machine |
| `backend/app/services/itinerary.py` | `into=` on `generate`; new `set_origin`, `drop_day` |
| `backend/app/services/orchestrator.py` | new tool schemas + handlers; `intercept` in `call_tool`; `_llm` → delegate; replace rescue round; trim prompt |
| `frontend/src/components/ChatPanel.tsx` | remove the fabricated bubble, keep the chips |
| `backend/app/routers/chat.py` | unchanged |

---

## 9. Build order

Each phase leaves `pytest` green before the next starts.

**Phase 1 — the missing mutators.** The four gaps from §4, including `drop_day`'s confirmation
path. **This alone fixes the reported transcript's user-visible symptom** and is independently
shippable.

**Phase 2 — `policy.py`.** `intercept()`, `applied()`, `claim_check()`, `plan_fingerprint()`,
chitchat check. Wire `intercept` into `call_tool` and move the rebuild gate into it. `claim_check`
lands here as a tested pure function; it is wired in at Phase 4. New `test_policy.py`. Existing
suite untouched — `TOOLS` stays static.

**Phase 3 — `reviewer.py`.** Strict schema, fail-open, `test_reviewer.py` stubbing OpenAI with the
`_FakeStream` pattern already at `test_chat.py:1956`. Not yet called by anything.

**Phase 4 — `turn.py` + conditional review + ChatPanel.** `run_turn` per §3; conditional review per
§5.4; buffering only on reviewed turns; the ChatPanel fix lands here since both change what the
user sees per turn. `_llm` becomes a delegate — `yield from run_turn(self, msg)`.

**Phase 5 — trace persistence.** Pass `tool_calls=` to `record()`; extend `history()` to append the
trace as prose. Keep the "history never contains `tool_calls`" test.

**Phase 6 — prompt reduction.** One paragraph per commit, deleting its paired assertion in the same
commit, so a regression is never confused with an intended deletion.

### Expected test breakage (out of 148 in `test_chat.py`)

| Cause | Broken |
|---|---|
| `_llm` kept as a delegate — all 7 monkeypatch sites, all 176 `call_tool` sites | **0** |
| `test_a_turn_that_spends_every_round_on_tools_still_answers` (`:1979`) — asserts on `fake.calls[-1]`, now sees a review step | **1** |
| Phase 6 prompt deletions — lines 314, 502, 842, 909, 1002, 1233, 1379, 1474, 1878, 2332, 2408 | **11** |

Six other `system_prompt()` assertion sites are context assertions and safe (`:308-310`, `:979`,
`:990-991`, `:997`). **Do not combine Phase 4 and Phase 6.**

---

## 10. Known risks

Dropping the framework voids the entire previous risk list — worker threads, cross-thread sessions,
`check_same_thread` becoming load-bearing, `max_concurrency`, `copy_context`, `recursion_limit`,
and the custom-stream smoke test are all gone. What remains:

- **Buffering on reviewed turns is a visible latency change.** Mitigated by §5.4 restricting it to
  turns that actually mutated or made a suspect claim. Pre-approved fallback: stream the draft and
  emit a correction frame — measure first.
- **`claim_check` is a verb-list heuristic.** It can miss a paraphrase. It is a *cheap first
  filter*, not the guarantee — the LLM reviewer still runs on every mutating turn regardless of
  what `claim_check` returns, so a miss costs nothing. Its false *positives* (flagging a correct
  answer) cost only an extra reviewer call.
- **Phase 6 is the only phase that should break tests.** If an earlier phase breaks
  `test_chat.py`, the delegate shim is wrong; fix the shim, not the tests.
- **The suite takes >10 minutes.** Budget for it in a six-phase plan gated on green tests.

---

## 11. Verification

**Automated.** `cd backend && .venv/bin/pytest` — 340 tests. New: `test_policy.py` (interception
verdicts and `claim_check`, pure), `test_reviewer.py` (verdicts against canned traces, stubbed
LLM), mutator cases in `test_itineraries.py`. `conftest.py:23-64` already blanks every billable
key, so no test spends money.

**End-to-end, replaying the reported bug.** `python -m app.seed`, then
`uvicorn app.main:app --reload` (:8000) and `npm run dev` (:5173); sign in as
`demo1@rihla.app` / `demo123`.

1. Plan a Dec 10 birthday. Confirm it lands in Dubai.
2. "We live in Abu Dhabi." → home base recorded, **origin** moved; the three Dubai stops **stay**,
   drive times and taxi fares re-cost. It must not claim the trip moved.
3. "Change the location of the plan to Abu Dhabi." → it says the region cannot change without
   replacing every stop, and asks. **It must not claim success.**
4. "Ok" → `replace_plan` fires; the right pane re-renders with Abu Dhabi stops.
5. "Drop day 2." (non-final) → returns `needs_confirmation` asking shift-or-gap; **nothing
   mutates**. Answer "shift them earlier" → applies, dates shift, budget re-totals. Separately,
   dropping the *final* day applies directly.
6. The chat's stop names match the pane exactly at every step.

**Added assertions.**

- A read-only turn produces **no** reviewer LLM call (assert on the stubbed client's call log) and
  streams without buffering.
- A turn whose draft claims a change with all `applied == false` is caught by `claim_check`
  **before** the LLM reviewer is consulted — the deterministic path covers the headline regression
  even with the reviewer disabled.
- No assistant-styled prose appears in the UI that did not arrive over the stream.

**Adversarial.** Trigger a `WindowOverrunRequired` confirmation, answer "yes" on the **next** turn,
and confirm the structured proposal survived via `tool_calls_json`. Kill the network mid-
`replace_plan` and confirm the reply says the change did not go through.

**Definition of done:** *whenever every call in a turn's trace has `applied == false`, the reply
contains no claim of change.*
