# Rihla — UAE Event-Based Itinerary Planner

Plans complete multi-day itineraries (max 5 days, UAE only) around your upcoming personal events,
with per-user family personalisation, long-term preference memory, budget tracking and
travel-time-aware scheduling.

**Core principle: the LLM never builds the itinerary.** It extracts intent from chat and calls
tools; a deterministic Python planner assembles and validates the schedule. The app keeps working
with the assistant, the maps API, or both switched off.

---

## Quick start

```bash
# 1. Backend
cd backend
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # set JWT_SECRET and OPENAI_API_KEY
.venv/bin/python -m app.seed  # 164 places, two demo accounts, their events + embeddings
.venv/bin/python -m uvicorn app.main:app --reload

# 2. Frontend, in a second terminal
cd frontend
npm install
cp .env.example .env          # optional: VITE_MAPTILER_KEY
npm run dev                   # http://localhost:5173
```

Sign in with either demo account — the login screen has one-tap buttons for both:

| Account | Password | Party | Plans skew toward |
|---|---|---|---|
| `demo1@rihla.app` | `demo123` | 2 adults, kids 7 and 13 | parks, aquariums, waterparks |
| `demo2@rihla.app` | `demo123` | 2 adults, no children | romantic evenings, fine dining |

The two accounts share nothing. demo2 never sees demo1's events, preferences or plans, and neither
account's likes influence the other's scoring.

## Environment

| Variable | Required | Absence means |
|---|---|---|
| `JWT_SECRET` | **yes** | A dev-only default is used; unsafe in production |
| `OPENAI_API_KEY` | to seed | `app.seed` exits non-zero rather than seeding a catalog without embeddings. At runtime, chat degrades to form-based intake and retrieval to SQL keyword scoring |
| `ORS_API_KEY` | no | Travel times fall back to haversine estimates, drawn as dashed `~35 min` segments |
| `LANGSMITH_API_KEY` | no | Tracing decorators become transparent no-ops |
| `VITE_MAPTILER_KEY` | no | The map falls back to OpenStreetMap tiles under the same palette filter |

## How it fits together

```
React + TypeScript (Vite)                    FastAPI
├─ ChatPanel      SSE via fetch/ReadableStream  ├─ orchestrator.py   OpenAI function calling
├─ MapView        react-leaflet + MapTiler      ├─ planner.py        deterministic, pure
├─ ItineraryStrip day tabs → slot cards         ├─ validator.py      constraint checker + repair
├─ SlotEditor     replace / adjust / remove     ├─ retrieval.py      Chroma → SQL filters
└─ BudgetPanel    live server-computed totals   ├─ travel.py         ORS → cache → haversine
                                                ├─ budget.py         age-tier pricing
                                                ├─ memory.py         per-user Chroma memory
                                                └─ repo.py           the scoping choke point
                                        SQLite (truth) + ChromaDB (semantic)
```

### Three things worth knowing

**`repo.py` is the only place user-owned tables are queried.** Every function takes `user_id` from
the auth dependency, cross-user access returns 404 rather than 403 (a 403 would confirm the row
exists), and asking it for a shared table raises `TypeError` — so misusing the choke point fails
loudly instead of silently returning everything.

**The planner is pure.** No DB session, no HTTP client, no clock; travel is injected as a callable.
That is what makes the hypothesis property tests — random party profiles, catalogs and budgets,
asserting the validator always passes — cheap enough to run on every commit.

**SSE is consumed with `fetch` + `ReadableStream`, not `EventSource`,** because `EventSource`
cannot send an `Authorization` header.

## Testing

```bash
cd backend && .venv/bin/python -m pytest        # 154 tests, ~50s
cd frontend && npm run build                    # tsc + vite, clean
```

Coverage includes the scheduler (overlaps, budget caps, opening hours, venues open past midnight,
age constraints, meal placement), the travel provider (cache hit, timeout → haversine fallback),
slot patching (neighbours' segments recomputed, whole-day revalidation, server-recomputed budget),
per-user isolation, and seven hypothesis properties.

## CLI: seeding one user's events

Separate from the global bootstrap, for demo prep or bulk-loading a calendar:

```bash
# From a JSON file
python -m app.seed_events --user demo2@rihla.app --file anniversary_events.json

# Or a single event inline
python -m app.seed_events --user demo1@rihla.app \
    --title "School winter break trip" --type family_visit \
    --date 2026-12-14 --notes "5 days, whole family"
```

Resolves `--user` by email and **exits non-zero before any write** if it does not exist — it never
creates users. Validates the whole batch before inserting the first row, so a bad entry halfway
down a file cannot leave a half-applied import. Idempotent on `(user_id, title, date)`, reporting
`inserted: N, skipped: M`.

## Known limits

- **The map is real geography, not the illustration.** MapTiler tiles are recoloured toward the
  design's cream and teal, and the pins, routes, chips and popovers are drawn to match — but tiles
  cannot reproduce hand-drawn road ribbons or lettering.
- **Prayer times are approximate.** A static monthly UAE table, accurate to roughly ±10 minutes,
  which is well inside the slack of a 20-minute gap. `services/prayer.py` documents the upgrade
  path to a live API.
- **Place photos are placeholders.** `image_url` is seeded `null` and cards render a per-category
  illustration, so a card can never show a broken image.
- **Re-running `app.seed` without `OPENAI_API_KEY`** is a no-op for the SQL rows but still exits
  non-zero at the embeddings step, by design.
