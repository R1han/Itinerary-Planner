# Rihla — UAE Event-Based Itinerary Planner

Rihla plans complete itineraries (up to 5 days, UAE only) around your upcoming personal events —
a birthday, an anniversary, cousins visiting — with per-user family personalisation, long-term
preference memory, live budget tracking and travel-time-aware scheduling.

**Core principle: the LLM never builds the itinerary.** It reads intent from chat and calls tools;
a deterministic Python planner assembles and validates the schedule. Planning stays correct when
the maps API is down, and the numbers on screen always come from the server.

---

## Quick start

```bash
# 1. Backend
cd backend
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # set JWT_SECRET and OPENAI_API_KEY
.venv/bin/python -m app.seed  # 351 places, 2 demo accounts, their events + embeddings
.venv/bin/python -m uvicorn app.main:app --reload    # http://localhost:8000

# 2. Frontend, in a second terminal
cd frontend
npm install
cp .env.example .env          # optional: VITE_MAPTILER_KEY
npm run dev                   # http://localhost:5173
```

Sign in with either demo account — the login screen has one-tap buttons for both:

| Account | Password | Party | Plans skew toward |
|---|---|---|---|
| `demo1@rihla.app` | `demo123` | 2 adults, kids aged 7 and 13 | parks, aquariums, waterparks |
| `demo2@rihla.app` | `demo123` | 2 adults, no children | romantic evenings, fine dining |

The two accounts share nothing — no events, preferences, plans or scoring influence.

---

## Where to get the API keys

Only the first two matter. Everything else degrades to a documented fallback rather than failing.

| Key | Get it from | Free tier | Needed for |
|---|---|---|---|
| **`OPENAI_API_KEY`** | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | Pay-as-you-go, no free tier | The assistant, place embeddings at seed time |
| **`JWT_SECRET`** | Generate your own: `openssl rand -hex 32` | — | Signing login tokens |
| `ORS_API_KEY` | [openrouteservice.org/dev/#/signup](https://openrouteservice.org/dev/#/signup) | 2,000 requests/day | Real driving times and route geometry |
| `WEB_SEARCH_API_KEY` | [app.tavily.com](https://app.tavily.com) | 1,000 searches/month | Finding one-off live events (concerts, festivals) |
| `LANGSMITH_API_KEY` | [smith.langchain.com](https://smith.langchain.com) → Settings → API Keys | 5,000 traces/month | Tracing LLM and planner calls |
| `VITE_MAPTILER_KEY` | [cloud.maptiler.com/account/keys](https://cloud.maptiler.com/account/keys/) | 100,000 tiles/month | The map's cream-and-teal palette |

### What happens without each one

| Missing | Behaviour |
|---|---|
| `JWT_SECRET` | A dev-only default is used. **Unsafe in production.** |
| `OPENAI_API_KEY` | Seeding fails at the embeddings step. Chat has no scripted fallback — a failed call is reported as an error. |
| `ORS_API_KEY` | Travel times fall back to haversine estimates, drawn as dashed `~35 min` segments. |
| `WEB_SEARCH_API_KEY` | Live event lookup is skipped; seeded data only. |
| `LANGSMITH_API_KEY` | Every tracing decorator becomes a transparent no-op. |
| `VITE_MAPTILER_KEY` | The map falls back to OpenStreetMap tiles under the same CSS palette filter. |

---

## Environment variables

`backend/.env` — see `backend/.env.example`:

```bash
JWT_SECRET=              # required
JWT_EXPIRY_MINUTES=1440

OPENAI_API_KEY=          # required
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

ORS_API_KEY=             # optional
WEB_SEARCH_API_KEY=      # optional (Tavily)
LANGSMITH_API_KEY=       # optional

TAXI_AED_PER_KM=2.5      # fare model, see "Travel costs"
FUEL_AED_PER_KM=0.35
PARKING_AED_PER_STOP=15.0

CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

`frontend/.env`:

```bash
VITE_MAPTILER_KEY=       # optional
```

---

## System architecture

```mermaid
flowchart TB
    subgraph FE["Frontend — React + TypeScript (Vite)"]
        direction LR
        CP["ChatPanel<br/><i>the only way to plan</i>"]
        MV["MapView<br/><i>react-leaflet</i>"]
        IS["ItineraryStrip<br/><i>day tabs, slot cards</i>"]
        BP["BudgetPanel<br/><i>totals, transport mode</i>"]
        ZS["Zustand store<br/><i>server is truth</i>"]
    end

    subgraph BE["Backend — FastAPI"]
        RT["routers/<br/>auth · chat · itineraries · events · family"]
        ORCH["orchestrator.py<br/><i>OpenAI tool calling</i>"]
        RP["repo.py<br/><i>per-user scoping choke point</i>"]
        subgraph CORE["Planning core"]
            direction LR
            PL["planner.py<br/><i>pure, no I/O</i>"]
            VA["validator.py<br/><i>check + repair</i>"]
            RE["retrieval.py"]
            TV["travel.py"]
            BU["budget.py"]
            ME["memory.py"]
        end
    end

    subgraph DATA["Storage"]
        direction LR
        SQL[("SQLite<br/>source of truth")]
        CDB[("ChromaDB<br/>embeddings")]
    end

    subgraph EXT["External APIs"]
        direction LR
        OPENAI["OpenAI"]
        ORSVC["OpenRouteService"]
        TAV["Tavily"]
        MTL["MapTiler"]
    end

    CP -->|"POST /chat · SSE"| RT
    ZS -->|REST| RT
    RT --> ORCH
    RT --> RP
    ORCH --> CORE
    ORCH --> RP
    CORE --> RP
    RP --> SQL
    RE --> CDB
    ME --> CDB
    ORCH --> OPENAI
    RE --> OPENAI
    TV --> ORSVC
    ORCH --> TAV
    MV --> MTL
```

### How a planning request flows

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant C as ChatPanel
    participant O as Orchestrator
    participant P as Planner core
    participant D as SQLite

    U->>C: "plan Aisha's birthday, 4500 AED"
    C->>O: POST /chat
    O-->>C: conversation · thread id
    O-->>C: token · token · token …

    Note over O: model calls generate_itinerary
    O-->>C: tool · "Building your itinerary"

    O->>P: retrieve → score → cluster → assemble
    P->>P: route the chosen legs for real
    P->>P: validate + repair
    P->>D: persist slots and segments

    O-->>C: tool_done · "AED 2,629 of AED 4,500"
    O-->>C: itinerary_updated
    O-->>C: budget_updated
    C->>C: right pane redraws live
    O-->>C: done
```

### The planning pipeline

```mermaid
flowchart LR
    A["1 · intake gate<br/><i>family, budget, dates</i>"] --> B["2 · retrieval<br/><i>Chroma → SQL filters</i>"]
    B --> C["3 · scoring<br/><i>similarity + fit + preference</i>"]
    C --> D["4 · clustering<br/><i>one region per day</i>"]
    D --> E["5 · day assembly<br/><i>hours, travel, budget</i>"]
    E --> F["6 · real routing<br/><i>chosen legs only</i>"]
    F --> G["7 · validate + repair<br/><i>drop the worst offender</i>"]
```

### Why the LLM is kept out of planning

The model decides **when** to plan and **what was asked for**; it never decides times, prices,
places or totals. Every tool result is recomputed from persisted rows, so a reply cannot quote a
figure that disagrees with the budget bar next to it. Tool schemas use OpenAI **strict mode**, so
arguments are guaranteed to match their shape before a handler ever sees them.

---

## Database design

SQLite is the source of truth. ChromaDB holds only derived vectors — delete it and reseed and
nothing is lost.

```mermaid
erDiagram
    users ||--o{ family_members : "who is going"
    users ||--o{ preferences : "likes and dislikes"
    users ||--o{ events : "the calendar"
    users ||--o{ itineraries : owns
    users ||--o{ conversations : owns

    events |o--o{ itineraries : "planned by"
    events }o--o| places : "held at"

    itineraries ||--o{ slots : "one per stop"
    itineraries ||--o{ travel_segments : "one per leg"
    places ||--o{ slots : "booked as"

    conversations ||--o{ messages : "chat history"
    conversations }o--o| itineraries : "shows in right pane"
    conversations }o--o| events : "planning for"

    places ||--o{ travel_cache : "route endpoints"

    users {
        int id PK
        string email UK
        float home_base_lat
        float home_base_lng
        float default_budget
    }
    itineraries {
        int id PK
        int user_id FK
        int event_id FK "nullable"
        date start_date
        int num_days
        float total_budget
        string transport_mode "taxi or own_car"
    }
    slots {
        int id PK
        int itinerary_id FK
        int place_id FK
        int day_index
        int position
        string start_time
        string end_time
        json cost_breakdown_json
        bool locked
    }
    travel_segments {
        int id PK
        int itinerary_id FK
        int from_slot_id FK "null on the first leg"
        int to_slot_id FK
        float distance_km
        int duration_min
        float est_cost
        bool estimated "haversine fallback"
    }
    places {
        int id PK
        string name
        string emirate
        string category
        float price_adult
        float price_child
        int min_age
        json closed_months
        float kid_score
        float teen_score
        float romance_score
    }
    travel_cache {
        int from_place_id PK
        int to_place_id PK
        string mode PK
        float distance_km
        int duration_min
        json geometry_json
    }
```

### The tables

| Table | Holds | Notes |
|---|---|---|
| `users` | Account, home base coordinates, default currency and budget | Home base is the origin every day starts and ends from |
| `family_members` | One row per person: `role` (adult/child), `age` | Drives scoring weights, pricing tiers and age gates |
| `preferences` | `kind` (like/dislike), `subject`, `category`, `strength` | Written from chat *and* from slot edits |
| `events` | The occasion being planned: type, date, notes | `UNIQUE(user_id, title, date)` makes imports idempotent |
| `places` | 351 venues across all 7 emirates, 12 categories | Prices, opening hours, `closed_months`, kid/teen/romance scores |
| `itineraries` | One plan: dates, budget cap, `transport_mode`, origin | `status`, and a nullable link to the event it serves |
| `slots` | One stop: `day_index`, `position`, times, `cost_breakdown_json` | `locked` protects a slot from repair passes |
| `travel_segments` | One leg between slots: distance, duration, cost, geometry | `estimated` flags a haversine fallback, drawn dashed |
| `travel_cache` | Shared `(from, to, mode)` route lookups | See the note below |
| `conversations` | A chat thread, optionally bound to one itinerary and event | The thread owns what the right pane shows |
| `messages` | Chat history, plus `tool_calls_json` for the activity trace | Last 20 are replayed into the model's context |

### Three design decisions worth knowing

**`travel_cache` caches distance, never price.** Distance between two places is a property of the
road and is shared freely between all users. Price depends on who is travelling and how — a
6-person van costs 1.6× a saloon — so fares are computed at read time from the cached distance. A
cached price would let one party's fare leak into another's plan.

**Costs are stored, not recomputed on read.** `slots.cost_breakdown_json` and
`travel_segments.est_cost` are written when the plan is built, so a price change in the catalog
never silently rewrites a plan someone already saw. Changing transport mode re-prices the stored
legs explicitly, via `POST /itineraries/{id}/transport`.

**There are no migrations.** `create_all()` builds missing tables, and `db.py` carries a short
list of columns added after the fact, applied with an idempotent `ALTER TABLE`. That is a
deliberate trade for a single-column change; if the list grows or anything needs a data backfill,
bring in Alembic.

---

## The assistant's tools

Eleven, all strict-schema, none of which accepts a `user_id` — the model cannot address another
user however it is prompted.

| Tool | Does |
|---|---|
| `save_family_details` | Record who is in the family and what they like |
| `create_event` | Add an event to the calendar |
| `get_upcoming_events` | Look further ahead than the calendar already in context |
| `find_live_events` | Search the web for concerts and festivals — **read-only, saves nothing** |
| `generate_itinerary` | Build a plan. `focus` = `full_day` or `dinner_only`; `adults_only` for a couple's night |
| `get_itinerary` | Read a plan's current state before describing it |
| `make_day_cheaper` | Re-solve one day against a smaller budget; reports what it actually saved |
| `add_prayer_breaks` | Insert prayer breaks and reflow the day |
| `set_transport` | Switch between taxi fares and driving yourself, and re-price |
| `edit_stop` | Remove a stop, or move it to a different time |
| `record_preference` | Note a like or dislike mentioned in passing |

The user's calendar, family and preferences are **injected into the system prompt**, not fetched —
so the assistant never asks for a date it already has.

---

## How the planner decides

**Geography.** Days are clustered by farthest-point seeding so one day does not zig-zag between
emirates. Within a day, a stop more than **60 km** from the previous one is rejected outright — a
soft distance penalty ranks, but only a hard cap stops a thin candidate pool producing a dinner in
the next emirate.

**Meals are measured by detour, not distance.** A restaurant is scored on what it *adds* to the
journey you were already making: `d(here → restaurant) + d(restaurant → next) − d(here → next)`,
where "next" is the following attraction if one is still ahead, and home if none is. One on the
way costs zero.

**Restaurants may repeat, attractions may not.** A good restaurant on a road you are already
driving is worth returning to; seeing the same aquarium twice is not a trip. A repeat needs a
detour under 10 km, and loses ties to somewhere new.

**Travel costs** scale with party size, because five people need a bigger vehicle:

| Party | Vehicle | Multiplier |
|---|---|---|
| ≤ 4 | standard | ×1.0 |
| 5–6 | 6-seater | ×1.6 |
| 7+ | two vehicles | ×2.0 |

Taxi is `km × 2.50 × multiplier`. Own car is `km × 0.35 × multiplier + 15 per stop` for parking.

---

## API reference

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/auth/register`, `/auth/login` | Returns a JWT |
| `GET` | `/me` | Current user |
| `GET` `PUT` | `/family` | Read / replace the family |
| `GET` `POST` `DELETE` | `/preferences` | Likes and dislikes |
| `GET` `POST` `DELETE` | `/events` | The calendar |
| `POST` | `/chat` | SSE stream — the only way to plan |
| `GET` `POST` | `/conversations`, `/{id}/messages`, `/{id}/seen` | Threads |
| `POST` | `/itineraries/generate` | Build a plan directly |
| `GET` | `/itineraries/{id}` | The full plan the workspace renders |
| `GET` | `/itineraries/{id}/slots/{slot_id}/alternatives` | Three swaps that fit the window |
| `PATCH` | `/itineraries/{id}/slots/{slot_id}` | Replace / adjust / remove one slot |
| `POST` | `/itineraries/{id}/days/{n}/cheaper` | Re-solve one day cheaper |
| `POST` | `/itineraries/{id}/prayer-breaks` | Insert prayer breaks |
| `POST` | `/itineraries/{id}/transport` | `taxi` or `own_car`, re-prices in place |

Interactive docs at `http://localhost:8000/docs` while the server is running.

---

## Testing

```bash
cd backend && .venv/bin/python -m pytest    # 288 tests, ~70s (the model is stubbed)
cd frontend && npm run build                # tsc + vite
```

No test ever makes a billable API call. Coverage includes the scheduler (overlaps, budget caps,
opening hours, venues open past midnight, age gates, meal placement, the 60 km hop cap, detour
scoring, restaurant revisits), the travel provider (cache hit, timeout → haversine, fare tiers,
transport modes), slot patching, per-user isolation, the web search adapter, strict tool schemas,
and eight Hypothesis properties.

---

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

---

## Known limits

- **The map is real geography, not the illustration.** MapTiler's `basic-v2` tiles are recoloured
  toward the design's cream and teal, but tiles cannot reproduce hand-drawn road ribbons.
- **Seasonal closures are modelled, not flagged.** Seventeen venues carry `closed_months` — Global
  Village runs October to April — and the planner will not schedule into them.
- **Some coordinates are unroutable.** ORS returns 404 for a few beach and island points with no
  road access; those legs fall back to dashed estimates by design.
- **Prayer times are approximate.** A static monthly UAE table, accurate to roughly ±10 minutes,
  well inside the slack of a 20-minute gap. `services/prayer.py` documents the upgrade path.
- **Place photos are placeholders.** `image_url` seeds as `null` and cards render a per-category
  illustration, so a card can never show a broken image.
- **The trip home is not costed.** A day's travel total covers the legs between stops, not the
  drive back from the last one.
- **"The best restaurant" is not a concept.** Scoring weights cuisine preference above romance,
  and price only ever constrains — it never attracts — so "budget is no issue" does not make an
  expensive venue win.
- **Web search needs both keys.** Tavily finds the pages; pulling an event name out of scraped
  page chrome needs comprehension, so without `OPENAI_API_KEY` the adapter returns nothing rather
  than feeding the planner noise.
