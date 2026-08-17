# UAE Event-Based Itinerary Planner — Project Specification

A multi-user application that plans complete multi-day itineraries (max 5 days, UAE only) around upcoming personal events, with per-user family personalization, long-term preference memory, budget tracking, and travel-time-aware scheduling. Every user has their own family composition, preferences, events, and itineraries; the places catalog and travel cache are shared.

**Stack (fixed):** React + TypeScript (Vite) frontend · Python FastAPI backend · SQLite (source of truth) · ChromaDB embedded (semantic retrieval) · OpenAI API (chat orchestration + embeddings) · OpenRouteService (travel times/routes) with haversine fallback · react-leaflet + OpenStreetMap (map).

---

## 1. Product Requirements

1. Plan a complete itinerary for an upcoming event: at most **5 days**, **UAE only**.
2. Conversational intake collects: number of adults, number of children, children's ages, likes/dislikes, budget, dates, start location. All fields required before generation (validated server-side, not only by the LLM).
3. **Long-term memory** of family composition and preferences, reused in future planning. Memory is strictly per-user: user A's likes/dislikes, family, and edit history must never influence or be visible in user B's plans or chat.
4. **Events**: insertable manually (CRUD UI + API) and via chat. Asking "What events are upcoming?" returns upcoming events; for any unplanned upcoming event, the system offers: "Want me to plan an itinerary for {event}?"
5. Planner accounts for **travel time between places** and the user's **budget constraint**.
6. UI shows the itinerary as a **strip**: one card (slot) per plan, per day. The user can **edit a single slot** in any day (replace / adjust time / remove) without regenerating the rest. Edits that reject a place/category are recorded as preferences (`source=slot_edit`) and confirmed with the user before being treated as strong signals.
7. **Budget split-up visible at every stage**: per-slot cost breakdown (adults / children / travel), per-day totals, trip total vs cap, live-updating on every change.
8. Each slot shows its **time range**; travel time between consecutive slots is displayed and enforced before the next slot may start.
9. **Age & occasion adaptation**: young kids → parks/zoos/aquariums, shorter slots, midday rest; teens/young adults → waterparks, theme parks, adventure; anniversary (adults only) → romantic evening-heavy plan, fine dining.
10. Web search is an optional secondary path for live one-off events only; primary retrieval is embeddings over seeded data.
11. **Foolproof**: the system must never produce an invalid plan and must keep working when the LLM, web search, or maps API fails.

## 2. Locked Decisions

- LLM: **OpenAI API** (chat + `text-embedding-3-small` for Chroma).
- **Multi-user with authentication.** Email + password (bcrypt via passlib) issuing JWT access tokens (`python-jose`); all API routes except `/auth/*` require `Authorization: Bearer <token>`. The authenticated `user_id` is always taken from the token — never from request bodies or query params. A FastAPI dependency (`get_current_user`) injects it into every router; every DB query on user-owned tables filters by it.
- **Ownership model:** per-user tables — `family_members`, `preferences`, `events`, `itineraries` (and their `slots`/`travel_segments`). Shared/global tables — `places`, `travel_cache` (place-to-place travel is user-agnostic, so all users benefit from one cache).
- Travel times: **real maps API** — OpenRouteService (free tier), behind a `TravelTimeProvider` interface, with SQLite caching and silent haversine fallback (segments marked `estimated=true`, rendered as "~35 min" and dashed polylines).
- Chroma runs **embedded** (`chromadb.PersistentClient`) inside the FastAPI process.
- UX reference: **Mindtrip** split-view (chat left, map + strip + budget right). Out of scope: booking integrations, group collaboration, screenshot-to-itinerary.

## 3. Architecture

```
┌─────────────────────────────────────────────────┐
│  React + TypeScript (Vite)                       │
│  ├─ ChatPanel (left pane, ~38%, SSE streaming)   │
│  ├─ MapView (react-leaflet: pins + polylines)    │
│  ├─ ItineraryStrip (day tabs → slot cards)       │
│  ├─ SlotEditor (replace/adjust/remove one slot)  │
│  └─ BudgetPanel (pinned bar, live split-up)      │
└───────────────┬─────────────────────────────────┘
                │ REST + SSE
┌───────────────▼─────────────────────────────────┐
│  FastAPI                                         │
│  ├─ Chat orchestrator (OpenAI function calling)  │
│  ├─ Planner engine (deterministic, pure Python)  │
│  ├─ Validator (constraint checker)               │
│  ├─ Retrieval (Chroma query + SQL filters)       │
│  ├─ TravelTimeProvider (ORS → cache → haversine) │
│  ├─ Budget allocator (server-side recompute)     │
│  ├─ Memory service (preferences, Chroma prefs)   │
│  └─ Web search adapter (optional, normalizing)   │
└─────┬──────────────────────────┬────────────────┘
 ┌────▼─────┐              ┌─────▼──────┐
 │  SQLite   │              │  ChromaDB  │
 │  (truth)  │              │ (semantic) │
 └───────────┘              └────────────┘
```

**Core principle: the LLM never builds the itinerary.** It extracts intent and preferences from chat and calls tools; a deterministic Python planner assembles and validates the schedule. The LLM supplies inputs and phrasing only. The app must fully function with the LLM down (form-based intake fallback + rule-based planning).

## 4. SQLite Schema

```sql
users(id, email UNIQUE, password_hash, name, home_base_lat, home_base_lng,
      default_currency, default_budget, created_at)
family_members(id, user_id REFERENCES users(id), role TEXT CHECK(role IN ('adult','child')), age, name)
preferences(id, user_id, kind TEXT CHECK(kind IN ('like','dislike')), subject,
            category, source TEXT CHECK(source IN ('stated','slot_edit')),
            strength REAL, created_at)
events(id, user_id, title, event_type, date, notes, planned BOOLEAN)
places(id, name, emirate, lat, lng, category, price_adult, price_child,
       min_age, open_time, close_time, avg_duration_min, tags,          -- tags: JSON array
       kid_score REAL, teen_score REAL, romance_score REAL,
       image_url, category_icon, description)
itineraries(id, event_id, user_id, start_date, num_days INTEGER CHECK(num_days <= 5),
            total_budget, currency, status)
slots(id, itinerary_id, day_index, position, place_id, start_time, end_time,
      cost_breakdown_json, locked BOOLEAN)
travel_segments(id, itinerary_id, from_slot_id, to_slot_id, distance_km,
                duration_min, mode, est_cost, estimated BOOLEAN, geometry_json)
travel_cache(from_place_id, to_place_id, mode, distance_km, duration_min,
             est_cost, geometry_json, provider, fetched_at)
```

All `user_id` columns are foreign keys to `users(id)`. `preferences`, `events`, `itineraries` carry `user_id` directly; `slots` and `travel_segments` inherit ownership through their itinerary. Enforce scoping in one place: a repository/query layer where every read/write on user-owned tables takes `user_id` from the auth dependency — no route handler ever queries these tables without it. Cross-user access attempts return 404 (not 403, to avoid leaking existence).

Boundary validation (Pydantic): reject >5 days, past dates, coordinates outside the UAE bounding box, negative budgets.

## 5. ChromaDB Collections

1. `places` — embedding of `name + description + tags` per place. Enables semantic candidate retrieval ("my 6-year-old loves animals" → zoos/aquariums), followed by SQL hard filters (age, price, emirate, hours).
2. `preference_memory` — embeddings of stated/inferred preference sentences for cross-session retrieval. Every document stores `{"user_id": <id>}` in its metadata, and **every query must pass `where={"user_id": current_user.id}`** — wrap Chroma access in the memory service so an unfiltered query is impossible to write from a route. The shared `places` collection needs no user filter.

Embeddings built by the seed script using `text-embedding-3-small`.

## 6. Planner Engine (deterministic core)

Scoring: `score(place) = w_semantic·chroma_similarity + w_profile·profile_fit + w_pref·preference_boost − dislike_penalty`. Weights derive from the party profile, never from the LLM.

**Party profile** from `family_members` + `events.event_type`:
- Kids under ~8 present → weight `kid_score`; cap slot durations (~2h); enforce a midday rest gap; prefer parks/zoos/soft-play.
- Teens/young adults → weight `teen_score` (waterparks, theme parks, desert safari, ziplines).
- `event_type=anniversary`, adults only → weight `romance_score`; evening-heavy; fine dining.
- Mixed party → a slot is valid only if **every attendee** clears `min_age`; days mix weighted categories.

**Generation pipeline:**
1. Constraint intake (validated).
2. Candidate retrieval: Chroma semantic query → SQL hard filters → scored shortlist (~40–60 places).
3. Geographic clustering: group candidates by proximity (k-means or greedy on lat/lng) and assign clusters to days, so no day zig-zags across emirates.
4. Day assembly (greedy): pick highest-scored feasible place; compute travel from previous slot via TravelTimeProvider; check `arrival ≥ open_time`, `departure ≤ close_time`, running budget ≤ envelope; insert travel segment; repeat. Meal slots are first-class slots (dining categories at meal windows).
5. Budget allocation: per-day envelope (total ÷ days, adjustable); each slot writes `cost_breakdown_json` like `{"adults":[199,199],"children":[155],"travel_in":38,"total":591}`. Totals are **always recomputed server-side**, never trusted from the client.
6. **Validation pass** (the foolproof guarantee) asserts: no slot overlaps; travel time honored between consecutive slots; budget ≤ cap; venues open during their slots; all age constraints met. On failure: repair (drop lowest-score slot), re-validate. Runs on generation **and on every manual edit**.

**Single-slot edit:** `PATCH /itineraries/{id}/slots/{slot_id}` locks all other slots, re-solves only the gap (time window between neighbors including both recomputed travel segments, remaining budget), re-runs the validator on the whole day, and returns the **full updated day + budget** — the client never patches locally. Replace flow offers 3 alternatives that fit the exact window and budget.

## 7. Travel Time

- Primary: OpenRouteService driving directions (duration, distance, route geometry).
- Cache every `(from_place, to_place, mode)` in `travel_cache` — the place set is finite, so demos become cache hits.
- Timeout budget 2s; on error/timeout fall back to haversine × 1.3 road factor at 45 km/h intra-city / 90 km/h inter-emirate + 10 min parking buffer; mark `estimated=true`.
- Travel cost estimate: taxi ≈ AED 2.5/km; shown as its own budget line.
- Route geometry stored per segment (`geometry_json`, encoded polyline) and drawn on the map; haversine fallback draws a dashed straight line.

## 8. Chat Orchestration (OpenAI function calling)

The orchestrator runs in the context of the authenticated user: it is constructed with `current_user`, loads that user's family, preferences, and preference-memory (Chroma, user-filtered) into its system context, and every tool implementation writes/reads only that user's rows. Tool schemas never expose a `user_id` parameter — the LLM cannot address another user even if prompted to.

Tools:
- `save_family_details(adults, children[ages], likes, dislikes)` → `family_members` + `preferences`
- `create_event(title, type, date, notes)`
- `get_upcoming_events(horizon_days)` → powers "What events are upcoming?"; response offers to plan for unplanned events
- `generate_itinerary(event_id, days, budget)` — server rejects if intake checklist incomplete
- `record_preference(kind, subject)` — fired whenever a like/dislike surfaces mid-conversation

SSE stream emits typed events: `{type: "token" | "itinerary_updated" | "budget_updated", data}` so the right pane re-renders live while the assistant is still responding.

Failure behavior: OpenAI call fails → degrade to form-based intake; planner runs rule-based. Web search fails → seeded data only, silently. Anything found via web search is normalized into `places`/`events` rows before the planner sees it — raw search text never reaches scheduling.

## 9. Frontend

Layout (Mindtrip-style split view):

```
┌────────────────┬──────────────────────────────────┐
│                │  MapView (pins, route polylines)  │
│   ChatPanel    ├──────────────────────────────────┤
│   (~38%)       │  ItineraryStrip (day tabs→cards)  │
│                ├──────────────────────────────────┤
│                │  BudgetPanel (pinned bottom bar)  │
└────────────────┴──────────────────────────────────┘
```

- State: Zustand store with `selectedDay`, `hoveredSlotId`, `selectedSlotId` shared by strip and map.
- SlotCard: time range, place name, thumbnail (`image_url` with `onError` → bundled local category illustration; cards never show broken images), per-person cost chips. Hover highlights the map pin and vice versa; click pans/zooms.
- TravelConnector between cards: "🚗 35 min · AED 40" ("~35 min" when estimated).
- Map: numbered pins per day, day tab filters pins, per-day route polyline; pin popup reuses the slot thumbnail + name + time + cost.
- SlotEditor: Replace (3 fitting alternatives) / Adjust time / Remove; on replace/remove show a one-tap "Avoid this type in future?" → `preferences(source=slot_edit)`.
- BudgetPanel: total, per-day bars, category split (activities / food / travel), remaining vs cap, live on every mutation.
- Mobile: chat collapses to a bottom sheet over map/strip.

Component tree:
```
AppShell.tsx, ChatPanel.tsx,
MapView.tsx → SlotMarker.tsx,
ItineraryStrip.tsx → SlotCard.tsx, TravelConnector.tsx, SlotEditor.tsx,
BudgetPanel.tsx
```

`GET /itineraries/{id}` includes `geometry_json` per segment and `image_url` per slot's place — no extra round trips to render.

## 10. Seed Data (`seed.py` — idempotent)

Data lives in `places.json` + `events.json` checked into the repo; loaded on startup when tables are empty; Chroma embeddings built in the same pass. Prices/hours are plausible-realistic fake data (real UAE coordinates so the maps API returns genuine travel times).

**Places: ~150 rows.** Distribution:

| Category | ~Count | Notes |
|---|---|---|
| parks / playgrounds | 20 | high kid_score, mostly ≤ AED 20 |
| waterparks / theme parks | 12 | high teen_score, expensive, long duration |
| museums / culture | 18 | mid scores |
| aquariums / zoos / wildlife | 10 | high kid_score, indoor flags |
| beaches / outdoor free | 15 | budget relief valves |
| adventure (zipline, safari, kayak) | 15 | min_age ≥ 8–12 |
| casual dining | 25 | 2–3 meal slots needed per day |
| fine dining / romantic | 15 | romance_score ≥ 0.8, evening hours |
| malls / entertainment | 10 | midday heat fallback |
| shows / cruises | 10 | evening, fixed showtimes |

Spread: ~45% Dubai, ~30% Abu Dhabi, remainder across Sharjah, RAK, Fujairah, Ajman, UAQ. Every category must include budget options so tight caps are satisfiable by substitution, not failure.

Representative rows:

```json
[
  {"name":"Yas Waterworld","emirate":"Abu Dhabi","lat":24.4887,"lng":54.5995,
   "category":"waterpark","price_adult":295,"price_child":250,"min_age":4,
   "open_time":"10:00","close_time":"19:00","avg_duration_min":300,
   "kid_score":0.7,"teen_score":0.95,"romance_score":0.1,
   "tags":["thrill","water","outdoor","full-day"],
   "description":"Large waterpark with 40+ rides from kids' splash zones to high-adrenaline slides. Best for teens and older kids."},
  {"name":"Umm Al Emarat Park","emirate":"Abu Dhabi","lat":24.4643,"lng":54.3773,
   "category":"park","price_adult":10,"price_child":5,"min_age":0,
   "open_time":"08:00","close_time":"22:00","avg_duration_min":120,
   "kid_score":0.95,"teen_score":0.3,"romance_score":0.4,
   "tags":["outdoor","playground","animal-barn","picnic","budget"],
   "description":"Family park with shaded playgrounds, a small animal barn, and botanic garden. Ideal for children under 10; very low cost."},
  {"name":"Pierchic Reflection","emirate":"Dubai","lat":25.1318,"lng":55.1841,
   "category":"fine_dining","price_adult":550,"price_child":0,"min_age":12,
   "open_time":"18:00","close_time":"23:30","avg_duration_min":120,
   "kid_score":0.0,"teen_score":0.2,"romance_score":0.98,
   "tags":["romantic","seafood","overwater","sunset","reservation"],
   "description":"Overwater fine-dining restaurant with sunset views. Signature anniversary venue."},
  {"name":"Dubai Aquarium & Underwater Zoo","emirate":"Dubai","lat":25.1975,"lng":55.2790,
   "category":"aquarium","price_adult":199,"price_child":155,"min_age":0,
   "open_time":"10:00","close_time":"22:00","avg_duration_min":90,
   "kid_score":0.9,"teen_score":0.6,"romance_score":0.3,
   "tags":["indoor","animals","air-conditioned","mall-adjacent"],
   "description":"Giant mall aquarium with tunnel walk-through. Indoor — good midday option for young kids."},
  {"name":"Jebel Jais Zipline","emirate":"Ras Al Khaimah","lat":25.9339,"lng":56.1273,
   "category":"adventure","price_adult":399,"price_child":399,"min_age":12,
   "open_time":"08:00","close_time":"17:00","avg_duration_min":180,
   "kid_score":0.1,"teen_score":0.98,"romance_score":0.35,
   "tags":["thrill","mountain","outdoor","booking-required","weight-limits"],
   "description":"World's longest zipline. Strict age/weight limits; teens and adults only."},
  {"name":"Al Fanar Cafeteria","emirate":"Dubai","lat":25.2211,"lng":55.2540,
   "category":"casual_dining","price_adult":85,"price_child":45,"min_age":0,
   "open_time":"08:00","close_time":"23:00","avg_duration_min":60,
   "kid_score":0.7,"teen_score":0.6,"romance_score":0.3,
   "tags":["emirati-cuisine","family","budget","lunch"],
   "description":"Traditional Emirati restaurant, relaxed and family-friendly, mid prices."}
]
```

**Seed events** (cover the three demo arcs — young-kid birthday, anniversary, teen visit):

```json
[
  {"title":"Aisha's 7th birthday","event_type":"birthday","date":"2026-08-29",
   "notes":"loves animals, afraid of loud rides","planned":false},
  {"title":"Wedding anniversary","event_type":"anniversary","date":"2026-09-14",
   "notes":"dinner, just the two of us","planned":false},
  {"title":"Cousins visiting from India","event_type":"family_visit","date":"2026-10-02",
   "notes":"4 days, teens aged 14 and 16","planned":false}
]
```

**Seed users: two demo accounts**, so per-user isolation and personalization divergence are demonstrable from the first run:

- `demo1@rihla.app` / `demo123` — the mixed-age family below (2 adults, kids 7 and 13), owns the three seed events, likes animals/waterparks, dislikes loud thrill rides. Plans should skew parks/aquarium/waterpark.
- `demo2@rihla.app` / `demo123` — a couple (2 adults, no children), one seed event (`"Anniversary weekend", event_type=anniversary`), preferences: like fine dining and beaches, dislike theme parks. Plans should skew romantic evenings; asking this account "what events are upcoming?" must never return demo1's events.

**Seed family for demo1** (mixed ages on purpose — exercises per-slot age fit):

```json
{"family_members":[
   {"role":"adult","age":34,"name":"Dad"},{"role":"adult","age":31,"name":"Mom"},
   {"role":"child","age":7,"name":"Aisha"},{"role":"child","age":13,"name":"Omar"}],
 "preferences":[
   {"kind":"like","subject":"animals and zoos","category":"aquarium","source":"stated","strength":0.9},
   {"kind":"like","subject":"waterparks","category":"waterpark","source":"stated","strength":0.8},
   {"kind":"dislike","subject":"long queues","category":null,"source":"stated","strength":0.6},
   {"kind":"dislike","subject":"very loud thrill rides","category":"adventure","source":"slot_edit","strength":0.7}]}
```

`travel_cache` is populated lazily by real API calls, not seeded.

**Per-user event seeding — `seed_events.py` (CLI).** Separate from the global bootstrap: inserts events for one specific user, usable any number of times after initial setup (demo prep, testing a fresh account, bulk-loading a user's calendar).

```
# From a JSON file:
python -m app.seed_events --user demo2@rihla.app --file anniversary_events.json

# Or a single event inline:
python -m app.seed_events --user demo1@rihla.app \
    --title "School winter break trip" --type family_visit \
    --date 2026-12-14 --notes "5 days, whole family"
```

Behavior requirements:
- `--user` takes an email; the script resolves it to `user_id` and **fails loudly if the user doesn't exist** — it never creates users implicitly.
- JSON file format is a list of objects matching the `events` schema minus `id`/`user_id`/`planned` (`title`, `event_type`, `date`, optional `notes`); `planned` defaults to `false`.
- Validates `event_type` against the allowed set and `date` format; warns (but doesn't fail) on past dates.
- **Idempotent per event**: skips any row where `(user_id, title, date)` already exists, reports `inserted: N, skipped: M` on exit.
- Goes through the same SQLAlchemy models/session as the app (no raw SQL), so constraints and FKs are exercised.
- `seed.py` (global bootstrap) reuses this module internally to attach the demo accounts' seed events, so there's one code path for event insertion.

## 11. API Endpoints

```
POST  /auth/register                         # email, password, name → JWT
POST  /auth/login                            # → JWT access token
GET   /me                PATCH /me           # profile + settings (home base, currency, default budget)

# All routes below require Authorization: Bearer <token>;
# every one is implicitly scoped to the authenticated user.
POST  /chat                                  # SSE stream, typed events
GET   /events            POST /events        # CRUD (own events only)
GET   /events/upcoming?horizon_days=30
PATCH /events/{id}       DELETE /events/{id}
GET   /family            PUT /family         # family_members for current user
POST  /itineraries/generate                  # validated intake required
GET   /itineraries                           # list own itineraries
GET   /itineraries/{id}                      # full: days, slots, segments, budget (404 if not owner)
PATCH /itineraries/{id}/slots/{slot_id}      # replace/adjust/remove one slot
GET   /itineraries/{id}/slots/{slot_id}/alternatives   # 3 fitting options
GET   /preferences       POST /preferences   # incl. slot_edit confirmations
```

Frontend: token stored in memory + refresh-on-load pattern (or localStorage for demo simplicity), attached by `api/client.ts`; add `LoginPage.tsx` / `RegisterPage.tsx` and a route guard around the workspace; the top bar shows the signed-in user's name and a sign-out action.

## 12. Repo Structure

```
itinerary-planner/
├─ backend/
│  ├─ app/
│  │  ├─ main.py            # FastAPI app, CORS, routers
│  │  ├─ config.py          # env: OPENAI_API_KEY, ORS_API_KEY
│  │  ├─ db.py              # SQLite engine/session
│  │  ├─ models.py          # SQLAlchemy models
│  │  ├─ schemas.py         # Pydantic
│  │  ├─ auth.py            # JWT creation/verification, get_current_user dependency
│  │  ├─ routers/  auth.py chat.py events.py family.py itineraries.py preferences.py
│  │  ├─ services/ orchestrator.py planner.py validator.py retrieval.py
│  │  │            travel.py budget.py memory.py
│  │  ├─ data/     places.json events.json
│  │  ├─ seed.py            # global bootstrap: places, demo users, embeddings
│  │  └─ seed_events.py     # CLI: insert events for a specific user
│  └─ tests/  test_planner.py test_validator.py
├─ frontend/
│  └─ src/
│     ├─ api/client.ts      # attaches bearer token
│     ├─ pages/ LoginPage.tsx RegisterPage.tsx
│     ├─ components/ (tree in §9)
│     ├─ state/store.ts
│     └─ types.ts           # mirrors Pydantic schemas
└─ docker-compose.yml       # optional
```

## 13. Testing & Acceptance Criteria

Tests:
- Unit tests on the scheduler: overlaps, budget caps, opening-hours edge cases (Friday hours, prayer-time closures), age constraints, meal-window placement.
- Property-based tests (hypothesis): random party profiles + budgets → validator always passes on planner output.
- Travel provider: cache hit path, timeout → haversine fallback marked estimated.
- Slot patch: neighbors' travel segments recomputed; whole-day revalidation; server-recomputed budget.
- **Isolation tests:** user B requesting user A's itinerary/event/preference IDs gets 404; Chroma preference queries return only the querying user's documents; chat tools invoked as user B never read or write user A's rows; `/events/upcoming` returns only own events.

Acceptance checklist:
1. Insert "daughter's birthday" via chat or UI → "What events are upcoming?" returns it and offers to plan.
2. Intake refuses to generate until adults/children/ages/budget/dates/start location are known.
3. Generated 3-day plan: no overlaps, every consecutive pair separated by ≥ travel time, total ≤ budget, all venues open during slots, all attendees clear min_age.
4. Aisha's-birthday plan skews parks/aquarium; anniversary plan is adults-only romantic evening; teen-visit plan includes waterpark/zipline days.
5. Replacing one slot changes only that slot + its two travel segments; budget panel updates; "avoid in future?" prompt appears and persists a preference.
6. Kill the OpenAI key → form intake + rule-based planning still produce valid plans. Kill ORS → dashed estimated segments, planning still works.
7. Re-running `seed.py` is a no-op on a populated DB.
8. Sign in as demo1 → animal/waterpark-skewed family plans and demo1's three events; sign out, sign in as demo2 → romantic adults-only plans, only the anniversary event, and none of demo1's preferences leaking into scoring. Unauthenticated requests to any non-auth route return 401.
9. Running `seed_events.py --user demo2@rihla.app --file f.json` makes those events appear in demo2's "upcoming events" (chat and API) and nowhere in demo1's; re-running the same command reports all rows skipped and inserts nothing; running it with an unknown email exits non-zero without touching the DB.

## 14. Build Order

1. Schema + auth (register/login, JWT, `get_current_user`) + seed data (two demo users) + events CRUD + upcoming-events endpoint — auth comes first so every subsequent router is written user-scoped from day one, not retrofitted
2. Planner engine + validator (pure Python, tests first — the testable core)
3. Travel provider (ORS + cache + fallback) + budget allocator
4. Strip UI + map + budget panel (read-only)
5. Slot editing + preference capture
6. Chat orchestration (OpenAI tools, SSE) + Chroma memory
7. Web search adapter (optional garnish, last)

Env vars: `OPENAI_API_KEY`, `ORS_API_KEY` (both optional at runtime — absence triggers the documented fallbacks), `JWT_SECRET` (required), `JWT_EXPIRY_MINUTES` (default 1440).
