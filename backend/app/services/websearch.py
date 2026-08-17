"""Optional web search for live one-off events (spec §1.10, §8).

Deliberately narrow. The seeded catalog is the primary retrieval path; this exists only to notice
things happening on specific dates — a concert, a festival weekend — which no static catalog can
know about.

Three rules, all load-bearing:

1. **Raw search text never reaches scheduling.** Search results are pages, not events, so they are
   extracted into structured rows and then validated by `seed_events.validate_event` — the same
   function the CLI and the bootstrap use. Anything that fails validation, or falls outside the
   horizon, is dropped.

2. **Failure is silent.** No key, a timeout, a bad response, a rate limit: the caller gets an empty
   list and planning proceeds on seeded data alone.

3. **Extraction is not scheduling.** The LLM reads scraped page text and returns `{title, date}`
   rows. It never sees the itinerary, never picks places and never assigns times — the
   deterministic planner still does all of that. Regex over scraped chrome was tried first and
   produced titles like "ng interactive adventure" and "Held from 20"; pulling an event name out
   of concatenated page text needs comprehension, so without an OpenAI key this returns nothing
   rather than feeding the planner noise.

Results are normalised into `events`, never `places`. A place needs coordinates, opening hours and
prices to be schedulable, and inventing those from a search snippet would put fabricated figures
into a plan the user is asked to trust. An event carries a title and a date, which a page can
actually support.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from typing import Protocol

from ..config import settings
from ..models import EVENT_TYPES
from ..seed_events import SeedError, validate_event
from .tracing import traced, wrap_openai

log = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"
MAX_RESULTS = 8
TIMEOUT_SECONDS = 6.0
MAX_EVENTS = 12
# Enough of each page for the dates and their surrounding names; the rest is navigation chrome.
SNIPPET_CHARS = 1600


# --- search ------------------------------------------------------------------------------------


class WebSearchProvider(Protocol):
    def search(self, query: str, limit: int) -> list[dict]: ...


class TavilyProvider:
    """Tavily's search API. Any transport or shape error propagates for the caller to swallow."""

    name = "tavily"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def search(self, query: str, limit: int = MAX_RESULTS) -> list[dict]:
        import httpx

        response = httpx.post(
            TAVILY_URL,
            json={
                "api_key": self.api_key,
                "query": query,
                "max_results": limit,
                "search_depth": "basic",
            },
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return list(response.json().get("results") or [])


def default_provider() -> WebSearchProvider | None:
    return TavilyProvider(settings.web_search_api_key) if settings.web_search_api_key else None


# --- extraction --------------------------------------------------------------------------------


class EventExtractor(Protocol):
    def extract(self, pages: list[dict], today: date) -> list[dict]: ...


EXTRACTION_PROMPT = (
    "You are given text scraped from event-listing web pages. Extract the individual dated events "
    "you can see.\n\n"
    "Rules:\n"
    "- Only include an event if the page states a specific calendar date for it.\n"
    "- Resolve dates to YYYY-MM-DD. If a listing gives a day and month with no year, use the next "
    "occurrence on or after today.\n"
    "- The title must be the event's own name, not a page heading, a venue, a price, or a listicle "
    "title like 'Top 30 things to do'.\n"
    "- Skip anything you are unsure about. Returning fewer, correct events is better than guessing.\n"
    "- Do not invent events, dates, prices or venues.\n\n"
    'Reply with JSON: {"events": [{"title": str, "date": "YYYY-MM-DD", '
    f'"event_type": one of {list(EVENT_TYPES)}}}]}}'
)


class LLMExtractor:
    """Turns scraped page text into structured event rows. Extraction only — never scheduling."""

    name = "openai"

    def __init__(self, api_key: str, model: str | None = None) -> None:
        self.api_key = api_key
        self.model = model or settings.openai_chat_model

    @traced("websearch.extract", run_type="llm")
    def extract(self, pages: list[dict], today: date) -> list[dict]:
        from openai import OpenAI

        corpus = "\n\n".join(
            f"PAGE: {page.get('title', '')}\nURL: {page.get('url', '')}\n"
            f"{str(page.get('content') or '')[:SNIPPET_CHARS]}"
            for page in pages
        )
        if not corpus.strip():
            return []

        client = wrap_openai(OpenAI(api_key=self.api_key))
        response = client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            temperature=0,
            messages=[
                {"role": "system", "content": f"{EXTRACTION_PROMPT}\n\nToday is {today.isoformat()}."},
                {"role": "user", "content": corpus},
            ],
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        events = payload.get("events")
        return list(events) if isinstance(events, list) else []


def default_extractor() -> EventExtractor | None:
    return LLMExtractor(settings.openai_api_key) if settings.openai_api_key else None


# --- the adapter -------------------------------------------------------------------------------


def to_event_row(raw: dict, today: date, horizon: date, source: str | None = None) -> dict | None:
    """One extracted row → a validated event, or None. The last gate before anything is stored."""
    if not isinstance(raw, dict):
        return None

    event_type = str(raw.get("event_type") or "other")
    candidate = {
        "title": str(raw.get("title") or "").strip()[:200],
        # A search result is never trustworthy enough to classify beyond the allowed set.
        "event_type": event_type if event_type in EVENT_TYPES else "other",
        "date": raw.get("date"),
        "notes": source,
    }

    try:
        event = validate_event(candidate, 0)
    except SeedError as exc:
        log.info("dropped a web result that failed validation: %s", exc)
        return None

    # An event the planner cannot place on a calendar is worse than no event at all.
    if not today <= event["date"] <= horizon:
        return None
    return event


@traced("websearch.live_events", run_type="tool")
def find_live_events(
    query: str,
    *,
    provider: WebSearchProvider | None = None,
    extractor: EventExtractor | None = None,
    limit: int = MAX_RESULTS,
    horizon_days: int = 365,
) -> list[dict]:
    """Search for dated live events. Returns validated event rows, or [] on any failure."""
    provider = provider or default_provider()
    if provider is None:
        log.debug("web search is not configured; using seeded data only")
        return []

    extractor = extractor or default_extractor()
    if extractor is None:
        log.info("no extractor available for web search; using seeded data only")
        return []

    try:
        pages = provider.search(f"{query} UAE events", limit)
    except Exception as exc:  # noqa: BLE001 — an optional garnish must never break planning
        log.info("web search failed (%s); using seeded data only", exc)
        return []

    today = date.today()
    horizon = today + timedelta(days=horizon_days)
    source = ", ".join(dict.fromkeys(str(page.get("url") or "") for page in pages if page.get("url")))

    try:
        extracted = extractor.extract(pages, today)
    except Exception as exc:  # noqa: BLE001
        log.info("event extraction failed (%s); using seeded data only", exc)
        return []

    events: list[dict] = []
    seen: set[tuple[str, date]] = set()
    for raw in extracted:
        event = to_event_row(raw, today, horizon, source[:2000] or None)
        if event is None:
            continue
        key = (event["title"].lower(), event["date"])
        if key in seen:
            continue
        seen.add(key)
        events.append(event)
        if len(events) >= MAX_EVENTS:
            break

    events.sort(key=lambda event: event["date"])
    log.info("web search: %s pages → %s usable events", len(pages), len(events))
    return events
