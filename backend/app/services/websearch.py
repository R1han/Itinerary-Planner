"""Optional web search for live one-off events (spec §1.10, §8).

Deliberately narrow. The seeded catalog is the primary retrieval path; this exists only to notice
things that are happening on specific dates — a concert, a festival weekend — which no static
catalog can know about.

Two rules, both load-bearing:

1. **Raw search text never reaches scheduling.** Every result is normalised into an `events` row
   and validated by `seed_events.validate_event` — the same function the CLI and the bootstrap
   use — before anything downstream sees it. A result that does not survive validation is dropped.

2. **Failure is silent.** No key, a timeout, a bad response, a rate limit: the caller gets an
   empty list and the planner proceeds on seeded data alone.

Results are NOT normalised into `places`. A place needs coordinates, opening hours and prices to be
schedulable, and inventing those from a search snippet would put fabricated data into a plan the
user is asked to trust. Events carry only a title, a type and a date, all of which a search result
can actually support.
"""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from typing import Protocol

from ..config import settings
from ..seed_events import SeedError, validate_event
from .tracing import traced

log = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"
MAX_RESULTS = 8
TIMEOUT_SECONDS = 4.0

# Loose ISO-date match; anything else is not specific enough to plan around.
_DATE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")


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
    key = getattr(settings, "web_search_api_key", None)
    return TavilyProvider(key) if key else None


def normalise(result: dict, *, horizon_days: int = 365) -> dict | None:
    """One search result → a validated event row, or None if it cannot be trusted.

    Anything without a parseable future date inside the horizon is dropped: an "event" the planner
    cannot place on a calendar is worse than no event at all.
    """
    title = str(result.get("title") or "").strip()
    if not title:
        return None
    # Search titles are usually "Thing — Site Name"; keep the thing.
    title = re.split(r"\s+[|–—]\s+", title)[0].strip()[:200]

    haystack = f"{result.get('title', '')} {result.get('content', '')}"
    match = _DATE.search(haystack)
    if not match:
        return None

    try:
        when = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None

    today = date.today()
    if not today <= when <= today + timedelta(days=horizon_days):
        return None

    candidate = {
        "title": title,
        "event_type": "other",  # a search result is never trustworthy enough to classify further
        "date": when,
        "notes": (str(result.get("url") or "") or None),
    }

    try:
        return validate_event(candidate, 0)
    except SeedError as exc:
        log.info("dropped a web result that failed validation: %s", exc)
        return None


@traced("websearch.live_events", run_type="tool")
def find_live_events(
    query: str,
    *,
    provider: WebSearchProvider | None = None,
    limit: int = MAX_RESULTS,
    horizon_days: int = 365,
) -> list[dict]:
    """Search for dated live events. Returns validated event rows, or [] on any failure."""
    provider = provider or default_provider()
    if provider is None:
        log.debug("web search is not configured; using seeded data only")
        return []

    try:
        results = provider.search(f"{query} UAE events", limit)
    except Exception as exc:  # noqa: BLE001 — an optional garnish must never break planning
        log.info("web search failed (%s); using seeded data only", exc)
        return []

    normalised: list[dict] = []
    seen: set[tuple[str, date]] = set()
    for result in results:
        event = normalise(result, horizon_days=horizon_days)
        if event is None:
            continue
        key = (event["title"], event["date"])
        if key in seen:
            continue
        seen.add(key)
        normalised.append(event)

    log.info("web search: %s results → %s usable events", len(results), len(normalised))
    return normalised
