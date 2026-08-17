"""Web search adapter: validation, silent failure and the events-not-places rule (spec §1.10, §8).

The search provider and the extractor are both stubbed. What is under test is the gate between
them and the database — that nothing reaches an `events` row without surviving the same validator
the seed CLI uses.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.services.websearch import find_live_events, to_event_row

TODAY = date.today()
HORIZON = TODAY + timedelta(days=365)
SOON = (TODAY + timedelta(days=30)).isoformat()
LONG_AGO = (TODAY - timedelta(days=30)).isoformat()


class StubProvider:
    def __init__(self, pages=None, error: Exception | None = None) -> None:
        self.pages = pages if pages is not None else [{"title": "p", "content": "c", "url": "u"}]
        self.error = error
        self.calls = 0

    def search(self, query: str, limit: int) -> list[dict]:
        self.calls += 1
        if self.error:
            raise self.error
        return self.pages


class StubExtractor:
    def __init__(self, events=None, error: Exception | None = None) -> None:
        self.events = events or []
        self.error = error
        self.calls = 0

    def extract(self, pages: list[dict], today: date) -> list[dict]:
        self.calls += 1
        if self.error:
            raise self.error
        return self.events


def run(events=None, **kwargs) -> list[dict]:
    return find_live_events(
        "concerts", provider=StubProvider(), extractor=StubExtractor(events or []), **kwargs
    )


# --- the validation gate -----------------------------------------------------------------------


def test_a_well_formed_row_survives():
    event = to_event_row({"title": "Dubai Jazz Festival", "date": SOON}, TODAY, HORIZON, "url")
    assert event is not None
    assert event["title"] == "Dubai Jazz Festival"
    assert event["date"].isoformat() == SOON
    assert event["planned"] is False
    assert event["notes"] == "url"


def test_an_unknown_event_type_is_coerced_rather_than_trusted():
    """A search result is never trustworthy enough to classify beyond the allowed set."""
    event = to_event_row({"title": "X", "date": SOON, "event_type": "rave"}, TODAY, HORIZON)
    assert event is not None and event["event_type"] == "other"


def test_a_known_event_type_is_kept():
    event = to_event_row({"title": "X", "date": SOON, "event_type": "holiday"}, TODAY, HORIZON)
    assert event is not None and event["event_type"] == "holiday"


@pytest.mark.parametrize(
    "raw",
    [
        {"title": "No date"},
        {"title": "Bad date", "date": "next Tuesday"},
        {"title": "Impossible", "date": "2027-02-31"},
        {"title": "   ", "date": SOON},
        {"date": SOON},
        "not even a dict",
    ],
)
def test_unusable_rows_are_dropped(raw):
    assert to_event_row(raw, TODAY, HORIZON) is None


def test_a_past_date_is_dropped():
    assert to_event_row({"title": "Old Festival", "date": LONG_AGO}, TODAY, HORIZON) is None


def test_a_date_beyond_the_horizon_is_dropped():
    far = (TODAY + timedelta(days=900)).isoformat()
    assert to_event_row({"title": "Expo 2028", "date": far}, TODAY, HORIZON) is None


# --- the adapter -------------------------------------------------------------------------------


def test_no_provider_configured_returns_nothing_and_does_not_raise():
    assert find_live_events("concerts", provider=None, extractor=StubExtractor()) == []


def test_no_extractor_returns_nothing_rather_than_feeding_the_planner_noise():
    """Without comprehension, scraped page text cannot be turned into trustworthy event names."""
    provider = StubProvider()
    assert find_live_events("concerts", provider=provider, extractor=None) == []


def test_a_search_failure_is_silent():
    provider = StubProvider(error=TimeoutError("took too long"))
    extractor = StubExtractor()
    assert find_live_events("concerts", provider=provider, extractor=extractor) == []
    assert provider.calls == 1
    assert extractor.calls == 0, "extraction should not run on a failed search"


def test_an_extraction_failure_is_silent():
    """Planning must proceed on seeded data alone when the garnish breaks."""
    extractor = StubExtractor(error=ValueError("bad json"))
    assert find_live_events("concerts", provider=StubProvider(), extractor=extractor) == []


def test_only_usable_rows_survive():
    events = run(
        [
            {"title": "Jazz Festival", "date": SOON},
            {"title": "Generic listicle"},
            {"title": "Old Show", "date": LONG_AGO},
        ]
    )
    assert [e["title"] for e in events] == ["Jazz Festival"]


def test_duplicates_are_collapsed_case_insensitively():
    events = run(
        [{"title": "Jazz Festival", "date": SOON}, {"title": "JAZZ FESTIVAL", "date": SOON}]
    )
    assert len(events) == 1


def test_events_come_back_in_date_order():
    later = (TODAY + timedelta(days=90)).isoformat()
    events = run([{"title": "Later", "date": later}, {"title": "Sooner", "date": SOON}])
    assert [e["title"] for e in events] == ["Sooner", "Later"]


def test_output_is_shaped_for_the_events_table_and_nothing_else():
    """Raw search text never reaches scheduling: results are events, never places."""
    event = run([{"title": "Jazz Festival", "date": SOON}])[0]
    assert set(event) == {"title", "event_type", "date", "notes", "planned"}


def test_the_result_count_is_bounded():
    from app.services.websearch import MAX_EVENTS

    many = [
        {"title": f"Event {i}", "date": (TODAY + timedelta(days=i + 1)).isoformat()}
        for i in range(MAX_EVENTS + 10)
    ]
    assert len(run(many)) == MAX_EVENTS
