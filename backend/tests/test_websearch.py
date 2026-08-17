"""Web search adapter: normalisation, validation and silent failure (spec §1.10, §8)."""

from __future__ import annotations

from datetime import date, timedelta

from app.services.websearch import find_live_events, normalise

SOON = (date.today() + timedelta(days=30)).isoformat()
LONG_AGO = (date.today() - timedelta(days=30)).isoformat()


class StubProvider:
    def __init__(self, results=None, error: Exception | None = None) -> None:
        self.results = results or []
        self.error = error
        self.calls = 0

    def search(self, query: str, limit: int) -> list[dict]:
        self.calls += 1
        if self.error:
            raise self.error
        return self.results


# --- normalisation -----------------------------------------------------------------------------


def test_a_dated_result_becomes_a_validated_event():
    event = normalise(
        {
            "title": "Dubai Jazz Festival — VisitDubai",
            "content": f"Runs on {SOON} at Media City Amphitheatre.",
            "url": "https://example.com/jazz",
        }
    )
    assert event is not None
    assert event["title"] == "Dubai Jazz Festival"  # site suffix stripped
    assert event["date"].isoformat() == SOON
    assert event["event_type"] == "other"
    assert event["planned"] is False


def test_a_result_without_a_date_is_dropped():
    """An event the planner cannot place on a calendar is worse than no event."""
    assert normalise({"title": "Things to do in Dubai", "content": "So much to see!"}) is None


def test_a_past_date_is_dropped():
    assert normalise({"title": "Old Festival", "content": f"Held on {LONG_AGO}."}) is None


def test_a_date_beyond_the_horizon_is_dropped():
    far = (date.today() + timedelta(days=900)).isoformat()
    assert normalise({"title": "Expo 2028", "content": f"Opens {far}."}) is None


def test_an_impossible_date_is_dropped_rather_than_crashing():
    assert normalise({"title": "Broken", "content": "Scheduled 2027-02-31."}) is None


def test_a_titleless_result_is_dropped():
    assert normalise({"title": "   ", "content": f"On {SOON}."}) is None


# --- the adapter -------------------------------------------------------------------------------


def test_no_provider_configured_returns_nothing_and_does_not_raise():
    assert find_live_events("concerts", provider=None) == []


def test_a_provider_failure_is_silent():
    """Web search failing must never break planning — seeded data carries on alone."""
    provider = StubProvider(error=TimeoutError("took too long"))
    assert find_live_events("concerts", provider=provider) == []
    assert provider.calls == 1


def test_only_usable_results_survive():
    provider = StubProvider(
        results=[
            {"title": "Jazz Festival", "content": f"On {SOON}."},
            {"title": "Generic listicle", "content": "25 things to do."},
            {"title": "Old Show", "content": f"On {LONG_AGO}."},
        ]
    )
    events = find_live_events("concerts", provider=provider)
    assert [e["title"] for e in events] == ["Jazz Festival"]


def test_duplicates_are_collapsed():
    provider = StubProvider(
        results=[
            {"title": "Jazz Festival", "content": f"On {SOON}."},
            {"title": "Jazz Festival — Timeout Dubai", "content": f"On {SOON}."},
        ]
    )
    assert len(find_live_events("concerts", provider=provider)) == 1


def test_results_are_shaped_for_the_events_table_and_nothing_else():
    """Raw search text never reaches scheduling: results are events, never places."""
    provider = StubProvider(results=[{"title": "Jazz Festival", "content": f"On {SOON}."}])
    event = find_live_events("concerts", provider=provider)[0]
    assert set(event) == {"title", "event_type", "date", "notes", "planned"}
