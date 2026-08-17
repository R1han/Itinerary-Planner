"""Travel provider: ORS happy path, cache hits, and the timeout → haversine fallback (spec §7)."""

from __future__ import annotations

import httpx
import pytest

from app.models import Place, TravelCache
from app.services.travel import (
    DRIVING,
    PARKING_BUFFER_MIN,
    HaversineProvider,
    ORSProvider,
    TravelService,
)

# Dubai Mall → Yas Waterworld: a genuine ~120 km inter-emirate hop.
DUBAI_MALL = (25.1972, 55.2796)
YAS = (24.4887, 54.5995)


@pytest.fixture
def places(db) -> tuple[Place, Place]:
    a = Place(name="Dubai Aquarium", emirate="Dubai", lat=DUBAI_MALL[0], lng=DUBAI_MALL[1],
              category="aquarium", tags=[])
    b = Place(name="Yas Waterworld", emirate="Abu Dhabi", lat=YAS[0], lng=YAS[1],
              category="waterpark", tags=[])
    db.add_all([a, b])
    db.commit()
    return a, b


class StubORS:
    """Stands in for OpenRouteService. Records calls so we can assert the cache prevented them."""

    name = "ors"

    def __init__(self, *, fail: Exception | None = None) -> None:
        self.fail = fail
        self.calls = 0

    def route(self, from_lat, from_lng, to_lat, to_lng, mode=DRIVING):
        self.calls += 1
        if self.fail:
            raise self.fail
        from app.services.planner import TravelInfo

        return TravelInfo(
            distance_km=137.4,
            duration_min=95,
            est_cost=343.5,
            estimated=False,
            geometry=[[from_lat, from_lng], [24.8, 54.9], [to_lat, to_lng]],
        )


# --- haversine fallback ------------------------------------------------------------------------


def test_haversine_uses_highway_speed_between_emirates():
    info = HaversineProvider().route(*DUBAI_MALL, *YAS)
    assert info.estimated is True
    assert 100 < info.distance_km < 200
    # ~120 km at 90 km/h plus parking, not ~160 min at city speed.
    assert 90 < info.duration_min < 150


def test_haversine_uses_city_speed_for_a_short_hop():
    info = HaversineProvider().route(25.197, 55.279, 25.221, 55.254)
    assert info.estimated is True
    assert info.duration_min < 30


def test_haversine_always_includes_the_parking_buffer():
    info = HaversineProvider().route(25.1972, 55.2796, 25.1972, 55.2796)
    assert info.duration_min == PARKING_BUFFER_MIN
    assert info.est_cost == 0.0


def test_haversine_draws_a_straight_line_for_the_dashed_polyline():
    info = HaversineProvider().route(*DUBAI_MALL, *YAS)
    assert info.geometry == [list(DUBAI_MALL), list(YAS)]


def test_haversine_cost_follows_the_taxi_rate():
    from app.config import settings

    info = HaversineProvider().route(*DUBAI_MALL, *YAS)
    # Both figures are rounded for display, so compare to the nearest few fils.
    assert info.est_cost == pytest.approx(info.distance_km * settings.taxi_aed_per_km, abs=0.05)


# --- ORS parsing -------------------------------------------------------------------------------


def test_ors_parses_geojson_and_flips_coordinates_to_lat_lng(monkeypatch):
    """ORS speaks [lng, lat]; everything else in this codebase speaks [lat, lng]."""
    payload = {
        "features": [
            {
                "properties": {"summary": {"distance": 137_400.0, "duration": 5_100.0}},
                "geometry": {"coordinates": [[55.2796, 25.1972], [54.5995, 24.4887]]},
            }
        ]
    }

    def fake_post(*_args, **_kwargs):
        return httpx.Response(200, json=payload, request=httpx.Request("POST", "http://ors"))

    monkeypatch.setattr(httpx, "post", fake_post)

    info = ORSProvider("key").route(*DUBAI_MALL, *YAS)
    assert info.estimated is False
    assert info.distance_km == pytest.approx(137.4)
    assert info.duration_min == 85 + PARKING_BUFFER_MIN
    assert info.geometry == [[25.1972, 55.2796], [24.4887, 54.5995]]


def test_ors_request_sends_lng_lat_order(monkeypatch):
    captured = {}

    def fake_post(_url, **kwargs):
        captured.update(kwargs["json"])
        return httpx.Response(
            200,
            json={"features": [{"properties": {"summary": {"distance": 0, "duration": 0}},
                                "geometry": {"coordinates": []}}]},
            request=httpx.Request("POST", "http://ors"),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    ORSProvider("key").route(25.1972, 55.2796, 24.4887, 54.5995)
    assert captured["coordinates"] == [[55.2796, 25.1972], [54.5995, 24.4887]]


# --- service: cache and fallback ---------------------------------------------------------------


def test_first_call_hits_the_provider_and_is_cached(db, places):
    a, b = places
    stub = StubORS()
    service = TravelService(db, provider=stub)

    info = service.between_places(a, b)
    db.commit()

    assert stub.calls == 1
    assert info.estimated is False
    cached = db.get(TravelCache, (a.id, b.id, DRIVING))
    assert cached is not None
    assert cached.duration_min == 95
    assert cached.provider == "ors"


def test_a_second_service_reads_the_shared_cache_without_calling_the_provider(db, places):
    a, b = places
    first = StubORS()
    TravelService(db, provider=first).between_places(a, b)
    db.commit()

    # A different user, a different request — same shared place-to-place cache.
    second = StubORS()
    info = TravelService(db, provider=second).between_places(a, b)

    assert second.calls == 0, "cache miss — the provider was called again"
    assert info.duration_min == 95
    assert info.geometry is not None


def test_timeout_falls_back_to_haversine_and_marks_it_estimated(db, places):
    a, b = places
    stub = StubORS(fail=httpx.TimeoutException("ORS took too long"))
    service = TravelService(db, provider=stub)

    info = service.between_places(a, b)
    db.commit()

    assert stub.calls == 1
    assert info.estimated is True
    assert info.duration_min > 0
    assert info.geometry == [[a.lat, a.lng], [b.lat, b.lng]]


def test_an_estimate_is_never_cached_so_a_later_real_route_can_replace_it(db, places):
    """Caching a guess would freeze it in place and stop ORS ever improving on it."""
    a, b = places
    failing = StubORS(fail=httpx.ConnectError("down"))
    TravelService(db, provider=failing).between_places(a, b)
    db.commit()
    assert db.get(TravelCache, (a.id, b.id, DRIVING)) is None

    working = StubORS()
    info = TravelService(db, provider=working).between_places(a, b)
    db.commit()
    assert working.calls == 1
    assert info.estimated is False
    assert db.get(TravelCache, (a.id, b.id, DRIVING)) is not None


def test_http_error_falls_back_rather_than_propagating(db, places):
    a, b = places
    stub = StubORS(fail=httpx.HTTPStatusError("403", request=None, response=None))
    info = TravelService(db, provider=stub).between_places(a, b)
    assert info.estimated is True


def test_no_ors_key_never_calls_out_and_never_caches(db, places):
    a, b = places
    service = TravelService(db, provider=HaversineProvider())
    info = service.between_places(a, b)
    db.commit()
    assert info.estimated is True
    assert db.get(TravelCache, (a.id, b.id, DRIVING)) is None


# --- the planner adapter -----------------------------------------------------------------------


def test_travel_fn_routes_known_places_through_the_cache(db, places):
    a, b = places
    stub = StubORS()
    service = TravelService(db, provider=stub)
    travel = service.travel_fn([a, b])

    travel(a.lat, a.lng, b.lat, b.lng)
    travel(a.lat, a.lng, b.lat, b.lng)  # repeat within one planning run

    assert stub.calls == 1, "the same leg was fetched twice in one run"


def test_travel_fn_falls_back_to_coordinates_for_the_start_location(db, places):
    """The trip's start location is not a Place, so it cannot be cache-keyed."""
    a, b = places
    stub = StubORS()
    travel = TravelService(db, provider=stub).travel_fn([a, b])

    info = travel(25.0, 55.0, a.lat, a.lng)  # a home base, not a catalog place
    assert stub.calls == 1
    assert info is not None
    assert db.query(TravelCache).count() == 0
