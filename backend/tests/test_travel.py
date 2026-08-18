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


# --- fares: vehicle size and transport mode -----------------------------------------------------


@pytest.mark.parametrize(
    ("party", "label", "multiplier"),
    [
        (1, "standard", 1.0),
        (4, "standard", 1.0),
        (5, "6-seater", 1.6),
        (6, "6-seater", 1.6),
        (7, "two vehicles", 2.0),
        (11, "two vehicles", 2.0),
    ],
)
def test_vehicle_tier_steps_with_the_party_size(party, label, multiplier):
    """A fifth passenger is a step change — a bigger vehicle — not a gradual surcharge."""
    from app.services.travel import vehicle_for

    assert vehicle_for(party) == (label, multiplier)


def test_a_taxi_costs_the_metered_rate_times_the_vehicle_tier():
    from app.services.travel import TAXI, fare

    assert fare(10.0, mode=TAXI, party_size=2) == pytest.approx(25.0)
    assert fare(10.0, mode=TAXI, party_size=6) == pytest.approx(40.0)


def test_a_taxi_is_never_charged_for_parking():
    """You are not parking a taxi. Only the driver pays to leave a car somewhere."""
    from app.services.travel import TAXI, fare

    assert fare(10.0, mode=TAXI, party_size=2, arriving_stops=3) == fare(
        10.0, mode=TAXI, party_size=2
    )


def test_own_car_charges_fuel_plus_parking_at_the_stop_it_arrives_at():
    from app.services.travel import OWN_CAR, fare

    fuel_only = fare(10.0, mode=OWN_CAR, party_size=2, arriving_stops=0)
    with_parking = fare(10.0, mode=OWN_CAR, party_size=2, arriving_stops=1)

    assert fuel_only == pytest.approx(3.5)
    assert with_parking == pytest.approx(3.5 + 15.0)


def test_parking_does_not_scale_with_the_vehicle_tier():
    """One car parked once is one parking charge, whatever size it is."""
    from app.services.travel import OWN_CAR, fare

    small = fare(0.0, mode=OWN_CAR, party_size=2, arriving_stops=1)
    large = fare(0.0, mode=OWN_CAR, party_size=6, arriving_stops=1)
    assert small == large == pytest.approx(15.0)


def test_driving_your_own_car_is_far_cheaper_than_the_taxi_it_replaces():
    """The reported complaint: AED 419 of taxi fare on a plan the family drives itself."""
    from app.services.travel import OWN_CAR, TAXI, fare

    taxi = fare(160.0, mode=TAXI, party_size=4)
    car = fare(160.0, mode=OWN_CAR, party_size=4, arriving_stops=4)
    assert car < taxi / 3


def test_the_service_prices_a_cached_leg_for_the_party_that_asked(db, places):
    """Cost is derived, never cached: the cache is shared, the party and the mode are not."""
    from app.services.travel import OWN_CAR, TAXI, TravelService

    a, b = places
    taxi_pair = TravelService(db, StubORS(), mode=TAXI, party_size=2).between_places(a, b)
    van_pair = TravelService(db, StubORS(), mode=TAXI, party_size=6).between_places(a, b)
    own_car = TravelService(db, StubORS(), mode=OWN_CAR, party_size=6).between_places(a, b)

    assert taxi_pair.distance_km == van_pair.distance_km == own_car.distance_km
    assert van_pair.est_cost == pytest.approx(taxi_pair.est_cost * 1.6)
    assert own_car.est_cost < taxi_pair.est_cost


def test_a_shared_cache_row_cannot_freeze_one_partys_fare_for_everyone(db, places):
    """The bug this guards: cache est_cost, and a 2-person taxi prices a 6-person van."""
    a, b = places
    TravelService(db, StubORS(), party_size=2).between_places(a, b)

    cached = db.get(TravelCache, (a.id, b.id, DRIVING))
    assert cached is not None, "the distance should still be cached — that part is user-agnostic"

    from app.services.travel import TAXI

    big = TravelService(db, StubORS(), mode=TAXI, party_size=6).between_places(a, b)
    assert big.est_cost == pytest.approx(cached.distance_km * 2.5 * 1.6)
