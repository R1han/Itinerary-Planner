"""Travel times: OpenRouteService → SQLite cache → haversine fallback (spec §7).

The fallback is silent and always available: a segment computed without ORS is marked
`estimated=True`, which the UI renders as "~35 min" with a dashed polyline. Planning never fails
because a maps API is down or unkeyed.

Only place-to-place pairs are cached — the place set is finite, so a demo becomes cache hits after
one run. The trip's start location is per-user and appears once per day, so caching it would cost
more than it saves.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

from sqlalchemy.orm import Session

from ..config import settings
from ..models import TravelCache
from .planner import TravelFn, TravelInfo, haversine_km
from .tracing import traced

log = logging.getLogger(__name__)

DRIVING = "driving-car"
ORS_URL = "https://api.openrouteservice.org/v2/directions/{mode}/geojson"

# Fallback model (spec §7): straight-line × road factor, at city or highway speed, plus parking.
ROAD_FACTOR = 1.3
CITY_KMH = 45.0
INTERCITY_KMH = 90.0
INTERCITY_THRESHOLD_KM = 40.0
PARKING_BUFFER_MIN = 10


class TravelTimeProvider(Protocol):
    def route(
        self, from_lat: float, from_lng: float, to_lat: float, to_lng: float, mode: str = DRIVING
    ) -> TravelInfo: ...


class HaversineProvider:
    """Always-available estimate. Never raises — this is the floor the whole system stands on."""

    name = "haversine"

    def route(
        self, from_lat: float, from_lng: float, to_lat: float, to_lng: float, mode: str = DRIVING
    ) -> TravelInfo:
        straight = haversine_km(from_lat, from_lng, to_lat, to_lng)
        road_km = straight * ROAD_FACTOR
        speed = INTERCITY_KMH if straight > INTERCITY_THRESHOLD_KM else CITY_KMH
        minutes = int(round(road_km / speed * 60)) + PARKING_BUFFER_MIN
        return TravelInfo(
            distance_km=round(road_km, 2),
            duration_min=minutes,
            est_cost=round(road_km * settings.taxi_aed_per_km, 2),
            estimated=True,
            # A straight line, drawn dashed, so the map is honest about being an estimate.
            geometry=[[from_lat, from_lng], [to_lat, to_lng]],
        )


class ORSProvider:
    """OpenRouteService driving directions. Raises on any failure so the caller can fall back."""

    name = "ors"

    def __init__(self, api_key: str, timeout: float | None = None) -> None:
        self.api_key = api_key
        self.timeout = timeout if timeout is not None else settings.ors_timeout_seconds

    def route(
        self, from_lat: float, from_lng: float, to_lat: float, to_lng: float, mode: str = DRIVING
    ) -> TravelInfo:
        import httpx

        response = httpx.post(
            ORS_URL.format(mode=mode),
            headers={"Authorization": self.api_key, "Content-Type": "application/json"},
            # ORS takes [lng, lat] — the opposite order to everything else in this codebase.
            json={"coordinates": [[from_lng, from_lat], [to_lng, to_lat]]},
            timeout=self.timeout,
        )
        response.raise_for_status()
        feature = response.json()["features"][0]
        summary = feature["properties"]["summary"]

        distance_km = float(summary.get("distance", 0.0)) / 1000.0
        duration_min = int(round(float(summary.get("duration", 0.0)) / 60.0))
        coordinates = feature.get("geometry", {}).get("coordinates") or []

        return TravelInfo(
            distance_km=round(distance_km, 2),
            duration_min=max(1, duration_min) + PARKING_BUFFER_MIN,
            est_cost=round(distance_km * settings.taxi_aed_per_km, 2),
            estimated=False,
            geometry=[[lat, lng] for lng, lat in coordinates],
        )


def default_provider() -> TravelTimeProvider:
    if settings.ors_api_key:
        return ORSProvider(settings.ors_api_key)
    return HaversineProvider()


class TravelService:
    """Cache-aware travel lookups, and the `travel_fn` the pure planner consumes."""

    def __init__(self, db: Session, provider: TravelTimeProvider | None = None) -> None:
        self.db = db
        self.provider = provider or default_provider()
        self.fallback = HaversineProvider()
        # Within one planning run the same leg is asked for many times; don't re-hit ORS or SQL.
        self._memo: dict[tuple, TravelInfo] = {}

    # --- lookups ---------------------------------------------------------------------------

    @traced("travel.between_places", run_type="tool")
    def between_places(self, from_place, to_place, mode: str = DRIVING) -> TravelInfo:
        key = (from_place.id, to_place.id, mode)
        if key in self._memo:
            return self._memo[key]

        cached = self.db.get(TravelCache, key)
        if cached is not None:
            info = TravelInfo(
                distance_km=cached.distance_km,
                duration_min=cached.duration_min,
                est_cost=cached.est_cost,
                estimated=cached.provider == HaversineProvider.name,
                geometry=cached.geometry_json,
            )
            self._memo[key] = info
            return info

        info, provider_name = self._fetch(
            from_place.lat, from_place.lng, to_place.lat, to_place.lng, mode
        )

        # Only real routes are worth persisting; caching an estimate would freeze it in place and
        # stop a later ORS call from ever improving on it.
        if provider_name != HaversineProvider.name:
            self.db.merge(
                TravelCache(
                    from_place_id=from_place.id,
                    to_place_id=to_place.id,
                    mode=mode,
                    distance_km=info.distance_km,
                    duration_min=info.duration_min,
                    est_cost=info.est_cost,
                    geometry_json=info.geometry,
                    provider=provider_name,
                )
            )
            self.db.flush()

        self._memo[key] = info
        return info

    def between_coords(
        self, from_lat: float, from_lng: float, to_lat: float, to_lng: float, mode: str = DRIVING
    ) -> TravelInfo:
        key = (round(from_lat, 5), round(from_lng, 5), round(to_lat, 5), round(to_lng, 5), mode)
        if key not in self._memo:
            self._memo[key], _ = self._fetch(from_lat, from_lng, to_lat, to_lng, mode)
        return self._memo[key]

    def _fetch(
        self, from_lat: float, from_lng: float, to_lat: float, to_lng: float, mode: str
    ) -> tuple[TravelInfo, str]:
        if isinstance(self.provider, HaversineProvider):
            return self.provider.route(from_lat, from_lng, to_lat, to_lng, mode), self.fallback.name
        try:
            return self.provider.route(from_lat, from_lng, to_lat, to_lng, mode), self.provider.name
        except Exception as exc:  # noqa: BLE001 — timeout, HTTP error, quota, malformed body
            log.warning("travel provider failed (%s); falling back to haversine", exc)
            return self.fallback.route(from_lat, from_lng, to_lat, to_lng, mode), self.fallback.name

    # --- adapter for the pure planner --------------------------------------------------------

    def estimate_fn(self) -> TravelFn:
        """Network-free travel estimates, for choosing between candidates.

        The planner asks "how far is this one?" for every candidate at every step — hundreds of
        pairs for a three-day trip, of which a dozen survive. Routing all of those for real burns
        the provider's rate limit (ORS free tier allows 40 requests a minute) to answer a question
        haversine answers well enough: ranking only needs relative distance. The chosen legs are
        then routed properly by `travel_fn`, so what the user sees is still real.
        """
        memo: dict[tuple, TravelInfo] = {}

        def _estimate(from_lat: float, from_lng: float, to_lat: float, to_lng: float) -> TravelInfo:
            key = (round(from_lat, 4), round(from_lng, 4), round(to_lat, 4), round(to_lng, 4))
            if key not in memo:
                memo[key] = self.fallback.route(from_lat, from_lng, to_lat, to_lng)
            return memo[key]

        return _estimate

    def travel_fn(self, places: Sequence) -> TravelFn:
        """Wrap this service as the coordinate-based callable the planner expects.

        The planner deals in coordinates only, so legs between two known places are resolved back
        to their ids here in order to hit the shared cache; anything else (the start location)
        goes straight to the provider.
        """
        index = {(round(p.lat, 5), round(p.lng, 5)): p for p in places}

        def _travel(from_lat: float, from_lng: float, to_lat: float, to_lng: float) -> TravelInfo:
            origin = index.get((round(from_lat, 5), round(from_lng, 5)))
            destination = index.get((round(to_lat, 5), round(to_lng, 5)))
            if origin is not None and destination is not None and origin.id != destination.id:
                return self.between_places(origin, destination)
            return self.between_coords(from_lat, from_lng, to_lat, to_lng)

        return _travel
