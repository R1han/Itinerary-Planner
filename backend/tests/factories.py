"""Builders for planner tests. Keeps the tests about behaviour rather than about construction."""

from __future__ import annotations

from app.services.budget import Attendee
from app.services.planner import PlaceCandidate, TravelInfo

# Roughly Dubai Mall — a convenient default origin.
ORIGIN = (25.1972, 55.2796)


def place(
    place_id: int,
    name: str = "Somewhere",
    category: str = "park",
    *,
    lat: float = 25.20,
    lng: float = 55.27,
    price_adult: float = 50.0,
    price_child: float = 25.0,
    price_bands: tuple | None = None,
    min_age: int = 0,
    open_time: str = "09:00",
    close_time: str = "22:00",
    avg_duration_min: int = 90,
    emirate: str = "Dubai",
    tags: tuple[str, ...] = (),
    indoor: bool = False,
    booking_required: bool = False,
    closed_months: tuple[int, ...] = (),
    kid_score: float = 0.5,
    teen_score: float = 0.5,
    romance_score: float = 0.5,
    similarity: float = 0.0,
) -> PlaceCandidate:
    return PlaceCandidate(
        id=place_id,
        name=name,
        category=category,
        emirate=emirate,
        lat=lat,
        lng=lng,
        price_adult=price_adult,
        price_child=price_child,
        price_bands=price_bands,
        min_age=min_age,
        open_time=open_time,
        close_time=close_time,
        avg_duration_min=avg_duration_min,
        tags=tags,
        indoor=indoor,
        booking_required=booking_required,
        closed_months=closed_months,
        kid_score=kid_score,
        teen_score=teen_score,
        romance_score=romance_score,
        similarity=similarity,
    )


def family(adults: int = 2, child_ages: tuple[int, ...] = ()) -> list[Attendee]:
    people = [Attendee(role="adult", age=34) for _ in range(adults)]
    people += [Attendee(role="child", age=age) for age in child_ages]
    return people


def fixed_travel(minutes: int = 20, km: float = 8.0, cost: float = 20.0):
    """A travel_fn that always returns the same leg — makes schedules exactly predictable."""

    def _travel(_a: float, _b: float, _c: float, _d: float) -> TravelInfo:
        return TravelInfo(distance_km=km, duration_min=minutes, est_cost=cost, estimated=True)

    return _travel


def distance_travel(kmh: float = 45.0, aed_per_km: float = 2.5):
    """A travel_fn derived from real distance — used where geography should matter."""
    from app.services.planner import haversine_km

    def _travel(a: float, b: float, c: float, d: float) -> TravelInfo:
        km = haversine_km(a, b, c, d) * 1.3
        minutes = int(round(km / kmh * 60)) + 10
        return TravelInfo(
            distance_km=round(km, 2),
            duration_min=minutes,
            est_cost=round(km * aed_per_km, 2),
            estimated=True,
        )

    return _travel
