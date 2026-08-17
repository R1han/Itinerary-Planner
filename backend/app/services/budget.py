"""Pricing and budget arithmetic. Pure functions — no DB, no network, no clock.

Totals are computed here and only here, on the server. The client never sends a total and the LLM
never produces one (spec §6.5).

Age-tier pricing: a place may carry `price_bands`, checked in order, e.g.

    [{"max_age": 2, "price": 0}, {"max_age": 12, "price": 155}, {"max_age": null, "price": 199}]

so a 13-year-old correctly pays the adult rate at venues that charge that way. Places without
bands fall back to the flat price_child (≤12) / price_adult split.
"""

from __future__ import annotations

from dataclasses import dataclass, field

FOOD_CATEGORIES = frozenset({"casual_dining", "fine_dining"})


@dataclass(frozen=True)
class Attendee:
    role: str  # "adult" | "child"
    age: int
    name: str | None = None


@dataclass
class CostBreakdown:
    """Mirrors the API's CostBreakdown and drives the cost chips on each slot card."""

    adults: list[float] = field(default_factory=list)
    children: list[float] = field(default_factory=list)
    free_children: int = 0
    free_under_age: int | None = None
    travel_in: float = 0.0
    total: float = 0.0
    chips: list[dict] = field(default_factory=list)

    @property
    def admission(self) -> float:
        return sum(self.adults) + sum(self.children)

    def as_dict(self) -> dict:
        return {
            "adults": self.adults,
            "children": self.children,
            "free_children": self.free_children,
            "free_under_age": self.free_under_age,
            "travel_in": self.travel_in,
            "total": self.total,
            "chips": self.chips,
        }


def price_for_age(place, age: int) -> float:
    """The admission price a person of `age` pays at `place`."""
    bands = getattr(place, "price_bands", None)
    if bands:
        for band in bands:
            max_age = band.get("max_age")
            if max_age is None or age <= max_age:
                return float(band.get("price", 0.0))
        return float(bands[-1].get("price", 0.0))
    return float(place.price_child if age <= 12 else place.price_adult)


def free_under_age(place) -> int | None:
    """The age below which admission is free, if the place has such a band. Powers the
    'N child free (under 3)' chip — it renders only when a real free tier applies."""
    bands = getattr(place, "price_bands", None)
    if not bands:
        return None
    for band in bands:
        if float(band.get("price", 0.0)) == 0.0 and band.get("max_age") is not None:
            return int(band["max_age"]) + 1
    return None


def _people(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular}" if count == 1 else f"{count} {plural}"


def slot_cost_breakdown(place, attendees: list[Attendee], travel_in: float = 0.0) -> CostBreakdown:
    """Per-slot cost for the whole party, plus the chips the slot card renders."""
    adult_prices: list[float] = []
    child_prices: list[float] = []
    free_children = 0

    for person in attendees:
        price = price_for_age(place, person.age)
        if person.role == "adult":
            adult_prices.append(round(price, 2))
        elif price <= 0.0:
            free_children += 1
        else:
            child_prices.append(round(price, 2))

    threshold = free_under_age(place) if free_children else None
    breakdown = CostBreakdown(
        adults=adult_prices,
        children=child_prices,
        free_children=free_children,
        free_under_age=threshold,
        travel_in=round(travel_in, 2),
    )
    breakdown.total = round(breakdown.admission + breakdown.travel_in, 2)

    chips: list[dict] = []
    if adult_prices:
        who = _people(len(adult_prices), "adult", "adults")
        chips.append(
            {
                "label": f"{who} · AED {sum(adult_prices):,.0f}",
                "count": len(adult_prices),
                "amount": round(sum(adult_prices), 2),
                "tone": "adult",
            }
        )
    if child_prices:
        who = _people(len(child_prices), "child", "children")
        chips.append(
            {
                "label": f"{who} · AED {sum(child_prices):,.0f}",
                "count": len(child_prices),
                "amount": round(sum(child_prices), 2),
                "tone": "child",
            }
        )
    if free_children:
        who = _people(free_children, "child", "children")
        suffix = f" (under {threshold})" if threshold else ""
        chips.append(
            {
                "label": f"{who} free{suffix}",
                "count": free_children,
                "amount": 0.0,
                "tone": "free",
            }
        )
    breakdown.chips = chips
    return breakdown


def category_bucket(category: str) -> str:
    """Which BudgetPanel segment a slot's admission belongs to."""
    return "food" if category in FOOD_CATEGORIES else "activities"


def summarise(days, total_budget: float, currency: str = "AED") -> dict:
    """Aggregate a planned or persisted itinerary into the BudgetPanel payload.

    `days` is any sequence of objects exposing `.slots` (each with `.cost` and a `.place.category`)
    and `.segments` (each with `.info.est_cost`).
    """
    per_day: list[float] = []
    activities = food = travel = 0.0

    for day in days:
        day_total = 0.0
        for slot in day.slots:
            admission = slot.cost.admission
            day_total += admission
            if category_bucket(slot.place.category) == "food":
                food += admission
            else:
                activities += admission
        for segment in day.segments:
            day_total += segment.info.est_cost
            travel += segment.info.est_cost
        per_day.append(round(day_total, 2))

    total = round(sum(per_day), 2)
    return {
        "total": total,
        "cap": round(total_budget, 2),
        "remaining": round(total_budget - total, 2),
        "currency": currency,
        "over_budget": total > total_budget + 0.01,
        "per_day": per_day,
        "categories": {
            "activities": round(activities, 2),
            "food": round(food, 2),
            "travel": round(travel, 2),
        },
    }
