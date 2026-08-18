"""The deterministic planner. The LLM never builds an itinerary — it only supplies inputs.

Everything here is pure: no DB session, no HTTP client, no clock. Travel is injected as a
`travel_fn(from_lat, from_lng, to_lat, to_lng) -> TravelInfo` callable, which is what makes the
property tests in tests/test_properties.py cheap enough to be worth running.

Pipeline (spec §6):
    1. constraint intake (validated upstream by Pydantic)
    2. candidate scoring
    3. geographic clustering — one cluster per day, so no day zig-zags across emirates
    4. greedy day assembly with travel, opening hours, meal windows and a budget envelope
    5. budget allocation (services/budget.py)
    6. validation + repair (services/validator.py)
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta

from .budget import Attendee, CostBreakdown, slot_cost_breakdown

# --- primitives --------------------------------------------------------------------------------

DINING_CATEGORIES = frozenset({"casual_dining", "fine_dining"})
MAX_DAYS = 5
MAX_SLOTS_PER_DAY = 6
# How long we are willing to idle outside a venue waiting for it to open.
MAX_WAIT_MIN = 40
# A meal may start a little after its window closes, but not hours later: a "lunch" that a long
# drive pushes to 15:50 is not lunch, and would misreport the day's shape to the user.
MEAL_GRACE_MIN = 45
# A meal becomes due slightly before its window opens. Without this, whatever is chosen at 11:57
# starts at 12:20 and runs to 14:02, swallowing the lunch window whole — and lunch is then
# recorded as missed despite nothing having eaten it.
MEAL_LOOKAHEAD_MIN = 30
# Per-day spend may exceed its even share by this much, as long as the trip total still fits.
DAY_ENVELOPE_FLEX = 1.35
# Score penalty per kilometre from the previous stop. Clustering decides which places belong to a
# day; this decides the order within it, and is what stops a day hopping Abu Dhabi → Dubai →
# Sharjah and spending more on taxis than on admission.
PROXIMITY_PENALTY_PER_KM = 0.012
# UAE summer. Between these hours in these months, an air-conditioned venue is worth a nudge —
# the spec calls malls and indoor attractions the "midday heat fallback".
HOT_MONTHS = frozenset({5, 6, 7, 8, 9})
HEAT_WINDOW = (11 * 60, 16 * 60)
INDOOR_HEAT_BONUS = 0.35
# How far from the trip's start location a place may be and still be worth a day trip. The UAE is
# small, so this still spans several emirates — it exists to stop a 7-year-old's birthday being
# planned around a mountain two hours away because the scorer liked it.
TRIP_RADIUS_KM = 140
# Below this many reachable candidates the radius is ignored rather than leaving the day empty.
MIN_REACHABLE = 20


def to_minutes(hhmm: str) -> int:
    hours, _, minutes = hhmm.partition(":")
    return int(hours) * 60 + int(minutes)


def to_hhmm(minutes: int) -> str:
    minutes = max(0, min(minutes, 24 * 60 - 1))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lng / 2) ** 2
    )
    return 2 * radius * math.asin(min(1.0, math.sqrt(a)))


@dataclass(frozen=True)
class TravelInfo:
    distance_km: float
    duration_min: int
    est_cost: float
    estimated: bool = True
    geometry: list | None = None


TravelFn = Callable[[float, float, float, float], TravelInfo]


@dataclass(frozen=True)
class PlaceCandidate:
    """A place as the planner sees it — decoupled from SQLAlchemy so tests can build one inline."""

    id: int
    name: str
    category: str
    emirate: str
    lat: float
    lng: float
    price_adult: float = 0.0
    price_child: float = 0.0
    price_bands: tuple | None = None
    min_age: int = 0
    open_time: str = "09:00"
    close_time: str = "22:00"
    avg_duration_min: int = 90
    tags: tuple[str, ...] = ()
    indoor: bool = False
    booking_required: bool = False
    closed_months: tuple[int, ...] = ()
    kid_score: float = 0.5
    teen_score: float = 0.5
    romance_score: float = 0.5
    similarity: float = 0.0

    def open_in_month(self, month: int) -> bool:
        return month not in self.closed_months

    @property
    def opens_at(self) -> int:
        return to_minutes(self.open_time)

    @property
    def closes_at(self) -> int:
        # A venue closing at or after midnight is stored as "00:00"–"03:00"; treat it as next-day.
        close = to_minutes(self.close_time)
        return close + 24 * 60 if close <= self.opens_at else close


@dataclass(frozen=True)
class PreferenceSignal:
    kind: str  # "like" | "dislike"
    subject: str
    category: str | None = None
    strength: float = 0.6


@dataclass
class PartyProfile:
    adults: int
    children_ages: list[int]
    event_type: str = "other"
    w_kid: float = 0.0
    w_teen: float = 0.0
    w_romance: float = 0.0
    w_semantic: float = 0.35
    w_preference: float = 0.45
    max_slot_min: int = 240
    needs_midday_rest: bool = False
    evening_bias: bool = False
    day_start: int = 9 * 60
    day_end: int = 21 * 60 + 30
    meal_windows: tuple[tuple[str, int, int], ...] = ()
    attendees: list[Attendee] = field(default_factory=list)

    @property
    def youngest_age(self) -> int:
        return min([a.age for a in self.attendees], default=18)

    @property
    def adults_only(self) -> bool:
        return not self.children_ages


@dataclass
class PlannedSlot:
    place: PlaceCandidate
    day_index: int
    position: int
    start_min: int
    end_min: int
    score: float
    cost: CostBreakdown
    locked: bool = False
    # The persisted Slot row this came from, when loaded from the database. Carried through edits
    # so ids stay stable — the client holds them for hover and selection across the strip and map.
    row_id: int | None = None

    @property
    def start_time(self) -> str:
        return to_hhmm(self.start_min)

    @property
    def end_time(self) -> str:
        return to_hhmm(self.end_min)


@dataclass
class PlannedSegment:
    day_index: int
    to_position: int
    info: TravelInfo
    from_position: int | None = None  # None = leaving the trip's start location


@dataclass
class DayPlan:
    day_index: int
    day_date: date
    slots: list[PlannedSlot] = field(default_factory=list)
    segments: list[PlannedSegment] = field(default_factory=list)

    @property
    def driving_total_min(self) -> int:
        return sum(s.info.duration_min for s in self.segments)

    @property
    def subtotal(self) -> float:
        admissions = sum(slot.cost.admission for slot in self.slots)
        travel = sum(seg.info.est_cost for seg in self.segments)
        return round(admissions + travel, 2)


@dataclass
class Plan:
    days: list[DayPlan] = field(default_factory=list)
    total_budget: float = 0.0
    currency: str = "AED"
    warnings: list[str] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return round(sum(day.subtotal for day in self.days), 2)


# --- 1. party profile --------------------------------------------------------------------------


def build_profile(
    attendees: Sequence[Attendee], event_type: str = "other", *, prayer_breaks: bool = False
) -> PartyProfile:
    """Derive scoring weights and day shape from who is going and why (spec §6).

    Weights come from the party — never from the LLM.
    """
    attendees = list(attendees)
    adults = [a for a in attendees if a.role == "adult"]
    children = [a for a in attendees if a.role == "child"]
    child_ages = sorted(a.age for a in children)

    young_kids = any(age < 8 for age in child_ages)
    teens = any(13 <= age <= 17 for age in child_ages)
    romantic = event_type == "anniversary" and not children

    profile = PartyProfile(
        adults=len(adults),
        children_ages=child_ages,
        event_type=event_type,
        attendees=attendees,
    )

    if romantic:
        profile.w_romance, profile.w_teen, profile.w_kid = 1.0, 0.0, 0.0
        profile.evening_bias = True
        profile.day_start = 11 * 60
        profile.day_end = 23 * 60 + 30
        profile.max_slot_min = 180
        profile.meal_windows = (("lunch", 13 * 60, 15 * 60), ("dinner", 19 * 60, 21 * 60 + 30))
    elif young_kids:
        # Short slots, an early finish and a protected midday rest (spec §6).
        profile.w_kid = 1.0
        profile.w_teen = 0.35 if teens else 0.1
        profile.w_romance = 0.05
        profile.max_slot_min = 120
        profile.needs_midday_rest = True
        profile.day_start = 9 * 60
        profile.day_end = 19 * 60 + 30
        profile.meal_windows = (("lunch", 12 * 60, 14 * 60), ("dinner", 17 * 60 + 30, 19 * 60 + 30))
    elif child_ages:
        profile.w_teen = 1.0
        profile.w_kid = 0.3
        profile.w_romance = 0.1
        profile.max_slot_min = 300
        profile.day_start = 9 * 60 + 30
        profile.day_end = 22 * 60
        profile.meal_windows = (("lunch", 12 * 60 + 30, 15 * 60), ("dinner", 18 * 60 + 30, 21 * 60))
    else:
        profile.w_teen = 0.6
        profile.w_romance = 0.5
        profile.w_kid = 0.05
        profile.max_slot_min = 240
        profile.day_start = 9 * 60 + 30
        profile.day_end = 22 * 60 + 30
        profile.meal_windows = (("lunch", 12 * 60 + 30, 15 * 60), ("dinner", 19 * 60, 21 * 60 + 30))

    if prayer_breaks:
        # Later start and a slightly shorter day to leave room for the inserted gaps.
        profile.day_end = min(profile.day_end, 22 * 60)
    return profile


# --- 2. scoring --------------------------------------------------------------------------------


def preference_signal(place: PlaceCandidate, preferences: Sequence[PreferenceSignal]) -> float:
    """Net like/dislike pull for a place. Positive boosts, negative penalises."""
    signal = 0.0
    haystack = f"{place.name} {place.category} {' '.join(place.tags)}".lower()
    for pref in preferences:
        matched = False
        if pref.category and pref.category == place.category:
            matched = True
        else:
            for token in pref.subject.lower().split():
                if len(token) > 3 and token in haystack:
                    matched = True
                    break
        if not matched:
            continue
        signal += pref.strength if pref.kind == "like" else -pref.strength * 1.4
    return max(-2.0, min(2.0, signal))


def score_place(
    place: PlaceCandidate,
    profile: PartyProfile,
    preferences: Sequence[PreferenceSignal] = (),
) -> float:
    """score = w_semantic·similarity + w_profile·profile_fit + w_pref·preference − dislike_penalty."""
    weight_sum = profile.w_kid + profile.w_teen + profile.w_romance or 1.0
    profile_fit = (
        profile.w_kid * place.kid_score
        + profile.w_teen * place.teen_score
        + profile.w_romance * place.romance_score
    ) / weight_sum

    return (
        profile.w_semantic * place.similarity
        + 1.0 * profile_fit
        + profile.w_preference * preference_signal(place, preferences)
    )


def attendees_clear_min_age(place: PlaceCandidate, profile: PartyProfile) -> bool:
    """A slot is valid only if EVERY attendee clears min_age (spec §6, mixed party)."""
    return all(person.age >= place.min_age for person in profile.attendees)


# --- 3. geographic clustering ------------------------------------------------------------------


def ensure_dining(
    bucket: list[PlaceCandidate], dining_pool: Sequence[PlaceCandidate], minimum: int = 3
) -> list[PlaceCandidate]:
    """Top a day's cluster up with nearby restaurants if it has too few.

    Clustering is geographic and blind to category, so a cluster can legitimately come out with no
    dining in it at all — and that day then gets no lunch and no dinner, because the meal windows
    have nothing to choose from. The nearest restaurants to that cluster are added back.
    """
    if not bucket:
        return bucket

    have = [p for p in bucket if p.category in DINING_CATEGORIES]
    if len(have) >= minimum:
        return bucket

    present = {p.id for p in bucket}
    centre_lat = sum(p.lat for p in bucket) / len(bucket)
    centre_lng = sum(p.lng for p in bucket) / len(bucket)

    nearest = sorted(
        (p for p in dining_pool if p.id not in present),
        key=lambda p: haversine_km(centre_lat, centre_lng, p.lat, p.lng),
    )
    return bucket + nearest[: minimum - len(have)]


def cluster_by_proximity(
    candidates: Sequence[PlaceCandidate], num_days: int, origin: tuple[float, float]
) -> list[list[PlaceCandidate]]:
    """Greedy clustering: pick spread-out, high-scoring seeds, assign each place to its nearest.

    Deliberately not k-means — with ~50 candidates and ≤5 days, one greedy pass gives the property
    that actually matters (a day does not zig-zag between emirates) with no numpy dependency and
    no random initialisation to make results non-reproducible.
    """
    if num_days <= 1 or not candidates:
        return [list(candidates)]

    ordered = list(candidates)
    seeds: list[PlaceCandidate] = [ordered[0]]  # candidates arrive score-sorted
    for _ in range(min(num_days, len(ordered)) - 1):
        # Farthest-point seeding, so clusters spread across the map instead of collapsing.
        farthest = max(
            (p for p in ordered if p not in seeds),
            key=lambda p: min(haversine_km(p.lat, p.lng, s.lat, s.lng) for s in seeds),
            default=None,
        )
        if farthest is None:
            break
        seeds.append(farthest)

    buckets: list[list[PlaceCandidate]] = [[] for _ in seeds]
    for place in ordered:
        nearest = min(
            range(len(seeds)),
            key=lambda i: haversine_km(place.lat, place.lng, seeds[i].lat, seeds[i].lng),
        )
        buckets[nearest].append(place)

    # Put the cluster nearest the trip's start location first, so day 1 begins close to home.
    buckets.sort(key=lambda b: haversine_km(b[0].lat, b[0].lng, *origin) if b else 1e9)
    while len(buckets) < num_days:
        buckets.append([])
    return buckets[:num_days]


# --- 4. day assembly ---------------------------------------------------------------------------


def _meal_due(profile: PartyProfile, cursor: int, filled: set[str]) -> tuple[str, int, int] | None:
    for window in profile.meal_windows:
        label, start, end = window
        if label not in filled and start - MEAL_LOOKAHEAD_MIN <= cursor <= end:
            return window
    return None


def _meal_missed(profile: PartyProfile, cursor: int, filled: set[str]) -> set[str]:
    return {label for label, _, end in profile.meal_windows if label not in filled and cursor > end}


def _next_meal_start(profile: PartyProfile, cursor: int, filled: set[str]) -> int | None:
    """Start of the next unfilled meal window after `cursor`, if any."""
    upcoming = [
        start for label, start, _ in profile.meal_windows if label not in filled and start > cursor
    ]
    return min(upcoming) if upcoming else None


def _cheapest_meal(candidates: Sequence[PlaceCandidate], profile: PartyProfile) -> float:
    """What the party would pay at the cheapest reachable dining option in this pool."""
    costs = [
        slot_cost_breakdown(place, profile.attendees).total
        for place in candidates
        if place.category in DINING_CATEGORIES and attendees_clear_min_age(place, profile)
    ]
    return min(costs) if costs else 0.0


def _meal_reserve(profile: PartyProfile, cursor: int, filled: set[str], unit: float) -> float:
    """Budget to hold back for the meals still ahead today.

    Without this, one expensive anchor venue eats the whole day envelope and every meal window is
    then skipped for being unaffordable — which is how a plan ends up with a waterpark and no lunch.
    """
    ahead = sum(
        1 for label, _, end in profile.meal_windows if label not in filled and end >= cursor
    )
    return ahead * unit


def assemble_day(
    day_index: int,
    day_date: date,
    candidates: Sequence[PlaceCandidate],
    profile: PartyProfile,
    travel_fn: TravelFn,
    origin: tuple[float, float],
    *,
    day_envelope: float,
    remaining_total: float,
    scores: dict[int, float],
    used: set[int],
) -> DayPlan:
    """Greedily build one day: highest-scored feasible place, honouring travel, hours and budget."""
    day = DayPlan(day_index=day_index, day_date=day_date)
    cursor = profile.day_start
    spent = 0.0
    rest_taken = not profile.needs_midday_rest
    filled_meals: set[str] = set()
    previous: tuple[float, float] = origin
    previous_position: int | None = None

    pool = list(candidates)
    meal_unit = _cheapest_meal(pool, profile)

    while len(day.slots) < MAX_SLOTS_PER_DAY and cursor < profile.day_end:
        # A midday rest for young children is a gap in the schedule, not a slot. It must yield to
        # a meal that is currently due: taking the rest first pushed the cursor past the end of
        # the lunch window, which then counted as missed — so a family with a small child got a
        # nap instead of lunch, every single day.
        if not rest_taken and cursor >= 13 * 60 and _meal_due(profile, cursor, filled_meals) is None:
            cursor += 60
            rest_taken = True
            continue

        due = _meal_due(profile, cursor, filled_meals)
        due_label = due[0] if due else None
        best: tuple[PlannedSlot, TravelInfo] | None = None

        # Re-rank from where we currently are, so the next stop is a good place that is also
        # near, and prefer somewhere air-conditioned if this is a summer midday.
        hot = day_date.month in HOT_MONTHS and HEAT_WINDOW[0] <= cursor <= HEAT_WINDOW[1]
        ordered = sorted(
            pool,
            key=lambda p: scores.get(p.id, 0.0)
            - PROXIMITY_PENALTY_PER_KM * haversine_km(previous[0], previous[1], p.lat, p.lng)
            + (INDOOR_HEAT_BONUS if hot and p.indoor else 0.0),
            reverse=True,
        )

        for place in ordered:
            if place.id in used:
                continue
            is_dining = place.category in DINING_CATEGORIES
            if due and not is_dining:
                continue
            if not due and is_dining:
                continue
            if profile.evening_bias and place.category == "fine_dining" and cursor < 17 * 60:
                continue
            if not attendees_clear_min_age(place, profile):
                continue
            # A venue shut for the season cannot be scheduled at all — this is a correctness
            # filter, not a preference.
            if not place.open_in_month(day_date.month):
                continue

            travel = travel_fn(previous[0], previous[1], place.lat, place.lng)
            arrival = cursor + travel.duration_min
            if arrival < place.opens_at:
                if place.opens_at - arrival > MAX_WAIT_MIN:
                    continue
                arrival = place.opens_at
            if due is not None and arrival > due[2] + MEAL_GRACE_MIN:
                continue

            duration = min(place.avg_duration_min, profile.max_slot_min)
            departure = arrival + duration
            if departure > place.closes_at or departure > profile.day_end:
                continue

            cost = slot_cost_breakdown(place, profile.attendees, travel.est_cost)
            # Hold back enough for the meals still to come; a meal itself spends its own reserve.
            reserve = 0.0 if is_dining else _meal_reserve(profile, cursor, filled_meals, meal_unit)
            if spent + cost.total + reserve > day_envelope * DAY_ENVELOPE_FLEX:
                continue
            if cost.total > remaining_total:
                continue

            best = (
                PlannedSlot(
                    place=place,
                    day_index=day_index,
                    position=len(day.slots),
                    start_min=arrival,
                    end_min=departure,
                    score=scores.get(place.id, 0.0),
                    cost=cost,
                ),
                travel,
            )
            break

        if best is None:
            if due_label:
                # No affordable, reachable meal in this window — skip it rather than stalling.
                filled_meals.add(due_label)
                continue
            missed = _meal_missed(profile, cursor, filled_meals)
            if missed:
                filled_meals |= missed
                continue
            # Nothing bookable right now, but a meal window is still ahead: wait for it rather
            # than ending the day early. Without this, an evening-heavy anniversary plan stops
            # just short of dinner once the daytime candidates are exhausted.
            next_meal = _next_meal_start(profile, cursor, filled_meals)
            if next_meal is not None and next_meal < profile.day_end:
                cursor = next_meal
                continue
            break

        slot, travel = best
        day.slots.append(slot)
        day.segments.append(
            PlannedSegment(
                day_index=day_index,
                from_position=previous_position,
                to_position=slot.position,
                info=travel,
            )
        )
        used.add(slot.place.id)
        spent += slot.cost.total
        remaining_total -= slot.cost.total
        cursor = slot.end_min
        previous = (slot.place.lat, slot.place.lng)
        previous_position = slot.position
        if due_label:
            filled_meals.add(due_label)

    return day


# --- 5. top-level generation -------------------------------------------------------------------


def generate_plan(
    candidates: Sequence[PlaceCandidate],
    profile: PartyProfile,
    travel_fn: TravelFn,
    *,
    start_date: date,
    num_days: int,
    total_budget: float,
    origin: tuple[float, float],
    preferences: Sequence[PreferenceSignal] = (),
    currency: str = "AED",
) -> Plan:
    """Score → cluster → assemble → repair. Always returns a plan the validator accepts."""
    from .validator import repair_plan  # local import: validator imports our dataclasses

    if not 1 <= num_days <= MAX_DAYS:
        raise ValueError(f"num_days must be between 1 and {MAX_DAYS}")

    reachable = [
        place
        for place in candidates
        if haversine_km(origin[0], origin[1], place.lat, place.lng) <= TRIP_RADIUS_KM
    ]
    if len(reachable) >= MIN_REACHABLE:
        candidates = reachable

    scores = {p.id: score_place(p, profile, preferences) for p in candidates}
    ranked = sorted(candidates, key=lambda p: scores[p.id], reverse=True)
    buckets = cluster_by_proximity(ranked, num_days, origin)
    dining_pool = [p for p in ranked if p.category in DINING_CATEGORIES]
    buckets = [ensure_dining(bucket, dining_pool) for bucket in buckets]

    plan = Plan(total_budget=total_budget, currency=currency)
    used: set[int] = set()
    envelope = total_budget / num_days

    for day_index in range(num_days):
        bucket = buckets[day_index] if day_index < len(buckets) else []
        # Fall back to the full pool if this cluster is too thin to fill a day; a sparse cluster
        # should cost geographic tightness, never leave the day empty.
        pool = bucket if len(bucket) >= 4 else ranked

        def build(candidate_pool):
            return assemble_day(
                day_index=day_index,
                day_date=start_date + timedelta(days=day_index),
                candidates=candidate_pool,
                profile=profile,
                travel_fn=travel_fn,
                origin=origin,
                day_envelope=envelope,
                remaining_total=total_budget - plan.total_cost,
                scores=scores,
                used=used,
            )

        day = build(pool)
        if not day.slots and pool is not ranked:
            # This cluster is unreachable on the remaining budget — usually a far-flung group on a
            # tight cap, where the drive alone exceeds the day's envelope. Geographic tightness is
            # a preference; an empty day is a failure, so fall back to the whole pool.
            day = build(ranked)

        plan.days.append(day)

    if not any(day.slots for day in plan.days):
        plan.warnings.append(
            "No places matched the constraints — try a longer stay, a wider area or a higher budget."
        )

    return repair_plan(plan, profile, travel_fn, origin)


def day_theme(day: DayPlan) -> str:
    """Human label for a day, from its dominant non-dining categories ('Waterpark & Wildlife')."""
    return theme_from_categories(
        [(slot.place.category, slot.end_min - slot.start_min) for slot in day.slots]
    )


def theme_from_categories(pairs: Sequence[tuple[str, int]]) -> str:
    """Day theme from (category, minutes) pairs, so it works from planner objects or DB rows."""
    if not pairs:
        return "Open day"

    pretty = {
        "casual_dining": "Food",
        "fine_dining": "Fine dining",
        "theme_park": "Theme park",
        "waterpark": "Waterpark",
        "aquarium": "Wildlife",
        "museum": "Culture",
        "park": "Parks",
        "beach": "Beach",
        "adventure": "Adventure",
        "mall": "Shopping",
        "show": "Show",
        "cruise": "Cruise",
    }

    weights: dict[str, float] = {}
    for category, minutes in pairs:
        if category in DINING_CATEGORIES:
            continue
        weights[category] = weights.get(category, 0.0) + minutes

    if not weights:  # a dining-only day
        category = max(pairs, key=lambda pair: pair[1])[0]
        return pretty.get(category, category.replace("_", " ").title())

    top = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:2]
    labels = [pretty.get(category, category.replace("_", " ").title()) for category, _ in top]
    return " & ".join(labels)


def reflow_day(
    day: DayPlan, profile: PartyProfile, travel_fn: TravelFn, origin: tuple[float, float]
) -> DayPlan:
    """Push slots later until the day is internally consistent, preserving order and durations.

    Used after a manual edit: moving one slot should cascade the rest of the day forward rather
    than leave an overlap for the repair pass to resolve by deleting something the user kept.
    """
    ordered = sorted(day.slots, key=lambda s: s.start_min)
    previous_end = profile.day_start
    previous_coord = origin

    for slot in ordered:
        duration = slot.end_min - slot.start_min
        leg = travel_fn(previous_coord[0], previous_coord[1], slot.place.lat, slot.place.lng)
        earliest = previous_end + leg.duration_min
        start = max(slot.start_min, earliest, slot.place.opens_at)
        slot.start_min = start
        slot.end_min = start + duration
        previous_end = slot.end_min
        previous_coord = (slot.place.lat, slot.place.lng)

    return rebuild_segments(day, travel_fn, origin)


def rebuild_segments(day: DayPlan, travel_fn: TravelFn, origin: tuple[float, float]) -> DayPlan:
    """Recompute every travel segment for a day after its slots have changed."""
    day.slots.sort(key=lambda s: s.start_min)
    for index, slot in enumerate(day.slots):
        slot.position = index

    segments: list[PlannedSegment] = []
    previous = origin
    previous_position: int | None = None
    for slot in day.slots:
        info = travel_fn(previous[0], previous[1], slot.place.lat, slot.place.lng)
        segments.append(
            PlannedSegment(
                day_index=day.day_index,
                from_position=previous_position,
                to_position=slot.position,
                info=info,
            )
        )
        slot.cost.travel_in = round(info.est_cost, 2)
        slot.cost.total = round(slot.cost.admission + info.est_cost, 2)
        previous = (slot.place.lat, slot.place.lng)
        previous_position = slot.position

    day.segments = segments
    return day
