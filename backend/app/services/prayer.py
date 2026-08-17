"""Prayer break insertion for the "Add prayer breaks" action chip.

ponytail: a static monthly table for the UAE, accurate to roughly ±10 minutes across the country,
rather than a solar-position calculation or a live prayer-times API. Ten minutes is well inside
the slack of a 20-minute rest gap, and this keeps the feature dependency-free and offline. Upgrade
path if minute-accuracy is ever wanted: swap `prayer_times_for` for an Aladhan API call plus
cache, leaving the reflow logic below untouched.
"""

from __future__ import annotations

from datetime import date

from .planner import DayPlan, TravelFn, rebuild_segments

BREAK_MINUTES = 20

# (dhuhr, asr, maghrib) in minutes past midnight, by month, averaged for the UAE.
_MONTHLY: dict[int, tuple[int, int, int]] = {
    1: (12 * 60 + 30, 15 * 60 + 30, 17 * 60 + 45),
    2: (12 * 60 + 33, 15 * 60 + 45, 18 * 60 + 5),
    3: (12 * 60 + 28, 15 * 60 + 50, 18 * 60 + 22),
    4: (12 * 60 + 18, 15 * 60 + 45, 18 * 60 + 37),
    5: (12 * 60 + 15, 15 * 60 + 40, 18 * 60 + 52),
    6: (12 * 60 + 20, 15 * 60 + 42, 19 * 60 + 5),
    7: (12 * 60 + 25, 15 * 60 + 45, 19 * 60 + 8),
    8: (12 * 60 + 25, 15 * 60 + 40, 18 * 60 + 52),
    9: (12 * 60 + 15, 15 * 60 + 30, 18 * 60 + 25),
    10: (12 * 60 + 5, 15 * 60 + 15, 17 * 60 + 57),
    11: (12 * 60 + 5, 15 * 60 + 5, 17 * 60 + 40),
    12: (12 * 60 + 15, 15 * 60 + 10, 17 * 60 + 37),
}


def prayer_times_for(day: date) -> dict[str, int]:
    dhuhr, asr, maghrib = _MONTHLY[day.month]
    return {"dhuhr": dhuhr, "asr": asr, "maghrib": maghrib}


def insert_prayer_breaks(
    day: DayPlan, travel_fn: TravelFn, origin: tuple[float, float]
) -> tuple[DayPlan, list[str]]:
    """Push slots later so each prayer time falls in a gap, then rebuild the day's travel.

    Slots are never dropped here — the caller re-runs the validator afterwards, which will trim
    anything that no longer fits the day.
    """
    if not day.slots:
        return day, []

    times = prayer_times_for(day.day_date)
    inserted: list[str] = []

    for name, moment in sorted(times.items(), key=lambda kv: kv[1]):
        ordered = sorted(day.slots, key=lambda s: s.start_min)

        # Already free at that moment? Then nothing to do for this prayer.
        covering = next(
            (s for s in ordered if s.start_min < moment < s.end_min),
            None,
        )
        if covering is None:
            continue

        # Shift the covering slot and everything after it, so the prayer lands in the gap before it.
        shift = (covering.end_min - moment) + BREAK_MINUTES
        for slot in ordered:
            if slot.start_min >= covering.start_min and not slot.locked:
                slot.start_min += shift
                slot.end_min += shift
        inserted.append(name)

    rebuild_segments(day, travel_fn, origin)
    return day, inserted
