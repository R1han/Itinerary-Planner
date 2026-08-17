"""Constraint checker and repair loop — the "foolproof" guarantee (spec §6.6).

`validate_plan` asserts, for every day:
  * no slot overlaps
  * travel time between consecutive slots is honoured
  * every venue is open for the whole of its slot
  * every attendee clears each venue's min_age
  * the trip total stays within the budget cap
  * at most 5 days

`repair_plan` drops the lowest-scored offending slot, rebuilds that day's travel segments and
re-validates, bounded by MAX_REPAIR_PASSES. It runs on generation AND on every manual edit, so a
slot patch cannot leave an invalid day behind.
"""

from __future__ import annotations

from dataclasses import dataclass

from .planner import (
    MAX_DAYS,
    DayPlan,
    PartyProfile,
    Plan,
    TravelFn,
    attendees_clear_min_age,
    rebuild_segments,
    to_hhmm,
)

MAX_REPAIR_PASSES = 24

# Codes are stable identifiers — the UI and tests match on these, not on the prose.
SLOT_OVERLAP = "slot_overlap"
TRAVEL_TIME_VIOLATED = "travel_time_violated"
VENUE_CLOSED = "venue_closed"
VENUE_CLOSED_SEASONALLY = "venue_closed_seasonally"
MIN_AGE_NOT_MET = "min_age_not_met"
BUDGET_EXCEEDED = "budget_exceeded"
TOO_MANY_DAYS = "too_many_days"


@dataclass(frozen=True)
class Violation:
    code: str
    day_index: int
    detail: str
    position: int | None = None

    def __str__(self) -> str:  # pragma: no cover - diagnostics only
        where = f"day {self.day_index + 1}"
        if self.position is not None:
            where += f", slot {self.position + 1}"
        return f"[{self.code}] {where}: {self.detail}"


def validate_day(day: DayPlan, profile: PartyProfile) -> list[Violation]:
    violations: list[Violation] = []
    slots = sorted(day.slots, key=lambda s: s.start_min)

    travel_into = {
        segment.to_position: segment.info.duration_min for segment in day.segments
    }

    for index, slot in enumerate(slots):
        place = slot.place

        if slot.end_min <= slot.start_min:
            violations.append(
                Violation(SLOT_OVERLAP, day.day_index, "slot ends before it starts", slot.position)
            )

        if slot.start_min < place.opens_at or slot.end_min > place.closes_at:
            violations.append(
                Violation(
                    VENUE_CLOSED,
                    day.day_index,
                    f"{place.name} is open {place.open_time}–{place.close_time} but the slot runs "
                    f"{to_hhmm(slot.start_min)}–{to_hhmm(slot.end_min)}",
                    slot.position,
                )
            )

        if not place.open_in_month(day.day_date.month):
            violations.append(
                Violation(
                    VENUE_CLOSED_SEASONALLY,
                    day.day_index,
                    f"{place.name} is closed in "
                    f"{day.day_date.strftime('%B')}",
                    slot.position,
                )
            )

        if not attendees_clear_min_age(place, profile):
            violations.append(
                Violation(
                    MIN_AGE_NOT_MET,
                    day.day_index,
                    f"{place.name} requires age {place.min_age}+ but the party includes a "
                    f"{profile.youngest_age}-year-old",
                    slot.position,
                )
            )

        if index == 0:
            continue

        previous = slots[index - 1]
        if slot.start_min < previous.end_min:
            violations.append(
                Violation(
                    SLOT_OVERLAP,
                    day.day_index,
                    f"{place.name} starts at {to_hhmm(slot.start_min)} but {previous.place.name} "
                    f"runs until {to_hhmm(previous.end_min)}",
                    slot.position,
                )
            )
            continue

        required = travel_into.get(slot.position)
        if required is not None and slot.start_min - previous.end_min < required:
            violations.append(
                Violation(
                    TRAVEL_TIME_VIOLATED,
                    day.day_index,
                    f"only {slot.start_min - previous.end_min} min between {previous.place.name} "
                    f"and {place.name}, but the drive takes {required} min",
                    slot.position,
                )
            )

    return violations


def validate_plan(plan: Plan, profile: PartyProfile) -> list[Violation]:
    violations: list[Violation] = []

    if len(plan.days) > MAX_DAYS:
        violations.append(
            Violation(TOO_MANY_DAYS, 0, f"{len(plan.days)} days planned, the maximum is {MAX_DAYS}")
        )

    for day in plan.days:
        violations.extend(validate_day(day, profile))

    total = plan.total_cost
    if total > plan.total_budget + 0.01:
        violations.append(
            Violation(
                BUDGET_EXCEEDED,
                0,
                f"trip total {total:.0f} exceeds the {plan.total_budget:.0f} cap "
                f"by {total - plan.total_budget:.0f}",
            )
        )

    return violations


def _drop_weakest_slot(day: DayPlan) -> bool:
    """Remove the lowest-scored slot from a day. Returns False if there was nothing to drop."""
    if not day.slots:
        return False
    weakest = min(day.slots, key=lambda s: (s.locked, s.score))
    if weakest.locked and all(slot.locked for slot in day.slots):
        return False  # every slot is pinned by the user; nothing we may remove
    day.slots.remove(weakest)
    return True


def repair_plan(
    plan: Plan, profile: PartyProfile, travel_fn: TravelFn, origin: tuple[float, float]
) -> Plan:
    """Drop the weakest offending slot and re-validate until the plan is clean or we run out.

    An invalid plan must never reach the user, so a plan that cannot be repaired ends up empty
    with a warning rather than subtly wrong.
    """
    for _ in range(MAX_REPAIR_PASSES):
        violations = validate_plan(plan, profile)
        if not violations:
            return plan

        # Budget overruns are trip-wide: trim the most expensive day rather than an arbitrary one.
        budget_violations = [v for v in violations if v.code == BUDGET_EXCEEDED]
        if budget_violations and len(violations) == len(budget_violations):
            target = max(plan.days, key=lambda d: d.subtotal, default=None)
            if target is None or not _drop_weakest_slot(target):
                plan.warnings.append("Could not bring the plan within budget.")
                return plan
            rebuild_segments(target, travel_fn, origin)
            continue

        for violation in violations:
            if violation.code == BUDGET_EXCEEDED or violation.day_index >= len(plan.days):
                continue
            day = plan.days[violation.day_index]
            if violation.position is not None:
                offending = next(
                    (s for s in day.slots if s.position == violation.position), None
                )
                if offending is not None and not offending.locked:
                    day.slots.remove(offending)
                elif not _drop_weakest_slot(day):
                    continue
            elif not _drop_weakest_slot(day):
                continue
            rebuild_segments(day, travel_fn, origin)
            break
        else:
            plan.warnings.append("Could not repair the plan automatically.")
            return plan

    remaining = validate_plan(plan, profile)
    if remaining:
        plan.warnings.append(
            f"Gave up repairing after {MAX_REPAIR_PASSES} passes ({len(remaining)} issues left)."
        )
    return plan


def assert_valid(plan: Plan, profile: PartyProfile) -> None:
    """Raise if a plan is invalid. Used at the API boundary as a last line of defence."""
    violations = validate_plan(plan, profile)
    if violations:
        raise ValueError("; ".join(str(v) for v in violations))
