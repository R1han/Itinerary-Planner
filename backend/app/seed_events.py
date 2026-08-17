"""CLI: insert events for one specific user.

    python -m app.seed_events --user demo2@rihla.app --file anniversary_events.json
    python -m app.seed_events --user demo1@rihla.app \
        --title "School winter break trip" --type family_visit \
        --date 2026-12-14 --notes "5 days, whole family"

Guarantees (spec §10):
  * --user takes an email, resolved to a user_id; fails loudly and touches nothing if unknown.
  * Idempotent per event — skips any row where (user_id, title, date) already exists.
  * Goes through the same SQLAlchemy models/session as the app, so FKs and CHECKs are exercised.
  * seed.py reuses `insert_events` so there is exactly one event-insertion code path.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from sqlalchemy.orm import Session

from .db import SessionLocal, create_all
from .models import EVENT_TYPES, Event, User


class SeedError(Exception):
    """Raised for bad input; the CLI turns this into a non-zero exit without writing anything."""


@dataclass
class SeedResult:
    inserted: int = 0
    skipped: int = 0
    warnings: list[str] = field(default_factory=list)


def resolve_user(db: Session, email: str) -> User:
    user = db.query(User).filter(User.email == email.lower().strip()).one_or_none()
    if user is None:
        raise SeedError(
            f"No user with email {email!r}. This script never creates users — "
            f"register the account first, or run `python -m app.seed` for the demo accounts."
        )
    return user


def validate_event(raw: dict, index: int) -> dict:
    """Normalise and validate one raw event dict. Raises SeedError with a locating message."""
    where = f"event #{index + 1}"

    title = str(raw.get("title", "")).strip()
    if not title:
        raise SeedError(f"{where}: 'title' is required")

    event_type = str(raw.get("event_type", "")).strip()
    if event_type not in EVENT_TYPES:
        raise SeedError(
            f"{where} ({title!r}): event_type {event_type!r} is not one of {', '.join(EVENT_TYPES)}"
        )

    raw_date = raw.get("date")
    if isinstance(raw_date, date):
        parsed = raw_date
    else:
        try:
            parsed = date.fromisoformat(str(raw_date))
        except (TypeError, ValueError) as exc:
            raise SeedError(
                f"{where} ({title!r}): date {raw_date!r} is not an ISO date (YYYY-MM-DD)"
            ) from exc

    notes = raw.get("notes")
    unknown = set(raw) - {"title", "event_type", "date", "notes", "planned"}
    if unknown:
        raise SeedError(f"{where} ({title!r}): unexpected field(s) {', '.join(sorted(unknown))}")

    return {
        "title": title,
        "event_type": event_type,
        "date": parsed,
        "notes": str(notes) if notes else None,
        "planned": bool(raw.get("planned", False)),
    }


def insert_events(db: Session, user: User, raw_events: list[dict]) -> SeedResult:
    """Insert events for `user`, skipping any (user_id, title, date) that already exists.

    Validation happens for every event before the first insert, so a bad row in the middle of a
    file cannot leave a half-applied batch behind.
    """
    cleaned = [validate_event(raw, i) for i, raw in enumerate(raw_events)]

    result = SeedResult()
    today = date.today()
    for ev in cleaned:
        if ev["date"] < today:
            result.warnings.append(f"{ev['title']!r} is in the past ({ev['date'].isoformat()})")

        exists = (
            db.query(Event)
            .filter(Event.user_id == user.id, Event.title == ev["title"], Event.date == ev["date"])
            .first()
        )
        if exists is not None:
            result.skipped += 1
            continue

        db.add(Event(user_id=user.id, **ev))
        result.inserted += 1

    db.flush()
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.seed_events",
        description="Insert events for one specific user (by email).",
    )
    parser.add_argument("--user", required=True, metavar="EMAIL", help="Existing user's email")

    parser.add_argument("--file", metavar="PATH", help="JSON file: a list of event objects")
    parser.add_argument("--title", help="Single inline event: title")
    parser.add_argument("--type", dest="event_type", help=f"One of: {', '.join(EVENT_TYPES)}")
    parser.add_argument("--date", help="ISO date, YYYY-MM-DD")
    parser.add_argument("--notes", default=None)
    return parser.parse_args(argv)


def _load_raw_events(args: argparse.Namespace) -> list[dict]:
    inline = any([args.title, args.event_type, args.date])
    if args.file and inline:
        raise SeedError("Use either --file or the inline --title/--type/--date flags, not both")
    if not args.file and not inline:
        raise SeedError("Nothing to insert: pass --file, or --title/--type/--date")

    if args.file:
        path = Path(args.file)
        if not path.is_file():
            raise SeedError(f"File not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SeedError(f"{path} is not valid JSON: {exc}") from exc
        if not isinstance(payload, list):
            raise SeedError(f"{path} must contain a JSON list of event objects")
        if not all(isinstance(item, dict) for item in payload):
            raise SeedError(f"{path} must contain a JSON list of event *objects*")
        return payload

    missing = [
        flag
        for flag, value in (("--title", args.title), ("--type", args.event_type), ("--date", args.date))
        if not value
    ]
    if missing:
        raise SeedError(f"Inline event is missing {', '.join(missing)}")
    return [
        {
            "title": args.title,
            "event_type": args.event_type,
            "date": args.date,
            "notes": args.notes,
        }
    ]


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        raw_events = _load_raw_events(args)
    except SeedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    create_all()
    db = SessionLocal()
    try:
        user = resolve_user(db, args.user)
        result = insert_events(db, user, raw_events)
    except SeedError as exc:
        db.rollback()
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception:
        db.rollback()
        raise
    else:
        db.commit()
    finally:
        db.close()

    for warning in result.warnings or []:
        print(f"warning: {warning}", file=sys.stderr)
    print(f"inserted: {result.inserted}, skipped: {result.skipped}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
