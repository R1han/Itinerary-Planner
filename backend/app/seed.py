"""Global bootstrap: places catalog, Chroma embeddings, two demo accounts and their events.

Idempotent — re-running against a populated database is a no-op (spec §13.7). Event insertion is
delegated to `seed_events.insert_events` so there is exactly one code path for it.

Seeding the places collection requires OPENAI_API_KEY: embeddings are built with
text-embedding-3-small and the script exits non-zero rather than seeding a half-usable catalog.
The key is only demanded when there is embedding work to do, so a re-run on a seeded database
still succeeds without one.

    python -m app.seed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .auth import hash_password
from .db import SessionLocal, create_all
from .models import FamilyMember, Place, Preference, User
from .seed_events import insert_events
from .services import vectors

DATA_DIR = Path(__file__).resolve().parent / "data"

DEMO_USERS = [
    {
        "email": "demo1@rihla.app",
        "password": "demo123",
        "name": "Yusuf Rahman",
        # Abu Dhabi home base — the mixed-age family arc.
        "home_base_lat": 24.4539,
        "home_base_lng": 54.3773,
        "default_budget": 3500.0,
        "family": [
            {"role": "adult", "age": 34, "name": "Dad"},
            {"role": "adult", "age": 31, "name": "Mom"},
            {"role": "child", "age": 7, "name": "Aisha"},
            {"role": "child", "age": 13, "name": "Omar"},
        ],
        "preferences": [
            {"kind": "like", "subject": "animals and zoos", "category": "aquarium",
             "source": "stated", "strength": 0.9},
            {"kind": "like", "subject": "waterparks", "category": "waterpark",
             "source": "stated", "strength": 0.8},
            {"kind": "dislike", "subject": "long queues", "category": None,
             "source": "stated", "strength": 0.6},
            {"kind": "dislike", "subject": "very loud thrill rides", "category": "adventure",
             "source": "slot_edit", "strength": 0.7},
        ],
    },
    {
        "email": "demo2@rihla.app",
        "password": "demo123",
        "name": "Layla Haddad",
        # Dubai home base — the adults-only romantic arc.
        "home_base_lat": 25.2048,
        "home_base_lng": 55.2708,
        "default_budget": 2800.0,
        "family": [
            {"role": "adult", "age": 36, "name": "Layla"},
            {"role": "adult", "age": 38, "name": "Karim"},
        ],
        "preferences": [
            {"kind": "like", "subject": "fine dining and tasting menus", "category": "fine_dining",
             "source": "stated", "strength": 0.9},
            {"kind": "like", "subject": "quiet beaches at sunset", "category": "beach",
             "source": "stated", "strength": 0.8},
            {"kind": "dislike", "subject": "crowded theme parks", "category": "theme_park",
             "source": "stated", "strength": 0.8},
        ],
    },
]


def _log(msg: str) -> None:
    print(msg, flush=True)


# --- places -----------------------------------------------------------------------------------


def seed_places(db: Session) -> int:
    """Insert the catalog if the table is empty. Returns the number of rows inserted."""
    existing = db.scalar(select(func.count()).select_from(Place)) or 0
    if existing:
        _log(f"places: {existing} rows already present — skipped")
        return 0

    raw = json.loads((DATA_DIR / "places.json").read_text(encoding="utf-8"))
    for row in raw:
        db.add(
            Place(
                name=row["name"],
                emirate=row["emirate"],
                lat=row["lat"],
                lng=row["lng"],
                category=row["category"],
                price_adult=row.get("price_adult", 0.0),
                price_child=row.get("price_child", 0.0),
                price_bands=row.get("price_bands") or default_price_bands(row),
                min_age=row.get("min_age", 0),
                open_time=row.get("open_time", "09:00"),
                close_time=row.get("close_time", "22:00"),
                avg_duration_min=row.get("avg_duration_min", 90),
                tags=row.get("tags", []),
                kid_score=row.get("kid_score", 0.5),
                teen_score=row.get("teen_score", 0.5),
                romance_score=row.get("romance_score", 0.5),
                image_url=row.get("image_url"),
                category_icon=row.get("category_icon") or row["category"],
                description=row.get("description", ""),
            )
        )
    db.flush()
    _log(f"places: inserted {len(raw)} rows")
    return len(raw)


def default_price_bands(row: dict) -> list[dict]:
    """Standard UAE ticketing: under-3 free, 3–12 child price, 13+ adult price.

    Venues that price differently (flat fees, under-18 free, height limits) override this with an
    explicit `price_bands` in places.json.
    """
    return [
        {"max_age": 2, "price": 0.0},
        {"max_age": 12, "price": float(row.get("price_child", 0.0))},
        {"max_age": None, "price": float(row.get("price_adult", 0.0))},
    ]


def seed_place_embeddings(db: Session) -> int:
    """Embed every place that is not already in the Chroma collection. Returns count embedded."""
    collection = vectors.get_collection(vectors.PLACES_COLLECTION)
    places = list(db.scalars(select(Place).order_by(Place.id)))
    if not places:
        return 0

    existing_ids = set(collection.get(include=[]).get("ids", []))
    pending = [p for p in places if str(p.id) not in existing_ids]
    if not pending:
        _log(f"embeddings: {len(existing_ids)} places already embedded — skipped")
        return 0

    if not vectors.embeddings_available():
        raise SystemExit(
            "error: OPENAI_API_KEY is required to build place embeddings.\n"
            "       Set it in backend/.env and re-run `python -m app.seed`."
        )

    _log(f"embeddings: embedding {len(pending)} places with {vectors.settings.openai_embedding_model}…")
    documents = [vectors.place_document(p) for p in pending]
    try:
        embedded = vectors.embed(documents)
    except vectors.EmbeddingUnavailable as exc:
        raise SystemExit(f"error: could not build embeddings — {exc}") from exc

    collection.add(
        ids=[str(p.id) for p in pending],
        documents=documents,
        embeddings=embedded,
        metadatas=[
            {"category": p.category, "emirate": p.emirate, "min_age": p.min_age} for p in pending
        ],
    )
    _log(f"embeddings: added {len(pending)} place vectors")
    return len(pending)


# --- users ------------------------------------------------------------------------------------


def seed_users(db: Session) -> tuple[int, int]:
    """Create the demo accounts, their family and their stated preferences. Idempotent."""
    created = existing = 0
    for spec in DEMO_USERS:
        user = db.query(User).filter(User.email == spec["email"]).one_or_none()
        if user is None:
            user = User(
                email=spec["email"],
                password_hash=hash_password(spec["password"]),
                name=spec["name"],
                home_base_lat=spec["home_base_lat"],
                home_base_lng=spec["home_base_lng"],
                default_budget=spec["default_budget"],
                default_currency="AED",
            )
            db.add(user)
            db.flush()
            created += 1
            _log(f"users: created {spec['email']}")
        else:
            existing += 1

        if not db.query(FamilyMember).filter(FamilyMember.user_id == user.id).first():
            for member in spec["family"]:
                db.add(FamilyMember(user_id=user.id, **member))

        if not db.query(Preference).filter(Preference.user_id == user.id).first():
            for pref in spec["preferences"]:
                db.add(Preference(user_id=user.id, **pref))

        db.flush()
    return created, existing


def seed_demo_events(db: Session) -> tuple[int, int]:
    """Attach each demo account's events, reusing the seed_events insertion path."""
    payload = json.loads((DATA_DIR / "events.json").read_text(encoding="utf-8"))
    inserted = skipped = 0
    for email, raw_events in payload.items():
        user = db.query(User).filter(User.email == email).one_or_none()
        if user is None:
            _log(f"events: no user {email} — skipped")
            continue
        result = insert_events(db, user, raw_events)
        inserted += result.inserted
        skipped += result.skipped
        for warning in result.warnings:
            print(f"warning: {warning}", file=sys.stderr)
    return inserted, skipped


def main() -> int:
    create_all()
    db = SessionLocal()
    try:
        places_added = seed_places(db)
        users_created, users_existing = seed_users(db)
        events_inserted, events_skipped = seed_demo_events(db)
        db.commit()
        # Embeddings run after the commit so place ids are final and a Chroma failure cannot
        # roll back a good SQL seed.
        vectors_added = seed_place_embeddings(db)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    touched = places_added or users_created or events_inserted or vectors_added
    _log(
        f"\nseed complete — places: +{places_added}, users: +{users_created} "
        f"({users_existing} existing), events: +{events_inserted} ({events_skipped} skipped), "
        f"embeddings: +{vectors_added}"
    )
    if not touched:
        _log("nothing to do — database was already seeded")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
