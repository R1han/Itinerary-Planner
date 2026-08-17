"""seed_events CLI — the behaviour contract from spec §10 and acceptance criterion 9."""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from app.models import Event, User
from app.seed_events import SeedError, insert_events, main, resolve_user, validate_event

FUTURE = (date.today() + timedelta(days=40)).isoformat()
PAST = (date.today() - timedelta(days=40)).isoformat()


@pytest.fixture
def seeded_user(db) -> User:
    user = User(email="owner@rihla.app", password_hash="x", name="Owner")
    db.add(user)
    db.commit()
    return user


# --- validation --------------------------------------------------------------------------------


def test_rejects_unknown_event_type():
    with pytest.raises(SeedError, match="not one of"):
        validate_event({"title": "X", "event_type": "wedding", "date": FUTURE}, 0)


def test_rejects_malformed_date():
    with pytest.raises(SeedError, match="ISO date"):
        validate_event({"title": "X", "event_type": "birthday", "date": "29-08-2026"}, 0)


def test_rejects_unexpected_field():
    with pytest.raises(SeedError, match="unexpected field"):
        validate_event(
            {"title": "X", "event_type": "birthday", "date": FUTURE, "user_id": 3}, 0
        )


def test_requires_title():
    with pytest.raises(SeedError, match="'title' is required"):
        validate_event({"event_type": "birthday", "date": FUTURE}, 0)


def test_planned_defaults_to_false():
    assert validate_event({"title": "X", "event_type": "eid", "date": FUTURE}, 0)["planned"] is False


# --- insertion ---------------------------------------------------------------------------------


def test_insert_is_idempotent_on_user_title_date(db, seeded_user):
    events = [{"title": "Anniversary", "event_type": "anniversary", "date": FUTURE}]

    first = insert_events(db, seeded_user, events)
    db.commit()
    assert (first.inserted, first.skipped) == (1, 0)

    second = insert_events(db, seeded_user, events)
    db.commit()
    assert (second.inserted, second.skipped) == (0, 1)
    assert db.query(Event).count() == 1


def test_same_title_different_date_is_a_new_event(db, seeded_user):
    other = (date.today() + timedelta(days=90)).isoformat()
    insert_events(db, seeded_user, [{"title": "Trip", "event_type": "holiday", "date": FUTURE}])
    insert_events(db, seeded_user, [{"title": "Trip", "event_type": "holiday", "date": other}])
    db.commit()
    assert db.query(Event).count() == 2


def test_past_date_warns_but_still_inserts(db, seeded_user):
    result = insert_events(db, seeded_user, [{"title": "Old", "event_type": "eid", "date": PAST}])
    db.commit()
    assert result.inserted == 1
    assert any("in the past" in w for w in result.warnings)


def test_a_bad_row_aborts_the_whole_batch(db, seeded_user):
    """Validation runs over every row before the first insert, so a batch is all-or-nothing."""
    events = [
        {"title": "Good", "event_type": "birthday", "date": FUTURE},
        {"title": "Bad", "event_type": "not-a-type", "date": FUTURE},
    ]
    with pytest.raises(SeedError):
        insert_events(db, seeded_user, events)
    db.rollback()
    assert db.query(Event).count() == 0


def test_resolve_user_fails_loudly_and_never_creates(db):
    with pytest.raises(SeedError, match="never creates users"):
        resolve_user(db, "nobody@rihla.app")
    assert db.query(User).count() == 0


# --- CLI ---------------------------------------------------------------------------------------


def test_cli_unknown_email_exits_nonzero_without_touching_db(db, capsys):
    code = main(["--user", "ghost@rihla.app", "--title", "T", "--type", "eid", "--date", FUTURE])
    assert code == 1
    assert "No user with email" in capsys.readouterr().err
    assert db.query(Event).count() == 0


def test_cli_reports_inserted_then_skipped(db, seeded_user, tmp_path, capsys):
    path = tmp_path / "events.json"
    path.write_text(
        json.dumps(
            [
                {"title": "Beach weekend", "event_type": "holiday", "date": FUTURE},
                {"title": "Cousins visit", "event_type": "family_visit", "date": FUTURE},
            ]
        )
    )

    assert main(["--user", seeded_user.email, "--file", str(path)]) == 0
    assert "inserted: 2, skipped: 0" in capsys.readouterr().out

    assert main(["--user", seeded_user.email, "--file", str(path)]) == 0
    assert "inserted: 0, skipped: 2" in capsys.readouterr().out


def test_cli_rejects_file_and_inline_together(db, seeded_user, tmp_path, capsys):
    path = tmp_path / "e.json"
    path.write_text("[]")
    code = main(["--user", seeded_user.email, "--file", str(path), "--title", "T"])
    assert code == 1
    assert "not both" in capsys.readouterr().err


def test_cli_rejects_non_list_json(db, seeded_user, tmp_path, capsys):
    path = tmp_path / "e.json"
    path.write_text('{"title": "nope"}')
    assert main(["--user", seeded_user.email, "--file", str(path)]) == 1
    assert "must contain a JSON list" in capsys.readouterr().err


def test_cli_inline_event_requires_all_three_flags(db, seeded_user, capsys):
    assert main(["--user", seeded_user.email, "--title", "T", "--type", "eid"]) == 1
    assert "--date" in capsys.readouterr().err
