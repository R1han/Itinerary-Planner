"""Per-user isolation — spec §13 "Isolation tests" and acceptance criterion 8.

These are the tests that must never be allowed to go red: a failure here is a privacy bug, not a
functional one.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.models import Preference
from app.repo import owned_query
from app.services.memory import MemoryService

FUTURE = (date.today() + timedelta(days=30)).isoformat()

PROTECTED_ROUTES = [
    ("get", "/me"),
    ("get", "/events"),
    ("post", "/events"),
    ("get", "/family"),
    ("get", "/preferences"),
    ("post", "/preferences"),
]


@pytest.fixture
def two_users(make_user):
    alice_headers, alice = make_user("alice@rihla.app", "Alice")
    bob_headers, bob = make_user("bob@rihla.app", "Bob")
    return (alice_headers, alice), (bob_headers, bob)


# --- authentication ----------------------------------------------------------------------------


@pytest.mark.parametrize(("method", "path"), PROTECTED_ROUTES)
def test_unauthenticated_requests_are_401(client, method, path):
    kwargs = {"json": {}} if method == "post" else {}
    assert getattr(client, method)(path, **kwargs).status_code == 401


def test_garbage_token_is_401(client):
    assert client.get("/me", headers={"Authorization": "Bearer not-a-jwt"}).status_code == 401


def test_token_signed_with_another_secret_is_401(client, monkeypatch):
    from jose import jwt

    forged = jwt.encode({"sub": "1"}, "attacker-secret", algorithm="HS256")
    assert client.get("/me", headers={"Authorization": f"Bearer {forged}"}).status_code == 401


def test_login_does_not_reveal_whether_an_account_exists(client, make_user):
    make_user("real@rihla.app")
    unknown = client.post("/auth/login", json={"email": "ghost@rihla.app", "password": "x"})
    wrong_password = client.post("/auth/login", json={"email": "real@rihla.app", "password": "x"})
    assert unknown.status_code == wrong_password.status_code == 401
    assert unknown.json()["detail"] == wrong_password.json()["detail"]


# --- cross-user access -------------------------------------------------------------------------


def test_user_b_gets_404_for_user_a_event(client, two_users):
    (alice_headers, _), (bob_headers, _) = two_users
    created = client.post(
        "/events",
        headers=alice_headers,
        json={"title": "Aisha's birthday", "event_type": "birthday", "date": FUTURE},
    )
    event_id = created.json()["id"]

    # 404 rather than 403 — a 403 would confirm the row exists.
    assert client.delete(f"/events/{event_id}", headers=bob_headers).status_code == 404
    assert client.get("/events", headers=bob_headers).json() == []


def test_the_event_list_returns_only_own_events(client, two_users):
    (alice_headers, _), (bob_headers, _) = two_users
    client.post(
        "/events",
        headers=alice_headers,
        json={"title": "Alice only", "event_type": "birthday", "date": FUTURE},
    )
    client.post(
        "/events",
        headers=bob_headers,
        json={"title": "Bob only", "event_type": "anniversary", "date": FUTURE},
    )

    alice_titles = [e["title"] for e in client.get("/events", headers=alice_headers).json()]
    bob_titles = [e["title"] for e in client.get("/events", headers=bob_headers).json()]
    assert alice_titles == ["Alice only"]
    assert bob_titles == ["Bob only"]


def test_user_b_gets_404_for_user_a_preference(client, two_users):
    (alice_headers, _), (bob_headers, _) = two_users
    created = client.post(
        "/preferences",
        headers=alice_headers,
        json={"kind": "dislike", "subject": "loud rides", "category": "adventure"},
    )
    pref_id = created.json()["id"]
    assert client.delete(f"/preferences/{pref_id}", headers=bob_headers).status_code == 404
    assert client.get("/preferences", headers=bob_headers).json() == []


def test_family_is_not_shared_between_users(client, two_users):
    (alice_headers, _), (bob_headers, _) = two_users
    client.put(
        "/family",
        headers=alice_headers,
        json={"members": [{"role": "adult", "age": 34}, {"role": "child", "age": 7}]},
    )
    assert len(client.get("/family", headers=alice_headers).json()) == 2
    assert client.get("/family", headers=bob_headers).json() == []


def test_user_id_in_the_request_body_is_ignored(client, two_users):
    """The authenticated id comes from the token; a body field must never override it."""
    (alice_headers, _), (_, bob) = two_users
    client.post(
        "/events",
        headers=alice_headers,
        json={
            "title": "Smuggled",
            "event_type": "birthday",
            "date": FUTURE,
            "user_id": bob["id"],
        },
    )
    assert client.get("/events", headers=alice_headers).json()[0]["title"] == "Smuggled"


# --- the scoping layer itself ------------------------------------------------------------------


def test_owned_query_refuses_shared_tables(db):
    """Reaching for a global table through the user-scoped helper is a loud programming error."""
    from app.models import Place, TravelCache

    for shared in (Place, TravelCache):
        with pytest.raises(TypeError, match="not a user-owned table"):
            owned_query(db, shared, 1)


# --- vector memory -----------------------------------------------------------------------------


def test_preference_recall_is_scoped_to_the_querying_user(db, monkeypatch):
    """Chroma recall must return only the querying user's documents.

    Embeddings are stubbed with a deterministic bag-of-words vector so the test exercises the real
    Chroma `where` filter rather than a mock of it.
    """
    from app.models import User
    from app.services import vectors

    vocabulary = ["animals", "zoo", "dining", "beach", "theme", "queue"]

    def fake_embed(texts: list[str]) -> list[list[float]]:
        return [[float(word in text.lower()) for word in vocabulary] + [0.1] for text in texts]

    monkeypatch.setattr(vectors, "embed", fake_embed)

    alice = User(email="a@rihla.app", password_hash="x", name="A")
    bob = User(email="b@rihla.app", password_hash="x", name="B")
    db.add_all([alice, bob])
    db.commit()

    alice_pref = Preference(
        user_id=alice.id, kind="like", subject="animals and the zoo", category="aquarium"
    )
    bob_pref = Preference(
        user_id=bob.id, kind="like", subject="animals and the zoo", category="aquarium"
    )
    db.add_all([alice_pref, bob_pref])
    db.commit()

    MemoryService(db, alice.id).remember_preference(alice_pref)
    MemoryService(db, bob.id).remember_preference(bob_pref)

    recalled = MemoryService(db, alice.id).recall("animals and zoos", limit=10)
    assert recalled, "expected Alice's own preference back"
    assert len(recalled) == 1, "Bob's identically-worded preference leaked into Alice's recall"

    # And Bob still sees his own.
    assert len(MemoryService(db, bob.id).recall("animals and zoos", limit=10)) == 1


def test_recall_falls_back_to_own_sql_rows_without_embeddings(db):
    from app.models import User

    alice = User(email="a@rihla.app", password_hash="x", name="A")
    bob = User(email="b@rihla.app", password_hash="x", name="B")
    db.add_all([alice, bob])
    db.commit()
    db.add_all(
        [
            Preference(user_id=alice.id, kind="like", subject="alice likes parks"),
            Preference(user_id=bob.id, kind="like", subject="bob likes malls"),
        ]
    )
    db.commit()

    # The default fixture disables embeddings, so this exercises the SQL fallback path.
    recalled = MemoryService(db, alice.id).recall("anything", limit=10)
    assert [r["text"] for r in recalled] == ["The family likes alice likes parks."]
