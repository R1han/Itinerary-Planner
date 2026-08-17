"""Shared test fixtures.

Every test runs against a throwaway SQLite file and a throwaway Chroma directory, with no network
access: `vectors.embed` is stubbed to raise EmbeddingUnavailable by default, which is the same
path the app takes when OPENAI_API_KEY is missing. Tests that want embeddings opt in explicitly.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:  # pragma: no cover
    from fastapi.testclient import TestClient
    from sqlalchemy.orm import Session

# This block MUST stay at module scope, not in a fixture. pytest imports conftest before it
# collects test modules, and those modules import `app.*` at import time — which builds the
# lru_cached Settings. A session fixture would run after that, far too late, and the suite would
# quietly read and write the developer's real rihla.db.
_TMP_ROOT = tempfile.mkdtemp(prefix="rihla-tests-")
atexit.register(shutil.rmtree, _TMP_ROOT, ignore_errors=True)

os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_ROOT}/test.db"
os.environ["CHROMA_PATH"] = f"{_TMP_ROOT}/chroma"
os.environ.setdefault("JWT_SECRET", "test-secret")
os.environ["LANGSMITH_TRACING"] = "false"
for _key in ("OPENAI_API_KEY", "ORS_API_KEY", "LANGSMITH_API_KEY"):
    os.environ.pop(_key, None)


@pytest.fixture(scope="session", autouse=True)
def _storage_is_isolated() -> Iterator[None]:
    """Fail loudly if the suite is somehow pointed at a real database."""
    from app.config import settings

    assert _TMP_ROOT in settings.database_url, (
        f"tests are pointed at {settings.database_url!r}, not the temp directory — "
        "something imported app.config before conftest set DATABASE_URL"
    )
    yield


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default: no embeddings. Mirrors running with the OpenAI key removed."""
    from app.services import vectors

    def _unavailable(_texts: list[str]) -> list[list[float]]:
        raise vectors.EmbeddingUnavailable("disabled in tests")

    monkeypatch.setattr(vectors, "embed", _unavailable)


@pytest.fixture
def db() -> Iterator["Session"]:  # noqa: F821
    """A clean database for one test."""
    from app.db import Base, SessionLocal, engine

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db) -> Iterator["TestClient"]:  # noqa: F821
    """A TestClient bound to the same session the `db` fixture hands out."""
    from fastapi.testclient import TestClient

    from app.db import get_db
    from app.main import app

    app.dependency_overrides[get_db] = lambda: db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def make_user(client):
    """Register a user and return (headers, user_json)."""

    def _make(email: str, name: str = "Test User", password: str = "testpass123"):
        response = client.post(
            "/auth/register", json={"email": email, "password": password, "name": name}
        )
        assert response.status_code == 201, response.text
        payload = response.json()
        return {"Authorization": f"Bearer {payload['access_token']}"}, payload["user"]

    return _make
