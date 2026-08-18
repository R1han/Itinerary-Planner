"""SQLite engine and session plumbing."""

from __future__ import annotations

import logging
from collections.abc import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings

log = logging.getLogger(__name__)

_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(settings.database_url, connect_args=_connect_args, future=True)


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_conn, _record) -> None:
    """WAL for concurrent reads during SSE streaming; FKs are off by default in SQLite."""
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)


class Base(DeclarativeBase):
    pass


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_all() -> None:
    from . import models  # noqa: F401  (register mappers before create_all)

    Base.metadata.create_all(bind=engine)
    _add_missing_columns()


# Columns added after a database was first created. `create_all` only creates missing TABLES, so
# without this an existing rihla.db keeps working right up until something SELECTs the new column.
# ponytail: a two-entry list beats adding Alembic for one column. If this list reaches a handful
# of entries, or anything needs a data backfill, that trade has flipped — bring in migrations.
_ADDED_COLUMNS = (("itineraries", "transport_mode", "VARCHAR(16) NOT NULL DEFAULT 'taxi'"),)


def _add_missing_columns() -> None:
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    with engine.begin() as connection:
        for table, column, definition in _ADDED_COLUMNS:
            if table not in inspector.get_table_names():
                continue
            if column in {c["name"] for c in inspector.get_columns(table)}:
                continue
            log.info("adding missing column %s.%s", table, column)
            connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {definition}"))
