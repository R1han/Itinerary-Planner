"""LangSmith tracing, made optional at the seam rather than at every call site.

`@traced(...)` and `wrap_openai(...)` are always importable and always safe to call. With no
LANGSMITH_API_KEY they are transparent pass-throughs costing one attribute lookup; with a key
they become real LangSmith spans. That means service code never branches on whether tracing is
configured — the same rule the app follows for OpenAI and ORS.

Enable by putting these in backend/.env:
    LANGSMITH_API_KEY=ls__...
    LANGSMITH_PROJECT=rihla-itinerary-planner   # optional
    LANGSMITH_TRACING=true                      # optional, defaults on when a key exists
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from functools import lru_cache
from typing import Any, TypeVar

from ..config import settings

log = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@lru_cache
def tracing_enabled() -> bool:
    """True when LangSmith is configured. Also exports the SDK's expected env vars.

    The LangSmith SDK reads os.environ directly, but this app's configuration lives in
    backend/.env via pydantic-settings — so the values are mirrored across here, once.
    """
    if not settings.langsmith_api_key or not settings.langsmith_tracing:
        return False

    os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    os.environ.setdefault("LANGSMITH_ENDPOINT", settings.langsmith_endpoint)
    os.environ["LANGSMITH_TRACING"] = "true"

    try:
        import langsmith  # noqa: F401
    except ImportError:
        log.warning("LANGSMITH_API_KEY is set but the langsmith package is not installed")
        return False

    log.info("LangSmith tracing enabled (project=%s)", settings.langsmith_project)
    return True


def traced(name: str, run_type: str = "chain", **metadata: Any) -> Callable[[F], F]:
    """Decorate a function as a LangSmith span; a no-op when tracing is off.

    run_type is one of LangSmith's span kinds — "llm", "retriever", "tool", "chain".
    """

    def decorator(func: F) -> F:
        if not tracing_enabled():
            return func
        from langsmith import traceable

        return traceable(name=name, run_type=run_type, metadata=metadata or None)(func)  # type: ignore[return-value]

    return decorator


def wrap_openai(client: Any) -> Any:
    """Wrap an OpenAI client so its calls appear as LangSmith LLM spans. Identity when off."""
    if not tracing_enabled():
        return client
    try:
        from langsmith.wrappers import wrap_openai as _wrap

        return _wrap(client)
    except Exception:  # noqa: BLE001 — tracing must never break a user-facing call
        log.exception("could not wrap the OpenAI client for tracing")
        return client
