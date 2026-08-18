"""Environment configuration.

OPENAI_API_KEY and ORS_API_KEY are optional at runtime — their absence triggers the documented
fallbacks (form-based intake / rule-based planning, and haversine travel estimates). JWT_SECRET is
required in production; a dev default is used only when DEBUG is on.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- storage -----------------------------------------------------------------
    database_url: str = f"sqlite:///{BACKEND_DIR / 'rihla.db'}"
    chroma_path: str = str(BACKEND_DIR / "chroma")

    # --- auth --------------------------------------------------------------------
    jwt_secret: str = "dev-only-insecure-secret-change-me"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 1440

    # --- third parties (all optional at runtime) ---------------------------------
    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    ors_api_key: str | None = None
    ors_timeout_seconds: float = 2.0

    # --- web search (optional; live one-off events only) -------------------------
    web_search_api_key: str | None = None

    # --- observability (LangSmith; entirely optional) ----------------------------
    langsmith_api_key: str | None = None
    langsmith_project: str = "rihla-itinerary-planner"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_tracing: bool = True  

    # --- domain ------------------------------------------------------------------
    default_currency: str = "AED"
    taxi_aed_per_km: float = 2.5
    # Driving yourself is not free: petrol, and something to leave the car in. Both are flat
    # knobs rather than researched per-place figures — the catalog carries no parking data.
    fuel_aed_per_km: float = 0.35
    parking_aed_per_stop: float = 15.0
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
