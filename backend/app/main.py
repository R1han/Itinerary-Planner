"""FastAPI application: CORS, routers, and table creation on startup."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .db import create_all
from .routers import auth, chat, conversations, events, family, itineraries, preferences

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    create_all()
    yield


app = FastAPI(
    title="Rihla",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(events.router)
app.include_router(family.router)
app.include_router(itineraries.router)
app.include_router(preferences.router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Reports which optional integrations are live, so the UI can explain its own degradations."""
    return {
        "status": "ok",
        "openai": bool(settings.openai_api_key),
        "openrouteservice": bool(settings.ors_api_key),
    }
