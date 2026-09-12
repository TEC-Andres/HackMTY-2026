"""FastAPI entrypoint for the Altur / HackMTY 2026 detection service.

Run from the ``fastapi/`` directory:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.services.classifier import load_all
from app.services.turns import warmup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load every registered feature-family detector + warm up the turn extractor.
    load_all()
    warmup()
    yield


app = FastAPI(
    title="HackMTY 2026 — Caller Detection API",
    description="POST /detect classifies a caller as human or synthetic.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(api_router)
