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
from app.services.lexical import lexical_detector
from app.services.turns import warmup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load every registered detector + warm up the turn extractor at startup.
    load_all()
    try:
        lexical_detector.load()
    except FileNotFoundError as exc:
        logger.warning("%s", exc)
    warmup()
    yield


app = FastAPI(
    title="HackMTY 2026 — Caller Detection API",
    description=(
        "POST /detect/timeDiff classifies a caller as human or synthetic using "
        "turn-taking timing. POST /detect/NST does the same using endpoint "
        "acoustics (natural_speech_termination). POST /detect/STTLexicalAnalysis "
        "runs the turn-taking detector against the issue #21 lexical detector on "
        "the same clip."
    ),
    version="0.2.0",
    lifespan=lifespan,
)
app.include_router(api_router)
