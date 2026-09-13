"""Aggregate v1 API router."""

from fastapi import APIRouter

from app.api.v1.endpoints import acoustic_ensemble, compare, detect, report, resonance

api_router = APIRouter()
api_router.include_router(detect.router, tags=["detect"])
api_router.include_router(compare.router, tags=["compare"])
api_router.include_router(resonance.router, tags=["resonance"])
api_router.include_router(acoustic_ensemble.router, tags=["acoustic-ensemble"])
api_router.include_router(report.router, tags=["report"])
