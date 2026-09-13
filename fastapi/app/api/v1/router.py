"""Aggregate v1 API router."""

from fastapi import APIRouter

from app.api.v1.endpoints import compare, detect

api_router = APIRouter()
api_router.include_router(detect.router, tags=["detect"])
api_router.include_router(compare.router, tags=["compare"])
