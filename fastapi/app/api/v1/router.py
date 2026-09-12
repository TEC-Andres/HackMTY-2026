"""Aggregate v1 API router."""

from fastapi import APIRouter

from app.api.v1.endpoints import detect

api_router = APIRouter()
api_router.include_router(detect.router, tags=["detect"])
