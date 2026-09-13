"""Pydantic models for ``POST /detect/acousticEnsemble``."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.detect import Turn


class AcousticEnsembleRequest(BaseModel):
    """Both natural_speech_termination and resonance_stability need the raw
    waveform, so ``audio_base64`` is required rather than an alternative to
    ``turns`` (unlike timing-only ``DetectRequest``).
    """

    audio_base64: str = Field(
        ..., description="Base64-encoded stereo 8 kHz WAV; channel 0 is the caller."
    )
    turns: list[Turn] | None = Field(
        default=None,
        description="Optional precomputed turns (bypasses the VAD stage).",
    )
    channel: int = Field(default=0, description="Caller channel to analyse.")


class FamilyResult(BaseModel):
    """One family's own verdict, kept alongside its raw P(synthetic) so the
    self-weighted formula's math is visible in the breakdown."""

    is_synthetic: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    synthetic_probability: float = Field(..., ge=0.0, le=1.0)


class AcousticEnsembleResponse(BaseModel):
    is_synthetic: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    breakdown: dict[str, FamilyResult] = Field(
        default_factory=dict,
        description="Per-family verdicts; only populated when ?verbose=true.",
    )
    ensemble_method: str = "self_weighted_mean_square"
