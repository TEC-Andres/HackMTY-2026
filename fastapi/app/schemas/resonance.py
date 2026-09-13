"""Pydantic request/response models for ``POST /detect/resonance``."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.detect import Turn


class ResonanceDetectRequest(BaseModel):
    """Needs the raw waveform (Praat formant/pitch tracking), so unlike
    ``DetectRequest`` (timing), ``audio_base64`` is required rather than an
    alternative to ``turns``.
    """

    audio_base64: str = Field(
        ..., description="Base64-encoded stereo 8 kHz WAV; channel 0 is the caller."
    )
    turns: list[Turn] | None = Field(
        default=None,
        description="Optional precomputed turns (bypasses the VAD stage).",
    )
    channel: int = Field(default=0, description="Caller channel to analyse.")


class ResonanceDetectResponse(BaseModel):
    """Minimal contract, same shape as ``DetectResponse``."""

    is_synthetic: bool
    confidence: float = Field(..., ge=0.0, le=1.0)


class ResonanceDetectVerboseResponse(ResonanceDetectResponse):
    """``?verbose=true`` payload: the minimal verdict plus the evidence."""

    channel: int = Field(..., description="Caller channel that was analysed.")
    n_turns: int = Field(..., description="Caller turns used by the resonance model.")
    turns: list[Turn] = Field(..., description="Caller turns (seconds from start).")
    features: dict[str, float] = Field(
        ..., description="Formant/pitch feature vector fed to the classifier."
    )
