"""Pydantic request/response models for ``POST /detect``."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Turn(BaseModel):
    channel: int = 0
    start: float = Field(..., description="Turn start in seconds from clip start")
    end: float = Field(..., description="Turn end in seconds from clip start")


class DetectRequest(BaseModel):
    """At least one of ``audio_base64`` or ``turns`` must be provided.

    ``audio_base64`` is the real challenge contract. ``turns`` exists so the
    classifier can be exercised/tested without decoding audio.
    """

    audio_base64: str | None = Field(
        default=None,
        description="Base64-encoded stereo 8 kHz WAV; channel 0 is the caller.",
    )
    turns: list[Turn] | None = Field(
        default=None,
        description="Optional precomputed turns (bypasses the VAD stage).",
    )
    channel: int = Field(default=0, description="Caller channel to analyse.")

    @model_validator(mode="after")
    def _require_input(self) -> "DetectRequest":
        if not self.audio_base64 and not self.turns:
            raise ValueError("Provide either 'audio_base64' or 'turns'")
        return self


class DetectResponse(BaseModel):
    is_synthetic: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
