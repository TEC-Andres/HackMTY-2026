"""Pydantic models for ``POST /detect/STTLexicalAnalysis``.

Compares the turn-taking detector (``tiemposDeDistribucion`` /
``POST /detect/timeDiff``)
with the issue #21 lexical detector on the same clip.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.detect import Turn


class CompareRequest(BaseModel):
    """The lexical side needs audio (it runs STT), so ``audio_base64`` is required.

    ``turns`` is still accepted for the timing side so the two detectors can be
    fed the exact same segmentation.
    """

    audio_base64: str = Field(
        ..., description="Base64-encoded stereo 8 kHz WAV; channel 0 is the caller."
    )
    turns: list[Turn] | None = Field(
        default=None,
        description="Optional precomputed turns for the timing detector.",
    )
    channel: int = Field(default=0, description="Caller channel to analyse.")


class DetectorResult(BaseModel):
    """Common verdict shape for one detector or the ensemble."""

    is_synthetic: bool
    confidence: float = Field(..., ge=0.0, le=1.0, description="P(predicted class)")
    synthetic_probability: float = Field(
        ..., ge=0.0, le=1.0, description="P(caller is synthetic)"
    )


class TimingResult(DetectorResult):
    n_turns: int = Field(..., description="Caller turns used by the timing model")


class ResonanceResult(DetectorResult):
    """Formant/pitch-physics verdict; same shape as the base result (no extra
    diagnostics, unlike TimingResult/LexicalResult - see resonance_features.py
    for what the underlying 7 features mean)."""


class LexicalResult(DetectorResult):
    plsda_score: float
    n_segments: int
    n_words: int
    variable_contributions: dict[str, float] = Field(
        ...,
        description="Additive log-odds per issue #21 lexical variable group.",
    )


class CompareResponse(BaseModel):
    timing: TimingResult | None = Field(
        default=None,
        description="Turn-taking detector; null if the clip had too few turns or "
        "the timing model is not loaded.",
    )
    lexical: LexicalResult | None = Field(
        default=None,
        description="Lexical (#21) detector; null if the lexical model is not "
        "loaded (it is not part of the current shipped ensemble - see "
        "app.services.lexical) or the clip had no usable transcript.",
    )
    resonance: ResonanceResult | None = Field(
        default=None,
        description="Formant/pitch-physics detector; null if the clip had too "
        "little voiced signal or the resonance model is not loaded. Not "
        "included in `agreement` (kept as timing-vs-lexical) or in the "
        "format=text report - only in ensemble and this field.",
    )
    agreement: bool | None = Field(
        default=None,
        description="Whether both detectors predicted the same class; null if "
        "the timing side was unavailable.",
    )
    ensemble: DetectorResult = Field(
        ..., description="Mean-probability fusion of the available detectors."
    )
    ensemble_method: str = "mean_probability"
    report: str = Field(
        ..., description="Human-readable comparison table (same as format=text)."
    )
