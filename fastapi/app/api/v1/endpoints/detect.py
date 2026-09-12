"""``POST /detect`` — decide whether a caller is human or synthetic.

Contract (hackmty26/README.md):
    Input : stereo WAV clip (8 kHz, base64, channel 0 = caller)
    Output: {"is_synthetic": bool, "confidence": float}

Runs every registered feature family (distribution_time, natural_speech_termination,
...) whose inputs are available for this request, and reports each one's own verdict
in ``breakdown`` alongside the required top-level fields (currently a temporary
equal-weight average of the families that ran).
"""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException

from app.core import config
from app.schemas.detect import DetectRequest, DetectResponse, FamilyResult
from app.services.acoustic_features import extract_acoustic_features
from app.services.classifier import DETECTORS
from app.services.features import extract_features
from app.services.turns import caller_audio_and_turns_from_wav

logger = logging.getLogger(__name__)
router = APIRouter()


def _depad_turns(turns: list[dict], pad_s: float) -> list[dict]:
    """Undo VAD's speech_pad_ms before endpoint-acoustics analysis.

    Live VAD (extract_turns_vad) pads every turn outward by config.VAD_SPEECH_PAD_MS
    so downstream STT doesn't clip soft onsets/offsets - harmless for distribution_time's
    gap statistics, but fatal for natural_speech_termination, which is anchored to the
    exact acoustic offset: by the padded "end", the real energy drop already happened
    before the measurement window starts. The training data (hackmty26/turns/*.json)
    has no such pad, so we remove it here rather than retraining on padded boundaries.
    """
    depadded = []
    for t in turns:
        start, end = t["start"] + pad_s, t["end"] - pad_s
        if end <= start:
            continue  # turn was shorter than 2x the pad - nothing meaningful left
        depadded.append({**t, "start": start, "end": end})
    return depadded


@router.post(
    "/detect",
    response_model=DetectResponse,
    summary="Classify a caller as human or synthetic",
)
def detect(request: DetectRequest) -> DetectResponse:
    caller_audio = None
    sample_rate = None

    # 1. Obtain caller turns (+ raw audio when available - only the audio path
    #    can feed the natural_speech_termination family).
    if request.turns is not None:
        turns = [turn.model_dump() for turn in request.turns]
        logger.info("Using %d turns supplied in request (no audio -> "
                    "natural_speech_termination will be skipped)", len(turns))
    else:
        try:
            caller_audio, sample_rate, turns = caller_audio_and_turns_from_wav(
                request.audio_base64 or "", request.channel
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - surface framework failures clearly
            logger.exception("Turn extraction failed")
            raise HTTPException(
                status_code=500, detail=f"Turn extraction failed: {exc}"
            ) from exc

    breakdown: dict[str, FamilyResult] = {}
    probas_synthetic: list[float] = []

    # 2a. distribution_time (turn-taking timing) - needs only the turns.
    dist_detector = DETECTORS["distribution_time"]
    dist_features = extract_features(turns, request.channel)
    if dist_features is not None and dist_detector.ready:
        is_synth, confidence, proba = dist_detector.predict(dist_features)
        breakdown["distribution_time"] = FamilyResult(
            is_synthetic=is_synth, confidence=round(confidence, 4)
        )
        probas_synthetic.append(proba)

    # 2b. natural_speech_termination (endpoint acoustics) - needs raw audio.
    term_detector = DETECTORS["natural_speech_termination"]
    term_features = None
    if caller_audio is not None:
        # Only the default "vad" mode's pad amount is known/config-driven here;
        # "whisper" mode's internal VAD uses its own (unexposed) padding.
        acoustic_turns = (
            _depad_turns(turns, config.VAD_SPEECH_PAD_MS / 1000)
            if config.TURNS_MODE != "whisper"
            else turns
        )
        term_features = extract_acoustic_features(
            caller_audio, sample_rate, acoustic_turns, request.channel
        )
        if term_features is not None and any(np.isnan(v) for v in term_features.values()):
            term_features = None
    if term_features is not None and term_detector.ready:
        is_synth, confidence, proba = term_detector.predict(term_features)
        breakdown["natural_speech_termination"] = FamilyResult(
            is_synthetic=is_synth, confidence=round(confidence, 4)
        )
        probas_synthetic.append(proba)

    if not probas_synthetic:
        raise HTTPException(
            status_code=422,
            detail=(
                "Not enough signal to classify: need >=4 caller turns for "
                "distribution_time, or raw audio (audio_base64) for "
                "natural_speech_termination."
            ),
        )

    # 3. TEMPORARY: equal-weight average of whichever families ran, for the
    # required top-level fields. Replace once family weighting is decided.
    avg_proba = float(np.mean(probas_synthetic))
    is_synthetic = avg_proba >= 0.5
    confidence = avg_proba if is_synthetic else 1.0 - avg_proba

    return DetectResponse(
        is_synthetic=is_synthetic, confidence=round(confidence, 4), breakdown=breakdown
    )


@router.get("/detect/health", summary="Service and model readiness")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "detectors": {
            name: {
                "ready": det.ready,
                "features": len(det.feature_cols),
                "val_auc": det.metadata.get("val_auc"),
            }
            for name, det in DETECTORS.items()
        },
    }
