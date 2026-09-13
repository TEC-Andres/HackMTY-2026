"""``POST /detect/acousticEnsemble`` — combine timing, endpoint-acoustics and
resonance-stability (no lexical/STT - see compare.py for that combination)
with an AUC-weighted average, so a family that's proven more accurate on
held-out validation data counts for more than one that's barely better than
a coin flip.

Ensemble formula (AUC-weighted mean)::

    combined = sum(w_i * p_i) / sum(w_i)      where w_i = family i's val_auc

An earlier version weighted each probability by itself (p_i^2 / p_i), which
turned out to only ever pull the combined score toward "synthetic": since
that formula rewards being close to 1 but not being close to 0, one family
leaning synthetic could outvote two others confidently saying human. AUC
weighting is symmetric - it doesn't care which way a family leans, only how
reliable it has proven to be.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from app.core import config
from app.schemas.acoustic_ensemble import (
    AcousticEnsembleRequest,
    AcousticEnsembleResponse,
    FamilyResult,
)
from app.services.acoustic_features import extract_acoustic_features
from app.services.classifier import DETECTORS
from app.services.features import extract_features
from app.services.resonance_features import extract_resonance_features
from app.services.turns import caller_audio_and_turns_from_wav, decode_base64_wav

logger = logging.getLogger(__name__)
router = APIRouter()

timing_detector = DETECTORS["distribution_time"]
nst_detector = DETECTORS["natural_speech_termination"]
resonance_detector = DETECTORS["resonance_stability"]


def _depad_turns(turns: list[dict[str, Any]], pad_s: float) -> list[dict[str, Any]]:
    """Undo VAD's speech_pad_ms before endpoint-acoustics analysis.

    natural_speech_termination is anchored to the exact acoustic offset, so a
    padded "end" would measure the tail starting after the real energy drop
    already happened. This helper no longer exists anywhere else in the
    codebase (it was dropped when detect.py's logic moved to compare.py,
    which doesn't call natural_speech_termination) - recreated here from the
    original implementation rather than left unhandled.
    """
    depadded = []
    for t in turns:
        start, end = t["start"] + pad_s, t["end"] - pad_s
        if end <= start:
            continue  # turn was shorter than 2x the pad - nothing meaningful left
        depadded.append({**t, "start": start, "end": end})
    return depadded


def _auc_weighted_average(weighted_probabilities: list[tuple[float, float]]) -> float:
    """combined = sum(w_i * p_i) / sum(w_i), where w_i is each family's own
    validation AUC.

    Weighting by a family's *proven* reliability (not by how large its own
    current prediction happens to be) is symmetric - a family that is very
    confident of "human" pulls the combined score just as much as one very
    confident of "synthetic", unlike the self-weighted-square formula tried
    first, which only ever pulled toward "synthetic". Falls back to a plain
    mean if no family has a usable AUC (e.g. artifacts missing val stats).
    """
    total_weight = sum(w for w, _ in weighted_probabilities)
    if total_weight <= 0:
        probs = [p for _, p in weighted_probabilities]
        return sum(probs) / len(probs)
    return sum(w * p for w, p in weighted_probabilities) / total_weight


def _detector_weight(detector) -> float:
    """A family's own validation AUC, used as its ensemble weight. Falls back
    to 0.5 (coin-flip, i.e. no real signal) if metadata is missing it."""
    auc = detector.metadata.get("val_auc")
    return float(auc) if auc is not None else 0.5


@router.post(
    "/detect/acousticEnsemble",
    response_model=None,
    summary="Combine timing, endpoint-acoustics and resonance (no lexical)",
    description=(
        "Returns `{is_synthetic, confidence}` from a self-weighted average "
        "(sum(p^2)/sum(p)) of whichever of the 3 acoustic families are "
        "available. Pass `?verbose=true` for the per-family breakdown."
    ),
)
def acoustic_ensemble(
    request: AcousticEnsembleRequest,
    verbose: bool = Query(
        False, description="Return the full per-family breakdown too."
    ),
) -> AcousticEnsembleResponse:
    # 1. Obtain caller audio + turns (both natural_speech_termination and
    #    resonance_stability need the raw waveform, not just turns).
    if request.turns is not None:
        try:
            data, sample_rate = decode_base64_wav(request.audio_base64)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        caller_audio = data[:, request.channel]
        turns = [turn.model_dump() for turn in request.turns]
        logger.info("Using %d turns supplied in request", len(turns))
    else:
        try:
            caller_audio, sample_rate, turns = caller_audio_and_turns_from_wav(
                request.audio_base64, request.channel
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - surface framework failures clearly
            logger.exception("Turn extraction failed")
            raise HTTPException(
                status_code=500, detail=f"Turn extraction failed: {exc}"
            ) from exc

    breakdown: dict[str, FamilyResult] = {}
    weighted_probabilities: list[tuple[float, float]] = []

    # ── distribution_time (turn-taking timing) ─────────────────────────────
    timing_features = extract_features(turns, request.channel)
    if timing_features is not None and timing_detector.ready:
        is_synth, confidence, proba = timing_detector.predict(timing_features)
        breakdown["distribution_time"] = FamilyResult(
            is_synthetic=is_synth,
            confidence=round(confidence, 4),
            synthetic_probability=round(proba, 4),
        )
        weighted_probabilities.append((_detector_weight(timing_detector), proba))

    # ── natural_speech_termination (endpoint acoustics) ────────────────────
    # Only de-pad when we derived turns ourselves via live VAD; turns supplied
    # directly by the caller (testing) are assumed unpadded, like the
    # dataset's ground-truth turns.json.
    acoustic_turns = (
        _depad_turns(turns, config.VAD_SPEECH_PAD_MS / 1000)
        if request.turns is None and config.TURNS_MODE != "whisper"
        else turns
    )
    nst_features = extract_acoustic_features(
        caller_audio, sample_rate, acoustic_turns, request.channel
    )
    if nst_features is not None and any(np.isnan(v) for v in nst_features.values()):
        nst_features = None
    if nst_features is not None and nst_detector.ready:
        is_synth, confidence, proba = nst_detector.predict(nst_features)
        breakdown["natural_speech_termination"] = FamilyResult(
            is_synthetic=is_synth,
            confidence=round(confidence, 4),
            synthetic_probability=round(proba, 4),
        )
        weighted_probabilities.append((_detector_weight(nst_detector), proba))

    # ── resonance_stability (formant/pitch physics) ────────────────────────
    resonance_features = extract_resonance_features(
        caller_audio, sample_rate, turns, request.channel
    )
    if resonance_features is not None and any(
        np.isnan(v) for v in resonance_features.values()
    ):
        resonance_features = None
    if resonance_features is not None and resonance_detector.ready:
        is_synth, confidence, proba = resonance_detector.predict(resonance_features)
        breakdown["resonance_stability"] = FamilyResult(
            is_synthetic=is_synth,
            confidence=round(confidence, 4),
            synthetic_probability=round(proba, 4),
        )
        weighted_probabilities.append((_detector_weight(resonance_detector), proba))

    if not weighted_probabilities:
        raise HTTPException(
            status_code=422,
            detail="Not enough signal: none of distribution_time / "
            "natural_speech_termination / resonance_stability could be computed.",
        )

    combined_proba = _auc_weighted_average(weighted_probabilities)
    is_synthetic = combined_proba >= 0.5
    confidence = combined_proba if is_synthetic else 1.0 - combined_proba

    return AcousticEnsembleResponse(
        is_synthetic=is_synthetic,
        confidence=round(confidence, 4),
        breakdown=breakdown if verbose else {},
    )


@router.get("/detect/acousticEnsemble/health", summary="All 3 acoustic detectors' readiness")
def acoustic_ensemble_health() -> dict[str, object]:
    return {
        "status": "ok",
        "distribution_time_ready": timing_detector.ready,
        "distribution_time_val_auc": timing_detector.metadata.get("val_auc"),
        "natural_speech_termination_ready": nst_detector.ready,
        "natural_speech_termination_val_auc": nst_detector.metadata.get("val_auc"),
        "resonance_stability_ready": resonance_detector.ready,
        "resonance_stability_val_auc": resonance_detector.metadata.get("val_auc"),
    }
