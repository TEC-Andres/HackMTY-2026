"""Timing-based detection endpoints.

* ``POST /detect/timeDiff`` — turn-taking dynamics only.
* ``POST /detect/STTLexicalAnalysis`` — side-by-side timing vs lexical detectors.

The two signals answer the same question ("is the caller synthetic?") from
orthogonal evidence:

* **timing** — ``tiemposDeDistribucion``: turn-taking dynamics / VAD features
  served by the existing ``POST /detect/timeDiff`` model.
* **lexical** — issue #21: STT transcript -> the five lexical variables ->
  PLS-DA + VIP model (``app.services.lexical``).

The response keeps both verdicts, whether they agree, and a mean-probability
ensemble. The point is comparison/diagnostics, so it is more expensive than
``/detect/timeDiff``: it runs full speech-to-text on the clip.
"""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

from app.schemas.compare import (
    CompareRequest,
    CompareResponse,
    DetectorResult,
    LexicalResult,
    ResonanceResult,
    TimingResult,
)
from app.schemas.detect import (
    DetectRequest,
    DetectResponse,
    DetectVerboseResponse,
    Turn,
)
from app.services.classifier import DETECTORS
from app.services.features import extract_features
from app.services.lexical import lexical_detector
from app.services.report import render_comparison
from app.services.resonance_features import extract_resonance_features
from app.services.turns import (
    caller_audio_and_turns_from_wav,
    caller_turns_from_wav,
    decode_base64_wav,
)

logger = logging.getLogger(__name__)
router = APIRouter()

#: Turn-taking ("distribution_time") family, served by POST /detect/timeDiff.
detector = DETECTORS["distribution_time"]

#: Formant/pitch-physics family - independent evidence from lexical/timing,
#: added alongside them in /detect/STTLexicalAnalysis (see _run_resonance below).
resonance_detector = DETECTORS["resonance_stability"]


def _run_timing(request: CompareRequest) -> TimingResult | None:
    """Best-effort turn-taking verdict; ``None`` when it cannot be computed."""
    if request.turns is not None:
        turns = [turn.model_dump() for turn in request.turns]
    else:
        turns = caller_turns_from_wav(request.audio_base64, request.channel)

    features = extract_features(turns, request.channel)
    if features is None or not detector.ready:
        return None

    is_synthetic, confidence, probability = detector.predict(features)
    n_turns = len(
        [t for t in turns if int(t.get("channel", -1)) == request.channel]
    )
    return TimingResult(
        is_synthetic=is_synthetic,
        confidence=round(confidence, 4),
        synthetic_probability=round(probability, 4),
        n_turns=n_turns,
    )


def _run_resonance(request: CompareRequest) -> ResonanceResult | None:
    """Best-effort formant/pitch verdict; ``None`` when it cannot be computed.

    Unlike timing/lexical, this needs the raw caller waveform (Praat formant
    tracking), not just turn boundaries - CompareRequest.audio_base64 is
    required, so it's always available here regardless of whether the caller
    also supplied precomputed turns.
    """
    if request.turns is not None:
        data, sample_rate = decode_base64_wav(request.audio_base64)
        caller_audio = data[:, request.channel]
        turns = [turn.model_dump() for turn in request.turns]
    else:
        caller_audio, sample_rate, turns = caller_audio_and_turns_from_wav(
            request.audio_base64, request.channel
        )

    features = extract_resonance_features(
        caller_audio, sample_rate, turns, request.channel
    )
    if features is None or any(np.isnan(v) for v in features.values()):
        return None
    if not resonance_detector.ready:
        return None

    is_synthetic, confidence, probability = resonance_detector.predict(features)
    return ResonanceResult(
        is_synthetic=is_synthetic,
        confidence=round(confidence, 4),
        synthetic_probability=round(probability, 4),
    )


@router.post(
    "/detect/timeDiff",
    response_model=None,
    summary="Classify a caller as human or synthetic",
    description=(
        "Returns `{is_synthetic, confidence}`. Pass `?verbose=true` to also get "
        "the caller turns and the timing feature vector."
    ),
)
def detect_time_diff(
    request: DetectRequest,
    verbose: bool = Query(
        False, description="Include turns and features alongside the verdict."
    ),
) -> DetectResponse | DetectVerboseResponse:
    # 1. Obtain caller turns: either supplied (testing) or derived via the
    #    colab framework's VAD from the raw audio (production path).
    if request.turns is not None:
        turns = [turn.model_dump() for turn in request.turns]
        logger.info("Using %d turns supplied in request", len(turns))
    else:
        try:
            turns = caller_turns_from_wav(request.audio_base64 or "", request.channel)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - surface framework failures clearly
            logger.exception("Turn extraction failed")
            raise HTTPException(
                status_code=500, detail=f"Turn extraction failed: {exc}"
            ) from exc

    # 2. Feature engineering (same maths as the playground).
    features = extract_features(turns, request.channel)
    if features is None:
        raise HTTPException(
            status_code=422,
            detail="Not enough caller turn structure to classify (need >=4 turns).",
        )

    # 3. Classify.
    if not detector.ready:
        raise HTTPException(
            status_code=503,
            detail="Detector not loaded. Run scripts/train.py to build artifacts.",
        )
    is_synthetic, confidence, _probability = detector.predict(features)
    verdict = DetectResponse(
        is_synthetic=is_synthetic, confidence=round(confidence, 4)
    )
    if not verbose:
        return verdict

    caller_turns = [
        turn for turn in turns if int(turn.get("channel", -1)) == request.channel
    ]
    return DetectVerboseResponse(
        **verdict.model_dump(),
        channel=request.channel,
        n_turns=len(caller_turns),
        turns=[Turn(**turn) for turn in caller_turns],
        features=features,
    )


@router.post(
    "/detect/STTLexicalAnalysis",
    response_model=None,
    summary="Compare the timing detector with the lexical (#21) detector",
    description=(
        "Returns the ensemble verdict `{is_synthetic, confidence}`. Pass "
        "`?verbose=true` for the per-detector breakdown (timing, lexical, "
        "agreement, contributions and report). Use `?format=text` for the "
        "plain comparison table."
    ),
)
def compare(
    request: CompareRequest,
    format: str = Query(
        "json",
        pattern="^(json|text)$",
        description="json (default) or text for the plain comparison table.",
    ),
    verbose: bool = Query(
        False,
        description="Return the full per-detector breakdown instead of just "
        "the ensemble verdict.",
    ),
) -> CompareResponse | DetectResponse | PlainTextResponse:
    # ── timing detector (best effort) ──────────────────────────────────────
    timing: TimingResult | None
    try:
        timing = _run_timing(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - a timing failure must not kill lexical
        logger.exception("Timing detector failed")
        timing = None

    # ── lexical detector (required) ────────────────────────────────────────
    if not lexical_detector.ready:
        raise HTTPException(
            status_code=503,
            detail="Lexical detector not loaded. Train it in "
            "_playingGround/lexicalAnalysis (artifacts/lexical_model.joblib).",
        )
    try:
        lexical_payload, _segments = lexical_detector.predict_from_audio(
            request.audio_base64, request.channel
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Lexical analysis failed")
        raise HTTPException(
            status_code=500, detail=f"Lexical analysis failed: {exc}"
        ) from exc

    lexical = LexicalResult(**lexical_payload)

    # ── resonance detector (best effort, like timing) ──────────────────────
    resonance: ResonanceResult | None
    try:
        resonance = _run_resonance(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - a resonance failure must not kill lexical
        logger.exception("Resonance detector failed")
        resonance = None

    # ── comparison + mean-probability ensemble ─────────────────────────────
    # `agreement` stays timing-vs-lexical only (unchanged) - resonance only
    # joins the probability average below.
    probabilities = [lexical.synthetic_probability]
    if timing is not None:
        probabilities.append(timing.synthetic_probability)
    if resonance is not None:
        probabilities.append(resonance.synthetic_probability)
    ensemble_probability = sum(probabilities) / len(probabilities)
    ensemble = DetectorResult(
        is_synthetic=ensemble_probability >= 0.5,
        confidence=round(max(ensemble_probability, 1 - ensemble_probability), 4),
        synthetic_probability=round(ensemble_probability, 4),
    )
    agreement = (
        None if timing is None else timing.is_synthetic == lexical.is_synthetic
    )

    report = render_comparison(timing, lexical, ensemble)
    logger.info("\n%s", report)
    if format == "text":
        return PlainTextResponse(report)

    if not verbose:
        return DetectResponse(
            is_synthetic=ensemble.is_synthetic,
            confidence=ensemble.confidence,
        )

    return CompareResponse(
        timing=timing,
        lexical=lexical,
        resonance=resonance,
        agreement=agreement,
        ensemble=ensemble,
        report=report,
    )


@router.get("/detect/timeDiff/health", summary="Service and model readiness")
def time_diff_health() -> dict[str, object]:
    return {
        "status": "ok",
        "detector_ready": detector.ready,
        "features": len(detector.feature_cols),
        "val_auc": detector.metadata.get("val_auc"),
    }


@router.get("/detect/STTLexicalAnalysis/health", summary="Both detectors' readiness")
def compare_health() -> dict[str, object]:
    return {
        "status": "ok",
        "timing_ready": detector.ready,
        "timing_val_auc": detector.metadata.get("val_auc"),
        "lexical_ready": lexical_detector.ready,
        "lexical_cv_auc": lexical_detector.metadata.get("plsda_cv_auc_mean"),
        "lexical_pls_components": lexical_detector.metadata.get("pls_components"),
        "resonance_ready": resonance_detector.ready,
        "resonance_val_auc": resonance_detector.metadata.get("val_auc"),
    }
