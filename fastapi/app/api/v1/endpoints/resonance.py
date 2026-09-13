"""``POST /detect/resonance`` — classify a caller using only formant/pitch
resonance-stability physics, standalone and independent of the timing/lexical
detectors in ``compare.py``.

Measures how the caller's vocal tract behaves (formant jitter, pitch-formant
coupling, long-timescale drift, vocoder frame periodicity) rather than
turn-taking timing or transcript content. See ``app/services/resonance_features.py``
for what each of the 7 underlying features means.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from app.schemas.detect import Turn
from app.schemas.resonance import (
    ResonanceDetectRequest,
    ResonanceDetectResponse,
    ResonanceDetectVerboseResponse,
)
from app.services.classifier import DETECTORS
from app.services.resonance_features import extract_resonance_features
from app.services.turns import caller_audio_and_turns_from_wav, decode_base64_wav

logger = logging.getLogger(__name__)
router = APIRouter()

detector = DETECTORS["resonance_stability"]


@router.post(
    "/detect/resonance",
    response_model=None,
    summary="Classify a caller using only formant/pitch resonance physics",
    description=(
        "Returns `{is_synthetic, confidence}`. Pass `?verbose=true` to also get "
        "the caller turns and the resonance feature vector."
    ),
)
def detect_resonance(
    request: ResonanceDetectRequest,
    verbose: bool = Query(
        False, description="Include turns and features alongside the verdict."
    ),
) -> ResonanceDetectResponse | ResonanceDetectVerboseResponse:
    # 1. Obtain caller audio + turns: either supplied turns (testing, still
    #    needs the audio decoded since resonance can't work from turns alone)
    #    or derived via the framework's VAD from the raw audio (production).
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

    # 2. Feature engineering (Praat formant/pitch tracking, frame by frame).
    features = extract_resonance_features(
        caller_audio, sample_rate, turns, request.channel
    )
    if features is None:
        raise HTTPException(
            status_code=422,
            detail="Not enough voiced signal to classify (need >=20 voiced frames).",
        )

    # 3. Classify.
    if not detector.ready:
        raise HTTPException(
            status_code=503,
            detail="Detector not loaded. Run scripts/train.py to build artifacts.",
        )
    is_synthetic, confidence, _probability = detector.predict(features)
    verdict = ResonanceDetectResponse(
        is_synthetic=is_synthetic, confidence=round(confidence, 4)
    )
    if not verbose:
        return verdict

    caller_turns = [
        turn for turn in turns if int(turn.get("channel", -1)) == request.channel
    ]
    return ResonanceDetectVerboseResponse(
        **verdict.model_dump(),
        channel=request.channel,
        n_turns=len(caller_turns),
        turns=[Turn(**turn) for turn in caller_turns],
        features=features,
    )


@router.get("/detect/resonance/health", summary="Resonance detector readiness")
def resonance_health() -> dict[str, object]:
    return {
        "status": "ok",
        "detector_ready": detector.ready,
        "features": len(detector.feature_cols),
        "val_auc": detector.metadata.get("val_auc"),
    }
