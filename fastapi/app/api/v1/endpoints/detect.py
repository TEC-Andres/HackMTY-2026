"""``POST /detect`` — decide whether a caller is human or synthetic.

Contract (hackmty26/README.md):
    Input : stereo WAV clip (8 kHz, base64, channel 0 = caller)
    Output: {"is_synthetic": bool, "confidence": float}
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.schemas.detect import DetectRequest, DetectResponse
from app.services.classifier import detector
from app.services.features import extract_features
from app.services.turns import caller_turns_from_wav

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/detect",
    response_model=DetectResponse,
    summary="Classify a caller as human or synthetic",
)
def detect(request: DetectRequest) -> DetectResponse:
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
    is_synthetic, confidence = detector.predict(features)
    return DetectResponse(is_synthetic=is_synthetic, confidence=round(confidence, 4))


@router.get("/detect/health", summary="Service and model readiness")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "detector_ready": detector.ready,
        "features": len(detector.feature_cols),
        "val_auc": detector.metadata.get("val_auc"),
    }
