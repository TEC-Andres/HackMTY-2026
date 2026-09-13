"""Caller speech-to-text for the lexical endpoint (issue #21).

Reuses the vendored local pipeline in ``colab/_localPipeline`` (the same code
``dialogue_stt.py`` uses offline) to turn a base64 stereo WAV into
speaker-tagged transcript segments. Channel isolation already separates the
caller (channel 0) from the agent (channel 1), so no diarization is needed.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from app.core import config
from app.services.turns import _resample_to_target, decode_base64_wav

logger = logging.getLogger(__name__)

_PATH_READY = False
_MODEL: Any | None = None
_MODEL_NAME: str | None = None


def ensure_local_pipeline_on_path() -> None:
    """Add ``colab/_localPipeline`` to ``sys.path`` (idempotent)."""
    global _PATH_READY
    if _PATH_READY:
        return
    pipeline = str(config.COLAB_LOCAL_PIPELINE_DIR)
    if config.COLAB_LOCAL_PIPELINE_DIR.is_dir() and pipeline not in sys.path:
        sys.path.insert(0, pipeline)
        logger.info("Added colab local STT pipeline to sys.path: %s", pipeline)
    _PATH_READY = True


def get_model(model_name: str | None = None) -> Any:
    """Load (and cache) the multilingual faster-whisper model, CPU by default."""
    global _MODEL, _MODEL_NAME
    ensure_local_pipeline_on_path()
    model_name = model_name or config.STT_MODEL
    if _MODEL is None or _MODEL_NAME != model_name:
        from device_utils import detect_device_and_compute_type  # type: ignore
        from transcriber_faster_whisper import WhisperModel  # type: ignore

        device, compute_type = detect_device_and_compute_type()
        logger.info("Loading STT model '%s' on %s (%s)", model_name, device, compute_type)
        _MODEL = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=str(config.COLAB_LOCAL_PIPELINE_DIR / "models"),
        )
        _MODEL_NAME = model_name
    return _MODEL


def transcribe_channel(
    audio_base64: str,
    channel: int = config.CALLER_CHANNEL,
    model_name: str | None = None,
    language: str | None = None,
) -> list[dict[str, Any]]:
    """Transcribe one channel and return Altur-shaped segment dicts."""
    model = get_model(model_name)
    language = language or config.STT_LANGUAGE

    data, sample_rate = decode_base64_wav(audio_base64)
    if channel >= data.shape[1]:
        raise ValueError(
            f"Requested channel {channel} but audio has {data.shape[1]} channel(s)"
        )
    audio16 = _resample_to_target(data[:, channel], sample_rate)

    segments, _info = model.transcribe(
        audio16,
        language=language,
        task="transcribe",
        vad_filter=True,
        word_timestamps=True,
        beam_size=config.STT_BEAM_SIZE,
        condition_on_previous_text=False,
    )

    out: list[dict[str, Any]] = []
    for seg in segments or []:
        text = (seg.text or "").strip()
        if not text:
            continue
        out.append(
            {
                "channel": int(channel),
                "speaker": f"user-{int(channel)}",
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "text": text,
                "avg_logprob": round(float(seg.avg_logprob), 4),
                "compression_ratio": round(float(seg.compression_ratio), 4),
                "no_speech_prob": round(float(seg.no_speech_prob), 4),
                "words": [
                    {
                        "word": w.word,
                        "probability": round(float(w.probability), 4),
                        "start": round(float(w.start), 3),
                        "end": round(float(w.end), 3),
                    }
                    for w in (seg.words or [])
                ],
            }
        )
    return out


def warmup() -> None:
    """Optionally preload the STT model (set STT_WARMUP=1)."""
    import os

    if os.getenv("STT_WARMUP", "0") not in ("1", "true", "yes"):
        return
    try:
        get_model()
        logger.info("STT model warmup complete")
    except Exception as exc:  # pragma: no cover - depends on model download
        logger.warning("STT warmup skipped: %s", exc)
