"""Audio -> caller turns using the colab HRI framework.

Issue #20 asks for a real communication path between the ``colab`` submodule
and the FastAPI service. Instead of shelling out to
``colab/_localPipeline/stt_test.py`` (explicitly *not* the goal), we import the
submodule's framework directly:

* ``colab/hri/microservices/stt/device_utils.py`` provides the device /
  compute-type detection used across the HRI STT microservice.
* The framework's VAD is the upstream ``faster_whisper.vad`` Silero model
  (the same one ``transcriber_faster_whisper.WhisperModel(..., vad_filter=True)``
  uses inside the submodule).

We turn a raw challenge clip into the same ``{"channel", "start", "end"}``
records that the Altur dataset ships in ``hackmty26/turns/*.json``.
"""

from __future__ import annotations

import base64
import io
import logging
import sys
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from app.core import config

logger = logging.getLogger(__name__)

_COLAB_PATH_READY = False


def ensure_colab_on_path() -> None:
    """Add the colab STT microservice to ``sys.path`` (idempotent)."""
    global _COLAB_PATH_READY
    if _COLAB_PATH_READY:
        return
    stt_dir = str(config.COLAB_STT_DIR)
    if config.COLAB_STT_DIR.is_dir() and stt_dir not in sys.path:
        sys.path.insert(0, stt_dir)
        logger.info("Added colab STT framework to sys.path: %s", stt_dir)
    _COLAB_PATH_READY = True


def framework_device() -> tuple[str, str]:
    """Device + compute type reported by the colab framework's ``device_utils``.

    Falls back to CPU/int8 if the submodule is not checked out.
    """
    ensure_colab_on_path()
    try:
        from device_utils import detect_device_and_compute_type  # type: ignore

        return detect_device_and_compute_type()
    except Exception as exc:  # pragma: no cover - depends on submodule presence
        logger.warning("Could not import colab device_utils (%s); using cpu/int8", exc)
        return "cpu", "int8"


def decode_base64_wav(audio_base64: str) -> tuple[np.ndarray, int]:
    """Decode a base64 WAV clip into a ``(frames, channels)`` float32 array."""
    try:
        raw = base64.b64decode(audio_base64, validate=False)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid base64 audio payload: {exc}") from exc

    data, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    if data.size == 0:
        raise ValueError("Decoded audio is empty")
    return data, int(sample_rate)


def _resample_to_target(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == config.TARGET_SAMPLE_RATE:
        return audio.astype(np.float32)
    # resample_poly wants an integer up/down ratio.
    gcd = np.gcd(sample_rate, config.TARGET_SAMPLE_RATE)
    up = config.TARGET_SAMPLE_RATE // gcd
    down = sample_rate // gcd
    return resample_poly(audio, up, down).astype(np.float32)


def extract_turns_vad(
    audio_mono: np.ndarray, sample_rate: int, channel: int = config.CALLER_CHANNEL
) -> list[dict[str, Any]]:
    """Extract speech turns with the framework's Silero VAD (fast path)."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    audio16 = _resample_to_target(audio_mono, sample_rate)
    options = VadOptions(
        threshold=config.VAD_THRESHOLD,
        min_silence_duration_ms=config.VAD_MIN_SILENCE_MS,
        speech_pad_ms=config.VAD_SPEECH_PAD_MS,
    )
    segments = get_speech_timestamps(
        audio16, vad_options=options, sampling_rate=config.TARGET_SAMPLE_RATE
    )
    turns = [
        {
            "channel": channel,
            "start": round(seg["start"] / config.TARGET_SAMPLE_RATE, 3),
            "end": round(seg["end"] / config.TARGET_SAMPLE_RATE, 3),
        }
        for seg in segments
    ]
    logger.info("VAD produced %d caller turns", len(turns))
    return turns


def extract_turns_whisper(
    audio_mono: np.ndarray, sample_rate: int, channel: int = config.CALLER_CHANNEL
) -> list[dict[str, Any]]:
    """Extract turns using the colab framework's ``WhisperModel`` (slow path).

    Kept as an opt-in alternative (``TURNS_MODE=whisper``) to demonstrate the
    framework integration end to end. It requires downloading a Whisper model.
    """
    ensure_colab_on_path()
    from transcriber_faster_whisper import WhisperModel  # type: ignore

    device, compute_type = framework_device()
    audio16 = _resample_to_target(audio_mono, sample_rate)
    model = WhisperModel(
        "base",
        device=device,
        compute_type=compute_type,
        download_root=str(config.COLAB_STT_DIR / "models"),
    )
    segments, _info = model.transcribe(
        audio16, language="es", vad_filter=True, without_timestamps=False
    )
    turns = [
        {
            "channel": channel,
            "start": round(float(seg.start), 3),
            "end": round(float(seg.end), 3),
        }
        for seg in segments
    ]
    logger.info("Whisper produced %d caller turns", len(turns))
    return turns


def caller_audio_and_turns_from_wav(
    audio_base64: str, channel: int = config.CALLER_CHANNEL
) -> tuple[np.ndarray, int, list[dict[str, Any]]]:
    """Full pipeline: base64 stereo WAV -> (native-rate caller mono audio, sample_rate, turns).

    Added for endpoint-acoustics feature extraction (Prosidy branch), which needs the
    raw caller waveform alongside the turn boundaries. The audio is returned at its
    native sample rate (unresampled) - turn timestamps are in seconds, so they apply
    directly to it regardless of what sample rate VAD internally resampled to.
    """
    data, sample_rate = decode_base64_wav(audio_base64)
    if channel >= data.shape[1]:
        raise ValueError(
            f"Requested channel {channel} but audio has {data.shape[1]} channel(s)"
        )
    mono = data[:, channel]

    if config.TURNS_MODE == "whisper":
        turns = extract_turns_whisper(mono, sample_rate, channel)
    else:
        turns = extract_turns_vad(mono, sample_rate, channel)
    return mono, sample_rate, turns


def caller_turns_from_wav(
    audio_base64: str, channel: int = config.CALLER_CHANNEL
) -> list[dict[str, Any]]:
    """Full pipeline: base64 stereo WAV -> caller-channel speech turns."""
    _, _, turns = caller_audio_and_turns_from_wav(audio_base64, channel)
    return turns


def warmup() -> None:
    """Force the VAD model to load so the first request is not cold."""
    try:
        if config.TURNS_MODE == "vad":
            from faster_whisper.vad import get_speech_timestamps  # noqa: F401

        device, compute_type = framework_device()
        logger.info(
            "Turn extractor warmup | mode=%s | framework device=%s/%s",
            config.TURNS_MODE,
            device,
            compute_type,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("Warmup skipped: %s", exc)
