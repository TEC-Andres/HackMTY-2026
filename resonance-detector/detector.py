"""
Resonance-stability detector: one signal among several in the team's
ensemble for POST /detect. Give it a raw call recording, get back a
synthetic-voice probability based on formant/pitch physics (see
resonance_core.py for what each feature means).

Usage (for Andres to wire into the actual endpoint):

    from detector import detect_from_wav_bytes

    result = detect_from_wav_bytes(wav_bytes)
    # {"is_synthetic": True, "confidence": 0.87}

`confidence` here is this module's own synthetic-probability score
(0-1) — meant to be combined with the team's other detectors (voting,
averaging, or a meta-classifier), not returned as the final system
verdict by itself.

CLI test:
    python detector.py path/to/call.wav
"""
import base64
import io
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resonance_core import features_from_segments, FEATURE_NAMES  # noqa: E402
from vad import detect_speech_segments  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
# Below this many voiced frames, features are unreliable — the caller likely
# barely spoke. Falls back to a neutral, low-confidence guess instead of
# feeding noise into the model.
MIN_VOICED_FRAMES = 20
FALLBACK_RESULT = {"is_synthetic": False, "confidence": 0.5}

_model_cache = None


def _load_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


def detect_from_channel0(ch0: np.ndarray, sr: int) -> dict:
    """Core entry point once you already have channel 0 as a float array."""
    segments = detect_speech_segments(ch0, sr)
    feats = features_from_segments(ch0, sr, segments)

    if feats.get("n_voiced_frames", 0) < MIN_VOICED_FRAMES:
        return dict(FALLBACK_RESULT)

    model = _load_model()
    x = pd.DataFrame([{name: feats[name] for name in model["feature_names"]}])
    proba_synthetic = float(model["pipeline"].predict_proba(x)[0, 1])

    return {
        "is_synthetic": proba_synthetic >= 0.5,
        "confidence": proba_synthetic,
    }


def detect_from_wav_bytes(wav_bytes: bytes) -> dict:
    """wav_bytes: raw bytes of a stereo WAV file (channel 0 = caller)."""
    audio, sr = sf.read(io.BytesIO(wav_bytes))
    if audio.ndim < 2:
        raise ValueError("Expected stereo WAV (channel 0 = caller, channel 1 = agent)")
    ch0 = audio[:, 0]
    return detect_from_channel0(ch0, sr)


def detect_from_base64(b64_audio: str) -> dict:
    """b64_audio: base64-encoded WAV, exactly what POST /detect receives."""
    return detect_from_wav_bytes(base64.b64decode(b64_audio))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python detector.py path/to/call.wav")
        sys.exit(1)
    with open(sys.argv[1], "rb") as f:
        result = detect_from_wav_bytes(f.read())
    print(result)
