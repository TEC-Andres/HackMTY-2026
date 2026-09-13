"""Central configuration and filesystem paths for the detection service.

The service lives in ``<repo>/fastapi/app``, so the repository root is three
parents up. Every path can be overridden with an environment variable which
keeps the code portable across machines and CI.
"""

from __future__ import annotations

import os
from pathlib import Path

# <repo>/fastapi/app/core/config.py -> parents[0]=core [1]=app [2]=fastapi [3]=repo
FASTAPI_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = FASTAPI_DIR.parent


def _env_path(var: str, default: Path) -> Path:
    """Return an absolute path from an env var, falling back to a default."""
    value = os.getenv(var)
    return Path(value).expanduser().resolve() if value else default.resolve()


#: Altur challenge data (manifest.csv + turns/ + optionally audio/).
HACKMTY26_DIR = _env_path("HACKMTY26_DIR", REPO_ROOT / "hackmty26")

#: Root of the colab submodule (framework source).
COLAB_DIR = _env_path("COLAB_DIR", REPO_ROOT / "colab")

#: The reusable STT/VAD microservice inside the colab submodule.
#: This directory is added to ``sys.path`` so its modules can be imported
#: directly (see ``app.services.turns``).
COLAB_STT_DIR = _env_path(
    "COLAB_STT_DIR", COLAB_DIR / "hri" / "microservices" / "stt"
)

#: The vendored local STT pipeline (faster-whisper transcription + models/).
#: Added to ``sys.path`` by ``app.services.stt`` for the lexical endpoint.
COLAB_LOCAL_PIPELINE_DIR = _env_path(
    "COLAB_LOCAL_PIPELINE_DIR", COLAB_DIR / "_localPipeline"
)

#: Issue #21 lexical analysis: shared feature code + trained lexical model.
LEXICAL_DIR = _env_path("LEXICAL_DIR", REPO_ROOT / "_playingGround" / "lexicalAnalysis")
LEXICAL_MODEL_PATH = LEXICAL_DIR / "artifacts" / "lexical_model.joblib"
LEXICAL_METADATA_PATH = LEXICAL_DIR / "artifacts" / "metadata.json"

#: Serialized model + scaler produced by ``scripts/train.py``.
ARTIFACTS_DIR = _env_path("ARTIFACTS_DIR", FASTAPI_DIR / "app" / "artifacts")
MODEL_PATH = ARTIFACTS_DIR / "detector.joblib"
SCALER_PATH = ARTIFACTS_DIR / "scaler.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"

#: Audio / inference settings aligned with the challenge contract.
TARGET_SAMPLE_RATE = 16_000  # Silero VAD / Whisper expect 16 kHz.
CALLER_CHANNEL = 0

#: Turn extraction strategy: "vad" (fast Silero) or "whisper" (framework model).
TURNS_MODE = os.getenv("TURNS_MODE", "vad").lower()

#: VAD tuning (seconds/milliseconds) for turn extraction.
VAD_MIN_SILENCE_MS = int(os.getenv("VAD_MIN_SILENCE_MS", "500"))
VAD_SPEECH_PAD_MS = int(os.getenv("VAD_SPEECH_PAD_MS", "100"))
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))

#: Lexical endpoint STT (issue #21). Multilingual model for Spanish telephony.
STT_MODEL = os.getenv("STT_MODEL", "small")
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "es")
STT_BEAM_SIZE = int(os.getenv("STT_BEAM_SIZE", "5"))

