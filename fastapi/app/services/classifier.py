"""Classifier service: independent per-feature-family detectors.

Each feature family (turn-timing "distribution_time", endpoint-acoustics
"natural_speech_termination", and any added later) is trained and scored
independently - see ``scripts/train.py``. ``/detect`` runs every family whose
inputs are available for a given request and reports each one's own verdict,
rather than forcing a single combined score. Add a new family by giving it a
subdirectory under ``app/artifacts/`` and registering it in ``DETECTORS``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.core import config

logger = logging.getLogger(__name__)


class Detector:
    """Thin wrapper around one family's scaler + logistic-regression pair."""

    def __init__(self, name: str, artifacts_dir: Path) -> None:
        self.name = name
        self.artifacts_dir = artifacts_dir
        self.model: Any | None = None
        self.scaler: Any | None = None
        self.calibrator: Any | None = None
        self.feature_cols: list[str] = []
        self.metadata: dict[str, Any] = {}

    @property
    def ready(self) -> bool:
        return self.model is not None and self.scaler is not None

    def load(self) -> "Detector":
        """Load artifacts from disk. Raises if they are missing.

        ``calibrator.joblib`` is optional - a family trained without calibration
        (e.g. distribution_time) simply won't have one, and predict() below skips
        the calibration step entirely when self.calibrator is None.
        """
        model_path = self.artifacts_dir / "detector.joblib"
        scaler_path = self.artifacts_dir / "scaler.joblib"
        metadata_path = self.artifacts_dir / "metadata.json"
        calibrator_path = self.artifacts_dir / "calibrator.joblib"
        if not model_path.exists() or not scaler_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Artifacts for '{self.name}' not found in {self.artifacts_dir}. "
                "Run `python scripts/train.py` from the fastapi/ directory first."
            )
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.calibrator = joblib.load(calibrator_path) if calibrator_path.exists() else None
        if metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.feature_cols = self.metadata.get("feature_cols", self.feature_cols)
        logger.info(
            "Detector '%s' loaded | features=%d | val_auc=%s | calibrated=%s",
            self.name,
            len(self.feature_cols),
            self.metadata.get("val_auc"),
            self.calibrator is not None,
        )
        return self

    def predict(self, features: dict[str, float]) -> tuple[bool, float, float]:
        """Return ``(is_synthetic, confidence, proba_synthetic)``.

        ``confidence`` is the model's probability for the predicted class, so it
        always lies in ``[0.5, 1.0]``. ``proba_synthetic`` is the raw P(synthetic)
        (Platt-calibrated when a calibrator is present - see scripts/train.py's
        ``calibrate=True`` path), exposed separately so callers combining multiple
        families (see ``app/api/v1/endpoints/detect.py``) work with a consistent,
        trustworthy scale rather than each family's own "confidence in its own
        predicted class".
        """
        if not self.ready:
            raise RuntimeError(f"Detector '{self.name}' is not loaded")

        vector = np.array([features[c] for c in self.feature_cols], dtype=float)
        scaled = self.scaler.transform(vector.reshape(1, -1))
        proba_synthetic = float(self.model.predict_proba(scaled)[0, 1])
        if self.calibrator is not None:
            proba_synthetic = float(self.calibrator.predict_proba([[proba_synthetic]])[0, 1])
        is_synthetic = proba_synthetic >= 0.5
        confidence = proba_synthetic if is_synthetic else 1.0 - proba_synthetic
        return is_synthetic, confidence, proba_synthetic


#: Registry of feature-family detectors served by /detect.
DETECTORS: dict[str, Detector] = {
    "distribution_time": Detector(
        "distribution_time", config.ARTIFACTS_DIR / "distribution_time"
    ),
    "natural_speech_termination": Detector(
        "natural_speech_termination", config.ARTIFACTS_DIR / "natural_speech_termination"
    ),
    "resonance_stability": Detector(
        "resonance_stability", config.ARTIFACTS_DIR / "resonance_stability"
    ),
}


def load_all() -> None:
    """Load every registered detector, warning (not failing) on missing artifacts."""
    for detector in DETECTORS.values():
        try:
            detector.load()
        except FileNotFoundError as exc:
            logger.warning("%s", exc)
