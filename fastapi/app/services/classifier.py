"""Classifier service: load persisted artifacts and score a feature vector."""

from __future__ import annotations

import json
import logging
from typing import Any

import joblib
import numpy as np

from app.core import config
from app.services.features import FEATURE_COLS

logger = logging.getLogger(__name__)


class Detector:
    """Thin wrapper around the scaler + logistic-regression pair."""

    def __init__(self) -> None:
        self.model: Any | None = None
        self.scaler: Any | None = None
        self.feature_cols: list[str] = list(FEATURE_COLS)
        self.metadata: dict[str, Any] = {}

    @property
    def ready(self) -> bool:
        return self.model is not None and self.scaler is not None

    def load(self) -> "Detector":
        """Load artifacts from disk. Raises if they are missing."""
        if not config.MODEL_PATH.exists() or not config.SCALER_PATH.exists():
            raise FileNotFoundError(
                "Detector artifacts not found. Run `python scripts/train.py` from "
                f"the fastapi/ directory first. Looked in {config.ARTIFACTS_DIR}."
            )
        self.model = joblib.load(config.MODEL_PATH)
        self.scaler = joblib.load(config.SCALER_PATH)
        if config.METADATA_PATH.exists():
            self.metadata = json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
            self.feature_cols = self.metadata.get("feature_cols", self.feature_cols)
        logger.info(
            "Detector loaded | features=%d | val_auc=%s",
            len(self.feature_cols),
            self.metadata.get("val_auc"),
        )
        return self

    def predict(self, features: dict[str, float]) -> tuple[bool, float]:
        """Return ``(is_synthetic, confidence)``.

        ``confidence`` is the model's probability for the predicted class, so
        it always lies in ``[0.5, 1.0]``.
        """
        if not self.ready:
            raise RuntimeError("Detector is not loaded")

        vector = np.array([features[c] for c in self.feature_cols], dtype=float)
        scaled = self.scaler.transform(vector.reshape(1, -1))
        proba_synthetic = float(self.model.predict_proba(scaled)[0, 1])
        is_synthetic = proba_synthetic >= 0.5
        confidence = proba_synthetic if is_synthetic else 1.0 - proba_synthetic
        return is_synthetic, confidence


detector = Detector()
