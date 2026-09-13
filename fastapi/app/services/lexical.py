"""Issue #21 lexical detector: caller transcript -> human/synthetic verdict.

Wraps the leakage-controlled model trained in
``_playingGround/lexicalAnalysis/train_lexical.py`` (PLS-DA + VIP, with a
calibrated logistic head and the per-fold char-n-gram fluency LM).

Because the model consumes the five lexical variables from issue #21, the
response also exposes each variable group's additive log-odds contribution,
which is what makes the two detectors comparable rather than just two opaque
numbers.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import joblib
import numpy as np

from app.core import config

logger = logging.getLogger(__name__)

_PATH_READY = False

#: Prefix -> human-readable #21 variable name.
VARIABLE_GROUPS: dict[str, str] = {
    "correct": "sentence_correctness",
    "filler": "fillers_over_fillers",
    "congru": "sentence_congruence",
    "repair": "error_correction_pronunciation",
    "idea": "main_sub_idea_decomposition",
    "lm": "fluency_language_model",
    "call": "call_level_controls",
}


def ensure_lexical_on_path() -> None:
    """Add the playground lexical-analysis dir to ``sys.path`` (idempotent)."""
    global _PATH_READY
    if _PATH_READY:
        return
    directory = str(config.LEXICAL_DIR)
    if config.LEXICAL_DIR.is_dir() and directory not in sys.path:
        sys.path.insert(0, directory)
        logger.info("Added lexical analysis dir to sys.path: %s", directory)
    _PATH_READY = True


def _group_of(feature_name: str) -> str | None:
    for prefix, group in VARIABLE_GROUPS.items():
        if feature_name.startswith(prefix + "_"):
            return group
    return None


class LexicalDetector:
    """Scaler + PLS-DA/logistic pair trained on the #21 lexical features."""

    def __init__(self) -> None:
        self.artifact: dict[str, Any] | None = None
        self.metadata: dict[str, Any] = {}

    @property
    def ready(self) -> bool:
        return self.artifact is not None

    def load(self) -> "LexicalDetector":
        # The artifact pickles ``build_features.CharNGramLM``, so the lexical
        # dir must be importable before unpickling.
        ensure_lexical_on_path()
        if not config.LEXICAL_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Lexical model not found. Run the training in "
                f"_playingGround/lexicalAnalysis first. Looked in {config.LEXICAL_MODEL_PATH}."
            )
        self.artifact = joblib.load(config.LEXICAL_MODEL_PATH)
        if config.LEXICAL_METADATA_PATH.exists():
            self.metadata = json.loads(
                config.LEXICAL_METADATA_PATH.read_text(encoding="utf-8")
            )
        logger.info(
            "Lexical detector loaded | features=%d | cv_auc=%s | lm=%s",
            len(self.artifact.get("feature_names", [])),
            self.metadata.get("plsda_cv_auc_mean"),
            self.artifact.get("lm") is not None,
        )
        return self

    # ── feature assembly ───────────────────────────────────────────────────
    def _features_from_segments(self, segments: list[dict[str, Any]]) -> dict[str, float]:
        assert self.artifact is not None
        ensure_lexical_on_path()
        from build_features import call_features  # type: ignore

        features = call_features({"segments": segments})

        caller_text = " ".join(
            str(s.get("text", "")).strip()
            for s in segments
            if int(s.get("channel", -1)) == config.CALLER_CHANNEL
        )
        lm = self.artifact.get("lm")
        if lm is not None and caller_text:
            logprobs = lm.logprobs(caller_text)
            features["lm_logprob_mean"] = (
                float(np.mean(logprobs)) if logprobs else 0.0
            )
            features["lm_logprob_std"] = float(np.std(logprobs)) if logprobs else 0.0
        else:
            features["lm_logprob_mean"] = 0.0
            features["lm_logprob_std"] = 0.0
        return features

    def predict(self, segments: list[dict[str, Any]]) -> dict[str, Any]:
        """Score caller transcript segments; returns the lexical verdict."""
        if not self.ready:
            raise RuntimeError("Lexical detector is not loaded")

        features = self._features_from_segments(segments)
        names = self.artifact["feature_names"]
        vector = np.array([features.get(n, 0.0) for n in names], dtype=float).reshape(1, -1)
        scaled = self.artifact["scaler"].transform(vector)

        probability = float(self.artifact["logistic"].predict_proba(scaled)[0, 1])
        plsda_score = float(self.artifact["pls"].predict(scaled)[:, 1][0])

        is_synthetic = probability >= 0.5
        confidence = probability if is_synthetic else 1.0 - probability

        coefficients = self.artifact["logistic"].coef_[0]
        contributions: dict[str, float] = {group: 0.0 for group in set(VARIABLE_GROUPS.values())}
        for index, name in enumerate(names):
            group = _group_of(name)
            if group is not None:
                contributions[group] += float(coefficients[index] * scaled[0, index])

        caller_segments = [
            s for s in segments if int(s.get("channel", -1)) == config.CALLER_CHANNEL
        ]
        return {
            "is_synthetic": bool(is_synthetic),
            "confidence": round(confidence, 4),
            "synthetic_probability": round(probability, 4),
            "plsda_score": round(plsda_score, 4),
            "n_segments": len(caller_segments),
            "n_words": int(
                sum(len(s.get("words") or []) for s in caller_segments)
            ),
            "variable_contributions": {
                k: round(v, 4) for k, v in sorted(contributions.items())
            },
        }

    def predict_from_audio(
        self, audio_base64: str, channel: int = config.CALLER_CHANNEL
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Transcribe the caller channel and score it."""
        from app.services.stt import transcribe_channel

        segments = transcribe_channel(audio_base64, channel=channel)
        if not segments:
            raise ValueError("No speech detected in the caller channel")
        return self.predict(segments), segments


lexical_detector = LexicalDetector()
