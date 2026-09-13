#!/usr/bin/env python3
"""Train and persist the human-vs-synthetic caller detector.

This reproduces the modelling step of ``_playingGround/minMaxConfidence.py``
but writes the fitted scaler + model to ``app/artifacts/`` so the FastAPI
service can serve them.

Run from the ``fastapi/`` directory:

    python scripts/train.py
    python scripts/train.py --hackmty26-dir ../hackmty26
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Make `app` importable when invoked as a script.
FASTAPI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FASTAPI_DIR))

from app.core import config  # noqa: E402
from app.services.acoustic_features import (  # noqa: E402
    ACOUSTIC_FEATURE_COLS,
    extract_acoustic_features,
)
from app.services.features import FEATURE_COLS, extract_features  # noqa: E402

#: Each feature family gets its own scaler + model, trained and scored independently
#: (see app/services/classifier.py). /detect runs every family whose inputs are
#: available and reports each one's own verdict, rather than one combined score.
#: Add a new family here (and its extract_*() in build_dataset below) without
#: touching the others.
FAMILIES: dict[str, list[str]] = {
    "distribution_time": FEATURE_COLS,
    "natural_speech_termination": ACOUSTIC_FEATURE_COLS,
}

#: Which families get a Platt/sigmoid calibrator on top of their raw predict_proba.
#: distribution_time is untouched (False) - only natural_speech_termination's
#: calibration was validated (see the calibration experiment: Brier 0.1437->0.1381,
#: log loss 0.4531->0.4345, AUC and 0.5-threshold accuracy unchanged).
CALIBRATE: dict[str, bool] = {
    "distribution_time": False,
    "natural_speech_termination": True,
}


def build_dataset(hackmty26_dir: Path) -> pd.DataFrame:
    """Turn ``manifest.csv`` + ``turns/*.json`` + ``audio/*.wav`` into a feature DataFrame."""
    manifest_path = hackmty26_dir / "manifest.csv"
    turns_dir = hackmty26_dir / "turns"
    audio_dir = hackmty26_dir / "audio"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    rows: list[dict] = []
    skipped = 0
    skipped_acoustic = 0
    for record in manifest.itertuples(index=False):
        turn_file = turns_dir / f"{record.anon_id}.json"
        if not turn_file.exists():
            skipped += 1
            continue
        payload = json.loads(turn_file.read_text(encoding="utf-8"))
        features = extract_features(payload["turns"], channel=config.CALLER_CHANNEL)
        if features is None:
            skipped += 1
            continue

        audio_file = audio_dir / f"{record.anon_id}.wav"
        acoustic = None
        if audio_file.exists():
            data, sr = sf.read(audio_file, dtype="float32", always_2d=True)
            caller = data[:, config.CALLER_CHANNEL].astype(np.float64)
            acoustic = extract_acoustic_features(
                caller, sr, payload["turns"], channel=config.CALLER_CHANNEL
            )
        if acoustic is None or any(np.isnan(v) for v in acoustic.values()):
            skipped_acoustic += 1
        else:
            features.update(acoustic)

        features["anon_id"] = record.anon_id
        features["label"] = record.label
        features["split"] = record.split
        rows.append(features)

    print(
        f"Loaded {len(rows)} calls ({skipped} skipped for insufficient turns, "
        f"{skipped_acoustic} with missing/insufficient audio for acoustic features)"
    )
    return pd.DataFrame(rows)


def _fit_platt_calibrator(raw_x: np.ndarray, y: np.ndarray, n_splits: int = 5) -> LogisticRegression:
    """Fit a Platt/sigmoid calibrator from out-of-fold training predictions only.

    Never touches the validation split, and never evaluates the final model's own
    (in-sample) predictions - each fold's held-out predictions come from a model
    that never saw that fold during fitting, matching the standard prevent-leakage
    calibration procedure. The calibrator itself is a plain 1-feature
    LogisticRegression on the raw probability, i.e. exactly Platt scaling
    (calibrated = sigmoid(A * raw_proba + B)) via a fully public sklearn API.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    oof_proba = np.zeros(len(y))
    for fold_train_idx, fold_hold_idx in cv.split(raw_x, y):
        fold_scaler = StandardScaler().fit(raw_x[fold_train_idx])
        fold_model = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0, random_state=0)
        fold_model.fit(fold_scaler.transform(raw_x[fold_train_idx]), y[fold_train_idx])
        oof_proba[fold_hold_idx] = fold_model.predict_proba(fold_scaler.transform(raw_x[fold_hold_idx]))[:, 1]
    return LogisticRegression().fit(oof_proba.reshape(-1, 1), y)


def train_family(df: pd.DataFrame, feature_cols: list[str], out_dir: Path, calibrate: bool = False) -> dict:
    """Fit one family's scaler + model (+ optional Platt calibrator) and persist it.

    When ``calibrate=True``: out-of-fold predictions are generated on the training
    split only (via a fresh scaler+model per fold), a sigmoid calibrator is fit on
    those OOF predictions, and only then is the final scaler+model refit on the
    *complete* training split - the calibrator is applied on top of that final
    model's output at inference time (see Detector.predict). The validation split
    is never used to fit anything, calibrator included.
    """
    family_df = df.copy()
    for col in feature_cols:
        if col not in family_df.columns:
            family_df[col] = np.nan
    family_df = family_df.dropna(subset=feature_cols)

    train = family_df[family_df.split == "train"]
    val = family_df[family_df.split == "val"]
    if train.empty or val.empty:
        raise SystemExit("Need both train and val splits to fit and evaluate.")

    y_train = (train.label == "synthetic").astype(int).values
    y_val = (val.label == "synthetic").astype(int).values
    raw_x_train = train[feature_cols].values

    calibrator = None
    if calibrate:
        calibrator = _fit_platt_calibrator(raw_x_train, y_train)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(raw_x_train)
    x_val = scaler.transform(val[feature_cols].values)

    model = LogisticRegression(
        max_iter=5000, class_weight="balanced", C=1.0, random_state=0
    )
    model.fit(x_train, y_train)

    # AUC is rank-invariant under Platt's monotonic sigmoid, so it's reported from
    # the raw model regardless of `calibrate` - calibration changes probability
    # values, not their ranking.
    train_auc = roc_auc_score(y_train, model.predict_proba(x_train)[:, 1])
    val_auc = roc_auc_score(y_val, model.predict_proba(x_val)[:, 1])

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "detector.joblib")
    joblib.dump(scaler, out_dir / "scaler.joblib")
    calibrator_path = out_dir / "calibrator.joblib"
    if calibrator is not None:
        joblib.dump(calibrator, calibrator_path)
    elif calibrator_path.exists():
        calibrator_path.unlink()  # stale calibrator from a previous run with calibrate=True
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_cols": feature_cols,
        "train_auc": round(float(train_auc), 4),
        "val_auc": round(float(val_auc), 4),
        "train_calls": int(len(train)),
        "val_calls": int(len(val)),
        "model": "LogisticRegression(class_weight=balanced, C=1.0)",
        "label_positive": "synthetic",
        "calibrated": calibrate,
        "calibration_method": "platt_sigmoid_oof_5fold" if calibrate else None,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hackmty26-dir",
        type=Path,
        default=config.HACKMTY26_DIR,
        help="Path to the hackmty26 submodule (manifest.csv + turns/).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=config.ARTIFACTS_DIR,
        help="Where to write <family>/detector.joblib, scaler.joblib, metadata.json.",
    )
    args = parser.parse_args()

    df = build_dataset(args.hackmty26_dir)

    for name, feature_cols in FAMILIES.items():
        metadata = train_family(df, feature_cols, args.out_dir / name, calibrate=CALIBRATE.get(name, False))
        print(
            f"[{name}] train AUC: {metadata['train_auc']:.3f}  "
            f"val AUC: {metadata['val_auc']:.3f}  "
            f"(train={metadata['train_calls']} val={metadata['val_calls']} "
            f"features={len(feature_cols)} calibrated={metadata['calibrated']})"
        )

    print(f"Saved artifacts under {args.out_dir}/<family>/")


if __name__ == "__main__":
    main()
