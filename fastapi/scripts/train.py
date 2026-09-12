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

#: Turn-timing features (features.py) + endpoint-acoustics features (acoustic_features.py).
#: Persisted into metadata.json so the FastAPI service builds the same vector at inference.
COMBINED_FEATURE_COLS = FEATURE_COLS + ACOUSTIC_FEATURE_COLS


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
            continue
        features.update(acoustic)

        features["anon_id"] = record.anon_id
        features["label"] = record.label
        features["split"] = record.split
        rows.append(features)

    print(
        f"Loaded {len(rows)} calls ({skipped} skipped for insufficient turns, "
        f"{skipped_acoustic} skipped for missing/insufficient audio)"
    )
    return pd.DataFrame(rows)


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
        help="Where to write detector.joblib, scaler.joblib, metadata.json.",
    )
    args = parser.parse_args()

    df = build_dataset(args.hackmty26_dir)
    train = df[df.split == "train"]
    val = df[df.split == "val"]
    if train.empty or val.empty:
        raise SystemExit("Need both train and val splits to fit and evaluate.")

    y_train = (train.label == "synthetic").astype(int).values
    y_val = (val.label == "synthetic").astype(int).values

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[COMBINED_FEATURE_COLS].values)
    x_val = scaler.transform(val[COMBINED_FEATURE_COLS].values)

    model = LogisticRegression(
        max_iter=5000, class_weight="balanced", C=1.0, random_state=0
    )
    model.fit(x_train, y_train)

    train_auc = roc_auc_score(y_train, model.predict_proba(x_train)[:, 1])
    val_auc = roc_auc_score(y_val, model.predict_proba(x_val)[:, 1])
    print(f"Train AUC: {train_auc:.3f}  Val AUC: {val_auc:.3f}")
    print(f"Train calls: {len(train)}  Val calls: {len(val)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out_dir / "detector.joblib")
    joblib.dump(scaler, args.out_dir / "scaler.joblib")
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_cols": COMBINED_FEATURE_COLS,
        "train_auc": round(float(train_auc), 4),
        "val_auc": round(float(val_auc), 4),
        "train_calls": int(len(train)),
        "val_calls": int(len(val)),
        "model": "LogisticRegression(class_weight=balanced, C=1.0)",
        "label_positive": "synthetic",
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
