"""
Builds a resonance-features CSV for the train/val split, using the
ground-truth turns/<id>.json segments provided with the dataset.

(At real inference time there's no turns file — see ../detector.py, which
uses vad.py instead. This script is only for building training data.)

Run (from anywhere):
    python resonance_features.py --n-per-class 999 --split train --out resonance_features_train_full.csv
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent  # resonance-detector/
sys.path.insert(0, str(ROOT))
from resonance_core import features_from_segments  # noqa: E402

DATA_DIR = ROOT.parent / "hackmty26"


def call_features(cid: str) -> dict:
    audio, sr = sf.read(DATA_DIR / "audio" / f"{cid}.wav")
    ch0 = audio[:, 0]
    with open(DATA_DIR / "turns" / f"{cid}.json") as f:
        turns = [t for t in json.load(f)["turns"] if t["channel"] == 0]
    segments = [(t["start"], t["end"]) for t in turns]
    return features_from_segments(ch0, sr, segments)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=30)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="resonance_features_sample.csv")
    args = ap.parse_args()

    df = pd.read_csv(DATA_DIR / "manifest.csv")
    df = df[df.split == args.split]

    rows = []
    for label in ["human", "synthetic"]:
        subset = df[df.label == label].head(args.n_per_class)
        for _, row in subset.iterrows():
            feats = call_features(row.anon_id)
            feats["anon_id"] = row.anon_id
            feats["label"] = label
            rows.append(feats)
            print(f"{label:10s} {row.anon_id}  n_voiced={feats.get('n_voiced_frames')}")

    out = pd.DataFrame(rows)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out)} rows to {out_path}")

    print("\n=== Mean by class ===")
    print(out.groupby("label").mean(numeric_only=True).T)


if __name__ == "__main__":
    main()
