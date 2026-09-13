"""
Rebuilds val features using vad.py instead of turns/<id>.json, to check
whether the detector still works when segments come from real VAD
(as at actual inference time) instead of the dataset's ground-truth turns.

Run (from anywhere):
    python build_val_with_vad.py
"""
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent  # resonance-detector/
sys.path.insert(0, str(ROOT))
from resonance_core import features_from_segments  # noqa: E402
from vad import detect_speech_segments  # noqa: E402

DATA_DIR = ROOT.parent / "hackmty26"


def call_features_vad(cid: str) -> dict:
    audio, sr = sf.read(DATA_DIR / "audio" / f"{cid}.wav")
    ch0 = audio[:, 0]
    segments = detect_speech_segments(ch0, sr)
    return features_from_segments(ch0, sr, segments)


def main():
    df = pd.read_csv(DATA_DIR / "manifest.csv")
    df = df[df.split == "val"]

    rows = []
    for _, row in df.iterrows():
        feats = call_features_vad(row.anon_id)
        feats["anon_id"] = row.anon_id
        feats["label"] = row.label
        rows.append(feats)
        print(f"{row.label:10s} {row.anon_id}  n_voiced={feats.get('n_voiced_frames')}")

    out = pd.DataFrame(rows)
    out_path = ROOT / "resonance_features_val_vad.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
