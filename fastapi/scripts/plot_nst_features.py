#!/usr/bin/env python3
"""Visualize how the 5 Natural Speech Termination (NST) features differ between
human and synthetic callers, across the full labeled dataset (train + val).

This is the justification artifact for using these 5 features / NST at all - not
a model-evaluation script (see evaluate.py for held-out /detect performance).
Features are computed from the dataset's own ground-truth turns/*.json (the same
methodology used to train and validate the natural_speech_termination detector -
see acoustic_features.py), not the live VAD path.

Run from the ``fastapi/`` directory:

    python scripts/plot_nst_features.py
    python scripts/plot_nst_features.py --out ../reports/nst_features.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import mannwhitneyu

FASTAPI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FASTAPI_DIR))

from app.core import config  # noqa: E402
from app.services.acoustic_features import (  # noqa: E402
    ACOUSTIC_FEATURE_COLS,
    extract_acoustic_features,
)

FEATURE_LABELS = {
    "abruptness_db_mean": "Endpoint abruptness\n(dB drop at turn end)",
    "tail_ms_mean": "Tail duration\n(ms to reach noise floor)",
    "floor_db": "Ambient/line floor\n(dBFS)",
    "voiced_frac_near_end_mean": "Voiced fraction\nnear turn end",
    "echo_delay_ms_mean": "Echo delay\n(ms, 8-60ms band)",
}

HUMAN_COLOR = "#4C72B0"
SYNTH_COLOR = "#DD8452"


def build_dataset(hackmty26_dir: Path) -> pd.DataFrame:
    manifest = pd.read_csv(hackmty26_dir / "manifest.csv")
    rows: list[dict] = []
    skipped = 0
    for record in manifest.itertuples(index=False):
        turn_file = hackmty26_dir / "turns" / f"{record.anon_id}.json"
        audio_file = hackmty26_dir / "audio" / f"{record.anon_id}.wav"
        if not turn_file.exists() or not audio_file.exists():
            skipped += 1
            continue
        turns = json.loads(turn_file.read_text(encoding="utf-8"))["turns"]
        data, sr = sf.read(audio_file, dtype="float32", always_2d=True)
        caller = data[:, config.CALLER_CHANNEL].astype(np.float64)
        feats = extract_acoustic_features(caller, sr, turns, channel=config.CALLER_CHANNEL)
        if feats is None or any(np.isnan(v) for v in feats.values()):
            skipped += 1
            continue
        feats["anon_id"] = record.anon_id
        feats["label"] = record.label
        rows.append(feats)
    print(f"Loaded {len(rows)} calls ({skipped} skipped for missing files/insufficient turns)")
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out_path: Path) -> None:
    human = df[df.label == "human"]
    synth = df[df.label == "synthetic"]
    rng = np.random.default_rng(0)

    n = len(ACOUSTIC_FEATURE_COLS)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 5.2))
    for ax, col in zip(axes, ACOUSTIC_FEATURE_COLS):
        h, s = human[col].dropna(), synth[col].dropna()
        bp = ax.boxplot(
            [h.values, s.values],
            tick_labels=["human", "synthetic"],
            widths=0.5,
            showfliers=False,
            patch_artist=True,
        )
        for patch, color in zip(bp["boxes"], [HUMAN_COLOR, SYNTH_COLOR]):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)

        for i, vals in enumerate([h.values, s.values], start=1):
            x = rng.normal(i, 0.06, size=len(vals))
            ax.scatter(x, vals, s=8, alpha=0.35, color="black", zorder=3)

        _, p = mannwhitneyu(h, s, alternative="two-sided")
        pooled_sd = np.sqrt((h.std() ** 2 + s.std() ** 2) / 2) + 1e-9
        d = (h.mean() - s.mean()) / pooled_sd
        sig = "**" if p < 0.001 else ("*" if p < 0.05 else "")
        ax.set_title(f"{FEATURE_LABELS[col]}\nCohen's d={d:.2f}  p={p:.2g} {sig}", fontsize=9.5)

    fig.suptitle(
        "Natural Speech Termination (NST): human vs. synthetic callers\n"
        f"n_human={len(human)}  n_synthetic={len(synth)}  (full labeled dataset)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hackmty26-dir", type=Path, default=config.HACKMTY26_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=FASTAPI_DIR / "reports" / "nst_feature_distributions.png",
    )
    args = parser.parse_args()

    df = build_dataset(args.hackmty26_dir)
    plot(df, args.out)


if __name__ == "__main__":
    main()
