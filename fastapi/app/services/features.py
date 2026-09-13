"""Turn-taking feature engineering.

This is a direct port of the feature extraction that lives in
``_playingGround/minMaxConfidence.py`` (lines 26-99). The only change is that
it is now a pure, reusable function instead of top-level script code, so the
same maths can be used both to train the offline model and to score a live
``POST /detect/timeDiff`` request.

The intuition: a synthetic caller produces turn-taking dynamics (gaps between
caller utterances, their regularity, reaction to the agent) that differ from a
human caller. We describe those dynamics with summary statistics and let a
classifier separate them.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

#: Exact feature order produced below. Persisted with the model so that
#: inference always builds the vector in the same order as training.
FEATURE_COLS: list[str] = [
    "frac_gt5",
    "turn_dur_mean",
    "mean_rel_roc",
    "mean_gap",
    "std_gap",
    "cv_gap",
    "mad_gap",
    "iqr_gap",
    "range_gap",
    "mean_abs_roc",
    "std_roc",
    "slope",
    "norm_slope",
    "mean_abs_jerk",
    "std_jerk",
    "p90_p50",
    "p50_p10",
    "frac_gt2",
    "frac_gt3",
    "skew",
    "kurt",
    "entropy",
    "ac1",
    "sign_changes",
    "mult_roc",
    "peak_ratio",
    "gaps_per_s",
    "turns_per_s",
    "silence_fraction",
    "turn_dur_std",
    "turn_dur_cv",
    "z_roc_mean",
    "z_roc_std",
    "z_gap_std",
]


def turns_for_channel(turns: Iterable[dict[str, Any]], channel: int = 0) -> list[dict[str, Any]]:
    """Return the turns of one channel, sorted by start time."""
    selected = [t for t in turns if int(t["channel"]) == channel]
    return sorted(selected, key=lambda t: float(t["start"]))


def extract_features(
    turns: Iterable[dict[str, Any]], channel: int = 0
) -> dict[str, float] | None:
    """Build the feature dict for one call.

    Returns ``None`` when the caller channel does not contain enough turns to
    compute the gap statistics (mirrors the guards in the playground script).
    """
    ch0 = turns_for_channel(turns, channel)
    if len(ch0) < 4:
        return None

    turn_durs = np.array([float(t["end"]) - float(t["start"]) for t in ch0])
    gaps = np.array(
        [float(ch0[i]["start"]) - float(ch0[i - 1]["end"]) for i in range(1, len(ch0))]
    )
    gaps = gaps[gaps >= 0]
    if len(gaps) < 3:
        return None

    dur = float(ch0[-1]["end"]) - float(ch0[0]["start"])
    if dur <= 0:
        return None

    d1 = np.diff(gaps)
    d2 = np.diff(gaps, 2)
    tt = np.arange(len(gaps), dtype=float)
    slope = np.polyfit(tt, gaps, 1)[0]
    q = np.percentile(gaps, [10, 25, 50, 75, 90])

    gz = (gaps - gaps.mean()) / (gaps.std() + 1e-9)
    ac1 = (
        np.corrcoef(gaps[:-1], gaps[1:])[0, 1]
        if gaps.std() > 0 and len(gaps) > 2
        else 0.0
    )
    sign_changes = np.mean(np.diff(np.sign(d1)) != 0) if len(d1) > 1 else 0.0
    fft_mag = np.abs(np.fft.rfft(gaps - gaps.mean()))
    peak_ratio = (
        fft_mag[1:].max() / (fft_mag[0] + 1e-9) if len(fft_mag) > 1 else 0
    )
    hist, _ = np.histogram(gaps, bins=8)
    p = hist / hist.sum()
    p = p[p > 0]
    entropy = -(p * np.log2(p)).sum()

    return {
        "frac_gt5": float((gaps > 5).mean()),
        "turn_dur_mean": float(turn_durs.mean()),
        "mean_rel_roc": float(np.abs(d1 / np.maximum(gaps[:-1], 1e-6)).mean()),
        "mean_gap": float(gaps.mean()),
        "std_gap": float(gaps.std()),
        "cv_gap": float(gaps.std() / (gaps.mean() + 1e-9)),
        "mad_gap": float(np.median(np.abs(gaps - np.median(gaps)))),
        "iqr_gap": float(q[3] - q[1]),
        "range_gap": float(gaps.max() - gaps.min()),
        "mean_abs_roc": float(np.abs(d1).mean()),
        "std_roc": float(d1.std()),
        "slope": float(slope),
        "norm_slope": float(slope / (gaps.mean() + 1e-9)),
        "mean_abs_jerk": float(np.abs(d2).mean() if len(d2) > 0 else 0),
        "std_jerk": float(d2.std() if len(d2) > 0 else 0),
        "p90_p50": float(q[4] / (q[2] + 1e-9)),
        "p50_p10": float(q[2] / (q[0] + 1e-9)),
        "frac_gt2": float((gaps > 2).mean()),
        "frac_gt3": float((gaps > 3).mean()),
        "skew": float(pd.Series(gaps).skew()),
        "kurt": float(pd.Series(gaps).kurt()),
        "entropy": float(entropy),
        "ac1": float(ac1),
        "sign_changes": float(sign_changes),
        "mult_roc": float(np.abs(np.diff(np.log(np.maximum(gaps, 1e-6)))).mean()),
        "peak_ratio": float(peak_ratio),
        "gaps_per_s": float(len(gaps) / dur),
        "turns_per_s": float(len(ch0) / dur),
        "silence_fraction": float(gaps.sum() / dur),
        "turn_dur_std": float(turn_durs.std()),
        "turn_dur_cv": float(turn_durs.std() / (turn_durs.mean() + 1e-9)),
        "z_roc_mean": float(np.abs(np.diff(gz)).mean()),
        "z_roc_std": float(np.abs(np.diff(gz)).std()),
        "z_gap_std": float(gz.std()),
    }


def feature_vector(features: dict[str, float]) -> np.ndarray:
    """Return the feature dict as an ordered 1-D array ready for the scaler."""
    return np.array([features[c] for c in FEATURE_COLS], dtype=float)
