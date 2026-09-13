"""Resonance-stability feature engineering: does the caller's vocal tract behave
like a physical vocal tract, or like a synthesis model's approximation of one?

Companion to ``features.py`` (turn-taking timing) and ``acoustic_features.py``
(endpoint acoustics) - this family only looks at the waveform *within* each
caller turn (formants + pitch, frame by frame), never at turn boundaries or
timing between turns. All three families are independent inputs to the same
ensemble.

Seven features, validated on the hackmty26 train split (282 calls, 113 human /
169 synthetic) with individual AUCs of 0.60-0.83 standalone, 0.94-0.95 combined
(see resonance-detector/team_summary.html for the full write-up):

    F1_cv                mean-normalized spread of the first formant (F1)
                          across all voiced frames of the call
    F2_cv                same, for the second formant (F2)
    F1_jitter             mean frame-to-frame |delta F1| / mean F1
    F2_jitter             mean frame-to-frame |delta F2| / mean F2
    F0_F1_corr            correlation between pitch (F0) and F1 across voiced
                          frames - a real vocal tract couples them (shared
                          physiology); many TTS pipelines generate them more
                          independently
    F1_drift_abs           |mean F1 in the call's last third - first third| -
                          humans drift over a multi-minute call (posture,
                          fatigue); most TTS doesn't model that
    vocoder_periodicity    fraction of the F1-derivative's spectral energy
                          concentrated near a neural vocoder's typical
                          frame-hop rate (50-150Hz) - an architectural
                          artifact of frame-by-frame generation

Counter-intuitive finding versus the naive "flat resonance = synthetic"
hypothesis: on this dataset, F1_jitter/F1_cv/vocoder_periodicity all run
*higher* for synthetic callers, not lower - the TTS in use appears to
over-correct by injecting more micro-variation than a real vocal tract
produces. F0_F1_corr and F1_drift_abs run in the intuitive direction (lower
for synthetic). The classifier is fit on the data either way, so this only
matters for interpretation, not for correctness.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import parselmouth

#: Exact feature order. Persisted with the model so inference always builds the vector
#: in the same order as training - mirrors ``features.FEATURE_COLS``.
RESONANCE_FEATURE_COLS: list[str] = [
    "F1_cv",
    "F2_cv",
    "F1_jitter",
    "F2_jitter",
    "F0_F1_corr",
    "F1_drift_abs",
    "vocoder_periodicity",
]

_MIN_SEGMENT_S = 0.3
_MAX_FORMANT_HZ = 4000.0  # Nyquist at 8kHz is already 4kHz; telephone band only holds F1-F3.
_MIN_VOICED_FRAMES = 20  # below this, per-frame stats are too noisy to trust
_MIN_PERIODICITY_FRAMES = 30  # below this, the F1-derivative spectrum is too coarse


def turns_for_channel(turns: Iterable[dict[str, Any]], channel: int = 0) -> list[dict[str, Any]]:
    """Return the turns of one channel, sorted by start time. Mirrors ``features.py``."""
    selected = [t for t in turns if int(t["channel"]) == channel]
    return sorted(selected, key=lambda t: float(t["start"]))


def _voiced_series(caller: np.ndarray, sr: int, turns_ch0: list[dict[str, Any]]):
    """Run Praat formant + pitch tracking over each turn, keep only voiced frames.

    Returns concatenated (f0s, f1s, f2s) across all turns, plus:

    * ``raw_turn_f1s``: per-turn raw (uniformly-spaced, unfiltered) F1
      trajectories with their true frame step, for the vocoder-periodicity
      spectral analysis - that needs uniform spacing, which the voiced-only
      series (with gaps where frames were dropped) doesn't have.
    * ``turn_f1s`` / ``turn_f2s``: per-turn voiced F1/F2 frames, so the jitter
      features can difference within a turn instead of across turn boundaries.
    """
    f0s: list[float] = []
    f1s: list[float] = []
    f2s: list[float] = []
    raw_turn_f1s: list[tuple[np.ndarray, float]] = []
    turn_f1s: list[np.ndarray] = []
    turn_f2s: list[np.ndarray] = []

    for t in turns_ch0:
        start_s, end_s = float(t["start"]), float(t["end"])
        s, e = int(start_s * sr), int(end_s * sr)
        seg = caller[s:e]
        if len(seg) < sr * _MIN_SEGMENT_S:
            continue

        snd = parselmouth.Sound(seg, sampling_frequency=sr)
        formant = snd.to_formant_burg(max_number_of_formants=5, maximum_formant=_MAX_FORMANT_HZ)
        pitch = snd.to_pitch()
        ts = formant.ts()

        raw_f1 = np.array([formant.get_value_at_time(1, tt) or np.nan for tt in ts])
        if len(ts) > 1:
            raw_turn_f1s.append((raw_f1, float(np.mean(np.diff(ts)))))

        turn_f1: list[float] = []
        turn_f2: list[float] = []
        for i, tt in enumerate(ts):
            f0 = pitch.get_value_at_time(tt)
            if not f0 or np.isnan(f0):
                continue
            f1 = raw_f1[i]
            f2 = formant.get_value_at_time(2, tt)
            if not f1 or not f2 or np.isnan(f1) or np.isnan(f2):
                continue
            f0s.append(f0)
            f1s.append(f1)
            f2s.append(f2)
            turn_f1.append(f1)
            turn_f2.append(f2)

        if turn_f1:
            turn_f1s.append(np.array(turn_f1))
            turn_f2s.append(np.array(turn_f2))

    return (
        np.array(f0s),
        np.array(f1s),
        np.array(f2s),
        raw_turn_f1s,
        turn_f1s,
        turn_f2s,
    )


def _voiced_runs(raw_f1: np.ndarray) -> list[np.ndarray]:
    """Split a uniformly-sampled F1 trajectory at its unvoiced (NaN) gaps.

    Dropping the NaNs and diffing the survivors would place samples seconds
    apart next to each other while keeping the original frame step ``dt``,
    manufacturing spurious periodicity. Keeping each contiguous voiced run
    intact preserves the true spacing for the downstream FFT.
    """
    runs: list[np.ndarray] = []
    start: int | None = None
    for i, voiced in enumerate(~np.isnan(raw_f1)):
        if voiced and start is None:
            start = i
        elif not voiced and start is not None:
            runs.append(raw_f1[start:i])
            start = None
    if start is not None:
        runs.append(raw_f1[start:])
    return runs


def _within_turn_diffs(turn_series: list[np.ndarray]) -> np.ndarray:
    """Absolute frame-to-frame differences, computed independently per turn.

    Differencing the cross-turn concatenation would add a spurious jump
    between each turn's last frame and the next turn's first, so jitter would
    reflect segmentation/silence rather than within-turn vocal-tract motion.
    """
    diffs = [np.abs(np.diff(series)) for series in turn_series if len(series) > 1]
    return np.concatenate(diffs) if diffs else np.array([], dtype=float)


def _vocoder_periodicity(raw_turn_f1s: list[tuple[np.ndarray, float]]) -> float:
    """Energy fraction of each turn's F1-derivative spectrum concentrated near
    a neural vocoder's typical frame-hop rate (50-150Hz), vs. total energy -
    averaged across contiguous voiced runs. NaN when no run has enough frames
    to judge."""
    scores: list[float] = []
    for raw_f1, dt in raw_turn_f1s:
        if dt <= 0:
            continue
        for f1 in _voiced_runs(raw_f1):
            if len(f1) < _MIN_PERIODICITY_FRAMES:
                continue
            d = np.diff(f1)
            d = d - d.mean()
            spec = np.abs(np.fft.rfft(d))
            freqs = np.fft.rfftfreq(len(d), d=dt)
            band = (freqs > 50) & (freqs < 150)
            if not band.any() or spec.sum() == 0:
                continue
            scores.append(float(spec[band].sum() / spec.sum()))
    return float(np.mean(scores)) if scores else float("nan")


def extract_resonance_features(
    caller: np.ndarray, sr: int, turns: Iterable[dict[str, Any]], channel: int = 0
) -> dict[str, float] | None:
    """Build the resonance-stability feature dict for one call.

    ``caller`` must be the mono waveform for ``channel`` (float, any amplitude
    scale - only relative levels matter, same contract as
    ``acoustic_features.extract_acoustic_features``). Deliberately does not
    de-pad VAD's speech_pad_ms the way natural_speech_termination must: every
    feature here is an aggregate over all voiced frames within a turn, not
    anchored to an exact boundary instant, and padded silence at a turn's
    edges simply has no detectable pitch so it drops out of the voiced-frame
    filter on its own.

    Returns ``None`` when there isn't enough voiced signal to trust the
    statistics (mirrors the guards in the other two families).
    """
    ch0 = turns_for_channel(turns, channel)
    if not ch0:
        return None

    f0s, f1s, f2s, raw_turn_f1s, turn_f1s, turn_f2s = _voiced_series(
        caller, sr, ch0
    )
    if len(f1s) < _MIN_VOICED_FRAMES:
        return None

    d1 = _within_turn_diffs(turn_f1s)
    d2 = _within_turn_diffs(turn_f2s)
    if len(d1) == 0 or len(d2) == 0:
        return None
    third = max(len(f1s) // 3, 1)
    drift_f1 = float(f1s[-third:].mean() - f1s[:third].mean())

    return {
        "F1_cv": float(f1s.std() / f1s.mean()),
        "F2_cv": float(f2s.std() / f2s.mean()),
        "F1_jitter": float(d1.mean() / f1s.mean()),
        "F2_jitter": float(d2.mean() / f2s.mean()),
        "F0_F1_corr": float(np.corrcoef(f0s, f1s)[0, 1]),
        "F1_drift_abs": abs(drift_f1),
        "vocoder_periodicity": _vocoder_periodicity(raw_turn_f1s),
    }


def resonance_feature_vector(features: dict[str, float]) -> np.ndarray:
    """Return the feature dict as an ordered 1-D array. Mirrors ``features.feature_vector``."""
    return np.array([features[c] for c in RESONANCE_FEATURE_COLS], dtype=float)
