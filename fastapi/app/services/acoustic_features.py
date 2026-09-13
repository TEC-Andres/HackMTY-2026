"""Endpoint-acoustics feature engineering: how naturally caller speech decays into silence.

Companion to ``features.py`` (turn-taking timing). Where that module never looks at the
waveform itself, this one only looks at the waveform around each caller turn's end -
the two feature families are independent inputs to the same classifier.

Six features. The first five were validated on the hackmty26 train split (282 calls,
113 human / 169 synthetic) with duration-independent, statistically significant
separation; the sixth (`voiced_frac_near_end_range`) was added after a controlled NST v2
ablation (see scripts/nst_v2_experiment.py and scripts/nst_b_internal_ablation.py)
showed it alone reproduces the full benefit of a much larger 9-feature variant on live
(VAD+depad) validation - 80.3% vs the original 74.6%, identical confusion matrix - while
`median`/`std`/`iqr` variants of the same statistic proved redundant once `range` is
present. The feature set is frozen at these six; do not add more without a new
controlled ablation:

    abruptness_db_mean          dB energy drop right at the labeled turn end
                                 (human 18.0dB vs synthetic 25.9dB - synthetic cuts off harder)
    tail_ms_mean                 ms to decay into the call's own noise floor
                                 (human 72ms vs synthetic 19ms - synthetic reaches silence ~4x faster)
    floor_db                     the call's own ambient/line noise floor (once per call).
                                 NOTE: documented generalization risk - its correlation with
                                 the model's own output (Spearman ~0.44) is substantially
                                 stronger than its raw correlation with the true label
                                 (~0.21, not significant at p<0.05). Removing or discretizing
                                 it was tested and rejected: it drops live accuracy from 0.803
                                 to 0.62-0.68 (see the floor-robustness experiment), so it
                                 carries real signal, not just a dataset artifact - but this
                                 gap is a real risk if the grading set's recording pipeline
                                 differs from this dataset's, and is worth revisiting later.
    voiced_frac_near_end_mean    fraction of frames near turn-end that are clearly voiced
                                 (synthetic stays more cleanly voiced right up to the cut)
    echo_delay_ms_mean           dominant autocorrelation lag in the 8-60ms band, excluding
                                 lags that are just multiples of the pitch period
    voiced_frac_near_end_range   spread (max-min) of voiced_frac_near_end across a call's
                                 turns - humans vary turn-to-turn far more than synthetic
                                 callers (d~1.07 in the original ablation)

Candidates tested and dropped for showing no separation, redundancy, or live-pipeline
instability: pitch slope near the endpoint, raw decay slope in dB/ms, and (in the v2
ablation) abruptness_db/tail_ms distributional stats, plus voiced_frac_near_end's own
median/std/iqr once range is present. See the EXPERIMENTAL section below for the
superseded exploration code (not used by production).
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

#: Exact feature order. Persisted with the model so inference always builds the vector
#: in the same order as training - mirrors ``features.FEATURE_COLS``.
ACOUSTIC_FEATURE_COLS: list[str] = [
    "abruptness_db_mean",
    "tail_ms_mean",
    "floor_db",
    "voiced_frac_near_end_mean",
    "echo_delay_ms_mean",
    "voiced_frac_near_end_range",
]

_MIN_TURN_DUR_S = 0.6
_END_GUARD_MS = 150.0


def _rms_envelope_db(x: np.ndarray, sr: int, win_ms: float = 10.0) -> np.ndarray:
    win = max(1, int(sr * win_ms / 1000))
    n = len(x) // win
    if n == 0:
        return np.array([])
    x = x[: n * win].reshape(n, win).astype(np.float64)
    rms = np.sqrt(np.mean(x**2, axis=1) + 1e-12)
    return 20.0 * np.log10(np.maximum(rms, 1e-9))


def _call_floor_db(caller: np.ndarray, sr: int, turns_ch0: list[dict[str, Any]], pad_ms: float = 150.0) -> float:
    """Per-call ambient/line floor: median dBFS of regions with no caller speech nearby.

    Deliberately per-call rather than a fixed dBFS threshold - telephony lines vary in
    their own noise floor independently of who's speaking, so an absolute threshold
    conflates the two (see Phase 3 notes: a fixed threshold hit an identical -72.2dBFS
    floor on both human and synthetic calls, which is a property of the line, not the
    speaker).
    """
    mask = np.ones(len(caller), dtype=bool)
    pad = int(pad_ms / 1000 * sr)
    for t in turns_ch0:
        lo = max(0, int(t["start"] * sr) - pad)
        hi = min(len(caller), int(t["end"] * sr) + pad)
        mask[lo:hi] = False
    silence = caller[mask]
    if len(silence) < int(sr * 0.2):
        env_db = _rms_envelope_db(caller, sr)
        return float(np.percentile(env_db, 10)) if len(env_db) else -90.0
    env_db = _rms_envelope_db(silence, sr)
    return float(np.median(env_db)) if len(env_db) else -90.0


def _analyze_decay(
    caller: np.ndarray, sr: int, end_s: float, floor_db: float, pre_ms: float = 300.0, max_post_ms: float = 1000.0
) -> dict[str, float] | None:
    """Endpoint abruptness + adaptive tail duration around one turn's end."""
    end_idx = int(end_s * sr)
    lo = max(0, end_idx - int(pre_ms / 1000 * sr))
    hi = min(len(caller), end_idx + int(max_post_ms / 1000 * sr))
    seg = caller[lo:hi]
    if hi - end_idx < int(0.05 * sr):
        return None  # not enough post-end audio in this file to measure a tail

    env_db = _rms_envelope_db(seg, sr)
    end_frame = (end_idx - lo) // int(sr * 0.01)
    if end_frame >= len(env_db) or end_frame < 5:
        return None

    pre_level = float(np.mean(env_db[max(0, end_frame - 10) : end_frame]))
    post = env_db[end_frame:]
    if len(post) < 3:
        return None
    post_first = float(np.mean(post[: min(5, len(post))]))
    abruptness_db = pre_level - post_first

    thresh = floor_db + 3.0
    reached = np.where(post <= thresh)[0]
    converged = len(reached) > 0
    tail_ms = float(reached[0] * 10.0) if converged else float(max_post_ms)

    return dict(abruptness_db=abruptness_db, tail_ms=tail_ms, converged=float(converged))


def _autocorr_pitch(frame: np.ndarray, sr: int, fmin: float = 70.0, fmax: float = 400.0) -> tuple[float, float]:
    frame = frame - frame.mean()
    if np.allclose(frame, 0):
        return 0.0, 0.0
    ac = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
    if ac[0] <= 1e-12:
        return 0.0, 0.0
    ac = ac / ac[0]
    lag_min, lag_max = int(sr / fmax), min(len(ac) - 1, int(sr / fmin))
    if lag_max <= lag_min:
        return 0.0, 0.0
    seg = ac[lag_min : lag_max + 1]
    k = int(np.argmax(seg))
    lag = lag_min + k
    return (sr / lag if lag > 0 else 0.0), float(seg[k])


def _voiced_fraction_near_end(
    caller: np.ndarray,
    sr: int,
    end_s: float,
    clarity_thresh: float = 0.35,
    window_ms: float = 200.0,
    frame_ms: float = 40.0,
    hop_ms: float = 15.0,
) -> float:
    end_idx = int(end_s * sr)
    frame_len = int(frame_ms / 1000 * sr)
    hop = int(hop_ms / 1000 * sr)
    lo = max(0, end_idx - int(window_ms / 1000 * sr) - frame_len)
    voiced = 0
    n_frames = 0
    pos = lo
    while pos + frame_len <= end_idx:
        _, clarity = _autocorr_pitch(caller[pos : pos + frame_len], sr)
        n_frames += 1
        if clarity >= clarity_thresh:
            voiced += 1
        pos += hop
    return voiced / n_frames if n_frames else 0.0


def _echo_delay_ms(
    caller: np.ndarray,
    sr: int,
    end_s: float,
    pre_ms: float = 300.0,
    lag_min_ms: float = 8.0,
    lag_max_ms: float = 60.0,
    harmonic_guard_frac: float = 0.15,
) -> float:
    """Dominant autocorrelation lag in the 8-60ms band, guarding against pitch harmonics.

    A room/line echo shows up as a secondary autocorrelation peak at a delay unrelated
    to the pitch period. Voiced speech also produces peaks at integer multiples of the
    pitch period (harmonics) - those are excluded so they aren't mistaken for an echo.
    """
    end_idx = int(end_s * sr)
    lo = max(0, end_idx - int(pre_ms / 1000 * sr))
    seg = caller[lo:end_idx].astype(np.float64)
    if len(seg) < int(0.05 * sr) or np.allclose(seg, seg.mean()):
        return float("nan")
    seg = seg - seg.mean()
    n = len(seg)
    fft = np.fft.rfft(seg, n=2 * n)
    ac = np.fft.irfft(fft * np.conj(fft))[:n]
    if ac[0] <= 1e-12:
        return float("nan")
    ac = ac / ac[0]

    pitch_hz, _ = _autocorr_pitch(seg, sr)
    pitch_lag = sr / pitch_hz if pitch_hz > 0 else None

    lag_min, lag_max = int(lag_min_ms / 1000 * sr), min(len(ac) - 1, int(lag_max_ms / 1000 * sr))
    if lag_max <= lag_min:
        return float("nan")

    best_val, best_lag = -1.0, None
    for lag in range(lag_min, lag_max + 1):
        if pitch_lag:
            k = round(lag / pitch_lag)
            if k >= 1 and abs(lag - k * pitch_lag) < harmonic_guard_frac * pitch_lag:
                continue
        if ac[lag] > best_val:
            best_val, best_lag = float(ac[lag]), lag
    return float(best_lag / sr * 1000) if best_lag is not None else float("nan")


def turns_for_channel(turns: Iterable[dict[str, Any]], channel: int = 0) -> list[dict[str, Any]]:
    """Return the turns of one channel, sorted by start time. Mirrors ``features.py``."""
    selected = [t for t in turns if int(t["channel"]) == channel]
    return sorted(selected, key=lambda t: float(t["start"]))


def extract_acoustic_features(
    caller: np.ndarray, sr: int, turns: Iterable[dict[str, Any]], channel: int = 0
) -> dict[str, float] | None:
    """Build the endpoint-acoustics feature dict for one call.

    ``caller`` must be the mono waveform for ``channel`` (float, any amplitude scale -
    only relative levels matter). Returns ``None`` when there aren't enough usable
    turns to measure (mirrors the guard in ``features.extract_features``).
    """
    ch0 = turns_for_channel(turns, channel)
    if not ch0:
        return None

    floor_db = _call_floor_db(caller, sr, ch0)
    call_len_s = len(caller) / sr

    abruptness_vals: list[float] = []
    tail_vals: list[float] = []
    voiced_vals: list[float] = []
    echo_vals: list[float] = []

    for t in ch0:
        if (t["end"] - t["start"]) < _MIN_TURN_DUR_S:
            continue
        if (call_len_s - t["end"]) * 1000 < _END_GUARD_MS:
            continue  # too close to file end, no real tail to measure

        decay = _analyze_decay(caller, sr, t["end"], floor_db)
        if decay is None:
            continue
        abruptness_vals.append(decay["abruptness_db"])
        tail_vals.append(decay["tail_ms"])
        voiced_vals.append(_voiced_fraction_near_end(caller, sr, t["end"]))
        echo = _echo_delay_ms(caller, sr, t["end"])
        if not np.isnan(echo):
            echo_vals.append(echo)

    if not abruptness_vals:
        return None

    return {
        "abruptness_db_mean": float(np.mean(abruptness_vals)),
        "tail_ms_mean": float(np.mean(tail_vals)),
        "floor_db": floor_db,
        "voiced_frac_near_end_mean": float(np.mean(voiced_vals)),
        "echo_delay_ms_mean": float(np.mean(echo_vals)) if echo_vals else float("nan"),
        "voiced_frac_near_end_range": float(np.max(voiced_vals) - np.min(voiced_vals)) if len(voiced_vals) >= 2 else 0.0,
    }


def acoustic_feature_vector(features: dict[str, float]) -> np.ndarray:
    """Return the feature dict as an ordered 1-D array. Mirrors ``features.feature_vector``."""
    return np.array([features[c] for c in ACOUSTIC_FEATURE_COLS], dtype=float)


# ============================================================================
# NST v2 (EXPERIMENTAL) - distributional aggregation over turns, instead of
# collapsing straight to a per-call mean. Not wired into production /detect;
# see scripts/nst_v2_experiment.py. Reuses the exact same per-turn primitives
# as v1 above (_analyze_decay, _voiced_fraction_near_end, _echo_delay_ms,
# _call_floor_db) - this is not a parallel implementation.
# ============================================================================

NST_V2_TURN_LEVEL_FEATURES: list[str] = [
    "abruptness_db", "tail_ms", "voiced_frac_near_end", "echo_delay_ms",
]
NST_V2_AGGS: list[str] = ["mean", "median", "std", "range", "iqr"]
NST_V2_ALL_STATS_COLS: list[str] = [
    f"{feat}_{agg}" for feat in NST_V2_TURN_LEVEL_FEATURES for agg in NST_V2_AGGS
]

#: Historical 5-feature "v1 baseline" used throughout the ablation record. Kept as its
#: own constant (not derived from ACOUSTIC_FEATURE_COLS, which is now the frozen
#: 6-feature production set) so these configs stay reproducible and don't silently pick
#: up duplicate columns now that voiced_frac_near_end_range has moved into production.
_NST_V1_ORIGINAL_5: list[str] = [
    "abruptness_db_mean", "tail_ms_mean", "floor_db",
    "voiced_frac_near_end_mean", "echo_delay_ms_mean",
]

#: Config A - NST v1 baseline (unchanged).
NST_V2_CONFIG_A: list[str] = list(_NST_V1_ORIGINAL_5)
#: Config B - v1 + voiced_frac_near_end distributional stats.
NST_V2_CONFIG_B: list[str] = list(_NST_V1_ORIGINAL_5) + [
    "voiced_frac_near_end_median", "voiced_frac_near_end_std",
    "voiced_frac_near_end_range", "voiced_frac_near_end_iqr",
]
#: Config C - v1 + distributional stats for all 4 turn-level features (dedup with
#: the *_mean columns already in _NST_V1_ORIGINAL_5, order preserved).
NST_V2_CONFIG_C: list[str] = list(dict.fromkeys(list(_NST_V1_ORIGINAL_5) + NST_V2_ALL_STATS_COLS))
#: Config D - Config C + hard_silence_flag (floor_db itself is kept, not replaced).
NST_V2_CONFIG_D: list[str] = NST_V2_CONFIG_C + ["hard_silence_flag"]


def _agg_stats(vals: list[float], prefix: str) -> dict[str, float]:
    arr = np.array([v for v in vals if v is not None], dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {f"{prefix}_{agg}": float("nan") for agg in NST_V2_AGGS}
    if len(arr) == 1:
        return {
            f"{prefix}_mean": float(arr[0]), f"{prefix}_median": float(arr[0]),
            f"{prefix}_std": 0.0, f"{prefix}_range": 0.0, f"{prefix}_iqr": 0.0,
        }
    q75, q25 = np.percentile(arr, [75, 25])
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_range": float(np.max(arr) - np.min(arr)),
        f"{prefix}_iqr": float(q75 - q25),
    }


def extract_acoustic_features_v2(
    caller: np.ndarray,
    sr: int,
    turns: Iterable[dict[str, Any]],
    channel: int = 0,
    hard_silence_floor_db: float | None = None,
    hard_silence_epsilon_db: float = 0.05,
) -> dict[str, float] | None:
    """NST v2 (experimental): per-turn distribution (mean/median/std/range/iqr)
    instead of only the per-call mean. Returns ``None`` under the same conditions
    as ``extract_acoustic_features`` (no usable turns).

    ``hard_silence_floor_db``: reference "hard digital silence" floor_db value used
    to compute ``hard_silence_flag`` - must be derived from the *train-split-only*,
    *unsupervised* marginal distribution of floor_db (e.g. its mode), never from
    validation labels. If None, ``hard_silence_flag`` is omitted from the output.
    """
    ch0 = turns_for_channel(turns, channel)
    if not ch0:
        return None

    floor_db = _call_floor_db(caller, sr, ch0)
    call_len_s = len(caller) / sr

    per_turn: dict[str, list[float]] = {feat: [] for feat in NST_V2_TURN_LEVEL_FEATURES}
    for t in ch0:
        if (t["end"] - t["start"]) < _MIN_TURN_DUR_S:
            continue
        if (call_len_s - t["end"]) * 1000 < _END_GUARD_MS:
            continue
        decay = _analyze_decay(caller, sr, t["end"], floor_db)
        if decay is None:
            continue
        per_turn["abruptness_db"].append(decay["abruptness_db"])
        per_turn["tail_ms"].append(decay["tail_ms"])
        per_turn["voiced_frac_near_end"].append(_voiced_fraction_near_end(caller, sr, t["end"]))
        echo = _echo_delay_ms(caller, sr, t["end"])
        if not np.isnan(echo):
            per_turn["echo_delay_ms"].append(echo)

    if not per_turn["abruptness_db"]:
        return None

    out: dict[str, float] = {"floor_db": floor_db}
    for feat, vals in per_turn.items():
        out.update(_agg_stats(vals, feat))

    if hard_silence_floor_db is not None:
        out["hard_silence_flag"] = float(abs(floor_db - hard_silence_floor_db) < hard_silence_epsilon_db)

    return out
