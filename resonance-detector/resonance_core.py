"""
Shared feature-extraction logic for the resonance-stability detector.

Used by both:
  - resonance_features.py (training/validation, using the ground-truth
    turns/<id>.json segments provided with the dataset)
  - detector.py (real inference, using VAD-detected segments — the
    turns file does not exist for a live /detect call)

Features, computed over voiced frames of channel 0 only:
  - F1/F2 coefficient of variation (spread relative to mean)
  - F1/F2 frame-to-frame jitter (mean abs delta / mean)
  - F0-F1 correlation (do pitch and resonance move together, like in a
    real vocal tract, or independently, like in many TTS pipelines?)
  - long-timescale F1 drift (start of call vs. end of call)
  - vocoder step-periodicity: energy concentrated at a neural vocoder's
    typical frame-hop rate, in the derivative of the F1 trajectory
"""
import numpy as np
import parselmouth

MIN_SEGMENT_SECONDS = 0.3
MAX_FORMANT_HZ = 4000  # Nyquist at 8kHz is already 4kHz; telephone band only holds F1-F3.
MIN_VOICED_FRAMES = 20  # below this, features are too noisy to trust


def series_from_segments(ch0: np.ndarray, sr: int, segments):
    """segments: list of (start_s, end_s) tuples on ch0.
    Returns concatenated (f0s, f1s, f2s) over voiced frames, plus a list
    of per-segment raw (uniformly-spaced) F1 trajectories with their true
    dt, for periodicity analysis."""
    f0s, f1s, f2s = [], [], []
    raw_seg_f1s = []  # list of (f1_array, dt) — unfiltered, uniform spacing

    for start_s, end_s in segments:
        s, e = int(start_s * sr), int(end_s * sr)
        seg = ch0[s:e]
        if len(seg) < sr * MIN_SEGMENT_SECONDS:
            continue
        snd = parselmouth.Sound(seg, sampling_frequency=sr)
        formant = snd.to_formant_burg(max_number_of_formants=5, maximum_formant=MAX_FORMANT_HZ)
        pitch = snd.to_pitch()
        ts = formant.ts()
        raw_f1 = np.array([formant.get_value_at_time(1, tt) or np.nan for tt in ts])
        if len(ts) > 1:
            raw_seg_f1s.append((raw_f1, float(np.mean(np.diff(ts)))))
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

    return np.array(f0s), np.array(f1s), np.array(f2s), raw_seg_f1s


def vocoder_periodicity_score(raw_seg_f1s):
    """Energy fraction of each segment's F1-derivative spectrum concentrated
    near the typical neural-vocoder hop rate (~50-150Hz, i.e. 6.5-20ms
    hops), vs. total energy — averaged across segments."""
    scores = []
    for raw_f1, dt in raw_seg_f1s:
        f1 = raw_f1[~np.isnan(raw_f1)]
        if len(f1) < 30 or dt <= 0:
            continue
        d = np.diff(f1)
        d = d - d.mean()
        spec = np.abs(np.fft.rfft(d))
        freqs = np.fft.rfftfreq(len(d), d=dt)
        band = (freqs > 50) & (freqs < 150)
        if not band.any() or spec.sum() == 0:
            continue
        scores.append(spec[band].sum() / spec.sum())
    return float(np.mean(scores)) if scores else np.nan


FEATURE_NAMES = [
    "F1_cv", "F2_cv", "F1_jitter", "F2_jitter",
    "F0_F1_corr", "F1_drift_abs", "vocoder_periodicity",
]


def features_from_segments(ch0: np.ndarray, sr: int, segments) -> dict:
    f0s, f1s, f2s, raw_seg_f1s = series_from_segments(ch0, sr, segments)
    if len(f1s) < MIN_VOICED_FRAMES:
        return {"n_voiced_frames": len(f1s)}

    d1 = np.abs(np.diff(f1s))
    d2 = np.abs(np.diff(f2s))
    third = max(len(f1s) // 3, 1)
    drift_f1 = f1s[-third:].mean() - f1s[:third].mean()

    return {
        "n_voiced_frames": len(f1s),
        "F1_mean": f1s.mean(),
        "F2_mean": f2s.mean(),
        "F1_cv": f1s.std() / f1s.mean(),
        "F2_cv": f2s.std() / f2s.mean(),
        "F1_jitter": d1.mean() / f1s.mean(),
        "F2_jitter": d2.mean() / f2s.mean(),
        "F0_F1_corr": np.corrcoef(f0s, f1s)[0, 1],
        "F1_drift_abs": abs(drift_f1),
        "vocoder_periodicity": vocoder_periodicity_score(raw_seg_f1s),
    }
