"""
Energy-based voice activity detection.

At inference time, POST /detect only receives the raw stereo WAV — there
is no turns/<id>.json (that file only exists for the provided train/val
data). This gives a rough substitute: find where channel 0 is actually
speaking, so we can feed the same kind of speech segments into the
formant/pitch extraction that resonance_core.py expects.
"""
import numpy as np


def detect_speech_segments(signal, sr, frame_ms=30, hop_ms=10,
                            min_duration=0.3, merge_gap=0.2, energy_margin_db=12):
    """Return a list of (start_s, end_s) tuples where `signal` has speech."""
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)

    energies, times = [], []
    for start in range(0, max(len(signal) - frame_len, 1), hop_len):
        frame = signal[start:start + frame_len]
        rms = np.sqrt(np.mean(frame.astype(np.float64) ** 2) + 1e-12)
        energies.append(20 * np.log10(rms + 1e-12))
        times.append(start / sr)
    energies = np.array(energies)
    times = np.array(times)
    if len(energies) == 0:
        return []

    noise_floor = np.percentile(energies, 10)
    is_speech = energies > (noise_floor + energy_margin_db)

    segments = []
    in_seg = False
    seg_start = 0.0
    for i, sp in enumerate(is_speech):
        if sp and not in_seg:
            in_seg, seg_start = True, times[i]
        elif not sp and in_seg:
            in_seg = False
            segments.append((seg_start, times[i]))
    if in_seg:
        segments.append((seg_start, times[-1] + hop_ms / 1000))

    merged = []
    for s, e in segments:
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    return [(s, e) for s, e in merged if e - s >= min_duration]
