"""Legacy detect router — intentionally empty.

All detection endpoints (``/detect/timeDiff``, ``/detect/NST``,
``/detect/STTLexicalAnalysis`` and their health checks) live in ``compare.py``.
This module is kept only so the router wiring in ``app/api/v1/router.py`` stays
unchanged, and because it hosts ``_depad_turns`` below, which both
``compare.py`` and ``scripts/nst_v2_experiment.py`` import.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


def _depad_turns(turns: list[dict], pad_s: float) -> list[dict]:
    """Undo VAD's speech_pad_ms before endpoint-acoustics analysis.

    Live VAD (extract_turns_vad) pads every turn outward by config.VAD_SPEECH_PAD_MS
    so downstream STT doesn't clip soft onsets/offsets - harmless for distribution_time's
    gap statistics, but fatal for natural_speech_termination, which is anchored to the
    exact acoustic offset: by the padded "end", the real energy drop already happened
    before the measurement window starts. The training data (hackmty26/turns/*.json)
    has no such pad, so we remove it here rather than retraining on padded boundaries.
    """
    depadded = []
    for t in turns:
        start, end = t["start"] + pad_s, t["end"] - pad_s
        if end <= start:
            continue  # turn was shorter than 2x the pad - nothing meaningful left
        depadded.append({**t, "start": start, "end": end})
    return depadded
