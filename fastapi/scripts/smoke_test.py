#!/usr/bin/env python3
"""Smoke test for the detection service.

Exercises:
  1. ``POST /detect/timeDiff`` with real (val-split) turns -> compares to label.
  2. ``GET /detect/timeDiff/health``.
  3. The colab-framework VAD turn extractor on a local WAV (no model needed).

Run from the ``fastapi/`` directory:  python scripts/smoke_test.py
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

FASTAPI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FASTAPI_DIR))

from fastapi.testclient import TestClient  # noqa: E402

from app.core import config  # noqa: E402
from app.main import app  # noqa: E402
from app.services.turns import caller_turns_from_wav  # noqa: E402


def test_detect_with_turns(client: TestClient) -> None:
    print("\n=== 1. /detect/timeDiff with precomputed turns ===")
    manifest = config.HACKMTY26_DIR / "manifest.csv"
    lines = manifest.read_text(encoding="utf-8").splitlines()[1:]
    val_rows = [ln.split(",") for ln in lines if ln.split(",")[2] == "val"]

    ok = 0
    for anon_id, label, _split, _dur in val_rows[:10]:
        turns = json.loads(
            (config.HACKMTY26_DIR / "turns" / f"{anon_id}.json").read_text(
                encoding="utf-8"
            )
        )["turns"]
        response = client.post("/detect/timeDiff", json={"turns": turns})
        response.raise_for_status()
        body = response.json()
        correct = body["is_synthetic"] == (label == "synthetic")
        ok += correct
        print(
            f"  {anon_id}  truth={label:9s} pred_synthetic={body['is_synthetic']!s:5s} "
            f"conf={body['confidence']:.3f}  {'OK' if correct else 'MISS'}"
        )
    print(f"  -> {ok}/{min(len(val_rows), 10)} correct on first val calls")


def test_health(client: TestClient) -> None:
    print("\n=== 2. GET /detect/timeDiff/health ===")
    body = client.get("/detect/timeDiff/health").json()
    print(f"  {body}")


def test_vad_on_wav() -> None:
    print("\n=== 3. Framework VAD turn extraction on a local WAV ===")
    wav = config.COLAB_DIR / "_localPipeline" / "warmup.wav"
    if not wav.exists():
        print(f"  skipped (missing {wav})")
        return
    payload = base64.b64encode(wav.read_bytes()).decode()
    turns = caller_turns_from_wav(payload, channel=0)
    print(f"  {wav.name} -> {len(turns)} caller turns: {turns[:5]}")


def main() -> None:
    with TestClient(app) as client:
        test_health(client)
        test_detect_with_turns(client)
        test_vad_on_wav()


if __name__ == "__main__":
    main()
