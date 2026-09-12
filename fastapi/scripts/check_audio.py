#!/usr/bin/env python3
"""Classify a single local audio file as human or synthetic, from the terminal.

Runs the exact same /detect pipeline used by the live service (base64 -> VAD ->
distribution_time + natural_speech_termination -> combined), in-process - no
server needs to be running.

Usage:
    python scripts/check_audio.py path/to/call.wav
    python scripts/check_audio.py path/to/call.wav --channel 0
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

FASTAPI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FASTAPI_DIR))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


def _line(label: str, is_synthetic: bool, confidence: float) -> str:
    verdict = "SYNTHETIC" if is_synthetic else "HUMAN"
    return f"  {label:28s} {verdict:10s} confidence={confidence:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio_path", type=Path, help="Path to a local WAV file (stereo, channel 0 = caller).")
    parser.add_argument("--channel", type=int, default=0, help="Caller channel (default: 0).")
    args = parser.parse_args()

    if not args.audio_path.exists():
        raise SystemExit(f"File not found: {args.audio_path}")

    payload = base64.b64encode(args.audio_path.read_bytes()).decode()

    with TestClient(app) as client:
        resp = client.post("/detect", json={"audio_base64": payload, "channel": args.channel})

    if resp.status_code != 200:
        raise SystemExit(f"Error {resp.status_code}: {resp.text}")

    body = resp.json()

    print(f"\nFile: {args.audio_path.name}\n")
    print(_line("Overall (combined):", body["is_synthetic"], body["confidence"]))
    for family, result in body["breakdown"].items():
        print(_line(f"  - {family}:", result["is_synthetic"], result["confidence"]))
    if not body["breakdown"]:
        print("  (no family produced a result - check /detect/health)")
    print()


if __name__ == "__main__":
    main()
