#!/usr/bin/env python3
"""Call ``POST /detect/STTLexicalAnalysis`` and print the comparison table.

Examples (service running on :8000):
    python scripts/compare_test.py --anon-id call_76856257e3ef
    python scripts/compare_test.py --audio ../audio/call_76856257e3ef.wav
    python scripts/compare_test.py --split val --limit 5

Output per call:

    Tiempos de Distribucion
    is_synthetic confidence
    ------------ ----------
            True     0.9382

    Lexical Analysis
    is_synthetic confidence
    ------------ ----------
            True     0.9699

    Together
    is_synthetic confidence
    ------------ ----------
            True     0.9541
"""

from __future__ import annotations

import argparse
import base64
import csv
import sys
from pathlib import Path

import requests

FASTAPI_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = FASTAPI_DIR.parent
MANIFEST = REPO_ROOT / "hackmty26" / "manifest.csv"
AUDIO_DIR = REPO_ROOT / "audio"


def load_manifest(split: str | None) -> list[dict]:
    with MANIFEST.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if split:
        rows = [r for r in rows if r["split"] == split]
    return rows


def call_compare(
    session: requests.Session, url: str, wav: Path, channel: int, timeout: float
) -> dict:
    body = {
        "audio_base64": base64.b64encode(wav.read_bytes()).decode("ascii"),
        "channel": channel,
    }
    response = session.post(url, json=body, timeout=timeout)
    response.raise_for_status()
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default="http://127.0.0.1:8000/detect/STTLexicalAnalysis"
    )
    parser.add_argument("--audio", type=Path, default=None, help="Single WAV file.")
    parser.add_argument("--anon-id", default=None, help="Challenge call id in audio/.")
    parser.add_argument("--split", choices=["train", "val"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    if args.audio:
        jobs = [(args.audio.stem, args.audio, None)]
    elif args.anon_id:
        jobs = [(args.anon_id, AUDIO_DIR / f"{args.anon_id}.wav", None)]
    else:
        rows = load_manifest(args.split)
        if args.limit:
            rows = rows[: args.limit]
        jobs = [
            (r["anon_id"], AUDIO_DIR / f"{r['anon_id']}.wav", r["label"]) for r in rows
        ]

    if not jobs:
        print("Nothing to do.")
        return 1

    session = requests.Session()
    agreements: list[bool] = []
    misses = 0
    for i, (anon_id, wav, truth) in enumerate(jobs, 1):
        if not wav.exists():
            print(f"[{i}/{len(jobs)}] {anon_id}: missing {wav}")
            misses += 1
            continue
        try:
            body = call_compare(session, args.url, wav, args.channel, args.timeout)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(jobs)}] {anon_id}: {type(exc).__name__}: {exc}")
            misses += 1
            continue

        header = f"[{i}/{len(jobs)}] {anon_id}"
        if truth:
            header += f"  truth={truth}"
        print(f"{header}\n{body.get('report', '')}")

        if body.get("agreement") is not None:
            agreements.append(bool(body["agreement"]))

    if len(jobs) > 1:
        total = len(jobs)
        print(f"--- {total} call(s) | {misses} error(s) ---")
        if agreements:
            agree = sum(agreements)
            print(
                f"Timing/lexical agreement: {agree}/{len(agreements)} "
                f"({100 * agree / len(agreements):.1f}%)"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
