#!/usr/bin/env python3
"""Confidence + accuracy statistics across the dataset, per feature family.

Sends every call through the real /detect pipeline blind - the label is never
given to the API, only used afterward, locally, to score the result. Reports:
  - average confidence, per family and combined
  - accuracy against the manifest label, per family and combined

Automatically picks up whatever families appear in /detect's `breakdown` field,
so adding a new feature family later needs no changes here.

Run from the ``fastapi/`` directory:

    python scripts/accuracy_stats.py                 # whole dataset (train+val)
    python scripts/accuracy_stats.py --split val      # only the held-out val split
    python scripts/accuracy_stats.py --split train
    python scripts/accuracy_stats.py --limit 20       # quick subset, any split
"""

from __future__ import annotations

import argparse
import base64
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

FASTAPI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FASTAPI_DIR))

from fastapi.testclient import TestClient  # noqa: E402

from app.core import config  # noqa: E402
from app.main import app  # noqa: E402

Entry = tuple[bool, float, str]  # (correct, confidence, split)


def load_rows(split: str, limit: int | None) -> list[dict]:
    with open(config.HACKMTY26_DIR / "manifest.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if split != "all":
        rows = [r for r in rows if r["split"] == split]
    if limit:
        rows = rows[:limit]
    return rows


def report(entries: list[Entry], label: str) -> None:
    if not entries:
        print(f"  {label:28s} (no data)")
        return
    correct = [c for c, _, _ in entries]
    conf = [cf for _, cf, _ in entries]
    print(
        f"  {label:28s} accuracy={np.mean(correct):.3f}  "
        f"avg_confidence={np.mean(conf):.3f}  n={len(entries)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["all", "train", "val"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_rows(args.split, args.limit)
    if not rows:
        raise SystemExit("No matching calls found.")

    stats: dict[str, list[Entry]] = defaultdict(list)
    errors = 0

    with TestClient(app) as client:
        for i, r in enumerate(rows, start=1):
            wav_path = config.HACKMTY26_DIR / "audio" / f"{r['anon_id']}.wav"
            if not wav_path.exists():
                errors += 1
                continue
            payload = base64.b64encode(wav_path.read_bytes()).decode()
            resp = client.post("/detect", json={"audio_base64": payload})
            if resp.status_code != 200:
                errors += 1
                continue
            body = resp.json()
            truth_synth = r["label"] == "synthetic"

            stats["combined"].append((body["is_synthetic"] == truth_synth, body["confidence"], r["split"]))
            for family, result in body["breakdown"].items():
                stats[family].append((result["is_synthetic"] == truth_synth, result["confidence"], r["split"]))

            if i % 25 == 0:
                print(f"  ... {i}/{len(rows)} processed")

    print(f"\nProcessed {len(rows)} calls ({errors} errors/skips)\n")

    keys = ["combined"] + sorted(k for k in stats if k != "combined")

    print(f"=== Overall ({args.split}) ===")
    for key in keys:
        report(stats[key], key)

    if args.split == "all":
        for split_name in ("train", "val"):
            print(f"\n=== split={split_name} ===")
            for key in keys:
                sub = [(c, cf, s) for c, cf, s in stats[key] if s == split_name]
                report(sub, key)


if __name__ == "__main__":
    main()
