#!/usr/bin/env python3
"""Evaluate /detect end-to-end on the val split: individual calls + aggregate stats.

Sends each val call's real audio through the *actual* pipeline (base64 -> VAD ->
both feature families -> both detectors -> combined score), exactly as a grader
would call it - not a shortcut through precomputed turns.

Run from the ``fastapi/`` directory (needs the full requirements.txt installed,
including faster-whisper, since this exercises the real VAD path):

    python scripts/evaluate.py                 # all 71 val calls
    python scripts/evaluate.py --limit 10       # first 10 only, for a quick look
    python scripts/evaluate.py --ids call_abc123 call_def456   # specific calls
"""

from __future__ import annotations

import argparse
import base64
import csv
import sys
from pathlib import Path

import numpy as np

FASTAPI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FASTAPI_DIR))

from fastapi.testclient import TestClient  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_auc_score  # noqa: E402

from app.core import config  # noqa: E402
from app.main import app  # noqa: E402

FAMILIES = ("distribution_time", "natural_speech_termination")


def load_val_rows(limit: int | None, ids: list[str] | None) -> list[dict]:
    with open(config.HACKMTY26_DIR / "manifest.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "val"]
    if ids:
        wanted = set(ids)
        rows = [r for r in rows if r["anon_id"] in wanted]
    elif limit:
        rows = rows[:limit]
    return rows


def run_one(client: TestClient, anon_id: str) -> dict | None:
    wav_path = config.HACKMTY26_DIR / "audio" / f"{anon_id}.wav"
    payload = base64.b64encode(wav_path.read_bytes()).decode()
    resp = client.post("/detect", json={"audio_base64": payload})
    if resp.status_code != 200:
        print(f"  {anon_id}: ERROR {resp.status_code} {resp.text[:200]}")
        return None
    return resp.json()


def proba_synthetic(is_synthetic: bool, confidence: float) -> float:
    """Recover P(synthetic) from a (verdict, confidence-in-predicted-class) pair."""
    return confidence if is_synthetic else 1.0 - confidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N val calls.")
    parser.add_argument("--ids", nargs="+", default=None, help="Evaluate only these specific anon_ids.")
    args = parser.parse_args()

    rows = load_val_rows(args.limit, args.ids)
    if not rows:
        raise SystemExit("No matching val calls found.")

    results = []
    print(f"=== Individual calls ({len(rows)}) ===")
    with TestClient(app) as client:
        for r in rows:
            aid = r["anon_id"]
            body = run_one(client, aid)
            if body is None:
                continue
            truth_synth = r["label"] == "synthetic"
            correct = body["is_synthetic"] == truth_synth
            per_family = ", ".join(
                f"{fam}={'synth' if v['is_synthetic'] else 'human'}({v['confidence']:.2f})"
                for fam, v in body["breakdown"].items()
            )
            print(
                f"  {aid}  truth={r['label']:9s}  combined={'synth' if body['is_synthetic'] else 'human':5s}"
                f"({body['confidence']:.2f})  {'OK ' if correct else 'MISS'}  [{per_family}]"
            )
            results.append({"anon_id": aid, "truth_synth": truth_synth, **body})

    if not results:
        raise SystemExit("No successful /detect calls to aggregate.")

    print(f"\n=== Overall statistics ({len(results)} calls) ===")
    y_true = np.array([r["truth_synth"] for r in results])

    y_pred = np.array([r["is_synthetic"] for r in results])
    proba = np.array([proba_synthetic(r["is_synthetic"], r["confidence"]) for r in results])
    print(f"  {'combined':28s} acc={(y_true == y_pred).mean():.3f}  auc={roc_auc_score(y_true, proba):.3f}  n={len(results)}")

    for family in FAMILIES:
        sub = [r for r in results if family in r["breakdown"]]
        if not sub:
            print(f"  {family:28s} (no calls had this family available)")
            continue
        yt = np.array([r["truth_synth"] for r in sub])
        fam_pred = np.array([r["breakdown"][family]["is_synthetic"] for r in sub])
        fam_proba = np.array(
            [proba_synthetic(r["breakdown"][family]["is_synthetic"], r["breakdown"][family]["confidence"]) for r in sub]
        )
        print(f"  {family:28s} acc={(yt == fam_pred).mean():.3f}  auc={roc_auc_score(yt, fam_proba):.3f}  n={len(sub)}")

    print("\n  confusion matrix (combined) [[TN, FP], [FN, TP]] (positive = synthetic):")
    print(" ", confusion_matrix(y_true, y_pred).tolist())


if __name__ == "__main__":
    main()
