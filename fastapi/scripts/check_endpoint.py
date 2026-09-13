#!/usr/bin/env python3
"""Hit a running /detect/<family> endpoint over real HTTP and report accuracy stats.

Unlike evaluate.py/accuracy_stats.py (which use FastAPI's in-process TestClient),
this drives an actual ``uvicorn`` server the same way a grader would - real
sockets, real latency. Point it at any single-family route (``/detect/timeDiff``,
``/detect/NST``, ...) or the combined ``/detect``.

Run from the ``fastapi/`` directory, with the server already running elsewhere
(``uvicorn app.main:app --port 8000``):

    python scripts/check_endpoint.py --url http://127.0.0.1:8000/detect/NST \\
        --audio-dir ../hackmty26/audio --n 50
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

FASTAPI_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = FASTAPI_DIR / "reports" / "latest_run.json"


def write_report(
    report_path: Path,
    url: str,
    total: int,
    results: list[dict],
    summary: dict | None,
) -> None:
    """Persist the run so far as JSON for the web frontend (GET /detect/report).

    Called after every row (partial, summary=None) and once more at the end
    (summary filled in), so a page polling this file sees results appear live
    alongside the terminal output above. Writes to a temp file then renames,
    so a concurrent reader never sees a half-written file.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "url": url,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "results": results,
        "summary": summary,
    }
    tmp_path = report_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    os.replace(tmp_path, report_path)


def load_rows(manifest_path: Path, split: str, n: int | None) -> list[dict]:
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if split != "all":
        rows = [r for r in rows if r["split"] == split]
    if n:
        rows = rows[:n]
    return rows


def proba_synthetic(is_synthetic: bool, confidence: float) -> float:
    """Recover P(synthetic) from a (verdict, confidence-in-predicted-class) pair."""
    return confidence if is_synthetic else 1.0 - confidence


def call_endpoint(url: str, audio_base64: str, timeout: float) -> tuple[dict | None, float, str | None]:
    body = json.dumps({"audio_base64": audio_base64}).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
        return payload, time.monotonic() - start, None
    except urllib.error.HTTPError as exc:
        return None, time.monotonic() - start, f"HTTP {exc.code} {exc.read()[:200]!r}"
    except Exception as exc:  # noqa: BLE001
        return None, time.monotonic() - start, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="Full endpoint URL, e.g. http://127.0.0.1:8000/detect/NST")
    parser.add_argument("--audio-dir", required=True, type=Path, help="Directory of <anon_id>.wav files")
    parser.add_argument("--n", type=int, default=None, help="Limit to the first N calls")
    parser.add_argument("--split", choices=["all", "train", "val"], default="val")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=FASTAPI_DIR.parent / "hackmty26" / "manifest.csv",
        help="Path to manifest.csv (default: ../hackmty26/manifest.csv)",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Where to write the JSON report for the web frontend (default: reports/latest_run.json)",
    )
    args = parser.parse_args()

    rows = load_rows(args.manifest, args.split, args.n)
    if not rows:
        raise SystemExit("No matching calls found.")

    y_true: list[bool] = []
    y_pred: list[bool] = []
    proba: list[float] = []
    latencies: list[float] = []
    answered = 0
    errors = 0
    results: list[dict] = []

    for i, r in enumerate(rows, start=1):
        aid = r["anon_id"]
        wav_path = args.audio_dir / f"{aid}.wav"
        if not wav_path.exists():
            print(f"{i}/{len(rows)} {aid:20s} SKIP (no audio file)")
            errors += 1
            results.append({"anon_id": aid, "truth_label": r["label"], "status": "skip", "error": "no audio file"})
            write_report(args.report_path, args.url, len(rows), results, None)
            continue

        audio_b64 = base64.b64encode(wav_path.read_bytes()).decode()
        body, latency, err = call_endpoint(args.url, audio_b64, args.timeout)
        truth_label = r["label"]
        truth_synth = truth_label == "synthetic"

        if err is not None or body is None:
            print(f"{i}/{len(rows)} {aid:20s} truth={truth_label:9s} ERROR {err} {latency:.2f}s")
            errors += 1
            results.append(
                {"anon_id": aid, "truth_label": truth_label, "status": "error", "error": err, "latency_s": latency}
            )
            write_report(args.report_path, args.url, len(rows), results, None)
            continue

        answered += 1
        latencies.append(latency)
        got_synth = bool(body["is_synthetic"])
        confidence = float(body["confidence"])
        ok = got_synth == truth_synth
        got_label = "synthetic" if got_synth else "human"

        y_true.append(truth_synth)
        y_pred.append(got_synth)
        proba.append(proba_synthetic(got_synth, confidence))

        print(
            f"{i}/{len(rows)} {aid:20s} truth={truth_label:9s} got={got_label:9s} "
            f"{'ok' if ok else 'MISS':4s} conf={confidence:.2f} {latency:.2f}s"
        )
        results.append(
            {
                "anon_id": aid,
                "truth_label": truth_label,
                "status": "ok",
                "is_synthetic": got_synth,
                "confidence": confidence,
                "correct": ok,
                "latency_s": latency,
            }
        )
        write_report(args.report_path, args.url, len(rows), results, None)

    if not y_true:
        raise SystemExit("\nNo successful calls to aggregate.")

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    proba_arr = np.array(proba)

    correct = y_true_arr == y_pred_arr
    synth_mask = y_true_arr
    human_mask = ~y_true_arr
    tpr_synthetic = correct[synth_mask].mean() if synth_mask.any() else float("nan")
    tnr_human = correct[human_mask].mean() if human_mask.any() else float("nan")
    balanced_accuracy = (tpr_synthetic + tnr_human) / 2
    auc = roc_auc_score(y_true_arr, proba_arr) if len(set(y_true)) > 1 else float("nan")
    brier = brier_score_loss(y_true_arr, proba_arr)

    print("\nsummary:")
    print(f"  calls: {len(rows)}")
    print(f"  answered: {answered}")
    print(f"  errors: {errors}")
    print(f"  accuracy: {correct.mean():.3f}")
    print(f"  tpr_synthetic: {tpr_synthetic:.3f}")
    print(f"  tnr_human: {tnr_human:.3f}")
    print(f"  mean_latency_s: {np.mean(latencies):.3f}")
    print(f"  max_latency_s: {np.max(latencies):.3f}")
    print(f"  balanced_accuracy: {balanced_accuracy:.3f}")
    print(f"  auc: {auc:.3f}")
    print(f"  brier: {brier:.3f}")

    write_report(
        args.report_path,
        args.url,
        len(rows),
        results,
        summary={
            "calls": len(rows),
            "answered": answered,
            "errors": errors,
            "accuracy": float(correct.mean()),
            "tpr_synthetic": float(tpr_synthetic),
            "tnr_human": float(tnr_human),
            "mean_latency_s": float(np.mean(latencies)),
            "max_latency_s": float(np.max(latencies)),
            "balanced_accuracy": float(balanced_accuracy),
            "auc": float(auc),
            "brier": float(brier),
        },
    )


if __name__ == "__main__":
    main()
