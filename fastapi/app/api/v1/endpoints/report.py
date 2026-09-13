"""``GET /detect/report`` — latest batch-evaluation results, for the web frontend.

Read-only mirror of whatever ``scripts/check_endpoint.py`` last wrote to
``reports/latest_run.json``. Does not call /detect and does not affect it in
any way; it only lets a browser poll the same numbers that script prints to
the terminal while it runs.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

REPORT_PATH = Path(__file__).resolve().parents[4] / "reports" / "latest_run.json"

router = APIRouter()


@router.get("/detect/report", summary="Latest scripts/check_endpoint.py run, if any")
def latest_report() -> dict:
    if not REPORT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No evaluation report yet. Run scripts/check_endpoint.py first.",
        )
    try:
        return json.loads(REPORT_PATH.read_text())
    except json.JSONDecodeError:
        # Being polled mid-write is possible in theory even with the
        # tmp-file-then-rename swap (e.g. a reader mid-read across the
        # rename); ask the client to retry rather than 500.
        raise HTTPException(status_code=503, detail="Report is being written, try again shortly.")
