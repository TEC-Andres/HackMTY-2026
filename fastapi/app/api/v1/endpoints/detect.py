"""Legacy detect router — intentionally empty.

All detection endpoints (``/detect/timeDiff``, ``/detect/STTLexicalAnalysis``
and their health checks) live in ``compare.py``. This module is kept only so the
router wiring in ``app/api/v1/router.py`` stays unchanged.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()
