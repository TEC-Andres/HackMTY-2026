"""Plain-text rendering of the detector comparison (issue #21).

Produces the console-friendly table:

    Tiempos de Distribucion
    is_synthetic confidence
    ------------ ----------
            True     0.7284

    Lexical Analysis
    ...
"""

from __future__ import annotations

from typing import Any

_HEADER = "is_synthetic confidence"
_SEPARATOR = "------------ ----------"


def _block(title: str, result: Any | None) -> str:
    if result is None:
        is_synthetic, confidence = "n/a", "n/a"
    else:
        is_synthetic = str(bool(result.is_synthetic))
        confidence = f"{float(result.confidence):.4f}"
    row = f"{is_synthetic:>12} {confidence:>10} "
    return f"{title}\n{_HEADER}\n{_SEPARATOR}\n{row}\n"


def render_comparison(timing: Any | None, lexical: Any, ensemble: Any) -> str:
    """Render the three verdicts as the comparison table."""
    return "\n".join(
        [
            _block("Tiempos de Distribucion", timing),
            _block("Lexical Analysis", lexical),
            _block("Together", ensemble),
        ]
    )
