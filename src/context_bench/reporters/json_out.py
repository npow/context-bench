"""Machine-readable JSON reporter."""

from __future__ import annotations

from context_bench.results import EvalResult


def to_json(result: EvalResult) -> str:
    """Serialize EvalResult to JSON string."""
    return result.to_json()
