"""Protocol definitions for context-bench.

Users implement these protocols -- never subclass.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from context_bench.results import EvalRow


@runtime_checkable
class System(Protocol):
    """Anything that transforms context."""

    @property
    def name(self) -> str: ...

    def process(self, example: dict[str, Any]) -> dict[str, Any]: ...


@runtime_checkable
class Evaluator(Protocol):
    """Scores a (input, output) pair. Returns dict of metric_name -> float."""

    @property
    def name(self) -> str: ...

    def score(
        self, original: dict[str, Any], processed: dict[str, Any]
    ) -> dict[str, float]: ...


@runtime_checkable
class Metric(Protocol):
    """Aggregates per-example scores into summary stats."""

    @property
    def name(self) -> str: ...

    def compute(self, rows: list[EvalRow]) -> dict[str, float]: ...
