"""Result data structures for context-bench."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalRow:
    """One (system, example) pair with all scores."""

    system: str
    example_id: str | int
    scores: dict[str, float]
    input_tokens: int
    output_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Collection of eval rows with summary statistics."""

    rows: list[EvalRow]
    summary: dict[str, dict[str, float]]  # system -> metric -> value
    timing: dict[str, float] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    def filter(self, **kwargs: Any) -> EvalResult:
        """Filter rows by attribute values.

        Example: result.filter(system="truncate") returns only rows for that system.
        """
        filtered = self.rows
        for key, value in kwargs.items():
            filtered = [r for r in filtered if getattr(r, key, None) == value]

        # Recompute summary for filtered systems
        systems = {r.system for r in filtered}
        summary = {s: v for s, v in self.summary.items() if s in systems}

        return EvalResult(
            rows=filtered,
            summary=summary,
            timing=self.timing,
            config=self.config,
        )

    def to_dataframe(self) -> Any:
        """Convert to pandas DataFrame. Requires pandas."""
        import pandas as pd

        records = []
        for row in self.rows:
            record = {
                "system": row.system,
                "example_id": row.example_id,
                "input_tokens": row.input_tokens,
                "output_tokens": row.output_tokens,
                **row.scores,
                **row.metadata,
            }
            records.append(record)
        return pd.DataFrame(records)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(
            {
                "rows": [
                    {
                        "system": r.system,
                        "example_id": r.example_id,
                        "scores": r.scores,
                        "input_tokens": r.input_tokens,
                        "output_tokens": r.output_tokens,
                        "metadata": r.metadata,
                    }
                    for r in self.rows
                ],
                "summary": self.summary,
                "timing": self.timing,
                "config": self.config,
            },
            indent=2,
        )
