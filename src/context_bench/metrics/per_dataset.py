"""Per-dataset breakdown metric.

When running mixed-dataset evaluations, this metric slices scores by the
``dataset`` field on each EvalRow and computes mean score per dataset.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from context_bench.results import EvalRow


@dataclass
class PerDatasetBreakdown:
    """Mean score broken down by dataset tag.

    Emits ``dataset:<name>`` keys so they appear alongside aggregate metrics
    in the summary without collisions.
    """

    score_field: str = "f1"

    @property
    def name(self) -> str:
        return "per_dataset"

    def compute(self, rows: list[EvalRow]) -> dict[str, float]:
        if not rows:
            return {}

        buckets: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            ds = r.dataset or "unknown"
            buckets[ds].append(r.scores.get(self.score_field, 0.0))

        result: dict[str, float] = {}
        for ds_name, values in sorted(buckets.items()):
            key = f"dataset:{ds_name}"
            result[key] = sum(values) / len(values)

        return result
