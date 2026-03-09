"""Per-QA-type breakdown metric.

Slices scores by the ``qa_type`` key in each EvalRow's metadata dict and
computes mean score per type.  Intended for memory-system evaluations where
qa_type is one of: single_hop, multi_hop, temporal, open_domain, adversarial.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from context_bench.results import EvalRow


@dataclass
class PerQATypeMetric:
    """Mean score broken down by qa_type from row metadata.

    Emits ``{score_field}_{qa_type}`` keys plus an overall ``{score_field}_mean``.
    """

    score_field: str = "f1"

    @property
    def name(self) -> str:
        return "per_qa_type"

    def compute(self, rows: list[EvalRow]) -> dict[str, float]:
        if not rows:
            return {}

        buckets: dict[str, list[float]] = defaultdict(list)
        all_values: list[float] = []

        for r in rows:
            qa_type = r.metadata.get("qa_type", "unknown")
            value = r.scores.get(self.score_field, 0.0)
            buckets[qa_type].append(value)
            all_values.append(value)

        result: dict[str, float] = {}
        for qa_type, values in sorted(buckets.items()):
            key = f"{self.score_field}_{qa_type}"
            result[key] = sum(values) / len(values)

        result[f"{self.score_field}_mean"] = sum(all_values) / len(all_values)

        return result
