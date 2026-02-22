"""Cost-of-pass metric (arXiv:2504.13359).

Measures tokens spent per successful task completion.
"""

from __future__ import annotations

from dataclasses import dataclass

from context_bench.results import EvalRow


@dataclass
class CostOfPass:
    """Tokens per successful task.

    For each system, computes: total_output_tokens / num_passing_examples.
    An example passes if its score field >= threshold.
    """

    threshold: float = 0.7
    score_field: str = "score"

    @property
    def name(self) -> str:
        return "cost_of_pass"

    def compute(self, rows: list[EvalRow]) -> dict[str, float]:
        if not rows:
            return {"cost_of_pass": float("inf"), "num_passing": 0}

        total_tokens = sum(r.output_tokens for r in rows)
        passing = [r for r in rows if r.scores.get(self.score_field, 0) >= self.threshold]
        num_passing = len(passing)

        cost = total_tokens / num_passing if num_passing > 0 else float("inf")

        return {
            "cost_of_pass": cost,
            "num_passing": float(num_passing),
        }
