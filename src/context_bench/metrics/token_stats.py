"""Token statistics metrics."""

from __future__ import annotations

from dataclasses import dataclass

from context_bench.results import EvalRow


@dataclass
class CompressionRatio:
    """Compression ratio: 1 - output_tokens/input_tokens."""

    @property
    def name(self) -> str:
        return "compression_ratio"

    def compute(self, rows: list[EvalRow]) -> dict[str, float]:
        if not rows:
            return {
                "compression_ratio": 0.0,
                "mean_input_tokens": 0.0,
                "mean_output_tokens": 0.0,
            }

        total_in = sum(r.input_tokens for r in rows)
        total_out = sum(r.output_tokens for r in rows)

        ratio = 1.0 - (total_out / total_in) if total_in > 0 else 0.0
        mean_in = total_in / len(rows)
        mean_out = total_out / len(rows)

        return {
            "compression_ratio": ratio,
            "mean_input_tokens": mean_in,
            "mean_output_tokens": mean_out,
        }


@dataclass
class ParetoRank:
    """Rank systems on the quality-cost Pareto frontier.

    Lower rank = better (rank 1 = Pareto-optimal).
    Operates across systems, so should be applied to EvalResult.summary directly.
    Per-system compute returns rank 0 (placeholder) since ranking needs all systems.
    """

    quality_field: str = "score"
    cost_field: str = "cost_of_pass"

    @property
    def name(self) -> str:
        return "pareto_rank"

    def compute(self, rows: list[EvalRow]) -> dict[str, float]:
        # Pareto ranking requires cross-system comparison.
        # Per-system compute returns placeholder; use rank_systems() for actual ranking.
        return {"pareto_rank": 0.0}

    @staticmethod
    def rank_systems(
        summary: dict[str, dict[str, float]],
        quality_field: str = "mean_score",
        cost_field: str = "cost_of_pass",
    ) -> dict[str, int]:
        """Rank systems by Pareto dominance. Lower = better."""
        systems = list(summary.keys())
        ranks: dict[str, int] = {}

        for sys_a in systems:
            rank = 1
            q_a = summary[sys_a].get(quality_field, 0)
            c_a = summary[sys_a].get(cost_field, float("inf"))
            for sys_b in systems:
                if sys_a == sys_b:
                    continue
                q_b = summary[sys_b].get(quality_field, 0)
                c_b = summary[sys_b].get(cost_field, float("inf"))
                # sys_b dominates sys_a if better quality AND lower cost
                if q_b >= q_a and c_b <= c_a and (q_b > q_a or c_b < c_a):
                    rank += 1
            ranks[sys_a] = rank

        return ranks
