"""Token efficiency metric.

Computes accuracy-per-token as the primary optimization target for the
autoresearch loop.  The key output ``token_efficiency`` is score per 1K input
tokens, giving a single number to maximize when trading off quality vs. cost.
"""

from __future__ import annotations

from dataclasses import dataclass

from context_bench.results import EvalRow


@dataclass
class TokenEfficiencyMetric:
    """Score per 1K input tokens, plus supporting latency and token averages.

    ``token_efficiency = mean(scores[score_field]) / (mean(input_tokens) / 1000)``

    The autoresearch loop should maximize ``token_efficiency``.
    """

    score_field: str = "f1"

    @property
    def name(self) -> str:
        return "token_efficiency"

    def compute(self, rows: list[EvalRow]) -> dict[str, float]:
        if not rows:
            return {
                "token_efficiency": 0.0,
                "mean_score": 0.0,
                "mean_input_tokens": 0.0,
                "mean_ingest_latency": 0.0,
                "mean_query_latency": 0.0,
            }

        n = len(rows)
        mean_score = sum(r.scores.get(self.score_field, 0.0) for r in rows) / n
        mean_input_tokens = sum(r.input_tokens for r in rows) / n
        mean_ingest_latency = sum(r.metadata.get("ingest_latency", 0.0) for r in rows) / n
        mean_query_latency = sum(r.metadata.get("query_latency", 0.0) for r in rows) / n

        efficiency = mean_score / (mean_input_tokens / 1000) if mean_input_tokens > 0 else 0.0

        # Penalise token bloat softly: F1 * (ref / tokens)^0.1 where ref=100 words.
        # This rewards genuine F1 improvements while only weakly penalising more tokens.
        # Using F1/tokens directly causes the loop to game the metric by driving tokens
        # to near-zero (e.g. k=1 retrieval) even when F1 also falls.
        _ref = 100.0
        if mean_input_tokens > 0 and mean_score > 0:
            adjusted = mean_score * (_ref / mean_input_tokens) ** 0.1
        else:
            adjusted = 0.0

        return {
            "token_efficiency": adjusted,       # primary optimisation target (F1-dominant)
            "token_efficiency_raw": efficiency,  # old ratio, kept for analysis
            "mean_score": mean_score,
            "mean_input_tokens": mean_input_tokens,
            "mean_ingest_latency": mean_ingest_latency,
            "mean_query_latency": mean_query_latency,
        }
