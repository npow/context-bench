"""End-to-end answer quality evaluator.

Scores LLM responses against ground-truth answers using F1, exact match,
and recall. Works with any System that adds a "response" key to the example.
"""

from __future__ import annotations

from typing import Any

from context_bench.metrics.quality import exact_match, f1_score, recall_score


class AnswerQuality:
    """Score LLM response against ground-truth answer.

    Expects processed dict to have a "response" key (the LLM output)
    and original dict to have an "answer" key (the ground truth).

    Returns scores: f1, exact_match, recall, contains (binary substring check).
    """

    @property
    def name(self) -> str:
        return "answer_quality"

    def score(
        self, original: dict[str, Any], processed: dict[str, Any]
    ) -> dict[str, float]:
        answer = str(original.get("answer", ""))
        response = str(processed.get("response", ""))

        if not answer:
            return {"f1": 1.0, "exact_match": 1.0, "recall": 1.0, "contains": 1.0}
        if not response:
            return {"f1": 0.0, "exact_match": 0.0, "recall": 0.0, "contains": 0.0}

        contains = 1.0 if answer.lower() in response.lower() else 0.0

        return {
            "f1": f1_score(response, answer),
            "exact_match": exact_match(response, answer),
            "recall": recall_score(response, answer),
            "contains": contains,
        }
