"""ROUGE-L evaluator for summarization tasks.

Pure-Python implementation using longest common subsequence — no external
dependencies required.
"""

from __future__ import annotations

from typing import Any


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenization after lowercasing."""
    return text.lower().split()


def _lcs_length(x: list[str], y: list[str]) -> int:
    """Compute length of longest common subsequence between two token lists."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    # Space-optimized DP: only keep two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def rouge_l(prediction: str, reference: str) -> dict[str, float]:
    """Compute ROUGE-L precision, recall, and F1 between two strings.

    Returns dict with keys: rouge_l_precision, rouge_l_recall, rouge_l_f1.
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return {"rouge_l_precision": 0.0, "rouge_l_recall": 0.0, "rouge_l_f1": 0.0}

    lcs_len = _lcs_length(pred_tokens, ref_tokens)

    precision = lcs_len / len(pred_tokens) if pred_tokens else 0.0
    recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "rouge_l_precision": precision,
        "rouge_l_recall": recall,
        "rouge_l_f1": f1,
    }


class SummarizationQuality:
    """Score LLM response against ground-truth summary using ROUGE-L.

    Expects processed dict to have a "response" key (the LLM output)
    and original dict to have an "answer" key (the ground truth summary).

    Returns scores: rouge_l_precision, rouge_l_recall, rouge_l_f1,
    plus the standard f1/exact_match/recall/contains from AnswerQuality
    for comparability.
    """

    @property
    def name(self) -> str:
        return "summarization_quality"

    def score(
        self, original: dict[str, Any], processed: dict[str, Any]
    ) -> dict[str, float]:
        answer = str(original.get("answer", ""))
        response = str(processed.get("response", ""))

        if not answer:
            return {
                "rouge_l_precision": 1.0,
                "rouge_l_recall": 1.0,
                "rouge_l_f1": 1.0,
            }
        if not response:
            return {
                "rouge_l_precision": 0.0,
                "rouge_l_recall": 0.0,
                "rouge_l_f1": 0.0,
            }

        return rouge_l(response, answer)
