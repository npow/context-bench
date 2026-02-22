"""Quality metrics: F1, exact match, pass rate, mean score."""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass

from context_bench.results import EvalRow


@dataclass
class PassRate:
    """Fraction of examples above a score threshold."""

    threshold: float = 0.7
    score_field: str = "score"

    @property
    def name(self) -> str:
        return "pass_rate"

    def compute(self, rows: list[EvalRow]) -> dict[str, float]:
        if not rows:
            return {"pass_rate": 0.0}

        passing = sum(
            1 for r in rows if r.scores.get(self.score_field, 0) >= self.threshold
        )
        return {"pass_rate": passing / len(rows)}


@dataclass
class MeanScore:
    """Average of a score field across examples."""

    score_field: str = "score"

    @property
    def name(self) -> str:
        return "mean_score"

    def compute(self, rows: list[EvalRow]) -> dict[str, float]:
        if not rows:
            return {"mean_score": 0.0}

        values = [r.scores.get(self.score_field, 0.0) for r in rows]
        return {"mean_score": sum(values) / len(values)}


def normalize_answer(text: str) -> str:
    """Lowercase, strip articles/punctuation, collapse whitespace (SQuAD standard)."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def f1_score(prediction: str, reference: str) -> float:
    """Token-level F1 between prediction and reference strings (SQuAD standard)."""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    """Exact match after normalization (SQuAD standard)."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(reference) else 0.0


def recall_score(prediction: str, reference: str) -> float:
    """Token-level recall with Counter multiset intersection."""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    return sum(common.values()) / len(ref_tokens)
