"""Tests for the core evaluation runner."""

from __future__ import annotations

from context_bench import EvalResult, evaluate
from context_bench.metrics import CompressionRatio, MeanScore, PassRate


class TruncateSystem:
    """Test system that truncates context to first 100 chars."""

    @property
    def name(self) -> str:
        return "truncate_100"

    def process(self, example: dict) -> dict:
        result = dict(example)
        result["context"] = example["context"][:100]
        return result


class IdentitySystem:
    """Test system that passes context through unchanged."""

    @property
    def name(self) -> str:
        return "identity"

    def process(self, example: dict) -> dict:
        return dict(example)


class LengthEvaluator:
    """Test evaluator that scores based on context length preservation."""

    @property
    def name(self) -> str:
        return "length"

    def score(self, original: dict, processed: dict) -> dict[str, float]:
        orig_len = len(original.get("context", ""))
        proc_len = len(processed.get("context", ""))
        ratio = proc_len / orig_len if orig_len > 0 else 1.0
        return {"score": ratio}


DATASET = [
    {"id": 0, "context": "A " * 200, "question": "What?", "answer": "A"},
    {"id": 1, "context": "B " * 150, "question": "What?", "answer": "B"},
    {"id": 2, "context": "C " * 100, "question": "What?", "answer": "C"},
]


def test_evaluate_basic():
    """evaluate() returns EvalResult with correct structure."""
    result = evaluate(
        systems=[TruncateSystem(), IdentitySystem()],
        dataset=DATASET,
        evaluators=[LengthEvaluator()],
        progress=False,
    )
    assert isinstance(result, EvalResult)
    # 2 systems x 3 examples = 6 rows
    assert len(result.rows) == 6
    assert all(r.system in ("truncate_100", "identity") for r in result.rows)


def test_evaluate_with_metrics():
    """evaluate() computes summary when metrics are provided."""
    result = evaluate(
        systems=[TruncateSystem(), IdentitySystem()],
        dataset=DATASET,
        evaluators=[LengthEvaluator()],
        metrics=[MeanScore(), PassRate(threshold=0.5), CompressionRatio()],
        progress=False,
    )
    assert "truncate_100" in result.summary
    assert "identity" in result.summary
    # Identity should have mean_score ~1.0
    assert result.summary["identity"]["mean_score"] > 0.99
    # Truncate should compress
    assert result.summary["truncate_100"]["compression_ratio"] > 0


def test_evaluate_max_examples():
    """max_examples limits the number of examples processed."""
    result = evaluate(
        systems=[IdentitySystem()],
        dataset=DATASET,
        evaluators=[LengthEvaluator()],
        max_examples=2,
        progress=False,
    )
    assert len(result.rows) == 2


def test_evaluate_timing():
    """Timing info is recorded per system."""
    result = evaluate(
        systems=[IdentitySystem()],
        dataset=DATASET,
        evaluators=[LengthEvaluator()],
        progress=False,
    )
    assert "identity" in result.timing
    assert result.timing["identity"] >= 0


def test_result_filter():
    """EvalResult.filter() works correctly."""
    result = evaluate(
        systems=[TruncateSystem(), IdentitySystem()],
        dataset=DATASET,
        evaluators=[LengthEvaluator()],
        progress=False,
    )
    filtered = result.filter(system="identity")
    assert len(filtered.rows) == 3
    assert all(r.system == "identity" for r in filtered.rows)


def test_result_to_json():
    """EvalResult.to_json() returns valid JSON."""
    import json

    result = evaluate(
        systems=[IdentitySystem()],
        dataset=DATASET,
        evaluators=[LengthEvaluator()],
        progress=False,
    )
    j = result.to_json()
    parsed = json.loads(j)
    assert "rows" in parsed
    assert "summary" in parsed
