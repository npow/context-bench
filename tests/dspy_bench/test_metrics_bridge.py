"""Tests for the metrics bridge module (TDD — written before implementation)."""

import pytest
from unittest.mock import MagicMock, patch

dspy = pytest.importorskip("dspy")

from context_bench.dspy_bench.metrics_bridge import (
    PRIMARY_METRIC_KEY,
    METRIC_FACTORIES,
    build_metric,
    make_dspy_metric,
    make_gepa_metric,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockEvaluator:
    """Minimal evaluator stub with .name and .score()."""

    def __init__(self, name: str, scores: dict):
        self._name = name
        self._scores = scores

    @property
    def name(self) -> str:
        return self._name

    def score(self, original: dict, processed: dict) -> dict:
        return dict(self._scores)


class FakeExample:
    """Stand-in for dspy.Example with _cb_original and dict-like prediction."""

    def __init__(self, cb_original: dict | None = None):
        if cb_original is not None:
            self._cb_original = cb_original

    def get(self, key, default=None):
        return getattr(self, key, default)


# ---------------------------------------------------------------------------
# make_dspy_metric tests
# ---------------------------------------------------------------------------

class TestMakeDspyMetric:
    """Standard 3-param metric wrapper."""

    def test_returns_primary_score(self):
        evaluator = MockEvaluator("answer_quality", {"f1": 0.8, "exact_match": 0.5})
        metric = make_dspy_metric(evaluator, "qa")

        example = FakeExample(cb_original={"question": "What?", "answer": "42"})
        prediction = {"answer": "test response"}

        result = metric(example, prediction)
        assert result == 0.8

    def test_raises_for_unknown_evaluator_name(self):
        evaluator = MockEvaluator("unknown_evaluator_xyz", {"score": 1.0})
        with pytest.raises(ValueError, match="No PRIMARY_METRIC_KEY registered"):
            make_dspy_metric(evaluator, "qa")

    def test_raises_when_cb_original_missing(self):
        evaluator = MockEvaluator("answer_quality", {"f1": 0.8})
        metric = make_dspy_metric(evaluator, "qa")

        example = FakeExample()  # no _cb_original
        prediction = {"answer": "test"}

        with pytest.raises(ValueError, match="_cb_original"):
            metric(example, prediction)

    def test_trace_not_none_returns_bool_true(self):
        evaluator = MockEvaluator("answer_quality", {"f1": 0.8})
        metric = make_dspy_metric(evaluator, "qa")

        example = FakeExample(cb_original={"question": "Q", "answer": "A"})
        prediction = {"answer": "response"}

        result = metric(example, prediction, trace="some_trace")
        assert result is True
        assert isinstance(result, bool)

    def test_trace_not_none_returns_bool_false(self):
        evaluator = MockEvaluator("answer_quality", {"f1": 0.3})
        metric = make_dspy_metric(evaluator, "qa")

        example = FakeExample(cb_original={"question": "Q", "answer": "A"})
        prediction = {"answer": "response"}

        result = metric(example, prediction, trace="some_trace")
        assert result is False
        assert isinstance(result, bool)

    def test_call_count_increments(self):
        evaluator = MockEvaluator("answer_quality", {"f1": 0.8})
        metric = make_dspy_metric(evaluator, "qa")

        assert metric._call_count == 0

        example = FakeExample(cb_original={"question": "Q", "answer": "A"})
        prediction = {"answer": "response"}

        metric(example, prediction)
        assert metric._call_count == 1

        metric(example, prediction)
        assert metric._call_count == 2

    def test_custom_primary_key_overrides_default(self):
        evaluator = MockEvaluator("answer_quality", {"f1": 0.8, "exact_match": 0.5})
        metric = make_dspy_metric(evaluator, "qa", primary_key="exact_match")

        example = FakeExample(cb_original={"question": "Q", "answer": "A"})
        prediction = {"answer": "response"}

        result = metric(example, prediction)
        assert result == 0.5

    def test_raises_when_metric_key_missing_from_scores(self):
        evaluator = MockEvaluator("answer_quality", {"bleu": 0.9})  # no "f1" key
        metric = make_dspy_metric(evaluator, "qa")

        example = FakeExample(cb_original={"question": "Q", "answer": "A"})
        prediction = {"answer": "response"}

        with pytest.raises(ValueError, match="PRIMARY_METRIC_KEY expects 'f1'"):
            metric(example, prediction)


# ---------------------------------------------------------------------------
# make_gepa_metric tests
# ---------------------------------------------------------------------------

class TestMakeGepaMetric:
    """5-param GEPA metric wrapper."""

    def test_returns_score_and_feedback(self):
        evaluator = MockEvaluator("answer_quality", {"f1": 0.8})
        metric = make_gepa_metric(evaluator, "qa")

        example = FakeExample(cb_original={"question": "Q", "answer": "A"})
        prediction = {"answer": "response"}

        result = metric(example, prediction)
        assert hasattr(result, "score")
        assert hasattr(result, "feedback")
        assert result.score == 0.8
        assert isinstance(result.feedback, str)

    def test_includes_pred_name_in_feedback(self):
        evaluator = MockEvaluator("answer_quality", {"f1": 0.6})
        metric = make_gepa_metric(evaluator, "qa")

        example = FakeExample(cb_original={"question": "Q", "answer": "A"})
        prediction = {"answer": "response"}

        result = metric(example, prediction, pred_name="generate_answer")
        assert "[generate_answer]" in result.feedback

    def test_trace_not_none_returns_bool(self):
        evaluator = MockEvaluator("answer_quality", {"f1": 0.8})
        metric = make_gepa_metric(evaluator, "qa")

        example = FakeExample(cb_original={"question": "Q", "answer": "A"})
        prediction = {"answer": "response"}

        result = metric(example, prediction, trace="some_trace")
        assert isinstance(result, bool)

    def test_excellent_feedback_for_high_score(self):
        evaluator = MockEvaluator("answer_quality", {"f1": 0.95})
        metric = make_gepa_metric(evaluator, "qa")

        example = FakeExample(cb_original={"question": "Q", "answer": "A"})
        prediction = {"answer": "response"}

        result = metric(example, prediction)
        assert "Excellent" in result.feedback

    def test_poor_feedback_for_low_score(self):
        evaluator = MockEvaluator("answer_quality", {"f1": 0.2})
        metric = make_gepa_metric(evaluator, "qa")

        example = FakeExample(cb_original={"question": "Q", "answer": "A"})
        prediction = {"answer": "response"}

        result = metric(example, prediction)
        assert "Poor" in result.feedback

    def test_has_call_count(self):
        evaluator = MockEvaluator("answer_quality", {"f1": 0.5})
        metric = make_gepa_metric(evaluator, "qa")
        assert hasattr(metric, "_call_count")
        assert metric._call_count == 0


# ---------------------------------------------------------------------------
# METRIC_FACTORIES tests
# ---------------------------------------------------------------------------

class TestMetricFactories:
    def test_contains_standard(self):
        assert "standard" in METRIC_FACTORIES
        assert METRIC_FACTORIES["standard"] is make_dspy_metric

    def test_contains_gepa_feedback(self):
        assert "gepa_feedback" in METRIC_FACTORIES
        assert METRIC_FACTORIES["gepa_feedback"] is make_gepa_metric


# ---------------------------------------------------------------------------
# build_metric tests
# ---------------------------------------------------------------------------

class TestBuildMetric:
    """build_metric dispatches to the correct factory based on REGISTRY."""

    @patch("context_bench.dspy_bench.metrics_bridge.REGISTRY", {
        "BootstrapFewShot": {"metric_factory": "standard"},
    })
    def test_dispatches_standard(self):
        evaluator = MockEvaluator("answer_quality", {"f1": 0.7})
        metric = build_metric("BootstrapFewShot", evaluator, "qa")

        example = FakeExample(cb_original={"question": "Q", "answer": "A"})
        prediction = {"answer": "response"}

        result = metric(example, prediction)
        assert result == 0.7

    @patch("context_bench.dspy_bench.metrics_bridge.REGISTRY", {
        "GEPA": {"metric_factory": "gepa_feedback"},
    })
    def test_dispatches_gepa(self):
        evaluator = MockEvaluator("answer_quality", {"f1": 0.7})
        metric = build_metric("GEPA", evaluator, "qa")

        example = FakeExample(cb_original={"question": "Q", "answer": "A"})
        prediction = {"answer": "response"}

        result = metric(example, prediction)
        assert hasattr(result, "score")
        assert result.score == 0.7

    @patch("context_bench.dspy_bench.metrics_bridge.REGISTRY", {})
    def test_raises_for_unknown_optimizer(self):
        evaluator = MockEvaluator("answer_quality", {"f1": 0.7})
        with pytest.raises(KeyError):
            build_metric("NonExistentOptimizer", evaluator, "qa")
