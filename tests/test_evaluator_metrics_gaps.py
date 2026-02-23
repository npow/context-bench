"""Tests for gap-closing evaluators, metrics, and runner concurrency."""

from __future__ import annotations

import time
from unittest import mock

import pytest

from context_bench.results import EvalRow


# ===========================================================================
# MathEquivalence evaluator
# ===========================================================================


class TestMathEquivalence:
    def test_exact_match(self):
        from context_bench.evaluators.math_equivalence import MathEquivalence
        ev = MathEquivalence()
        scores = ev.score({"answer": "42"}, {"response": "42"})
        assert scores["math_equiv"] == 1.0

    def test_latex_frac(self):
        from context_bench.evaluators.math_equivalence import MathEquivalence
        ev = MathEquivalence()
        scores = ev.score({"answer": r"\frac{1}{2}"}, {"response": "0.5"})
        assert scores["math_equiv"] == 1.0

    def test_latex_frac_reverse(self):
        from context_bench.evaluators.math_equivalence import MathEquivalence
        ev = MathEquivalence()
        scores = ev.score({"answer": "0.5"}, {"response": r"\frac{1}{2}"})
        assert scores["math_equiv"] == 1.0

    def test_boxed_extraction_from_response(self):
        from context_bench.evaluators.math_equivalence import MathEquivalence
        ev = MathEquivalence()
        scores = ev.score(
            {"answer": "7"},
            {"response": r"The answer is $\boxed{7}$."},
        )
        assert scores["math_equiv"] == 1.0

    def test_different_values(self):
        from context_bench.evaluators.math_equivalence import MathEquivalence
        ev = MathEquivalence()
        scores = ev.score({"answer": "42"}, {"response": "43"})
        assert scores["math_equiv"] == 0.0

    def test_empty_response(self):
        from context_bench.evaluators.math_equivalence import MathEquivalence
        ev = MathEquivalence()
        scores = ev.score({"answer": "42"}, {"response": ""})
        assert scores["math_equiv"] == 0.0

    def test_empty_reference(self):
        from context_bench.evaluators.math_equivalence import MathEquivalence
        ev = MathEquivalence()
        scores = ev.score({"answer": ""}, {"response": "42"})
        assert scores["math_equiv"] == 1.0

    def test_whitespace_normalization(self):
        from context_bench.evaluators.math_equivalence import MathEquivalence
        ev = MathEquivalence()
        scores = ev.score({"answer": "  42  "}, {"response": "42"})
        assert scores["math_equiv"] == 1.0

    def test_percentage(self):
        from context_bench.evaluators.math_equivalence import MathEquivalence
        ev = MathEquivalence()
        scores = ev.score({"answer": "0.25"}, {"response": "25%"})
        assert scores["math_equiv"] == 1.0

    def test_latex_cdot(self):
        from context_bench.evaluators.math_equivalence import MathEquivalence
        ev = MathEquivalence()
        scores = ev.score({"answer": r"2\cdot 3"}, {"response": "2*3"})
        assert scores["math_equiv"] == 1.0

    def test_comma_in_number(self):
        from context_bench.evaluators.math_equivalence import MathEquivalence
        ev = MathEquivalence()
        scores = ev.score({"answer": "1000"}, {"response": "1,000"})
        assert scores["math_equiv"] == 1.0

    def test_nested_frac(self):
        from context_bench.evaluators.math_equivalence import MathEquivalence
        ev = MathEquivalence()
        # \frac{1}{3} should parse to (1)/(3) = 0.333...
        scores = ev.score(
            {"answer": r"\frac{1}{3}"},
            {"response": r"\frac{2}{6}"},
        )
        assert scores["math_equiv"] == 1.0

    def test_name_property(self):
        from context_bench.evaluators.math_equivalence import MathEquivalence
        assert MathEquivalence().name == "math_equivalence"


class TestMathHelpers:
    def test_normalize_latex_spacing(self):
        from context_bench.evaluators.math_equivalence import _normalize_latex
        assert r"\," not in _normalize_latex(r"1\,000")

    def test_normalize_latex_text(self):
        from context_bench.evaluators.math_equivalence import _normalize_latex
        result = _normalize_latex(r"\text{cm}")
        assert result == "cm"

    def test_try_parse_number_basic(self):
        from context_bench.evaluators.math_equivalence import _try_parse_number
        assert _try_parse_number("42") == 42.0
        assert _try_parse_number("3.14") == pytest.approx(3.14)
        assert _try_parse_number("not a number") is None

    def test_try_parse_number_fraction(self):
        from context_bench.evaluators.math_equivalence import _try_parse_number
        assert _try_parse_number("1/2") == pytest.approx(0.5)
        assert _try_parse_number("(1)/(2)") == pytest.approx(0.5)

    def test_try_parse_number_percentage(self):
        from context_bench.evaluators.math_equivalence import _try_parse_number
        assert _try_parse_number("25%") == pytest.approx(0.25)


# ===========================================================================
# NLILabelMatch evaluator
# ===========================================================================


class TestNLILabelMatch:
    def test_exact_label_match(self):
        from context_bench.evaluators.nli_label_match import NLILabelMatch
        ev = NLILabelMatch()
        scores = ev.score({"answer": "Entailment"}, {"response": "Entailment"})
        assert scores["nli_accuracy"] == 1.0

    def test_case_insensitive(self):
        from context_bench.evaluators.nli_label_match import NLILabelMatch
        ev = NLILabelMatch()
        scores = ev.score({"answer": "SUPPORTS"}, {"response": "supports"})
        assert scores["nli_accuracy"] == 1.0

    def test_synonym_match(self):
        from context_bench.evaluators.nli_label_match import NLILabelMatch
        ev = NLILabelMatch()
        scores = ev.score({"answer": "Entailment"}, {"response": "Yes"})
        assert scores["nli_accuracy"] == 1.0

    def test_extraction_from_sentence(self):
        from context_bench.evaluators.nli_label_match import NLILabelMatch
        ev = NLILabelMatch()
        scores = ev.score(
            {"answer": "Contradiction"},
            {"response": "Based on the evidence, I conclude this is a contradiction."},
        )
        assert scores["nli_accuracy"] == 1.0

    def test_extraction_answer_is_pattern(self):
        from context_bench.evaluators.nli_label_match import NLILabelMatch
        ev = NLILabelMatch()
        scores = ev.score(
            {"answer": "SUPPORTS"},
            {"response": "The answer is: supports"},
        )
        assert scores["nli_accuracy"] == 1.0

    def test_wrong_label(self):
        from context_bench.evaluators.nli_label_match import NLILabelMatch
        ev = NLILabelMatch()
        scores = ev.score({"answer": "Entailment"}, {"response": "Contradiction"})
        assert scores["nli_accuracy"] == 0.0

    def test_empty_response(self):
        from context_bench.evaluators.nli_label_match import NLILabelMatch
        ev = NLILabelMatch()
        scores = ev.score({"answer": "SUPPORTS"}, {"response": ""})
        assert scores["nli_accuracy"] == 0.0

    def test_empty_reference(self):
        from context_bench.evaluators.nli_label_match import NLILabelMatch
        ev = NLILabelMatch()
        scores = ev.score({"answer": ""}, {"response": "something"})
        assert scores["nli_accuracy"] == 1.0

    def test_not_mentioned(self):
        from context_bench.evaluators.nli_label_match import NLILabelMatch
        ev = NLILabelMatch()
        scores = ev.score(
            {"answer": "Not mentioned"},
            {"response": "The answer is: Not mentioned."},
        )
        assert scores["nli_accuracy"] == 1.0

    def test_neutral_maps_to_not_mentioned(self):
        from context_bench.evaluators.nli_label_match import NLILabelMatch
        ev = NLILabelMatch()
        scores = ev.score(
            {"answer": "Not mentioned"},
            {"response": "neutral"},
        )
        assert scores["nli_accuracy"] == 1.0

    def test_name_property(self):
        from context_bench.evaluators.nli_label_match import NLILabelMatch
        assert NLILabelMatch().name == "nli_label_match"


# ===========================================================================
# Latency metric
# ===========================================================================


class TestLatencyMetric:
    def test_basic_latency(self):
        from context_bench.metrics.latency import Latency
        rows = [
            EvalRow("sys", 0, {}, 0, 0, latency=0.1),
            EvalRow("sys", 1, {}, 0, 0, latency=0.2),
            EvalRow("sys", 2, {}, 0, 0, latency=0.3),
        ]
        result = Latency().compute(rows)
        assert result["latency_mean"] == pytest.approx(0.2)
        assert result["latency_median"] == pytest.approx(0.2)

    def test_empty_rows(self):
        from context_bench.metrics.latency import Latency
        result = Latency().compute([])
        assert result["latency_mean"] == 0.0
        assert result["latency_median"] == 0.0

    def test_single_row(self):
        from context_bench.metrics.latency import Latency
        rows = [EvalRow("sys", 0, {}, 0, 0, latency=0.5)]
        result = Latency().compute(rows)
        assert result["latency_mean"] == pytest.approx(0.5)
        assert result["latency_median"] == pytest.approx(0.5)
        assert result["latency_p95"] == pytest.approx(0.5)
        assert result["latency_p99"] == pytest.approx(0.5)

    def test_percentiles(self):
        from context_bench.metrics.latency import Latency
        # 100 rows with increasing latency
        rows = [EvalRow("sys", i, {}, 0, 0, latency=float(i) / 100) for i in range(100)]
        result = Latency().compute(rows)
        assert result["latency_p95"] >= result["latency_median"]
        assert result["latency_p99"] >= result["latency_p95"]

    def test_name(self):
        from context_bench.metrics.latency import Latency
        assert Latency().name == "latency"


# ===========================================================================
# PerDatasetBreakdown metric
# ===========================================================================


class TestPerDatasetBreakdown:
    def test_single_dataset(self):
        from context_bench.metrics.per_dataset import PerDatasetBreakdown
        rows = [
            EvalRow("sys", 0, {"f1": 0.8}, 0, 0, dataset="hotpotqa"),
            EvalRow("sys", 1, {"f1": 0.6}, 0, 0, dataset="hotpotqa"),
        ]
        result = PerDatasetBreakdown(score_field="f1").compute(rows)
        assert result["dataset:hotpotqa"] == pytest.approx(0.7)

    def test_multiple_datasets(self):
        from context_bench.metrics.per_dataset import PerDatasetBreakdown
        rows = [
            EvalRow("sys", 0, {"f1": 1.0}, 0, 0, dataset="hotpotqa"),
            EvalRow("sys", 1, {"f1": 0.5}, 0, 0, dataset="gsm8k"),
        ]
        result = PerDatasetBreakdown(score_field="f1").compute(rows)
        assert result["dataset:hotpotqa"] == pytest.approx(1.0)
        assert result["dataset:gsm8k"] == pytest.approx(0.5)

    def test_empty_rows(self):
        from context_bench.metrics.per_dataset import PerDatasetBreakdown
        result = PerDatasetBreakdown().compute([])
        assert result == {}

    def test_unknown_dataset_tag(self):
        from context_bench.metrics.per_dataset import PerDatasetBreakdown
        rows = [EvalRow("sys", 0, {"f1": 0.9}, 0, 0, dataset="")]
        result = PerDatasetBreakdown(score_field="f1").compute(rows)
        assert "dataset:unknown" in result

    def test_name(self):
        from context_bench.metrics.per_dataset import PerDatasetBreakdown
        assert PerDatasetBreakdown().name == "per_dataset"


# ===========================================================================
# EvalRow new fields
# ===========================================================================


class TestEvalRowNewFields:
    def test_latency_default(self):
        row = EvalRow("sys", 0, {}, 100, 50)
        assert row.latency == 0.0

    def test_dataset_default(self):
        row = EvalRow("sys", 0, {}, 100, 50)
        assert row.dataset == ""

    def test_latency_and_dataset_set(self):
        row = EvalRow("sys", 0, {}, 100, 50, latency=0.5, dataset="hotpotqa")
        assert row.latency == 0.5
        assert row.dataset == "hotpotqa"

    def test_to_json_includes_new_fields(self):
        import json
        from context_bench.results import EvalResult
        row = EvalRow("sys", 0, {"f1": 0.9}, 100, 50, latency=0.5, dataset="test_ds")
        result = EvalResult(rows=[row], summary={})
        parsed = json.loads(result.to_json())
        assert parsed["rows"][0]["latency"] == 0.5
        assert parsed["rows"][0]["dataset"] == "test_ds"


# ===========================================================================
# Runner: per-example latency and dataset tracking
# ===========================================================================


class TestRunnerLatencyTracking:
    def test_rows_have_latency(self):
        from context_bench import evaluate

        class SlowSystem:
            @property
            def name(self):
                return "slow"
            def process(self, example):
                time.sleep(0.01)
                return dict(example)

        class DummyEvaluator:
            @property
            def name(self):
                return "dummy"
            def score(self, original, processed):
                return {"score": 1.0}

        result = evaluate(
            systems=[SlowSystem()],
            dataset=[{"id": 0, "context": "x"}],
            evaluators=[DummyEvaluator()],
            progress=False,
        )
        assert len(result.rows) == 1
        assert result.rows[0].latency > 0

    def test_rows_have_dataset_tag(self):
        from context_bench import evaluate

        class IdentitySystem:
            @property
            def name(self):
                return "id"
            def process(self, example):
                return dict(example)

        class DummyEvaluator:
            @property
            def name(self):
                return "dummy"
            def score(self, original, processed):
                return {"score": 1.0}

        result = evaluate(
            systems=[IdentitySystem()],
            dataset=[{"id": 0, "context": "x", "dataset": "my_dataset"}],
            evaluators=[DummyEvaluator()],
            progress=False,
        )
        assert result.rows[0].dataset == "my_dataset"


# ===========================================================================
# Runner: concurrent execution
# ===========================================================================


class TestRunnerConcurrency:
    def test_concurrent_produces_same_results(self):
        from context_bench import evaluate

        class EchoSystem:
            @property
            def name(self):
                return "echo"
            def process(self, example):
                return {**example, "response": example.get("answer", "")}

        class DummyEvaluator:
            @property
            def name(self):
                return "dummy"
            def score(self, original, processed):
                return {"score": 1.0 if processed.get("response") == original.get("answer") else 0.0}

        dataset = [{"id": i, "context": f"c{i}", "answer": f"a{i}"} for i in range(20)]

        result_seq = evaluate(
            systems=[EchoSystem()],
            dataset=dataset,
            evaluators=[DummyEvaluator()],
            progress=False,
            max_workers=None,
        )
        result_par = evaluate(
            systems=[EchoSystem()],
            dataset=dataset,
            evaluators=[DummyEvaluator()],
            progress=False,
            max_workers=4,
        )

        # Same number of rows
        assert len(result_seq.rows) == len(result_par.rows)
        # Same scores in same order
        for seq_row, par_row in zip(result_seq.rows, result_par.rows):
            assert seq_row.example_id == par_row.example_id
            assert seq_row.scores == par_row.scores

    def test_concurrent_config_recorded(self):
        from context_bench import evaluate

        class IdSys:
            @property
            def name(self):
                return "id"
            def process(self, example):
                return dict(example)

        class DummyEval:
            @property
            def name(self):
                return "d"
            def score(self, o, p):
                return {"s": 1.0}

        result = evaluate(
            systems=[IdSys()],
            dataset=[{"id": 0}],
            evaluators=[DummyEval()],
            progress=False,
            max_workers=4,
        )
        assert result.config["max_workers"] == 4

    def test_sequential_config_default(self):
        from context_bench import evaluate

        class IdSys:
            @property
            def name(self):
                return "id"
            def process(self, example):
                return dict(example)

        class DummyEval:
            @property
            def name(self):
                return "d"
            def score(self, o, p):
                return {"s": 1.0}

        result = evaluate(
            systems=[IdSys()],
            dataset=[{"id": 0}],
            evaluators=[DummyEval()],
            progress=False,
        )
        assert result.config["max_workers"] == 1


# ===========================================================================
# Evaluator and metric exports
# ===========================================================================


class TestNewExports:
    def test_evaluator_exports(self):
        from context_bench.evaluators import (
            MathEquivalence,
            NLILabelMatch,
        )
        assert MathEquivalence is not None
        assert NLILabelMatch is not None

    def test_metric_exports(self):
        from context_bench.metrics import (
            Latency,
            ParetoRank,
            PerDatasetBreakdown,
        )
        assert Latency is not None
        assert ParetoRank is not None
        assert PerDatasetBreakdown is not None
