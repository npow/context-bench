"""Tests for built-in metrics."""

from __future__ import annotations

from context_bench.metrics.cost_of_pass import CostOfPass
from context_bench.metrics.quality import (
    MeanScore,
    PassRate,
    exact_match,
    f1_score,
    recall_score,
)
from context_bench.metrics.token_stats import CompressionRatio, ParetoRank
from context_bench.results import EvalRow


def _make_rows() -> list[EvalRow]:
    return [
        EvalRow("sys_a", 0, {"score": 0.9}, input_tokens=100, output_tokens=50),
        EvalRow("sys_a", 1, {"score": 0.3}, input_tokens=100, output_tokens=60),
        EvalRow("sys_a", 2, {"score": 0.8}, input_tokens=100, output_tokens=40),
    ]


def test_pass_rate():
    metric = PassRate(threshold=0.7)
    result = metric.compute(_make_rows())
    # 2 out of 3 pass (0.9 and 0.8)
    assert abs(result["pass_rate"] - 2 / 3) < 1e-9


def test_mean_score():
    metric = MeanScore()
    result = metric.compute(_make_rows())
    expected = (0.9 + 0.3 + 0.8) / 3
    assert abs(result["mean_score"] - expected) < 1e-9


def test_compression_ratio():
    metric = CompressionRatio()
    result = metric.compute(_make_rows())
    total_in = 300
    total_out = 150
    expected = 1.0 - total_out / total_in
    assert abs(result["compression_ratio"] - expected) < 1e-9


def test_cost_of_pass():
    metric = CostOfPass(threshold=0.7)
    result = metric.compute(_make_rows())
    # Total output tokens = 150, passing examples = 2
    assert abs(result["cost_of_pass"] - 75.0) < 1e-9
    assert result["num_passing"] == 2.0


def test_cost_of_pass_no_passing():
    rows = [EvalRow("sys_a", 0, {"score": 0.1}, input_tokens=100, output_tokens=50)]
    metric = CostOfPass(threshold=0.7)
    result = metric.compute(rows)
    assert result["cost_of_pass"] == float("inf")


def test_f1_score():
    assert f1_score("the cat sat", "the cat sat") == 1.0
    assert f1_score("", "hello") == 0.0
    # Partial overlap
    score = f1_score("the cat sat on mat", "the cat sat")
    assert 0.5 < score < 1.0


def test_exact_match():
    assert exact_match("Hello World", "hello world") == 1.0
    assert exact_match("Hello", "World") == 0.0


def test_recall_score():
    assert recall_score("the cat sat on mat", "the cat sat") == 1.0
    assert recall_score("the cat", "the cat sat on mat") == 0.4  # 2/5 unique words


def test_pareto_rank():
    summary = {
        "system_a": {"mean_score": 0.9, "cost_of_pass": 100},
        "system_b": {"mean_score": 0.8, "cost_of_pass": 50},
        "system_c": {"mean_score": 0.7, "cost_of_pass": 200},
    }
    ranks = ParetoRank.rank_systems(summary)
    # system_c is dominated by both a (better quality, lower cost) and b
    assert ranks["system_c"] > ranks["system_a"]
    assert ranks["system_c"] > ranks["system_b"]
