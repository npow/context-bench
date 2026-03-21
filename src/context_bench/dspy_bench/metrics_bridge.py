"""Metrics bridge: wraps context-bench evaluators as DSPy-compatible metric functions.

Provides two flavours:
  - make_dspy_metric  — standard 3-param (example, prediction, trace) -> float|bool
  - make_gepa_metric  — GEPA 5-param (gold, pred, trace, pred_name, pred_trace) -> ScoreWithFeedback

And a dispatch helper:
  - build_metric(optimizer_name, evaluator, task_type) -> metric callable
"""

from __future__ import annotations

from typing import Any

try:
    from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
except Exception:
    # Fallback if GEPA utilities are unavailable (older dspy versions)
    class ScoreWithFeedback:  # type: ignore[no-redef]
        """Minimal stand-in for dspy ScoreWithFeedback."""

        def __init__(self, score: float, feedback: str):
            self.score = score
            self.feedback = feedback

try:
    from context_bench.dspy_bench.registry import REGISTRY
except Exception:
    REGISTRY: dict[str, Any] = {}  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Primary metric key mapping
# ---------------------------------------------------------------------------

PRIMARY_METRIC_KEY: dict[str, str] = {
    "answer_quality": "f1",
    "math_equivalence": "math_equiv",
    "multiple_choice_accuracy": "mc_accuracy",
    "summarization_quality": "rougeL",
    "code_execution": "pass_at_1",
    "ifeval": "ifeval",
    "llm_judge": "judge_score",
}


# ---------------------------------------------------------------------------
# Standard 3-param metric
# ---------------------------------------------------------------------------

def make_dspy_metric(evaluator: Any, task_type: str, primary_key: str | None = None):
    """Wrap a context-bench evaluator as a standard DSPy metric.

    Returns a callable ``(example, prediction, trace=None) -> float | bool``.
    When *trace* is not None the metric returns a bool (score >= 0.5) for
    bootstrap filtering.
    """
    metric_key = primary_key or PRIMARY_METRIC_KEY.get(evaluator.name)
    if metric_key is None:
        raise ValueError(
            f"No PRIMARY_METRIC_KEY registered for evaluator '{evaluator.name}'. "
            f"Add an entry to PRIMARY_METRIC_KEY in metrics_bridge.py."
        )

    def metric_fn(example: Any, prediction: Any, trace: Any = None) -> float | bool:
        original = getattr(example, "_cb_original", None)
        if original is None:
            raise ValueError(
                "dspy.Example missing '_cb_original'. "
                "Use convert_to_dspy_example() to create examples."
            )
        response = prediction.get("answer", "") if hasattr(prediction, "get") else ""
        processed = {"response": response}
        scores = evaluator.score(original, processed)
        if metric_key not in scores:
            raise ValueError(
                f"Evaluator '{evaluator.name}' returned {list(scores.keys())} "
                f"but PRIMARY_METRIC_KEY expects '{metric_key}'."
            )
        score = scores[metric_key]
        if trace is not None:
            return score >= 0.5
        return score

    # Wrap with a counting wrapper that tracks invocations.
    def counting_wrapper(example: Any, prediction: Any, trace: Any = None) -> float | bool:
        counting_wrapper._call_count += 1  # type: ignore[attr-defined]
        return metric_fn(example, prediction, trace)

    counting_wrapper._call_count = 0  # type: ignore[attr-defined]
    return counting_wrapper


# ---------------------------------------------------------------------------
# GEPA 5-param metric
# ---------------------------------------------------------------------------

def make_gepa_metric(evaluator: Any, task_type: str, primary_key: str | None = None):
    """Wrap a context-bench evaluator as a GEPA-compatible metric.

    GEPA calls this with 5 arguments during per-predictor reflection:
        gold       — the gold dspy.Example
        pred       — the prediction dict
        trace      — the DSPy trace (may be None)
        pred_name  — name of the predictor being evaluated (str)
        pred_trace — the per-predictor trace (list or None)

    Returns ``ScoreWithFeedback(score, feedback)`` unless *trace* is not None,
    in which case returns a bool for bootstrap filtering.
    """
    base_metric = make_dspy_metric(evaluator, task_type, primary_key)

    def gepa_metric_fn(
        gold: Any,
        pred: Any,
        trace: Any = None,
        pred_name: str | None = None,
        pred_trace: Any = None,
    ):
        score = base_metric(gold, pred, trace)
        if isinstance(score, bool):
            return score

        original = getattr(gold, "_cb_original", gold.toDict() if hasattr(gold, "toDict") else {})
        response = pred.get("answer", "") if hasattr(pred, "get") else ""
        expected = original.get("answer", "") if isinstance(original, dict) else ""

        if score >= 0.9:
            feedback = "Excellent response. Closely matches the expected answer."
        elif score >= 0.5:
            feedback = (
                f"Partial match. Expected: {expected[:200]}. "
                f"Got: {response[:200]}."
            )
        else:
            feedback = (
                f"Poor match. Expected: {expected[:200]}. "
                f"Got: {response[:200]}."
            )

        if pred_name:
            feedback = f"[{pred_name}] " + feedback

        return ScoreWithFeedback(score=score, feedback=feedback)

    gepa_metric_fn._call_count = 0  # type: ignore[attr-defined]
    return gepa_metric_fn


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------

METRIC_FACTORIES: dict[str, Any] = {
    "standard": make_dspy_metric,
    "gepa_feedback": make_gepa_metric,
}


def build_metric(optimizer_name: str, evaluator: Any, task_type: str):
    """Look up the optimizer in REGISTRY and dispatch to the right metric factory."""
    entry = REGISTRY[optimizer_name]
    factory_fn = METRIC_FACTORIES[entry["metric_factory"]]
    return factory_fn(evaluator, task_type)
