"""Scoring function for the autoresearch loop.

Evaluates a ContextPipeline (or any MemorySystem implementation) on a sample
of conversation examples and returns a scalar optimization target.

The primary metric is ``token_efficiency = f1 / (mean_input_tokens / 1000)``.
Higher token efficiency means the pipeline answers correctly while consuming
fewer input tokens — exactly the trade-off the autoresearch loop is optimising.
"""

from __future__ import annotations

from typing import Any, Type

from context_bench.evaluators.answer_quality import AnswerQuality
from context_bench.memory_runner import evaluate_memory
from context_bench.metrics.token_efficiency import TokenEfficiencyMetric


def score_pipeline(
    pipeline_class: Type[Any],
    examples: list[dict[str, Any]],
    relay_url: str,
    model: str = "claude-haiku-4-5-20251001",
    api_key: str = "",
) -> dict[str, float]:
    """Evaluate a pipeline on a sample of examples.

    Instantiates ``pipeline_class``, runs evaluate_memory() over ``examples``
    with AnswerQuality as the evaluator and TokenEfficiencyMetric for
    aggregation, then returns a summary dict.

    Args:
        pipeline_class: A class implementing the MemorySystem protocol.
            Its constructor must accept ``relay_url``, ``model``, and
            ``api_key`` keyword arguments.
        examples:       List of conversation examples in evaluate_memory()
                        format (keys: id, turns, qa_pairs, dataset).
        relay_url:      Base URL of the OpenAI-compatible relay.
        model:          Chat model name forwarded to pipeline_class.
        api_key:        Bearer token for the relay.

    Returns:
        A dict with the following keys:

        ``score``
            Primary optimisation target — equal to ``token_efficiency``.
            Maximise this value.
        ``f1``
            Mean token-level F1 across all QA pairs.
        ``token_efficiency``
            F1 per 1 000 input words (proxy for tokens).
        ``mean_input_tokens``
            Mean word count of ingested conversation histories.
        ``n``
            Number of QA pairs evaluated.
    """
    if not examples:
        return {"score": 0.0, "f1": 0.0, "token_efficiency": 0.0, "mean_input_tokens": 0.0, "n": 0}

    # Build the pipeline instance using keyword-only relay args.
    pipeline = pipeline_class(
        relay_url=relay_url,
        model=model,
        api_key=api_key if api_key else None,
    )

    evaluator = AnswerQuality()
    metric = TokenEfficiencyMetric(score_field="f1")

    result = evaluate_memory(
        systems=[pipeline],
        dataset=examples,
        evaluators=[evaluator],
        metrics=[metric],
        progress=False,
    )

    rows = result.rows
    n = len(rows)
    if n == 0:
        return {"score": 0.0, "f1": 0.0, "token_efficiency": 0.0, "mean_input_tokens": 0.0, "n": 0}

    # Pull aggregated stats from the summary computed by TokenEfficiencyMetric
    system_summary = result.summary.get(pipeline.name, {})
    token_efficiency = system_summary.get("token_efficiency", 0.0)
    mean_score = system_summary.get("mean_score", 0.0)
    mean_input_tokens = system_summary.get("mean_input_tokens", 0.0)

    return {
        "score": mean_score,                # primary target: pure F1, no token penalty
        "f1": mean_score,
        "token_efficiency": token_efficiency,
        "mean_input_tokens": mean_input_tokens,
        "n": n,
    }
