"""Scoring function for the autoresearch loop.

Evaluates a ContextPipeline on a sample of conversation examples and returns
a scalar optimization target.

Uses an LLM judge (matching MemMachine/Mem0 LoCoMo evaluation protocol) as
the primary metric so scores are directly comparable to published SOTA numbers.
"""

from __future__ import annotations

from typing import Any, Type

from context_bench.evaluators.llm_judge_locomo import LLMJudgeLoCoMo
from context_bench.memory_runner import evaluate_memory
from context_bench.metrics.token_efficiency import TokenEfficiencyMetric


def score_pipeline(
    pipeline_class: Type[Any],
    examples: list[dict[str, Any]],
    relay_url: str,
    model: str = "claude-haiku-4-5-20251001",
    api_key: str = "",
) -> dict[str, float]:
    """Evaluate a pipeline on a sample of examples using LLM judge scoring.

    Uses the same ACCURACY_PROMPT and CORRECT/WRONG protocol as MemMachine's
    llm_judge.py so results are directly comparable to published SOTA numbers
    on the LoCoMo benchmark.

    Args:
        pipeline_class: A class implementing the MemorySystem protocol.
            Constructor must accept ``relay_url``, ``model``, ``api_key``.
        examples:   List of conversation examples (keys: id, turns, qa_pairs).
        relay_url:  Base URL of the OpenAI-compatible relay.
        model:      Chat model name forwarded to pipeline_class.
        api_key:    Bearer token for the relay.

    Returns:
        ``score``            Primary optimisation target — LLM judge accuracy.
        ``llm_judge``        Mean LLM judge score (0/1 per question).
        ``mean_input_tokens`` Mean word count of ingested conversations.
        ``n``                Number of QA pairs evaluated.
    """
    if not examples:
        return {"score": 0.0, "llm_judge": 0.0, "mean_input_tokens": 0.0, "n": 0}

    pipeline = pipeline_class(
        relay_url=relay_url,
        model=model,
        api_key=api_key if api_key else None,
    )

    evaluator = LLMJudgeLoCoMo(relay_url=relay_url, model="haiku", api_key=api_key)
    metric = TokenEfficiencyMetric(score_field="llm_judge")

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
        return {"score": 0.0, "llm_judge": 0.0, "mean_input_tokens": 0.0, "n": 0}

    system_summary = result.summary.get(pipeline.name, {})
    mean_score = system_summary.get("mean_score", 0.0)
    mean_input_tokens = system_summary.get("mean_input_tokens", 0.0)

    return {
        "score": mean_score,
        "llm_judge": mean_score,
        "f1": mean_score,   # alias for backward compat with loop.py
        "mean_input_tokens": mean_input_tokens,
        "n": n,
    }
