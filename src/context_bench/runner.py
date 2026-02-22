"""Core evaluation orchestration."""

from __future__ import annotations

import time
from typing import Any, Iterable

from context_bench.results import EvalResult, EvalRow
from context_bench.utils.tokens import count_tokens_dict


def evaluate(
    systems: list[Any],  # list[System], but avoid import cycle
    dataset: Iterable[dict[str, Any]],
    evaluators: list[Any],  # list[Evaluator]
    metrics: list[Any] | None = None,
    max_examples: int | None = None,
    progress: bool = True,
    text_fields: list[str] | None = None,
) -> EvalResult:
    """Run evaluation: apply systems to dataset, score with evaluators, aggregate with metrics.

    Args:
        systems: Objects implementing the System protocol (.name, .process()).
        dataset: Iterable of example dicts. Each should have an "id" field.
        evaluators: Objects implementing the Evaluator protocol (.name, .score()).
        metrics: Optional list of Metric objects for aggregation.
        max_examples: Limit number of examples processed.
        progress: Whether to print progress (simple stderr output).
        text_fields: Optional list of dict keys to count tokens for. If None,
            counts tokens in all string values.

    Returns:
        EvalResult with per-row scores and summary statistics.
    """
    if metrics is None:
        metrics = []

    # Materialize dataset
    examples = list(dataset)
    if max_examples is not None:
        examples = examples[:max_examples]

    rows: list[EvalRow] = []
    timing: dict[str, float] = {}

    for system in systems:
        sys_start = time.monotonic()
        for i, example in enumerate(examples):
            example_id = example.get("id", i)

            # Count input tokens
            input_tokens = count_tokens_dict(example, text_fields=text_fields)

            # Process
            processed = system.process(example)

            # Count output tokens
            output_tokens = count_tokens_dict(processed, text_fields=text_fields)

            # Score with all evaluators
            scores: dict[str, float] = {}
            for evaluator in evaluators:
                eval_scores = evaluator.score(example, processed)
                scores.update(eval_scores)

            rows.append(
                EvalRow(
                    system=system.name,
                    example_id=example_id,
                    scores=scores,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    metadata={},
                )
            )

            if progress and (i + 1) % 10 == 0:
                import sys as _sys

                print(
                    f"  {system.name}: {i + 1}/{len(examples)}",
                    file=_sys.stderr,
                )

        sys_elapsed = time.monotonic() - sys_start
        timing[system.name] = sys_elapsed

    # Compute summary via metrics
    summary: dict[str, dict[str, float]] = {s.name: {} for s in systems}
    for metric in metrics:
        for system in systems:
            sys_rows = [r for r in rows if r.system == system.name]
            metric_values = metric.compute(sys_rows)
            summary[system.name].update(metric_values)

    return EvalResult(
        rows=rows,
        summary=summary,
        timing=timing,
        config={
            "systems": [s.name for s in systems],
            "evaluators": [e.name for e in evaluators],
            "metrics": [m.name for m in metrics],
            "num_examples": len(examples),
        },
    )
