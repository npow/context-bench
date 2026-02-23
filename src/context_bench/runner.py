"""Core evaluation orchestration."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable

from context_bench.results import EvalResult, EvalRow
from context_bench.utils.tokens import count_tokens_dict


def _process_example(
    system: Any,
    example: dict[str, Any],
    index: int,
    evaluators: list[Any],
    text_fields: list[str] | None,
) -> EvalRow:
    """Process a single example through a system and score it.

    This function is the unit of work for both sequential and concurrent
    execution.
    """
    example_id = example.get("id", index)
    dataset_tag = example.get("dataset", "")

    # Count input tokens
    input_tokens = count_tokens_dict(example, text_fields=text_fields)

    # Process and time it
    t0 = time.monotonic()
    processed = system.process(example)
    latency = time.monotonic() - t0

    # Count output tokens
    output_tokens = count_tokens_dict(processed, text_fields=text_fields)

    # Score with all evaluators
    scores: dict[str, float] = {}
    for evaluator in evaluators:
        eval_scores = evaluator.score(example, processed)
        scores.update(eval_scores)

    return EvalRow(
        system=system.name,
        example_id=example_id,
        scores=scores,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        metadata={},
        latency=latency,
        dataset=dataset_tag,
    )


def evaluate(
    systems: list[Any],  # list[System], but avoid import cycle
    dataset: Iterable[dict[str, Any]],
    evaluators: list[Any],  # list[Evaluator]
    metrics: list[Any] | None = None,
    max_examples: int | None = None,
    progress: bool = True,
    text_fields: list[str] | None = None,
    max_workers: int | None = None,
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
        max_workers: Max threads for concurrent example processing. If None or
            1, examples are processed sequentially. Values > 1 use a
            ThreadPoolExecutor.

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

    use_concurrent = max_workers is not None and max_workers > 1

    for system in systems:
        sys_start = time.monotonic()

        if use_concurrent:
            sys_rows = _run_concurrent(
                system, examples, evaluators, text_fields, max_workers, progress,
            )
        else:
            sys_rows = _run_sequential(
                system, examples, evaluators, text_fields, progress,
            )

        rows.extend(sys_rows)
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
            "max_workers": max_workers or 1,
        },
    )


def _run_sequential(
    system: Any,
    examples: list[dict[str, Any]],
    evaluators: list[Any],
    text_fields: list[str] | None,
    progress: bool,
) -> list[EvalRow]:
    """Process examples sequentially."""
    sys_rows: list[EvalRow] = []
    for i, example in enumerate(examples):
        row = _process_example(system, example, i, evaluators, text_fields)
        sys_rows.append(row)

        if progress and (i + 1) % 10 == 0:
            import sys as _sys
            print(
                f"  {system.name}: {i + 1}/{len(examples)}",
                file=_sys.stderr,
            )
    return sys_rows


def _run_concurrent(
    system: Any,
    examples: list[dict[str, Any]],
    evaluators: list[Any],
    text_fields: list[str] | None,
    max_workers: int,
    progress: bool,
) -> list[EvalRow]:
    """Process examples concurrently using ThreadPoolExecutor.

    Results are collected and sorted by original index to maintain
    deterministic ordering.
    """
    # Map future -> original index for ordering
    results: dict[int, EvalRow] = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for i, example in enumerate(examples):
            future = executor.submit(
                _process_example, system, example, i, evaluators, text_fields,
            )
            future_to_idx[future] = i

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            completed += 1

            if progress and completed % 10 == 0:
                import sys as _sys
                print(
                    f"  {system.name}: {completed}/{len(examples)}",
                    file=_sys.stderr,
                )

    # Return in original order
    return [results[i] for i in range(len(examples))]
