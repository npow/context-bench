"""Core evaluation orchestration."""

from __future__ import annotations

import sys as _sys
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

    Handles both single-turn and multi-turn examples. Multi-turn examples
    have a ``turns`` list of user messages; the system's
    ``process_conversation()`` is called if available, otherwise ``process()``
    is used with the turns embedded in the example.
    """
    example_id = example.get("id", index)
    dataset_tag = example.get("dataset", "")

    # Count input tokens
    input_tokens = count_tokens_dict(example, text_fields=text_fields)

    # Process and time it
    t0 = time.monotonic()
    is_multi_turn = example.get("multi_turn", False) and "turns" in example

    if is_multi_turn and hasattr(system, "process_conversation"):
        user_turns = example["turns"]
        responses = system.process_conversation(user_turns)
        # Final assistant response becomes "response"
        final_response = responses[-1]["content"] if responses else ""
        processed = {
            **example,
            "response": final_response,
            "turn_responses": [r["content"] for r in responses],
        }
    else:
        processed = system.process(example)

    latency = time.monotonic() - t0

    # Count output tokens
    output_tokens = count_tokens_dict(processed, text_fields=text_fields)

    # Extract API usage if present
    api_usage = processed.get("api_usage")
    metadata: dict[str, Any] = {}
    if api_usage and isinstance(api_usage, dict):
        metadata["prompt_tokens"] = api_usage.get("prompt_tokens", 0)
        metadata["completion_tokens"] = api_usage.get("completion_tokens", 0)
        metadata["total_tokens"] = api_usage.get("total_tokens", 0)

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
        metadata=metadata,
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
    cache_dir: str | None = None,
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
        cache_dir: Directory for result caching. If provided, completed rows
            are saved and reused on subsequent runs with the same configuration.

    Returns:
        EvalResult with per-row scores and summary statistics.
    """
    if metrics is None:
        metrics = []

    # Materialize dataset
    examples = list(dataset)
    if max_examples is not None:
        examples = examples[:max_examples]

    # Set up cache if requested
    result_cache = None
    if cache_dir is not None:
        from context_bench.cache import ResultCache
        result_cache = ResultCache(
            cache_dir=cache_dir,
            systems=[s.name for s in systems],
            datasets=sorted({ex.get("dataset", "") for ex in examples}),
            evaluators=[e.name for e in evaluators],
        )
        if result_cache.cached_count > 0 and progress:
            print(
                f"  Cache: {result_cache.cached_count} rows loaded from {result_cache.cache_path}",
                file=_sys.stderr,
            )

    rows: list[EvalRow] = []
    timing: dict[str, float] = {}

    use_concurrent = max_workers is not None and max_workers > 1

    for system in systems:
        sys_start = time.monotonic()

        if use_concurrent:
            sys_rows = _run_concurrent(
                system, examples, evaluators, text_fields, max_workers,
                progress, result_cache,
            )
        else:
            sys_rows = _run_sequential(
                system, examples, evaluators, text_fields, progress,
                result_cache,
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
    cache: Any | None,
) -> list[EvalRow]:
    """Process examples sequentially."""
    sys_rows: list[EvalRow] = []
    cached = 0
    for i, example in enumerate(examples):
        # Check cache
        if cache is not None:
            hit = cache.get(
                system.name,
                example.get("dataset", ""),
                example.get("id", i),
            )
            if hit is not None:
                sys_rows.append(hit)
                cached += 1
                continue

        row = _process_example(system, example, i, evaluators, text_fields)
        sys_rows.append(row)

        # Save to cache
        if cache is not None:
            cache.put(row)

        if progress and (i + 1) % 10 == 0:
            print(
                f"  {system.name}: {i + 1}/{len(examples)}"
                + (f" ({cached} cached)" if cached else ""),
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
    cache: Any | None,
) -> list[EvalRow]:
    """Process examples concurrently using ThreadPoolExecutor.

    Results are collected and sorted by original index to maintain
    deterministic ordering.
    """
    results: dict[int, EvalRow] = {}
    to_process: list[tuple[int, dict[str, Any]]] = []

    # Check cache first
    for i, example in enumerate(examples):
        if cache is not None:
            hit = cache.get(
                system.name,
                example.get("dataset", ""),
                example.get("id", i),
            )
            if hit is not None:
                results[i] = hit
                continue
        to_process.append((i, example))

    if progress and results:
        print(
            f"  {system.name}: {len(results)} cached, {len(to_process)} to process",
            file=_sys.stderr,
        )

    completed = 0

    if to_process:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for i, example in to_process:
                future = executor.submit(
                    _process_example, system, example, i, evaluators, text_fields,
                )
                future_to_idx[future] = i

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                row = future.result()
                results[idx] = row
                completed += 1

                # Save to cache
                if cache is not None:
                    cache.put(row)

                if progress and completed % 10 == 0:
                    print(
                        f"  {system.name}: {completed}/{len(to_process)}",
                        file=_sys.stderr,
                    )

    # Return in original order
    return [results[i] for i in range(len(examples))]
