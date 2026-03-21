"""Evaluation runner for MemorySystem implementations (v2).

Supports both the new typed protocol (BenchmarkExample with Item types)
and the legacy dict-based format for backward compatibility with loop.py.

The loop per example:
    system.reset()
    system.ingest(example.items)
    for query in example.queries:
        result = system.query(query.question)
        score(query, result)
"""

from __future__ import annotations

import sys
import time
from typing import Any, Iterable

from context_bench.results import EvalResult, EvalRow


def evaluate_memory(
    systems: list[Any],
    dataset: Iterable[Any],
    evaluators: list[Any],
    metrics: list[Any] | None = None,
    max_examples: int | None = None,
    progress: bool = True,
) -> EvalResult:
    """Run evaluation over a memory benchmark dataset.

    Accepts either:
    - list[BenchmarkExample] (new typed format)
    - list[dict] with "turns" and "qa_pairs" keys (legacy format)

    Args:
        systems: Objects implementing the MemorySystem protocol.
        dataset: Iterable of examples.
        evaluators: Objects implementing the Evaluator protocol.
        metrics: Optional aggregation metrics.
        max_examples: Limit number of examples evaluated.
        progress: Print progress to stderr.

    Returns:
        EvalResult with one EvalRow per (system, example, query).
    """
    if metrics is None:
        metrics = []

    examples = list(dataset)
    if max_examples is not None:
        examples = examples[:max_examples]

    rows: list[EvalRow] = []
    timing: dict[str, float] = {}

    for system in systems:
        sys_start = time.monotonic()
        sys_rows = _run_system(system, examples, evaluators, progress)
        rows.extend(sys_rows)
        timing[system.name] = time.monotonic() - sys_start

    summary: dict[str, dict[str, float]] = {s.name: {} for s in systems}
    for metric in metrics:
        for system in systems:
            sys_rows = [r for r in rows if r.system == system.name]
            summary[system.name].update(metric.compute(sys_rows))

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


def _run_system(
    system: Any,
    examples: list[Any],
    evaluators: list[Any],
    progress: bool,
) -> list[EvalRow]:
    rows: list[EvalRow] = []

    for i, example in enumerate(examples):
        # Detect whether this is a BenchmarkExample or a legacy dict
        if _is_benchmark_example(example):
            rows.extend(_run_typed_example(system, example, evaluators))
        else:
            rows.extend(_run_legacy_example(system, example, i, evaluators))

        if progress and (i + 1) % 5 == 0:
            total = len(examples)
            print(f"  {system.name}: {i + 1}/{total} examples", file=sys.stderr)

    return rows


def _is_benchmark_example(example: Any) -> bool:
    """Check if example is a BenchmarkExample (has .items and .queries attrs)."""
    return hasattr(example, "items") and hasattr(example, "queries")


def _run_typed_example(
    system: Any,
    example: Any,
    evaluators: list[Any],
) -> list[EvalRow]:
    """Run a BenchmarkExample through the system."""
    from context_bench.memory_types import QueryResult

    rows: list[EvalRow] = []
    system.reset()

    t_ingest = time.monotonic()
    ingest_result = system.ingest(example.items)
    ingest_latency = time.monotonic() - t_ingest

    for j, bq in enumerate(example.queries):
        t_query = time.monotonic()
        try:
            result = system.query(bq.question)
        except Exception:
            result = QueryResult(answer="", total_latency_ms=0.0)
        query_latency = time.monotonic() - t_query

        # Build pseudo dicts for existing evaluators
        context_snippet = _items_to_context_snippet(example.items, max_chars=3000)
        pseudo_input = {
            "id": f"{example.id}_{j}",
            "dataset": example.dataset,
            "question": bq.question,
            "answer": bq.answer,
            "context": context_snippet,
        }
        pseudo_output = {**pseudo_input, "response": result.answer}

        scores: dict[str, float] = {}
        for evaluator in evaluators:
            scores.update(evaluator.score(pseudo_input, pseudo_output))

        rows.append(EvalRow(
            system=system.name,
            example_id=f"{example.id}_{j}",
            scores=scores,
            input_tokens=result.context_tokens,
            output_tokens=len(result.answer.split()),
            metadata={
                "qa_type": bq.query_type,
                "ingest_latency": ingest_latency,
                "query_latency": query_latency,
                "conversation_id": example.id,
                "turn_count": len(example.items),
                **result.details,
            },
            latency=query_latency,
            dataset=example.dataset,
        ))

    return rows


def _run_legacy_example(
    system: Any,
    example: dict[str, Any],
    index: int,
    evaluators: list[Any],
) -> list[EvalRow]:
    """Run a legacy dict-format example through the system."""
    rows: list[EvalRow] = []
    conversation_id = example.get("id", index)
    dataset_tag = example.get("dataset", "")
    turns = example.get("turns", [])
    qa_pairs = example.get("qa_pairs", [])

    system.reset()

    t_ingest = time.monotonic()
    system.ingest(turns)
    ingest_latency = time.monotonic() - t_ingest

    for j, qa in enumerate(qa_pairs):
        question = qa["question"]
        gold = qa["answer"]
        qa_type = qa.get("type", "")

        t_query = time.monotonic()
        try:
            answer = system.query(question)
            # Handle both str and QueryResult returns
            if hasattr(answer, "answer"):
                answer_str = answer.answer
                context_tokens = answer.context_tokens
            else:
                answer_str = str(answer)
                context_tokens = getattr(system, "last_context_tokens", None)
                if context_tokens is None:
                    context_tokens = sum(
                        len(t.get("content", "").split()) for t in turns
                    )
        except Exception:
            answer_str = ""
            context_tokens = 0
        query_latency = time.monotonic() - t_query

        context_snippet = _truncate_turns(turns, max_chars=3000)
        pseudo_input = {
            "id": f"{conversation_id}_{j}",
            "dataset": dataset_tag,
            "question": question,
            "answer": gold,
            "context": context_snippet,
        }
        pseudo_output = {**pseudo_input, "response": answer_str}

        scores: dict[str, float] = {}
        for evaluator in evaluators:
            scores.update(evaluator.score(pseudo_input, pseudo_output))

        rows.append(EvalRow(
            system=system.name,
            example_id=f"{conversation_id}_{j}",
            scores=scores,
            input_tokens=context_tokens,
            output_tokens=len(answer_str.split()),
            metadata={
                "qa_type": qa_type,
                "ingest_latency": ingest_latency,
                "query_latency": query_latency,
                "conversation_id": conversation_id,
                "turn_count": len(turns),
            },
            latency=query_latency,
            dataset=dataset_tag,
        ))

    return rows


def _items_to_context_snippet(items: list[Any], max_chars: int = 3000) -> str:
    """Serialize typed items to a string, keeping the most recent content."""
    from context_bench.memory_types import ConversationTurn, DocumentChunk, PlatformEvent, Declaration

    lines: list[str] = []
    total = 0
    for item in reversed(items):
        if isinstance(item, ConversationTurn):
            line = f"{item.role.upper()}: {item.content}"
        elif isinstance(item, DocumentChunk):
            line = item.content
        elif isinstance(item, PlatformEvent):
            line = f"[{item.platform}] {item.content}"
        elif isinstance(item, Declaration):
            line = f"{item.key}: {item.value}"
        else:
            line = str(item)
        if total + len(line) > max_chars and lines:
            break
        lines.append(line)
        total += len(line) + 1
    lines.reverse()
    return "\n".join(lines)


def _truncate_turns(turns: list[dict[str, Any]], max_chars: int = 3000) -> str:
    """Serialize legacy conversation turns to a string."""
    lines: list[str] = []
    total = 0
    for turn in reversed(turns):
        role = turn.get("role", "user").upper()
        content = turn.get("content", "").strip()
        line = f"{role}: {content}"
        if total + len(line) > max_chars and lines:
            break
        lines.append(line)
        total += len(line) + 1
    lines.reverse()
    return "\n".join(lines)
