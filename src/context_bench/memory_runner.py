"""Evaluation runner for stateful MemorySystem implementations.

A MemorySystem is evaluated over conversation examples where each example
contains a full conversation history and one or more QA pairs.

The loop per example:
    system.reset()
    system.ingest(turns)
    for qa in qa_pairs:
        answer = system.query(qa["question"])
        score(answer, qa["answer"])
"""

from __future__ import annotations

import sys
import time
from typing import Any, Iterable

from context_bench.results import EvalResult, EvalRow


def evaluate_memory(
    systems: list[Any],
    dataset: Iterable[dict[str, Any]],
    evaluators: list[Any],
    metrics: list[Any] | None = None,
    max_examples: int | None = None,
    progress: bool = True,
) -> EvalResult:
    """Run evaluation over a memory benchmark dataset.

    Each example in the dataset must have:
        - ``"id"``: unique identifier
        - ``"turns"``: list of ``{"role": ..., "content": ...}`` dicts
        - ``"qa_pairs"``: list of ``{"question": str, "answer": str}`` dicts
        - ``"dataset"`` (optional): dataset tag string

    Args:
        systems: Objects implementing the MemorySystem protocol.
        dataset: Iterable of conversation examples.
        evaluators: Objects implementing the Evaluator protocol.
        metrics: Optional aggregation metrics.
        max_examples: Limit number of conversations evaluated.
        progress: Print progress to stderr.

    Returns:
        EvalResult with one EvalRow per (system, conversation, qa_pair).
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
    examples: list[dict[str, Any]],
    evaluators: list[Any],
    progress: bool,
) -> list[EvalRow]:
    rows: list[EvalRow] = []

    for i, example in enumerate(examples):
        conversation_id = example.get("id", i)
        dataset_tag = example.get("dataset", "")
        turns = example.get("turns", [])
        qa_pairs = example.get("qa_pairs", [])

        # Reset memory state before each conversation
        system.reset()

        # Ingest the full conversation history
        t_ingest = time.monotonic()
        system.ingest(turns)
        ingest_latency = time.monotonic() - t_ingest

        # Query once per QA pair
        for j, qa in enumerate(qa_pairs):
            question = qa["question"]
            gold = qa["answer"]
            qa_type = qa.get("type", "")

            t_query = time.monotonic()
            answer = system.query(question)
            query_latency = time.monotonic() - t_query

            # Prefer the system's own context-token count (words actually sent
            # to the final LLM call) over the full conversation word count.
            # This makes token_efficiency meaningful: a pipeline that retrieves
            # 5 chunks gets credit for sending fewer tokens than one that sends
            # the full history.  Falls back to conversation word count for
            # systems that don't expose last_context_tokens.
            context_tokens = getattr(system, "last_context_tokens", None)
            if context_tokens is None:
                context_tokens = sum(
                    len(t.get("content", "").split()) for t in turns
                )

            # Build pseudo-example/processed dicts for existing evaluators.
            # Include a truncated conversation snippet so LLM-judge evaluators
            # (e.g. FalseMemoryRate) can distinguish recalled facts from hallucinations.
            context_snippet = _truncate_turns(turns, max_chars=3000)
            pseudo_input = {
                "id": f"{conversation_id}_{j}",
                "dataset": dataset_tag,
                "question": question,
                "answer": gold,
                "context": context_snippet,
            }
            pseudo_output = {**pseudo_input, "response": answer}

            scores: dict[str, float] = {}
            for evaluator in evaluators:
                scores.update(evaluator.score(pseudo_input, pseudo_output))

            rows.append(EvalRow(
                system=system.name,
                example_id=f"{conversation_id}_{j}",
                scores=scores,
                input_tokens=context_tokens,
                output_tokens=len(answer.split()),
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

        if progress and (i + 1) % 5 == 0:
            print(f"  {system.name}: {i + 1}/{len(examples)} conversations", file=sys.stderr)

    return rows


def _truncate_turns(turns: list[dict[str, Any]], max_chars: int = 3000) -> str:
    """Serialize conversation turns to a string, keeping the most recent content.

    Starts from the end of the conversation and prepends turns until the
    character budget is exhausted.  This prioritises recent context (where the
    answer fact is most likely to appear) while keeping the context window
    manageable for LLM-judge evaluators.
    """
    lines: list[str] = []
    total = 0
    for turn in reversed(turns):
        role = turn.get("role", "user").upper()
        content = turn.get("content", "").strip()
        line = f"{role}: {content}"
        if total + len(line) > max_chars and lines:
            break
        lines.append(line)
        total += len(line) + 1  # +1 for newline
    lines.reverse()
    return "\n".join(lines)
