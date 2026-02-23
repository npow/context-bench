"""Reasoning dataset loaders.

Requires: pip install context-bench[datasets]
"""

from __future__ import annotations

import re
from typing import Any


def _require_datasets() -> Any:
    try:
        import datasets
        return datasets
    except ImportError:
        raise ImportError(
            "HuggingFace datasets required. Install with: pip install context-bench[datasets]"
        )


def _extract_boxed(text: str) -> str:
    r"""Extract content from \boxed{...} with brace matching.

    Returns the content inside the outermost \boxed{}, or empty string if
    no match is found.
    """
    start = text.find(r"\boxed{")
    if start == -1:
        return ""
    # Start after \boxed{
    i = start + len(r"\boxed{")
    depth = 1
    result = []
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
            result.append(text[i])
        elif text[i] == "}":
            depth -= 1
            if depth > 0:
                result.append(text[i])
        else:
            result.append(text[i])
        i += 1
    return "".join(result)


def drop(
    n: int | None = None, split: str = "validation",
) -> list[dict[str, Any]]:
    """Load DROP (Discrete Reasoning Over Paragraphs) dataset.

    Each example has: id, context, question, answer.
    Answers can be spans, numbers, or dates.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("ucinlp/drop", split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break

        # Extract first answer span; fall back to number or date
        spans = item.get("answers_spans", {})
        span_list = spans.get("spans", []) if isinstance(spans, dict) else []
        if span_list:
            answer = span_list[0]
        else:
            answer = ""

        examples.append({
            "id": item.get("query_id", i),
            "context": item.get("passage", ""),
            "question": item.get("question", ""),
            "answer": answer,
        })

    return examples


def math_dataset(
    n: int | None = None, split: str = "test",
) -> list[dict[str, Any]]:
    r"""Load MATH (Competition Mathematics) dataset.

    Extracts the final answer from \boxed{} in the solution field.
    Each example has: id, context, question, answer.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("hendrycks/competition_math", split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break

        problem = item.get("problem", "")
        solution = item.get("solution", "")
        answer = _extract_boxed(solution) or solution

        examples.append({
            "id": i,
            "context": problem,
            "question": problem,
            "answer": answer,
        })

    return examples
