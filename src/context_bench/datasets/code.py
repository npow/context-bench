"""Code generation dataset loaders.

Requires: pip install context-bench[datasets]
"""

from __future__ import annotations

from typing import Any


def _require_datasets() -> Any:
    try:
        import datasets
        return datasets
    except ImportError:
        raise ImportError(
            "HuggingFace datasets required. Install with: pip install context-bench[datasets]"
        )


def humaneval(n: int | None = None, split: str = "test") -> list[dict[str, Any]]:
    """Load OpenAI HumanEval code generation dataset.

    Each example has: id, context, question, answer.
    Note: F1 against canonical_solution is a rough proxy for code quality;
    proper evaluation requires execution-based pass@k.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("openai/openai_humaneval", split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        examples.append({
            "id": item.get("task_id", i),
            "context": item["prompt"],
            "question": item["prompt"],
            "answer": item.get("canonical_solution", ""),
            "test": item.get("test", ""),
            "entry_point": item.get("entry_point", ""),
        })

    return examples


def mbpp(n: int | None = None, split: str = "test") -> list[dict[str, Any]]:
    """Load Google MBPP (sanitized) code generation dataset.

    Each example has: id, context, question, answer.
    Note: F1 against reference code is a rough proxy; proper evaluation
    requires execution-based pass@k.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset(
        "google-research-datasets/mbpp", "sanitized", split=split,
    )

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        examples.append({
            "id": item.get("task_id", i),
            "context": item.get("prompt", ""),
            "question": item.get("prompt", ""),
            "answer": item.get("code", ""),
            "test_list": item.get("test_list", []),
            "test_setup_code": item.get("test_setup_code", ""),
        })

    return examples
