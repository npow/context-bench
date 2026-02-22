"""Agent trace dataset loaders.

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


def apigen_mt(n: int | None = None) -> list[dict[str, Any]]:
    """Load APIGen multi-turn tool calling traces (Salesforce).

    Each example has: id, context (conversation), tools, answer.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("Salesforce/xlam-function-calling-60k", split="train")

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        examples.append({
            "id": i,
            "context": str(item.get("query", "")),
            "tools": item.get("tools", ""),
            "answer": item.get("answers", ""),
        })

    return examples


def swebench(n: int | None = None, split: str = "test") -> list[dict[str, Any]]:
    """Load SWE-bench full dataset (2,294 examples).

    Each example has: id, context (repo + issue), answer (patch).
    Context is the problem statement — the text an agent sees before exploring code.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("princeton-nlp/SWE-bench", split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        context = f"Repository: {item['repo']}\n\nIssue:\n{item['problem_statement']}"
        examples.append({
            "id": item.get("instance_id", i),
            "context": context,
            "repo": item["repo"],
            "answer": item.get("patch", ""),
            "base_commit": item.get("base_commit", ""),
        })

    return examples


def swebench_verified(n: int | None = None) -> list[dict[str, Any]]:
    """Load SWE-bench Verified (500 human-validated examples).

    The current standard subset for comparing coding agents.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        context = f"Repository: {item['repo']}\n\nIssue:\n{item['problem_statement']}"
        examples.append({
            "id": item.get("instance_id", i),
            "context": context,
            "repo": item["repo"],
            "answer": item.get("patch", ""),
            "base_commit": item.get("base_commit", ""),
        })

    return examples


def swebench_lite(n: int | None = None) -> list[dict[str, Any]]:
    """Load SWE-bench Lite (300 examples).

    Smaller subset for quick evaluation.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        context = f"Repository: {item['repo']}\n\nIssue:\n{item['problem_statement']}"
        examples.append({
            "id": item.get("instance_id", i),
            "context": context,
            "repo": item["repo"],
            "answer": item.get("patch", ""),
            "base_commit": item.get("base_commit", ""),
        })

    return examples


# Keep old name as alias
swe_agent_traces = swebench_lite
