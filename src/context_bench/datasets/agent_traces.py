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


def swe_agent_traces(n: int | None = None) -> list[dict[str, Any]]:
    """Load SWE-agent coding traces (JetBrains SWE-bench).

    Each example has: id, context (issue + repo), answer (patch).
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
