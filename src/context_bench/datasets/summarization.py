"""Summarization dataset loaders.

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


def multi_news(n: int | None = None, split: str = "test") -> list[dict[str, Any]]:
    """Load Multi-News multi-document summarization dataset.

    Each example has: id, context, question, answer.
    Documents are joined by ``|||||`` separators in the source field.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("alexfabbri/multi_news", split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        examples.append({
            "id": i,
            "context": item.get("document", ""),
            "question": "Summarize the news articles.",
            "answer": item.get("summary", ""),
        })

    return examples


def dialogsum(n: int | None = None, split: str = "test") -> list[dict[str, Any]]:
    """Load DialogSum dialogue summarization dataset.

    Each example has: id, context, question, answer.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("knkarthick/dialogsum", split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        examples.append({
            "id": item.get("id", i),
            "context": item.get("dialogue", ""),
            "question": "Summarize the dialogue.",
            "answer": item.get("summary", ""),
        })

    return examples


def qmsum(n: int | None = None, split: str = "validation") -> list[dict[str, Any]]:
    """Load QMSum query-based meeting summarization dataset (via SCROLLS).

    Each example has: id, context, question, answer.
    The ``input`` field contains the transcript followed by a query;
    we split on the last occurrence of ``\\nQuery:`` to separate them.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset(
        "tau/scrolls", "qmsum", split=split, trust_remote_code=True,
    )

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        raw_input = item.get("input", "")
        # SCROLLS QMSum format: transcript text followed by "\nQuery: <query>"
        sep = "\nQuery:"
        if sep in raw_input:
            idx = raw_input.rfind(sep)
            context = raw_input[:idx].strip()
            question = raw_input[idx + len(sep):].strip()
        else:
            context = raw_input
            question = "Summarize the meeting transcript."

        examples.append({
            "id": item.get("id", i),
            "context": context,
            "question": question,
            "answer": item.get("output", ""),
        })

    return examples


def summscreenfd(n: int | None = None, split: str = "validation") -> list[dict[str, Any]]:
    """Load SummScreenFD TV transcript summarization dataset (via SCROLLS).

    Each example has: id, context, question, answer.
    Uses validation split because test split has empty outputs.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset(
        "tau/scrolls", "summ_screen_fd", split=split, trust_remote_code=True,
    )

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        examples.append({
            "id": item.get("id", i),
            "context": item.get("input", ""),
            "question": "Summarize the TV episode transcript.",
            "answer": item.get("output", ""),
        })

    return examples
