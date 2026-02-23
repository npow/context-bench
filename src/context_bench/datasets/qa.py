"""QA dataset loaders.

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


def natural_questions(n: int | None = None, split: str = "validation") -> list[dict[str, Any]]:
    """Load Natural Questions open-domain QA dataset.

    Each example has: id, context, question, answer.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("google-research-datasets/nq_open", split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        examples.append({
            "id": i,
            "context": item["question"],
            "question": item["question"],
            "answer": item["answer"][0] if item.get("answer") else "",
        })

    return examples


def musique(n: int | None = None, split: str = "validation") -> list[dict[str, Any]]:
    """Load MuSiQue multi-hop QA dataset (answerable subset).

    Each example has: id, context, question, answer.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("bdsaglam/musique", "answerable", split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        context_parts = []
        for p in item.get("paragraphs", []):
            title = p.get("title", "")
            text = p.get("paragraph_text", "")
            context_parts.append(f"{title}: {text}")

        examples.append({
            "id": item.get("id", i),
            "context": "\n".join(context_parts),
            "question": item["question"],
            "answer": item["answer"],
        })

    return examples


def narrativeqa(n: int | None = None, split: str = "test") -> list[dict[str, Any]]:
    """Load NarrativeQA reading comprehension dataset.

    Uses document summaries as context (not full books).
    Each example has: id, context, question, answer.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("deepmind/narrativeqa", split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        doc = item.get("document", {})
        summary = doc.get("summary", {}) if isinstance(doc, dict) else {}
        context = summary.get("text", "") if isinstance(summary, dict) else ""

        answers = item.get("answers", [])
        answer = answers[0]["text"] if answers and isinstance(answers[0], dict) else ""

        examples.append({
            "id": i,
            "context": context,
            "question": item["question"],
            "answer": answer,
        })

    return examples


def triviaqa(n: int | None = None, split: str = "validation") -> list[dict[str, Any]]:
    """Load TriviaQA reading comprehension dataset.

    Uses search result contexts when available.
    Each example has: id, context, question, answer.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("mandarjoshi/trivia_qa", "rc", split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        search = item.get("search_results", {})
        contexts = search.get("search_context", []) if isinstance(search, dict) else []
        context = "\n".join(c for c in contexts if c) if contexts else item.get("question", "")

        answer_obj = item.get("answer", {})
        answer = answer_obj.get("value", "") if isinstance(answer_obj, dict) else ""

        examples.append({
            "id": item.get("question_id", i),
            "context": context,
            "question": item["question"],
            "answer": answer,
        })

    return examples


def frames(n: int | None = None, split: str = "test") -> list[dict[str, Any]]:
    """Load FRAMES multi-hop factual reasoning benchmark.

    Each example has: id, context, question, answer.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("google/frames-benchmark", split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        examples.append({
            "id": i,
            "context": item.get("Prompt", ""),
            "question": item.get("Prompt", ""),
            "answer": item.get("Answer", ""),
        })

    return examples


def quality(n: int | None = None, split: str = "validation") -> list[dict[str, Any]]:
    """Load QuALITY long-document multiple-choice QA dataset.

    Resolves 1-indexed gold_label to the actual option text.
    Each example has: id, context, question, answer.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("tasksource/QuALITY", split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        options = item.get("options", [])
        gold_label = item.get("gold_label", 0)
        # gold_label is 1-indexed
        idx = gold_label - 1 if gold_label >= 1 else 0
        answer = options[idx] if 0 <= idx < len(options) else ""

        examples.append({
            "id": i,
            "context": item.get("article", ""),
            "question": item.get("question", ""),
            "answer": answer,
        })

    return examples
