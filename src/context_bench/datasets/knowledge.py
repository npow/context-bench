"""Knowledge and multiple-choice dataset loaders.

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


_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def _format_choices(choices: list[str]) -> str:
    """Format a list of choices as 'A) ... B) ... C) ...'."""
    return "  ".join(
        f"{_LETTERS[i]}) {c}" for i, c in enumerate(choices) if i < len(_LETTERS)
    )


def mmlu(
    n: int | None = None, config: str = "all", split: str = "test",
) -> list[dict[str, Any]]:
    """Load MMLU (Massive Multitask Language Understanding) dataset.

    Each example has: id, context, question, answer, choices, correct_letter.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("cais/mmlu", config, split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        choices = item["choices"]
        answer_idx = item["answer"]
        correct_letter = _LETTERS[answer_idx] if answer_idx < len(_LETTERS) else "A"
        answer_text = choices[answer_idx] if answer_idx < len(choices) else ""

        question = item["question"]
        context = f"{question}\n\n{_format_choices(choices)}"

        examples.append({
            "id": i,
            "context": context,
            "question": question,
            "answer": answer_text,
            "choices": choices,
            "correct_letter": correct_letter,
        })

    return examples


def arc_challenge(
    n: int | None = None, split: str = "test",
) -> list[dict[str, Any]]:
    """Load ARC-Challenge (AI2 Reasoning Challenge) dataset.

    Each example has: id, context, question, answer, choices, correct_letter.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break

        labels = item["choices"]["label"]
        texts = item["choices"]["text"]
        answer_key = item["answerKey"]

        # Find correct answer index
        correct_idx = None
        for j, label in enumerate(labels):
            if label == answer_key:
                correct_idx = j
                break

        if correct_idx is None:
            continue

        answer_text = texts[correct_idx]
        correct_letter = _LETTERS[correct_idx] if correct_idx < len(_LETTERS) else answer_key

        question = item["question"]
        context = f"{question}\n\n{_format_choices(texts)}"

        examples.append({
            "id": item.get("id", i),
            "context": context,
            "question": question,
            "answer": answer_text,
            "choices": texts,
            "correct_letter": correct_letter,
        })

    return examples


def truthfulqa(
    n: int | None = None, split: str = "validation",
) -> list[dict[str, Any]]:
    """Load TruthfulQA generation dataset.

    Each example has: id, context, question, answer.
    Only has a validation split.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("truthfulqa/truthful_qa", "generation", split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        examples.append({
            "id": i,
            "context": item["question"],
            "question": item["question"],
            "answer": item.get("best_answer", ""),
        })

    return examples


def gpqa(
    n: int | None = None, split: str = "train",
) -> list[dict[str, Any]]:
    """Load GPQA Diamond (Graduate-level Google-Proof QA) dataset.

    Gated dataset with title-case columns. Each example has: id, context,
    question, answer, choices, correct_letter.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("Idavidrein/gpqa", "gpqa_diamond", split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break

        correct_answer = item.get("Correct Answer", "")
        choices = [
            correct_answer,
            item.get("Incorrect Answer 1", ""),
            item.get("Incorrect Answer 2", ""),
            item.get("Incorrect Answer 3", ""),
        ]
        # The correct answer is always at index 0 after our construction
        correct_letter = "A"

        question = item.get("Question", "")
        context = f"{question}\n\n{_format_choices(choices)}"

        examples.append({
            "id": i,
            "context": context,
            "question": question,
            "answer": correct_answer,
            "choices": choices,
            "correct_letter": correct_letter,
        })

    return examples
