"""Task feature extraction for the DSPy optimizer meta-model."""

from __future__ import annotations


def _avg_field_len(examples: list[dict], field: str) -> float:
    """Average character length of *field* across examples.

    Examples missing the field contribute 0 to the sum but still count
    toward the denominator (i.e. they are treated as zero-length).
    Returns 0.0 for an empty list.
    """
    if not examples:
        return 0.0
    total = sum(len(ex.get(field, "")) for ex in examples)
    return total / len(examples)


def _fraction_numeric(examples: list[dict]) -> float:
    """Fraction of examples whose ``answer`` field parses as a number.

    Handles integers, floats, and negative numbers.  Returns 0.0 for
    an empty list.
    """
    if not examples:
        return 0.0
    count = 0
    for ex in examples:
        answer = ex.get("answer", "")
        try:
            float(answer)
            count += 1
        except (ValueError, TypeError):
            pass
    return count / len(examples)


def _answer_vocab_size(examples: list[dict]) -> int:
    """Number of unique answer strings across *examples*."""
    return len({ex.get("answer", "") for ex in examples}) if examples else 0


def extract_task_features(
    dataset_name: str,
    task_type: str,
    train: list[dict],
    val: list[dict],
    test: list[dict],
) -> dict:
    """Extract per-task features for the meta-model.

    Returns a dict with the following keys:

    - dataset: str
    - task_type: str
    - n_train: int
    - n_val: int
    - n_test: int
    - avg_context_len: float  (avg char length of "context" field)
    - avg_question_len: float (avg char length of "question" field)
    - avg_answer_len: float   (avg char length of "answer" field)
    - has_context: bool       (any example has a non-empty "context")
    - has_choices: bool       (any example has a "choices" key)
    - answer_is_numeric: float (fraction of answers that are numeric)
    - answer_vocab_size: int   (unique answer strings across all splits)
    """
    all_examples = [*train, *val, *test]

    has_context = any(ex.get("context", "") for ex in all_examples)
    has_choices = any("choices" in ex for ex in all_examples)

    return {
        "dataset": dataset_name,
        "task_type": task_type,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "avg_context_len": _avg_field_len(all_examples, "context"),
        "avg_question_len": _avg_field_len(all_examples, "question"),
        "avg_answer_len": _avg_field_len(all_examples, "answer"),
        "has_context": has_context,
        "has_choices": has_choices,
        "answer_is_numeric": _fraction_numeric(all_examples),
        "answer_vocab_size": _answer_vocab_size(all_examples),
    }
