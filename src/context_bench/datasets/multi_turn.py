"""Multi-turn conversation dataset loaders.

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


def mt_bench(n: int | None = None, split: str = "train") -> list[dict[str, Any]]:
    """Load MT-Bench multi-turn conversation prompts.

    Each example is a 2-turn conversation. The ``turns`` field contains user
    messages as ``{"role": "user", "content": ...}`` dicts, and
    ``multi_turn`` is set to ``True`` so the runner dispatches to
    ``process_conversation()``.

    The ``answer`` field is empty because MT-Bench is judged by an LLM judge,
    not by reference matching.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("HuggingFaceH4/mt_bench_prompts", split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break

        raw_turns = item.get("turns", [])
        # Each turn is a user message string
        turns = [{"role": "user", "content": t} for t in raw_turns]

        category = item.get("category", "")
        prompt_id = item.get("prompt_id", i)

        examples.append({
            "id": prompt_id,
            "multi_turn": True,
            "turns": turns,
            "context": raw_turns[0] if raw_turns else "",
            "question": raw_turns[-1] if raw_turns else "",
            "answer": "",
            "category": category,
        })

    return examples
