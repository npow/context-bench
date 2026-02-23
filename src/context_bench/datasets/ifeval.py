"""IFEval instruction-following dataset loader.

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


def ifeval(
    n: int | None = None, split: str = "train",
) -> list[dict[str, Any]]:
    """Load Google IFEval instruction-following dataset.

    Each example has: id, context, question, answer (empty), plus
    instruction_id_list and kwargs for the evaluator.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("google/IFEval", split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break

        prompt = item.get("prompt", "")
        examples.append({
            "id": i,
            "context": prompt,
            "question": prompt,
            "answer": "",
            "instruction_id_list": item.get("instruction_id_list", []),
            "kwargs": item.get("kwargs", []),
        })

    return examples
