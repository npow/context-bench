"""Instruction-following dataset loaders.

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


def alpaca_eval(n: int | None = None, split: str = "eval") -> list[dict[str, Any]]:
    """Load AlpacaEval instruction-following dataset.

    Each example has: id, context, question, answer.
    The ``answer`` field contains the reference output from text-davinci-003.
    Best evaluated with an LLM judge (``--judge-url``).
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("tatsu-lab/alpaca_eval", split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break

        instruction = item.get("instruction", "")
        output = item.get("output", "")

        examples.append({
            "id": i,
            "context": instruction,
            "question": instruction,
            "answer": output,
        })

    return examples
