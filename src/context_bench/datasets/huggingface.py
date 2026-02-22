"""HuggingFace dataset loaders.

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


def hotpotqa(n: int | None = None, split: str = "validation") -> list[dict[str, Any]]:
    """Load HotpotQA multi-hop QA dataset.

    Each example has: id, context, question, answer.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("hotpot_qa", "fullwiki", split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        # Combine supporting facts into context
        context_parts = []
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            context_parts.append(f"{title}: {''.join(sentences)}")

        examples.append({
            "id": item["id"],
            "context": "\n".join(context_parts),
            "question": item["question"],
            "answer": item["answer"],
            "type": item.get("type", ""),
        })

    return examples


def gsm8k(n: int | None = None, split: str = "test") -> list[dict[str, Any]]:
    """Load GSM8K math reasoning dataset.

    Each example has: id, context (question), answer.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("gsm8k", "main", split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        # Extract numeric answer from "#### <number>" format
        answer_text = item["answer"]
        numeric = answer_text.split("####")[-1].strip() if "####" in answer_text else answer_text

        examples.append({
            "id": i,
            "context": item["question"],
            "question": item["question"],
            "answer": numeric,
            "reasoning": answer_text,
        })

    return examples


def bfcl_simple(n: int | None = None) -> list[dict[str, Any]]:
    """Load BFCL function-calling examples from multiple subsets.

    Loads live_multiple, live_simple, rest, and exec_multiple JSONL files
    from the BFCL v3 dataset. Each example has: id, context, tools, answer.
    """
    import json
    from pathlib import Path

    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        repo_type="dataset",
    )
    snap = Path(local_dir)

    data: list[dict] = []
    for filename in [
        "BFCL_v3_live_multiple.json",
        "BFCL_v3_live_simple.json",
        "BFCL_v3_rest.json",
        "BFCL_v3_exec_multiple.json",
    ]:
        f = snap / filename
        if f.exists():
            with open(f) as fh:
                for line in fh:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    examples = []
    for i, item in enumerate(data):
        if n is not None and i >= n:
            break
        functions = item.get("function", [])
        if not functions:
            continue
        context = json.dumps(functions, indent=2)
        gt = item.get("ground_truth", [])
        if isinstance(gt, list) and gt:
            answer = gt[0] if isinstance(gt[0], str) else json.dumps(gt[0])
        else:
            answer = functions[0].get("name", "") if functions else ""
        examples.append({
            "id": item.get("id", i),
            "context": context,
            "tools": functions,
            "answer": answer,
        })

    return examples
