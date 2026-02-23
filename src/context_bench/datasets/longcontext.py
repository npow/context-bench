"""Long-context benchmark dataset loaders.

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


def longbench(n: int | None = None, config: str = "hotpotqa", split: str = "test") -> list[dict[str, Any]]:
    """Load LongBench long-context evaluation dataset.

    Supports 21 sub-task configs (e.g. hotpotqa, qasper, narrativeqa, etc.).
    Each example has: id, context, question, answer.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("THUDM/LongBench", config, split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        answers = item.get("answers", [])
        answer = answers[0] if answers else ""

        examples.append({
            "id": i,
            "context": item.get("context", ""),
            "question": item.get("input", ""),
            "answer": answer,
        })

    return examples


def longbench_v2(n: int | None = None, split: str = "train") -> list[dict[str, Any]]:
    """Load LongBench v2 long-context evaluation dataset.

    Resolves letter answer (A-D) to the actual choice text.
    Each example has: id, context, question, answer.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("THUDM/LongBench-v2", split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        answer_letter = item.get("answer", "")
        choice_key = f"choice_{answer_letter}" if answer_letter else ""
        answer = item.get(choice_key, answer_letter) if choice_key else ""

        examples.append({
            "id": i,
            "context": item.get("context", ""),
            "question": item.get("question", ""),
            "answer": answer,
        })

    return examples


def infinitebench(n: int | None = None, split: str = "longbook_qa_en") -> list[dict[str, Any]]:
    """Load InfiniteBench long-context evaluation dataset.

    The split parameter selects the subtask (e.g. longbook_qa_en, longdialogue_qa_eng).
    Each example has: id, context, question, answer.
    """
    ds = _require_datasets()
    # InfiniteBench requires explicit features schema for some splits
    features = ds.Features({
        "id": ds.Value("int64"),
        "context": ds.Value("string"),
        "input": ds.Value("string"),
        "answer": ds.Sequence(ds.Value("string")),
        "options": ds.Sequence(ds.Value("string")),
    })
    dataset = ds.load_dataset(
        "xinrongzhang2022/InfiniteBench",
        split=split,
        features=features,
    )

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        answers = item.get("answer", [])
        answer = answers[0] if answers else ""

        examples.append({
            "id": i,
            "context": item.get("context", ""),
            "question": item.get("input", ""),
            "answer": answer,
        })

    return examples


def nolima(n: int | None = None) -> list[dict[str, Any]]:
    """Load NoLiMa needle-in-a-haystack evaluation dataset.

    Downloads via snapshot_download (same pattern as bfcl_simple).
    Each example has: id, context, question, answer.
    """
    import json
    from pathlib import Path

    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        "amodaresi/NoLiMa",
        repo_type="dataset",
    )
    snap = Path(local_dir)

    data: list[dict] = []
    for json_file in sorted(snap.glob("*.json")):
        with open(json_file) as fh:
            try:
                content = json.load(fh)
                if isinstance(content, list):
                    data.extend(content)
                elif isinstance(content, dict):
                    data.append(content)
            except json.JSONDecodeError:
                continue

    # Also try jsonl files
    for jsonl_file in sorted(snap.glob("*.jsonl")):
        with open(jsonl_file) as fh:
            for line in fh:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    examples = []
    for i, item in enumerate(data):
        if n is not None and i >= n:
            break
        haystack = item.get("haystack", "")
        needle = item.get("needle", "")
        context = f"{haystack}\n{needle}" if needle else haystack

        examples.append({
            "id": item.get("id", i),
            "context": context,
            "question": item.get("question", item.get("query", "")),
            "answer": item.get("answer", ""),
        })

    return examples


def bbh(n: int | None = None, config: str = "boolean_expressions", split: str = "test") -> list[dict[str, Any]]:
    """Load BIG-Bench Hard reasoning benchmark.

    Supports 27 sub-task configs (e.g. boolean_expressions, causal_judgement, etc.).
    Each example has: id, context, question, answer.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("lukaemon/bbh", config, split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        examples.append({
            "id": i,
            "context": item.get("input", ""),
            "question": item.get("input", ""),
            "answer": item.get("target", ""),
        })

    return examples


def meetingbank(n: int | None = None, split: str = "test") -> list[dict[str, Any]]:
    """Load MeetingBank meeting summarization dataset.

    Each example has: id, context (transcript), question, answer (summary).
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("huuuyeah/meetingbank", split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        examples.append({
            "id": i,
            "context": item.get("transcript", ""),
            "question": "Summarize the meeting transcript.",
            "answer": item.get("summary", ""),
        })

    return examples


def govreport(n: int | None = None, split: str = "test") -> list[dict[str, Any]]:
    """Load GovReport government report summarization dataset.

    Each example has: id, context (report), question, answer (summary).
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("ccdv/govreport-summarization", split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        examples.append({
            "id": i,
            "context": item.get("report", ""),
            "question": "Summarize the government report.",
            "answer": item.get("summary", ""),
        })

    return examples
