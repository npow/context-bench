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


def _nolima_expand(needle_cfg: dict, haystack_text: str) -> list[dict[str, Any]]:
    """Expand a single NoLiMa needle config into concrete examples.

    Each needle config has template placeholders ({CHAR}, {1}, {2}, …) and
    multiple test cases.  We use the first character and the ``onehop``
    question for each test, inserting the expanded needle into the haystack.
    """
    import random

    needle_tpl = needle_cfg.get("needle", "")
    questions = needle_cfg.get("questions", {})
    question_tpl = questions.get("onehop", questions.get("twohop", ""))
    characters = needle_cfg.get("character_set", [])
    tests = needle_cfg.get("tests", {})
    cfg_id = needle_cfg.get("id", "")

    results = []
    for test_id, test_data in tests.items():
        args = test_data.get("input_args", [])
        # Pick a deterministic character from the set
        char = characters[0] if characters else "Alice"

        # Expand needle template: {CHAR} -> character, {1},{2},… -> args
        expanded_needle = needle_tpl.replace("{CHAR}", char)
        for j, arg in enumerate(args, 1):
            expanded_needle = expanded_needle.replace(f"{{{j}}}", arg)

        # Expand question template
        expanded_question = question_tpl.replace("{CHAR}", char)
        for j, arg in enumerate(args, 1):
            expanded_question = expanded_question.replace(f"{{{j}}}", arg)

        # Insert needle at a random position in the haystack
        rng = random.Random(hash(f"{cfg_id}_{test_id}"))
        lines = haystack_text.split("\n")
        insert_pos = rng.randint(0, max(len(lines) - 1, 0))
        lines.insert(insert_pos, expanded_needle)
        context = "\n".join(lines)

        # The answer is the character name (the needle reveals info about CHAR)
        results.append({
            "id": f"{cfg_id}_{test_id}",
            "context": context,
            "question": expanded_question,
            "answer": char,
        })

    return results


def nolima(n: int | None = None, needle_set: str = "needle_set.json") -> list[dict[str, Any]]:
    """Load NoLiMa needle-in-a-haystack evaluation dataset.

    Downloads via snapshot_download (same pattern as bfcl_simple).
    Expands needle templates with haystack text to produce concrete examples.

    Args:
        n: Maximum number of examples to return.
        needle_set: Which needle set file to use from the ``needlesets/``
            directory (default: ``needle_set.json``).

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

    # Load needle set
    needle_file = snap / "needlesets" / needle_set
    if not needle_file.exists():
        available = [f.name for f in (snap / "needlesets").glob("*.json")] if (snap / "needlesets").exists() else []
        raise FileNotFoundError(
            f"Needle set {needle_set!r} not found. Available: {available}"
        )

    with open(needle_file) as fh:
        needles = json.load(fh)

    # Load first haystack book
    haystack_text = ""
    haystack_dir = snap / "haystack" / "rand_shuffle"
    if haystack_dir.exists():
        for txt_file in sorted(haystack_dir.glob("*.txt")):
            with open(txt_file) as fh:
                haystack_text = fh.read()
            break  # Use only the first book

    if not haystack_text:
        # Fallback: try any .txt file in haystack/
        for txt_file in sorted((snap / "haystack").rglob("*.txt")):
            with open(txt_file) as fh:
                haystack_text = fh.read()
            break

    # Expand all needle configs into concrete examples
    examples = []
    for needle_cfg in needles:
        expanded = _nolima_expand(needle_cfg, haystack_text)
        for ex in expanded:
            if n is not None and len(examples) >= n:
                return examples
            examples.append(ex)

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
