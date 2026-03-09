"""Dataset loaders for memory system benchmarks.

Each loader returns a list of conversation examples compatible with
evaluate_memory() and the MemorySystem protocol.

Every example has:
    id:       unique conversation identifier
    turns:    list of {"role": "user"|"assistant", "content": str}
    qa_pairs: list of {"question": str, "answer": str, "type": str}
    dataset:  dataset tag string

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


# LoCoMo category codes from the paper
_LOCOMO_CATEGORY = {
    1: "single_hop",
    2: "multi_hop",
    3: "temporal",
    4: "open_domain",
    5: "adversarial",
}


def locomo(
    n: int | None = None,
    qa_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Load the LoCoMo long-term conversational memory benchmark.

    50 multi-session conversations with QA pairs covering:
    single_hop, multi_hop, temporal, open_domain, adversarial.

    Reference: Maharana et al., ACL 2024. https://arxiv.org/abs/2402.17753
    HuggingFace: KhangPTT373/locomo_preprocess (split: test)

    Schema:
        turns:     list of turn strings "timestamp\\n Speaker: text\\n"
        sessions:  list of session strings (grouped turns)
        questions: list of question strings
        answers:   list of answer strings
        category:  list of int codes (1-5)

    Args:
        n: Max number of conversations to load.
        qa_types: Filter to specific QA types. Options:
            "single_hop", "multi_hop", "temporal", "open_domain", "adversarial".
            None loads all types.

    Returns:
        List of conversation examples for evaluate_memory().
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("KhangPTT373/locomo_preprocess", split="test")

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break

        # turns is a list of raw strings: "timestamp\n Speaker: text\n"
        # Convert to role/content dicts by detecting speaker alternation.
        # Speaker_1 -> user, Speaker_2 -> assistant
        raw_turns = item.get("turns", [])
        turns: list[dict[str, str]] = []
        for raw in raw_turns:
            # Strip timestamp line, keep content
            lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
            for line in lines:
                if line.startswith("Speaker_1:"):
                    turns.append({"role": "user", "content": line[len("Speaker_1:"):].strip()})
                elif line.startswith("Speaker_2:"):
                    turns.append({"role": "assistant", "content": line[len("Speaker_2:"):].strip()})
                elif ":" in line and not any(c.isdigit() and line[i+1:i+2] == ":" for i, c in enumerate(line)):
                    # Generic "Name: text" format — odd turns are user, even assistant
                    content = line.split(":", 1)[1].strip()
                    role = "user" if len(turns) % 2 == 0 else "assistant"
                    turns.append({"role": role, "content": content})

        questions = item.get("questions", [])
        answers = item.get("answers", [])
        categories = item.get("category", [])

        qa_pairs: list[dict[str, str]] = []
        for q, a, cat in zip(questions, answers, categories):
            qa_type = _LOCOMO_CATEGORY.get(cat, str(cat))
            if qa_types is not None and qa_type not in qa_types:
                continue
            if q and a:
                qa_pairs.append({"question": q, "answer": a, "type": qa_type})

        if not turns or not qa_pairs:
            continue

        examples.append({
            "id": item.get("dialogue_id", i),
            "turns": turns,
            "qa_pairs": qa_pairs,
            "dataset": "locomo",
        })

    return examples


def longmemeval(
    n: int | None = None,
    question_types: list[str] | None = None,
    variant: str = "s",
) -> list[dict[str, Any]]:
    """Load the LongMemEval chat assistant memory benchmark.

    500 QA pairs, each with a full multi-session chat history as context.

    Reference: Wu et al., ICLR 2025. https://arxiv.org/abs/2410.10813
    HuggingFace: xiaowu0162/longmemeval (downloaded via hf_hub_download)

    Schema:
        question_id:       str
        question_type:     str (e.g. "single-session-user")
        question:          str
        answer:            str
        haystack_sessions: list of sessions, each session is a list of
                           {"role": str, "content": str} dicts

    Args:
        n: Max number of examples to load.
        question_types: Filter by question type substring. Options include:
            "single-session-user", "single-session-assistant",
            "multi-session", "temporal-reasoning", "knowledge-update",
            "abstention".
            None loads all types.
        variant: "s" for LongMemEval-S (~115K token histories),
                 "m" for LongMemEval-M (~1.5M token histories),
                 "oracle" for oracle-retrieval variant.

    Returns:
        List of conversation examples for evaluate_memory().
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub required. Install with: pip install context-bench[datasets]"
        )
    import json

    path = hf_hub_download(
        "xiaowu0162/longmemeval",
        f"longmemeval_{variant}",
        repo_type="dataset",
    )
    with open(path) as f:
        data = json.load(f)

    examples = []
    for i, item in enumerate(data):
        if n is not None and i >= n:
            break

        question_type = item.get("question_type", "")
        if question_types is not None:
            if not any(qt in question_type for qt in question_types):
                continue

        # Flatten all haystack sessions into one turn list in order
        turns: list[dict[str, str]] = []
        for session in item.get("haystack_sessions", []):
            # Each session is a list of {"role": ..., "content": ...} dicts
            if isinstance(session, list):
                for turn in session:
                    if turn.get("content", "").strip():
                        turns.append({
                            "role": turn.get("role", "user"),
                            "content": turn["content"].strip(),
                        })

        question = item.get("question", "").strip()
        answer = item.get("answer", "")
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        answer = str(answer).strip()

        if not turns or not question or not answer:
            continue

        examples.append({
            "id": item.get("question_id", i),
            "turns": turns,
            "qa_pairs": [{"question": question, "answer": answer, "type": question_type}],
            "dataset": f"longmemeval_{variant}",
        })

    return examples


# ---------------------------------------------------------------------------
# Train / test split utilities
# ---------------------------------------------------------------------------

import random as _random


def split_conversations(
    examples: list,
    train_frac: float = 0.8,
    seed: int = 42,
) -> tuple[list, list]:
    """Deterministically split conversation examples into (train, test).

    The list is shuffled with ``random.Random(seed)`` before splitting so the
    result is reproducible without requiring numpy.

    Args:
        examples:   Output of locomo() or longmemeval() (or any list of dicts
                    sharing the same schema).
        train_frac: Fraction of examples to place in the training split.
                    Defaults to 0.8 (80 % train, 20 % test).
        seed:       Integer seed for the internal RNG.  Change this value to
                    obtain a different but still deterministic split.

    Returns:
        ``(train_examples, test_examples)`` where *train_examples* is the
        search pool seen by the autoresearch loop and *test_examples* is the
        held-out set never used during search.
    """
    shuffled = list(examples)
    _random.Random(seed).shuffle(shuffled)
    split_idx = int(len(shuffled) * train_frac)
    return shuffled[:split_idx], shuffled[split_idx:]


def sample_search_pool(
    examples: list,
    n: int = 100,
    seed: int | None = None,
) -> list:
    """Randomly sample *n* examples from a pool for fast iteration during search.

    When *seed* is ``None`` the sample is different on every call — this is
    intentional so that each autoresearch iteration sees a fresh, diverse
    subset.  Pass an integer *seed* to make the sample reproducible.

    Args:
        examples: Pool of conversation examples (typically the train split
                  returned by :func:`split_conversations`).
        n:        Number of examples to draw.  If *n* >= ``len(examples)`` the
                  entire pool is returned in a (possibly shuffled) copy.
        seed:     RNG seed.  ``None`` means non-deterministic.

    Returns:
        A list of at most *n* sampled examples.
    """
    rng = _random.Random(seed)
    population = list(examples)
    k = min(n, len(population))
    return rng.sample(population, k)


__all__ = [
    "locomo",
    "longmemeval",
    "split_conversations",
    "sample_search_pool",
]
