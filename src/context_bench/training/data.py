"""Convert LoCoMo benchmark examples into GRPO training format.

Each training example:
    prompt:       REPL system prompt + "Question: {question}"
    turns:        flat list of turn strings (for reward function)
    ground_truth: expected answer string
    conv_id:      conversation identifier
    query_type:   single_hop / multi_hop / temporal / open_domain / adversarial

Train/eval split: first 40 conversations → train, last 10 → eval.
"""

from __future__ import annotations

import textwrap
from typing import Any

# Import here to keep this module importable without sentence-transformers
_SYSTEM_PROMPT = textwrap.dedent("""
You are an agent that answers questions by managing memory programmatically.

You have access to these functions:

  memory_read(query, k=20) -> list[str]
    Retrieve k relevant memory entries for the given query string.

  memory_write(content: str, memory_type: str = "episodic") -> None
    Write new content to persistent memory.
    memory_type: "episodic" | "factual" | "procedural"

  consolidate() -> str
    Compress working context into a dense paragraph. Writes to memory.

  answer: dict  {"content": str, "ready": bool}

REQUIRED: always end with these two lines:
    answer["content"] = "YOUR SHORT ANSWER HERE"
    answer["ready"] = True

Example complete script:
    items = memory_read("Caroline support group visit date")
    answer["content"] = "7 May 2023"
    answer["ready"] = True

Write Python only. No text outside code. Last two lines MUST be the answer assignment.
""").strip()


def _turn_to_str(item: Any) -> str:
    """Convert a ConversationTurn or Item to a plain string."""
    if hasattr(item, "content"):
        parts = []
        if hasattr(item, "timestamp") and item.timestamp:
            parts.append(f"[{item.timestamp}]")
        if hasattr(item, "speaker") and item.speaker:
            parts.append(f"{item.speaker}:")
        parts.append(item.content)
        return " ".join(parts)
    return str(item)


def build_training_dataset(
    n_train: int | None = None,
    n_eval: int = 2,
    max_questions_per_conv: int | None = None,
    exclude_adversarial: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Load LoCoMo and return (train_examples, eval_examples).

    The public LoCoMo split has 10 conversations. Default: 8 train / 2 eval.

    Each example is a dict with keys:
        prompt, turns, ground_truth, conv_id, query_type
    """
    from context_bench.datasets.memory.locomo import locomo

    all_convs = locomo()  # load all available (10 in public split)
    total = len(all_convs)

    if n_train is None:
        n_train = max(1, total - n_eval)

    # Clamp to available data
    n_eval = min(n_eval, total - n_train)
    if n_train + n_eval > total:
        n_train = total - n_eval

    train_convs = all_convs[:n_train]
    eval_convs = all_convs[n_train: n_train + n_eval]

    print(
        f"[data] LoCoMo: {total} conversations → "
        f"{len(train_convs)} train, {len(eval_convs)} eval",
        flush=True,
    )

    def conv_to_examples(convs: list) -> list[dict]:
        out: list[dict] = []
        for conv in convs:
            turns = [_turn_to_str(item) for item in conv.items]
            questions = conv.queries
            if max_questions_per_conv:
                questions = questions[:max_questions_per_conv]
            for q in questions:
                if exclude_adversarial and q.query_type == "adversarial":
                    continue
                prompt = f"{_SYSTEM_PROMPT}\n\nQuestion: {q.question}"
                out.append({
                    "prompt": prompt,
                    "turns": turns,
                    "ground_truth": q.answer,
                    "conv_id": conv.id,
                    "query_type": q.query_type or "unknown",
                })
        return out

    return conv_to_examples(train_convs), conv_to_examples(eval_convs)


def to_hf_dataset(examples: list[dict]):
    """Convert list of examples to a HuggingFace Dataset."""
    from datasets import Dataset

    # HF datasets require all values to be serializable.
    # turns (list[str]) is fine; other fields are strings.
    return Dataset.from_list(examples)
