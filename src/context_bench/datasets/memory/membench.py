"""MemBench dataset loader returning typed BenchmarkExample instances.

MemBench (ACL 2025) tests memory across multi-turn conversations in 8 categories:
  highlevel, lowlevel, knowledge_update, comparative, conditional,
  noisy, aggregative, RecMultiSession.

Particularly relevant: knowledge_update tests facts that change over time,
which is where temporal-aware memory systems should shine.

Reference: https://aclanthology.org/2025.findings-acl.989/
Data: https://github.com/import-myself/Membench (HuggingFace mirror)

Requires: pip install context-bench[datasets]
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from context_bench.memory_types import BenchmarkExample, BenchmarkQuery, ConversationTurn


# HuggingFace raw file URLs for MemBench
_HF_BASE = "https://huggingface.co/datasets/MemBench/MemData/resolve/main/FirstAgent"

_CATEGORIES = [
    "highlevel",
    "lowlevel",
    "knowledge_update",
    "comparative",
    "conditional",
    "noisy",
    "aggregative",
    "RecMultiSession",
]

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "context-bench", "membench")


def _download(category: str) -> list[dict[str, Any]]:
    """Download a MemBench category JSON file, caching locally."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(_CACHE_DIR, f"{category}.json")

    if os.path.isfile(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    url = f"{_HF_BASE}/{category}.json"
    try:
        urllib.request.urlretrieve(url, cache_path)
        with open(cache_path) as f:
            return json.load(f)
    except Exception as e:
        # Try alternate HuggingFace datasets API
        try:
            import datasets
            ds = datasets.load_dataset("MemBench/MemData", data_dir="FirstAgent", split=category)
            items = [dict(row) for row in ds]
            with open(cache_path, "w") as f:
                json.dump(items, f)
            return items
        except Exception:
            raise RuntimeError(
                f"Could not download MemBench/{category}: {e}. "
                "Download manually from https://github.com/import-myself/Membench"
            )


def membench(
    n: int | None = None,
    categories: list[str] | None = None,
) -> list[BenchmarkExample]:
    """Load the MemBench multi-turn conversation memory benchmark.

    Args:
        n: Max number of examples per category to load.
        categories: Filter to specific categories. Options:
            "highlevel", "lowlevel", "knowledge_update", "comparative",
            "conditional", "noisy", "aggregative", "RecMultiSession".
            None loads all categories.

    Returns:
        List of BenchmarkExample instances for evaluate_memory().
    """
    cats = categories or _CATEGORIES
    examples: list[BenchmarkExample] = []

    for cat in cats:
        if cat not in _CATEGORIES:
            continue
        try:
            items = _download(cat)
        except Exception:
            continue

        for i, item in enumerate(items):
            if n is not None and i >= n:
                break

            # Parse conversation turns
            message_list = item.get("message_list", [[]])[0] if item.get("message_list") else []
            turns: list[ConversationTurn] = []

            for msg in message_list:
                role = "user" if msg.get("role") == "user" else "assistant"
                content = msg.get("content", "")
                timestamp = msg.get("time", None)
                if content.strip():
                    turns.append(ConversationTurn(
                        content=content,
                        role=role,
                        timestamp=str(timestamp) if timestamp else None,
                        speaker=msg.get("place", None),
                    ))

            if not turns:
                continue

            # Parse QA
            qa = item.get("QA", {})
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            ground_truth = qa.get("ground_truth", "")

            if not question:
                continue

            # Use ground_truth as the answer if available (it's the letter choice)
            # but the full answer text is more useful for F1 scoring
            gold = answer if answer else ground_truth

            queries = [BenchmarkQuery(
                question=question,
                answer=gold,
                query_type=cat,
                metadata={
                    "choices": qa.get("choices", {}),
                    "ground_truth_letter": ground_truth,
                    "target_step_id": qa.get("target_step_id", ""),
                },
            )]

            examples.append(BenchmarkExample(
                id=f"membench_{cat}_{i}",
                items=turns,
                queries=queries,
                dataset="membench",
                metadata={"category": cat},
            ))

    return examples
