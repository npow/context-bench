"""ConvoMem dataset loader returning typed BenchmarkExample instances.

ConvoMem (Salesforce): 75,336 QA pairs across 6 evidence categories testing
conversational memory systems on different types of recall.

Categories:
  user_evidence, assistant_facts_evidence, changing_evidence,
  abstention_evidence, preference_evidence, implicit_connection_evidence.

Particularly relevant: changing_evidence tests recall of facts that update,
abstention_evidence tests knowing when you DON'T know.

Reference: https://huggingface.co/datasets/Salesforce/ConvoMem
Requires: pip install context-bench[datasets]
"""

from __future__ import annotations

import json
import os
import random
import urllib.request
from typing import Any

from context_bench.memory_types import BenchmarkExample, BenchmarkQuery, ConversationTurn


_HF_BASE = (
    "https://huggingface.co/datasets/Salesforce/ConvoMem/resolve/main"
    "/core_benchmark/evidence_questions"
)

_CATEGORIES = {
    "user_evidence": "User Facts",
    "assistant_facts_evidence": "Assistant Facts",
    "changing_evidence": "Changing Facts",
    "abstention_evidence": "Abstention",
    "preference_evidence": "Preferences",
    "implicit_connection_evidence": "Implicit Connections",
}

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "context-bench", "convomem")


def _discover_files(category: str) -> list[str]:
    """Discover available evidence files for a category via HuggingFace API."""
    try:
        import datasets
        ds_files = datasets.load_dataset(
            "Salesforce/ConvoMem",
            data_dir=f"core_benchmark/evidence_questions/{category}",
            split="train",
        )
        return [f"{i}" for i in range(len(ds_files))]
    except Exception:
        # Fallback: try known subpath patterns
        return ["1_evidence"]


def _download_evidence(category: str, subpath: str) -> dict[str, Any] | None:
    """Download a single evidence JSON file."""
    os.makedirs(os.path.join(_CACHE_DIR, category), exist_ok=True)
    safe_name = subpath.replace("/", "_").replace("\\", "_")
    cache_path = os.path.join(_CACHE_DIR, category, f"{safe_name}.json")

    if os.path.isfile(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    url = f"{_HF_BASE}/{category}/{subpath}"
    try:
        urllib.request.urlretrieve(url, cache_path)
        with open(cache_path) as f:
            return json.load(f)
    except Exception:
        return None


def convomem(
    n: int | None = None,
    categories: list[str] | None = None,
    seed: int = 42,
) -> list[BenchmarkExample]:
    """Load the ConvoMem conversational memory benchmark.

    Since ConvoMem has 75K+ items across many files, this loader samples
    from each category. Use n to control sample size.

    Args:
        n: Max examples per category (default: all available cached).
        categories: Filter categories. None loads all.
        seed: Random seed for sampling.

    Returns:
        List of BenchmarkExample instances for evaluate_memory().
    """
    try:
        import datasets as _ds
    except ImportError:
        raise ImportError(
            "HuggingFace datasets required. Install with: pip install context-bench[datasets]"
        )

    cats = categories or list(_CATEGORIES.keys())
    examples: list[BenchmarkExample] = []
    rng = random.Random(seed)

    for cat in cats:
        if cat not in _CATEGORIES:
            continue

        try:
            ds = _ds.load_dataset(
                "Salesforce/ConvoMem",
                data_dir=f"core_benchmark/evidence_questions/{cat}",
                split="train",
                trust_remote_code=True,
            )
        except Exception:
            # Try loading individual files
            continue

        items = list(ds)
        if n is not None and len(items) > n:
            items = rng.sample(items, n)

        for i, item in enumerate(items):
            # Parse conversation
            conversations = item.get("conversations", [])
            turns: list[ConversationTurn] = []

            for msg in conversations:
                role = msg.get("from", "user")
                if role == "human":
                    role = "user"
                elif role == "gpt":
                    role = "assistant"
                content = msg.get("value", "")
                if content.strip():
                    turns.append(ConversationTurn(
                        content=content,
                        role=role,
                    ))

            if not turns:
                continue

            # Parse QA
            question = item.get("question", "")
            answer = item.get("answer", "")
            evidence_messages = item.get("evidence_messages", [])

            if not question:
                continue

            queries = [BenchmarkQuery(
                question=question,
                answer=answer,
                query_type=cat,
                evidence=[str(e) for e in evidence_messages] if evidence_messages else None,
            )]

            examples.append(BenchmarkExample(
                id=f"convomem_{cat}_{i}",
                items=turns,
                queries=queries,
                dataset="convomem",
                metadata={"category": cat},
            ))

    return examples
