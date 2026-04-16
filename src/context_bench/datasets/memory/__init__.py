"""Dataset loaders for memory system benchmarks.

Each loader returns a list of BenchmarkExample instances compatible with
evaluate_memory() and the MemorySystem protocol.

Requires: pip install context-bench[datasets]
"""

from __future__ import annotations

import random as _random

from context_bench.datasets.memory.locomo import locomo
from context_bench.datasets.memory.longmemeval import longmemeval
from context_bench.datasets.memory.membench import membench
from context_bench.datasets.memory.convomem import convomem


def split_conversations(
    examples: list,
    train_frac: float = 0.8,
    seed: int = 42,
) -> tuple[list, list]:
    """Deterministically split conversation examples into (train, test).

    The list is shuffled with ``random.Random(seed)`` before splitting so the
    result is reproducible without requiring numpy.

    Args:
        examples:   Output of locomo() or longmemeval() (or any list of
                    BenchmarkExample sharing the same schema).
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
    "membench",
    "convomem",
    "split_conversations",
    "sample_search_pool",
]
