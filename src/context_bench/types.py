"""Protocol definitions for context-bench.

Users implement these protocols -- never subclass.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from context_bench.results import EvalRow


@runtime_checkable
class System(Protocol):
    """Anything that transforms context."""

    @property
    def name(self) -> str: ...

    def process(self, example: dict[str, Any]) -> dict[str, Any]: ...


@runtime_checkable
class Evaluator(Protocol):
    """Scores a (input, output) pair. Returns dict of metric_name -> float."""

    @property
    def name(self) -> str: ...

    def score(
        self, original: dict[str, Any], processed: dict[str, Any]
    ) -> dict[str, float]: ...


@runtime_checkable
class Metric(Protocol):
    """Aggregates per-example scores into summary stats."""

    @property
    def name(self) -> str: ...

    def compute(self, rows: list[EvalRow]) -> dict[str, float]: ...


@runtime_checkable
class MemorySystem(Protocol):
    """A stateful memory system evaluated over conversation histories.

    The runner calls reset() between conversations, ingest() to load the
    conversation history, then query() for each QA pair.  Implementations
    point their internal LLM calls at whatever relay/proxy they already use.
    """

    @property
    def name(self) -> str: ...

    def reset(self) -> None:
        """Clear all stored memory.  Called once before each conversation."""
        ...

    def ingest(self, turns: list[dict[str, Any]]) -> None:
        """Store a conversation history.

        Args:
            turns: List of ``{"role": "user"|"assistant", "content": str}``
                dicts in chronological order.
        """
        ...

    def query(self, question: str) -> str:
        """Answer a question using stored memory.

        Args:
            question: Natural-language question about the conversation.

        Returns:
            The system's answer string.
        """
        ...
