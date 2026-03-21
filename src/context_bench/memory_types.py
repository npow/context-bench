"""Typed data structures for the memory benchmark framework.

Item types use a tagged union (dataclass per variant) so systems can
pattern-match on item type and access typed fields directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Item types (tagged union)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConversationTurn:
    """A single turn in a conversation."""
    content: str
    role: str                           # "user" | "assistant"
    timestamp: str | None = None
    speaker: str | None = None
    session_id: str | None = None

@dataclass(frozen=True)
class DocumentChunk:
    """A chunk of a document."""
    content: str
    document_id: str
    position: int
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class PlatformEvent:
    """An event from a platform (Slack, Linear, Git, etc.)."""
    content: str
    platform: str
    timestamp: str
    author: str | None = None
    channel: str | None = None
    event_type: str | None = None

@dataclass(frozen=True)
class Declaration:
    """An explicit user declaration (preference, fact, reminder)."""
    key: str
    value: str
    source_turn_id: str | None = None

Item = ConversationTurn | DocumentChunk | PlatformEvent | Declaration


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class IngestResult:
    """Result of ingesting items into a memory system."""
    num_items: int
    latency_ms: float
    details: dict[str, Any] = field(default_factory=dict)

@dataclass
class QueryResult:
    """Result of querying a memory system."""
    answer: str
    total_latency_ms: float
    context_tokens: int = 0
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Benchmark schema
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkQuery:
    """A question to ask a memory system after ingestion."""
    question: str
    answer: str
    query_type: str = ""
    evidence: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkExample:
    """A complete benchmark example: items to ingest + queries to ask."""
    id: str
    items: list[Item]
    queries: list[BenchmarkQuery]
    dataset: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class MemorySystem(Protocol):
    """A stateful memory system evaluated over benchmark examples.

    The runner calls reset() between examples, ingest() to load items,
    then query() for each benchmark query.
    """

    @property
    def name(self) -> str: ...

    def reset(self) -> None: ...

    def ingest(self, items: list[Item]) -> IngestResult: ...

    def query(self, question: str, budget: int | None = None) -> QueryResult: ...
