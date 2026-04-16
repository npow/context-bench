"""Tests for MemPalace system adapter and MemBench/ConvoMem dataset loaders.

No network required — tests cover system lifecycle, ingest, chunking,
hybrid ranking, dataset parsing, and typed-item handling.

MemPalaceSystem tests require chromadb (skipped in CI if not installed).
Dataset loader tests only check imports and category definitions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from context_bench.memory_types import (
    BenchmarkExample,
    BenchmarkQuery,
    ConversationTurn,
    Declaration,
    DocumentChunk,
    IngestResult,
    PlatformEvent,
    QueryResult,
)

# chromadb is not a core context-bench dep — skip system tests if missing
try:
    import chromadb
    _HAS_CHROMADB = True
except ImportError:
    _HAS_CHROMADB = False

_skip_no_chromadb = pytest.mark.skipif(
    not _HAS_CHROMADB, reason="chromadb not installed"
)


# ===========================================================================
# MemPalaceSystem — unit tests (no network, requires chromadb)
# ===========================================================================


@_skip_no_chromadb
class TestMemPalaceSystemLifecycle:
    def _system(self, **kwargs):
        from context_bench.systems.mempalace_system import MemPalaceSystem
        return MemPalaceSystem(base_url="http://localhost:9999", **kwargs)

    def test_name(self):
        s = self._system(top_k=10)
        assert s.name == "mempalace_k10"

    def test_name_custom_k(self):
        s = self._system(top_k=25)
        assert s.name == "mempalace_k25"

    def test_reset_creates_fresh_palace(self):
        s = self._system()
        s.reset()
        assert s._collection is not None
        assert s._collection.count() == 0
        assert s._tmpdir is not None

    def test_reset_clears_previous(self):
        s = self._system()
        s.reset()
        first_tmp = s._tmpdir
        s.reset()
        assert s._tmpdir != first_tmp
        assert s._collection.count() == 0

    def test_ingest_conversation_turns(self):
        s = self._system()
        s.reset()
        items = [
            ConversationTurn(content="I like pizza", role="user", timestamp="2026-01-01"),
            ConversationTurn(content="Good choice!", role="assistant"),
            ConversationTurn(content="Actually I prefer sushi now", role="user", timestamp="2026-03-01"),
        ]
        result = s.ingest(items)
        assert isinstance(result, IngestResult)
        assert result.num_items == 3
        assert result.latency_ms > 0
        assert s._collection.count() == 3

    def test_ingest_document_chunks(self):
        s = self._system()
        s.reset()
        items = [
            DocumentChunk(content="Chapter 1: The beginning", document_id="doc1", position=0, source="book.txt"),
            DocumentChunk(content="Chapter 2: The middle", document_id="doc1", position=1, source="book.txt"),
        ]
        result = s.ingest(items)
        assert s._collection.count() == 2

    def test_ingest_platform_events(self):
        s = self._system()
        s.reset()
        items = [
            PlatformEvent(content="Deployed v2.0", platform="github", timestamp="2026-01-15", author="alice"),
        ]
        result = s.ingest(items)
        assert s._collection.count() == 1

    def test_ingest_declarations(self):
        s = self._system()
        s.reset()
        items = [
            Declaration(key="favorite_food", value="sushi"),
        ]
        result = s.ingest(items)
        assert s._collection.count() == 1

    def test_ingest_skips_short_content(self):
        s = self._system()
        s.reset()
        items = [
            ConversationTurn(content="ok", role="user"),  # too short (<10 chars)
            ConversationTurn(content="This is a longer message that should be stored", role="user"),
        ]
        s.ingest(items)
        assert s._collection.count() == 1

    def test_ingest_mixed_types(self):
        s = self._system()
        s.reset()
        items = [
            ConversationTurn(content="I was born in 1990 in New York City", role="user"),
            DocumentChunk(content="User profile indicates birthplace", document_id="d1", position=0),
            Declaration(key="birth_year", value="1990"),
        ]
        s.ingest(items)
        assert s._collection.count() == 3

    def test_query_empty_palace_graceful(self):
        s = self._system()
        s.reset()
        result = s.query("anything")
        assert isinstance(result, QueryResult)
        assert "No memories" in result.answer

    def test_cleanup_on_del(self):
        import os
        s = self._system()
        s.reset()
        tmpdir = s._tmpdir
        assert os.path.isdir(tmpdir)
        del s
        # tmpdir may or may not exist depending on __del__ timing
        # just verify no crash


@_skip_no_chromadb
class TestMemPalaceChunking:
    def _chunk(self, text, max_chars=1500):
        from context_bench.systems.mempalace_system import MemPalaceSystem
        return MemPalaceSystem._chunk(text, max_chars)

    def test_short_text_no_split(self):
        chunks = self._chunk("Short text", 1500)
        assert len(chunks) == 1
        assert chunks[0] == "Short text"

    def test_long_text_split(self):
        text = "word " * 500  # ~2500 chars
        chunks = self._chunk(text, 1500)
        assert len(chunks) >= 2
        # All content preserved
        rejoined = " ".join(c.strip() for c in chunks)
        assert rejoined.count("word") == 500

    def test_split_at_paragraph(self):
        text = "A" * 700 + "\n\n" + "B" * 700 + "\n\n" + "C" * 700
        chunks = self._chunk(text, 1500)
        assert len(chunks) >= 2

    def test_empty_text(self):
        chunks = self._chunk("")
        assert chunks == []

    def test_exact_boundary(self):
        text = "x" * 1500
        chunks = self._chunk(text, 1500)
        assert len(chunks) == 1


@_skip_no_chromadb
class TestMemPalaceHybridRank:
    def _rank(self, query, docs, distances):
        from context_bench.systems.mempalace_system import MemPalaceSystem
        return MemPalaceSystem._hybrid_rank(query, docs, distances)

    def test_basic_ranking(self):
        docs = ["Python programming language", "Java coffee beans", "Python snake species"]
        dists = [0.3, 0.5, 0.4]
        ranked = self._rank("Python programming", docs, dists)
        assert ranked[0] == "Python programming language"

    def test_bm25_boost(self):
        """BM25 should boost exact keyword matches over semantic-only matches."""
        docs = [
            "The quick brown fox jumps over the lazy dog",  # no keyword match
            "Pizza restaurant review with excellent marinara sauce",  # keyword match
        ]
        dists = [0.2, 0.25]  # first is closer in vector space
        ranked = self._rank("pizza marinara", docs, dists)
        # BM25 should boost pizza doc despite slightly worse vector distance
        assert "Pizza" in ranked[0]

    def test_empty_docs(self):
        ranked = self._rank("query", [], [])
        assert ranked == []

    def test_single_doc(self):
        ranked = self._rank("test", ["only doc"], [0.5])
        assert ranked == ["only doc"]


@_skip_no_chromadb
class TestMemPalaceItemConversion:
    def _to_text(self, item):
        from context_bench.systems.mempalace_system import MemPalaceSystem
        return MemPalaceSystem._item_to_text(item)

    def test_conversation_turn_with_timestamp(self):
        t = ConversationTurn(content="Hello", role="user", timestamp="2026-01-01")
        text = self._to_text(t)
        assert "2026-01-01" in text
        assert "Hello" in text

    def test_conversation_turn_with_speaker(self):
        t = ConversationTurn(content="Hi there", role="user", speaker="Alice")
        text = self._to_text(t)
        assert "Alice" in text
        assert "Hi there" in text

    def test_document_chunk(self):
        d = DocumentChunk(content="Chapter 1", document_id="d1", position=0)
        assert self._to_text(d) == "Chapter 1"

    def test_platform_event(self):
        e = PlatformEvent(content="Deployed", platform="github", timestamp="2026-01-01")
        text = self._to_text(e)
        assert "github" in text
        assert "Deployed" in text

    def test_declaration(self):
        d = Declaration(key="food", value="sushi")
        text = self._to_text(d)
        assert "food" in text
        assert "sushi" in text


@_skip_no_chromadb
class TestMemPalaceWithRunner:
    """Integration test: MemPalace through evaluate_memory (no LLM)."""

    def test_runs_through_evaluate_memory(self):
        from context_bench.memory_runner import evaluate_memory
        from context_bench.evaluators.answer_quality import AnswerQuality

        # Create a system that returns canned answers (mock LLM)
        from context_bench.systems.mempalace_system import MemPalaceSystem
        s = MemPalaceSystem(base_url="http://localhost:9999")

        # Override _llm_answer to avoid network
        s._llm_answer = lambda q, ctx: "Alice"

        example = BenchmarkExample(
            id="test1",
            items=[
                ConversationTurn(content="My name is Alice and I was born in 1990", role="user"),
                ConversationTurn(content="Nice to meet you Alice!", role="assistant"),
            ],
            queries=[
                BenchmarkQuery(question="What is my name?", answer="Alice", query_type="single_hop"),
            ],
            dataset="test",
        )

        result = evaluate_memory([s], [example], [AnswerQuality()], progress=False)
        assert len(result.rows) == 1
        assert result.rows[0].scores["f1"] == 1.0
        assert result.rows[0].system == s.name


# ===========================================================================
# MemBench dataset loader — parsing tests (no download)
# ===========================================================================


class TestMemBenchLoader:
    def test_import(self):
        from context_bench.datasets.memory.membench import membench
        assert callable(membench)

    def test_categories_defined(self):
        from context_bench.datasets.memory.membench import _CATEGORIES
        assert "knowledge_update" in _CATEGORIES
        assert "highlevel" in _CATEGORIES
        assert len(_CATEGORIES) == 8

    def test_import_from_init(self):
        from context_bench.datasets.memory import membench
        assert callable(membench)


# ===========================================================================
# ConvoMem dataset loader — parsing tests (no download)
# ===========================================================================


class TestConvoMemLoader:
    def test_import(self):
        from context_bench.datasets.memory.convomem import convomem
        assert callable(convomem)

    def test_categories_defined(self):
        from context_bench.datasets.memory.convomem import _CATEGORIES
        assert "changing_evidence" in _CATEGORIES
        assert "abstention_evidence" in _CATEGORIES
        assert len(_CATEGORIES) == 6

    def test_import_from_init(self):
        from context_bench.datasets.memory import convomem
        assert callable(convomem)
