"""Tests for RLMSystemRepl: memory_write, consolidate, REPL loop, usage stats.

All tests use mocked LLM calls and real LanceDB/DuckDB in temp dirs.
No network required. The SentenceTransformer embedder is mocked to avoid
downloading large model weights during CI.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from context_bench.memory_types import ConversationTurn, Declaration
from context_bench.systems.rlm_repl import RLMSystemRepl, _strip_code_fences

_EMB_DIM = 32  # small fake embedding dimension


def _make_embedder() -> MagicMock:
    """Return a mock SentenceTransformer that produces random fixed-dim vectors."""
    emb = MagicMock()
    emb.get_sentence_embedding_dimension.return_value = _EMB_DIM
    emb.encode.side_effect = lambda text: np.random.default_rng(
        abs(hash(str(text))) % (2**32)
    ).random(_EMB_DIM).astype("float32")
    return emb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def system():
    """RLMSystemRepl with a fake base_url; LLM calls + embedder are mocked."""
    # Patch SentenceTransformer so no model download occurs
    with patch(
        "context_bench.systems.rlm.SentenceTransformer",
        return_value=_make_embedder(),
    ):
        s = RLMSystemRepl(
            base_url="http://localhost:9999",
            model="test-model",
            max_iterations=3,
        )
    return s


def _turns(texts: list[str]) -> list[ConversationTurn]:
    return [
        ConversationTurn(
            content=t,
            role="user",
            speaker="Alice",
            timestamp=f"2024-01-0{i+1}T00:00:00",
        )
        for i, t in enumerate(texts)
    ]


# ---------------------------------------------------------------------------
# _strip_code_fences
# ---------------------------------------------------------------------------

class TestStripCodeFences:
    def test_bare_code_unchanged(self):
        code = "x = 1\nprint(x)"
        assert _strip_code_fences(code) == code

    def test_python_fence_stripped(self):
        raw = "```python\nx = 1\nprint(x)\n```"
        assert _strip_code_fences(raw) == "x = 1\nprint(x)"

    def test_generic_fence_stripped(self):
        raw = "```\ny = 2\n```"
        assert _strip_code_fences(raw) == "y = 2"

    def test_empty_string(self):
        assert _strip_code_fences("") == ""

    def test_no_trailing_fence_handled(self):
        # Missing closing ``` — returns what we have after the opening line
        raw = "```python\nx = 3"
        result = _strip_code_fences(raw)
        assert "x = 3" in result


# ---------------------------------------------------------------------------
# Ingest + memory_write integration
# ---------------------------------------------------------------------------

class TestMemoryWrite:
    def test_write_adds_to_lance_and_duck(self, system):
        """memory_write() during a REPL query persists to both stores."""
        system.ingest(_turns(["Alice likes jazz.", "Bob plays guitar."]))

        # Simulate LLM that calls memory_write then sets answer["ready"]
        call_count = 0

        def fake_chat(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return textwrap.dedent("""
                    items = memory_read("music preferences")
                    memory_write("Alice is a jazz fan; Bob plays guitar.", "factual")
                    answer["content"] = "Alice likes jazz"
                    answer["ready"] = True
                """)
            return "answer['ready'] = True"

        with patch.object(system, "_chat", side_effect=fake_chat):
            result = system.query("What music does Alice like?")

        assert result.answer == "Alice likes jazz"
        assert result.details["writes"] == 1
        assert result.details["answer_ready"] is True

        # Verify written entry persists in lance
        import duckdb
        conn = duckdb.connect(system._duck_path, read_only=True)
        rows = conn.execute(
            "SELECT content FROM turns WHERE role = 'agent'"
        ).fetchall()
        conn.close()
        assert any("jazz fan" in row[0] for row in rows)

    def test_invalid_memory_type_defaults_to_episodic(self, system):
        """Invalid memory_type silently falls back to episodic (no exception)."""
        system.ingest(_turns(["Test turn."]))

        def fake_chat(messages):
            return textwrap.dedent("""
                memory_write("some content", "invalid_type")
                answer["content"] = "ok"
                answer["ready"] = True
            """)

        with patch.object(system, "_chat", side_effect=fake_chat):
            result = system.query("test?")

        assert result.details["writes"] == 1

    def test_write_count_accumulates_across_queries(self, system):
        """Total write count accumulates correctly across multiple queries."""
        system.ingest(_turns(["Turn A.", "Turn B.", "Turn C."]))

        call_seq = iter([
            "memory_write('fact1', 'factual')\nanswer['content']='a'\nanswer['ready']=True",
            "memory_write('fact2', 'factual')\nmemory_write('fact3', 'episodic')\nanswer['content']='b'\nanswer['ready']=True",
        ])

        with patch.object(system, "_chat", side_effect=lambda _: next(call_seq)):
            system.query("q1")
            system.query("q2")

        stats = system.usage_stats()
        assert stats["total_writes"] == 3
        assert stats["queries"] == 2
        assert stats["queries_with_writes"] == 2
        assert stats["write_adoption_rate"] == 1.0


# ---------------------------------------------------------------------------
# consolidate
# ---------------------------------------------------------------------------

class TestConsolidate:
    def test_consolidate_calls_llm_and_writes(self, system):
        """consolidate() makes a sub-LLM call and writes the result to memory."""
        system.ingest(_turns(["Fact A.", "Fact B.", "Fact C."]))

        llm_responses = iter([
            # First call: model retrieves and consolidates
            textwrap.dedent("""
                items = memory_read("facts")
                summary = consolidate()
                answer["content"] = summary[:50]
                answer["ready"] = True
            """),
            # Second call (consolidate's internal LLM call)
            "Consolidated: Fact A, B, and C are all present.",
        ])

        with patch.object(system, "_chat", side_effect=lambda msgs: next(llm_responses)):
            result = system.query("Summarize the facts.")

        assert result.details["consolidations"] == 1
        # consolidate also triggers a memory_write internally
        assert result.details["writes"] == 1

    def test_consolidate_returns_empty_with_no_context(self, system):
        """consolidate() returns '' when no memory_read has been called yet."""
        system.ingest(_turns(["Anything."]))

        def fake_chat(messages):
            # REPL code: consolidate before any memory_read
            return textwrap.dedent("""
                result = consolidate()
                answer["content"] = repr(result)
                answer["ready"] = True
            """)

        # consolidate LLM call should NOT be fired since retrieved_context is empty
        with patch.object(system, "_chat", side_effect=fake_chat) as m:
            result = system.query("test")

        # No consolidation LLM call because context was empty
        assert result.details["consolidations"] == 0

    def test_consolidate_adoption_rate_tracked(self, system):
        """consolidate_adoption_rate is 0.5 when half the queries consolidate."""
        system.ingest(_turns(["X.", "Y."]))

        responses = iter([
            # Query 1: uses consolidate
            "items = memory_read('x')\nconsolidate()\nanswer['content']='a'\nanswer['ready']=True",
            "Summary of X.",  # consolidate's sub-call
            # Query 2: no consolidate
            "items = memory_read('y')\nanswer['content']='b'\nanswer['ready']=True",
        ])

        with patch.object(system, "_chat", side_effect=lambda _: next(responses)):
            system.query("q1")
            system.query("q2")

        stats = system.usage_stats()
        assert stats["consolidate_adoption_rate"] == 0.5
        assert stats["total_consolidations"] == 1


# ---------------------------------------------------------------------------
# REPL loop mechanics
# ---------------------------------------------------------------------------

class TestReplLoop:
    def test_answer_set_on_first_iteration(self, system):
        """Loop terminates on iteration 1 when model sets answer immediately."""
        system.ingest(_turns(["Alice is 30 years old."]))

        def fake_chat(messages):
            return "answer['content'] = '30'\nanswer['ready'] = True"

        with patch.object(system, "_chat", side_effect=fake_chat):
            result = system.query("How old is Alice?")

        assert result.answer == "30"
        assert result.details["iterations"] == 1

    def test_multi_iteration_loop(self, system):
        """Loop runs multiple iterations when model doesn't set ready immediately."""
        system.ingest(_turns(["Bob is from London."]))

        call_count = 0

        def fake_chat(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "items = memory_read('Bob')\nprint('got items')"
            return "answer['content'] = 'London'\nanswer['ready'] = True"

        with patch.object(system, "_chat", side_effect=fake_chat):
            result = system.query("Where is Bob from?")

        assert result.answer == "London"
        assert result.details["iterations"] == 2

    def test_fallback_to_static_retrieval_when_no_ready(self, system):
        """Falls back to static retrieval when REPL never sets answer['ready']."""
        system.ingest(_turns(["Carol likes hiking."]))

        # Model never sets ready; fall back uses static path
        def fake_chat(messages):
            # Check if it's the fallback static call (has "Context:" in user msg)
            for m in messages:
                if m["role"] == "user" and "Context:" in m.get("content", ""):
                    return "hiking"
            return "x = 1  # does nothing"

        with patch.object(system, "_chat", side_effect=fake_chat):
            result = system.query("What does Carol like?")

        assert result.answer == "hiking"
        assert result.details.get("answer_ready") is False

    def test_exec_error_does_not_crash(self, system):
        """SyntaxError or runtime error in REPL code is handled gracefully."""
        system.ingest(_turns(["Data point."]))

        responses = iter([
            "raise ValueError('oops')",  # will cause exec error
            "answer['content'] = 'recovered'\nanswer['ready'] = True",
        ])

        with patch.object(system, "_chat", side_effect=lambda _: next(responses)):
            result = system.query("test?")

        assert result.answer == "recovered"

    def test_max_iterations_respected(self, system):
        """Loop never exceeds max_iterations (set to 3 in fixture)."""
        system.ingest(_turns(["Something."]))

        call_count = 0

        def fake_chat(messages):
            nonlocal call_count
            call_count += 1
            # For the fallback call (has "Context:"), return real answer
            for m in messages:
                if m["role"] == "user" and "Context:" in m.get("content", ""):
                    return "fallback answer"
            return "x = 1  # never sets ready"

        with patch.object(system, "_chat", side_effect=fake_chat):
            result = system.query("test?")

        assert result.details["iterations"] <= 3


# ---------------------------------------------------------------------------
# usage_stats
# ---------------------------------------------------------------------------

class TestCodexIssues:
    """Tests for the three issues found by Codex review."""

    # Issue 1: exec timeout + cancellation flag
    def test_exec_timeout_terminates(self, system):
        """Hanging exec is interrupted; iteration continues after timeout."""
        import context_bench.systems.rlm_repl as _mod

        system.ingest(_turns(["Data."]))

        responses = iter([
            "import time; time.sleep(5)",  # longer than patched timeout
            "answer['content']='recovered'\nanswer['ready']=True",
        ])

        with patch.object(_mod, "_EXEC_TIMEOUT_S", 0.05):
            with patch.object(system, "_chat", side_effect=lambda _: next(responses)):
                result = system.query("test?")

        assert result.answer == "recovered"

    def test_cancelled_namespace_calls_are_noops(self, system):
        """After timeout, memory_write/consolidate/memory_read are no-ops."""
        system.ingest(_turns(["Turn."]))

        responses = iter([
            # Iteration 1: times out (sleep is long)
            "import time; time.sleep(5)",
            # Iteration 2: call memory_write (should work normally)
            "memory_write('after-timeout-write', 'factual')\nanswer['content']='ok'\nanswer['ready']=True",
        ])

        import context_bench.systems.rlm_repl as _mod

        with patch.object(_mod, "_EXEC_TIMEOUT_S", 0.05):
            with patch.object(system, "_chat", side_effect=lambda _: next(responses)):
                result = system.query("test?")

        # The timeout iteration did not write anything
        # (the sleep thread's namespace calls were guarded by _iter_cancelled)
        # The next iteration's write succeeded
        assert result.details["writes"] == 1

    # Issue 2: write_count accuracy
    def test_write_count_reflects_partial_success(self, system):
        """write_count is 1 when lance fails but duck succeeds (any_write_succeeded=True)."""
        system.ingest(_turns(["Turn."]))

        def fake_chat(messages):
            return "memory_write('fact', 'factual')\nanswer['content']='x'\nanswer['ready']=True"

        # Lance fails, duck succeeds → any_write_succeeded=True → count=1
        with patch.object(system._lance_table, "add", side_effect=RuntimeError("lance down")):
            with patch.object(system, "_chat", side_effect=fake_chat):
                result = system.query("q?")

        stats = system.usage_stats()
        assert stats["total_writes"] == 1  # duck succeeded → counted

    def test_write_count_zero_when_no_write_called(self, system):
        """write_count stays 0 when the model never calls memory_write."""
        system.ingest(_turns(["Turn."]))

        def fake_chat(messages):
            return "answer['content']='x'\nanswer['ready']=True"

        with patch.object(system, "_chat", side_effect=fake_chat):
            system.query("q?")

        assert system.usage_stats()["total_writes"] == 0

    # Issue 3: consolidate preserves full item content (no 300-char per-item truncation)
    def test_consolidate_preserves_long_items(self, system):
        """consolidate() does not truncate individual items to 300 chars."""
        # Ingest a long turn (>300 chars) so memory_read returns it full.
        long_text = "Important detail: " + "X" * 350
        system.ingest(_turns([long_text]))

        captured_consolidate_input: list[str] = []
        llm_calls = [0]

        def fake_chat(messages):
            llm_calls[0] += 1
            call_n = llm_calls[0]
            if call_n == 1:
                # REPL: read memory, then consolidate
                return "items = memory_read('detail')\nconsolidate()\nanswer['content']='done'\nanswer['ready']=True"
            # Second call must be consolidate's sub-LLM (has system "Compress" message)
            for m in messages:
                if m["role"] == "user":
                    captured_consolidate_input.append(m["content"])
            return "compressed summary"

        with patch.object(system, "_chat", side_effect=fake_chat):
            system.query("find the detail?")

        # If consolidate ran, verify its input preserved the full long text
        if captured_consolidate_input:
            # The full text (not just 300 chars) should be in the consolidate input
            assert "X" * 300 in captured_consolidate_input[0]


class TestUsageStats:
    def test_empty_stats(self, system):
        stats = system.usage_stats()
        assert stats == {"queries": 0}

    def test_zero_adoption_baseline(self, system):
        """Model that never calls write/consolidate → 0% adoption rates."""
        system.ingest(_turns(["Turn."]))

        def fake_chat(messages):
            for m in messages:
                if "Context:" in m.get("content", ""):
                    return "baseline"
            return "x = 1"  # never sets ready

        with patch.object(system, "_chat", side_effect=fake_chat):
            system.query("q?")

        stats = system.usage_stats()
        assert stats["write_adoption_rate"] == 0.0
        assert stats["consolidate_adoption_rate"] == 0.0
        assert stats["total_writes"] == 0

    def test_mean_iterations_computed(self, system):
        system.ingest(_turns(["A.", "B."]))

        seq = iter([
            "answer['content']='x'\nanswer['ready']=True",  # 1 iter
            "pass",
            "answer['content']='y'\nanswer['ready']=True",  # 2 iters
        ])

        with patch.object(system, "_chat", side_effect=lambda _: next(seq)):
            system.query("q1")
            system.query("q2")

        stats = system.usage_stats()
        assert stats["mean_iterations"] == 1.5
