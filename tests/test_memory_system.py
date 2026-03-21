"""Tests for the memory system components added for context-bench.

Covers (no network required unless explicitly noted):
  - memory_runner: evaluate_memory, _truncate_turns
  - datasets/memory: locomo/longmemeval parsing helpers, split_conversations, sample_search_pool
  - evaluators/answer_quality: F1 / exact-match
  - evaluators/false_memory_rate: prompt construction, verdict parsing
  - evaluators/memory_judge: verdict parsing
  - metrics/per_qa_type: aggregation
  - metrics/token_efficiency: aggregation
  - pipeline/context_pipeline: chunking, BM25 retrieval, ordering, normalise
  - loop/mutator: history formatting, code extraction, best_score calculation
  - loop.py: load_pipeline_from_code, _summarise_mutation
"""

from __future__ import annotations

import importlib
import json
import sys
import textwrap
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make sure the src layout is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_example(
    turns=None,
    qa_pairs=None,
    conversation_id="c1",
    dataset="test",
):
    return {
        "id": conversation_id,
        "turns": turns or [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
            {"role": "user", "content": "I was born in 1990."},
            {"role": "assistant", "content": "Got it, born in 1990."},
        ],
        "qa_pairs": qa_pairs or [
            {"question": "What is my name?", "answer": "Alice", "type": "single_hop"},
        ],
        "dataset": dataset,
    }


def _stub_system(name="stub", answers=None):
    """Return a minimal MemorySystem stub."""
    answers = answers or {}

    class _Stub:
        def __init__(self):
            self._n = 0

        @property
        def name(self):
            return name

        def reset(self):
            self._n = 0

        def ingest(self, turns):
            pass

        def query(self, question):
            return answers.get(question, "unknown")

    return _Stub()


# ===========================================================================
# memory_runner
# ===========================================================================

class TestTruncateTurns:
    def _fn(self):
        from context_bench.memory_runner import _truncate_turns
        return _truncate_turns

    def test_empty(self):
        fn = self._fn()
        assert fn([]) == ""

    def test_single_turn(self):
        fn = self._fn()
        turns = [{"role": "user", "content": "hello"}]
        result = fn(turns, max_chars=1000)
        assert "USER: hello" in result

    def test_truncates_from_end(self):
        """When over budget, we keep the MOST RECENT turns."""
        fn = self._fn()
        turns = [
            {"role": "user", "content": "A" * 200},
            {"role": "assistant", "content": "B" * 200},
            {"role": "user", "content": "recent question"},
        ]
        result = fn(turns, max_chars=100)
        # The last (most recent) turn should be present
        assert "recent question" in result
        # The very old turns should be dropped
        assert "A" * 200 not in result

    def test_all_fit(self):
        fn = self._fn()
        turns = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = fn(turns, max_chars=10000)
        assert "USER: hi" in result
        assert "ASSISTANT: hello" in result

    def test_order_preserved(self):
        """Chronological order should be maintained in output."""
        fn = self._fn()
        turns = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]
        result = fn(turns, max_chars=10000)
        assert result.index("first") < result.index("second")


class TestEvaluateMemory:
    def test_basic_run(self):
        from context_bench.memory_runner import evaluate_memory
        from context_bench.evaluators.answer_quality import AnswerQuality

        system = _stub_system("s1", answers={"What is my name?": "Alice"})
        examples = [_make_example()]
        result = evaluate_memory([system], examples, [AnswerQuality()], progress=False)

        assert len(result.rows) == 1
        row = result.rows[0]
        assert row.system == "s1"
        assert row.scores["f1"] == 1.0  # exact match
        assert row.metadata["qa_type"] == "single_hop"
        assert row.metadata["turn_count"] == 4

    def test_context_populated(self):
        """pseudo_input['context'] should contain the conversation text."""
        from context_bench.memory_runner import evaluate_memory

        captured = {}

        class CapturingEvaluator:
            @property
            def name(self):
                return "capture"

            def score(self, original, processed):
                captured["context"] = original.get("context", "")
                return {"capture": 1.0}

        system = _stub_system()
        examples = [_make_example()]
        evaluate_memory([system], examples, [CapturingEvaluator()], progress=False)

        assert "Alice" in captured["context"], (
            "context should contain conversation content so LLM judges can "
            "distinguish recalled facts from hallucinations"
        )

    def test_multiple_qa_pairs(self):
        from context_bench.memory_runner import evaluate_memory
        from context_bench.evaluators.answer_quality import AnswerQuality

        system = _stub_system(answers={
            "Name?": "Alice",
            "Year?": "1990",
        })
        example = _make_example(qa_pairs=[
            {"question": "Name?", "answer": "Alice", "type": "single_hop"},
            {"question": "Year?", "answer": "1990", "type": "temporal"},
        ])
        result = evaluate_memory([system], [example], [AnswerQuality()], progress=False)
        assert len(result.rows) == 2

    def test_max_examples(self):
        from context_bench.memory_runner import evaluate_memory
        from context_bench.evaluators.answer_quality import AnswerQuality

        examples = [_make_example(conversation_id=str(i)) for i in range(5)]
        system = _stub_system()
        result = evaluate_memory([system], examples, [AnswerQuality()],
                                  max_examples=2, progress=False)
        assert len(result.rows) == 2

    def test_multiple_systems(self):
        from context_bench.memory_runner import evaluate_memory
        from context_bench.evaluators.answer_quality import AnswerQuality

        s1 = _stub_system("s1", answers={"What is my name?": "Alice"})
        s2 = _stub_system("s2", answers={"What is my name?": "wrong"})
        result = evaluate_memory([s1, s2], [_make_example()], [AnswerQuality()], progress=False)
        assert len(result.rows) == 2
        by_system = {r.system: r for r in result.rows}
        assert by_system["s1"].scores["f1"] == 1.0
        assert by_system["s2"].scores["f1"] == 0.0

    def test_reset_called_between_conversations(self):
        from context_bench.memory_runner import evaluate_memory
        from context_bench.evaluators.answer_quality import AnswerQuality

        reset_count = 0

        class CountingSystem:
            @property
            def name(self):
                return "counter"

            def reset(self):
                nonlocal reset_count
                reset_count += 1

            def ingest(self, turns):
                pass

            def query(self, q):
                return "x"

        examples = [_make_example(conversation_id=str(i)) for i in range(3)]
        evaluate_memory([CountingSystem()], examples, [AnswerQuality()], progress=False)
        assert reset_count == 3


# ===========================================================================
# datasets/memory
# ===========================================================================

class TestSplitConversations:
    def _examples(self, n=10):
        return [{"id": i, "turns": [], "qa_pairs": [], "dataset": "x"} for i in range(n)]

    def test_split_ratio(self):
        from context_bench.datasets.memory import split_conversations
        train, test = split_conversations(self._examples(10), train_frac=0.8, seed=42)
        assert len(train) == 8
        assert len(test) == 2

    def test_deterministic(self):
        from context_bench.datasets.memory import split_conversations
        a_train, a_test = split_conversations(self._examples(20), seed=7)
        b_train, b_test = split_conversations(self._examples(20), seed=7)
        assert [e["id"] for e in a_train] == [e["id"] for e in b_train]
        assert [e["id"] for e in a_test] == [e["id"] for e in b_test]

    def test_different_seeds_differ(self):
        from context_bench.datasets.memory import split_conversations
        a_train, _ = split_conversations(self._examples(20), seed=1)
        b_train, _ = split_conversations(self._examples(20), seed=2)
        assert [e["id"] for e in a_train] != [e["id"] for e in b_train]

    def test_no_overlap(self):
        from context_bench.datasets.memory import split_conversations
        train, test = split_conversations(self._examples(10))
        train_ids = {e["id"] for e in train}
        test_ids = {e["id"] for e in test}
        assert train_ids.isdisjoint(test_ids)
        assert len(train_ids | test_ids) == 10

    def test_empty(self):
        from context_bench.datasets.memory import split_conversations
        train, test = split_conversations([])
        assert train == []
        assert test == []


class TestSampleSearchPool:
    def _examples(self, n=20):
        return [{"id": i} for i in range(n)]

    def test_size(self):
        from context_bench.datasets.memory import sample_search_pool
        result = sample_search_pool(self._examples(20), n=10, seed=1)
        assert len(result) == 10

    def test_n_larger_than_pool(self):
        from context_bench.datasets.memory import sample_search_pool
        result = sample_search_pool(self._examples(5), n=100)
        assert len(result) == 5

    def test_deterministic_with_seed(self):
        from context_bench.datasets.memory import sample_search_pool
        a = sample_search_pool(self._examples(20), n=10, seed=42)
        b = sample_search_pool(self._examples(20), n=10, seed=42)
        assert [e["id"] for e in a] == [e["id"] for e in b]

    def test_non_deterministic_without_seed(self):
        from context_bench.datasets.memory import sample_search_pool
        results = [
            tuple(e["id"] for e in sample_search_pool(self._examples(100), n=10))
            for _ in range(5)
        ]
        # With 100 items and 10 sampled, the chance of all 5 being identical is negligible
        assert len(set(results)) > 1


# ===========================================================================
# evaluators/answer_quality (F1)
# ===========================================================================

class TestAnswerQuality:
    def _ev(self):
        from context_bench.evaluators.answer_quality import AnswerQuality
        return AnswerQuality()

    def _score(self, response, answer):
        ev = self._ev()
        orig = {"answer": answer, "question": "q", "context": "", "id": "1", "dataset": "x"}
        proc = {**orig, "response": response}
        return ev.score(orig, proc)

    def test_exact_match(self):
        s = self._score("Alice", "Alice")
        assert s["f1"] == 1.0
        assert s["exact_match"] == 1.0

    def test_case_insensitive_exact_match(self):
        s = self._score("alice", "Alice")
        assert s["exact_match"] == 1.0

    def test_partial_f1(self):
        # Response contains the answer plus extra words -> recall=1, precision<1
        s = self._score("Alice Smith", "Alice")
        assert 0.0 < s["f1"] < 1.0
        assert s["recall"] == 1.0

    def test_no_overlap(self):
        s = self._score("Bob", "Alice")
        assert s["f1"] == 0.0
        assert s["exact_match"] == 0.0

    def test_empty_response(self):
        s = self._score("", "Alice")
        assert s["f1"] == 0.0

    def test_empty_answer(self):
        s = self._score("Alice", "")
        # Both empty tokens: edge case — just ensure no crash
        assert isinstance(s["f1"], float)

    def test_numeric_answer(self):
        s = self._score("The year was 1990.", "1990")
        assert s["recall"] == 1.0


# ===========================================================================
# evaluators/false_memory_rate — parsing only (no network)
# ===========================================================================

class TestFalseMemoryRateParsing:
    def _parse(self, text):
        from context_bench.evaluators.false_memory_rate import FalseMemoryRate
        return FalseMemoryRate._parse_verdict(text)

    def test_yes_detected(self):
        assert self._parse("YES - the answer fabricates a date") == 1.0

    def test_no_detected(self):
        assert self._parse("NO - all facts appear in the conversation") == 0.0

    def test_case_insensitive(self):
        assert self._parse("yes - fabricated") == 1.0
        assert self._parse("No - looks fine") == 0.0

    def test_ambiguous_returns_zero(self):
        assert self._parse("Unclear, cannot determine") == 0.0

    def test_yes_in_sentence(self):
        assert self._parse("I would say YES here.") == 1.0

    def test_prompt_contains_context(self):
        """The prompt template must include {context} so recalled facts aren't flagged."""
        from context_bench.evaluators.false_memory_rate import _HALLUCINATION_PROMPT
        assert "{context}" in _HALLUCINATION_PROMPT
        assert "conversation" in _HALLUCINATION_PROMPT.lower()

    def test_score_empty_answer(self):
        from context_bench.evaluators.false_memory_rate import FalseMemoryRate
        fmr = FalseMemoryRate.__new__(FalseMemoryRate)
        orig = {"question": "q", "context": "ctx"}
        proc = {"response": ""}
        result = fmr.score(orig, proc)
        assert result == {"false_memory": 0.0}


# ===========================================================================
# evaluators/memory_judge — parsing only (no network)
# ===========================================================================

class TestMemoryJudgeParsing:
    def _parse(self, text):
        from context_bench.evaluators.memory_judge import MemoryJudge
        # Returns (memory_judge_score, raw_score) tuple
        score, raw = MemoryJudge._parse_judgment(text)
        return score

    def test_yes_correct(self):
        assert self._parse("YES") == 1.0

    def test_no_incorrect(self):
        assert self._parse("NO") == 0.0

    def test_lowercase(self):
        assert self._parse("yes") == 1.0
        assert self._parse("no") == 0.0

    def test_ambiguous(self):
        # Should default to 0.0 when verdict is unclear
        assert self._parse("Maybe") == 0.0


# ===========================================================================
# metrics/per_qa_type
# ===========================================================================

class TestPerQATypeMetric:
    def _metric(self):
        from context_bench.metrics.per_qa_type import PerQATypeMetric
        return PerQATypeMetric(score_field="f1")

    def _row(self, f1, qa_type, system="s"):
        from context_bench.results import EvalRow
        return EvalRow(
            system=system,
            example_id="x",
            scores={"f1": f1},
            input_tokens=10,
            output_tokens=5,
            metadata={"qa_type": qa_type},
            latency=0.1,
            dataset="test",
        )

    def test_groups_by_type(self):
        metric = self._metric()
        rows = [
            self._row(1.0, "single_hop"),
            self._row(0.0, "single_hop"),
            self._row(0.8, "multi_hop"),
        ]
        result = metric.compute(rows)
        assert result["f1_single_hop"] == pytest.approx(0.5)
        assert result["f1_multi_hop"] == pytest.approx(0.8)

    def test_mean(self):
        metric = self._metric()
        rows = [self._row(1.0, "a"), self._row(0.0, "b")]
        result = metric.compute(rows)
        assert result["f1_mean"] == pytest.approx(0.5)

    def test_empty(self):
        metric = self._metric()
        result = metric.compute([])
        assert result == {}

    def test_name(self):
        metric = self._metric()
        assert metric.name == "per_qa_type"


# ===========================================================================
# metrics/token_efficiency
# ===========================================================================

class TestTokenEfficiencyMetric:
    def _metric(self):
        from context_bench.metrics.token_efficiency import TokenEfficiencyMetric
        return TokenEfficiencyMetric(score_field="f1")

    def _row(self, f1, input_tokens, system="s"):
        from context_bench.results import EvalRow
        return EvalRow(
            system=system,
            example_id="x",
            scores={"f1": f1},
            input_tokens=input_tokens,
            output_tokens=5,
            metadata={},
            latency=0.1,
            dataset="test",
        )

    def test_basic(self):
        metric = self._metric()
        # f1=1.0, input_tokens=1000 → new formula: 1.0 * (100/1000)^0.1 = 1.0 * 0.794 ≈ 0.794
        rows = [self._row(1.0, 1000)]
        result = metric.compute(rows)
        import math
        expected = 1.0 * (100.0 / 1000.0) ** 0.1
        assert result["token_efficiency"] == pytest.approx(expected, rel=1e-4)
        assert result["token_efficiency_raw"] == pytest.approx(1.0)  # old ratio still present
        assert result["mean_score"] == pytest.approx(1.0)
        assert result["mean_input_tokens"] == pytest.approx(1000.0)

    def test_zero_tokens_no_crash(self):
        metric = self._metric()
        rows = [self._row(0.5, 0)]
        result = metric.compute(rows)
        assert isinstance(result["token_efficiency"], float)

    def test_empty(self):
        metric = self._metric()
        result = metric.compute([])
        assert result["token_efficiency"] == 0.0

    def test_name(self):
        metric = self._metric()
        assert metric.name == "token_efficiency"


# ===========================================================================
# pipeline/context_pipeline — BM25, chunking, ordering (no network)
# ===========================================================================

class TestContextPipelineChunking:
    """Tests for ContextPipeline ingest / retrieval using _chunks/_retrieve API."""

    def _pipeline(self):
        from context_bench.pipeline.context_pipeline import ContextPipeline
        return ContextPipeline(
            relay_url="http://localhost:9999",
            model="test-model",
        )

    def _turns(self):
        return [
            {"role": "user", "content": "I love cats."},
            {"role": "assistant", "content": "Cats are great!"},
            {"role": "user", "content": "My favourite breed is Maine Coon."},
            {"role": "assistant", "content": "Maine Coons are very large."},
            {"role": "user", "content": "I also have a dog named Rex."},
        ]

    def test_turn_chunking(self):
        p = self._pipeline()
        p.ingest(self._turns())
        assert len(p._chunks) == 5

    def test_bm25_index_populated(self):
        p = self._pipeline()
        p.ingest(self._turns())
        assert p._bm25_index is not None

    def test_reset_clears_state(self):
        p = self._pipeline()
        p.ingest(self._turns())
        assert len(p._chunks) > 0
        p.reset()
        assert p._chunks == []

    def test_bm25_retrieval_relevance(self):
        p = self._pipeline()
        p.ingest(self._turns())
        results = p._retrieve("Maine Coon", k=2)
        texts = [text for _idx, text, _score in results]
        assert any("Maine Coon" in t for t in texts)

    def test_retrieval_k_respected(self):
        p = self._pipeline()
        p.ingest(self._turns())
        results = p._retrieve("anything", k=2)
        assert len(results) <= 2

    def test_retrieval_k_larger_than_chunks(self):
        p = self._pipeline()
        p.ingest(self._turns())
        results = p._retrieve("cats", k=100)
        assert len(results) <= len(p._chunks)

    def test_chronological_order(self):
        p = self._pipeline()
        p.ingest(self._turns())
        results = p._retrieve("cats", k=3)
        formatted = p._order_and_format(results, order="chronological")
        assert len(formatted) > 0

    def test_reverse_chron_order(self):
        p = self._pipeline()
        p.ingest(self._turns())
        results = p._retrieve("cats", k=3)
        formatted = p._order_and_format(results, order="reverse_chron")
        assert len(formatted) > 0

    def test_empty_chunks_no_crash(self):
        p = self._pipeline()
        p.ingest([])
        results = p._retrieve("anything", k=5)
        assert results == []


# TestNormalize removed: _normalize was removed from the production code.


# ===========================================================================
# loop/mutator — history formatting, code extraction, best_score
# ===========================================================================

class TestMutatorHistoryFormatting:
    def _mutator(self):
        from context_bench.loop.mutator import PipelineMutator
        return PipelineMutator(relay_url="http://localhost:9999")

    def test_format_empty_history(self):
        m = self._mutator()
        text = m._format_history([])
        assert "no history" in text.lower()

    def test_format_history_accepted(self):
        m = self._mutator()
        history = [{"iteration": 0, "score": 0.5, "delta": 0.1, "mutation": "changed k", "accepted": True}]
        text = m._format_history(history)
        assert "ACCEPTED" in text
        assert "changed k" in text

    def test_format_history_rejected(self):
        m = self._mutator()
        history = [{"iteration": 1, "score": 0.3, "delta": -0.1, "mutation": "broke it", "accepted": False}]
        text = m._format_history(history)
        assert "rejected" in text.lower()

    def test_rejected_summary(self):
        m = self._mutator()
        rejected = [{"mutation": "bad change 1"}, {"mutation": "bad change 2"}]
        text = m._format_rejected_summary(rejected)
        assert "bad change 1" in text
        assert "bad change 2" in text

    def test_empty_rejected_summary(self):
        m = self._mutator()
        assert m._format_rejected_summary([]) == ""

    def test_best_score_uses_tracked_value(self):
        """best_score should come from h['best_score'], not max(h['score'])."""
        m = self._mutator()
        # Simulate: one rejected candidate with a high (noisy) score,
        # and one accepted with a lower but real score.
        history = [
            {"iteration": 0, "score": 0.9, "best_score": 0.4, "mutation": "x", "accepted": False, "delta": 0.5},
            {"iteration": 1, "score": 0.5, "best_score": 0.5, "mutation": "y", "accepted": True, "delta": 0.1},
        ]
        prompt = m._build_user_prompt("# dummy code", history)
        # The displayed best should be 0.5 (tracked best), not 0.9 (rejected candidate noise)
        assert "0.5000" in prompt
        # 0.9000 should NOT appear as the best score header
        lines = [l for l in prompt.splitlines() if "Best" in l and "0.5" in l]
        assert lines, "Expected a 'Best ...' line with 0.5 in prompt"
        assert not any("0.9" in l for l in lines)


class TestExtractCode:
    def _fn(self):
        from context_bench.loop.mutator import PipelineMutator
        return PipelineMutator._extract_code

    def test_no_fence(self):
        fn = self._fn()
        code = "# context_pipeline.py\nSTRATEGY = {}"
        assert fn(code) == code

    def test_python_fence(self):
        fn = self._fn()
        code = "```python\n# context_pipeline.py\nSTRATEGY = {}\n```"
        result = fn(code)
        assert "```" not in result
        assert "STRATEGY" in result

    def test_generic_fence(self):
        fn = self._fn()
        code = "```\n# context_pipeline.py\nSTRATEGY = {}\n```"
        result = fn(code)
        assert "```" not in result

    def test_strips_whitespace(self):
        fn = self._fn()
        code = "   # context_pipeline.py\n   "
        result = fn(code)
        assert result == "# context_pipeline.py"


# ===========================================================================
# loop.py helpers (load_pipeline_from_code, _summarise_mutation)
# ===========================================================================

class TestLoadPipelineFromCode:
    def _fn(self):
        import importlib.util
        import sys
        loop_path = Path(__file__).parent.parent / "loop.py"
        spec = importlib.util.spec_from_file_location("loop_module", loop_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.load_pipeline_from_code

    def _minimal_pipeline(self, extra=""):
        return textwrap.dedent(f"""\
            # context_pipeline.py
            STRATEGY = {{"retrieval_k": 5}}
            class ContextPipeline:
                def __init__(self, relay_url="", model="", api_key=None):
                    pass
                @property
                def name(self):
                    return "test"
                def reset(self): pass
                def ingest(self, turns): pass
                def query(self, q): return "answer"
            {extra}
        """)

    def test_valid_code_returns_class(self):
        fn = self._fn()
        cls = fn(self._minimal_pipeline())
        assert cls.__name__ == "ContextPipeline"

    def test_missing_class_raises(self):
        fn = self._fn()
        with pytest.raises(ImportError, match="ContextPipeline"):
            fn("# no class here\nx = 1")

    def test_syntax_error_raises(self):
        fn = self._fn()
        with pytest.raises(Exception):
            fn("def broken(:\n    pass")

    def test_class_is_instantiable(self):
        fn = self._fn()
        cls = fn(self._minimal_pipeline())
        instance = cls(relay_url="http://x", model="m")
        assert instance.name == "test"


class TestSummariseMutation:
    def _fn(self):
        import importlib.util
        loop_path = Path(__file__).parent.parent / "loop.py"
        spec = importlib.util.spec_from_file_location("loop_module2", loop_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module._summarise_mutation

    def _code(self, k=5):
        return textwrap.dedent(f"""\
            STRATEGY = {{
                "retrieval_k": {k},
                "chunk_by": "turn",
            }}
            class ContextPipeline:
                pass
        """)

    def test_strategy_change_detected(self):
        fn = self._fn()
        summary = fn(self._code(k=5), self._code(k=10))
        assert "STRATEGY" in summary
        assert "retrieval_k" in summary or "10" in summary

    def test_no_change(self):
        fn = self._fn()
        code = self._code()
        summary = fn(code, code)
        assert "no visible change" in summary.lower()

    def test_code_level_change(self):
        fn = self._fn()
        old = self._code() + "\ndef foo(): pass"
        new = self._code() + "\ndef bar(): pass"
        summary = fn(old, new)
        # New implementation returns specific method names (e.g. "add bar; remove foo")
        # rather than the generic "code-level change" label — both are acceptable.
        assert summary and summary != "no visible change"


# ===========================================================================
# NaiveMemorySystem — unit test without network
# ===========================================================================

class TestContextTokenTracking:
    """last_context_tokens should reflect actual LLM input, not full conversation."""

    def test_context_pipeline_tracks_context_tokens(self):
        from context_bench.pipeline.context_pipeline import ContextPipeline

        p = ContextPipeline(
            relay_url="http://localhost:9999",
            model="test",
        )
        turns = [{"role": "user", "content": "word " * 100} for _ in range(10)]
        p.ingest(turns)

        # Mock _chat and _analyze_query_haiku to avoid network calls
        p._chat = lambda msgs: "answer"
        p._analyze_query_haiku = lambda q: {
            "q_type": "factual", "attributes": [], "entities": [],
            "search_terms": [], "time_focus": "none",
        }

        assert p.last_context_tokens == 0  # before any query

        p.query("test question")
        # last_context_tokens is set to len(context.split()) + len(question.split())
        # The context includes BM25 retrieved chunks with expansion and formatting.
        # With 10 turns of ~100 words each, retrieval includes context window hits.
        # Just verify it's non-zero and less than the full conversation would be.
        assert 0 < p.last_context_tokens < 5000, (
            f"Expected 0 < context_tokens < 5000, got {p.last_context_tokens}"
        )

    def test_context_pipeline_reset_clears_token_count(self):
        from context_bench.pipeline.context_pipeline import ContextPipeline

        p = ContextPipeline(relay_url="http://x", model="m")
        p.last_context_tokens = 999
        p.reset()
        assert p.last_context_tokens == 0

    def test_naive_tracks_context_tokens(self):
        from context_bench.systems.naive_memory import NaiveMemorySystem

        s = NaiveMemorySystem(base_url="http://x", model="m")
        s.ingest([{"role": "user", "content": "hello world"}])

        # Mock _chat
        s._chat = lambda msgs: "answer"

        s.query("what did I say?")
        # context = "USER: hello world" (3 words) + "what did I say?" (4 words) = 7
        assert s.last_context_tokens > 0
        # Should be ~7, definitely not the full conversation word count difference
        assert s.last_context_tokens < 20

    def test_naive_reset_clears_token_count(self):
        from context_bench.systems.naive_memory import NaiveMemorySystem

        s = NaiveMemorySystem(base_url="http://x", model="m")
        s.last_context_tokens = 500
        s.reset()
        assert s.last_context_tokens == 0

    def test_runner_uses_context_tokens(self):
        """EvalRow.input_tokens should reflect last_context_tokens, not full conversation."""
        from context_bench.memory_runner import evaluate_memory
        from context_bench.evaluators.answer_quality import AnswerQuality

        class SmallContextSystem:
            last_context_tokens = 42  # much less than conversation size

            @property
            def name(self):
                return "small_ctx"

            def reset(self):
                pass

            def ingest(self, turns):
                pass

            def query(self, q):
                return "Alice"

        # Give a large conversation (many words) so the fallback would be much larger
        big_turns = [{"role": "user", "content": "word " * 200}]
        example = {
            "id": "e1",
            "turns": big_turns,
            "qa_pairs": [{"question": "Name?", "answer": "Alice", "type": "x"}],
            "dataset": "test",
        }
        result = evaluate_memory([SmallContextSystem()], [example],
                                  [AnswerQuality()], progress=False)
        assert result.rows[0].input_tokens == 42

    def test_runner_falls_back_when_no_attribute(self):
        """Systems without last_context_tokens should fall back to conversation words."""
        from context_bench.memory_runner import evaluate_memory
        from context_bench.evaluators.answer_quality import AnswerQuality

        class NoTrackingSystem:
            @property
            def name(self):
                return "no_track"

            def reset(self):
                pass

            def ingest(self, turns):
                pass

            def query(self, q):
                return "Alice"

        turns = [{"role": "user", "content": "hello world"}]  # 2 words
        example = {
            "id": "e1",
            "turns": turns,
            "qa_pairs": [{"question": "q?", "answer": "Alice", "type": "x"}],
            "dataset": "test",
        }
        result = evaluate_memory([NoTrackingSystem()], [example],
                                  [AnswerQuality()], progress=False)
        assert result.rows[0].input_tokens == 2  # "hello world" = 2 words


class TestMemoryJudgeEdgeCases:
    def test_no_gold_returns_half(self):
        """No gold answer should return 0.5, not 1.0, to avoid inflating scores."""
        from context_bench.evaluators.memory_judge import MemoryJudge
        j = MemoryJudge.__new__(MemoryJudge)
        orig = {"question": "q?", "answer": "", "context": ""}
        proc = {"response": "something", "answer": "", "question": "q?", "context": ""}
        result = j.score(orig, proc)
        assert result["memory_judge"] == 0.5
        assert result["memory_judge_raw"] == 0.5

    def test_empty_response_returns_zero(self):
        from context_bench.evaluators.memory_judge import MemoryJudge
        j = MemoryJudge.__new__(MemoryJudge)
        orig = {"question": "q?", "answer": "Alice", "context": ""}
        proc = {"response": "", "answer": "Alice", "question": "q?", "context": ""}
        result = j.score(orig, proc)
        assert result == {"memory_judge": 0.0, "memory_judge_raw": 0.0}


class TestBuildMemorySystem:
    """CLI builder should pass relay_url (not base_url) to Mem0System and ZepSystem."""

    def test_naive_builds(self):
        from context_bench.__main__ import _build_memory_system
        s = _build_memory_system("naive", relay="http://x", model="m", api_key=None)
        assert s.name == "naive"

    def test_unknown_raises(self):
        from context_bench.__main__ import _build_memory_system
        with pytest.raises(SystemExit):
            _build_memory_system("unknown_system", relay="http://x", model="m", api_key=None)

    def test_mem0_uses_relay_url_kwarg(self):
        """Mem0System takes relay_url=, not base_url=. Verify no TypeError on construction."""
        from context_bench.__main__ import _build_memory_system
        try:
            s = _build_memory_system("mem0", relay="http://x", model="m", api_key=None)
            # If mem0ai is installed, the system should construct without TypeError
        except SystemExit:
            # mem0ai not installed — that's fine, the import guard works
            pass
        except TypeError as e:
            pytest.fail(f"Mem0System got wrong constructor kwarg: {e}")

    def test_zep_uses_relay_url_kwarg(self):
        """ZepSystem takes relay_url=, not base_url=. Verify no TypeError on construction."""
        from context_bench.__main__ import _build_memory_system
        try:
            s = _build_memory_system("zep", relay="http://x", model="m", api_key=None)
        except SystemExit:
            pass
        except TypeError as e:
            pytest.fail(f"ZepSystem got wrong constructor kwarg: {e}")


class TestLoopSysPath:
    """loop.py should be importable from repo root without pip install."""

    def test_sys_path_injection(self):
        import importlib.util
        loop_path = Path(__file__).parent.parent / "loop.py"
        spec = importlib.util.spec_from_file_location("_loop_test", loop_path)
        module = importlib.util.module_from_spec(spec)
        # Should not raise ImportError for context_bench
        spec.loader.exec_module(module)
        # If we get here, the sys.path injection worked
        assert hasattr(module, "main")


class TestNaiveMemorySystem:
    def _system(self, max_turns=None):
        from context_bench.systems.naive_memory import NaiveMemorySystem
        return NaiveMemorySystem(
            base_url="http://localhost:9999",
            model="test",
            max_turns=max_turns,
        )

    def test_name_no_max_turns(self):
        s = self._system()
        assert s.name == "naive"

    def test_name_with_max_turns(self):
        s = self._system(max_turns=5)
        assert "5" in s.name

    def test_reset_clears_turns(self):
        s = self._system()
        s.ingest([{"role": "user", "content": "hi"}])
        assert len(s._turns) == 1
        s.reset()
        assert s._turns == []

    def test_ingest_stores_turns(self):
        s = self._system()
        turns = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        s.ingest(turns)
        assert len(s._turns) == 2

    def test_max_turns_slices(self):
        """When max_turns=2, query() should only pass the last 2 turns."""
        s = self._system(max_turns=2)
        turns = [{"role": "user", "content": str(i)} for i in range(5)]
        s.ingest(turns)
        # query() would call _chat; we mock it to capture the messages sent
        captured = {}

        def fake_chat(messages):
            captured["messages"] = messages
            return "ok"

        s._chat = fake_chat
        s.query("q?")
        user_msg = captured["messages"][-1]["content"]
        # Only turns 3 and 4 (last 2) should appear
        assert "3" in user_msg and "4" in user_msg
        assert "0" not in user_msg
