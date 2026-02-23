"""Tests for infrastructure features: caching, HTML reporter, multi-turn, API usage."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from context_bench.results import EvalResult, EvalRow


# ===========================================================================
# Result caching
# ===========================================================================


class TestResultCache:
    def test_put_and_get(self, tmp_path):
        from context_bench.cache import ResultCache
        cache = ResultCache(
            cache_dir=tmp_path,
            systems=["sys_a"],
            datasets=["ds_1"],
            evaluators=["eval_1"],
        )
        row = EvalRow("sys_a", "ex_0", {"f1": 0.9}, 100, 50, latency=0.1, dataset="ds_1")
        cache.put(row)
        hit = cache.get("sys_a", "ds_1", "ex_0")
        assert hit is not None
        assert hit.scores["f1"] == 0.9
        assert hit.latency == 0.1

    def test_miss_returns_none(self, tmp_path):
        from context_bench.cache import ResultCache
        cache = ResultCache(tmp_path, ["s"], ["d"], ["e"])
        assert cache.get("s", "d", "missing") is None

    def test_persists_across_instances(self, tmp_path):
        from context_bench.cache import ResultCache
        cfg = dict(cache_dir=tmp_path, systems=["s"], datasets=["d"], evaluators=["e"])
        cache1 = ResultCache(**cfg)
        cache1.put(EvalRow("s", 0, {"f1": 1.0}, 10, 5, dataset="d"))
        del cache1

        cache2 = ResultCache(**cfg)
        assert cache2.cached_count == 1
        assert cache2.get("s", "d", 0) is not None

    def test_different_config_different_cache(self, tmp_path):
        from context_bench.cache import ResultCache
        cache1 = ResultCache(tmp_path, ["s1"], ["d"], ["e"])
        cache1.put(EvalRow("s1", 0, {"f1": 1.0}, 10, 5, dataset="d"))

        cache2 = ResultCache(tmp_path, ["s2"], ["d"], ["e"])
        assert cache2.cached_count == 0

    def test_cached_count(self, tmp_path):
        from context_bench.cache import ResultCache
        cache = ResultCache(tmp_path, ["s"], ["d"], ["e"])
        assert cache.cached_count == 0
        cache.put(EvalRow("s", 0, {"f1": 1.0}, 10, 5, dataset="d"))
        cache.put(EvalRow("s", 1, {"f1": 0.8}, 10, 5, dataset="d"))
        assert cache.cached_count == 2


class TestRunnerCaching:
    def test_cache_skips_reprocessing(self, tmp_path):
        from context_bench import evaluate

        call_count = 0

        class CountingSystem:
            @property
            def name(self):
                return "counter"
            def process(self, example):
                nonlocal call_count
                call_count += 1
                return {**example, "response": "ok"}

        class DummyEval:
            @property
            def name(self):
                return "d"
            def score(self, o, p):
                return {"s": 1.0}

        dataset = [{"id": i, "context": f"c{i}", "dataset": "test"} for i in range(5)]

        # First run: all processed
        call_count = 0
        evaluate(
            systems=[CountingSystem()],
            dataset=dataset,
            evaluators=[DummyEval()],
            progress=False,
            cache_dir=str(tmp_path),
        )
        assert call_count == 5

        # Second run: all cached
        call_count = 0
        result = evaluate(
            systems=[CountingSystem()],
            dataset=dataset,
            evaluators=[DummyEval()],
            progress=False,
            cache_dir=str(tmp_path),
        )
        assert call_count == 0
        assert len(result.rows) == 5

    def test_cache_with_concurrent(self, tmp_path):
        from context_bench import evaluate

        class IdSystem:
            @property
            def name(self):
                return "id"
            def process(self, example):
                return {**example, "response": "r"}

        class DummyEval:
            @property
            def name(self):
                return "d"
            def score(self, o, p):
                return {"s": 1.0}

        dataset = [{"id": i, "context": f"c{i}", "dataset": "test"} for i in range(10)]

        # First run with concurrency
        r1 = evaluate(
            systems=[IdSystem()],
            dataset=dataset,
            evaluators=[DummyEval()],
            progress=False,
            max_workers=3,
            cache_dir=str(tmp_path),
        )

        # Second run: should use cache
        r2 = evaluate(
            systems=[IdSystem()],
            dataset=dataset,
            evaluators=[DummyEval()],
            progress=False,
            cache_dir=str(tmp_path),
        )
        assert len(r2.rows) == 10


# ===========================================================================
# HTML reporter
# ===========================================================================


class TestHTMLReporter:
    def test_basic_output(self):
        from context_bench.reporters.html import to_html
        result = EvalResult(
            rows=[EvalRow("sys_a", 0, {"f1": 0.9}, 100, 50, latency=0.1, dataset="test")],
            summary={"sys_a": {"mean_score": 0.9, "pass_rate": 1.0}},
            timing={"sys_a": 1.5},
            config={"num_examples": 1},
        )
        html = to_html(result)
        assert "<!DOCTYPE html>" in html
        assert "context-bench" in html
        assert "sys_a" in html
        assert "0.9000" in html

    def test_empty_result(self):
        from context_bench.reporters.html import to_html
        result = EvalResult(rows=[], summary={})
        html = to_html(result)
        assert "No results" in html

    def test_multiple_systems(self):
        from context_bench.reporters.html import to_html
        result = EvalResult(
            rows=[
                EvalRow("sys_a", 0, {"f1": 0.9}, 100, 50, dataset="test"),
                EvalRow("sys_b", 0, {"f1": 0.7}, 100, 50, dataset="test"),
            ],
            summary={
                "sys_a": {"mean_score": 0.9},
                "sys_b": {"mean_score": 0.7},
            },
        )
        html = to_html(result)
        assert "sys_a" in html
        assert "sys_b" in html

    def test_contains_javascript_table(self):
        from context_bench.reporters.html import to_html
        result = EvalResult(
            rows=[EvalRow("s", 0, {"f1": 0.5}, 10, 5, dataset="d")],
            summary={"s": {"mean_score": 0.5}},
        )
        html = to_html(result)
        assert "rowData" in html
        assert "filterRows" in html


# ===========================================================================
# CLI --output html
# ===========================================================================


class TestCLIHtmlOutput:
    def test_main_html_output(self, capsys):
        fake_response = {"choices": [{"message": {"content": "ok"}}]}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"id": 0, "context": "ctx", "answer": "ok"}) + "\n")
            path = f.name

        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = json.dumps(fake_response).encode()
        mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
        mock_resp.__exit__ = mock.Mock(return_value=False)

        from context_bench.__main__ import main
        with mock.patch("urllib.request.urlopen", return_value=mock_resp):
            main(["--proxy", "http://localhost:7878", "--dataset", path, "--output", "html"])

        captured = capsys.readouterr()
        assert "<!DOCTYPE html>" in captured.out
        assert "context-bench" in captured.out

        Path(path).unlink()


# ===========================================================================
# CLI --cache-dir
# ===========================================================================


class TestCLICacheDir:
    def test_cache_dir_arg_parsed(self):
        from context_bench.__main__ import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "--proxy", "http://localhost:7878",
            "--dataset", "hotpotqa",
            "--cache-dir", "/tmp/cache",
        ])
        assert args.cache_dir == "/tmp/cache"


# ===========================================================================
# Multi-turn support
# ===========================================================================


class TestMultiTurnRunner:
    def test_multi_turn_dispatches_to_process_conversation(self):
        from context_bench import evaluate

        class MultiTurnSystem:
            @property
            def name(self):
                return "mt"
            def process(self, example):
                return {**example, "response": "single-turn"}
            def process_conversation(self, turns):
                # Echo back with turn count
                return [
                    {"role": "assistant", "content": f"Response to turn {i+1}"}
                    for i in range(len(turns))
                ]

        class DummyEval:
            @property
            def name(self):
                return "d"
            def score(self, o, p):
                return {"s": 1.0 if "turn" in p.get("response", "") else 0.0}

        dataset = [{
            "id": 0,
            "multi_turn": True,
            "turns": [
                {"role": "user", "content": "Hello"},
                {"role": "user", "content": "How are you?"},
            ],
            "answer": "Response to turn 2",
        }]

        result = evaluate(
            systems=[MultiTurnSystem()],
            dataset=dataset,
            evaluators=[DummyEval()],
            progress=False,
        )
        assert len(result.rows) == 1
        assert result.rows[0].scores["s"] == 1.0

    def test_multi_turn_stores_turn_responses(self):
        from context_bench.runner import _process_example

        class MTSystem:
            @property
            def name(self):
                return "mt"
            def process_conversation(self, turns):
                return [{"role": "assistant", "content": f"R{i}"} for i in range(len(turns))]

        example = {
            "id": 0,
            "multi_turn": True,
            "turns": [
                {"role": "user", "content": "T1"},
                {"role": "user", "content": "T2"},
                {"role": "user", "content": "T3"},
            ],
        }

        class DummyEval:
            def score(self, o, p):
                # Check that turn_responses is set
                return {"n_turns": float(len(p.get("turn_responses", [])))}

        row = _process_example(MTSystem(), example, 0, [DummyEval()], None)
        assert row.scores["n_turns"] == 3.0

    def test_single_turn_ignores_multi_turn_protocol(self):
        from context_bench import evaluate

        class SimpleSystem:
            @property
            def name(self):
                return "simple"
            def process(self, example):
                return {**example, "response": "single"}

        class DummyEval:
            @property
            def name(self):
                return "d"
            def score(self, o, p):
                return {"s": 1.0 if p.get("response") == "single" else 0.0}

        dataset = [{"id": 0, "context": "hello", "answer": "single"}]
        result = evaluate(
            systems=[SimpleSystem()],
            dataset=dataset,
            evaluators=[DummyEval()],
            progress=False,
        )
        assert result.rows[0].scores["s"] == 1.0

    def test_fallback_when_no_process_conversation(self):
        """If system lacks process_conversation, process() is used even for multi_turn."""
        from context_bench import evaluate

        class SingleOnlySystem:
            @property
            def name(self):
                return "single"
            def process(self, example):
                return {**example, "response": "fallback"}

        class DummyEval:
            @property
            def name(self):
                return "d"
            def score(self, o, p):
                return {"s": 1.0 if p.get("response") == "fallback" else 0.0}

        dataset = [{
            "id": 0,
            "multi_turn": True,
            "turns": [{"role": "user", "content": "Hi"}],
            "answer": "fallback",
        }]
        result = evaluate(
            systems=[SingleOnlySystem()],
            dataset=dataset,
            evaluators=[DummyEval()],
            progress=False,
        )
        assert result.rows[0].scores["s"] == 1.0


# ===========================================================================
# OpenAI proxy: multi-turn messages and usage extraction
# ===========================================================================


class TestOpenAIProxyMultiTurn:
    def test_messages_with_turns_field(self):
        from context_bench.systems.openai_proxy import OpenAIProxy
        proxy = OpenAIProxy(base_url="http://localhost:7878")
        example = {
            "turns": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How's the weather?"},
            ],
        }
        messages = proxy._messages(example)
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[2]["content"] == "How's the weather?"

    def test_messages_with_turns_and_system_prompt(self):
        from context_bench.systems.openai_proxy import OpenAIProxy
        proxy = OpenAIProxy(base_url="http://localhost:7878", system_prompt="Be helpful")
        example = {
            "turns": [{"role": "user", "content": "Hello"}],
        }
        messages = proxy._messages(example)
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be helpful"
        assert messages[1]["role"] == "user"


class TestOpenAIProxyUsage:
    def test_process_extracts_usage(self):
        from context_bench.systems.openai_proxy import OpenAIProxy
        proxy = OpenAIProxy(base_url="http://localhost:7878")

        fake_response = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = json.dumps(fake_response).encode()
        mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
        mock_resp.__exit__ = mock.Mock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_resp):
            result = proxy.process({"context": "test"})

        assert result["api_usage"]["prompt_tokens"] == 10
        assert result["api_usage"]["completion_tokens"] == 5

    def test_process_without_usage(self):
        from context_bench.systems.openai_proxy import OpenAIProxy
        proxy = OpenAIProxy(base_url="http://localhost:7878")

        fake_response = {"choices": [{"message": {"content": "ok"}}]}
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = json.dumps(fake_response).encode()
        mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
        mock_resp.__exit__ = mock.Mock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_resp):
            result = proxy.process({"context": "test"})

        assert "api_usage" not in result


class TestRunnerUsageExtraction:
    def test_api_usage_stored_in_metadata(self):
        from context_bench.runner import _process_example

        class UsageSystem:
            @property
            def name(self):
                return "usage"
            def process(self, example):
                return {
                    **example,
                    "response": "ok",
                    "api_usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                }

        class DummyEval:
            def score(self, o, p):
                return {"s": 1.0}

        row = _process_example(UsageSystem(), {"id": 0, "context": "x"}, 0, [DummyEval()], None)
        assert row.metadata["prompt_tokens"] == 100
        assert row.metadata["completion_tokens"] == 20
        assert row.metadata["total_tokens"] == 120

    def test_no_usage_empty_metadata(self):
        from context_bench.runner import _process_example

        class SimpleSystem:
            @property
            def name(self):
                return "simple"
            def process(self, example):
                return {**example, "response": "ok"}

        class DummyEval:
            def score(self, o, p):
                return {"s": 1.0}

        row = _process_example(SimpleSystem(), {"id": 0, "context": "x"}, 0, [DummyEval()], None)
        assert "prompt_tokens" not in row.metadata
