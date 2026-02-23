"""Tests for retry logic and rich progress bar integration."""

from __future__ import annotations

import json
from unittest import mock

import pytest


# ===========================================================================
# Retry logic
# ===========================================================================


class TestRetryLogic:
    def _make_proxy(self, **kwargs):
        from context_bench.systems.openai_proxy import OpenAIProxy
        return OpenAIProxy(base_url="http://localhost:7878", **kwargs)

    def _mock_success_response(self, content="ok"):
        resp = mock.MagicMock()
        resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": content}}],
        }).encode()
        resp.__enter__ = mock.Mock(return_value=resp)
        resp.__exit__ = mock.Mock(return_value=False)
        return resp

    def test_no_retry_on_success(self):
        proxy = self._make_proxy(max_retries=3)
        mock_resp = self._mock_success_response()

        with mock.patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            result = proxy.process({"context": "test"})

        assert result["response"] == "ok"
        assert mock_open.call_count == 1

    def test_retry_on_429(self):
        import urllib.error
        proxy = self._make_proxy(max_retries=2, retry_base_delay=0.01)

        error_429 = urllib.error.HTTPError(
            "http://test", 429, "Too Many Requests", {}, None
        )
        mock_resp = self._mock_success_response()

        with mock.patch("urllib.request.urlopen", side_effect=[error_429, mock_resp]) as mock_open:
            result = proxy.process({"context": "test"})

        assert result["response"] == "ok"
        assert mock_open.call_count == 2

    def test_retry_on_500(self):
        import urllib.error
        proxy = self._make_proxy(max_retries=2, retry_base_delay=0.01)

        error_500 = urllib.error.HTTPError(
            "http://test", 500, "Internal Server Error", {}, None
        )
        mock_resp = self._mock_success_response()

        with mock.patch("urllib.request.urlopen", side_effect=[error_500, mock_resp]):
            result = proxy.process({"context": "test"})

        assert result["response"] == "ok"

    def test_retry_on_connection_error(self):
        import urllib.error
        proxy = self._make_proxy(max_retries=2, retry_base_delay=0.01)

        conn_error = urllib.error.URLError("Connection refused")
        mock_resp = self._mock_success_response()

        with mock.patch("urllib.request.urlopen", side_effect=[conn_error, mock_resp]):
            result = proxy.process({"context": "test"})

        assert result["response"] == "ok"

    def test_no_retry_on_400(self):
        import urllib.error
        proxy = self._make_proxy(max_retries=3, retry_base_delay=0.01)

        error_400 = urllib.error.HTTPError(
            "http://test", 400, "Bad Request", {}, None
        )

        with mock.patch("urllib.request.urlopen", side_effect=error_400):
            with pytest.raises(RuntimeError, match="HTTP 400"):
                proxy.process({"context": "test"})

    def test_exhausted_retries_raises(self):
        import urllib.error
        proxy = self._make_proxy(max_retries=2, retry_base_delay=0.01)

        error_503 = urllib.error.HTTPError(
            "http://test", 503, "Service Unavailable", {}, None
        )

        with mock.patch("urllib.request.urlopen", side_effect=error_503):
            with pytest.raises(RuntimeError, match="HTTP 503"):
                proxy.process({"context": "test"})

    def test_retry_disabled(self):
        import urllib.error
        proxy = self._make_proxy(max_retries=0)

        error_429 = urllib.error.HTTPError(
            "http://test", 429, "Too Many Requests", {}, None
        )

        with mock.patch("urllib.request.urlopen", side_effect=error_429):
            with pytest.raises(RuntimeError, match="HTTP 429"):
                proxy.process({"context": "test"})

    def test_respects_retry_after_header(self):
        import urllib.error
        proxy = self._make_proxy(max_retries=1, retry_base_delay=0.01)

        headers = mock.MagicMock()
        headers.get.return_value = "0.01"
        error_429 = urllib.error.HTTPError(
            "http://test", 429, "Too Many Requests", headers, None
        )
        mock_resp = self._mock_success_response()

        with mock.patch("urllib.request.urlopen", side_effect=[error_429, mock_resp]):
            with mock.patch("time.sleep") as mock_sleep:
                result = proxy.process({"context": "test"})

        assert result["response"] == "ok"
        # Should have slept at least once
        assert mock_sleep.call_count >= 1


# ===========================================================================
# Rich progress detection
# ===========================================================================


class TestRichDetection:
    def test_has_rich_returns_true_when_installed(self):
        from context_bench.runner import _has_rich
        # rich is in our dev dependencies, so it should be available
        assert _has_rich() is True

    def test_has_rich_returns_false_when_missing(self):
        from context_bench.runner import _has_rich
        with mock.patch.dict("sys.modules", {"rich": None}):
            assert _has_rich() is False


# ===========================================================================
# Runner with rich progress (integration)
# ===========================================================================


class TestRunnerProgress:
    def test_sequential_with_rich(self):
        """Runner works with rich progress bars enabled."""
        from context_bench import evaluate

        class EchoSystem:
            @property
            def name(self):
                return "echo"
            def process(self, example):
                return {**example, "response": "ok"}

        class DummyEval:
            @property
            def name(self):
                return "d"
            def score(self, o, p):
                return {"s": 1.0}

        dataset = [{"id": i, "context": f"c{i}"} for i in range(5)]
        result = evaluate(
            systems=[EchoSystem()],
            dataset=dataset,
            evaluators=[DummyEval()],
            progress=True,  # Will use rich since it's installed
        )
        assert len(result.rows) == 5

    def test_concurrent_with_rich(self):
        """Runner works with rich + concurrency."""
        from context_bench import evaluate

        class EchoSystem:
            @property
            def name(self):
                return "echo"
            def process(self, example):
                return {**example, "response": "ok"}

        class DummyEval:
            @property
            def name(self):
                return "d"
            def score(self, o, p):
                return {"s": 1.0}

        dataset = [{"id": i, "context": f"c{i}"} for i in range(10)]
        result = evaluate(
            systems=[EchoSystem()],
            dataset=dataset,
            evaluators=[DummyEval()],
            progress=True,
            max_workers=3,
        )
        assert len(result.rows) == 10

    def test_progress_false_skips_rich(self):
        """progress=False produces no output."""
        from context_bench import evaluate

        class EchoSystem:
            @property
            def name(self):
                return "echo"
            def process(self, example):
                return {**example, "response": "ok"}

        class DummyEval:
            @property
            def name(self):
                return "d"
            def score(self, o, p):
                return {"s": 1.0}

        result = evaluate(
            systems=[EchoSystem()],
            dataset=[{"id": 0, "context": "c"}],
            evaluators=[DummyEval()],
            progress=False,
        )
        assert len(result.rows) == 1
