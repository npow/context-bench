"""Tests for the OpenAI proxy system."""

from __future__ import annotations

import io
import json
from unittest import mock

import pytest

from context_bench.systems.openai_proxy import OpenAIProxy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(body: dict) -> mock.MagicMock:
    """Create a fake urllib response that returns *body* as JSON."""
    raw = json.dumps(body).encode()
    resp = mock.MagicMock()
    resp.read.return_value = raw
    resp.__enter__ = mock.Mock(return_value=resp)
    resp.__exit__ = mock.Mock(return_value=False)
    return resp


SIMPLE_RESPONSE = {
    "choices": [{"message": {"content": "hello world"}}],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicFlow:
    def test_process_returns_response(self):
        proxy = OpenAIProxy(base_url="http://localhost:8080", model="test-model")
        with mock.patch("urllib.request.urlopen", return_value=_mock_response(SIMPLE_RESPONSE)):
            result = proxy.process({"id": 1, "context": "some context"})
        assert result["response"] == "hello world"
        assert result["context"] == "some context"
        assert result["id"] == 1

    def test_url_construction(self):
        proxy = OpenAIProxy(base_url="http://localhost:8080/")
        with mock.patch("urllib.request.urlopen", return_value=_mock_response(SIMPLE_RESPONSE)) as mock_open:
            proxy.process({"id": 1, "context": "ctx"})
        req = mock_open.call_args[0][0]
        assert req.full_url == "http://localhost:8080/v1/chat/completions"


class TestMessageBuilding:
    def test_context_only(self):
        """Without a question, context is sent as user message."""
        proxy = OpenAIProxy(base_url="http://localhost:8080")
        with mock.patch("urllib.request.urlopen", return_value=_mock_response(SIMPLE_RESPONSE)) as mock_open:
            proxy.process({"id": 1, "context": "ctx text"})
        body = json.loads(mock_open.call_args[0][0].data)
        messages = body["messages"]
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "ctx text"}

    def test_context_and_question(self):
        """With a question, context is system and question is user."""
        proxy = OpenAIProxy(base_url="http://localhost:8080")
        with mock.patch("urllib.request.urlopen", return_value=_mock_response(SIMPLE_RESPONSE)) as mock_open:
            proxy.process({"id": 1, "context": "ctx", "question": "what?"})
        body = json.loads(mock_open.call_args[0][0].data)
        messages = body["messages"]
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "ctx"}
        assert messages[1] == {"role": "user", "content": "what?"}

    def test_system_prompt_prepended(self):
        proxy = OpenAIProxy(base_url="http://localhost:8080", system_prompt="Be helpful.")
        with mock.patch("urllib.request.urlopen", return_value=_mock_response(SIMPLE_RESPONSE)) as mock_open:
            proxy.process({"id": 1, "context": "ctx", "question": "q?"})
        body = json.loads(mock_open.call_args[0][0].data)
        messages = body["messages"]
        assert len(messages) == 3
        assert messages[0] == {"role": "system", "content": "Be helpful."}
        assert messages[1] == {"role": "system", "content": "ctx"}
        assert messages[2] == {"role": "user", "content": "q?"}

    def test_custom_build_messages(self):
        def custom(example):
            return [{"role": "user", "content": f"Custom: {example['context']}"}]

        proxy = OpenAIProxy(base_url="http://localhost:8080", build_messages=custom)
        with mock.patch("urllib.request.urlopen", return_value=_mock_response(SIMPLE_RESPONSE)) as mock_open:
            proxy.process({"id": 1, "context": "data"})
        body = json.loads(mock_open.call_args[0][0].data)
        assert body["messages"] == [{"role": "user", "content": "Custom: data"}]


class TestCustomParams:
    def test_model_in_body(self):
        proxy = OpenAIProxy(base_url="http://localhost:8080", model="gpt-4")
        with mock.patch("urllib.request.urlopen", return_value=_mock_response(SIMPLE_RESPONSE)) as mock_open:
            proxy.process({"id": 1, "context": "c"})
        body = json.loads(mock_open.call_args[0][0].data)
        assert body["model"] == "gpt-4"

    def test_extra_body_merged(self):
        proxy = OpenAIProxy(
            base_url="http://localhost:8080",
            extra_body={"temperature": 0, "max_tokens": 100},
        )
        with mock.patch("urllib.request.urlopen", return_value=_mock_response(SIMPLE_RESPONSE)) as mock_open:
            proxy.process({"id": 1, "context": "c"})
        body = json.loads(mock_open.call_args[0][0].data)
        assert body["temperature"] == 0
        assert body["max_tokens"] == 100


class TestAPIKey:
    def test_explicit_api_key(self):
        proxy = OpenAIProxy(base_url="http://localhost:8080", api_key="sk-test-123")
        with mock.patch("urllib.request.urlopen", return_value=_mock_response(SIMPLE_RESPONSE)) as mock_open:
            proxy.process({"id": 1, "context": "c"})
        req = mock_open.call_args[0][0]
        assert req.get_header("Authorization") == "Bearer sk-test-123"

    def test_env_var_fallback(self):
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env-456"}):
            proxy = OpenAIProxy(base_url="http://localhost:8080")
        with mock.patch("urllib.request.urlopen", return_value=_mock_response(SIMPLE_RESPONSE)) as mock_open:
            proxy.process({"id": 1, "context": "c"})
        req = mock_open.call_args[0][0]
        assert req.get_header("Authorization") == "Bearer sk-env-456"

    def test_no_api_key_no_header(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            proxy = OpenAIProxy(base_url="http://localhost:8080")
        with mock.patch("urllib.request.urlopen", return_value=_mock_response(SIMPLE_RESPONSE)) as mock_open:
            proxy.process({"id": 1, "context": "c"})
        req = mock_open.call_args[0][0]
        assert req.get_header("Authorization") is None


class TestErrorHandling:
    def test_http_error(self):
        import urllib.error
        exc = urllib.error.HTTPError(
            url="http://localhost:8080/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=io.BytesIO(b""),
        )
        proxy = OpenAIProxy(base_url="http://localhost:8080")
        with mock.patch("urllib.request.urlopen", side_effect=exc):
            with pytest.raises(RuntimeError, match="HTTP 429"):
                proxy.process({"id": 1, "context": "c"})

    def test_connection_error(self):
        import urllib.error
        exc = urllib.error.URLError("Connection refused")
        proxy = OpenAIProxy(base_url="http://localhost:8080")
        with mock.patch("urllib.request.urlopen", side_effect=exc):
            with pytest.raises(RuntimeError, match="Could not connect"):
                proxy.process({"id": 1, "context": "c"})

    def test_bad_json(self):
        resp = mock.MagicMock()
        resp.read.return_value = b"not json at all"
        resp.__enter__ = mock.Mock(return_value=resp)
        resp.__exit__ = mock.Mock(return_value=False)
        proxy = OpenAIProxy(base_url="http://localhost:8080")
        with mock.patch("urllib.request.urlopen", return_value=resp):
            with pytest.raises(RuntimeError, match="invalid JSON"):
                proxy.process({"id": 1, "context": "c"})

    def test_empty_choices(self):
        proxy = OpenAIProxy(base_url="http://localhost:8080")
        resp = _mock_response({"choices": []})
        with mock.patch("urllib.request.urlopen", return_value=resp):
            with pytest.raises(RuntimeError, match="empty choices"):
                proxy.process({"id": 1, "context": "c"})

    def test_missing_choices_key(self):
        proxy = OpenAIProxy(base_url="http://localhost:8080")
        resp = _mock_response({"error": "bad request"})
        with mock.patch("urllib.request.urlopen", return_value=resp):
            with pytest.raises(RuntimeError, match="Unexpected response"):
                proxy.process({"id": 1, "context": "c"})


class TestNameDefaults:
    def test_default_name_from_host(self):
        proxy = OpenAIProxy(base_url="http://localhost:8080")
        assert proxy.name == "openai_proxy_localhost:8080"

    def test_default_name_strips_path(self):
        proxy = OpenAIProxy(base_url="http://myhost.com/prefix")
        assert proxy.name == "openai_proxy_myhost.com"

    def test_custom_name(self):
        proxy = OpenAIProxy(base_url="http://localhost:8080", name="my_proxy")
        assert proxy.name == "my_proxy"

    def test_trailing_slash_stripped(self):
        proxy = OpenAIProxy(base_url="http://localhost:8080/")
        assert proxy.name == "openai_proxy_localhost:8080"
