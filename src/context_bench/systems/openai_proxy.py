"""OpenAI-compatible proxy system for context-bench.

Point at any OpenAI-compatible endpoint and benchmark it without writing
boilerplate HTTP code.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable


class OpenAIProxy:
    """System that forwards examples to an OpenAI-compatible chat endpoint.

    Args:
        base_url: Root URL of the proxy, e.g. ``"http://localhost:8080"``.
        model: Model name to send in the request body.
        name: Display name for benchmark results.  Defaults to
            ``"openai_proxy_<host>"``.
        api_key: Bearer token.  Falls back to the ``OPENAI_API_KEY``
            environment variable when *None*.
        system_prompt: If provided, prepended as a system message before the
            default message building logic.
        build_messages: Full override for message construction.  Receives the
            example dict and must return a list of message dicts.
        timeout: HTTP request timeout in seconds.
        extra_body: Additional keys merged into the request body
            (e.g. ``{"temperature": 0, "max_tokens": 256}``).
    """

    def __init__(
        self,
        base_url: str,
        model: str = "gpt-3.5-turbo",
        name: str | None = None,
        api_key: str | None = None,
        system_prompt: str | None = None,
        build_messages: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None,
        timeout: float = 30.0,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._system_prompt = system_prompt
        self._build_messages = build_messages
        self._timeout = timeout
        self._extra_body = extra_body or {}

        # Derive a default name from the host portion of the URL.
        if name is not None:
            self._name = name
        else:
            # Extract host from URL (e.g. "localhost:8080" from "http://localhost:8080")
            host = self._base_url.split("://", 1)[-1].split("/", 1)[0]
            self._name = f"openai_proxy_{host}"

    # ------------------------------------------------------------------
    # System protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    def process(self, example: dict[str, Any]) -> dict[str, Any]:
        """Send example to the proxy and return the dict with a ``response`` key."""
        messages = self._messages(example)
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            **self._extra_body,
        }
        content = self._post(body)
        return {**example, "response": content}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        """Build the messages list for the chat completion request."""
        if self._build_messages is not None:
            return self._build_messages(example)

        messages: list[dict[str, Any]] = []

        if self._system_prompt is not None:
            messages.append({"role": "system", "content": self._system_prompt})

        context = example.get("context", "")
        question = example.get("question")

        if question is not None:
            # Context as system message, question as user message.
            messages.append({"role": "system", "content": context})
            messages.append({"role": "user", "content": question})
        else:
            # Context only — send as user message.
            messages.append({"role": "user", "content": context})

        return messages

    def _post(self, body: dict[str, Any]) -> str:
        """POST to the chat completions endpoint and return assistant content."""
        url = f"{self._base_url}/v1/chat/completions"
        data = json.dumps(body).encode()

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                resp_data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"OpenAI proxy returned HTTP {exc.code}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not connect to OpenAI proxy at {url}: {exc.reason}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"OpenAI proxy returned invalid JSON: {exc}"
            ) from exc

        # Extract assistant content.
        try:
            choices = resp_data["choices"]
            if not choices:
                raise RuntimeError("OpenAI proxy returned empty choices list")
            return choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected response structure from OpenAI proxy: {resp_data}"
            ) from exc
