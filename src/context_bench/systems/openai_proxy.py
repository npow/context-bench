"""OpenAI-compatible proxy system for context-bench.

Point at any OpenAI-compatible endpoint and benchmark it without writing
boilerplate HTTP code.
"""

from __future__ import annotations

import json
import os
import time
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
        max_retries: Number of retries on transient failures (HTTP 429/5xx,
            connection errors). 0 disables retries.
        retry_base_delay: Base delay in seconds for exponential backoff.
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
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._system_prompt = system_prompt
        self._build_messages = build_messages
        self._timeout = timeout
        self._extra_body = extra_body or {}
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay

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
        content, usage = self._post(body)
        result = {**example, "response": content}
        if usage:
            result["api_usage"] = usage
        return result

    def process_conversation(
        self, turns: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Send a multi-turn conversation and collect all assistant responses.

        Args:
            turns: List of user-turn dicts ``{"role": "user", "content": ...}``.
                Assistant responses from prior turns are inserted automatically.

        Returns:
            List of assistant-response dicts ``{"role": "assistant", "content": ...}``,
            one per user turn.
        """
        history: list[dict[str, Any]] = []
        if self._system_prompt is not None:
            history.append({"role": "system", "content": self._system_prompt})

        responses: list[dict[str, str]] = []
        for turn in turns:
            history.append(turn)
            body: dict[str, Any] = {
                "model": self._model,
                "messages": list(history),
                **self._extra_body,
            }
            content, _ = self._post(body)
            assistant_msg = {"role": "assistant", "content": content}
            history.append(assistant_msg)
            responses.append(assistant_msg)

        return responses

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        """Build the messages list for the chat completion request.

        Supports multi-turn examples via a ``turns`` field containing a list
        of ``{"role": ..., "content": ...}`` dicts.
        """
        if self._build_messages is not None:
            return self._build_messages(example)

        messages: list[dict[str, Any]] = []

        if self._system_prompt is not None:
            messages.append({"role": "system", "content": self._system_prompt})

        # Multi-turn: the example has a pre-built turns list
        turns = example.get("turns")
        if turns is not None:
            messages.extend(turns)
            return messages

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

    def _post(self, body: dict[str, Any]) -> tuple[str, dict[str, int] | None]:
        """POST to the chat completions endpoint with retry on transient errors.

        Retries on HTTP 429 (rate limit), 5xx (server errors), and connection
        errors using exponential backoff.

        Returns:
            Tuple of (assistant_content, usage_dict_or_None).
        """
        url = f"{self._base_url}/v1/chat/completions"
        data = json.dumps(body).encode()

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        last_exc: Exception | None = None

        for attempt in range(1 + self._max_retries):
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    resp_data = json.loads(resp.read().decode())

                # Extract usage if present
                usage = resp_data.get("usage")

                # Extract assistant content.
                try:
                    choices = resp_data["choices"]
                    if not choices:
                        raise RuntimeError("OpenAI proxy returned empty choices list")
                    return choices[0]["message"]["content"], usage
                except (KeyError, IndexError, TypeError) as exc:
                    raise RuntimeError(
                        f"Unexpected response structure from OpenAI proxy: {resp_data}"
                    ) from exc

            except urllib.error.HTTPError as exc:
                last_exc = exc
                # Retry on rate limit (429) and server errors (5xx)
                if exc.code in (429, 500, 502, 503, 504) and attempt < self._max_retries:
                    delay = self._retry_base_delay * (2 ** attempt)
                    # Respect Retry-After header if present
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            pass
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"OpenAI proxy returned HTTP {exc.code}: {exc.reason}"
                ) from exc

            except urllib.error.URLError as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    time.sleep(self._retry_base_delay * (2 ** attempt))
                    continue
                raise RuntimeError(
                    f"Could not connect to OpenAI proxy at {url}: {exc.reason}"
                ) from exc

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"OpenAI proxy returned invalid JSON: {exc}"
                ) from exc

        # Should not reach here, but just in case
        raise RuntimeError(
            f"Failed after {self._max_retries} retries: {last_exc}"
        )
