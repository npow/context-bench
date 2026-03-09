"""Naive full-context memory system.

Baseline: concatenate the entire conversation history into the prompt and
answer the question. No compression, no retrieval — just raw context stuffing.

This is the baseline every memory system claims to beat.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any


class NaiveMemorySystem:
    """Stuffs the full conversation history into the LLM context.

    Uses any OpenAI-compatible relay for the final answer call.
    No memory management — everything goes in the prompt verbatim.

    Args:
        base_url: OpenAI-compatible relay URL (e.g. "http://localhost:7878").
        model: Model name to pass in the request body.
        api_key: Bearer token. Falls back to OPENAI_API_KEY env var.
        max_turns: If set, only the last N turns are included (sliding window).
            None means include all turns.
        timeout: HTTP timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        max_turns: int | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._max_turns = max_turns
        self._timeout = timeout
        self._turns: list[dict[str, str]] = []
        self.last_context_tokens: int = 0

    @property
    def name(self) -> str:
        suffix = f"_last{self._max_turns}" if self._max_turns else ""
        return f"naive{suffix}"

    # ------------------------------------------------------------------
    # MemorySystem protocol
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._turns = []
        self.last_context_tokens = 0

    def ingest(self, turns: list[dict[str, str]]) -> None:
        self._turns = list(turns)

    def query(self, question: str) -> str:
        turns = self._turns
        if self._max_turns is not None:
            turns = turns[-self._max_turns:]

        history = "\n".join(
            f"{t['role'].upper()}: {t['content']}" for t in turns
        )
        # Record the word count actually sent so the runner can use it for
        # token_efficiency instead of the full ingested conversation size.
        self.last_context_tokens = len(history.split()) + len(question.split())

        system_prompt = (
            "You are a helpful assistant with access to a conversation history. "
            "Answer the question based only on what was discussed in the conversation. "
            "Be concise and accurate."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{history}\n\n"
                    f"Question: {question}\n\n"
                    "Answer based only on the conversation above:"
                ),
            },
        ]
        return self._chat(messages)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _chat(self, messages: list[dict[str, Any]]) -> str:
        url = f"{self._base_url}/v1/chat/completions"
        body = json.dumps({"model": self._model, "messages": messages}).encode()
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        for attempt in range(3):
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    data = json.loads(resp.read().decode())
                    return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"HTTP {e.code}: {e.reason}") from e
            except urllib.error.URLError as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Connection error: {e.reason}") from e

        raise RuntimeError("Failed after 3 retries")
