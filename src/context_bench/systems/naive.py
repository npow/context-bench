"""Naive full-context memory system (new protocol).

Baseline: concatenate the entire history into the prompt and answer the
question.  No compression, no retrieval -- just raw context stuffing.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from context_bench.memory_types import (
    ConversationTurn,
    Declaration,
    DocumentChunk,
    IngestResult,
    Item,
    PlatformEvent,
    QueryResult,
)


class NaiveSystem:
    """Stuffs all ingested content into the LLM context verbatim.

    Args:
        base_url: OpenAI-compatible relay URL.
        model: Model name for the chat completion request.
        api_key: Bearer token.  Falls back to OPENAI_API_KEY env var.
        max_items: If set, only the last N items are included (sliding window).
        timeout: HTTP timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        max_items: int | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._max_items = max_items
        self._timeout = timeout
        self._items: list[Item] = []

    # ------------------------------------------------------------------
    # MemorySystem protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        suffix = f"_last{self._max_items}" if self._max_items else ""
        return f"naive{suffix}"

    def reset(self) -> None:
        self._items = []

    def ingest(self, items: list[Item]) -> IngestResult:
        t0 = time.monotonic()
        self._items = list(items)
        elapsed_ms = (time.monotonic() - t0) * 1000
        return IngestResult(num_items=len(items), latency_ms=elapsed_ms)

    def query(self, question: str, budget: int | None = None) -> QueryResult:
        t0 = time.monotonic()

        items = self._items
        if self._max_items is not None:
            items = items[-self._max_items:]

        history = "\n".join(_render_item(item) for item in items)

        context_tokens = len(history.split()) + len(question.split())

        system_prompt = (
            "You answer questions about conversations. "
            "Rules:\n"
            "1. Answer in ONE short sentence or phrase (under 15 words)\n"
            "2. No explanations, no elaboration, no dashes followed by reasoning\n"
            "3. Just the direct answer, nothing else\n"
            "4. Examples: 'Psychology, counseling certification' / 'Likely no' / "
            "'Yes, since she collects classic children's books' / 'Liberal' / "
            "'Thoughtful, authentic, driven' / 'National park; she likes the outdoors'"
        )

        # Log if context was truncated by max_items
        if self._max_items is not None and len(self._items) > self._max_items:
            print(
                f"[NaiveSystem] WARNING: context truncated from "
                f"{len(self._items)} to {self._max_items} items",
                flush=True,
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{history}\n\n"
                    f"Question: {question}\n\n"
                    "Short answer:"
                ),
            },
        ]

        answer = self._chat(messages)
        elapsed_ms = (time.monotonic() - t0) * 1000

        return QueryResult(
            answer=answer,
            total_latency_ms=elapsed_ms,
            context_tokens=context_tokens,
        )

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


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _render_item(item: Item) -> str:
    """Convert any Item variant to a single text line for the prompt."""
    if isinstance(item, ConversationTurn):
        prefix = item.role.upper()
        if item.speaker:
            prefix = f"{item.speaker} ({prefix})"
        return f"{prefix}: {item.content}"
    if isinstance(item, DocumentChunk):
        label = item.source or item.document_id
        return f"[DOC {label} #{item.position}] {item.content}"
    if isinstance(item, PlatformEvent):
        author = item.author or "unknown"
        return f"[{item.platform.upper()} {item.timestamp}] {author}: {item.content}"
    if isinstance(item, Declaration):
        return f"[DECL {item.key}] {item.value}"
    # Fallback — should not happen with typed Items.
    return str(getattr(item, "content", item))
