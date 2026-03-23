"""Mem0 memory system wrapper.

Points Mem0's internal LLM calls at your existing relay so you don't
pay for a separate API. Mem0 handles extraction, storage, and retrieval;
we call the relay for the final answer.

Requires: pip install context-bench[mem0]
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from typing import Any


def _require_mem0() -> Any:
    try:
        from mem0 import Memory
        return Memory
    except ImportError:
        raise ImportError(
            "mem0ai required. Install with: pip install context-bench[mem0]"
        )


class Mem0System:
    """Mem0 memory system evaluated against the MemorySystem protocol.

    Mem0's internal LLM calls (extraction, deduplication, summarization)
    are routed through the relay. The final question-answering call also
    uses the relay.

    Args:
        relay_url: OpenAI-compatible relay URL (e.g. "http://localhost:7878").
        model: Model name for both Mem0 internals and final answer.
        api_key: Bearer token. Falls back to OPENAI_API_KEY env var.
        timeout: HTTP timeout in seconds for the answer call.
    """

    def __init__(
        self,
        relay_url: str,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._relay_url = relay_url.rstrip("/")
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._timeout = timeout
        self._user_id: str = ""
        self._mem: Any = None
        self._init_mem0()

    def _init_mem0(self) -> None:
        Memory = _require_mem0()
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": self._model,
                    "openai_base_url": f"{self._relay_url}/v1",
                    "api_key": self._api_key or "relay",
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                },
            },
            # Use in-memory vector store — no external DB needed
            "vector_store": {
                "provider": "qdrant",
                "config": {"collection_name": "mem0_eval", "on_disk": False},
            },
        }
        self._mem = Memory.from_config(config)

    @property
    def name(self) -> str:
        return "mem0"

    # ------------------------------------------------------------------
    # MemorySystem protocol
    # ------------------------------------------------------------------

    def reset(self) -> None:
        # Use a fresh user_id per conversation to isolate memory state
        self._user_id = str(uuid.uuid4())

    def ingest(self, turns: list) -> None:
        # Convert typed Items to dicts if needed
        messages = []
        for t in turns:
            if hasattr(t, "role") and hasattr(t, "content"):
                messages.append({"role": t.role, "content": t.content})
            elif isinstance(t, dict):
                messages.append({"role": t.get("role", "user"), "content": t.get("content", "")})
        # Mem0 ingests the full conversation at once
        self._mem.add(messages, user_id=self._user_id)

    def query(self, question: str) -> str:
        # Retrieve relevant memories
        memories = self._mem.search(question, user_id=self._user_id, limit=10)
        memory_text = "\n".join(
            f"- {m['memory']}" for m in memories.get("results", [])
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the question using "
                    "the provided memory context. Be concise and accurate."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Relevant memory:\n{memory_text}\n\n"
                    f"Question: {question}"
                ),
            },
        ]
        return self._chat(messages)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _chat(self, messages: list[dict[str, Any]]) -> str:
        url = f"{self._relay_url}/v1/chat/completions"
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
