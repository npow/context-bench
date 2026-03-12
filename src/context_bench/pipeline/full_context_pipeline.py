# full_context_pipeline.py
# ===================================================================
# Full-context pipeline: stuff the entire conversation into Claude's
# context window and ask the question directly.
#
# LoCoMo conversations are 400-600 turns (~30-60K tokens), well within
# Claude Sonnet's 200K context window.  No retrieval, no extraction,
# no lossy intermediate representation — just the raw conversation.
#
# This is the true SOTA-equivalent approach: the top LoCoMo systems
# (Hindsight + TEMPR) achieve 90% F1 by using Gemini's 1M-token window
# to fit the whole conversation.  We replicate that with Claude.
# ===================================================================

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any


class FullContextPipeline:
    """Full-context pipeline for LoCoMo long-conversation QA.

    Ingest: store all turns verbatim.
    Query:  send entire conversation + question to Claude in one call.
    """

    def __init__(
        self,
        relay_url: str,
        model: str = "sonnet",
        api_key: str | None = None,
        strategy: dict[str, Any] | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = relay_url.rstrip("/")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model
        self._timeout = timeout
        self._turns: list[dict] = []
        self.last_context_tokens: int = 0

    @property
    def name(self) -> str:
        return "full_context_pipeline_v1"

    def reset(self) -> None:
        self._turns = []
        self.last_context_tokens = 0

    # ------------------------------------------------------------------
    # Ingest — just store turns verbatim
    # ------------------------------------------------------------------

    def ingest(self, turns: list[dict[str, Any]]) -> None:
        self._turns = turns

    # ------------------------------------------------------------------
    # Query — full conversation in context
    # ------------------------------------------------------------------

    def query(self, question: str) -> str:
        if not self._turns:
            return self._chat([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ])

        conversation_text = "\n".join(
            f"{t.get('role', 'user').upper()}: {t.get('content', '')}"
            for t in self._turns
        )

        self.last_context_tokens = len(conversation_text.split()) + len(question.split())

        system_prompt = (
            "You are a helpful assistant answering questions about a conversation. "
            "The full conversation is provided below. "
            "Give a short, direct answer — typically 1-10 words. "
            "Do not explain or use bullet points. "
            "If the answer is not in the conversation, say 'unknown'. "
            "Output ONLY the answer."
        )
        user_content = (
            f"Conversation:\n{conversation_text}\n\n"
            f"Question: {question}\n\nAnswer (1-10 words):"
        )

        return self._chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ])

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _chat(self, messages: list[dict[str, Any]]) -> str:
        url = f"{self._base_url}/v1/chat/completions"
        body = json.dumps({"model": self._model, "messages": messages}).encode()
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        for attempt in range(4):
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    data = json.loads(resp.read().decode())
                    return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                    time.sleep(15 * (2 ** attempt))
                    continue
                raise RuntimeError(f"Chat HTTP {e.code}: {e.reason}") from e
            except urllib.error.URLError as e:
                if attempt < 3:
                    time.sleep(15 * (2 ** attempt))
                    continue
                raise RuntimeError(f"Chat connection error: {e.reason}") from e

        raise RuntimeError("Chat request failed after 4 retries")


# Loop expects ContextPipeline class name
ContextPipeline = FullContextPipeline
