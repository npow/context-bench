"""Embedding-based retrieval memory system.

Embeds all ingested items locally with sentence-transformers, retrieves the
top-k most relevant chunks via cosine similarity, then calls an LLM to
synthesise the final answer.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from context_bench.memory_types import (
    ConversationTurn,
    Declaration,
    DocumentChunk,
    IngestResult,
    Item,
    PlatformEvent,
    QueryResult,
)


class EmbeddingSystem:
    """Retrieval-augmented memory using local embeddings.

    Args:
        base_url: OpenAI-compatible relay URL for the answer LLM.
        model: Model name for the chat completion request.
        embedding_model: sentence-transformers model id.
        top_k: Number of items to retrieve per query.
        recency_weight: 0.0-1.0 -- how much to boost recent items.
            The final score is ``(1 - w) * similarity + w * recency``
            where *recency* is linearly scaled so the last item = 1.0.
        api_key: Bearer token.  Falls back to OPENAI_API_KEY env var.
        timeout: HTTP timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        model: str = "claude-haiku-4-5-20251001",
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 10,
        recency_weight: float = 0.0,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._top_k = top_k
        self._recency_weight = max(0.0, min(1.0, recency_weight))
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._timeout = timeout

        self._encoder = SentenceTransformer(embedding_model)

        # Populated by ingest():
        self._store: list[dict[str, Any]] = []  # {embedding, content, metadata}
        self._embeddings: np.ndarray | None = None  # (N, D) matrix

    # ------------------------------------------------------------------
    # MemorySystem protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        w = self._recency_weight
        suffix = f"_recency{w}" if w > 0 else ""
        return f"embedding_top{self._top_k}{suffix}"

    def reset(self) -> None:
        self._store = []
        self._embeddings = None

    def ingest(self, items: list[Item]) -> IngestResult:
        t0 = time.monotonic()

        self._store = []
        texts: list[str] = []
        for idx, item in enumerate(items):
            content = _extract_content(item)
            metadata = _extract_metadata(item, idx)
            texts.append(content)
            self._store.append({"content": content, "metadata": metadata})

        if texts:
            vecs = self._encoder.encode(texts, show_progress_bar=False)
            self._embeddings = np.asarray(vecs, dtype=np.float32)
            # Assign embeddings back into store entries.
            for i, entry in enumerate(self._store):
                entry["embedding"] = self._embeddings[i]
        else:
            self._embeddings = None

        elapsed_ms = (time.monotonic() - t0) * 1000
        return IngestResult(
            num_items=len(items),
            latency_ms=elapsed_ms,
            details={"embedding_dim": int(self._embeddings.shape[1]) if self._embeddings is not None else 0},
        )

    def query(self, question: str, budget: int | None = None) -> QueryResult:
        t0 = time.monotonic()

        top_k = budget if budget is not None else self._top_k

        retrieved = self._retrieve(question, top_k)

        context_block = "\n".join(
            f"[{i+1}] {entry['content']}" for i, entry in enumerate(retrieved)
        )
        context_tokens = len(context_block.split()) + len(question.split())

        system_prompt = (
            "You are a helpful assistant. Use the retrieved context below to "
            "answer the question. Be concise and accurate. If the context does "
            "not contain enough information, say so."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Retrieved context:\n{context_block}\n\n"
                    f"Question: {question}\n\n"
                    "Answer:"
                ),
            },
        ]

        answer = self._chat(messages)
        elapsed_ms = (time.monotonic() - t0) * 1000

        return QueryResult(
            answer=answer,
            total_latency_ms=elapsed_ms,
            context_tokens=context_tokens,
            details={
                "num_retrieved": len(retrieved),
                "top_scores": [float(e["_score"]) for e in retrieved[:5]],
            },
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve(self, question: str, top_k: int) -> list[dict[str, Any]]:
        if not self._store or self._embeddings is None:
            return []

        q_vec = self._encoder.encode([question], show_progress_bar=False)
        q_vec = np.asarray(q_vec, dtype=np.float32).reshape(1, -1)

        # Cosine similarity: dot product of L2-normalised vectors.
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normed = self._embeddings / norms

        q_norm = q_vec / np.maximum(np.linalg.norm(q_vec), 1e-10)
        similarities = (normed @ q_norm.T).squeeze()  # (N,)

        # Optional recency boost.
        if self._recency_weight > 0 and len(self._store) > 1:
            n = len(self._store)
            recency = np.linspace(0.0, 1.0, n, dtype=np.float32)
            w = self._recency_weight
            scores = (1 - w) * similarities + w * recency
        else:
            scores = similarities

        k = min(top_k, len(self._store))
        top_indices = np.argpartition(-scores, k)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results = []
        for idx in top_indices:
            entry = dict(self._store[idx])
            entry["_score"] = float(scores[idx])
            results.append(entry)
        return results

    # ------------------------------------------------------------------
    # LLM call
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

def _extract_content(item: Item) -> str:
    """Pull the textual content from any Item variant."""
    if isinstance(item, Declaration):
        return f"{item.key}: {item.value}"
    return item.content


def _extract_metadata(item: Item, index: int) -> dict[str, Any]:
    """Build a metadata dict from any Item variant."""
    meta: dict[str, Any] = {"index": index, "type": type(item).__name__}
    if isinstance(item, ConversationTurn):
        meta["role"] = item.role
        if item.speaker:
            meta["speaker"] = item.speaker
        if item.timestamp:
            meta["timestamp"] = item.timestamp
    elif isinstance(item, DocumentChunk):
        meta["document_id"] = item.document_id
        meta["position"] = item.position
        if item.source:
            meta["source"] = item.source
    elif isinstance(item, PlatformEvent):
        meta["platform"] = item.platform
        meta["timestamp"] = item.timestamp
        if item.author:
            meta["author"] = item.author
    elif isinstance(item, Declaration):
        meta["key"] = item.key
    return meta
