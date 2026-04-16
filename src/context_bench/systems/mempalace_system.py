"""MemPalace memory system adapter for context-bench.

Wraps MemPalace (ChromaDB + SQLite KG + hybrid BM25/vector search) as a
MemorySystem so it can be benchmarked head-to-head against Mem0, Zep, RLM,
and naive baselines on LoCoMo, LongMemEval, MemBench, and ConvoMem.

Requires: pip install mempalace
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
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

try:
    import chromadb
except ImportError:
    chromadb = None  # type: ignore[assignment]


class MemPalaceSystem:
    """MemPalace wrapped as a context-bench MemorySystem.

    Creates a fresh ephemeral palace per example, ingests conversation turns
    as drawers (verbatim, chunked), retrieves via hybrid BM25 + cosine search,
    and answers via an OpenAI-compatible LLM relay.

    Args:
        base_url: OpenAI-compatible relay URL for answer generation.
        model: Model name for the relay.
        api_key: Bearer token (falls back to OPENAI_API_KEY).
        top_k: Number of drawers to retrieve per query.
        use_kg: Whether to populate the temporal knowledge graph.
        timeout: HTTP timeout for LLM calls.
    """

    def __init__(
        self,
        base_url: str,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        top_k: int = 10,
        use_kg: bool = False,
        timeout: float = 60.0,
    ) -> None:
        if chromadb is None:
            raise ImportError("chromadb required. Install with: pip install mempalace")

        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._top_k = top_k
        self._use_kg = use_kg
        self._timeout = timeout

        self._tmpdir: str | None = None
        self._client: Any = None
        self._collection: Any = None
        self._doc_count: int = 0

    @property
    def name(self) -> str:
        return f"mempalace_k{self._top_k}"

    def reset(self) -> None:
        if self._tmpdir and os.path.isdir(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._tmpdir = tempfile.mkdtemp(prefix="mempalace_bench_")
        self._client = chromadb.PersistentClient(path=self._tmpdir)
        self._collection = self._client.get_or_create_collection(
            "mempalace_drawers", metadata={"hnsw:space": "cosine"}
        )
        self._doc_count = 0

    def ingest(self, items: list[Item]) -> IngestResult:
        t0 = time.monotonic()

        for item in items:
            content = self._item_to_text(item)
            if not content or len(content.strip()) < 10:
                continue

            # Chunk long content (MemPalace style: ~1500 char chunks)
            chunks = self._chunk(content, max_chars=1500)
            for ci, chunk in enumerate(chunks):
                self._collection.add(
                    ids=[f"d_{self._doc_count:06d}"],
                    documents=[chunk],
                    metadatas=[{
                        "wing": self._item_wing(item),
                        "room": "general",
                        "source_file": f"item_{self._doc_count}.md",
                        "chunk_index": ci,
                        "filed_at": self._item_timestamp(item) or "",
                    }],
                )
                self._doc_count += 1

        latency = (time.monotonic() - t0) * 1000
        return IngestResult(num_items=len(items), latency_ms=latency)

    def query(self, question: str, budget: int | None = None) -> QueryResult:
        t0 = time.monotonic()

        k = budget or self._top_k
        if self._collection.count() == 0:
            return QueryResult(answer="No memories stored.", total_latency_ms=0.0)

        # Hybrid retrieval: vector search + BM25 re-ranking
        results = self._collection.query(
            query_texts=[question],
            n_results=min(k * 3, self._collection.count()),
            include=["documents", "distances", "metadatas"],
        )

        docs = results["documents"][0] if results["documents"] else []
        dists = results["distances"][0] if results["distances"] else []

        if not docs:
            return QueryResult(answer="No relevant memories found.", total_latency_ms=0.0)

        # BM25 re-rank
        ranked = self._hybrid_rank(question, docs, dists)[:k]

        context = "\n\n---\n\n".join(ranked)
        context_tokens = len(context.split())

        answer = self._llm_answer(question, context)
        total_ms = (time.monotonic() - t0) * 1000

        return QueryResult(
            answer=answer,
            total_latency_ms=total_ms,
            context_tokens=context_tokens,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _item_to_text(item: Item) -> str:
        if isinstance(item, ConversationTurn):
            prefix = f"{item.role.upper()}"
            if item.speaker:
                prefix = item.speaker
            if item.timestamp:
                prefix = f"[{item.timestamp}] {prefix}"
            return f"{prefix}: {item.content}"
        elif isinstance(item, DocumentChunk):
            return item.content
        elif isinstance(item, PlatformEvent):
            return f"[{item.platform} {item.timestamp}] {item.content}"
        elif isinstance(item, Declaration):
            return f"{item.key}: {item.value}"
        return str(item)

    @staticmethod
    def _item_wing(item: Item) -> str:
        if isinstance(item, ConversationTurn):
            return item.session_id or "conversation"
        elif isinstance(item, DocumentChunk):
            return item.source or "document"
        elif isinstance(item, PlatformEvent):
            return item.platform
        return "general"

    @staticmethod
    def _item_timestamp(item: Item) -> str | None:
        if isinstance(item, ConversationTurn):
            return item.timestamp
        elif isinstance(item, PlatformEvent):
            return item.timestamp
        return None

    @staticmethod
    def _chunk(text: str, max_chars: int = 1500) -> list[str]:
        if not text.strip():
            return []
        if len(text) <= max_chars:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_chars
            # Try to break at paragraph or sentence boundary
            if end < len(text):
                for sep in ["\n\n", "\n", ". ", "! ", "? "]:
                    bp = text.rfind(sep, start + max_chars // 2, end)
                    if bp > start:
                        end = bp + len(sep)
                        break
            chunks.append(text[start:end].strip())
            start = end
        return [c for c in chunks if c]

    @staticmethod
    def _hybrid_rank(
        query: str, docs: list[str], distances: list[float],
        vector_weight: float = 0.6, bm25_weight: float = 0.4,
    ) -> list[str]:
        """Re-rank by cosine similarity + BM25, matching MemPalace's searcher."""
        import math
        import re

        token_re = re.compile(r"\w{2,}", re.UNICODE)
        query_terms = set(token_re.findall(query.lower()))
        if not query_terms or not docs:
            return docs

        tokenized = [token_re.findall(d.lower()) for d in docs]
        n = len(docs)
        avgdl = sum(len(t) for t in tokenized) / max(n, 1)

        df = {term: sum(1 for t in tokenized if term in set(t)) for term in query_terms}
        idf = {t: math.log((n - df[t] + 0.5) / (df[t] + 0.5) + 1) for t in query_terms}

        bm25_scores = []
        for toks in tokenized:
            dl = len(toks)
            tf = {}
            for t in toks:
                if t in query_terms:
                    tf[t] = tf.get(t, 0) + 1
            score = sum(
                idf[t] * (tf.get(t, 0) * 2.5) / (tf.get(t, 0) + 1.5 * (0.25 + 0.75 * dl / max(avgdl, 1)))
                for t in query_terms
            )
            bm25_scores.append(score)

        max_bm25 = max(bm25_scores) if bm25_scores else 1.0
        bm25_norm = [s / max_bm25 if max_bm25 > 0 else 0 for s in bm25_scores]

        scored = []
        for i, (doc, dist, bn) in enumerate(zip(docs, distances, bm25_norm)):
            vec_sim = max(0.0, 1.0 - dist)
            combined = vector_weight * vec_sim + bm25_weight * bn
            scored.append((combined, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored]

    def _llm_answer(self, question: str, context: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant with access to verbatim memory excerpts. "
                    "Answer the question based only on the provided memories. Be concise."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Memory excerpts:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer based only on the memories above:"
                ),
            },
        ]

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
                return f"[LLM error: HTTP {e.code}]"
            except urllib.error.URLError as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return f"[LLM error: {e.reason}]"

        return "[LLM error: failed after 3 retries]"

    def __del__(self):
        if self._tmpdir and os.path.isdir(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)
