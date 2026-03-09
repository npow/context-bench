# context_pipeline.py
# ===================================================================
# THIS FILE IS MUTATED BY THE AUTORESEARCH LOOP.
# The agent edits this file to discover better context strategies.
#
# HOW TO MUTATE:
#   - Simple changes: edit only the STRATEGY dict below.
#   - Radical changes: also edit ContextPipeline methods.
#
# STRATEGY keys and what they do:
#   chunk_by          : how to split turns into retrievable chunks
#                       "turn"    = one chunk per turn (role + content)
#                       "session" = group session_size consecutive turns
#   session_size      : turns per chunk when chunk_by="session"
#   retrieval_k       : how many chunks to include as context
#   retrieval_method  : algorithm used to rank chunks against the query
#                       "bm25"      = keyword overlap (no API calls)
#                       "embedding" = semantic search via relay /v1/embeddings
#                       "hybrid"    = 0.5 * bm25_score + 0.5 * embed_score
#   context_order     : order of retrieved chunks sent to the LLM
#                       "relevant_first"  = highest-ranked chunk first
#                       "chronological"   = earliest turn first
#                       "reverse_chron"   = most-recent turn first
#   compression       : whether/how to shorten chunks before sending
#                       "none"         = send raw text
#                       "summarize"    = ask LLM to summarize each chunk
#                       "key_sentences"= ask LLM to extract key sentences
#   query_expansion   : rewrite the question via LLM before retrieval
#   iterative         : retrieve, reason, then retrieve again (2-pass)
#   max_context_tokens: hard cap (word count proxy) on context sent to LLM
# ===================================================================

from __future__ import annotations

STRATEGY = {
    "chunk_by": "turn",               # "turn" | "session"
    "session_size": 5,                # turns per session if chunk_by="session"
    "retrieval_k": 5,                 # number of chunks to retrieve
    "retrieval_method": "embedding",  # "bm25" | "embedding" | "hybrid"
    "context_order": "relevant_first",  # "relevant_first" | "chronological" | "reverse_chron"
    "compression": "none",            # "none" | "summarize" | "key_sentences"
    "query_expansion": False,         # rewrite question before retrieval
    "iterative": False,               # retrieve -> reason -> retrieve again
    "max_context_tokens": 2000,       # hard cap on context sent to LLM
}

# ===================================================================
# Implementation — edit below for more radical changes
# ===================================================================

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any


# ---------------------------------------------------------------------------
# Optional rank_bm25 import — falls back to simple TF-IDF word overlap
# ---------------------------------------------------------------------------
try:
    from rank_bm25 import BM25Okapi as _BM25Okapi  # type: ignore
    _HAVE_BM25 = True
except ImportError:
    _BM25Okapi = None  # type: ignore
    _HAVE_BM25 = False


# ---------------------------------------------------------------------------
# Local sentence-transformers embedding model
#
# Loaded via context_bench.embeddings so the singleton survives across
# the dynamic module reloads the autoresearch loop performs each iteration.
# ---------------------------------------------------------------------------

def _get_st_model():
    """Return the process-level cached SentenceTransformer."""
    try:
        from context_bench.embeddings import get_model  # type: ignore
        return get_model()
    except ImportError:
        # Fallback if running the pipeline file standalone (outside the package)
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            return SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers not installed. Run: uv sync"
            ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace-split tokenisation used by both BM25 and TF-IDF."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors (plain Python)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _word_count_proxy(text: str) -> int:
    """Cheap proxy for token count: split on whitespace."""
    return len(text.split())


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Hard truncate a string to approximately max_tokens words."""
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens]) + " [truncated]"


# ---------------------------------------------------------------------------
# Simple TF-IDF fallback (stdlib only)
# ---------------------------------------------------------------------------

class _TFIDFIndex:
    """Minimal TF-IDF scorer for BM25 fallback."""

    def __init__(self, corpus: list[list[str]]) -> None:
        self._corpus = corpus
        N = len(corpus)
        # Document frequency
        df: dict[str, int] = {}
        for doc in corpus:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        self._idf: dict[str, float] = {
            term: math.log((N + 1) / (count + 1)) + 1.0
            for term, count in df.items()
        }

    def get_scores(self, query: list[str]) -> list[float]:
        scores = []
        for doc in self._corpus:
            doc_len = len(doc) or 1
            tf: dict[str, float] = {}
            for term in doc:
                tf[term] = tf.get(term, 0.0) + 1.0 / doc_len
            score = sum(tf.get(t, 0.0) * self._idf.get(t, 0.0) for t in query)
            scores.append(score)
        return scores


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ContextPipeline:
    """RAG pipeline implementing the MemorySystem protocol.

    All behaviour is driven by the module-level STRATEGY dict so the
    autoresearch agent can tweak parameters without touching this class.

    Args:
        relay_url: Base URL of the OpenAI-compatible relay (no trailing slash).
        model:     Chat model name forwarded in every /v1/chat/completions call.
        api_key:   Bearer token; falls back to OPENAI_API_KEY env var.
        strategy:  Override STRATEGY dict (defaults to module-level STRATEGY).
        timeout:   HTTP request timeout in seconds.
    """

    def __init__(
        self,
        relay_url: str,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        strategy: dict[str, Any] | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = relay_url.rstrip("/")
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._strategy: dict[str, Any] = dict(STRATEGY)
        if strategy:
            self._strategy.update(strategy)
        self._timeout = timeout

        # State populated by ingest()
        self._chunks: list[str] = []          # raw text of each chunk
        self._chunk_indices: list[int] = []   # original turn index of each chunk
        self._bm25_index: Any = None          # BM25Okapi or _TFIDFIndex
        self._embeddings: list[list[float]] = []  # one vector per chunk

        # Set by query() so the runner can record how many words were actually
        # sent to the final LLM call (context + question), not the full
        # conversation size.  This makes token_efficiency meaningful across
        # pipeline configurations with different retrieval_k / compression.
        self.last_context_tokens: int = 0

    @property
    def name(self) -> str:
        m = self._strategy["retrieval_method"]
        k = self._strategy["retrieval_k"]
        c = self._strategy["chunk_by"]
        return f"context_pipeline_{m}_k{k}_{c}"

    # ------------------------------------------------------------------
    # MemorySystem protocol
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all stored memory. Called before each conversation."""
        self._chunks = []
        self._chunk_indices = []
        self._bm25_index = None
        self._embeddings = []
        self.last_context_tokens = 0

    def ingest(self, turns: list[dict[str, Any]]) -> None:
        """Chunk the conversation history and build the retrieval index.

        Chunking strategy is controlled by STRATEGY["chunk_by"]:
          - "turn":    one chunk per turn, labelled with role
          - "session": group session_size consecutive turns into one chunk
        """
        if not turns:
            return

        strategy = self._strategy

        # --- Build chunks ---
        if strategy["chunk_by"] == "session":
            size = max(1, strategy["session_size"])
            for start in range(0, len(turns), size):
                group = turns[start: start + size]
                text = "\n".join(
                    f"{t['role'].upper()}: {t.get('content', '')}" for t in group
                )
                self._chunks.append(text)
                self._chunk_indices.append(start)
        else:
            # Default: "turn" — one chunk per turn
            for i, turn in enumerate(turns):
                role = turn.get("role", "user").upper()
                content = turn.get("content", "")
                self._chunks.append(f"{role}: {content}")
                self._chunk_indices.append(i)

        # --- Build BM25 / TF-IDF index ---
        tokenized = [_tokenize(c) for c in self._chunks]
        if _HAVE_BM25:
            self._bm25_index = _BM25Okapi(tokenized)
        else:
            self._bm25_index = _TFIDFIndex(tokenized)

        # --- Build embedding index (only if needed) ---
        method = strategy["retrieval_method"]
        if method in ("embedding", "hybrid"):
            self._embeddings = self._embed_batch(self._chunks)

    def query(self, question: str) -> str:
        """Answer a question using stored memory.

        Pipeline:
          1. Optionally expand the query via LLM.
          2. Retrieve top-k chunks.
          3. Optionally compress chunks via LLM.
          4. Order context.
          5. Truncate to max_context_tokens.
          6. Call relay for final answer.
        """
        if not self._chunks:
            return self._chat([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ])

        strategy = self._strategy
        k = strategy["retrieval_k"]

        # Step 1: optional query expansion
        retrieval_query = question
        if strategy["query_expansion"]:
            retrieval_query = self._expand_query(question)

        # Step 2: retrieve
        if strategy["iterative"]:
            retrieved = self._retrieve_iterative(retrieval_query, k)
        else:
            retrieved = self._retrieve(retrieval_query, k)

        # Step 3: optional compression
        if strategy["compression"] != "none":
            retrieved = self._compress(retrieved, question, strategy["compression"])

        # Step 4: ordering
        context_text = self._order_and_format(retrieved, strategy["context_order"])

        # Step 5: truncate
        context_text = _truncate_to_tokens(context_text, strategy["max_context_tokens"])

        # Record how many words are actually sent to the LLM so the runner
        # can use this (not full-conversation size) for token_efficiency.
        self.last_context_tokens = _word_count_proxy(context_text) + _word_count_proxy(question)

        # Step 6: final answer
        system_prompt = (
            "You are a helpful assistant. Answer questions based on the provided "
            "conversation context. Give a short, direct answer — typically 1-10 words. "
            "Do not explain, do not use bullet points, do not use markdown. "
            "Output only the answer itself, nothing else."
        )
        user_content = (
            f"Relevant conversation context:\n{context_text}\n\n"
            f"Question: {question}\n\nAnswer (1-10 words only):"
        )
        return self._chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ])

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve(self, query: str, k: int) -> list[tuple[int, str, float]]:
        """Return top-k (original_index, chunk_text, score) tuples."""
        method = self._strategy["retrieval_method"]
        n = len(self._chunks)
        k = min(k, n)

        if method == "bm25":
            scores = self._bm25_scores(query)
        elif method == "embedding":
            scores = self._embedding_scores(query)
        else:  # hybrid
            bm25 = self._bm25_scores(query)
            emb = self._embedding_scores(query)
            # Normalise each to [0, 1] then blend 50/50
            bm25 = _normalize(bm25)
            emb = _normalize(emb)
            scores = [0.5 * b + 0.5 * e for b, e in zip(bm25, emb)]

        ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
        top_k = ranked[:k]
        return [(self._chunk_indices[i], self._chunks[i], scores[i]) for i in top_k]

    def _retrieve_iterative(
        self, query: str, k: int
    ) -> list[tuple[int, str, float]]:
        """Two-pass retrieval: first pass -> LLM reasoning -> second pass."""
        # First pass: retrieve half the budget
        half_k = max(1, k // 2)
        first_pass = self._retrieve(query, half_k)

        # Ask LLM for an intermediate reasoning step / refined query
        first_context = "\n\n".join(chunk for _, chunk, _ in first_pass)
        reasoning_prompt = [
            {"role": "system", "content": (
                "You are a helpful assistant. Given some context excerpts and a "
                "question, write a short refined search query (one sentence) that "
                "would help find additional relevant information."
            )},
            {"role": "user", "content": (
                f"Context so far:\n{first_context}\n\n"
                f"Original question: {query}\n\n"
                "Refined search query:"
            )},
        ]
        refined_query = self._chat(reasoning_prompt)

        # Second pass with refined query, retrieve remaining budget
        second_pass = self._retrieve(refined_query, k)

        # Merge, deduplicating by original turn index
        seen: set[int] = set()
        merged: list[tuple[int, str, float]] = []
        for item in first_pass + second_pass:
            if item[0] not in seen:
                seen.add(item[0])
                merged.append(item)
        return merged[:k]

    def _bm25_scores(self, query: str) -> list[float]:
        tq = _tokenize(query)
        if self._bm25_index is None:
            return [0.0] * len(self._chunks)
        return list(self._bm25_index.get_scores(tq))

    def _embedding_scores(self, query: str) -> list[float]:
        if not self._embeddings:
            return [0.0] * len(self._chunks)
        query_vec = self._embed_batch([query])[0]
        return [_cosine(query_vec, chunk_vec) for chunk_vec in self._embeddings]

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def _compress(
        self,
        retrieved: list[tuple[int, str, float]],
        question: str,
        mode: str,
    ) -> list[tuple[int, str, float]]:
        """Compress each chunk via LLM and return updated tuples."""
        compressed = []
        for orig_idx, chunk_text, score in retrieved:
            if mode == "summarize":
                prompt = (
                    f"Summarize the following conversation excerpt in 1-2 sentences, "
                    f"keeping information relevant to: {question}\n\n"
                    f"Excerpt:\n{chunk_text}\n\nSummary:"
                )
            else:  # key_sentences
                prompt = (
                    f"Extract the 1-3 most important sentences from this excerpt "
                    f"that are relevant to: {question}\n\n"
                    f"Excerpt:\n{chunk_text}\n\nKey sentences:"
                )
            compressed_text = self._chat([
                {"role": "user", "content": prompt},
            ])
            compressed.append((orig_idx, compressed_text.strip(), score))
        return compressed

    # ------------------------------------------------------------------
    # Context ordering
    # ------------------------------------------------------------------

    def _order_and_format(
        self,
        retrieved: list[tuple[int, str, float]],
        order: str,
    ) -> str:
        """Sort retrieved chunks and join into a single context string."""
        if order == "chronological":
            # Sort by original turn position (ascending)
            ordered = sorted(retrieved, key=lambda x: x[0])
        elif order == "reverse_chron":
            # Sort by original turn position (descending)
            ordered = sorted(retrieved, key=lambda x: x[0], reverse=True)
        else:
            # "relevant_first": keep retrieval ranking order (score descending)
            ordered = retrieved

        return "\n\n---\n\n".join(chunk for _, chunk, _ in ordered)

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using local sentence-transformers model (all-MiniLM-L6-v2).

        The model is loaded once at module level and reused across conversations,
        so the ~80 MB download only happens on first call.
        """
        if not texts:
            return []
        model = _get_st_model()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

    # ------------------------------------------------------------------
    # Query expansion helper
    # ------------------------------------------------------------------

    def _expand_query(self, question: str) -> str:
        """Rewrite the question to be more retrieval-friendly."""
        prompt = (
            "Rewrite the following question into a concise search query that "
            "captures its key concepts. Return only the rewritten query, no "
            "explanation.\n\n"
            f"Question: {question}\n\nSearch query:"
        )
        expanded = self._chat([{"role": "user", "content": prompt}])
        return expanded.strip() or question

    # ------------------------------------------------------------------
    # HTTP helper — relay calls
    # ------------------------------------------------------------------

    def _chat(self, messages: list[dict[str, Any]]) -> str:
        """POST to /v1/chat/completions via urllib. Retries on transient errors."""
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
                raise RuntimeError(f"Chat HTTP {e.code}: {e.reason}") from e
            except urllib.error.URLError as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Chat connection error: {e.reason}") from e

        raise RuntimeError("Chat request failed after 3 retries")


# ---------------------------------------------------------------------------
# Utility: min-max normalise a list of floats to [0, 1]
# ---------------------------------------------------------------------------

def _normalize(scores: list[float]) -> list[float]:
    lo, hi = min(scores, default=0.0), max(scores, default=0.0)
    if hi == lo:
        return [0.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]
