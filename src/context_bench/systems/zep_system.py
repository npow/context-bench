"""Zep/Graphiti memory system wrapper.

Uses graphiti-core to build a temporal knowledge graph from conversation turns,
then queries the graph for relevant facts to answer questions.

If graphiti-core is not installed or Neo4j is unreachable, falls back to a
simple in-memory keyword/TF-IDF search approach so tests can always run.

Requires (optional): pip install context-bench[zep]
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Graphiti import attempt
# ---------------------------------------------------------------------------

def _try_import_graphiti() -> Any:
    """Return the graphiti_core module, or None if unavailable."""
    try:
        import graphiti_core  # noqa: F401
        return graphiti_core
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Fallback: in-memory keyword search
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lower-case, split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _tfidf_scores(query_tokens: list[str], corpus: list[list[str]]) -> list[float]:
    """Return a TF-IDF cosine-style score for each document in *corpus*."""
    n = len(corpus)
    if n == 0:
        return []

    # Document frequency per token
    df: dict[str, int] = {}
    for doc in corpus:
        for tok in set(doc):
            df[tok] = df.get(tok, 0) + 1

    scores: list[float] = []
    for doc in corpus:
        if not doc:
            scores.append(0.0)
            continue
        doc_len = len(doc)
        tf: dict[str, float] = {}
        for tok in doc:
            tf[tok] = tf.get(tok, 0.0) + 1.0 / doc_len

        score = 0.0
        for tok in query_tokens:
            if tok in tf:
                idf = math.log((n + 1) / (df.get(tok, 0) + 1)) + 1.0
                score += tf[tok] * idf
        scores.append(score)

    return scores


class _FallbackStore:
    """Simple in-memory store with keyword/TF-IDF retrieval."""

    def __init__(self) -> None:
        self._chunks: list[str] = []
        self._tokenized: list[list[str]] = []

    def reset(self) -> None:
        self._chunks = []
        self._tokenized = []

    def add(self, text: str) -> None:
        self._chunks.append(text)
        self._tokenized.append(_tokenize(text))

    def search(self, query: str, top_k: int = 10) -> list[str]:
        if not self._chunks:
            return []
        query_tokens = _tokenize(query)
        scores = _tfidf_scores(query_tokens, self._tokenized)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._chunks[i] for i in ranked[:top_k] if scores[i] > 0]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ZepSystem:
    """Graphiti/Zep-backed memory system implementing the MemorySystem protocol.

    Attempts to use graphiti-core with a Neo4j backend. If graphiti-core is
    not installed, or if the Neo4j connection fails on first use, it
    transparently falls back to a lightweight in-memory keyword-search store.

    All LLM calls (both graphiti internals and the final answer) are routed
    through *relay_url*, which must expose an OpenAI-compatible endpoint.

    Args:
        relay_url: OpenAI-compatible relay URL (e.g. "http://localhost:7878").
        model: Model name for LLM calls.
        api_key: Bearer token. Falls back to OPENAI_API_KEY env var.
        neo4j_uri: Neo4j bolt URI. Defaults to "bolt://localhost:7687".
        neo4j_user: Neo4j username. Defaults to "neo4j".
        neo4j_password: Neo4j password. Defaults to "password".
        timeout: HTTP timeout in seconds for the final answer call.
    """

    def __init__(
        self,
        relay_url: str,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        timeout: float = 60.0,
    ) -> None:
        self._relay_url = relay_url.rstrip("/")
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password
        self._timeout = timeout

        # Conversation isolation
        self._group_id: str = str(uuid.uuid4())

        # Will be resolved in _ensure_backend()
        self._graphiti: Any = None       # Graphiti instance or None
        self._fallback: _FallbackStore | None = None
        self._using_fallback: bool = False
        self._backend_ready: bool = False

    # ------------------------------------------------------------------
    # MemorySystem protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "zep_fallback" if self._using_fallback else "zep"

    def reset(self) -> None:
        """Clear stored memory and prepare a fresh conversation namespace."""
        self._group_id = str(uuid.uuid4())
        self._backend_ready = False
        self._graphiti = None
        self._fallback = None
        self._using_fallback = False

    def ingest(self, turns: list[dict[str, Any]]) -> None:
        """Store conversation turns in the knowledge graph (or fallback store)."""
        self._ensure_backend()

        if self._using_fallback:
            assert self._fallback is not None
            for turn in turns:
                role = turn.get("role", "unknown")
                content = turn.get("content", "")
                self._fallback.add(f"{role}: {content}")
        else:
            self._graphiti_ingest(turns)

    def query(self, question: str) -> str:
        """Retrieve relevant facts and answer *question* via the relay."""
        self._ensure_backend()

        if self._using_fallback:
            assert self._fallback is not None
            results = self._fallback.search(question, top_k=10)
            context = "\n".join(f"- {r}" for r in results) if results else "(no relevant context found)"
        else:
            context = self._graphiti_search(question)

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
                    f"Relevant memory:\n{context}\n\n"
                    f"Question: {question}"
                ),
            },
        ]
        return self._chat(messages)

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def _ensure_backend(self) -> None:
        """Lazily initialise graphiti or fall back to in-memory store."""
        if self._backend_ready:
            return

        graphiti_core = _try_import_graphiti()
        if graphiti_core is None:
            self._init_fallback()
            return

        try:
            self._init_graphiti(graphiti_core)
        except Exception:
            # Any connection / import error -> silently fall back
            self._init_fallback()

    def _init_fallback(self) -> None:
        self._fallback = _FallbackStore()
        self._using_fallback = True
        self._backend_ready = True

    def _init_graphiti(self, graphiti_core: Any) -> None:
        """Initialise a Graphiti instance pointed at the relay for LLM/embed."""
        import asyncio

        from graphiti_core import Graphiti
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_client import OpenAIClient

        llm_config = LLMConfig(
            api_key=self._api_key or "relay",
            model=self._model,
            base_url=f"{self._relay_url}/v1",
        )
        llm_client = OpenAIClient(config=llm_config)

        embedder_config = OpenAIEmbedderConfig(
            api_key=self._api_key or "relay",
            base_url=f"{self._relay_url}/v1",
        )
        embedder = OpenAIEmbedder(config=embedder_config)

        graphiti_instance = Graphiti(
            uri=self._neo4j_uri,
            user=self._neo4j_user,
            password=self._neo4j_password,
            llm_client=llm_client,
            embedder=embedder,
        )

        # Probe connectivity by running build_indices_and_constraints.
        # This is a coroutine; run it synchronously.
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(graphiti_instance.build_indices_and_constraints())
        finally:
            loop.close()

        self._graphiti = graphiti_instance
        self._using_fallback = False
        self._backend_ready = True

    # ------------------------------------------------------------------
    # Graphiti operations (only called when _using_fallback is False)
    # ------------------------------------------------------------------

    def _graphiti_ingest(self, turns: list[dict[str, Any]]) -> None:
        """Add each conversation turn as an episode to the knowledge graph."""
        import asyncio
        from graphiti_core.nodes import EpisodeType

        async def _add_all() -> None:
            for i, turn in enumerate(turns):
                role = turn.get("role", "unknown")
                content = turn.get("content", "")
                episode_body = f"{role}: {content}"
                await self._graphiti.add_episode(
                    name=f"turn_{i}",
                    episode_body=episode_body,
                    source_description="conversation turn",
                    reference_time=datetime.now(tz=timezone.utc),
                    source=EpisodeType.message,
                    group_id=self._group_id,
                )

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_add_all())
        finally:
            loop.close()

    def _graphiti_search(self, question: str) -> str:
        """Search the knowledge graph and return a formatted context string."""
        import asyncio

        async def _search() -> list[Any]:
            return await self._graphiti.search(
                query=question,
                group_ids=[self._group_id],
                num_results=10,
            )

        loop = asyncio.new_event_loop()
        try:
            edges = loop.run_until_complete(_search())
        finally:
            loop.close()

        if not edges:
            return "(no relevant context found)"

        facts = []
        for edge in edges:
            fact = getattr(edge, "fact", None)
            if fact:
                facts.append(f"- {fact}")

        return "\n".join(facts) if facts else "(no relevant context found)"

    # ------------------------------------------------------------------
    # HTTP helper
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
