# entity_pipeline.py
# ===================================================================
# THIS FILE IS MUTATED BY THE AUTORESEARCH LOOP.
# The agent edits this file to discover better context strategies.
#
# HOW TO MUTATE:
#   - Simple changes: edit only the STRATEGY dict below.
#   - Radical changes: also edit EntityMemoryPipeline methods.
#
# STRATEGY keys and what they do:
#   extraction_model  : model used to extract facts during ingest
#                       "haiku" = fast/cheap, "sonnet" = higher quality
#   session_size      : turns grouped together for fact extraction
#   facts_per_session : max facts extracted per session chunk
#   retrieval_k       : how many facts to retrieve per query
#   context_order     : order of retrieved facts sent to LLM
#                       "relevant_first" | "chronological"
#   multi_hop         : whether to do a second retrieval pass using
#                       entities mentioned in first-pass facts
#   max_context_tokens: hard cap (word count proxy) on context sent to LLM
# ===================================================================

from __future__ import annotations

STRATEGY = {
    "extraction_model": "haiku",      # "haiku" | "sonnet"
    "session_size": 20,               # turns per extraction chunk
    "facts_per_session": 8,           # max facts extracted per chunk
    "retrieval_k": 10,                # facts to retrieve per query
    "context_order": "relevant_first",  # "relevant_first" | "chronological"
    "multi_hop": False,               # second retrieval pass on entities
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


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _word_count_proxy(text: str) -> int:
    return len(text.split())


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens]) + " [truncated]"


def _get_st_model():
    """Return the process-level cached SentenceTransformer."""
    try:
        from context_bench.embeddings import get_model  # type: ignore
        return get_model()
    except ImportError:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            return SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError as exc:
            raise RuntimeError("sentence-transformers not installed. Run: uv sync") from exc


class EntityMemoryPipeline:
    """Structured fact-extraction pipeline.

    During ingest: groups turns into sessions, calls an LLM to extract
    concise facts (subject-predicate-object style), and embeds each fact
    for semantic retrieval.

    During query: embeds the question, retrieves top-k relevant facts,
    optionally does a second hop for entities mentioned in those facts,
    then sends the assembled facts to the answer model.

    This approaches SOTA memory architectures (A-MEM, LiCoMemory) by
    storing structured facts rather than raw conversation chunks.
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
        self._answer_model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._strategy: dict[str, Any] = dict(STRATEGY)
        if strategy:
            self._strategy.update(strategy)
        self._timeout = timeout

        # State populated by ingest()
        self._facts: list[str] = []               # extracted fact strings
        self._fact_turn_idx: list[int] = []       # source turn index per fact
        self._fact_embeddings: list[list[float]] = []

        self.last_context_tokens: int = 0

    @property
    def name(self) -> str:
        k = self._strategy["retrieval_k"]
        ss = self._strategy["session_size"]
        mh = "multihop" if self._strategy["multi_hop"] else "singlehop"
        return f"entity_pipeline_k{k}_s{ss}_{mh}"

    def reset(self) -> None:
        self._facts = []
        self._fact_turn_idx = []
        self._fact_embeddings = []
        self.last_context_tokens = 0

    def ingest(self, turns: list[dict[str, Any]]) -> None:
        """Extract structured facts from conversation turns.

        Groups turns into sessions of session_size, calls the extraction
        model to produce up to facts_per_session facts per session, then
        embeds all facts for semantic retrieval.
        """
        if not turns:
            return

        strategy = self._strategy
        session_size = max(1, strategy["session_size"])
        facts_per_session = max(1, strategy["facts_per_session"])
        extraction_model = strategy["extraction_model"]

        for start in range(0, len(turns), session_size):
            group = turns[start: start + session_size]
            session_text = "\n".join(
                f"{t.get('role','user').upper()}: {t.get('content','')}"
                for t in group
            )

            facts = self._extract_facts(session_text, facts_per_session, extraction_model)
            for fact in facts:
                self._facts.append(fact)
                self._fact_turn_idx.append(start)

        if self._facts:
            model = _get_st_model()
            embeddings = model.encode(self._facts, convert_to_numpy=True, show_progress_bar=False)
            self._fact_embeddings = [e.tolist() for e in embeddings]

    def query(self, question: str) -> str:
        if not self._facts:
            return self._chat(self._answer_model, [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ])

        strategy = self._strategy
        k = strategy["retrieval_k"]

        # Retrieve top-k relevant facts
        retrieved_facts = self._retrieve_facts(question, k)

        # Optional multi-hop: extract entities from retrieved facts and
        # do a second retrieval pass to pull in connected facts
        if strategy["multi_hop"] and retrieved_facts:
            entities = self._extract_entities(retrieved_facts)
            if entities:
                hop2 = self._retrieve_facts(entities, k // 2 or 1)
                # Merge, deduplicating
                seen = set(retrieved_facts)
                for f in hop2:
                    if f not in seen:
                        retrieved_facts.append(f)
                        seen.add(f)

        # Order
        if strategy["context_order"] == "chronological":
            # Re-sort by source turn index
            paired = list(zip(retrieved_facts, self._fact_turn_idx[:len(retrieved_facts)]))
            paired.sort(key=lambda x: x[1])
            ordered_facts = [f for f, _ in paired]
        else:
            ordered_facts = retrieved_facts  # relevance order

        context_text = "\n".join(f"• {f}" for f in ordered_facts)
        context_text = _truncate_to_tokens(context_text, strategy["max_context_tokens"])

        self.last_context_tokens = _word_count_proxy(context_text) + _word_count_proxy(question)

        system_prompt = (
            "You are a helpful assistant. Answer questions based on the provided "
            "facts extracted from a conversation. Give a short, direct answer — "
            "typically 1-10 words. Do not explain, do not use bullet points, "
            "do not use markdown. Output only the answer itself, nothing else."
        )
        user_content = (
            f"Relevant facts from the conversation:\n{context_text}\n\n"
            f"Question: {question}\n\nAnswer (1-10 words only):"
        )
        return self._chat(self._answer_model, [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ])

    # ------------------------------------------------------------------
    # Fact extraction
    # ------------------------------------------------------------------

    def _extract_facts(self, session_text: str, n: int, model: str) -> list[str]:
        """Call LLM to extract up to n concise facts from a session."""
        prompt = (
            f"Extract up to {n} concise facts from this conversation excerpt. "
            f"Each fact should be a single sentence capturing a specific piece of "
            f"information (who did what, when, where, preferences, relationships). "
            f"Output one fact per line, no numbering, no preamble.\n\n"
            f"Conversation:\n{session_text}\n\nFacts:"
        )
        response = self._chat(model, [{"role": "user", "content": prompt}])
        facts = [line.strip("•- ").strip() for line in response.splitlines() if line.strip()]
        return facts[:n]

    def _extract_entities(self, facts: list[str]) -> str:
        """Extract key entity names from a list of facts for multi-hop retrieval."""
        combined = " ".join(facts)
        # Simple regex: capitalised words likely to be proper nouns/names
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', combined)
        unique = list(dict.fromkeys(entities))[:5]
        return " ".join(unique)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve_facts(self, query: str, k: int) -> list[str]:
        if not self._fact_embeddings:
            return []
        model = _get_st_model()
        query_vec = model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0].tolist()
        scores = [_cosine(query_vec, fv) for fv in self._fact_embeddings]
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._facts[i] for i in top_k]

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------

    def _chat(self, model: str, messages: list[dict[str, Any]]) -> str:
        url = f"{self._base_url}/v1/chat/completions"
        body = json.dumps({"model": model, "messages": messages}).encode()
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

# Loop expects ContextPipeline class name
ContextPipeline = EntityMemoryPipeline
