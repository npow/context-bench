# lean_pipeline.py
# ===================================================================
# Lean retrieval pipeline for LongMemEval-S.
#
# Design: ZERO LLM calls during ingest. All LLM calls at query time.
# This keeps evaluation fast (~2 min per conversation).
#
# Architecture:
# - Ingest: TF-IDF index over turns (pure Python, no LLM)
# - Query: retrieve top-K turns -> pack context -> single sonnet call
#
# ===================================================================

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EXPAND_CONTEXT = 3           # turns to include around each hit
_RETRIEVAL_K = 25             # top-K turns before expansion
_MAX_CONTEXT_CHARS = 25000    # guard against over-long prompts


class ContextPipeline:
    """Lean retrieval pipeline for LongMemEval-S.

    Zero LLM calls during ingest. Single sonnet call per query.
    """

    def __init__(
        self,
        relay_url: str,
        model: str = "sonnet",
        api_key: str | None = None,
        strategy: dict[str, Any] | None = None,
        timeout: float = 180.0,
    ) -> None:
        self._base_url = relay_url.rstrip("/")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model
        self._timeout = timeout

        self._turns: list[dict] = []
        self._index: dict[str, list[int]] = {}   # word -> [turn_idx]
        self._idf: dict[str, float] = {}
        self.last_context_tokens: int = 0

    @property
    def name(self) -> str:
        return "lean_retrieval_v1"

    def reset(self) -> None:
        self._turns = []
        self._index = {}
        self._idf = {}
        self.last_context_tokens = 0

    # ------------------------------------------------------------------
    # Stopwords
    # ------------------------------------------------------------------

    _STOPWORDS = frozenset({
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves", "he", "him",
        "his", "himself", "she", "her", "hers", "herself", "it", "its",
        "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "this", "that", "these", "those",
        "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having", "do", "does", "did", "doing",
        "a", "an", "the", "and", "but", "if", "or", "because", "as",
        "until", "while", "of", "at", "by", "for", "with", "about",
        "against", "between", "through", "during", "before", "after",
        "above", "below", "to", "from", "up", "down", "in", "out",
        "on", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all",
        "both", "each", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than", "too",
        "very", "s", "t", "can", "will", "just", "don", "should", "now",
        "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn",
        "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn",
        "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won",
        "wouldn", "also", "would", "could", "shall", "might", "must",
        "need", "may", "let", "like", "know", "think", "want", "tell",
        "said", "say", "one", "two", "yes", "no", "oh", "well",
        "really", "actually", "just", "thing", "things", "sure",
    })

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-z][a-z0-9']*", text.lower())
        return [t for t in tokens if t not in self._STOPWORDS and len(t) > 1]

    # ------------------------------------------------------------------
    # Ingest — pure Python, no LLM calls
    # ------------------------------------------------------------------

    def ingest(self, turns: list[dict[str, Any]]) -> None:
        self._turns = turns
        n = len(turns)

        # Build inverted index
        index: dict[str, list[int]] = defaultdict(list)
        doc_freq: dict[str, int] = defaultdict(int)

        for idx, turn in enumerate(turns):
            content = turn.get("content", "") or ""
            role = turn.get("role", "user")
            text = f"{role}: {content}"
            tokens = set(self._tokenize(text))
            for token in tokens:
                index[token].append(idx)
                doc_freq[token] += 1

        # Compute IDF weights
        self._idf = {}
        for word, df in doc_freq.items():
            self._idf[word] = math.log((n + 1) / (df + 1)) + 1

        self._index = dict(index)

    # ------------------------------------------------------------------
    # Retrieval — TF-IDF scoring
    # ------------------------------------------------------------------

    def _retrieve(self, query: str, top_k: int = _RETRIEVAL_K) -> list[int]:
        tokens = self._tokenize(query)
        if not tokens:
            return list(range(min(top_k, len(self._turns))))

        scores: dict[int, float] = defaultdict(float)
        for token in tokens:
            if token in self._index:
                weight = self._idf.get(token, 1.0)
                for idx in self._index[token]:
                    scores[idx] += weight

        if not scores:
            return list(range(min(top_k, len(self._turns))))

        ranked = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        return ranked[:top_k]

    def _expand(self, indices: list[int]) -> list[int]:
        """Expand each hit with surrounding context turns."""
        expanded: set[int] = set()
        n = len(self._turns)
        for idx in indices:
            for offset in range(-_EXPAND_CONTEXT, _EXPAND_CONTEXT + 1):
                neighbor = idx + offset
                if 0 <= neighbor < n:
                    expanded.add(neighbor)
        return sorted(expanded)

    def _format_turns(self, indices: list[int]) -> str:
        """Format selected turns as context text."""
        lines: list[str] = []
        total_chars = 0
        prev = -10

        for idx in indices:
            if total_chars > _MAX_CONTEXT_CHARS:
                break
            if prev >= 0 and idx > prev + 1:
                lines.append("...")

            turn = self._turns[idx]
            role = turn.get("role", "user").upper()
            content = (turn.get("content", "") or "")[:500]
            line = f"[T{idx}] {role}: {content}"
            lines.append(line)
            total_chars += len(line)
            prev = idx

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Query — single LLM call
    # ------------------------------------------------------------------

    def query(self, question: str) -> str:
        if not self._turns:
            return self._chat([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ])

        # Retrieve relevant turns
        hits = self._retrieve(question, top_k=_RETRIEVAL_K)
        expanded = self._expand(hits)
        context = self._format_turns(expanded)

        self.last_context_tokens = len(context.split()) + len(question.split())

        # Detect question type for targeted hints
        q_lower = question.lower()
        is_temporal = any(kw in q_lower for kw in (
            "when", "what year", "what date", "how long",
            "which year", "what month", "first time", "last time",
        ))
        is_knowledge_update = any(kw in q_lower for kw in (
            "current", "now", "latest", "recently", "changed",
            "updated", "new", "switched", "moved",
        ))

        if is_temporal:
            hint = (
                "This is a TEMPORAL question about when something happened. "
                "Look at the turn indices [T###] to determine ordering. "
                "Higher T numbers = later in conversation. "
                "Give a specific answer about timing/ordering."
            )
        elif is_knowledge_update:
            hint = (
                "This question asks about the CURRENT/LATEST state of something. "
                "The conversation may contain updates — look for the HIGHEST "
                "turn index [T###] that mentions the topic, as that's the most recent info."
            )
        else:
            hint = (
                "Give a SHORT, DIRECT answer (usually 1-15 words). "
                "Do not explain your reasoning."
            )

        system_prompt = (
            "You are a precise question-answering assistant. You answer questions "
            "based ONLY on the conversation excerpts provided below.\n\n"
            f"{hint}\n\n"
            "Rules:\n"
            "- Answer ONLY from the provided excerpts.\n"
            "- If a person's name is mentioned, use their full name.\n"
            "- Do NOT make up information not in the excerpts.\n"
            "- If the answer is truly not found, say 'unknown'.\n"
            "- Output ONLY the answer — no preamble, no explanation."
        )

        user_content = (
            f"## CONVERSATION EXCERPTS\n{context}\n\n"
            f"---\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        return self._chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ])

    # ------------------------------------------------------------------
    # HTTP with retry
    # ------------------------------------------------------------------

    def _chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
    ) -> str:
        url = f"{self._base_url}/v1/chat/completions"
        model_name = model or self._model
        body = json.dumps({"model": model_name, "messages": messages}).encode()
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        last_err: Exception | None = None
        for attempt in range(5):
            req = urllib.request.Request(
                url, data=body, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    data = json.loads(resp.read().decode())
                    return data["choices"][0]["message"]["content"].strip()
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                    wait = min(60, 10 * (2 ** attempt))
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Chat HTTP {e.code}: {e.reason}") from e
            except urllib.error.URLError as e:
                last_err = e
                if attempt < 4:
                    time.sleep(10 * (2 ** attempt))
                    continue
                raise RuntimeError(f"Chat connection error: {e.reason}") from e
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                last_err = e
                if attempt < 4:
                    time.sleep(5)
                    continue
                raise RuntimeError(f"Chat parse error: {e}") from e

        raise RuntimeError(f"Chat request failed after 5 retries: {last_err}")
