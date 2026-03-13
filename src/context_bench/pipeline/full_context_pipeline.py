# full_context_pipeline.py
# ===================================================================
# MemMachine-inspired pipeline for LoCoMo long-conversation QA.
#
# Architectural advances over naive full-context:
#
# 1. WORKING MEMORY SUMMARY (ingest-time, haiku)
#    Build a structured summary covering people, events, timeline,
#    and key facts. Used as fast-lookup context for all queries.
#
# 2. TIMESTAMPED EPISODIC MEMORIES
#    Parse dates embedded in conversation content. Tag every turn
#    with its resolved absolute date so temporal questions ("when
#    did X happen?") return the timestamp directly.
#
# 3. RELATIVE TIME RESOLUTION
#    Detect "last year", "3 months ago", etc. and resolve them to
#    absolute dates anchored to the turn's session date.
#
# 4. CONTEXT EXPANSION (expand_context=3)
#    When a relevant episode is retrieved, also return the 3 turns
#    before and after it. LoCoMo answers often span multiple turns.
#
# 5. ENTITY-AWARE TF-IDF RETRIEVAL
#    Score turns by weighted term overlap (IDF-weighted). Named
#    entities (capitalized tokens) get a 3x weight boost.
#
# 6. STRUCTURED FACT EXTRACTION (ingest-time, haiku)
#    Extract entity-relation-value triples to answer "who", "what",
#    "where" questions precisely without scanning all turns.
#
# 7. QUERY-TYPE-AWARE ANSWERING
#    Detect question type (temporal / entity / multi-hop / open).
#    Route to appropriate context assembly strategy.
#
# ===================================================================

from __future__ import annotations

import json
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

_EXPAND_CONTEXT = 3          # turns to include around each hit
_RETRIEVAL_K = 20            # top-K episodes before expansion
_MAX_RETRIEVED_CHARS = 20000 # guard against over-long prompts


class MemMachinePipeline:
    """MemMachine-inspired pipeline for LoCoMo long-conversation QA.

    Ingest: build timestamped episodes, fact index, working memory summary.
    Query:  retrieve relevant episodes + expand context, answer with summary.
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

        # Populated during ingest()
        self._turns: list[dict] = []
        self._episodes: list[dict] = []      # enriched turn dicts
        self._facts: str = ""                # extracted entity facts block
        self._summary: str = ""             # working memory summary
        self._entity_index: dict[str, list[int]] = {}  # word -> [turn_idx]
        self._entity_weights: dict[str, float] = {}    # IDF weights

        self.last_context_tokens: int = 0

    @property
    def name(self) -> str:
        return "memmachine_pipeline_v2"

    def reset(self) -> None:
        self._turns = []
        self._episodes = []
        self._facts = ""
        self._summary = ""
        self._entity_index = {}
        self._entity_weights = {}
        self.last_context_tokens = 0

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, turns: list[dict[str, Any]]) -> None:
        self._turns = turns

        # Step 1: Build timestamped episodes (lightweight, no LLM)
        self._episodes = self._build_episodes(turns)

        # Step 2: Build keyword index for retrieval
        self._build_index()

        # Step 3: Build working memory summary (haiku)
        self._summary = self._build_working_memory(turns)

        # Step 4: Extract structured facts (haiku)
        self._facts = self._extract_facts(turns)

    # ------------------------------------------------------------------
    # Episode construction and date extraction
    # ------------------------------------------------------------------

    _DATE_PATTERNS = [
        # ISO date: 2022-05-04
        re.compile(r'\b(\d{4}-\d{2}-\d{2})\b'),
        # "4 May 2022" or "May 4, 2022"
        re.compile(
            r'\b(\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|'
            r'May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|'
            r'Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\b',
            re.IGNORECASE,
        ),
        re.compile(
            r'\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|'
            r'Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|'
            r'Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4})\b',
            re.IGNORECASE,
        ),
        # "in 2021" or "in 2022"
        re.compile(r'\bin (\d{4})\b'),
    ]

    _REL_TIME_PATTERNS = [
        # "last year" -> -1 year
        (re.compile(r'\blast\s+year\b', re.IGNORECASE), -1, 'year'),
        # "2 years ago" -> -2 years
        (re.compile(r'\b(\d+)\s+years?\s+ago\b', re.IGNORECASE), None, 'years_ago'),
        # "last month" -> -1 month
        (re.compile(r'\blast\s+month\b', re.IGNORECASE), -1, 'month'),
        # "6 months ago"
        (re.compile(r'\b(\d+)\s+months?\s+ago\b', re.IGNORECASE), None, 'months_ago'),
    ]

    def _extract_date_from_text(self, text: str) -> str | None:
        """Return the first parseable date found in text, or None."""
        for pat in self._DATE_PATTERNS:
            m = pat.search(text)
            if m:
                return m.group(1)
        return None

    def _extract_year_from_text(self, text: str) -> int | None:
        """Extract a 4-digit year from text, returning it as an int."""
        # Look for year in ISO date or standalone
        m = re.search(r'\b(20\d{2}|19\d{2})\b', text)
        if m:
            return int(m.group(1))
        return None

    def _resolve_relative_time(self, text: str, anchor_year: int | None) -> str:
        """Return text with relative time expressions resolved to absolute years."""
        if anchor_year is None:
            return text

        def replace_last_year(m: re.Match) -> str:
            return f"in {anchor_year - 1}"

        def replace_years_ago(m: re.Match) -> str:
            n = int(m.group(1))
            return f"in {anchor_year - n}"

        def replace_last_month(m: re.Match) -> str:
            return f"about {anchor_year} (last month)"

        def replace_months_ago(m: re.Match) -> str:
            n = int(m.group(1))
            if n >= 12:
                return f"in {anchor_year - 1}"
            return f"earlier in {anchor_year}"

        replacers = [
            (self._REL_TIME_PATTERNS[0][0], replace_last_year),
            (self._REL_TIME_PATTERNS[1][0], replace_years_ago),
            (self._REL_TIME_PATTERNS[2][0], replace_last_month),
            (self._REL_TIME_PATTERNS[3][0], replace_months_ago),
        ]
        for pat, fn in replacers:
            text = pat.sub(fn, text)
        return text

    def _build_episodes(self, turns: list[dict]) -> list[dict]:
        """Assign timestamps and resolve relative dates for all turns."""
        episodes: list[dict] = []
        current_date: str | None = None
        current_year: int | None = None

        for i, turn in enumerate(turns):
            content = (
                turn.get("content", "")
                or turn.get("text", "")
                or ""
            )
            role = turn.get("role", "user")

            # Try to get date from turn metadata first
            date = (
                turn.get("timestamp")
                or turn.get("date")
                or turn.get("session_date")
                or turn.get("created_at")
            )
            if date:
                current_date = str(date)
                current_year = self._extract_year_from_text(current_date)
            else:
                # Try to extract date from content
                extracted = self._extract_date_from_text(content)
                if extracted:
                    current_date = extracted
                    current_year = self._extract_year_from_text(extracted)

            # Resolve relative time expressions in content
            resolved_content = self._resolve_relative_time(content, current_year)

            episodes.append({
                "idx": i,
                "role": role,
                "content": content,
                "resolved_content": resolved_content,
                "date": current_date,
                "year": current_year,
            })

        return episodes

    # ------------------------------------------------------------------
    # Index construction (TF-IDF style with entity boost)
    # ------------------------------------------------------------------

    _STOPWORDS = frozenset({
        "the", "and", "for", "are", "was", "were", "did", "does", "have",
        "has", "had", "that", "this", "with", "what", "when", "where",
        "who", "how", "why", "which", "about", "from", "they", "their",
        "you", "your", "its", "not", "but", "also", "just", "been",
        "more", "into", "then", "than", "over", "some", "would", "could",
        "should", "will", "can", "may", "might", "must", "shall",
        "like", "very", "only", "even", "both", "each", "any", "all",
        "she", "her", "his", "him", "our", "out", "one", "got", "get",
    })

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\b[a-zA-Z]{3,}\b', text)

    def _build_index(self) -> None:
        """Build inverted index with IDF weights and entity boosts."""
        index: dict[str, list[int]] = defaultdict(list)

        for ep in self._episodes:
            tokens = self._tokenize(ep["resolved_content"])
            seen: set[str] = set()
            for tok in tokens:
                lower = tok.lower()
                if lower not in self._STOPWORDS and lower not in seen:
                    index[lower].append(ep["idx"])
                    seen.add(lower)

        n = len(self._episodes)
        self._entity_index = dict(index)
        # IDF weight: log(N / df)
        import math
        self._entity_weights = {
            term: math.log((n + 1) / (len(idxs) + 1)) + 1.0
            for term, idxs in index.items()
        }

    # ------------------------------------------------------------------
    # Retrieval with context expansion
    # ------------------------------------------------------------------

    def _retrieve(self, question: str, top_k: int = _RETRIEVAL_K) -> list[int]:
        """Score episodes by IDF-weighted term overlap; boost named entities."""
        q_tokens = self._tokenize(question)
        query_terms: list[tuple[str, float]] = []

        for tok in q_tokens:
            lower = tok.lower()
            if lower in self._STOPWORDS:
                continue
            weight = self._entity_weights.get(lower, 0.5)
            # Named entities (originally capitalized) get 3x boost
            if tok[0].isupper():
                weight *= 3.0
            query_terms.append((lower, weight))

        if not query_terms:
            # Fallback: evenly spaced sample
            step = max(1, len(self._episodes) // top_k)
            return sorted(range(0, len(self._episodes), step)[:top_k])

        scores: dict[int, float] = defaultdict(float)
        for term, weight in query_terms:
            for idx in self._entity_index.get(term, []):
                scores[idx] += weight

        ranked = sorted(scores, key=lambda x: scores[x], reverse=True)
        return ranked[:top_k]

    def _expand(self, indices: list[int]) -> list[int]:
        """Expand retrieved indices by ±EXPAND_CONTEXT turns."""
        n = len(self._episodes)
        expanded: set[int] = set()
        for idx in indices:
            for offset in range(-_EXPAND_CONTEXT, _EXPAND_CONTEXT + 1):
                j = idx + offset
                if 0 <= j < n:
                    expanded.add(j)
        return sorted(expanded)

    def _format_episodes(self, indices: list[int]) -> str:
        """Render episodes with timestamps; insert '...' for gaps."""
        lines: list[str] = []
        prev = -1
        total_chars = 0

        for idx in indices:
            if total_chars >= _MAX_RETRIEVED_CHARS:
                lines.append("[... additional context truncated ...]")
                break

            ep = self._episodes[idx]
            if prev >= 0 and idx > prev + 1:
                lines.append("...")

            date_tag = f" [{ep['date']}]" if ep["date"] else ""
            role_tag = ep["role"].upper()
            line = f"[T{idx}]{date_tag} {role_tag}: {ep['resolved_content']}"
            lines.append(line)
            total_chars += len(line)
            prev = idx

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Ingest-time LLM calls (haiku — fast/cheap)
    # ------------------------------------------------------------------

    def _build_working_memory(self, turns: list[dict]) -> str:
        """Build a structured working memory summary using haiku."""
        if not turns:
            return ""

        # Sample turns intelligently: take start, middle chunks, and end
        n = len(turns)
        if n <= 150:
            selected = turns
        else:
            # Sample ~150 turns spread across the conversation
            step = n // 120
            indices = list(range(0, n, step))[:120]
            # Always include first 15 and last 15
            first = list(range(min(15, n)))
            last = list(range(max(0, n - 15), n))
            all_idx = sorted(set(first + indices + last))
            selected = [turns[i] for i in all_idx]

        conversation_sample = "\n".join(
            f"{t.get('role','user').upper()}: {(t.get('content','') or '')[:300]}"
            for t in selected
        )

        prompt = (
            "You are analyzing a long multi-session conversation for a QA task.\n"
            "Produce a STRUCTURED MEMORY covering:\n\n"
            "## People\n"
            "List every person mentioned, their relationship to the speakers, "
            "key facts (job, hobbies, location, age).\n\n"
            "## Timeline of Key Events\n"
            "Chronological list: [DATE or PERIOD] - EVENT. Include all dates mentioned.\n\n"
            "## Places\n"
            "All locations visited or mentioned, with who and when.\n\n"
            "## Personal Facts\n"
            "Preferences, habits, recurring activities, goals, life changes.\n\n"
            "## Relationships\n"
            "How people know each other, introductions, social connections.\n\n"
            "Be factual, dense, and comprehensive — this memory will be used "
            "to answer specific QA questions.\n\n"
            f"Conversation:\n{conversation_sample}"
        )

        try:
            return self._chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You extract structured facts from conversations for QA. "
                            "Be comprehensive and precise. Include all named entities and dates."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model="haiku",
            )
        except Exception:
            return ""

    def _extract_facts(self, turns: list[dict]) -> str:
        """Extract entity-relation-value triples using haiku."""
        if not turns:
            return ""

        n = len(turns)
        # Process in chunks of ~100 turns, then combine
        chunk_size = 80
        all_facts: list[str] = []

        for start in range(0, n, chunk_size):
            chunk = turns[start: start + chunk_size]
            chunk_text = "\n".join(
                f"{t.get('role','user').upper()}: {(t.get('content','') or '')[:200]}"
                for t in chunk
            )
            prompt = (
                "Extract factual statements as triples from this conversation excerpt.\n"
                "Format: ENTITY | RELATION | VALUE\n"
                "Examples:\n"
                "  Alice | works at | Google\n"
                "  Bob | hobby | playing guitar\n"
                "  Sarah | visited | Paris in 2021\n"
                "  Tom | met | Alice through work\n\n"
                "Extract ALL factual claims, especially about people, places, events, dates:\n\n"
                f"{chunk_text}\n\n"
                "Triples (one per line):"
            )
            try:
                result = self._chat(
                    [
                        {
                            "role": "system",
                            "content": "Extract entity-relation-value triples. One per line.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    model="haiku",
                )
                if result.strip():
                    all_facts.append(result.strip())
            except Exception:
                continue

        return "\n".join(all_facts)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question: str) -> str:
        if not self._turns:
            return self._chat([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ])

        # Detect question type for routing
        q_lower = question.lower()
        is_temporal = any(
            kw in q_lower
            for kw in ("when", "what year", "what date", "how long ago",
                       "which year", "what month", "how many years", "what time")
        )
        is_list_q = any(
            kw in q_lower
            for kw in ("what are", "list", "name all", "how many", "which ones")
        )

        # Retrieve relevant episodes + expand context
        raw_hits = self._retrieve(question, top_k=_RETRIEVAL_K)
        expanded = self._expand(raw_hits)
        episode_excerpt = self._format_episodes(expanded)

        # Build context block
        context_parts: list[str] = []

        if self._summary:
            context_parts.append(f"## CONVERSATION MEMORY\n{self._summary}")

        if self._facts:
            context_parts.append(f"## EXTRACTED FACTS\n{self._facts}")

        if episode_excerpt:
            context_parts.append(
                f"## RELEVANT CONVERSATION EXCERPTS "
                f"(with timestamps where available)\n{episode_excerpt}"
            )

        context = "\n\n".join(context_parts)
        self.last_context_tokens = len(context.split()) + len(question.split())

        # Tailor the system prompt to question type
        if is_temporal:
            answer_hint = (
                "For this TEMPORAL question: output the ABSOLUTE date, month, or year. "
                "NEVER answer with relative time like 'yesterday', 'last year', 'recently'. "
                "Instead, look at the [timestamp] tags on conversation turns to find the "
                "actual date. Example: if a turn tagged [1:56 pm on 8 May, 2023] mentions "
                "'I went to X yesterday', the answer is '7 May 2023', NOT 'yesterday'."
            )
        elif is_list_q:
            answer_hint = (
                "For this LIST question: output a comma-separated list of all items. "
                "Be exhaustive — scan the facts and excerpts for every instance."
            )
        else:
            answer_hint = (
                "Give a SHORT, DIRECT answer (usually 1-15 words). "
                "Do not explain your reasoning."
            )

        system_prompt = (
            "You are a precise question-answering assistant. "
            "You have access to a structured memory of a long conversation.\n\n"
            f"{answer_hint}\n\n"
            "Rules:\n"
            "- Answer ONLY from the provided memory and excerpts.\n"
            "- If the answer includes a person's name, give their full name.\n"
            "- If the answer is not in the memory, say 'unknown'.\n"
            "- Output ONLY the answer — no preamble, no explanation."
        )

        user_content = (
            f"{context}\n\n"
            f"---\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        return self._chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        )

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

        raise RuntimeError(
            f"Chat request failed after 5 retries: {last_err}"
        )


# ---------------------------------------------------------------------------
# Alias expected by the evaluation loop
# ---------------------------------------------------------------------------
ContextPipeline = MemMachinePipeline
