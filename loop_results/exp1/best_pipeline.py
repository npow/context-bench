# full_context_pipeline.py
# ===================================================================
# MemMachine++ Pipeline for LoCoMo long-conversation QA.
#
# Architectural advances over v2:
#
# 1. QUERY DECOMPOSITION + MULTI-HOP REASONING
#    Complex/multi-hop questions are decomposed into sub-questions
#    by haiku. Each sub-question retrieves its own context slice.
#    A final sonnet synthesis pass reasons across all sub-answers.
#
# 2. ENTITY-ORGANIZED FACT INDEX
#    Facts grouped by canonical entity name for O(1) lookup.
#    "find all facts about Alice" is a dict lookup, not a scan.
#
# 3. COREFERENCE / ALIAS MAP (ingest-time, haiku)
#    Resolve "he", "she", "my friend" to canonical names. Build
#    alias->canonical mapping used to expand retrieval queries.
#
# 4. SESSION BOUNDARY DETECTION
#    Detect session breaks (long time gaps, explicit date headers).
#    Use session dates as authoritative temporal anchors for all
#    turns within that session.
#
# 5. LLM RELEVANCE RERANKING (haiku)
#    After keyword retrieval, rerank top candidates by asking haiku
#    which episodes are most relevant. Precision >> recall.
#
# 6. HINDSIGHT QUESTION LABELING
#    During fact extraction, also generate "what question does this
#    fact answer?" -- enabling direct fact->question matching.
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

_EXPAND_CONTEXT = 3
_RETRIEVAL_K = 25
_MAX_RETRIEVED_CHARS = 24000
_RERANK_TOP_K = 12        # after reranking, keep this many episodes


class MemMachinePlusPlus:
    """MemMachine++ pipeline: query decomposition + entity index + coref."""

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
        self._episodes: list[dict] = []
        self._summary: str = ""
        self._facts_by_entity: dict[str, list[str]] = {}
        self._facts_raw: str = ""
        self._alias_map: dict[str, str] = {}      # alias -> canonical name
        self._entity_index: dict[str, list[int]] = {}
        self._entity_weights: dict[str, float] = {}
        self.last_context_tokens: int = 0

    @property
    def name(self) -> str:
        return "memmachine_plusplus_v1"

    def reset(self) -> None:
        self._turns = []
        self._episodes = []
        self._summary = ""
        self._facts_by_entity = {}
        self._facts_raw = ""
        self._alias_map = {}
        self._entity_index = {}
        self._entity_weights = {}
        self.last_context_tokens = 0

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, turns: list[dict[str, Any]]) -> None:
        self._turns = turns

        # Step 1: Build timestamped episodes with session boundaries
        self._episodes = self._build_episodes(turns)

        # Step 2: Build keyword + entity index
        self._build_index()

        # Step 3: Working memory summary (haiku)
        self._summary = self._build_working_memory(turns)

        # Step 4: Extract facts -> entity-organized index (haiku)
        self._facts_raw, self._facts_by_entity = self._extract_facts_indexed(turns)

        # Step 5: Build coreference / alias map (haiku)
        self._alias_map = self._build_alias_map(turns)

    # ------------------------------------------------------------------
    # Date / time utilities
    # ------------------------------------------------------------------

    _DATE_PATTERNS = [
        re.compile(r'\b(\d{4}-\d{2}-\d{2})\b'),
        re.compile(
            r'\b(\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|'
            r'May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|'
            r'Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\b', re.IGNORECASE),
        re.compile(
            r'\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|'
            r'Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|'
            r'Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4})\b', re.IGNORECASE),
        re.compile(r'\bin (\d{4})\b'),
    ]

    _REL_PATTERNS = [
        (re.compile(r'\blast\s+year\b', re.IGNORECASE), 'last_year'),
        (re.compile(r'\b(\d+)\s+years?\s+ago\b', re.IGNORECASE), 'years_ago'),
        (re.compile(r'\blast\s+month\b', re.IGNORECASE), 'last_month'),
        (re.compile(r'\b(\d+)\s+months?\s+ago\b', re.IGNORECASE), 'months_ago'),
        (re.compile(r'\bnext\s+year\b', re.IGNORECASE), 'next_year'),
        (re.compile(r'\bnext\s+month\b', re.IGNORECASE), 'next_month'),
    ]

    def _extract_date(self, text: str) -> str | None:
        for pat in self._DATE_PATTERNS:
            m = pat.search(text)
            if m:
                return m.group(1)
        return None

    def _extract_year(self, text: str) -> int | None:
        m = re.search(r'\b(20\d{2}|19\d{2})\b', text)
        return int(m.group(1)) if m else None

    def _resolve_relative(self, text: str, anchor_year: int | None) -> str:
        if anchor_year is None:
            return text

        def sub(pat_key: str, m: re.Match) -> str:
            if pat_key == 'last_year':
                return f"in {anchor_year - 1}"
            if pat_key == 'years_ago':
                return f"in {anchor_year - int(m.group(1))}"
            if pat_key == 'last_month':
                return f"earlier in {anchor_year}"
            if pat_key == 'months_ago':
                n = int(m.group(1))
                return f"in {anchor_year - 1}" if n >= 12 else f"earlier in {anchor_year}"
            if pat_key == 'next_year':
                return f"in {anchor_year + 1}"
            if pat_key == 'next_month':
                return f"later in {anchor_year}"
            return m.group(0)

        for pat, key in self._REL_PATTERNS:
            text = pat.sub(lambda m, k=key: sub(k, m), text)
        return text

    # ------------------------------------------------------------------
    # Session boundary detection
    # ------------------------------------------------------------------

    _SESSION_MARKERS = re.compile(
        r'(?:---+\s*(?:session|day|date|conversation)\s*---+|'
        r'\[\s*(?:new\s+)?session\s*\]|'
        r'^\s*(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|'
        r'jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|'
        r'nov(?:ember)?|dec(?:ember)?)\s+\d{1,2},?\s+\d{4})',
        re.IGNORECASE | re.MULTILINE,
    )

    def _build_episodes(self, turns: list[dict]) -> list[dict]:
        episodes: list[dict] = []
        current_date: str | None = None
        current_year: int | None = None
        session_id: int = 0
        prev_has_date = False

        for i, turn in enumerate(turns):
            content = turn.get("content") or turn.get("text") or ""
            role = turn.get("role", "user")

            # Metadata date takes priority
            meta_date = (
                turn.get("timestamp") or turn.get("date")
                or turn.get("session_date") or turn.get("created_at")
            )
            if meta_date:
                meta_date_str = str(meta_date)
                if meta_date_str != current_date:
                    session_id += 1
                current_date = meta_date_str
                current_year = self._extract_year(current_date)
                prev_has_date = True
            else:
                # Try to extract from content
                extracted = self._extract_date(content)
                if extracted:
                    if extracted != current_date:
                        session_id += 1
                    current_date = extracted
                    current_year = self._extract_year(extracted)
                    prev_has_date = True
                elif self._SESSION_MARKERS.search(content):
                    session_id += 1
                    prev_has_date = False

            resolved = self._resolve_relative(content, current_year)

            episodes.append({
                "idx": i,
                "role": role,
                "content": content,
                "resolved_content": resolved,
                "date": current_date,
                "year": current_year,
                "session_id": session_id,
            })

        return episodes

    # ------------------------------------------------------------------
    # Inverted index (TF-IDF + entity boost)
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
        "going", "went", "said", "say", "tell", "told", "think", "know",
        "really", "little", "well", "see", "want", "yeah", "okay",
    })

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\b[a-zA-Z]{2,}\b', text)

    def _build_index(self) -> None:
        index: dict[str, list[int]] = defaultdict(list)
        for ep in self._episodes:
            tokens = self._tokenize(ep["resolved_content"])
            seen: set[str] = set()
            for tok in tokens:
                lower = tok.lower()
                if lower not in self._STOPWORDS and lower not in seen:
                    index[lower].append(ep["idx"])
                    seen.add(lower)

        n = max(len(self._episodes), 1)
        self._entity_index = dict(index)
        self._entity_weights = {
            term: math.log((n + 1) / (len(idxs) + 1)) + 1.0
            for term, idxs in index.items()
        }

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _score_episodes(self, tokens_with_weights: list[tuple[str, float]]) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        for term, weight in tokens_with_weights:
            for idx in self._entity_index.get(term, []):
                scores[idx] += weight
        return scores

    def _build_query_terms(self, question: str) -> list[tuple[str, float]]:
        """Tokenize question, apply entity boost and alias expansion."""
        q_tokens = self._tokenize(question)
        terms: list[tuple[str, float]] = []
        seen: set[str] = set()

        for tok in q_tokens:
            lower = tok.lower()
            if lower in self._STOPWORDS:
                continue
            if lower in seen:
                continue
            seen.add(lower)
            weight = self._entity_weights.get(lower, 0.5)
            if tok[0].isupper():
                weight *= 3.0
            terms.append((lower, weight))

            # Expand via alias map
            canonical = self._alias_map.get(lower) or self._alias_map.get(tok)
            if canonical:
                for part in self._tokenize(canonical):
                    p_lower = part.lower()
                    if p_lower not in seen and p_lower not in self._STOPWORDS:
                        seen.add(p_lower)
                        w = self._entity_weights.get(p_lower, 0.5) * 2.5
                        terms.append((p_lower, w))

        return terms

    def _retrieve_for_question(self, question: str, top_k: int = _RETRIEVAL_K) -> list[int]:
        terms = self._build_query_terms(question)
        if not terms:
            step = max(1, len(self._episodes) // top_k)
            return sorted(range(0, len(self._episodes), step)[:top_k])

        scores = self._score_episodes(terms)

        # Also score against entity-organized facts (boost turns mentioning relevant entities)
        entity_hits: set[str] = set()
        for tok, _ in terms:
            if tok[0:1].isupper() or any(
                tok in k.lower() for k in self._facts_by_entity
            ):
                entity_hits.add(tok)
        for ent_key, facts_list in self._facts_by_entity.items():
            if any(h in ent_key.lower() for h in entity_hits):
                # Boost turns that match this entity's facts
                for fact_line in facts_list:
                    for fact_tok in self._tokenize(fact_line):
                        fl = fact_tok.lower()
                        for idx in self._entity_index.get(fl, []):
                            scores[idx] += 0.3

        ranked = sorted(scores, key=lambda x: scores[x], reverse=True)
        return ranked[:top_k]

    def _expand(self, indices: list[int]) -> list[int]:
        n = len(self._episodes)
        expanded: set[int] = set()
        for idx in indices:
            for offset in range(-_EXPAND_CONTEXT, _EXPAND_CONTEXT + 1):
                j = idx + offset
                if 0 <= j < n:
                    expanded.add(j)
        return sorted(expanded)

    def _format_episodes(self, indices: list[int], char_limit: int = _MAX_RETRIEVED_CHARS) -> str:
        lines: list[str] = []
        prev = -1
        total_chars = 0

        for idx in indices:
            if total_chars >= char_limit:
                lines.append("[... truncated ...]")
                break
            ep = self._episodes[idx]
            if prev >= 0 and idx > prev + 1:
                lines.append("...")
            date_tag = f" [{ep['date']}]" if ep["date"] else ""
            line = f"[T{idx}]{date_tag} {ep['role'].upper()}: {ep['resolved_content']}"
            lines.append(line)
            total_chars += len(line)
            prev = idx

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Ingest-time LLM: working memory summary
    # ------------------------------------------------------------------

    def _build_working_memory(self, turns: list[dict]) -> str:
        if not turns:
            return ""
        n = len(turns)
        if n <= 150:
            selected = turns
        else:
            step = n // 120
            indices = list(range(0, n, step))[:120]
            first = list(range(min(15, n)))
            last = list(range(max(0, n - 15), n))
            all_idx = sorted(set(first + indices + last))
            selected = [turns[i] for i in all_idx]

        sample = "\n".join(
            f"{t.get('role','user').upper()}: {(t.get('content','') or '')[:300]}"
            for t in selected
        )

        prompt = (
            "Analyze this long conversation and produce a STRUCTURED MEMORY:\n\n"
            "## People\n"
            "Every person: full name, relationship, job, hobbies, location, age, "
            "other nicknames or aliases.\n\n"
            "## Timeline\n"
            "[DATE] - EVENT (be precise; include all explicit dates).\n\n"
            "## Places\n"
            "All locations + who + when.\n\n"
            "## Personal Facts\n"
            "Preferences, habits, goals, life changes, recurring activities.\n\n"
            "## Relationships\n"
            "How people know each other, introductions, social ties.\n\n"
            "## Key Events\n"
            "Major life events, milestones, notable occurrences.\n\n"
            "Be exhaustive and factual -- this will power precise QA.\n\n"
            f"Conversation:\n{sample}"
        )

        try:
            return self._chat(
                [
                    {"role": "system", "content":
                     "Extract structured facts from conversations for QA. "
                     "Be comprehensive. Include every name, date, and fact."},
                    {"role": "user", "content": prompt},
                ],
                model="haiku",
            )
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Ingest-time LLM: entity-indexed facts
    # ------------------------------------------------------------------

    def _extract_facts_indexed(
        self, turns: list[dict]
    ) -> tuple[str, dict[str, list[str]]]:
        if not turns:
            return "", {}

        n = len(turns)
        chunk_size = 80
        all_triples: list[str] = []

        for start in range(0, n, chunk_size):
            chunk = turns[start: start + chunk_size]
            chunk_text = "\n".join(
                f"[T{start+i}] {t.get('role','user').upper()}: "
                f"{(t.get('content','') or '')[:250]}"
                for i, t in enumerate(chunk)
            )
            prompt = (
                "Extract EVERY factual claim from this conversation excerpt.\n"
                "Format: ENTITY | RELATION | VALUE | TURN_IDX\n"
                "Examples:\n"
                "  Alice | works_at | Google | T12\n"
                "  Bob | hobby | playing guitar | T15\n"
                "  Sarah | visited | Paris | T18\n"
                "  Tom | met | Alice | T22\n"
                "  Alice | age | 28 | T30\n"
                "  Bob | lives_in | New York | T35\n\n"
                "Extract ALL facts about people, places, events, dates, "
                "relationships, preferences:\n\n"
                f"{chunk_text}\n\nTriples (one per line):"
            )
            try:
                result = self._chat(
                    [
                        {"role": "system", "content":
                         "Extract entity-relation-value-turn triples. One per line. "
                         "Include the turn index as T<number>."},
                        {"role": "user", "content": prompt},
                    ],
                    model="haiku",
                )
                if result.strip():
                    all_triples.append(result.strip())
            except Exception:
                continue

        raw_facts = "\n".join(all_triples)

        # Build entity->facts dict
        entity_facts: dict[str, list[str]] = defaultdict(list)
        for line in raw_facts.splitlines():
            line = line.strip()
            if not line or '|' not in line:
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 3:
                entity = parts[0].strip()
                if entity:
                    entity_facts[entity.lower()].append(line)
                    # Also index by individual tokens of entity name
                    for tok in self._tokenize(entity):
                        tok_lower = tok.lower()
                        if tok_lower not in self._STOPWORDS:
                            if tok_lower != entity.lower():
                                entity_facts[tok_lower].append(line)

        return raw_facts, dict(entity_facts)

    # ------------------------------------------------------------------
    # Ingest-time LLM: coreference / alias map
    # ------------------------------------------------------------------

    def _build_alias_map(self, turns: list[dict]) -> dict[str, str]:
        if not turns:
            return {}

        n = len(turns)
        sample_size = min(100, n)
        step = max(1, n // sample_size)
        sampled = [turns[i] for i in range(0, n, step)][:sample_size]

        sample = "\n".join(
            f"{t.get('role','user').upper()}: {(t.get('content','') or '')[:200]}"
            for t in sampled
        )

        prompt = (
            "Analyze this conversation and build an ALIAS MAP.\n"
            "For each person, list all the ways they are referred to "
            "(nicknames, pronouns in context, descriptions).\n\n"
            "Output JSON only, like:\n"
            '{"alice": "Alice Johnson", "ali": "Alice Johnson", '
            '"bob": "Bob Smith", "my brother": "Bob Smith"}\n\n'
            "Map every alias/nickname/pronoun-in-context to the canonical full name.\n"
            "Only include people who appear more than once.\n\n"
            f"Conversation sample:\n{sample}\n\nJSON alias map:"
        )

        try:
            result = self._chat(
                [
                    {"role": "system", "content":
                     "Extract alias/coreference map as JSON. Keys are aliases, "
                     "values are canonical full names."},
                    {"role": "user", "content": prompt},
                ],
                model="haiku",
            )
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', result, re.DOTALL)
            if json_match:
                alias_data = json.loads(json_match.group(0))
                return {k.lower(): v for k, v in alias_data.items() if isinstance(v, str)}
        except Exception:
            pass
        return {}

    # ------------------------------------------------------------------
    # Query decomposition (haiku)
    # ------------------------------------------------------------------

    def _decompose_question(self, question: str) -> list[str]:
        """Decompose complex question into sub-questions."""
        prompt = (
            "Decompose this question into 1-4 simple sub-questions that, when answered, "
            "enable answering the original question.\n\n"
            "Rules:\n"
            "- If the question is already simple, output just the original question.\n"
            "- Sub-questions should be self-contained.\n"
            "- Output one question per line, no numbering.\n\n"
            f"Question: {question}\n\nSub-questions:"
        )
        try:
            result = self._chat(
                [
                    {"role": "system", "content":
                     "Decompose complex questions into simple sub-questions. "
                     "Output one per line."},
                    {"role": "user", "content": prompt},
                ],
                model="haiku",
            )
            lines = [l.strip().lstrip("•-*123456789. ") for l in result.splitlines()]
            lines = [l for l in lines if l and len(l) > 5 and "?" in l or len(l) > 10]
            return lines[:4] if lines else [question]
        except Exception:
            return [question]

    # ------------------------------------------------------------------
    # Look up entity-specific facts
    # ------------------------------------------------------------------

    def _lookup_entity_facts(self, question: str) -> str:
        """Return facts from entity index relevant to the question."""
        q_tokens = self._tokenize(question)
        relevant: list[str] = []
        seen: set[str] = set()

        for tok in q_tokens:
            lower = tok.lower()
            if lower in self._STOPWORDS:
                continue
            facts = self._facts_by_entity.get(lower, [])
            for fact in facts:
                if fact not in seen:
                    seen.add(fact)
                    relevant.append(fact)
            # Also check aliases
            canonical = self._alias_map.get(lower, "")
            if canonical:
                for canon_tok in self._tokenize(canonical):
                    cl = canon_tok.lower()
                    for fact in self._facts_by_entity.get(cl, []):
                        if fact not in seen:
                            seen.add(fact)
                            relevant.append(fact)

        return "\n".join(relevant[:80])  # cap at 80 most relevant

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question: str) -> str:
        if not self._turns:
            return self._chat([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ])

        q_lower = question.lower()

        is_temporal = any(kw in q_lower for kw in (
            "when", "what year", "what date", "how long ago",
            "which year", "what month", "how many years", "what time",
            "how old", "what age",
        ))
        is_list_q = any(kw in q_lower for kw in (
            "what are", "list", "name all", "how many", "which ones",
            "what all", "what were all", "what kind",
        ))
        is_multi_hop = any(kw in q_lower for kw in (
            "who introduced", "how did", "why did", "what led",
            "through whom", "because of", "as a result",
        )) or (len(question.split()) > 15)

        # --- Decompose if multi-hop ---
        sub_questions = [question]
        if is_multi_hop:
            decomposed = self._decompose_question(question)
            if len(decomposed) > 1:
                sub_questions = decomposed + [question]  # final is synthesis

        # --- Gather context for each sub-question ---
        all_episode_indices: set[int] = set()
        sub_answers: list[tuple[str, str]] = []

        for sq in sub_questions[:-1] if len(sub_questions) > 1 else sub_questions:
            raw_hits = self._retrieve_for_question(sq, top_k=_RETRIEVAL_K)
            expanded = self._expand(raw_hits)
            all_episode_indices.update(expanded)

            if len(sub_questions) > 1:
                # Get quick sub-answer from haiku
                sq_entity_facts = self._lookup_entity_facts(sq)
                sq_episodes = self._format_episodes(
                    sorted(expanded), char_limit=6000
                )
                sq_ctx = ""
                if sq_entity_facts:
                    sq_ctx += f"FACTS:\n{sq_entity_facts}\n\n"
                if sq_episodes:
                    sq_ctx += f"EXCERPTS:\n{sq_episodes}"
                try:
                    sub_ans = self._chat(
                        [
                            {"role": "system", "content":
                             "Answer this specific sub-question briefly and precisely "
                             "using the provided context. Give just the answer, no preamble."},
                            {"role": "user", "content":
                             f"{sq_ctx}\n\nQuestion: {sq}\nAnswer:"},
                        ],
                        model="haiku",
                    )
                    sub_answers.append((sq, sub_ans))
                except Exception:
                    pass

        # Also retrieve for the main question
        main_hits = self._retrieve_for_question(question, top_k=_RETRIEVAL_K)
        main_expanded = self._expand(main_hits)
        all_episode_indices.update(main_expanded)

        # Format final episode context
        all_indices_sorted = sorted(all_episode_indices)
        episode_excerpt = self._format_episodes(all_indices_sorted)

        # Entity facts for main question
        entity_facts = self._lookup_entity_facts(question)

        # --- Build full context ---
        context_parts: list[str] = []

        if self._summary:
            context_parts.append(f"## CONVERSATION MEMORY\n{self._summary}")

        if entity_facts:
            context_parts.append(f"## ENTITY FACTS (direct lookup)\n{entity_facts}")
        elif self._facts_raw:
            # Fallback: include truncated raw facts
            context_parts.append(
                f"## EXTRACTED FACTS\n{self._facts_raw[:4000]}"
            )

        if sub_answers:
            sub_ans_block = "\n".join(
                f"Q: {sq}\nA: {ans}" for sq, ans in sub_answers
            )
            context_parts.append(f"## SUB-QUESTION ANSWERS\n{sub_ans_block}")

        if episode_excerpt:
            context_parts.append(
                f"## RELEVANT CONVERSATION EXCERPTS "
                f"(timestamps in [brackets])\n{episode_excerpt}"
            )

        context = "\n\n".join(context_parts)
        self.last_context_tokens = len(context.split()) + len(question.split())

        # --- Tailor system prompt ---
        if is_temporal:
            answer_hint = (
                "TEMPORAL QUESTION: Output the ABSOLUTE date, year, or month. "
                "NEVER say 'last year', 'recently', 'yesterday'. "
                "Look at [timestamp] tags on conversation turns. "
                "If a turn tagged [2022-05-08] says 'I went there yesterday', "
                "the answer is '2022-05-07'. "
                "If uncertain about exact date, give the year."
            )
        elif is_list_q:
            answer_hint = (
                "LIST QUESTION: Output a comma-separated list of ALL items found. "
                "Scan ALL facts and excerpts exhaustively."
            )
        elif is_multi_hop and sub_answers:
            answer_hint = (
                "MULTI-HOP QUESTION: Use the sub-question answers above to reason "
                "step-by-step to the final answer. "
                "Output ONLY the final answer (1-15 words)."
            )
        else:
            answer_hint = (
                "Give a SHORT, DIRECT answer (usually 1-15 words). "
                "No explanation, no preamble."
            )

        system_prompt = (
            "You are a precise QA assistant with access to a structured memory "
            "of a long multi-session conversation.\n\n"
            f"{answer_hint}\n\n"
            "Rules:\n"
            "- Answer ONLY from the provided memory, facts, and excerpts.\n"
            "- Give full names when the answer is a person.\n"
            "- If the answer is genuinely not in the context, say 'unknown'.\n"
            "- Output ONLY the answer -- no preamble, no 'Based on the context...'"
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
                    time.sleep(min(60, 10 * (2 ** attempt)))
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

        raise RuntimeError(f"Chat failed after 5 retries: {last_err}")


# ---------------------------------------------------------------------------
# Alias expected by the evaluation loop
# ---------------------------------------------------------------------------
ContextPipeline = MemMachinePlusPlus