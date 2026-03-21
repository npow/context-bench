# full_context_pipeline.py
# ===================================================================
# MemMachine++ pipeline for LoCoMo long-conversation QA.
#
# Architectural advances over MemMachine v2:
#
# 1. LLM QUERY ANALYSIS (haiku, infer-time)
#    Before retrieval, use haiku to extract named entities, classify
#    question type precisely, generate expanded search terms, and
#    identify whether multi-hop reasoning is needed.
#
# 2. ENTITY-CENTRIC MEMORY (ingest-time, haiku)
#    Build a per-entity fact dictionary: entity -> list of facts.
#    Enables O(1) lookup for "what is Alice's job?" style questions
#    by surfacing all facts about the queried person directly.
#
# 3. MULTI-PASS RETRIEVAL (entity + TF-IDF merged)
#    Pass 1: For each entity extracted from the question, retrieve
#            turns that mention that entity (high precision).
#    Pass 2: Standard IDF-weighted TF-IDF retrieval.
#    Merge both pass results, deduplicated, before context expansion.
#
# 4. QUERY DECOMPOSITION + SUB-QUESTION ANSWERING
#    For multi-hop questions, decompose into up to 3 sub-questions,
#    run each through retrieval independently, collect mini-answers,
#    then synthesize the final answer from all evidence.
#
# 5. WORKING MEMORY SUMMARY + FACT TRIPLES (MemMachine baseline)
#    Retained: structured summary covering people/places/events,
#    entity-relation-value triples, timestamped episodes, relative
#    time resolution, context expansion.
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
_RETRIEVAL_K = 20             # top-K episodes before expansion
_MAX_RETRIEVED_CHARS = 22000  # guard against over-long prompts
_ENTITY_RETRIEVAL_K = 10      # per-entity top-K hits
_MAX_SUB_QUESTIONS = 3        # max sub-questions for decomposition


class MemMachinePipeline:
    """MemMachine++ pipeline for LoCoMo long-conversation QA.

    Ingest: build timestamped episodes, entity-centric memory, fact index,
            working memory summary.
    Query:  LLM query analysis -> multi-pass retrieval -> optional sub-question
            decomposition -> context assembly -> answer synthesis.
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
        self._episodes: list[dict] = []        # enriched turn dicts
        self._facts: str = ""                   # extracted entity facts block (triples)
        self._summary: str = ""                 # working memory summary
        self._entity_memory: dict[str, list[str]] = {}  # entity -> [facts]
        self._entity_index: dict[str, list[int]] = {}   # word -> [turn_idx]
        self._entity_weights: dict[str, float] = {}     # IDF weights
        self.last_context_tokens: int = 0

    @property
    def name(self) -> str:
        return "memmachine_pp_v3"

    def reset(self) -> None:
        self._turns = []
        self._episodes = []
        self._facts = ""
        self._summary = ""
        self._entity_memory = {}
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

        # Step 4: Extract structured facts (haiku) -- also populates entity_memory
        self._facts = self._extract_facts(turns)

    # ------------------------------------------------------------------
    # Episode construction and date extraction
    # ------------------------------------------------------------------

    _DATE_PATTERNS = [
        re.compile(r'\b(\d{4}-\d{2}-\d{2})\b'),
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
        re.compile(r'\bin (\d{4})\b'),
    ]

    _REL_TIME_PATTERNS = [
        (re.compile(r'\blast\s+year\b', re.IGNORECASE), -1, 'year'),
        (re.compile(r'\b(\d+)\s+years?\s+ago\b', re.IGNORECASE), None, 'years_ago'),
        (re.compile(r'\blast\s+month\b', re.IGNORECASE), -1, 'month'),
        (re.compile(r'\b(\d+)\s+months?\s+ago\b', re.IGNORECASE), None, 'months_ago'),
        (re.compile(r'\bnext\s+year\b', re.IGNORECASE), 1, 'year'),
        (re.compile(r'\b(\d+)\s+years?\s+later\b', re.IGNORECASE), None, 'years_later'),
    ]

    def _extract_date_from_text(self, text: str) -> str | None:
        for pat in self._DATE_PATTERNS:
            m = pat.search(text)
            if m:
                return m.group(1)
        return None

    def _extract_year_from_text(self, text: str) -> int | None:
        m = re.search(r'\b(20\d{2}|19\d{2})\b', text)
        if m:
            return int(m.group(1))
        return None

    def _resolve_relative_time(self, text: str, anchor_year: int | None) -> str:
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

        def replace_next_year(m: re.Match) -> str:
            return f"in {anchor_year + 1}"

        def replace_years_later(m: re.Match) -> str:
            n = int(m.group(1))
            return f"in {anchor_year + n}"

        replacers = [
            (self._REL_TIME_PATTERNS[0][0], replace_last_year),
            (self._REL_TIME_PATTERNS[1][0], replace_years_ago),
            (self._REL_TIME_PATTERNS[2][0], replace_last_month),
            (self._REL_TIME_PATTERNS[3][0], replace_months_ago),
            (self._REL_TIME_PATTERNS[4][0], replace_next_year),
            (self._REL_TIME_PATTERNS[5][0], replace_years_later),
        ]
        for pat, fn in replacers:
            text = pat.sub(fn, text)
        return text

    def _build_episodes(self, turns: list[dict]) -> list[dict]:
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
                extracted = self._extract_date_from_text(content)
                if extracted:
                    current_date = extracted
                    current_year = self._extract_year_from_text(extracted)

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
        "said", "say", "tell", "told", "ask", "asked", "think", "know",
        "going", "went", "come", "came", "make", "made", "see", "saw",
    })

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\b[a-zA-Z]{3,}\b', text)

    def _build_index(self) -> None:
        index: dict[str, list[int]] = defaultdict(list)

        for ep in self._episodes:
            combined = ep["resolved_content"] + " " + ep["content"]
            tokens = self._tokenize(combined)
            seen: set[str] = set()
            for tok in tokens:
                lower = tok.lower()
                if lower not in self._STOPWORDS and lower not in seen:
                    index[lower].append(ep["idx"])
                    seen.add(lower)

        n = len(self._episodes)
        self._entity_index = dict(index)
        self._entity_weights = {
            term: math.log((n + 1) / (len(idxs) + 1)) + 1.0
            for term, idxs in index.items()
        }

    # ------------------------------------------------------------------
    # Retrieval with context expansion
    # ------------------------------------------------------------------

    def _score_episodes(self, query_terms: list[tuple[str, float]]) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        for term, weight in query_terms:
            for idx in self._entity_index.get(term, []):
                scores[idx] += weight
        return scores

    def _tokenize_query(self, question: str, entity_boost: float = 3.0) -> list[tuple[str, float]]:
        q_tokens = self._tokenize(question)
        query_terms: list[tuple[str, float]] = []
        for tok in q_tokens:
            lower = tok.lower()
            if lower in self._STOPWORDS:
                continue
            weight = self._entity_weights.get(lower, 0.5)
            if tok[0].isupper():
                weight *= entity_boost
            query_terms.append((lower, weight))
        return query_terms

    def _retrieve(self, question: str, top_k: int = _RETRIEVAL_K) -> list[int]:
        query_terms = self._tokenize_query(question)
        if not query_terms:
            step = max(1, len(self._episodes) // top_k)
            return sorted(range(0, len(self._episodes), step)[:top_k])

        scores = self._score_episodes(query_terms)
        ranked = sorted(scores, key=lambda x: scores[x], reverse=True)
        return ranked[:top_k]

    def _retrieve_for_entity(self, entity_name: str) -> list[int]:
        query_terms = self._tokenize_query(entity_name, entity_boost=5.0)
        if not query_terms:
            return []
        scores = self._score_episodes(query_terms)
        ranked = sorted(scores, key=lambda x: scores[x], reverse=True)
        return ranked[:_ENTITY_RETRIEVAL_K]

    def _expand(self, indices: list[int]) -> list[int]:
        n = len(self._episodes)
        expanded: set[int] = set()
        for idx in indices:
            for offset in range(-_EXPAND_CONTEXT, _EXPAND_CONTEXT + 1):
                j = idx + offset
                if 0 <= j < n:
                    expanded.add(j)
        return sorted(expanded)

    def _format_episodes(self, indices: list[int]) -> str:
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
    # Ingest-time LLM calls (haiku -- fast/cheap)
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

        conversation_sample = "\n".join(
            f"{t.get('role', 'user').upper()}: {(t.get('content', '') or '')[:300]}"
            for t in selected
        )

        prompt = (
            "You are analyzing a long multi-session conversation for a QA task.\n"
            "Produce a STRUCTURED MEMORY covering:\n\n"
            "## People\n"
            "List every person mentioned (full name), their relationship to the speakers, "
            "key facts (job, hobbies, location, age, physical traits).\n"
            "Also list pronouns/nicknames that refer to them (e.g., 'he' likely = Bob).\n\n"
            "## Timeline of Key Events\n"
            "Chronological list: [DATE or PERIOD] - EVENT. Include ALL dates mentioned.\n\n"
            "## Places\n"
            "All locations visited or mentioned, with who and when.\n\n"
            "## Personal Facts\n"
            "Preferences, habits, recurring activities, goals, life changes, "
            "purchases, health, relationships.\n\n"
            "## Relationships & Social Connections\n"
            "How people know each other, who introduced whom, social network.\n\n"
            "## Numbers & Counts\n"
            "Any numeric facts: ages, durations, quantities, distances, prices.\n\n"
            "Be factual, dense, and comprehensive -- this memory will be used "
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
                            "Be comprehensive and precise. Include all named entities, "
                            "dates, and relationships. Use the exact names as mentioned."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model="haiku",
            )
        except Exception:
            return ""

    def _extract_facts(self, turns: list[dict]) -> str:
        if not turns:
            return ""

        n = len(turns)
        chunk_size = 80
        all_facts: list[str] = []

        for start in range(0, n, chunk_size):
            chunk = turns[start: start + chunk_size]
            chunk_text = "\n".join(
                f"{t.get('role', 'user').upper()}: {(t.get('content', '') or '')[:200]}"
                for t in chunk
            )
            prompt = (
                "Extract factual statements as triples from this conversation excerpt.\n"
                "Format: ENTITY | RELATION | VALUE\n"
                "Examples:\n"
                "  Alice | works at | Google\n"
                "  Bob | hobby | playing guitar\n"
                "  Sarah | visited | Paris in 2021\n"
                "  Tom | met | Alice through work\n"
                "  Carol | age | 32\n"
                "  Dave | introduced | Alice to Bob\n\n"
                "Extract ALL factual claims about people, places, events, dates, numbers:\n\n"
                f"{chunk_text}\n\n"
                "Triples (one per line):"
            )
            try:
                result = self._chat(
                    [
                        {
                            "role": "system",
                            "content": "Extract entity-relation-value triples. One per line. Be thorough.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    model="haiku",
                )
                if result.strip():
                    all_facts.append(result.strip())
                    self._parse_triples_into_entity_memory(result.strip())
            except Exception:
                continue

        return "\n".join(all_facts)

    def _parse_triples_into_entity_memory(self, triples_text: str) -> None:
        for line in triples_text.split("\n"):
            line = line.strip()
            if "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue
            entity = parts[0].strip()
            relation = parts[1].strip()
            value = parts[2].strip()
            if not entity or not relation or not value:
                continue
            key = entity.lower()
            if key not in self._entity_memory:
                self._entity_memory[key] = []
            self._entity_memory[key].append(f"{entity} | {relation} | {value}")

    # ------------------------------------------------------------------
    # Query-time LLM analysis (haiku)
    # ------------------------------------------------------------------

    def _analyze_query(self, question: str) -> dict:
        prompt = (
            "Analyze this question for a long-conversation QA task.\n"
            "Return a JSON object with these fields:\n"
            "- entities: list of named entities (person names, places, organizations) in the question\n"
            "- question_type: one of [temporal, entity_fact, list, count, multi_hop, yes_no, open]\n"
            "- is_multi_hop: true if answering requires combining facts from multiple parts\n"
            "- sub_questions: if multi_hop, list up to 3 simpler sub-questions to answer first\n"
            "- search_terms: list of key terms to search for in the conversation\n"
            "- temporal_focus: true if the question is primarily about when/date/time\n\n"
            f"Question: {question}\n\n"
            "Return ONLY valid JSON, no explanation."
        )
        try:
            raw = self._chat(
                [
                    {
                        "role": "system",
                        "content": "You analyze questions for information retrieval. Return valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model="haiku",
            )
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass
        return {
            "entities": self._extract_capitalized_names(question),
            "question_type": "open",
            "is_multi_hop": False,
            "sub_questions": [],
            "search_terms": self._tokenize(question),
            "temporal_focus": any(
                kw in question.lower()
                for kw in ("when", "what year", "what date", "how long", "which year", "what month")
            ),
        }

    def _extract_capitalized_names(self, text: str) -> list[str]:
        tokens = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
        return [t for t in tokens if t.lower() not in self._STOPWORDS]

    def _get_entity_memory_block(self, entities: list[str]) -> str:
        result_lines: list[str] = []
        for entity in entities:
            key = entity.lower()
            if key in self._entity_memory:
                result_lines.extend(self._entity_memory[key])
            else:
                for stored_key, facts in self._entity_memory.items():
                    if stored_key.startswith(key) or (
                        key and stored_key.split()[0] == key.split()[0]
                    ):
                        result_lines.extend(facts)
        seen: set[str] = set()
        unique: list[str] = []
        for line in result_lines:
            if line not in seen:
                seen.add(line)
                unique.append(line)
        return "\n".join(unique)

    def _answer_sub_question(self, sub_q: str, base_context: str) -> str:
        raw_hits = self._retrieve(sub_q, top_k=10)
        expanded = self._expand(raw_hits)
        excerpt = self._format_episodes(expanded)

        user_content = (
            f"{base_context}\n\n"
            f"## RELEVANT EXCERPTS FOR SUB-QUESTION\n{excerpt}\n\n"
            f"Sub-question: {sub_q}\n"
            "Answer (brief, direct):"
        )
        try:
            return self._chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You answer sub-questions from conversation memory. "
                            "Be brief and precise. Give the specific fact only."
                        ),
                    },
                    {"role": "user", "content": user_content},
                ],
                model="haiku",
            )
        except Exception:
            return "unknown"

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question: str) -> str:
        if not self._turns:
            return self._chat([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ])

        # Step 1: Analyze query with haiku
        analysis = self._analyze_query(question)
        entities: list[str] = analysis.get("entities", [])
        is_multi_hop: bool = bool(analysis.get("is_multi_hop", False))
        sub_questions: list[str] = analysis.get("sub_questions", [])[:_MAX_SUB_QUESTIONS]
        is_temporal: bool = bool(analysis.get("temporal_focus", False))
        q_type: str = analysis.get("question_type", "open")
        search_terms: list[str] = analysis.get("search_terms", [])
        is_list_q: bool = q_type in ("list", "count")

        # Fallback temporal detection
        if not is_temporal:
            is_temporal = any(
                kw in question.lower()
                for kw in ("when", "what year", "what date", "how long ago",
                           "which year", "what month", "how many years", "what time")
            )

        # Step 2: Multi-pass retrieval
        # Pass A: entity-focused retrieval for each named entity
        entity_hits: set[int] = set()
        for entity in entities:
            hits = self._retrieve_for_entity(entity)
            entity_hits.update(hits)

        # Pass B: standard TF-IDF retrieval on full question
        tfidf_hits = self._retrieve(question, top_k=_RETRIEVAL_K)
        tfidf_set = set(tfidf_hits)

        # Pass C: TF-IDF on expanded search terms (if any additional)
        term_hits: set[int] = set()
        if search_terms:
            augmented_q = " ".join(search_terms)
            additional = self._retrieve(augmented_q, top_k=10)
            term_hits.update(additional)

        # Merge: entity hits first (higher precision), then TF-IDF
        merged_hits: list[int] = list(entity_hits)
        for idx in tfidf_hits:
            if idx not in entity_hits:
                merged_hits.append(idx)
        for idx in term_hits:
            if idx not in entity_hits and idx not in tfidf_set:
                merged_hits.append(idx)

        # Limit before expansion
        cap = _RETRIEVAL_K + len(entity_hits)
        merged_hits = merged_hits[:cap]
        expanded = self._expand(sorted(merged_hits))
        episode_excerpt = self._format_episodes(expanded)

        # Step 3: Build base context block
        context_parts: list[str] = []

        if self._summary:
            context_parts.append(f"## CONVERSATION MEMORY\n{self._summary}")

        if entities:
            entity_facts = self._get_entity_memory_block(entities)
            if entity_facts:
                context_parts.append(f"## ENTITY-SPECIFIC FACTS\n{entity_facts}")

        if self._facts:
            context_parts.append(f"## ALL EXTRACTED FACTS\n{self._facts}")

        if episode_excerpt:
            context_parts.append(
                f"## RELEVANT CONVERSATION EXCERPTS "
                f"(timestamps in brackets)\n{episode_excerpt}"
            )

        base_context = "\n\n".join(context_parts)
        self.last_context_tokens = len(base_context.split()) + len(question.split())

        # Step 4: For multi-hop questions, answer sub-questions first
        sub_answers_block = ""
        if is_multi_hop and sub_questions:
            sub_answers: list[str] = []
            mini_context = ""
            if self._summary:
                mini_context = f"## CONVERSATION MEMORY\n{self._summary[:3000]}"
            for sq in sub_questions:
                try:
                    ans = self._answer_sub_question(sq, mini_context)
                    sub_answers.append(f"Q: {sq}\nA: {ans}")
                except Exception:
                    sub_answers.append(f"Q: {sq}\nA: unknown")
            if sub_answers:
                sub_answers_block = (
                    "\n\n## SUB-QUESTION ANSWERS (intermediate reasoning)\n"
                    + "\n\n".join(sub_answers)
                )

        # Step 5: Build answer hint based on question type
        if is_temporal:
            answer_hint = (
                "For this TEMPORAL question: output the ABSOLUTE date, month, or year. "
                "NEVER answer with relative time like 'yesterday', 'last year', 'recently'. "
                "Look at the [timestamp] tags on conversation turns -- if a turn tagged "
                "[2023-05-07] says 'I went to X yesterday', the answer is '2023-05-06'. "
                "If the turn itself IS the event, return that turn's date. "
                "Express dates as: Month Day, Year (e.g., 'May 7, 2023') or Year only."
            )
        elif q_type == "count":
            answer_hint = (
                "For this COUNT question: carefully count all instances in the facts "
                "and excerpts. Return ONLY the number."
            )
        elif q_type == "yes_no":
            answer_hint = (
                "For this YES/NO question: answer 'yes' or 'no' followed by a brief reason."
            )
        elif is_list_q:
            answer_hint = (
                "For this LIST question: output a comma-separated list of ALL items. "
                "Be exhaustive -- scan the facts and excerpts for every instance."
            )
        elif q_type == "entity_fact":
            answer_hint = (
                "For this ENTITY FACT question: look up the entity in ENTITY-SPECIFIC FACTS "
                "first. Give the precise value (name, place, job, etc.)."
            )
        elif is_multi_hop:
            answer_hint = (
                "For this MULTI-HOP question: use the sub-question answers above as "
                "intermediate reasoning steps to derive the final answer. "
                "State only the final answer."
            )
        else:
            answer_hint = (
                "Give a SHORT, DIRECT answer (usually 1-15 words). "
                "Do not explain your reasoning."
            )

        system_prompt = (
            "You are a precise question-answering assistant with access to structured "
            "memory of a long conversation.\n\n"
            f"{answer_hint}\n\n"
            "Rules:\n"
            "- Answer ONLY from the provided memory and excerpts.\n"
            "- If the answer includes a person's name, use their full name.\n"
            "- Do NOT make up information not found in the memory.\n"
            "- If the answer is truly not found, say 'unknown'.\n"
            "- Output ONLY the answer -- no preamble, no explanation, no 'Based on...'."
        )

        user_content = (
            f"{base_context}"
            f"{sub_answers_block}\n\n"
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