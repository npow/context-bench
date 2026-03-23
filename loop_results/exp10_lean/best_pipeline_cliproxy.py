# lean_pipeline.py
# ===================================================================
# BM25 + Entity Index + Knowledge-Update Tracking pipeline for LongMemEval-S.
#
# Design: ZERO LLM calls during ingest. Single sonnet call per query.
#
# Architecture:
# - Ingest: BM25 index + entity index (regex) + update-statement detection
# - Query: multi-signal retrieval (BM25 + entity + recency) -> relevance-ordered
#          chunk packing -> single sonnet call
#
# v3 improvements over v2:
# - Expanded synonym dict with inflected word forms (working/worked/works, etc.)
# - User-turn score boosting (personal facts live in user turns)
# - Recent-turns injection for knowledge-update questions
# - More robust _UPDATE_PATTERNS for fact-change detection
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

_EXPAND_CONTEXT = 2           # turns to include around each hit
_RETRIEVAL_K = 40             # top-K turns before expansion
_MAX_CONTEXT_CHARS = 30000    # guard against over-long prompts
_BM25_K1 = 1.5
_BM25_B = 0.75
_USER_TURN_BOOST = 1.4        # user turns hold personal facts; boost their scores
_RECENT_TURNS_KU = 40         # how many recent turns to prepend for knowledge-update


class ContextPipeline:
    """BM25 + Entity-aware retrieval pipeline for LongMemEval-S.

    Zero LLM calls during ingest. Single sonnet call per query.
    v3: inflection-aware synonyms, user-turn boosting, recent-turn injection.
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
        # BM25 index: word -> [(turn_idx, raw_tf)]
        self._index: dict[str, list[tuple[int, int]]] = {}
        self._idf: dict[str, float] = {}
        self._doc_lens: list[int] = []
        self._avg_dl: float = 1.0
        # Entity index: entity_str -> [turn_idx]
        self._entity_index: dict[str, list[int]] = {}
        # Update-statement turns: list of (turn_idx, snippet)
        self._update_turns: list[tuple[int, str]] = []
        # Role lookup for score boosting
        self._turn_roles: list[str] = []
        self.last_context_tokens: int = 0

    @staticmethod
    def _get_turn(turn, key, default=""):
        if isinstance(turn, dict):
            return self._get_turn(turn,key, default)
        return getattr(turn, key, default)

    @property
    def name(self) -> str:
        return "bm25_entity_v3"

    def reset(self) -> None:
        self._turns = []
        self._index = {}
        self._idf = {}
        self._doc_lens = []
        self._avg_dl = 1.0
        self._entity_index = {}
        self._update_turns = []
        self._turn_roles = []
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
        "said", "say", "one", "two", "yes", "oh", "well",
        "really", "actually", "thing", "things", "sure",
    })

    # ------------------------------------------------------------------
    # Query synonym expansion (pure Python, no LLM)
    # Expanded v3: inflected forms added so BM25 matches word variants.
    # ------------------------------------------------------------------

    _QUERY_SYNONYMS: dict[str, list[str]] = {
        # Work / career -- inflections added
        "job": ["work", "career", "employed", "occupation", "profession", "employer",
                "company", "role", "position", "working", "works", "worked"],
        "work": ["job", "career", "employed", "occupation", "profession", "employer",
                 "company", "working", "works", "worked"],
        "career": ["job", "work", "profession", "occupation", "role", "position"],
        "employed": ["job", "work", "company", "employer", "employment"],
        "profession": ["job", "work", "career", "occupation"],
        "occupation": ["job", "work", "career", "profession"],
        "company": ["employer", "firm", "organization", "workplace"],
        "employer": ["company", "firm", "organization", "workplace"],
        "hire": ["hired", "hiring", "job", "work", "position", "offer", "accepted"],
        "accept": ["accepted", "offer", "job", "position", "hired"],
        # Location / home -- inflections added
        "live": ["home", "reside", "located", "based", "stay", "city", "town",
                 "apartment", "house", "living", "lives", "lived"],
        "lived": ["home", "reside", "located", "based", "city", "moved", "living"],
        "living": ["live", "lives", "lived", "home", "city", "town", "apartment", "house"],
        "reside": ["live", "home", "located", "city", "town", "residing", "resided"],
        "city": ["town", "place", "location", "area", "live", "reside", "moved"],
        "town": ["city", "place", "location", "area", "live"],
        "home": ["live", "city", "apartment", "house", "place"],
        "move": ["relocate", "moved", "transfer", "migrated", "switched",
                 "moving", "relocated", "relocation"],
        "moved": ["relocate", "transfer", "migrated", "new", "moving", "relocation",
                  "relocated", "switch"],
        "relocate": ["move", "moved", "transfer", "relocating", "relocated"],
        "relocating": ["move", "moved", "relocate", "relocated", "moving"],
        # Education -- inflections added
        "study": ["school", "university", "college", "major", "degree", "course",
                  "program", "graduate", "studying", "studied", "studies"],
        "studying": ["study", "studied", "school", "university", "college", "major",
                     "degree", "course", "program"],
        "school": ["university", "college", "study", "education", "degree", "academic"],
        "university": ["college", "school", "study", "degree", "academic", "campus"],
        "college": ["university", "school", "study", "degree"],
        "degree": ["major", "study", "university", "college", "graduate"],
        "major": ["study", "degree", "subject", "field", "program"],
        "graduate": ["degree", "graduated", "alumni", "school", "university",
                     "graduating", "graduation"],
        "graduated": ["graduate", "degree", "alumni", "school", "university", "finished"],
        # Preferences / hobbies
        "hobby": ["activity", "interest", "enjoy", "pastime", "leisure", "like", "love",
                  "hobbies"],
        "interest": ["hobby", "like", "enjoy", "passion", "activity", "interested",
                     "interests"],
        "like": ["enjoy", "love", "prefer", "favorite", "favourite", "fond", "into",
                 "liked", "likes", "enjoying"],
        "enjoy": ["like", "love", "prefer", "favorite", "hobby", "activity",
                  "enjoying", "enjoyed"],
        "love": ["like", "enjoy", "adore", "favorite", "passion", "loving", "loved"],
        "prefer": ["like", "enjoy", "love", "favor", "favour", "rather", "favorite",
                   "preferred", "prefers"],
        "favorite": ["prefer", "like", "enjoy", "love", "favourite", "best"],
        "favourite": ["prefer", "like", "enjoy", "love", "favorite", "best"],
        # Temporal
        "first": ["initial", "begin", "start", "earliest", "original"],
        "last": ["recent", "latest", "final", "current", "most recent"],
        "current": ["now", "currently", "latest", "recent", "present", "today",
                    "these days"],
        "currently": ["now", "current", "latest", "recent", "present", "today"],
        "recent": ["latest", "current", "recently", "new", "last"],
        "recently": ["lately", "just", "new", "current", "now"],
        "when": ["time", "date", "year", "month", "moment", "period"],
        # Change / update
        "change": ["update", "switch", "new", "different", "modify", "switched",
                   "changed", "changing"],
        "changed": ["update", "switch", "new", "different", "modify", "moved",
                    "switched", "changing"],
        "update": ["change", "new", "switch", "latest", "recent", "updated"],
        "switch": ["change", "move", "switched", "new", "different", "switching"],
        "switched": ["change", "switch", "moved", "new", "different", "changing"],
        "new": ["recent", "latest", "current", "changed", "switched", "update"],
        # Food / eating -- inflections added
        "food": ["eat", "restaurant", "cuisine", "meal", "diet", "dish", "cooking",
                 "eating", "ate"],
        "eat": ["food", "restaurant", "meal", "cuisine", "diet", "eating", "ate",
                "eats"],
        "eating": ["eat", "ate", "food", "restaurant", "meal", "cuisine"],
        "restaurant": ["eat", "food", "dining", "cuisine", "place"],
        # Travel -- inflections added
        "travel": ["trip", "visit", "vacation", "holiday", "journey", "went",
                   "traveling", "travelled", "travelling"],
        "traveled": ["trip", "visit", "vacation", "journey", "went", "travel"],
        "visit": ["travel", "trip", "went", "vacation", "tour", "visited", "visiting"],
        "trip": ["travel", "visit", "vacation", "journey", "went"],
        # Relationships
        "friend": ["colleague", "acquaintance", "companion", "buddy", "pal", "mate",
                   "friends"],
        "partner": ["spouse", "boyfriend", "girlfriend", "husband", "wife", "significant"],
        "family": ["parent", "sibling", "brother", "sister", "mother", "father",
                   "relative"],
        # Sports / fitness
        "sport": ["exercise", "fitness", "gym", "play", "athletic", "activity",
                  "sports"],
        "exercise": ["workout", "fitness", "gym", "sport", "run", "train",
                     "exercising", "exercised"],
        "gym": ["exercise", "workout", "fitness", "train"],
        # Offer / acceptance
        "offer": ["job", "position", "hired", "accepted", "received", "got"],
        "accept": ["accepted", "offer", "got", "hired", "job", "position"],
    }

    def _expand_query_tokens(self, tokens: list[str]) -> list[str]:
        """Expand query tokens with synonyms for better BM25 recall."""
        expanded = list(tokens)
        seen = set(tokens)
        for token in tokens:
            syns = self._QUERY_SYNONYMS.get(token, [])
            for syn in syns:
                syn_tok = re.findall(r"[a-z][a-z0-9']*", syn.lower())
                for st in syn_tok:
                    if st not in seen and st not in self._STOPWORDS and len(st) > 1:
                        expanded.append(st)
                        seen.add(st)
        return expanded

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-z][a-z0-9']*", text.lower())
        return [t for t in tokens if t not in self._STOPWORDS and len(t) > 1]

    # ------------------------------------------------------------------
    # Entity extraction (pure Python, regex)
    # ------------------------------------------------------------------

    # v3: expanded patterns to catch more natural phrasing of fact changes
    _UPDATE_PATTERNS: list[re.Pattern] = [
        re.compile(r'\bi\s+(?:just|recently|now|finally|officially)\s+(?:got|found|started|moved|changed|switched|became|joined|left|quit|finished|accepted|signed|landed)\b', re.I),
        re.compile(r'\bi\s+(?:moved|relocated|transferred|switched|transitioned)\s+to\b', re.I),
        re.compile(r'\bi\'?m\s+(?:now|currently|officially|actually)\b', re.I),
        re.compile(r'\b(?:new\s+job|new\s+city|new\s+apartment|new\s+house|new\s+role|new\s+position|new\s+school|new\s+company)\b', re.I),
        re.compile(r'\bi\s+(?:changed|updated|got\s+a\s+new|have\s+a\s+new|found\s+a\s+new)\b', re.I),
        re.compile(r'\bstarting\s+(?:a\s+new|my\s+new)\b', re.I),
        re.compile(r'\b(?:as\s+of|since|from\s+now)\b.*\bi\b', re.I),
        # Additional patterns (v3)
        re.compile(r'\bi\s+(?:accepted|received|got)\s+(?:an?\s+)?(?:job\s+)?offer\b', re.I),
        re.compile(r'\bi\'?(?:m| am)\s+(?:going\s+to\s+be|going\s+to\s+start|moving\s+to|transferring\s+to)\b', re.I),
        re.compile(r'\b(?:big\s+news|exciting\s+news|update[:\s])\b', re.I),
        re.compile(r'\bi\s+(?:left|quit|resigned\s+from|got\s+fired|was\s+laid\s+off)\b', re.I),
        re.compile(r'\bby\s+the\s+way[,\s]+i\b', re.I),
        re.compile(r'\bi\s+(?:should\s+mention|wanted\s+to\s+tell\s+you|forgot\s+to\s+mention)\b', re.I),
        re.compile(r'\bi\'?m\s+(?:living|working|studying|staying)\s+(?:in|at|with)\b', re.I),
    ]

    def _extract_entities(self, text: str) -> list[str]:
        """Extract named entities, years, and key nouns using regex (no LLM)."""
        entities: list[str] = []

        # Capitalized name sequences (1-3 words, e.g. "John Smith", "New York")
        names = re.findall(r'\b([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){0,2})\b', text)
        for n in names:
            entities.append(n.lower())

        # Years
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', text)
        entities.extend(years)

        # Quoted strings (often proper names / titles)
        quoted = re.findall(r'"([^"]{2,40})"', text)
        entities.extend([q.lower() for q in quoted])

        return entities

    # ------------------------------------------------------------------
    # Ingest -- pure Python, no LLM calls
    # ------------------------------------------------------------------

    def ingest(self, turns: list[dict[str, Any]]) -> None:
        self._turns = turns
        n = len(turns)
        if n == 0:
            return

        bm25_index: dict[str, list] = defaultdict(list)
        doc_freq: dict[str, int] = defaultdict(int)
        entity_index: dict[str, list[int]] = defaultdict(list)
        doc_lens: list[int] = []
        update_turns: list[tuple[int, str]] = []
        turn_roles: list[str] = []

        for idx, turn in enumerate(turns):
            content = self._get_turn(turn,"content", "") or ""
            role = self._get_turn(turn,"role", "user")
            turn_roles.append(role)
            text = f"{role}: {content}"

            tokens = self._tokenize(text)
            doc_lens.append(len(tokens))

            # TF counting for BM25
            tf_counter: dict[str, int] = defaultdict(int)
            for t in tokens:
                tf_counter[t] += 1

            seen_in_doc: set[str] = set()
            for word, tf in tf_counter.items():
                bm25_index[word].append((idx, tf))
                if word not in seen_in_doc:
                    doc_freq[word] += 1
                    seen_in_doc.add(word)

            # Entity extraction (all turns -- assistant may mention user's info)
            entities = self._extract_entities(content)
            for entity in entities:
                if entity and len(entity) > 1:
                    entity_index[entity].append(idx)

            # Update-statement detection (user turns only)
            if role == "user":
                for pattern in self._UPDATE_PATTERNS:
                    if pattern.search(content):
                        update_turns.append((idx, content[:300]))
                        break

        # BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        self._idf = {}
        for word, df in doc_freq.items():
            self._idf[word] = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

        self._index = dict(bm25_index)
        self._doc_lens = doc_lens
        self._avg_dl = sum(doc_lens) / n if n > 0 else 1.0
        self._entity_index = dict(entity_index)
        self._update_turns = update_turns
        self._turn_roles = turn_roles

    # ------------------------------------------------------------------
    # BM25 scoring
    # ------------------------------------------------------------------

    def _bm25_scores(self, query_tokens: list[str]) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        k1, b = _BM25_K1, _BM25_B
        avg_dl = self._avg_dl or 1.0

        for token in query_tokens:
            if token not in self._index:
                continue
            idf = self._idf.get(token, 0.0)
            for (idx, tf) in self._index[token]:
                dl = self._doc_lens[idx] if idx < len(self._doc_lens) else avg_dl
                tf_norm = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avg_dl))
                scores[idx] += idf * tf_norm

        return scores

    # ------------------------------------------------------------------
    # Entity-overlap scoring
    # ------------------------------------------------------------------

    def _entity_scores(self, query_entities: list[str]) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        for entity in query_entities:
            ent_l = entity.lower().strip()
            if not ent_l or len(ent_l) < 2:
                continue
            # Exact match
            if ent_l in self._entity_index:
                for idx in self._entity_index[ent_l]:
                    scores[idx] += 2.0
            else:
                # Substring match (e.g. "John" matches "John Smith")
                for stored, idxs in self._entity_index.items():
                    if len(stored) > 2 and (ent_l in stored or stored in ent_l):
                        for idx in idxs:
                            scores[idx] += 1.0
        return scores

    # ------------------------------------------------------------------
    # Question type detection (pure Python)
    # ------------------------------------------------------------------

    def _detect_q_type(self, question: str) -> str:
        q = question.lower()
        if any(kw in q for kw in (
            "when", "what year", "what date", "how long",
            "which year", "what month", "first time", "last time",
            "before", "after", "earlier", "later", "sequence", "order",
        )):
            return "temporal"
        if any(kw in q for kw in (
            "current", "currently", "now", "latest", "recently",
            "changed", "updated", "new", "switched", "moved",
            "still", "anymore", "these days", "at the moment",
            "most recent", "last job", "last city", "last place",
        )):
            return "knowledge_update"
        if any(kw in q for kw in (
            "prefer", "like", "enjoy", "favorite", "favourite",
            "love", "hate", "dislike", "rather",
        )):
            return "preference"
        return "factual"

    # ------------------------------------------------------------------
    # Combined scoring -- v3: user-turn boosting
    # ------------------------------------------------------------------

    def _combined_scores(
        self, question: str, q_type: str
    ) -> dict[int, float]:
        """Compute per-turn relevance scores (BM25 + entity + role boost + type bias)."""
        tokens = self._tokenize(question)
        expanded_tokens = self._expand_query_tokens(tokens)
        entities = self._extract_entities(question)
        n = len(self._turns)

        bm25 = self._bm25_scores(expanded_tokens)
        ent = self._entity_scores(entities)

        all_idxs: set[int] = set(bm25.keys()) | set(ent.keys())
        combined: dict[int, float] = {}

        for idx in all_idxs:
            score = bm25.get(idx, 0.0) + 1.5 * ent.get(idx, 0.0)

            # v3: Boost user turns -- personal facts come from user, not assistant
            if idx < len(self._turn_roles) and self._turn_roles[idx] == "user":
                score *= _USER_TURN_BOOST

            # For knowledge-update: recency bias (later turns preferred)
            if q_type == "knowledge_update" and n > 1:
                recency = idx / (n - 1)          # 0.0 ... 1.0
                score *= (1.0 + 0.5 * recency)

            combined[idx] = score

        # For knowledge-update: also surface detected update-statement turns
        if q_type == "knowledge_update":
            for hint_idx, _ in self._update_turns:
                existing = combined.get(hint_idx, 0.0)
                combined[hint_idx] = max(existing * 1.4, existing + 0.8)

        return combined

    # ------------------------------------------------------------------
    # Chunk-based retrieval and relevance-ordered packing
    # ------------------------------------------------------------------

    def _make_chunks(
        self,
        hit_indices: list[int],
        scores: dict[int, float],
    ) -> list[tuple[float, list[int]]]:
        """Group hit indices with +-EXPAND_CONTEXT into contiguous chunks."""
        n = len(self._turns)
        expanded: dict[int, float] = {}
        for idx in hit_indices:
            hit_score = scores.get(idx, 0.0)
            for offset in range(-_EXPAND_CONTEXT, _EXPAND_CONTEXT + 1):
                neighbor = idx + offset
                if 0 <= neighbor < n:
                    decay = 1.0 if offset == 0 else 0.5
                    prev = expanded.get(neighbor, 0.0)
                    expanded[neighbor] = max(prev, hit_score * decay)

        if not expanded:
            return []

        sorted_indices = sorted(expanded.keys())

        chunks: list[tuple[float, list[int]]] = []
        current_chunk = [sorted_indices[0]]
        current_score = expanded[sorted_indices[0]]

        for idx in sorted_indices[1:]:
            if idx == current_chunk[-1] + 1:
                current_chunk.append(idx)
                current_score = max(current_score, expanded[idx])
            else:
                chunks.append((current_score, list(current_chunk)))
                current_chunk = [idx]
                current_score = expanded[idx]

        chunks.append((current_score, list(current_chunk)))
        return chunks

    def _retrieve_chunks(
        self, question: str, q_type: str, top_k: int = _RETRIEVAL_K
    ) -> list[tuple[float, list[int]]]:
        """Retrieve top-K turns, expand into chunks, sort by relevance."""
        n = len(self._turns)
        if n == 0:
            return []

        scores = self._combined_scores(question, q_type)
        if not scores:
            return []

        ranked = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)[:top_k]
        chunks = self._make_chunks(ranked, scores)
        chunks.sort(key=lambda c: c[0], reverse=True)

        return chunks

    # ------------------------------------------------------------------
    # Context formatting (relevance-ordered)
    # ------------------------------------------------------------------

    def _format_chunks(
        self, chunks: list[tuple[float, list[int]]], q_type: str
    ) -> str:
        """Format chunks into context string.

        Chunks are in relevance order (best first).
        Within each chunk, turns are in chronological order.
        """
        if not chunks:
            return ""

        lines: list[str] = []
        total_chars = 0
        prev_last_idx = -10

        for _chunk_score, chunk_indices in chunks:
            if total_chars >= _MAX_CONTEXT_CHARS:
                break

            if lines and chunk_indices[0] > prev_last_idx + 1:
                lines.append("...")

            chunk_text_lines: list[str] = []
            for idx in chunk_indices:
                turn = self._turns[idx]
                role = self._get_turn(turn,"role", "user").upper()
                content = (self._get_turn(turn,"content", "") or "")[:600]
                line = f"[T{idx}] {role}: {content}"
                chunk_text_lines.append(line)

            chunk_text = "\n".join(chunk_text_lines)
            if total_chars + len(chunk_text) > _MAX_CONTEXT_CHARS and lines:
                continue

            lines.extend(chunk_text_lines)
            total_chars += len(chunk_text)
            prev_last_idx = chunk_indices[-1]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Recent-turns section for knowledge-update questions (v3)
    # ------------------------------------------------------------------

    def _build_recent_context(self, n_turns: int = _RECENT_TURNS_KU) -> tuple[int, str]:
        """Return (start_idx, formatted recent turns) for the last n_turns."""
        n = len(self._turns)
        if n == 0:
            return 0, ""
        start = max(0, n - n_turns)
        lines = []
        for i in range(start, n):
            turn = self._turns[i]
            role = self._get_turn(turn,"role", "user").upper()
            content = (self._get_turn(turn,"content", "") or "")[:400]
            lines.append(f"[T{i}] {role}: {content}")
        return start, "\n".join(lines)

    # ------------------------------------------------------------------
    # Query -- single LLM call
    # ------------------------------------------------------------------

    def query(self, question: str) -> str:
        if not self._turns:
            return self._chat([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ])

        n_total = len(self._turns)
        q_type = self._detect_q_type(question)
        chunks = self._retrieve_chunks(question, q_type, top_k=_RETRIEVAL_K)
        retrieved_context = self._format_chunks(chunks, q_type)

        if not retrieved_context:
            # Fallback: use last N turns
            n = len(self._turns)
            fallback_idxs = list(range(max(0, n - 20), n))
            retrieved_context = "\n".join(
                f"[T{i}] {self._turns[i].get('role','user').upper()}: "
                f"{(self._turns[i].get('content','') or '')[:400]}"
                for i in fallback_idxs
            )

        # v3: For knowledge-update questions, prepend the most recent turns so
        # the LLM always has access to the latest facts, regardless of BM25 scores.
        if q_type == "knowledge_update":
            recent_start, recent_context = self._build_recent_context(_RECENT_TURNS_KU)
            if recent_context:
                recent_chars = len(recent_context)
                retrieved_budget = _MAX_CONTEXT_CHARS - recent_chars - 300
                if retrieved_budget > 1000:
                    # Truncate retrieved context to leave room for recent turns
                    truncated_retrieved = retrieved_context[:retrieved_budget]
                    context = (
                        f"## MOST RECENT TURNS [T{recent_start}-T{n_total - 1}] "
                        f"(authoritative -- most up-to-date values are here):\n"
                        + recent_context
                        + "\n\n## ADDITIONAL RETRIEVED TURNS (for context):\n"
                        + truncated_retrieved
                    )
                else:
                    context = (
                        f"## MOST RECENT TURNS [T{recent_start}-T{n_total - 1}]:\n"
                        + recent_context
                    )
            else:
                context = retrieved_context
        else:
            context = retrieved_context

        self.last_context_tokens = len(context.split()) + len(question.split())

        # Conversation metadata helps the LLM calibrate turn positions
        conv_meta = f"(Conversation total: {n_total} turns. [T0]=earliest, [T{n_total - 1}]=most recent.)"

        # Type-specific prompt hints
        if q_type == "temporal":
            hint = (
                "TEMPORAL QUESTION: Turn indices [T###] reflect absolute conversation order -- "
                "HIGHER index = LATER in the conversation. Use this ordering to determine "
                "what happened first, last, or when something occurred. "
                "Note: retrieved chunks may appear out of sequence (most relevant first), "
                "but the [T###] numbers always reflect true chronological position."
            )
        elif q_type == "knowledge_update":
            hint = (
                "KNOWLEDGE-UPDATE QUESTION: The conversation may contain multiple updates "
                "to the same fact. The MOST RECENT TURNS section at the top contains the "
                "latest, authoritative information. If a fact is mentioned multiple times, "
                "use the value from the turn with the HIGHEST [T###] index -- that is the "
                "most current state. Ignore outdated earlier values (lower [T###])."
            )
        elif q_type == "preference":
            hint = (
                "PREFERENCE QUESTION: Look for explicit statements of liking, disliking, "
                "preferring, enjoying, or favouring something. If preferences changed over "
                "time, use the most recent statement (highest [T###] index)."
            )
        else:
            hint = (
                "Give a SHORT, DIRECT answer (usually 1-15 words). "
                "Do not explain your reasoning."
            )

        system_prompt = (
            "You are a precise question-answering assistant. "
            "Answer questions based ONLY on the conversation excerpts provided below.\n\n"
            f"INSTRUCTION: {hint}\n\n"
            "Rules:\n"
            "- Answer ONLY from the provided excerpts -- do not invent information.\n"
            "- If a person's name appears, use their full name.\n"
            "- If the answer is truly not in the excerpts, respond with exactly: unknown\n"
            "- Output ONLY the answer -- no preamble, no explanation, no hedging."
        )

        user_content = (
            f"## CONVERSATION EXCERPTS\n"
            f"{conv_meta}\n"
            f"(retrieved chunks ordered by relevance; [T###] = chronological turn index)\n"
            f"{context}\n\n"
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

    # Model alias mapping for Anthropic API (cliproxyapi)
    _MODEL_MAP = {
        "sonnet": "claude-sonnet-4-5-20250929",
        "haiku": "claude-haiku-4-5-20251001",
        "opus": "claude-opus-4-6",
    }

    def _chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
    ) -> str:
        model_name = model or self._model
        model_name = self._MODEL_MAP.get(model_name, model_name)

        # Separate system message from conversation messages
        system_text = ""
        api_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                api_messages.append({"role": msg["role"], "content": msg["content"]})
        if not api_messages:
            api_messages = [{"role": "user", "content": "Hello"}]

        url = f"{self._base_url}/v1/messages"
        payload: dict[str, Any] = {
            "model": model_name,
            "max_tokens": 4096,
            "messages": api_messages,
        }
        if system_text:
            payload["system"] = system_text

        body = json.dumps(payload).encode()
        headers = {
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if self._api_key:
            headers["x-api-key"] = self._api_key

        last_err: Exception | None = None
        for attempt in range(5):
            req = urllib.request.Request(
                url, data=body, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    data = json.loads(resp.read().decode())
                    content = data.get("content", [])
                    if content and isinstance(content, list):
                        return content[0].get("text", "").strip()
                    return ""
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code in (429, 500, 502, 503, 504, 529) and attempt < 4:
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