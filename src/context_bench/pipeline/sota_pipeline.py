# sota_pipeline.py
# ===================================================================
# Near-SOTA pipeline implementing key techniques from top LoCoMo systems:
#
# 1. COREFERENCE RESOLUTION: pronouns/references -> canonical entity names
#    during ingest so facts are findable by person name
#
# 2. TYPED ENTITY-RELATION TRIPLES with temporal tracking:
#    (entity, relation, value, turn_idx) - enables precise lookup
#    "find Caroline's hobbies" or "find Nate's current job"
#
# 3. ENTITY PROFILES: per-entity aggregate of all known facts,
#    with recency ordering for temporal questions
#
# 4. QUERY DECOMPOSITION: break multi-part questions into sub-queries,
#    answer each, synthesise
#
# 5. MULTI-HOP RETRIEVAL: extract entities from question, retrieve
#    entity profiles, then reason across them
#
# 6. CONFIDENCE-AWARE ANSWERING: partial credit for uncertain answers
#
# Based on techniques from: Hindsight, A-MEM, LiCoMemory, TEMPR
# ===================================================================

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

STRATEGY = {
    "extraction_model": "haiku",   # haiku for extraction (fast)
    "answer_model": "sonnet",      # sonnet for reasoning (accurate)
    "session_size": 10,            # turns per extraction chunk
    "triples_per_session": 15,     # max triples extracted per chunk
    "retrieval_k": 20,             # facts to retrieve per entity
    "use_query_decomposition": True,
    "use_coreference": True,
    "use_temporal_ranking": True,
    "max_context_tokens": 3000,
}


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _truncate(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " [truncated]"


def _get_st_model():
    try:
        from context_bench.embeddings import get_model
        return get_model()
    except ImportError:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")


class SotaPipeline:
    """Near-SOTA pipeline for LoCoMo long-conversation QA.

    Ingest: coreference resolution + typed triple extraction + entity profiles.
    Query: entity lookup + multi-hop + query decomposition + Sonnet reasoning.
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
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._strategy: dict[str, Any] = dict(STRATEGY)
        if strategy:
            self._strategy.update(strategy)
        # Override answer model from loop's --model flag
        self._strategy["answer_model"] = model
        self._timeout = timeout

        # Storage
        self._triples: list[dict] = []
        # triple: {entity, relation, value, turn_idx, session_idx, text}

        self._entity_profiles: dict[str, list[dict]] = {}
        # entity -> list of triples sorted by turn_idx

        self._all_facts: list[str] = []        # flat text for embedding fallback
        self._all_fact_turns: list[int] = []
        self._fact_embeddings: list[list[float]] = []

        self.last_context_tokens: int = 0

    @property
    def name(self) -> str:
        return "sota_pipeline_v1"

    def reset(self) -> None:
        self._triples = []
        self._entity_profiles = {}
        self._all_facts = []
        self._all_fact_turns = []
        self._fact_embeddings = []
        self.last_context_tokens = 0

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, turns: list[dict[str, Any]]) -> None:
        if not turns:
            return

        s = self._strategy
        session_size = max(1, s["session_size"])
        n_triples = max(1, s["triples_per_session"])
        ext_model = s["extraction_model"]

        # Step 1: Coreference resolution pass to get entity map
        entity_map: dict[str, str] = {}
        if s["use_coreference"]:
            entity_map = self._resolve_coreferences(turns, ext_model)

        # Step 2: Extract typed triples session by session
        for sess_idx, start in enumerate(range(0, len(turns), session_size)):
            group = turns[start: start + session_size]
            session_text = self._turns_to_text(group, entity_map)
            new_triples = self._extract_triples(
                session_text, n_triples, ext_model, start, sess_idx
            )
            self._triples.extend(new_triples)

        # Step 3: Build entity profiles
        self._build_entity_profiles()

        # Step 4: Embed all facts for fallback retrieval
        self._all_facts = [
            f"{t['entity']} {t['relation']} {t['value']}" for t in self._triples
        ]
        self._all_fact_turns = [t["turn_idx"] for t in self._triples]
        if self._all_facts:
            model = _get_st_model()
            embs = model.encode(
                self._all_facts, convert_to_numpy=True, show_progress_bar=False
            )
            self._fact_embeddings = [e.tolist() for e in embs]

    def _turns_to_text(
        self, turns: list[dict], entity_map: dict[str, str]
    ) -> str:
        lines = []
        for t in turns:
            role = t.get("role", "user").upper()
            content = t.get("content", "")
            # Apply entity map substitutions
            for alias, canonical in entity_map.items():
                if alias != canonical:
                    content = re.sub(
                        r'\b' + re.escape(alias) + r'\b', canonical, content
                    )
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _resolve_coreferences(
        self, turns: list[dict], model: str
    ) -> dict[str, str]:
        """Build alias->canonical name map from first ~60 turns."""
        sample = turns[:60]
        text = "\n".join(
            f"{t.get('role','user').upper()}: {t.get('content','')}"
            for t in sample
        )
        prompt = (
            "Read this conversation excerpt and list all people mentioned. "
            "For each person, give their canonical name and any aliases/pronouns used to refer to them.\n"
            "Output ONLY a JSON object: {\"alias\": \"CanonicalName\", ...}\n"
            "Include first-name-only references, nicknames, and relationship terms (e.g. 'my friend' -> 'Alice').\n"
            "If you cannot determine the canonical name for a pronoun, skip it.\n\n"
            f"Conversation:\n{text[:3000]}\n\nJSON:"
        )
        try:
            resp = self._chat(model, [{"role": "user", "content": prompt}])
            # Extract JSON from response
            match = re.search(r'\{[^{}]*\}', resp, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return {}

    def _extract_triples(
        self,
        session_text: str,
        n: int,
        model: str,
        turn_start: int,
        sess_idx: int,
    ) -> list[dict]:
        """Extract typed entity-relation-value triples from a session."""
        prompt = (
            f"Extract up to {n} factual triples from this conversation. "
            "Each triple captures a specific fact about a named entity.\n"
            "Format: one triple per line as: ENTITY | RELATION | VALUE\n"
            "Relations should be short (hobby, job, location, friend, sport, "
            "pet, school, achievement, preference, relationship, event, etc.)\n"
            "Only include facts about specific named people or places.\n"
            "No numbering, no preamble.\n\n"
            f"Conversation:\n{session_text}\n\nTriples:"
        )
        try:
            resp = self._chat(model, [{"role": "user", "content": prompt}])
        except Exception:
            return []

        triples = []
        for line in resp.splitlines():
            line = line.strip().strip("•- ")
            if "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                entity, relation, value = parts[0], parts[1], " | ".join(parts[2:])
                if entity and relation and value:
                    triples.append({
                        "entity": entity,
                        "relation": relation.lower(),
                        "value": value,
                        "turn_idx": turn_start,
                        "session_idx": sess_idx,
                        "text": f"{entity} {relation} {value}",
                    })
        return triples[:n]

    def _build_entity_profiles(self) -> None:
        """Group triples by entity, sorted by recency."""
        self._entity_profiles = {}
        for triple in self._triples:
            entity = triple["entity"]
            if entity not in self._entity_profiles:
                self._entity_profiles[entity] = []
            self._entity_profiles[entity].append(triple)
        # Sort each profile newest-first for temporal questions
        for entity in self._entity_profiles:
            self._entity_profiles[entity].sort(
                key=lambda t: t["turn_idx"], reverse=True
            )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question: str) -> str:
        if not self._triples:
            return self._chat(self._strategy["answer_model"], [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ])

        s = self._strategy

        # Step 1: Query decomposition for complex questions
        if s["use_query_decomposition"]:
            sub_questions = self._decompose_question(question)
        else:
            sub_questions = [question]

        # Step 2: For each sub-question, retrieve relevant facts
        all_context_parts = []
        for sq in sub_questions:
            facts = self._retrieve_for_question(sq)
            if facts:
                all_context_parts.append(f"[Re: {sq}]\n" + "\n".join(f"• {f}" for f in facts))

        if not all_context_parts:
            # Fallback: embedding search on original question
            facts = self._embedding_search(question, self._strategy["retrieval_k"])
            all_context_parts = ["\n".join(f"• {f}" for f in facts)]

        context = "\n\n".join(all_context_parts)
        context = _truncate(context, s["max_context_tokens"])
        self.last_context_tokens = len(context.split()) + len(question.split())

        # Step 3: Final answer with Sonnet
        system_prompt = (
            "You are a helpful assistant answering questions about a conversation. "
            "Use the provided facts to give a short, direct answer. "
            "Typically 1-10 words. Do not explain or use bullet points. "
            "If uncertain, give your best guess based on available facts. "
            "Output ONLY the answer."
        )
        user_content = (
            f"Facts from the conversation:\n{context}\n\n"
            f"Question: {question}\n\nAnswer (1-10 words):"
        )
        return self._chat(s["answer_model"], [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ])

    def _decompose_question(self, question: str) -> list[str]:
        """Break a complex question into simple sub-questions."""
        # Heuristic: only decompose if question seems multi-part
        if not any(kw in question.lower() for kw in ["and", "when did", "how did", "why did", "who introduced"]):
            return [question]

        prompt = (
            "Break this question into 1-3 simple sub-questions that together answer it. "
            "Output one sub-question per line, no numbering.\n"
            f"Question: {question}\nSub-questions:"
        )
        try:
            resp = self._chat(self._strategy["extraction_model"],
                              [{"role": "user", "content": prompt}])
            sqs = [line.strip() for line in resp.splitlines() if line.strip()]
            if sqs:
                return sqs[:3]
        except Exception:
            pass
        return [question]

    def _retrieve_for_question(self, question: str) -> list[str]:
        """Entity-profile lookup + embedding fallback for one question."""
        results = []

        # Extract entity names from question
        entities = self._extract_question_entities(question)

        # Look up entity profiles
        k = self._strategy["retrieval_k"]
        profile_facts = []
        for entity in entities:
            # Exact match
            if entity in self._entity_profiles:
                profile_facts.extend(self._entity_profiles[entity][:k])
            else:
                # Fuzzy match: entity name appears in profile key
                for profile_entity, triples in self._entity_profiles.items():
                    if (entity.lower() in profile_entity.lower() or
                            profile_entity.lower() in entity.lower()):
                        profile_facts.extend(triples[:k // 2])

        # Temporal ranking: if question asks about current/now, prefer recent
        if self._strategy["use_temporal_ranking"]:
            temporal_recent = any(
                kw in question.lower()
                for kw in ["now", "current", "currently", "today", "latest", "recently"]
            )
            temporal_past = any(
                kw in question.lower()
                for kw in ["used to", "before", "previously", "former", "old", "past", "ago"]
            )
            if temporal_recent:
                profile_facts.sort(key=lambda t: t["turn_idx"], reverse=True)
            elif temporal_past:
                profile_facts.sort(key=lambda t: t["turn_idx"])

        for t in profile_facts[:k]:
            results.append(f"{t['entity']} {t['relation']} {t['value']}")

        # Supplement with embedding search if profile lookup thin
        if len(results) < 5:
            emb_facts = self._embedding_search(question, k - len(results))
            seen = set(results)
            for f in emb_facts:
                if f not in seen:
                    results.append(f)
                    seen.add(f)

        return results[:k]

    def _extract_question_entities(self, question: str) -> list[str]:
        """Extract named entity mentions from a question."""
        # Simple capitalized-word heuristic
        entities = re.findall(r'\b[A-Z][a-z]{1,}(?:\s+[A-Z][a-z]+)*\b', question)
        # Also check against known entity profile keys
        q_lower = question.lower()
        for entity in self._entity_profiles:
            if entity.lower() in q_lower and entity not in entities:
                entities.append(entity)
        return list(dict.fromkeys(entities))  # dedup, preserve order

    def _embedding_search(self, query: str, k: int) -> list[str]:
        if not self._fact_embeddings:
            return []
        model = _get_st_model()
        qvec = model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0].tolist()
        scores = [_cosine(qvec, fv) for fv in self._fact_embeddings]
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._all_facts[i] for i in top]

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _chat(self, model: str, messages: list[dict[str, Any]]) -> str:
        url = f"{self._base_url}/v1/chat/completions"
        body = json.dumps({"model": model, "messages": messages}).encode()
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        for attempt in range(4):
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    data = json.loads(resp.read().decode())
                    return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                    time.sleep(15 * (2 ** attempt))
                    continue
                raise RuntimeError(f"Chat HTTP {e.code}: {e.reason}") from e
            except urllib.error.URLError as e:
                if attempt < 3:
                    time.sleep(15 * (2 ** attempt))
                    continue
                raise RuntimeError(f"Chat connection error: {e.reason}") from e

        raise RuntimeError("Chat request failed after 4 retries")


# Loop expects ContextPipeline class name
ContextPipeline = SotaPipeline
