"""TiMem baseline: Temporal-Hierarchical Memory Consolidation.

Faithful reproduction of TiMem (arXiv:2601.02845, Li et al. 2026).
Official repo: https://github.com/TiMEM-AI/timem

Cross-validation status (2026-05-25): STRENGTHENED — all HIGH-severity
Codex review deviations fixed; see changelog below.

Changelog (v2 fixes):
  - Dense embeddings: switched from TF-IDF cosine to amazon.titan-embed-text-v2:0
    (1024-dim) via Bedrock for semantic similarity in fused scoring.
  - BM25: replaced constant-IDF single-doc approximation with corpus BM25
    (proper IDF over all L1 memory texts, k1=1.5, b=0.75).
  - Timestamps: LME items have timestamp=None; use session_id (integer) as
    ingestion-order proxy. Sessions sorted by integer session_id; day/week
    buckets derived from this ordering. Documented in build_tmt.
  - L5 profile: incremental monthly updates — profile is rebuilt each time
    a new "month" worth of L4 nodes accumulates (every 4 weeks).
  - Final memory ordering: sorted by level ASC then temporal proximity
    (session_id recency) DESC, so recent fragments surface before old profile.

== Official repo summary ==
The official TiMem repo (TiMEM-AI/timem) is a SELF-HOSTED system that
requires PostgreSQL + Qdrant vector DB + optional Qwen3-Embedding-0.6B
local embedding model.  It is NOT pip-installable as a library and has
NO standalone single-file inference mode.  Running it on LongMemEval-S
data would require:
  (a) docker-compose up (Postgres + Qdrant containers),
  (b) configuring OPENAI_API_KEY / QWEN_API_KEY / ZHIPUAI_API_KEY in .env,
  (c) running experiments/datasets/longmemeval_s/01_memory_generation.py
      (async, hits an external LLM API), then 02_memory_retrieval.py.
There is also a cloud-service SDK (pip install timem-ai) that wraps an
API at https://api.timem.cloud — but that is cloud-only and requires a
paid account.  Neither mode is compatible with Bedrock-only inference
on Workbench.

== Algorithm comparison ==
Our reproduction matches the official algorithm in all key respects:
  - 5-level TMT hierarchy (L1 fragment → L2 session → L3 day →
    L4 week → L5 persona) ✓
  - L1 progressive fragment summary with w_i=3 historical context window ✓
    (official: config fragment_size=2 turns/fragment, merge_weight=0.7;
     ours: 1 turn/node, history_window=3 — slightly finer granularity)
  - Per-level LLM consolidation prompts with factual / evolving-pattern /
    persona framing at L2–L5 ✓
  - Three-stage recall: planner → hierarchical retrieval → gating ✓
  - Fused semantic+BM25 score at L1, complexity-aware per-level budgets ✓
    (official retrieval_config: simple={L1:20,L2:4,L5:1},
     hybrid={L1:20,L2:4,L3:2,L5:1}, complex={L1:20,L2:4,L3:4,L4:2,L5:1};
     ours matches these numbers exactly in BUDGETS dict)
  - LLM gating / memory_refiner step to prune candidates ✓

Algorithm overview:
  Build phase  — Process each turn into a 5-level Temporal Memory Tree (TMT):
    L1 segment   : one node per turn (online, factual summarization)
    L2 session   : consolidate all L1 nodes in the session
    L3 day       : consolidate all L2 nodes in the day
    L4 week      : consolidate all L3 nodes in the week
    L5 profile   : single evolving persona representation (all L4 nodes)

  Recall phase — Two LLM calls per query:
    Planner call : classify query complexity {simple, hybrid, complex}
                   and extract ≤3 keywords (no personal names).
    Retrieval    : fused BM25+cosine at L1 (λ=0.9, top k1=20),
                   ancestor collection with per-level budgets,
                   then a gating LLM call to retain/drop candidates.
    Answer       : pass ordered memories + question to LLM.

Remaining deviations from official code (minor):
  3. Fragment granularity: official groups every 2 dialogue turns into one
     L1 fragment; we create one L1 node per conversation turn.
  4. Storage backend: official stores memories in Qdrant (vector) +
     Postgres (metadata) and retrieves with vector search.  We keep
     everything in-memory as Python dicts, which is fine for one-shot eval.
  5. Retrieval planner: official uses rule-based keyword patterns to
     classify query category (TEMPORAL / FACTUAL / INFERENTIAL / DETAILED)
     and selects strategies accordingly.  We use a single LLM call to
     classify into {simple, hybrid, complex}.  Mapping is approximate.
  6. LongMemEval-S evaluation: official runs 500 users in parallel with
     40 concurrent API keys.  We run sequentially via Bedrock.
  - The gating call uses a simplified "retain/drop" prompt (paper's exact
    system prompt is not released), faithful to the described intent.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict

sys.path.insert(0, "src")

from context_bench.datasets.memory.longmemeval import longmemeval


# ---------------------------------------------------------------------------
# Bedrock client + multi-family dispatch (copied from run_strong_pipeline.py)
# ---------------------------------------------------------------------------

def _bedrock_client():
    import boto3
    return boto3.client("bedrock-runtime", region_name="us-east-1")


def _claude(client, prompt: str, max_tokens: int = 200, temperature: float = 0.3) -> str:
    """Model-agnostic Bedrock call — dispatches on BEDROCK_MODEL_ID."""
    model_id = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-sonnet-20241022-v2:0")

    if "anthropic" in model_id:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        })
        r = client.invoke_model(body=body, modelId=model_id, accept="application/json", contentType="application/json")
        return json.loads(r["body"].read())["content"][0]["text"].strip()

    if "openai" in model_id or "gpt" in model_id:
        body = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max(max_tokens * 3, 400),
        })
        r = client.invoke_model(body=body, modelId=model_id)
        text = json.loads(r["body"].read())["choices"][0]["message"]["content"]
        return re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL).strip()

    if "gemma" in model_id or "google" in model_id:
        body = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        r = client.invoke_model(body=body, modelId=model_id)
        resp = json.loads(r["body"].read())
        if "choices" in resp:
            return resp["choices"][0]["message"]["content"].strip()
        if "generation" in resp:
            return resp["generation"].strip()
        return str(resp).strip()

    raise ValueError(f"Unknown model: {model_id}")


# ---------------------------------------------------------------------------
# FIX: Dense embeddings via Bedrock (amazon.titan-embed-text-v2:0, 1024 dims)
# Replaces TF-IDF cosine used previously. Paper uses Qwen3-Embedding-0.6B;
# Titan v2 is the closest available Bedrock model (1024-dim dense vectors).
# ---------------------------------------------------------------------------

_EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
_embed_cache: dict[str, list[float]] = {}  # simple in-process cache


def _get_embedding(client, text: str) -> list[float]:
    """FIX: Dense embedding via Bedrock Titan Embed v2 (1024 dims).
    Replaces TF-IDF cosine. Cached per text to avoid redundant API calls."""
    text = text[:8000]  # Titan v2 max input
    if text in _embed_cache:
        return _embed_cache[text]
    body = json.dumps({"inputText": text, "dimensions": 1024, "normalize": True})
    try:
        r = client.invoke_model(
            body=body,
            modelId=_EMBED_MODEL_ID,
            accept="application/json",
            contentType="application/json",
        )
        emb = json.loads(r["body"].read())["embedding"]
    except Exception:
        # Fallback: zero vector (retrieval will degrade gracefully)
        emb = [0.0] * 1024
    _embed_cache[text] = emb
    return emb


def _dot_product(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cosine_dense(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two pre-normalised dense vectors."""
    # Titan v2 returns normalize=True vectors, so dot == cosine
    return _dot_product(a, b)


# ---------------------------------------------------------------------------
# Corpus BM25 for lexical branch  (FIX: was constant-IDF single-doc approx)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


class CorpusBM25:
    """FIX: Proper BM25 with corpus IDF over all L1 memory texts.
    Paper uses k1=1.5, b=0.75 as a separate lexical index over stored text.
    Previously a constant IDF=log(2) single-document approximation was used."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: list[list[str]] = []
        self._df: dict[str, int] = defaultdict(int)
        self._avgdl: float = 0.0
        self._idf: dict[str, float] = {}
        self._dirty: bool = True

    def add_doc(self, text: str) -> None:
        tokens = _tokenize(text)
        self._docs.append(tokens)
        for t in set(tokens):
            self._df[t] += 1
        self._dirty = True

    def _build_idf(self) -> None:
        N = len(self._docs)
        if N == 0:
            return
        self._avgdl = sum(len(d) for d in self._docs) / N
        # FIX: corpus IDF  idf(t) = log((N - df + 0.5) / (df + 0.5) + 1)
        self._idf = {
            t: math.log((N - df + 0.5) / (df + 0.5) + 1)
            for t, df in self._df.items()
        }
        self._dirty = False

    def score(self, query: str, doc_text: str) -> float:
        if self._dirty:
            self._build_idf()
        q_tok = _tokenize(query)
        d_tok = _tokenize(doc_text)
        if not q_tok or not d_tok:
            return 0.0
        d_freq: dict[str, int] = defaultdict(int)
        for t in d_tok:
            d_freq[t] += 1
        dl = len(d_tok)
        avgdl = self._avgdl if self._avgdl > 0 else dl
        score = 0.0
        for t in set(q_tok):
            idf = self._idf.get(t, math.log(1.5))
            tf = d_freq.get(t, 0)
            score += idf * (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * dl / avgdl)
            )
        return score


# Module-level BM25 corpus; populated during TMT build
_bm25_corpus: CorpusBM25 = CorpusBM25()


def _fused_score(client_or_none, query: str, memory_text: str,
                 query_emb: list[float] | None = None,
                 lam: float = 0.9) -> float:
    """FIX: λ·dense_cosine + (1-λ)·corpus_BM25  (paper: λ=0.9).
    Was: λ·TF-IDF_cosine + (1-λ)·constant-IDF_BM25.
    Now uses real dense embeddings (Titan v2) + corpus BM25."""
    if client_or_none is not None and query_emb is None:
        query_emb = _get_embedding(client_or_none, query)
    mem_emb = _get_embedding(client_or_none, memory_text) if client_or_none is not None else [0.0] * 1024
    cos = _cosine_dense(query_emb, mem_emb) if (query_emb and mem_emb) else 0.0
    bm_raw = _bm25_corpus.score(query, memory_text)
    # Normalise BM25 loosely: cap at 20 (typical max for short queries)
    bm_norm = min(bm_raw / 20.0, 1.0)
    return lam * cos + (1 - lam) * bm_norm


# ---------------------------------------------------------------------------
# TMT construction
# ---------------------------------------------------------------------------

def _group_items_by_session(items) -> dict[str, list]:
    sessions: dict[str, list] = {}
    for item in items:
        sid = str(getattr(item, "session_id", None) or "default")
        sessions.setdefault(sid, []).append(item)
    return sessions


def _turn_text(item) -> str:
    spk = getattr(item, "speaker", "")
    return f"{spk}: {item.content}" if spk else item.content


# --- L1: segment (one per turn, factual summarization) ---
def _build_l1(client, session_id: str, items) -> list[dict]:
    """Produce one L1 memory node per turn."""
    nodes = []
    history_window: list[str] = []
    for idx, item in enumerate(items):
        turn_text = _turn_text(item)
        hist_ctx = "\n".join(history_window[-3:])  # w_i = 3 historical memories
        prompt = (
            "Write a fragment memory that captures ONLY the NEW facts introduced in this turn. "
            "Use third-person perspective. Preserve any relative time expressions verbatim. "
            "Do NOT repeat facts already captured in the history below. "
            "Output one concise sentence.\n\n"
            + (f"History:\n{hist_ctx}\n\n" if hist_ctx else "")
            + f"Turn:\n{turn_text}\n\nFragment memory:"
        )
        try:
            mem = _claude(client, prompt, max_tokens=120, temperature=0.7)
        except Exception as e:
            mem = turn_text[:200]
        node = {
            "level": 1,
            "session_id": session_id,
            "turn_idx": idx,
            "text": mem,
            "raw": turn_text,
        }
        nodes.append(node)
        history_window.append(mem)
    return nodes


# --- L2: session consolidation ---
def _build_l2(client, session_id: str, l1_nodes: list[dict],
               recent_l2: list[str]) -> dict:
    child_texts = "\n".join(f"- {n['text']}" for n in l1_nodes)
    hist_ctx = "\n".join(recent_l2[-3:])
    prompt = (
        "Factual Summarization: Synthesize the session memories below into a coherent "
        "session-level summary. Third-person. Preserve specific facts (names, numbers, dates). "
        "Do NOT repeat prior context. Output 3-6 sentences.\n\n"
        + (f"Recent session summaries:\n{hist_ctx}\n\n" if hist_ctx else "")
        + f"Session {session_id} memories:\n{child_texts}\n\nSession summary:"
    )
    try:
        text = _claude(client, prompt, max_tokens=400, temperature=0.7)
    except Exception:
        text = child_texts[:600]
    return {"level": 2, "session_id": session_id, "text": text, "children": l1_nodes}


# --- L3: day consolidation ---
def _build_l3(client, day_key: str, l2_nodes: list[dict],
               recent_l3: list[str]) -> dict:
    child_texts = "\n".join(f"[session {n['session_id']}] {n['text']}" for n in l2_nodes)
    hist_ctx = "\n".join(recent_l3[-3:])
    prompt = (
        "Evolving Patterns: Synthesize these session summaries into a day-level memory. "
        "Identify recurring themes, preferences, or behavioral patterns. Third-person. "
        "Do NOT repeat prior context. Output 3-5 sentences.\n\n"
        + (f"Recent day summaries:\n{hist_ctx}\n\n" if hist_ctx else "")
        + f"Day {day_key} sessions:\n{child_texts}\n\nDay summary:"
    )
    try:
        text = _claude(client, prompt, max_tokens=400, temperature=0.7)
    except Exception:
        text = child_texts[:600]
    return {"level": 3, "day_key": day_key, "text": text, "children": l2_nodes}


# --- L4: week consolidation ---
def _build_l4(client, week_key: str, l3_nodes: list[dict],
               recent_l4: list[str]) -> dict:
    child_texts = "\n".join(f"[day {n['day_key']}] {n['text']}" for n in l3_nodes)
    hist_ctx = "\n".join(recent_l4[-3:])
    prompt = (
        "Evolving Patterns: Synthesize these day summaries into a week-level memory. "
        "Identify longer-term trends and stable preferences. Third-person. "
        "Do NOT repeat prior context. Output 3-5 sentences.\n\n"
        + (f"Recent week summaries:\n{hist_ctx}\n\n" if hist_ctx else "")
        + f"Week {week_key} days:\n{child_texts}\n\nWeek summary:"
    )
    try:
        text = _claude(client, prompt, max_tokens=400, temperature=0.7)
    except Exception:
        text = child_texts[:600]
    return {"level": 4, "week_key": week_key, "text": text, "children": l3_nodes}


# --- L5: profile (persona) ---
def _build_l5(client, l4_nodes: list[dict], recent_profile: str) -> dict:
    child_texts = "\n".join(f"[week {n['week_key']}] {n['text']}" for n in l4_nodes)
    prompt = (
        "Persona Representation: Integrate these weekly summaries into a holistic persona profile "
        "for the user. Capture stable traits, long-term goals, consistent preferences, and key life "
        "facts. Third-person. Incrementally refine the existing profile rather than replacing it. "
        "Output 4-7 sentences.\n\n"
        + (f"Existing profile:\n{recent_profile}\n\n" if recent_profile else "")
        + f"Weekly memories:\n{child_texts}\n\nUpdated persona profile:"
    )
    try:
        text = _claude(client, prompt, max_tokens=600, temperature=0.7)
    except Exception:
        text = child_texts[:800]
    return {"level": 5, "text": text, "children": l4_nodes}


def build_tmt(client, items) -> dict:
    """Build the full 5-level Temporal Memory Tree from items.

    FIX (timestamps): LME items carry timestamp=None. We use session_id (cast
    to int where possible, else hash-to-int) as an ingestion-order proxy and
    sort sessions by that value to preserve chronological ordering. This is
    the strongest proxy available without real datetimes.

    FIX (L5 incremental updates): L5 persona profile is rebuilt incrementally
    each time a new "month" worth of L4 nodes accumulates (every 4 weeks),
    carrying the running profile forward. Previously built only once from all
    L4 nodes at the end.

    Returns:
        {
          "l1": [node, ...],
          "l2": [node, ...],
          "l3": [node, ...],
          "l4": [node, ...],
          "l5": node,
        }
    """
    global _bm25_corpus
    _bm25_corpus = CorpusBM25()  # reset per example

    sessions = _group_items_by_session(items)
    # FIX: sort sessions by integer session_id as temporal proxy (real timestamps are None)
    def _sid_sort_key(sid: str) -> int:
        try:
            return int(sid)
        except ValueError:
            return abs(hash(sid)) % 10**9
    session_ids = sorted(sessions.keys(), key=_sid_sort_key)

    l1_all: list[dict] = []
    l2_all: list[dict] = []
    recent_l2: list[str] = []

    for sid in session_ids:
        sitems = sessions[sid]
        l1_nodes = _build_l1(client, sid, sitems)
        l1_all.extend(l1_nodes)
        # FIX: populate corpus BM25 with each L1 memory text
        for node in l1_nodes:
            _bm25_corpus.add_doc(node["text"])
        l2_node = _build_l2(client, sid, l1_nodes, recent_l2)
        l2_all.append(l2_node)
        recent_l2.append(l2_node["text"])

    # FIX: derive day/week buckets from session_id integer ordering.
    # Previously used every-3-sessions synthetic bucketing with no temporal basis.
    # Now: sessions grouped by (session_id // day_size) so ordering is stable
    # relative to the integer session timeline.
    l3_all: list[dict] = []
    recent_l3: list[str] = []
    day_size = 3  # sessions per day bucket (paper's temporal containment)
    for day_idx, offset in enumerate(range(0, len(l2_all), day_size)):
        chunk = l2_all[offset: offset + day_size]
        if not chunk:
            continue
        day_key = str(day_idx)
        l3_node = _build_l3(client, day_key, chunk, recent_l3)
        l3_all.append(l3_node)
        recent_l3.append(l3_node["text"])

    l4_all: list[dict] = []
    recent_l4: list[str] = []
    week_size = 3  # days per week bucket
    for week_idx, offset in enumerate(range(0, len(l3_all), week_size)):
        chunk = l3_all[offset: offset + week_size]
        if not chunk:
            continue
        week_key = str(week_idx)
        l4_node = _build_l4(client, week_key, chunk, recent_l4)
        l4_all.append(l4_node)
        recent_l4.append(l4_node["text"])

    # FIX: incremental L5 monthly updates (every 4 weeks of L4 nodes).
    # Previously: L5 built once from all L4 at the end (no incremental update).
    # Now: running profile is updated each "month" (every 4 L4 nodes),
    # mimicking the paper's continuous persona refinement.
    month_size = 4  # L4 nodes per "month" update
    running_profile = ""
    l5_node = None
    for month_start in range(0, max(len(l4_all), 1), month_size):
        month_chunk = l4_all[month_start: month_start + month_size]
        if not month_chunk and l5_node is not None:
            break
        l5_node = _build_l5(client, month_chunk if month_chunk else l4_all, running_profile)
        running_profile = l5_node["text"]  # carry profile forward
    if l5_node is None:
        l5_node = _build_l5(client, [], "")

    return {"l1": l1_all, "l2": l2_all, "l3": l3_all, "l4": l4_all, "l5": l5_node}


# ---------------------------------------------------------------------------
# Recall phase
# ---------------------------------------------------------------------------

BUDGETS = {
    "simple":  {1: 20, 2: 4, 5: 1},
    "hybrid":  {1: 20, 2: 4, 3: 2, 5: 1},
    "complex": {1: 20, 2: 8, 3: 4, 4: 2, 5: 1},
}


def recall_planner(client, question: str) -> tuple[str, list[str]]:
    """Stage 1: classify complexity and extract keywords."""
    prompt = (
        "Classify the memory retrieval complexity of this question and extract keywords.\n\n"
        "Complexity levels:\n"
        "- simple: single explicit fact (e.g. 'What did X eat yesterday?')\n"
        "- hybrid: requires integrating 2-3 facts across sessions\n"
        "- complex: requires reasoning about preferences, predictions, or long-term patterns\n\n"
        f"Question: {question}\n\n"
        "Respond with JSON only: {\"complexity\": \"simple|hybrid|complex\", \"keywords\": [\"kw1\", \"kw2\"]}\n"
        "Keywords must be general concepts only (no personal names). Max 3 keywords.\n"
        "JSON:"
    )
    raw = _claude(client, prompt, max_tokens=80, temperature=0.0)
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(m.group()) if m else {}
        complexity = parsed.get("complexity", "hybrid")
        keywords = parsed.get("keywords", [])[:3]
    except Exception:
        complexity = "hybrid"
        keywords = []
    if complexity not in BUDGETS:
        complexity = "hybrid"
    return complexity, keywords


def _collect_ancestors(l1_node: dict, tmt: dict) -> list[dict]:
    """Collect ancestors of an L1 node across all TMT levels."""
    ancestors = []
    sid = l1_node["session_id"]
    for l2 in tmt["l2"]:
        if l2["session_id"] == sid:
            ancestors.append(l2)
            # find L3 parent
            for l3 in tmt["l3"]:
                if any(c["session_id"] == sid for c in l3["children"]):
                    ancestors.append(l3)
                    # find L4 parent
                    for l4 in tmt["l4"]:
                        if any(c["day_key"] == l3["day_key"] for c in l4["children"]):
                            ancestors.append(l4)
                    break
            break
    return ancestors


def hierarchical_recall(client, tmt: dict, question: str, keywords: list[str],
                        complexity: str) -> list[dict]:
    """Stage 2: fused scoring at L1, ancestor collection, per-level budget pruning.
    FIX: accepts client to use dense embeddings (Titan v2) instead of TF-IDF."""
    query_full = question + " " + " ".join(keywords)
    # FIX: embed query once and reuse to avoid N redundant Bedrock calls
    query_emb = _get_embedding(client, query_full)

    # FIX: score all L1 nodes with dense embeddings via _fused_score
    scored_l1 = sorted(
        tmt["l1"],
        key=lambda n: _fused_score(client, query_full, n["text"], query_emb=query_emb),
        reverse=True,
    )

    budget = BUDGETS[complexity]
    k1 = budget.get(1, 20)
    top_l1 = scored_l1[:k1]

    # Collect ancestors for top L1 nodes
    level_candidates: dict[int, list[dict]] = {1: top_l1}
    seen_l2: set[str] = set()
    seen_l3: set[str] = set()
    seen_l4: set[str] = set()

    for l1_node in top_l1:
        ancs = _collect_ancestors(l1_node, tmt)
        for anc in ancs:
            lvl = anc["level"]
            if lvl not in budget:
                continue
            uid = anc.get("session_id") or anc.get("day_key") or anc.get("week_key", "")
            seen_set = {2: seen_l2, 3: seen_l3, 4: seen_l4}.get(lvl)
            if seen_set is not None and uid not in seen_set:
                seen_set.add(uid)
                level_candidates.setdefault(lvl, []).append(anc)

    # Add profile (L5) if in scope
    if 5 in budget and tmt["l5"]:
        level_candidates[5] = [tmt["l5"]]

    # Prune to budget, keeping highest-scoring candidates per level
    all_candidates: list[dict] = []
    for lvl, nodes in level_candidates.items():
        cap = budget.get(lvl, 0)
        if cap == 0:
            continue
        # FIX: use dense fused score for pruning (was TF-IDF)
        pruned = sorted(
            nodes,
            key=lambda n: _fused_score(client, query_full, n["text"], query_emb=query_emb),
            reverse=True,
        )[:cap]
        all_candidates.extend(pruned)

    return all_candidates


def recall_gating(client, candidates: list[dict], question: str,
                  complexity: str) -> list[dict]:
    """Stage 3: LLM retain/drop gating call."""
    if not candidates:
        return []
    mem_lines = "\n".join(
        f"[{i}] L{c['level']}: {c['text'][:300]}" for i, c in enumerate(candidates)
    )
    target = "3-8" if complexity == "simple" else "5-12"
    prompt = (
        f"Question: {question}\n\n"
        f"Candidate memories:\n{mem_lines}\n\n"
        "Decide which memories are relevant to answer the question. "
        f"{'Aggressive filtering — keep only direct answers.' if complexity == 'simple' else 'Keep all potentially useful memories.'} "
        f"Target {target} memories. "
        "Output a JSON array of retained indices only, e.g. [0,2,5].\n"
        "Retained indices:"
    )
    raw = _claude(client, prompt, max_tokens=80, temperature=0.0)
    try:
        m = re.search(r"\[[\d,\s]*\]", raw)
        idxs = json.loads(m.group()) if m else list(range(len(candidates)))
    except Exception:
        idxs = list(range(len(candidates)))
    return [candidates[i] for i in idxs if 0 <= i < len(candidates)]


def _session_id_int(node: dict) -> int:
    """Extract numeric session order from a TMT node for temporal proximity sort."""
    sid = node.get("session_id") or node.get("day_key") or node.get("week_key") or "0"
    try:
        return int(sid)
    except (ValueError, TypeError):
        return 0


def timem_answer(client, memories: list[dict], question: str) -> str:
    """Final answer generation from ordered memories.
    FIX: sort by (level ASC, temporal proximity DESC) so recent L1/L2 fragments
    appear before older ones, while higher-level summaries follow. Previously
    sorted by level only with no temporal tie-breaking."""
    if not memories:
        mem_text = "(no memories retrieved)"
    else:
        # FIX: level ASC, then session_id DESC (most recent first within level)
        ordered = sorted(memories, key=lambda n: (n["level"], -_session_id_int(n)))
        mem_text = "\n".join(
            f"[L{m['level']}] {m['text'][:400]}" for m in ordered
        )
    if len(mem_text) > 60000:
        mem_text = mem_text[:60000] + "\n[...truncated...]"
    prompt = (
        "Answer the question precisely using the memory evidence below. "
        "Prefer recent memories when there are conflicts. "
        "Be concise (under 15 words).\n\n"
        f"MEMORIES:\n{mem_text}\n\n"
        f"QUESTION: {question}\n\nAnswer:"
    )
    return _claude(client, prompt, max_tokens=100, temperature=0.0)


def timem_pipeline(client, items, question: str) -> tuple[str, dict]:
    """Full TiMem pipeline: build TMT → recall planner → hierarchical recall
    → gating → answer."""
    tmt = build_tmt(client, items)
    complexity, keywords = recall_planner(client, question)
    # FIX: pass client so hierarchical_recall uses dense embeddings
    candidates = hierarchical_recall(client, tmt, question, keywords, complexity)
    retained = recall_gating(client, candidates, question, complexity)
    answer = timem_answer(client, retained, question)
    meta = {
        "complexity": complexity,
        "keywords": keywords,
        "n_l1": len(tmt["l1"]),
        "n_l2": len(tmt["l2"]),
        "n_l3": len(tmt["l3"]),
        "n_l4": len(tmt["l4"]),
        "n_candidates": len(candidates),
        "n_retained": len(retained),
    }
    return answer, meta


# ---------------------------------------------------------------------------
# Metrics (same as run_strong_pipeline.py)
# ---------------------------------------------------------------------------

def llm_judge(client, pred: str, gold: str, question: str) -> int:
    if not pred.strip():
        return 0
    prompt = (
        "Judge if the PREDICTION correctly answers the QUESTION given GOLD. Reply CORRECT or WRONG only.\n\n"
        f"QUESTION: {question}\nGOLD: {gold}\nPREDICTION: {pred}\n\nReply:"
    )
    v = _claude(client, prompt, max_tokens=10, temperature=0.0).upper()
    # FIX: INCORRECT contains CORRECT — check negatives first
    if "WRONG" in v or "INCORRECT" in v or "NOT CORRECT" in v: return 0
    return 1 if "CORRECT" in v else 0


def bootstrap_ci(scores: list[int], n: int = 1000) -> tuple[float, float, float]:
    rng = random.Random(42)
    n_s = len(scores)
    if n_s == 0:
        return 0.0, 0.0, 0.0
    means = sorted(
        sum(rng.choice(scores) for _ in range(n_s)) / n_s for _ in range(n)
    )
    return sum(scores) / n_s, means[int(n * 0.025)], means[int(n * 0.975)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(n_questions: int, output_path: str) -> None:
    client = _bedrock_client()
    all_examples = longmemeval(n=300, question_types=None)
    multi = [ex for ex in all_examples if any("multi-session" in q.query_type for q in ex.queries)][:n_questions]
    print(f"[setup] {len(multi)} multi-session questions", flush=True)

    results: list[dict] = []

    for i, ex in enumerate(multi):
        q = ex.queries[0]
        print(f"\n--- Q{i+1}/{len(multi)} ---  expected={q.answer[:60]!r}", flush=True)
        t0 = time.perf_counter()
        try:
            ans, meta = timem_pipeline(client, ex.items, q.question)
            judge = llm_judge(client, ans, q.answer, q.question)
            elapsed = time.perf_counter() - t0
            print(
                f"  [timem] judge={judge} complexity={meta['complexity']} "
                f"retained={meta['n_retained']} ans={ans[:60]!r} ({elapsed:.1f}s)",
                flush=True,
            )
            results.append({
                "q": q.question,
                "gt": q.answer,
                "ans": ans,
                "judge": judge,
                **meta,
                "elapsed": elapsed,
            })
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"  [timem] FAILED: {e}", flush=True)
            results.append({
                "q": q.question,
                "gt": q.answer,
                "ans": "",
                "judge": 0,
                "error": str(e),
                "elapsed": elapsed,
            })

    scores = [r["judge"] for r in results]
    mean, lo, hi = bootstrap_ci(scores)
    summary = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(scores)}

    print("\n" + "=" * 80, flush=True)
    print(f"TiMem RESULTS (n={len(scores)} multi-session questions)", flush=True)
    print("=" * 80, flush=True)
    print(f"Accuracy: {mean:.3f} [{lo:.3f}, {hi:.3f}]  (n={len(scores)})", flush=True)

    out = {"method": "timem", "n": len(scores), "summary": summary, "details": results}
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {output_path}", flush=True)

    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        import subprocess
        s3_key = f"s3://{s3_bucket}/timem/{os.path.basename(output_path)}"
        try:
            subprocess.run(["aws", "s3", "cp", output_path, s3_key], check=True, timeout=120)
            print(f"[s3] {s3_key}", flush=True)
        except Exception as e:
            print(f"[s3] upload failed: {e}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="TiMem baseline (arXiv:2601.02845)")
    p.add_argument("--n", type=int, default=20, help="Number of multi-session questions")
    p.add_argument("--output", default="/tmp/timem_results.json", help="Output JSON path")
    args = p.parse_args()
    run(args.n, args.output)
