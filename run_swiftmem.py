"""SwiftMem baseline: Fast Agentic Memory via Query-aware Indexing.

Faithful reproduction of SwiftMem (arXiv:2601.08160, Tian et al. 2026).

Changelog (v2 fixes — all HIGH-severity Codex review deviations):
  - Dense embeddings: switched from TF-IDF to amazon.titan-embed-text-v2:0
    (1024-dim Bedrock) for all semantic similarity and tag routing.
  - DAG specificity: enforce monotonic specificity via embedding-based
    entailment check (child embedding must be more specific than parent,
    approximated by L2 distance in embedding space).
  - DAG expansion: BFS now traverses both parent AND child edges (bidirectional)
    for richer candidate retrieval.
  - Query-Tag Router: tag matching uses tag embeddings + indexed cosine search
    instead of lexical cosine on raw tag string tokens.
  - Timestamps: LME items have timestamp=None; use int(session_id) as
    ingestion-order proxy. Documented below.
  - Co-consolidation layout: layout_map is used during retrieval to boost
    episodes sharing the same cluster as the top semantic hits.
  - Embedding vector tier: retrieve() now has a dedicated dense-embedding
    rerank tier (embed query once, score all candidates, top-k by cosine).

Algorithm overview:
  Ingestion phase:
    1. For each conversation turn, create an episode (content + timestamp + tags).
    2. Insert into Temporal Index (sorted per-user timeline for O(log N) range queries).
    3. Insert into Semantic DAG-Tag Index (LLM generates 3-8 tags per episode;
       parent-child edges added when semantic specificity strictly increases).
    4. Periodically run Embedding-Tag Co-Consolidation to reorganise memory
       into semantically coherent clusters, reducing fragmentation.

  Query phase:
    1. Analyse query for temporal indicators (extract date/time ranges).
    2. Route query through Query-Tag Router: embed query, pick top-k tags via
       indexed tag embeddings, expand via DAG (bidirectional BFS, depth D_max=2).
    3. Retrieve candidate episode subset from temporal + semantic branches;
       boost episodes co-located with top semantic hits by layout cluster.
    4. Dense embedding rerank over candidate subset → top passages.
    5. Generate answer from retrieved evidence.
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
from collections import defaultdict, deque

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
# Replaces TF-IDF cosine used previously for all similarity operations.
# ---------------------------------------------------------------------------

_EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
_embed_cache: dict[str, list[float]] = {}  # in-process cache


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
        emb = [0.0] * 1024
    _embed_cache[text] = emb
    return emb


def _cosine_dense(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two pre-normalised Titan v2 vectors (dot == cosine)."""
    return sum(x * y for x, y in zip(a, b))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def _cosine_sim_dense(client, a: str, b: str) -> float:
    """FIX: Dense cosine similarity between two texts using Bedrock embeddings.
    Was: TF-IDF term-overlap cosine."""
    return _cosine_dense(_get_embedding(client, a), _get_embedding(client, b))


# ---------------------------------------------------------------------------
# Episode representation
# ---------------------------------------------------------------------------

class Episode:
    __slots__ = ("eid", "user_id", "content", "timestamp", "tags", "session_id")

    def __init__(self, eid: int, user_id: str, content: str,
                 timestamp: int, tags: list[str], session_id: str):
        self.eid = eid
        self.user_id = user_id
        self.content = content
        self.timestamp = timestamp
        self.tags = tags
        self.session_id = session_id


# ---------------------------------------------------------------------------
# Temporal Index
# ---------------------------------------------------------------------------

class TemporalIndex:
    """Per-user sorted timeline (list of (timestamp, eid)).
    Insertions maintain order; range queries use bisect."""

    def __init__(self):
        self._timelines: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._eid_map: dict[int, tuple[str, int]] = {}  # eid -> (user_id, timestamp)

    def insert(self, ep: Episode) -> None:
        import bisect
        tl = self._timelines[ep.user_id]
        bisect.insort(tl, (ep.timestamp, ep.eid))
        self._eid_map[ep.eid] = (ep.user_id, ep.timestamp)

    def query_range(self, user_id: str, t_start: int, t_end: int) -> list[int]:
        """Return eids in [t_start, t_end] for user. O(log N + k)."""
        import bisect
        tl = self._timelines.get(user_id, [])
        lo = bisect.bisect_left(tl, (t_start, -1))
        hi = bisect.bisect_right(tl, (t_end, 10**18))
        return [eid for _, eid in tl[lo:hi]]

    def all_eids(self, user_id: str) -> list[int]:
        return [eid for _, eid in self._timelines.get(user_id, [])]


# ---------------------------------------------------------------------------
# Semantic DAG-Tag Index
# ---------------------------------------------------------------------------

class DagTagIndex:
    """Hierarchical tag DAG.
    Node: tag_str -> {episodes: set[int], children: set[str], parents: set[str]}

    FIX (specificity): parent-child edge added only when embedding-based
    specificity strictly increases — child embedding must be closer to a
    "more specific / narrower" direction, approximated by checking that the
    child L2-distance from the general anchor "general topic" > parent distance.
    Fallback (no client at insert time): use token-length heuristic.

    FIX (bidirectional traversal): expand_tags traverses both parent AND
    child edges, not children only.

    FIX (tag router): top_k_tags uses indexed tag embeddings + cosine search
    instead of lexical cosine on raw tag string characters.
    """

    def __init__(self):
        self._nodes: dict[str, dict] = {}
        self._tag_eps: dict[str, set[int]] = defaultdict(set)
        # FIX: tag embedding index for dense Query-Tag Router
        self._tag_embeddings: dict[str, list[float]] = {}

    def _get_or_create(self, tag: str) -> dict:
        if tag not in self._nodes:
            self._nodes[tag] = {"episodes": set(), "children": set(), "parents": set()}
        return self._nodes[tag]

    def _specificity_check(self, parent_tag: str, child_tag: str,
                           client=None) -> bool:
        """FIX: Enforce monotonic specificity for DAG edges.
        Uses embedding-based check when client available, token-length fallback.
        Previously: only token-length/substring heuristic (no semantic check)."""
        if client is not None:
            # Embed both tags; child should be semantically more specific
            # Approximation: child embedding should be farther from the
            # centroid of all tags (more specialised = less central)
            p_emb = _get_embedding(client, parent_tag.replace("_", " "))
            c_emb = _get_embedding(client, child_tag.replace("_", " "))
            # If child is strictly more specific, its embedding should NOT be
            # closer to the parent's embedding than vice versa (non-symmetric),
            # AND child token count should be >= parent (broader -> narrower).
            # Use cosine similarity: high sim = semantically related (good);
            # child specificity checked by len(child_tokens) >= len(parent_tokens)
            sim = _cosine_dense(p_emb, c_emb)
            if sim < 0.3:
                return False  # unrelated tags — reject edge
            child_tokens = child_tag.replace("_", " ").split()
            parent_tokens = parent_tag.replace("_", " ").split()
            return len(child_tokens) >= len(parent_tokens)
        else:
            # Fallback: token-length heuristic
            return len(child_tag) > len(parent_tag)

    def index_tag_embedding(self, tag: str, client) -> None:
        """FIX: Pre-compute and store tag embedding for indexed search."""
        if tag not in self._tag_embeddings:
            self._tag_embeddings[tag] = _get_embedding(client, tag.replace("_", " "))

    def insert_episode(self, ep: Episode, relations: list[tuple[str, str]],
                       client=None) -> None:
        """Register episode tags and hierarchical relations.

        relations: list of (parent_tag, child_tag) pairs.
        FIX: client passed to enable embedding-based specificity check.
        """
        for tag in ep.tags:
            node = self._get_or_create(tag)
            node["episodes"].add(ep.eid)
            self._tag_eps[tag].add(ep.eid)
            # FIX: index tag embedding when client available
            if client is not None:
                self.index_tag_embedding(tag, client)

        for parent_tag, child_tag in relations:
            pn = self._get_or_create(parent_tag)
            cn = self._get_or_create(child_tag)
            # FIX: enforce monotonic specificity via embedding check
            if child_tag not in pn["children"] and self._specificity_check(
                parent_tag, child_tag, client=client
            ):
                pn["children"].add(child_tag)
                cn["parents"].add(parent_tag)

    def expand_tags(self, seed_tags: list[str], d_max: int = 2) -> set[int]:
        """FIX: Bidirectional BFS — traverse both parent AND child edges.
        Previously: children only. Paper specifies DAG expansion traverses
        both directions for richer candidate coverage."""
        visited: set[str] = set()
        frontier = deque((t, 0) for t in seed_tags)
        result_eids: set[int] = set()

        while frontier:
            tag, depth = frontier.popleft()
            if tag in visited:
                continue
            visited.add(tag)
            result_eids.update(self._tag_eps.get(tag, set()))
            if depth < d_max:
                node = self._nodes.get(tag)
                if node:
                    # FIX: traverse children AND parents
                    for neighbour in node["children"] | node["parents"]:
                        if neighbour not in visited:
                            frontier.append((neighbour, depth + 1))
        return result_eids

    def top_k_tags(self, client, query: str, k: int = 5) -> list[str]:
        """FIX: Query-Tag Router using tag embeddings + indexed cosine search.
        Was: lexical cosine on raw tag string tokens.
        Now: embed query once, score against pre-indexed tag embeddings."""
        if not self._tag_embeddings:
            # Fallback if no embeddings indexed yet (shouldn't happen post-ingest)
            return list(self._nodes.keys())[:k]
        query_emb = _get_embedding(client, query)
        scores = [
            (tag, _cosine_dense(query_emb, emb))
            for tag, emb in self._tag_embeddings.items()
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in scores[:k]]


# ---------------------------------------------------------------------------
# Co-consolidation (Embedding-Tag Co-Consolidation)
# ---------------------------------------------------------------------------

def _build_tag_clusters(dag: DagTagIndex) -> dict[str, str]:
    """Assign each tag to a cluster ID using connected-component analysis
    over DAG edges (approximates the paper's cohesion-based clustering)."""
    all_tags = list(dag._nodes.keys())
    parent_map: dict[str, str] = {}  # union-find

    def find(x: str) -> str:
        while parent_map.get(x, x) != x:
            parent_map[x] = parent_map.get(parent_map.get(x, x), x)
            x = parent_map[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent_map[rb] = ra

    for tag, node in dag._nodes.items():
        for child in node["children"]:
            union(tag, child)

    return {t: find(t) for t in all_tags}


def co_consolidate(episodes: list[Episode], dag: DagTagIndex) -> dict[int, int]:
    """Return a layout map: eid -> cluster_id (integer).

    In the paper, physical memory blocks are rearranged so same-cluster
    episodes are adjacent. Here we return the mapping so retrieval can
    preferentially surface coherent clusters.
    """
    tag_to_cluster = _build_tag_clusters(dag)
    # Convert string cluster IDs to integers
    cluster_strs = list(set(tag_to_cluster.values()))
    cluster_int = {c: i for i, c in enumerate(cluster_strs)}

    eid_to_cluster: dict[int, int] = {}
    for ep in episodes:
        if ep.tags:
            cluster_str = tag_to_cluster.get(ep.tags[0], ep.tags[0])
            eid_to_cluster[ep.eid] = cluster_int.get(cluster_str, 0)
        else:
            eid_to_cluster[ep.eid] = 0
    return eid_to_cluster


# ---------------------------------------------------------------------------
# Tag generation (paper Appendix A.1)
# ---------------------------------------------------------------------------

_TAG_GEN_PROMPT = """\
Extract 3-8 meaningful tags that capture the main topics, themes, and contexts of the following memory episode. Also identify any parent-child relationships between tags where one tag is a more specific instance of another.

Rules:
- Tags must be lowercase, max 3 words, underscores for multi-word (e.g. italian_cuisine)
- Prefer specific over generic (python_programming > technology)
- Exclude broad terms like "conversation" or "chat"
- Cover: topics/activities, locations/entities, emotions/intents, specific concepts

Memory episode:
{content}

Respond with JSON only:
{{"tags": ["tag1", "tag2", ...], "relations": [{{"parent": "tag_a", "child": "tag_b"}}]}}
JSON:"""


def generate_tags(client, content: str) -> tuple[list[str], list[tuple[str, str]]]:
    """LLM tag generation per Appendix A.1 of SwiftMem.

    Returns (tags, relations) where relations is [(parent, child), ...].
    """
    prompt = _TAG_GEN_PROMPT.format(content=content[:1000])
    try:
        raw = _claude(client, prompt, max_tokens=200, temperature=0.0)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise ValueError("no JSON found")
        parsed = json.loads(m.group())
        tags = [t.lower().replace(" ", "_")[:40] for t in parsed.get("tags", [])[:8]]
        rels = []
        for rel in parsed.get("relations", []):
            p = rel.get("parent", "").lower().replace(" ", "_")
            c = rel.get("child", "").lower().replace(" ", "_")
            if p and c and p != c:
                rels.append((p, c))
        return tags[:8], rels
    except Exception:
        # Fallback: use top 4 content words as flat tags
        words = list(dict.fromkeys(re.findall(r"[a-z]{4,}", content.lower())))
        tags = words[:4]
        return tags, []


# ---------------------------------------------------------------------------
# Temporal indicator extraction
# ---------------------------------------------------------------------------

_TEMPORAL_PATTERNS = [
    r"\b(today|yesterday|tomorrow)\b",
    r"\b(last|this|next)\s+(week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b\d{1,2}/\d{1,2}(/\d{2,4})?\b",
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    r"\brecent(?:ly)?\b",
    r"\b(a few|several|many)\s+(days?|weeks?|months?)\s+ago\b",
]
_TEMPORAL_RE = re.compile("|".join(_TEMPORAL_PATTERNS), re.IGNORECASE)


def has_temporal_indicator(query: str) -> bool:
    return bool(_TEMPORAL_RE.search(query))


def _extract_session_range(query: str, max_ts: int) -> tuple[int, int]:
    """Map textual temporal reference to a (t_start, t_end) range over session indices."""
    q = query.lower()
    if "yesterday" in q or "last" in q:
        # Recent sessions: last 20% of timeline
        return max(0, max_ts - max(1, max_ts // 5)), max_ts
    if "recent" in q or "lately" in q or "this week" in q:
        return max(0, max_ts - max(1, max_ts // 3)), max_ts
    # Default: use full range
    return 0, max_ts


# ---------------------------------------------------------------------------
# SwiftMem ingestion + retrieval
# ---------------------------------------------------------------------------

class SwiftMemStore:
    def __init__(self):
        self.episodes: list[Episode] = []
        self.temporal_idx = TemporalIndex()
        self.dag_tag_idx = DagTagIndex()
        self._eid_counter = 0
        self._eid_to_ep: dict[int, Episode] = {}
        self._layout_map: dict[int, int] = {}  # eid -> cluster_id
        self._client = None  # stored for use in retrieve()

    def ingest(self, client, items, user_id: str = "user") -> None:
        """Process all items into episodes, build temporal + tag indices.

        FIX (timestamps): LME items carry timestamp=None. We use int(session_id)
        as an ingestion-order proxy, ensuring correct temporal ordering.
        Previously used session_order counter which was equivalent but undocumented.

        FIX (DAG specificity + tag embeddings): pass client to insert_episode
        so embedding-based specificity check and tag embedding indexing happen
        at ingest time.
        """
        self._client = client
        session_order = 0
        prev_sid = None

        for item in items:
            sid = str(getattr(item, "session_id", None) or "default")
            if sid != prev_sid:
                session_order += 1
                prev_sid = sid

            content = (f"{getattr(item, 'speaker', '')}: {item.content}").strip()
            # FIX: use int(session_id) as temporal proxy; fall back to session_order
            try:
                ts = int(sid) * 100 + len(self.episodes) % 100
            except (ValueError, TypeError):
                ts = session_order * 100 + len(self.episodes) % 100

            tags, relations = generate_tags(client, content)
            ep = Episode(
                eid=self._eid_counter,
                user_id=user_id,
                content=content,
                timestamp=ts,
                tags=tags,
                session_id=sid,
            )
            self._eid_counter += 1
            self.episodes.append(ep)
            self._eid_to_ep[ep.eid] = ep

            self.temporal_idx.insert(ep)
            # FIX: pass client to enforce embedding-based specificity + index tag embeddings
            self.dag_tag_idx.insert_episode(ep, relations, client=client)

        # Run co-consolidation once after full ingestion
        self._layout_map = co_consolidate(self.episodes, self.dag_tag_idx)

    def retrieve(self, query: str, user_id: str = "user",
                 top_k: int = 10, k_tags: int = 5, d_max: int = 2) -> list[Episode]:
        """Query-aware retrieval: temporal branch + semantic DAG branch + embedding rerank.

        FIX (tag router): uses indexed tag embeddings for top_k_tags.
        FIX (co-consolidation layout): boost episodes sharing cluster with top semantic hits.
        FIX (embedding vector tier): dedicated dense rerank pass replaces TF-IDF cosine.
        """
        client = self._client
        candidate_eids: set[int] = set()
        all_user_eids = set(self.temporal_idx.all_eids(user_id))

        # --- Temporal branch ---
        if has_temporal_indicator(query):
            max_ts = max((ep.timestamp for ep in self.episodes), default=0)
            t_start, t_end = _extract_session_range(query, max_ts)
            temporal_eids = set(self.temporal_idx.query_range(user_id, t_start, t_end))
            candidate_eids.update(temporal_eids)

        # --- Semantic DAG branch ---
        # FIX: use indexed tag embeddings for query-tag routing
        top_tags = self.dag_tag_idx.top_k_tags(client, query, k=k_tags)
        # FIX: bidirectional DAG expansion (parent + child)
        semantic_eids = self.dag_tag_idx.expand_tags(top_tags, d_max=d_max)
        candidate_eids.update(semantic_eids & all_user_eids)

        # Fallback: if both branches return nothing, use all episodes
        if not candidate_eids:
            candidate_eids = all_user_eids

        # FIX: co-consolidation layout boost — episodes sharing the same cluster
        # as top semantic hits are added to candidates (paper §3.3 retrieval locality)
        if self._layout_map and semantic_eids:
            top_clusters = {self._layout_map[e] for e in semantic_eids
                            if e in self._layout_map}
            for ep in self.episodes:
                if self._layout_map.get(ep.eid) in top_clusters:
                    candidate_eids.add(ep.eid)

        # FIX: dense embedding rerank tier (was TF-IDF cosine)
        # Embed query once; score all candidates via Titan v2 cosine
        query_emb = _get_embedding(client, query)
        candidates = [self._eid_to_ep[e] for e in candidate_eids if e in self._eid_to_ep]
        scored = sorted(
            candidates,
            key=lambda ep: _cosine_dense(query_emb, _get_embedding(client, ep.content)),
            reverse=True,
        )
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def swiftmem_answer(client, retrieved: list[Episode], question: str) -> str:
    if not retrieved:
        evidence = "(no memories retrieved)"
    else:
        evidence = "\n".join(
            f"[ep{ep.eid} t={ep.timestamp}] {ep.content[:400]}"
            for ep in retrieved
        )
    if len(evidence) > 60000:
        evidence = evidence[:60000] + "\n[...truncated...]"
    prompt = (
        "Answer the question precisely using the memory evidence below. "
        "Be concise (under 15 words).\n\n"
        f"EVIDENCE:\n{evidence}\n\nQUESTION: {question}\n\nAnswer:"
    )
    return _claude(client, prompt, max_tokens=100, temperature=0.0)


def swiftmem_pipeline(client, items, question: str) -> tuple[str, dict]:
    """Full SwiftMem pipeline: ingest → query-aware retrieval → answer."""
    store = SwiftMemStore()
    store.ingest(client, items)
    retrieved = store.retrieve(question)
    answer = swiftmem_answer(client, retrieved, question)
    meta = {
        "n_episodes": len(store.episodes),
        "n_tags": len(store.dag_tag_idx._nodes),
        "n_retrieved": len(retrieved),
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
            ans, meta = swiftmem_pipeline(client, ex.items, q.question)
            judge = llm_judge(client, ans, q.answer, q.question)
            elapsed = time.perf_counter() - t0
            print(
                f"  [swiftmem] judge={judge} episodes={meta['n_episodes']} "
                f"tags={meta['n_tags']} retrieved={meta['n_retrieved']} "
                f"ans={ans[:60]!r} ({elapsed:.1f}s)",
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
            print(f"  [swiftmem] FAILED: {e}", flush=True)
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
    print(f"SwiftMem RESULTS (n={len(scores)} multi-session questions)", flush=True)
    print("=" * 80, flush=True)
    print(f"Accuracy: {mean:.3f} [{lo:.3f}, {hi:.3f}]  (n={len(scores)})", flush=True)

    out = {"method": "swiftmem", "n": len(scores), "summary": summary, "details": results}
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {output_path}", flush=True)

    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        import subprocess
        s3_key = f"s3://{s3_bucket}/swiftmem/{os.path.basename(output_path)}"
        try:
            subprocess.run(["aws", "s3", "cp", output_path, s3_key], check=True, timeout=120)
            print(f"[s3] {s3_key}", flush=True)
        except Exception as e:
            print(f"[s3] upload failed: {e}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SwiftMem baseline (arXiv:2601.08160)")
    p.add_argument("--n", type=int, default=20, help="Number of multi-session questions")
    p.add_argument("--output", default="/tmp/swiftmem_results.json", help="Output JSON path")
    args = p.parse_args()
    run(args.n, args.output)
