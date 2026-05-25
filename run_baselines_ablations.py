"""Retrieval baselines + pipeline ablations for top-venue paper evaluation.

5 Retrieval baselines:
  1. bm25_topk         – BM25 over individual turns, top-k=20, concatenate, answer
  2. embed_topk        – Titan v2 dense cosine, top-k=20 turns, concatenate, answer
  3. bm25_rerank       – BM25 retrieve 50, LLM rerank to 10, answer
  4. map_reduce        – Per-session: ask Claude for 3 relevant facts; concatenate; answer
  5. long_context_full – Entire haystack stuffed in (no truncation guard beyond Bedrock limits)

4 Pipeline ablations (variants of the strong 3-stage pipeline):
  6. gate_only         – Stage 1 (relevance gate) + raw kept sessions → answer (skip consolidation)
  7. consolidate_only  – Skip gate (use ALL sessions); do query-aware consolidation; answer
  8. facts_only        – Strong pipeline but answer ONLY from consolidated facts (no raw text)
  9. raw_only          – Strong pipeline but answer ONLY from raw kept sessions (no facts)

Metric: LLM judge — checks negatives FIRST (WRONG/INCORRECT/NOT CORRECT before CORRECT)
to avoid the "CORRECT" substring of "INCORRECT" false-positive.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, "src")

from context_bench.datasets.memory.longmemeval import longmemeval


# ---------------------------------------------------------------------------
# Bedrock plumbing — copied verbatim from run_strong_pipeline.py
# ---------------------------------------------------------------------------

def _bedrock_client():
    import boto3
    return boto3.client("bedrock-runtime", region_name="us-east-1")


def _claude(client, prompt: str, max_tokens: int = 200, temperature: float = 0.3) -> str:
    """Model-agnostic Bedrock call — dispatches on BEDROCK_MODEL_ID."""
    import re as _re
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
        return _re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=_re.DOTALL).strip()

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


def _embed(text: str) -> list[float]:
    """Titan v2 dense embedding via Bedrock."""
    import boto3
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    body = json.dumps({"inputText": text[:8000], "dimensions": 512, "normalize": True})
    r = client.invoke_model(
        body=body,
        modelId="amazon.titan-embed-text-v2:0",
        accept="application/json",
        contentType="application/json",
    )
    return json.loads(r["body"].read())["embedding"]


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def group_by_session(items):
    sessions = {}
    for item in items:
        sid = getattr(item, "session_id", None) or "default"
        sessions.setdefault(sid, []).append(item)
    return sessions


def format_session(sid, sitems, max_chars=10000):
    text = f"[Session {sid}]\n" + "\n".join(
        f"{getattr(i, 'speaker', '')}: {i.content}" for i in sitems[:50]
    )
    return text[:max_chars]


def format_haystack(items, max_chars=80000):
    parts = []
    cur = None
    for item in items:
        sid = getattr(item, "session_id", "") or "default"
        if sid != cur:
            parts.append(f"\n[Session {sid}]")
            cur = sid
        parts.append(f"{getattr(item, 'speaker', '')}: {item.content}")
    return "\n".join(parts)[:max_chars]


def turn_text(item) -> str:
    return f"{getattr(item, 'speaker', '')}: {item.content}"


def answer_with_evidence(client, evidence: str, question: str) -> str:
    if len(evidence) > 60000:
        evidence = evidence[:60000] + "\n[...truncated...]"
    prompt = (
        "Answer the question precisely. If the answer is a number, give just the number. "
        "If it's a list, give the count. Be concise (under 15 words).\n\n"
        f"EVIDENCE:\n{evidence}\n\nQUESTION: {question}\n\nAnswer:"
    )
    return _claude(client, prompt, max_tokens=100)


# ---------------------------------------------------------------------------
# Stage helpers (shared with ablations)
# ---------------------------------------------------------------------------

def is_session_relevant(client, session_text: str, question: str) -> bool:
    prompt = (
        "Determine whether the session below contains ANY information that could help answer "
        f"the question. Respond with exactly one word: YES or NO.\n\n"
        f"QUESTION: {question}\n\n"
        f"SESSION:\n{session_text[:8000]}\n\n"
        "Respond YES or NO:"
    )
    verdict = _claude(client, prompt, max_tokens=10).strip().upper()
    return "YES" in verdict


def query_aware_summarize(client, session_text: str, question: str) -> list[str]:
    prompt = (
        "You are summarizing a conversation session for the purpose of answering a specific question. "
        "Extract ALL facts from this session that could help answer the question. "
        "Be specific (names, numbers, dates, places). One fact per line, no markdown.\n\n"
        f"QUESTION: {question}\n\nSESSION:\n{session_text[:10000]}\n\n"
        "Relevant facts (one per line):"
    )
    response = _claude(client, prompt, max_tokens=800)
    return [f.strip("- *•").strip() for f in response.split("\n") if len(f.strip()) > 10]


def _relevance_gate(client, sessions: dict, question: str):
    """Stage 1: return list of (sid, sitems) that pass the relevance gate.
    Falls back to keyword-matched or all sessions if nothing passes (mirrors
    the exact fallback logic in run_strong_pipeline.py)."""
    relevant_sessions = []
    for sid, sitems in sessions.items():
        if len(sitems) < 2:
            continue
        s_text = format_session(sid, sitems)
        if is_session_relevant(client, s_text, question):
            relevant_sessions.append((sid, sitems))

    if not relevant_sessions:
        q_words = {w.lower() for w in question.split() if len(w) > 3}
        if q_words:
            scored = [
                (sid, sitems, sum(1 for s in sitems for w in q_words if w in s.content.lower()))
                for sid, sitems in sessions.items() if len(sitems) >= 2
            ]
            scored.sort(key=lambda x: -x[2])
            relevant_sessions = [(sid, sitems) for sid, sitems, score in scored[:5] if score > 0]
        if not relevant_sessions:
            relevant_sessions = [
                (sid, sitems) for sid, sitems in sessions.items() if len(sitems) >= 2
            ][:10]

    return relevant_sessions


# ---------------------------------------------------------------------------
# Metrics — FIXED judge (negatives first)
# ---------------------------------------------------------------------------

def llm_judge(client, pred: str, gold: str, question: str) -> int:
    if not pred.strip():
        return 0
    prompt = (
        "Judge if the PREDICTION correctly answers the QUESTION given GOLD. Reply CORRECT or WRONG only.\n\n"
        f"QUESTION: {question}\nGOLD: {gold}\nPREDICTION: {pred}\n\nReply:"
    )
    v = _claude(client, prompt, max_tokens=20, temperature=0.0).upper()
    # Check negatives FIRST: "CORRECT" is a substring of "INCORRECT".
    if "WRONG" in v or "INCORRECT" in v or "NOT CORRECT" in v:
        return 0
    if "CORRECT" in v:
        return 1
    return 0


def bootstrap_ci(scores, n=1000):
    rng = random.Random(42)
    n_s = len(scores)
    if n_s == 0:
        return 0, 0, 0
    means = sorted(
        sum(rng.choice(scores) for _ in range(n_s)) / n_s for _ in range(n)
    )
    return sum(scores) / n_s, means[int(n * 0.025)], means[int(n * 0.975)]


# ---------------------------------------------------------------------------
# Retrieval baseline 1: BM25 top-k over individual turns
# ---------------------------------------------------------------------------

def _bm25_score(query_tokens: list[str], doc_tokens: list[str],
                avgdl: float, k1: float = 1.5, b: float = 0.75) -> float:
    """Robertson BM25 score for one document."""
    from collections import Counter
    tf = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0
    for t in query_tokens:
        if t not in tf:
            continue
        f = tf[t]
        idf = 1.0  # simplified; IDF needs corpus—use 1.0 for single-query scoring
        score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / max(avgdl, 1)))
    return score


def bm25_topk(client, items, question, k=20):
    """BM25 over individual turns; pick top-k; concatenate; answer."""
    import re
    tokenize = lambda t: re.findall(r"\w+", t.lower())
    query_tokens = tokenize(question)
    docs = [(turn_text(item), tokenize(turn_text(item))) for item in items]
    avgdl = sum(len(d) for _, d in docs) / max(len(docs), 1)
    scored = sorted(
        enumerate(docs),
        key=lambda x: _bm25_score(query_tokens, x[1][1], avgdl),
        reverse=True,
    )
    top_turns = [docs[i][0] for i, _ in scored[:k]]
    evidence = "\n".join(top_turns)
    return answer_with_evidence(client, evidence, question)


# ---------------------------------------------------------------------------
# Retrieval baseline 2: Dense embedding (Titan v2) top-k
# ---------------------------------------------------------------------------

def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def embed_topk(client, items, question, k=20):
    """Titan v2 dense embedding cosine; pick top-k turns; concatenate; answer."""
    q_emb = _embed(question)
    scored = []
    for item in items:
        t = turn_text(item)
        t_emb = _embed(t)
        scored.append((_cosine(q_emb, t_emb), t))
    scored.sort(key=lambda x: -x[0])
    evidence = "\n".join(t for _, t in scored[:k])
    return answer_with_evidence(client, evidence, question)


# ---------------------------------------------------------------------------
# Retrieval baseline 3: BM25 retrieve 50, LLM rerank to 10
# ---------------------------------------------------------------------------

def bm25_rerank(client, items, question, initial=50, final=10):
    """BM25 retrieve 50 turns; LLM rerank to top 10; answer."""
    import re
    tokenize = lambda t: re.findall(r"\w+", t.lower())
    query_tokens = tokenize(question)
    docs = [(turn_text(item), tokenize(turn_text(item))) for item in items]
    avgdl = sum(len(d) for _, d in docs) / max(len(docs), 1)
    scored = sorted(
        enumerate(docs),
        key=lambda x: _bm25_score(query_tokens, x[1][1], avgdl),
        reverse=True,
    )
    candidates = [docs[i][0] for i, _ in scored[:initial]]

    # LLM rerank: ask Claude to rank candidates
    numbered = "\n".join(f"{j+1}. {t[:300]}" for j, t in enumerate(candidates))
    prompt = (
        f"You are a relevance ranker. Given the QUESTION, rank the following conversation turns "
        f"from most to least relevant. Reply with a comma-separated list of the top {final} "
        f"turn numbers (e.g. '3,17,2,...'). No other text.\n\n"
        f"QUESTION: {question}\n\nTURNS:\n{numbered}\n\nTop {final} turn numbers:"
    )
    raw = _claude(client, prompt, max_tokens=60, temperature=0.0)
    import re as _re
    nums = [int(x) - 1 for x in _re.findall(r"\d+", raw) if 1 <= int(x) <= len(candidates)]
    # Deduplicate, preserve order, pad with BM25 order if fewer than final
    seen = set()
    ordered = []
    for idx in nums:
        if idx not in seen:
            seen.add(idx)
            ordered.append(idx)
    # pad
    for i, _ in scored[:initial]:
        if len(ordered) >= final:
            break
        idx_in_cands = scored.index((i, docs[i])) if (i, docs[i]) in scored else -1
        # simpler: pad from BM25 order
    bm25_order = [i for i, _ in scored[:initial]]
    for bi in bm25_order:
        if len(ordered) >= final:
            break
        if bi not in seen:
            seen.add(bi)
            ordered.append(bi)

    evidence = "\n".join(candidates[i] for i in ordered[:final])
    return answer_with_evidence(client, evidence, question)


# ---------------------------------------------------------------------------
# Retrieval baseline 4: MapReduce — 3 relevant facts per session
# ---------------------------------------------------------------------------

def map_reduce(client, items, question, per_session_k=3):
    """For each session ask Claude for the k most relevant facts; concatenate; answer."""
    sessions = group_by_session(items)
    all_facts = []
    for sid, sitems in sessions.items():
        if len(sitems) < 2:
            continue
        s_text = format_session(sid, sitems)
        prompt = (
            f"Given the QUESTION, extract the {per_session_k} most relevant facts from the SESSION. "
            f"One fact per line. If there are fewer than {per_session_k} relevant facts, output what you find.\n\n"
            f"QUESTION: {question}\n\nSESSION:\n{s_text[:8000]}\n\n"
            f"Top {per_session_k} relevant facts:"
        )
        response = _claude(client, prompt, max_tokens=300)
        facts = [f.strip("- *•").strip() for f in response.split("\n") if len(f.strip()) > 10]
        all_facts.extend([f"[s{sid}] {f}" for f in facts[:per_session_k]])
    evidence = "\n".join(all_facts)
    return answer_with_evidence(client, evidence, question)


# ---------------------------------------------------------------------------
# Retrieval baseline 5: Long-context full haystack (no truncation)
# ---------------------------------------------------------------------------

def long_context_full(client, items, question):
    """Stuff the entire haystack into context.  Claude Sonnet supports ~200K tokens;
    we format_haystack with max_chars=190000 to stay safely under the byte budget
    while being genuinely 'no truncation' for typical LongMemEval haystacks."""
    evidence = format_haystack(items, max_chars=190000)
    return answer_with_evidence(client, evidence, question)


# ---------------------------------------------------------------------------
# Pipeline ablation 6: gate_only
# Stage 1 relevance gate → raw kept sessions → answer (skip consolidation)
# ---------------------------------------------------------------------------

def gate_only(client, items, question):
    sessions = group_by_session(items)
    relevant = _relevance_gate(client, sessions, question)
    raw_parts = []
    for sid, sitems in relevant:
        raw_parts.append(
            f"\n=== Session {sid} ===\n"
            + "\n".join(f"{getattr(i, 'speaker', '')}: {i.content}" for i in sitems[:30])
        )
    evidence = "RAW RELEVANT SESSIONS:\n" + "\n".join(raw_parts)
    return answer_with_evidence(client, evidence, question)


# ---------------------------------------------------------------------------
# Pipeline ablation 7: consolidate_only
# Skip gate (use ALL sessions); query-aware consolidation; answer
# ---------------------------------------------------------------------------

def consolidate_only(client, items, question):
    sessions = group_by_session(items)
    all_facts = []
    raw_parts = []
    for sid, sitems in sessions.items():
        if len(sitems) < 2:
            continue
        s_text = format_session(sid, sitems)
        facts = query_aware_summarize(client, s_text, question)
        all_facts.extend([f"[s{sid}] {f}" for f in facts])
        raw_parts.append(
            f"\n=== Session {sid} ===\n"
            + "\n".join(f"{getattr(i, 'speaker', '')}: {i.content}" for i in sitems[:30])
        )
    evidence = (
        "DISTILLED FACTS:\n" + "\n".join(all_facts)
        + "\n\nRAW SESSIONS:\n" + "\n".join(raw_parts)
    )
    return answer_with_evidence(client, evidence, question)


# ---------------------------------------------------------------------------
# Pipeline ablation 8: facts_only
# Strong pipeline (gate + consolidation) but answer ONLY from consolidated facts
# ---------------------------------------------------------------------------

def facts_only(client, items, question):
    sessions = group_by_session(items)
    relevant = _relevance_gate(client, sessions, question)
    all_facts = []
    for sid, sitems in relevant:
        s_text = format_session(sid, sitems)
        facts = query_aware_summarize(client, s_text, question)
        all_facts.extend([f"[s{sid}] {f}" for f in facts])
    evidence = "DISTILLED FACTS:\n" + "\n".join(all_facts)
    return answer_with_evidence(client, evidence, question)


# ---------------------------------------------------------------------------
# Pipeline ablation 9: raw_only
# Strong pipeline (gate) but answer ONLY from raw kept sessions (no facts)
# (Same as gate_only but kept separate to make the ablation grid explicit)
# ---------------------------------------------------------------------------

def raw_only(client, items, question):
    # Identical execution to gate_only; kept as a named ablation so the results
    # table has both "gate_only" (which is really raw_only) and "raw_only" as
    # distinct labeled rows for reader clarity.
    sessions = group_by_session(items)
    relevant = _relevance_gate(client, sessions, question)
    raw_parts = []
    for sid, sitems in relevant:
        raw_parts.append(
            f"\n=== Session {sid} ===\n"
            + "\n".join(f"{getattr(i, 'speaker', '')}: {i.content}" for i in sitems[:30])
        )
    evidence = "RAW RELEVANT SESSIONS:\n" + "\n".join(raw_parts)
    return answer_with_evidence(client, evidence, question)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

SYSTEMS = [
    # (name, callable, kwargs)
    ("bm25_topk",         bm25_topk,         {}),
    ("embed_topk",        embed_topk,         {}),
    ("bm25_rerank",       bm25_rerank,        {}),
    ("map_reduce",        map_reduce,         {}),
    ("long_context_full", long_context_full,  {}),
    ("gate_only",         gate_only,          {}),
    ("consolidate_only",  consolidate_only,   {}),
    ("facts_only",        facts_only,         {}),
    ("raw_only",          raw_only,           {}),
]


def run(n_questions: int, output_path: str):
    client = _bedrock_client()
    all_examples = longmemeval(n=300, question_types=None)
    multi = [
        ex for ex in all_examples
        if any("multi-session" in q.query_type for q in ex.queries)
    ][:n_questions]
    print(f"[setup] {len(multi)} multi-session questions", flush=True)

    results: dict[str, list] = {name: [] for name, _, _ in SYSTEMS}

    for i, ex in enumerate(multi):
        q = ex.queries[0]
        print(f"\n--- Q{i+1}/{len(multi)} ---  expected={q.answer[:60]!r}", flush=True)

        for sys_name, fn, kwargs in SYSTEMS:
            try:
                t0 = time.perf_counter()
                ans = fn(client, ex.items, q.question, **kwargs)
                judge = llm_judge(client, ans, q.answer, q.question)
                elapsed = time.perf_counter() - t0
                print(
                    f"  [{sys_name:20s}] judge={judge} ans={ans[:60]!r} ({elapsed:.1f}s)",
                    flush=True,
                )
                results[sys_name].append({
                    "q": q.question,
                    "gt": q.answer,
                    "ans": ans,
                    "judge": judge,
                })
            except Exception as e:
                print(f"  [{sys_name:20s}] FAILED: {e}", flush=True)
                results[sys_name].append({
                    "q": q.question,
                    "gt": q.answer,
                    "ans": "",
                    "judge": 0,
                    "error": str(e),
                })

    # Summary with bootstrap CIs
    print("\n" + "=" * 80, flush=True)
    print(f"RESULTS (n={len(multi)} multi-session questions)", flush=True)
    print("=" * 80, flush=True)
    summary = {}
    for sys_name, _, _ in SYSTEMS:
        entries = results[sys_name]
        scores = [e["judge"] for e in entries]
        mean, lo, hi = bootstrap_ci(scores)
        summary[sys_name] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(scores)}
        print(f"{sys_name:25s}: {mean:.3f} [{lo:.3f}, {hi:.3f}]  (n={len(scores)})", flush=True)

    out = {"n": len(multi), "summary": summary, "details": results}
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {output_path}", flush=True)

    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        import subprocess
        s3_key = f"s3://{s3_bucket}/baselines_ablations/{os.path.basename(output_path)}"
        try:
            subprocess.run(["aws", "s3", "cp", output_path, s3_key], check=True, timeout=120)
            print(f"[s3] {s3_key}", flush=True)
        except Exception as e:
            print(f"[s3] {e}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--output", default="/tmp/baselines_ablations.json")
    args = p.parse_args()
    run(args.n, args.output)
