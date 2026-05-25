"""Multi-hop retrieval experiment for main-track contribution.

Central claim: retrieval failure — not reasoning failure — causes multi-hop
memory QA breakdown. We show query decomposition + multi-round retrieval
closes the gap from BM25 single-query toward oracle evidence on 3+-hop questions.

Conditions (all with same compression pipeline after retrieval):
  LC_naive        — long-context naive prompt (no compression, no retrieval)
  LC_cot          — long-context structured CoT (search-extract-answer)
  BM25_single     — single BM25 query → compression pipeline
  BM25_expand     — query expansion/paraphrase (HyDE-style) → BM25 → compression
  BM25_decompose  — LLM decomposes Q into sub-questions → BM25 each → union → compression
  BM25_iterative  — IRCoT-style: retrieve → extract → formulate next query → iterate
  oracle_evidence — gold evidence sessions → compression pipeline (upper bound)

All use SAME token budget (8K chars of retrieved context) for fair comparison.
All use SAME compression pipeline (relevance gate → query-aware extract → answer).

Evaluation: paired bootstrap CIs + McNemar test for each pair vs BM25_single.
Stratified by hop_count (from hop_labels.jsonl).
"""
from __future__ import annotations
import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")
from context_bench.datasets.memory.longmemeval import longmemeval


def _bedrock():
    import boto3
    return boto3.client("bedrock-runtime", region_name="us-east-1")


def _claude(client, prompt, max_tokens=400, model_id=None):
    model_id = model_id or os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    r = client.invoke_model(
        body=json.dumps(body), modelId=model_id,
        accept="application/json", contentType="application/json",
    )
    return json.loads(r["body"].read())["content"][0]["text"].strip()


def group_by_session(items):
    sessions = defaultdict(list)
    order = []
    for it in items:
        sid = getattr(it, "session_id", None) or "default"
        if sid not in sessions:
            order.append(sid)
        sessions[sid].append(it)
    return order, sessions


def bm25_retrieve(items, query, k=16):
    """BM25 retrieval over sessions. Returns top-k sessions (no char cap — use cap_to_sessions)."""
    order, sessions = group_by_session(items)
    docs = {sid: " ".join(i.content for i in sessions[sid]) for sid in order}
    q_terms = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 2]
    if not q_terms:
        return [(sid, sessions[sid]) for sid in order[:k]]
    N = len(docs)
    avgdl = sum(len(re.findall(r"\w+", d)) for d in docs.values()) / max(1, N)
    df = {t: sum(1 for d in docs.values() if t in d.lower()) for t in q_terms}
    k1, b = 1.5, 0.75
    scored = []
    for sid in order:
        words = [w.lower() for w in re.findall(r"\w+", docs[sid])]
        dl = len(words); s = 0.0
        for term in q_terms:
            if df[term] == 0: continue
            tf = words.count(term)
            idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
            s += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(1, avgdl)))
        scored.append((s, sid))
    scored.sort(key=lambda x: -x[0])
    return [(sid, sessions[sid]) for s, sid in scored[:k] if s > 0]


def bm25_retrieve_multi(items, queries, k_per_q=8):
    """Multi-query BM25: retrieve for each query, union by max score (no char cap)."""
    order, sessions = group_by_session(items)
    all_scored = []
    for query in queries:
        q_terms = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 2]
        if not q_terms:
            continue
        docs = {sid: " ".join(i.content for i in sessions[sid]) for sid in order}
        N = len(docs)
        avgdl = sum(len(re.findall(r"\w+", d)) for d in docs.values()) / max(1, N)
        df = {t: sum(1 for d in docs.values() if t in d.lower()) for t in q_terms}
        k1, b = 1.5, 0.75
        for sid in order:
            words = [w.lower() for w in re.findall(r"\w+", docs[sid])]
            dl = len(words); s = 0.0
            for term in q_terms:
                if df[term] == 0: continue
                tf = words.count(term)
                idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
                s += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(1, avgdl)))
            all_scored.append((s, sid))
    best = {}
    for s, sid in all_scored:
        best[sid] = max(best.get(sid, 0.0), s)
    sorted_sids = sorted(best, key=lambda x: -best[x])[:k_per_q * max(1, len(queries))]
    kept = [(sid, sessions[sid]) for sid in sorted_sids if best.get(sid, 0) > 0]
    return kept if kept else [(order[0], sessions[order[0]])]


def is_session_relevant(client, session_text, question):
    prompt = (
        f"Could this session contain information that would help answer the question? "
        f"Reply YES or NO only.\n\nQUESTION: {question}\n\nSESSION:\n{session_text[:6000]}\n\nReply:"
    )
    v = _claude(client, prompt, max_tokens=10).upper()
    return "YES" in v


def query_aware_extract(client, session_text, question):
    prompt = (
        "Extract ALL facts from this session that could help answer the question. "
        "Be specific (names, numbers, dates, quantities). One fact per line.\n\n"
        f"QUESTION: {question}\n\nSESSION:\n{session_text[:8000]}\n\nFacts:"
    )
    out = _claude(client, prompt, max_tokens=600)
    return [l.strip("- *•").strip() for l in out.split("\n") if len(l.strip()) > 10]


def answer_from_evidence(client, evidence_text, question):
    prompt = (
        "Answer the question precisely. If a number or count, give just the number. Under 15 words. "
        "If the evidence does not support an answer, reply NO_EVIDENCE.\n\n"
        f"EVIDENCE:\n{evidence_text}\n\nQUESTION: {question}\n\nAnswer:"
    )
    return _claude(client, prompt, max_tokens=100)


def compress_and_answer(client, kept_sessions, question, skip_gate=False):
    """Gate → extract → answer. Same compression pipeline for all retrieval conditions.

    skip_gate=True: bypass relevance gate (used for oracle to prevent destructive
    gate rejecting gold sessions). Codex bug #2: gate truncates at 6K chars, so
    a gold session with the answer in chars 6001+ gets rejected. Oracle must skip gate.
    """
    if skip_gate:
        relevant = [(sid, "\n".join(f"{getattr(i,'speaker','')}: {i.content}" for i in sitems))
                    for sid, sitems in kept_sessions]
    else:
        relevant = []
        for sid, sitems in kept_sessions:
            s_text = "\n".join(f"{getattr(i,'speaker','')}: {i.content}" for i in sitems)
            if is_session_relevant(client, s_text, question):
                relevant.append((sid, s_text))
        if not relevant:
            relevant = [(sid, "\n".join(f"{getattr(i,'speaker','')}: {i.content}" for i in sitems))
                        for sid, sitems in kept_sessions[:5]]
    all_facts = []
    all_raw = []
    for sid, s_text in relevant:
        # Codex bug #3: extraction truncated at 8K chars; use 20K for oracle-grade coverage
        extract_window = 20000 if skip_gate else 8000
        facts = query_aware_extract(client, s_text[:extract_window], question)
        all_facts.extend([f"[s{sid}] {f}" for f in facts])
        all_raw.append(f"[Session {sid}]\n{s_text}")
    evidence = "DISTILLED FACTS:\n" + "\n".join(all_facts)
    if all_raw:
        evidence += "\n\nRAW SESSIONS:\n" + "\n\n".join(all_raw)
    return answer_from_evidence(client, evidence, question)


def llm_judge(client, pred, gold, question):
    if not pred.strip() or "NO_EVIDENCE" in pred.upper():
        return 0
    prompt = (
        "Judge if the PREDICTION correctly answers the QUESTION given GOLD. Reply CORRECT or WRONG only.\n\n"
        f"QUESTION: {question}\nGOLD: {gold}\nPREDICTION: {pred}\n\nReply:"
    )
    v = _claude(client, prompt, max_tokens=20).upper()
    if "WRONG" in v or "INCORRECT" in v or "NOT CORRECT" in v:
        return 0
    return 1 if "CORRECT" in v else 0


def decompose_question(client, question):
    """Decompose a multi-hop question into atomic sub-questions."""
    prompt = (
        "Break this question into 2-4 simpler sub-questions, each answerable from a single session. "
        "Output one sub-question per line. If the question is already simple, output it once.\n\n"
        f"QUESTION: {question}\n\nSub-questions:"
    )
    out = _claude(client, prompt, max_tokens=200)
    subs = [l.strip().lstrip("0123456789.-) ").strip() for l in out.split("\n") if len(l.strip()) > 10]
    return subs[:4] if subs else [question]


def expand_query(client, question):
    """HyDE-style: generate a hypothetical answer, use as additional query."""
    prompt = (
        "Generate a plausible answer to this question in 1-2 sentences. "
        "Be specific with facts, names, and numbers.\n\n"
        f"QUESTION: {question}\n\nPlausible answer:"
    )
    hypo = _claude(client, prompt, max_tokens=100)
    return [question, hypo]


def iterative_retrieve(client, items, question, max_iters=2, k=5):
    """IRCoT-style: retrieve → extract clue → reformulate → repeat.
    Codex fix: no per-iter budget split; budget enforced uniformly by cap_to_sessions()
    at the call site. We accumulate candidate sessions across iterations, then cap.
    """
    order, sessions = group_by_session(items)
    all_kept = {}  # sid → sitems
    current_query = question
    for it in range(max_iters):
        q_terms = [w.lower() for w in re.findall(r"\w+", current_query) if len(w) > 2]
        if not q_terms:
            break
        docs = {sid: " ".join(i.content for i in sessions[sid]) for sid in order}
        N = len(docs); avgdl = sum(len(re.findall(r"\w+", d)) for d in docs.values()) / max(1, N)
        df = {t: sum(1 for d in docs.values() if t in d.lower()) for t in q_terms}
        k1, b_param = 1.5, 0.75
        scored = []
        for sid in order:
            ws = [w.lower() for w in re.findall(r"\w+", docs[sid])]
            dl = len(ws); s = 0.0
            for term in q_terms:
                if df[term] == 0: continue
                tf = ws.count(term)
                idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
                s += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b_param + b_param * dl / max(1, avgdl)))
            scored.append((s, sid))
        scored.sort(key=lambda x: -x[0])
        new_sids = [sid for s, sid in scored[:k] if s > 0 and sid not in all_kept]
        for sid in new_sids:
            all_kept[sid] = sessions[sid]
        if it < max_iters - 1 and new_sids:
            # Formulate next query from newly retrieved sessions
            excerpts = [f"[Session {sid}]\n" + "\n".join(
                f"{getattr(i,'speaker','')}: {i.content}" for i in sessions[sid])[:2000]
                for sid in new_sids[:3]]
            clue_prompt = (
                "Given these sessions and the question, what specific fact is STILL MISSING "
                "to answer it? Output ONLY a search query to find the missing fact.\n\n"
                f"QUESTION: {question}\n\nSESSIONS:\n{'---'.join(excerpts)}\n\nSearch query:"
            )
            current_query = _claude(client, clue_prompt, max_tokens=80)
    # Return in haystack order; budget cap applied at call site
    return [(sid, all_kept[sid]) for sid in order if sid in all_kept]


def full_haystack_answer(client, items, question, use_cot=False):
    """Long-context: full haystack in context."""
    parts = []
    cur = None
    for it in items:
        sid = getattr(it, "session_id", None) or "default"
        if sid != cur:
            parts.append(f"\n[Session {sid}]")
            cur = sid
        sp = getattr(it, "speaker", "") or ""
        parts.append(f"{sp}: {it.content}")
    haystack = "\n".join(parts)[:900_000]
    if use_cot:
        prompt = (
            "Follow these steps:\n"
            "Step 1: List ALL sessions that could contain relevant information.\n"
            "Step 2: Quote the specific evidence from each relevant session.\n"
            "Step 3: Synthesize and give the final answer (under 15 words).\n\n"
            f"CONVERSATION HISTORY:\n{haystack}\n\nQUESTION: {question}\n\n"
            "RELEVANT SESSIONS:\nEVIDENCE:\nANSWER:"
        )
    else:
        prompt = (
            "Answer the question precisely. If a number or count, give just the number. Under 15 words. "
            f"CONVERSATION HISTORY:\n{haystack}\n\nQUESTION: {question}\n\nAnswer:"
        )
    return _claude(client, prompt, max_tokens=400 if use_cot else 100)


def parse_cot_answer(response):
    for line in response.split("\n"):
        if line.strip().upper().startswith("ANSWER:"):
            return line.split(":", 1)[1].strip()
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    return lines[-1] if lines else ""


def mcnemar_test(results_a, results_b):
    """Exact McNemar's test (mid-p correction) for paired binary outcomes.
    Codex requirement: use exact binomial, not asymptotic z, for small n.
    Returns p-value (two-tailed).
    """
    import math
    b = sum(1 for a, bv in zip(results_a, results_b) if a == 1 and bv == 0)
    c = sum(1 for a, bv in zip(results_a, results_b) if a == 0 and bv == 1)
    n_disc = b + c
    if n_disc == 0:
        return 1.0
    # Exact binomial mid-p: P(X <= min(b,c)) + 0.5*P(X == min(b,c)) under H0: p=0.5
    # where X ~ Binomial(n_disc, 0.5)
    k = min(b, c)
    # Binomial CDF using log-gamma for numerical stability
    def log_binom_pmf(n, k, p=0.5):
        from math import lgamma, log
        return lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1) + k*log(p) + (n-k)*log(1-p)
    cumul = sum(math.exp(log_binom_pmf(n_disc, j)) for j in range(k))
    pmf_k = math.exp(log_binom_pmf(n_disc, k))
    p_one_tail = cumul + 0.5 * pmf_k  # mid-p correction
    return min(1.0, 2 * p_one_tail)


def bootstrap_ci(scores, n=1000):
    import random
    if not scores: return 0.0, 0.0, 0.0
    rng = random.Random(42)
    n_s = len(scores)
    means = sorted(sum(rng.choice(scores) for _ in range(n_s)) / n_s for _ in range(n))
    return sum(scores) / n_s, means[int(n * 0.025)], means[int(n * 0.975)]


def dense_retrieve(client, items, query, k=8):
    """Dense embedding retrieval using Titan v2 (1024-dim).
    Encodes query + each session, returns top-k by cosine similarity.
    """
    import json as _json, math as _math
    order, sessions = group_by_session(items)
    docs = {sid: " ".join(i.content for i in sessions[sid])[:6000] for sid in order}

    def embed(text):
        body = _json.dumps({"inputText": text[:6000]})
        r = client.invoke_model(
            body=body, modelId="amazon.titan-embed-text-v2:0",
            accept="application/json", contentType="application/json",
        )
        return _json.loads(r["body"].read())["embedding"]

    def cosine(a, b):
        dot = sum(x*y for x,y in zip(a,b))
        na = _math.sqrt(sum(x*x for x in a)); nb = _math.sqrt(sum(x*x for x in b))
        return dot / max(1e-9, na * nb)

    q_emb = embed(query)
    scored = []
    for sid in order:
        s_emb = embed(docs[sid])
        scored.append((cosine(q_emb, s_emb), sid))
    scored.sort(key=lambda x: -x[0])
    return [(sid, sessions[sid]) for _, sid in scored[:k]]


CONDITIONS = [
    "LC_naive", "LC_cot", "BM25_single", "BM25_expand",
    "BM25_decompose", "BM25_iterative", "dense_single",
    "dense_decompose", "oracle_evidence",
]

MAX_SESSIONS = 8  # session count limit, consistent with main baselines pipeline.
# NOTE: No char budget cap. 3+-evidence questions need full session content
# (~10-30K chars); an 8K char cap destroys oracle evidence making oracle ≈ BM25.
# Fair comparison: all conditions cap at MAX_SESSIONS retrieved/selected.
# The main baselines (run_baselines_ablations.py) use no char cap and top-8 sessions.


def cap_to_sessions(kept_sessions, max_sessions=MAX_SESSIONS):
    """Cap by session count, not chars — consistent with main baselines."""
    return kept_sessions[:max_sessions]


def resolve_oracle_sessions(ex, order, sessions):
    """Find gold sessions by ID matching with normalization.
    Codex bug #1: answer_session_ids might be int indices while session_ids
    are strings like 'session_3', causing all-empty match → silent BM25 fallback.
    Try multiple formats: str(id), f'session_{id}', direct string match.
    """
    raw_ids = ex.metadata.get("answer_session_ids") or []
    if not raw_ids:
        return [], False

    # Build all session_ids in this example
    all_sids = set(order)

    # Try each raw_id in multiple formats
    kept = []
    for raw_id in raw_ids:
        candidates = [str(raw_id), f"session_{raw_id}"]
        found = next((s for s in candidates if s in all_sids), None)
        if found and found not in {sid for sid, _ in kept}:
            kept.append((found, sessions[found]))

    matched = len(kept) > 0
    return kept, matched


def run_example(client, ex, q, conditions, client_bedrock=None):
    """Run all conditions on one example. Returns dict of {condition: {ans, judge}}."""
    order, sessions = group_by_session(ex.items)
    gold = q.answer
    question = q.question
    results = {}

    for cond in conditions:
        t0 = time.perf_counter()
        try:
            if cond == "LC_naive":
                ans = full_haystack_answer(client, ex.items, question, use_cot=False)
            elif cond == "LC_cot":
                raw = full_haystack_answer(client, ex.items, question, use_cot=True)
                # Codex bug #5: more robust LC_cot parsing
                ans = parse_cot_answer(raw)
                if not ans.strip():
                    # Fallback: last non-empty line that doesn't look like a header
                    lines = [l.strip() for l in raw.split("\n") if l.strip()
                             and not l.strip().upper().startswith(("STEP", "RELEVANT", "EVIDENCE", "ANSWER:"))]
                    ans = lines[-1] if lines else raw.strip()[:100]
            elif cond == "BM25_single":
                kept = bm25_retrieve(ex.items, question, k=16)
                ans = compress_and_answer(client, cap_to_sessions(kept), question)
            elif cond == "BM25_expand":
                queries = expand_query(client, question)
                kept = bm25_retrieve_multi(ex.items, queries, k_per_q=8)
                ans = compress_and_answer(client, cap_to_sessions(kept), question)
            elif cond == "BM25_decompose":
                sub_qs = decompose_question(client, question)
                all_queries = [question] + sub_qs
                kept = bm25_retrieve_multi(ex.items, all_queries, k_per_q=5)
                ans = compress_and_answer(client, cap_to_sessions(kept), question)
            elif cond == "BM25_iterative":
                kept = iterative_retrieve(client, ex.items, question, max_iters=2, k=8)
                ans = compress_and_answer(client, cap_to_sessions(kept), question)
            elif cond == "dense_single":
                kept = dense_retrieve(client, ex.items, question, k=8)
                ans = compress_and_answer(client, cap_to_sessions(kept), question)
            elif cond == "dense_decompose":
                sub_qs = decompose_question(client, question)
                # Dense retrieval for each sub-question, union
                all_kept = {}
                for sq in [question] + sub_qs:
                    for sid, sitems in dense_retrieve(client, ex.items, sq, k=5):
                        all_kept[sid] = sitems
                kept = list(all_kept.items())
                ans = compress_and_answer(client, cap_to_sessions(kept), question)
            elif cond == "oracle_evidence":
                # Codex bugs #1,2,4 fixes:
                # - Normalize IDs (bug #1)
                # - Skip relevance gate (bug #2: gate truncates at 6K, can reject gold sessions)
                # - Use ALL gold sessions, no MAX_SESSIONS cap (bug #4)
                kept, matched = resolve_oracle_sessions(ex, order, sessions)
                oracle_n_matched = len(kept)
                if not kept:
                    # Last resort: BM25 — but LOG this so we can audit
                    kept = bm25_retrieve(ex.items, question, k=16)
                    oracle_n_matched = 0
                # skip_gate=True bypasses destructive gate for oracle
                ans = compress_and_answer(client, kept, question, skip_gate=True)
            else:
                ans = ""
            judge = llm_judge(client, ans, gold, question)
            dt = time.perf_counter() - t0
            rec = {"ans": ans, "judge": judge, "dt": dt}
            if cond == "oracle_evidence":
                rec["oracle_n_matched"] = oracle_n_matched
            results[cond] = rec
        except Exception as e:
            results[cond] = {"ans": "", "judge": 0, "error": str(e)[:200], "dt": time.perf_counter() - t0}
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hop-min", type=int, default=3, help="min hop_count for filtering")
    p.add_argument("--n", type=int, default=None, help="max examples (default: all 3+-hop)")
    p.add_argument("--shard", type=int, default=0, help="shard index (0-based)")
    p.add_argument("--n-shards", type=int, default=1, help="total number of shards")
    p.add_argument("--hop-labels", default="/tmp/hop_labels.jsonl")
    p.add_argument("--conditions", default="all")
    p.add_argument("--output", default="/tmp/multihop_retrieval.json")
    args = p.parse_args()

    # Load hop labels
    hop_map = {}
    hop_path = Path(args.hop_labels)
    if hop_path.exists():
        for line in hop_path.read_text().splitlines():
            try:
                d = json.loads(line)
                qid = str(d.get("qid", ""))
                if qid:
                    hop_map[qid] = d
            except Exception:
                pass
    print(f"[setup] loaded {len(hop_map)} hop labels", flush=True)

    # Load examples and filter by GROUND-TRUTH n_evidence_sessions from LME-S metadata.
    # Codex fix: LLM-inferred hop_count from hop_labels != ground-truth n_evidence_sessions.
    # When oracle uses empty evidence_ids it falls back to BM25 → oracle ≈ BM25, useless.
    # Use ex.metadata["answer_session_ids"] for filtering to ensure oracle is meaningful.
    all_examples = longmemeval(n=500)
    filtered = []
    for ex in all_examples:
        for q in ex.queries:
            evidence_ids = ex.metadata.get("answer_session_ids") or []
            n_evidence = len(evidence_ids)
            # Primary filter: ground-truth evidence sessions >= hop_min
            if n_evidence >= args.hop_min:
                label = hop_map.get(str(ex.id), {})
                filtered.append((ex, q, label))
    if args.n:
        filtered = filtered[:args.n]
    # Shard: this job processes only its slice
    if args.n_shards > 1:
        filtered = filtered[args.shard::args.n_shards]
    print(f"[setup] {len(filtered)} questions (shard {args.shard}/{args.n_shards})", flush=True)

    conditions = CONDITIONS if args.conditions == "all" else args.conditions.split(",")
    client = _bedrock()
    all_results = []

    for i, (ex, q, label) in enumerate(filtered):
        print(
            f"\n--- Q{i+1}/{len(filtered)} id={ex.id} "
            f"hop={label.get('hop_count','?')} n_ev={len(ex.metadata.get('answer_session_ids') or [])} "
            f"class={label.get('question_class','?')} ---",
            flush=True,
        )
        cond_results = run_example(client, ex, q, conditions, client_bedrock=client)
        for cond, r in cond_results.items():
            print(f"  [{cond:16s}] judge={r['judge']} ans={r['ans'][:50]!r} ({r['dt']:.1f}s)", flush=True)
        all_results.append({
            "qid": ex.id, "question": q.question, "gold": q.answer,
            "hop_count": label.get("hop_count"), "question_class": label.get("question_class"),
            "n_evidence": len(ex.metadata.get("answer_session_ids") or []),
            "conditions": cond_results,
        })

    # Aggregate
    print("\n" + "=" * 80, flush=True)
    print(f"MULTI-HOP RETRIEVAL (n={len(all_results)}, hop≥{args.hop_min})", flush=True)
    print("=" * 80, flush=True)
    summary = {}
    all_judges = {c: [r["conditions"].get(c, {}).get("judge", 0) for r in all_results] for c in conditions}
    base = "BM25_single"
    for cond in conditions:
        scores = all_judges[cond]
        mean, lo, hi = bootstrap_ci(scores)
        p_val = mcnemar_test(all_judges[cond], all_judges[base]) if cond != base else None
        summary[cond] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(scores), "mcnemar_vs_bm25": p_val}
        sig_str = f"  p={p_val:.3f}{'*' if p_val and p_val < 0.05 else ''}" if p_val is not None else ""
        print(f"  {cond:18s} {mean:.3f} [{lo:.3f},{hi:.3f}] (n={len(scores)}){sig_str}", flush=True)

    out = {"n": len(all_results), "hop_min": args.hop_min, "summary": summary, "details": all_results}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {args.output}", flush=True)

    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        import subprocess
        s3_key = f"s3://{s3_bucket}/multihop/{os.path.basename(args.output)}"
        try:
            subprocess.run(["aws", "s3", "cp", args.output, s3_key], check=True, timeout=120)
            print(f"[s3] {s3_key}", flush=True)
        except Exception as e:
            print(f"[s3] {e}", flush=True)


if __name__ == "__main__":
    main()
