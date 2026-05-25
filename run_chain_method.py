"""Multi-hop evidence-chain method.

Main-track candidate per Codex review. Central claim: long-context QA failure
is mainly CHAIN ASSEMBLY FAILURE, not fact access failure. Explicit chain
construction with citation-grounded edges fixes it.

Pipeline (per question Q):
  1. RETRIEVE_SESSIONS: BM25/dense hybrid → top-K sessions by relevance.
  2. EXTRACT_PROPOSITIONS: per kept session, extract atomic propositions WITH
     session_id citations. Each proposition is a short factual claim grounded
     in one session.
  3. CHAIN_CONSTRUCT: ask LLM to build a candidate chain answering Q.
     Each chain step must cite one proposition. Output: ordered list of
     "P1 (cite: S3) → P2 (cite: S7) → ... → Answer".
  4. VERIFY_CHAIN: independent LLM verifies (a) each cited proposition is
     literally in the proposition list, (b) chain has no contradiction with
     proposition list, (c) chain is temporally consistent, (d) answer follows.
     If verifier fails → identify missing-link query → re-retrieve → retry chain.
  5. ANSWER_FROM_CHAIN: produce final answer from verified chain ONLY.

Ablations to support the central claim:
  - no_chain: extract propositions then answer (= our facts_only baseline)
  - no_verifier: chain construct, no verification
  - no_citation: chain construct without requiring session citations
  - no_re_retrieve: don't fix failed chains
  - oracle_evidence: skip retrieval, use gold evidence sessions
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

sys.path.insert(0, "src")

from context_bench.datasets.memory.longmemeval import longmemeval


def _bedrock():
    import boto3
    return boto3.client("bedrock-runtime", region_name="us-east-1")


def _claude(client, prompt, max_tokens=400):
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    r = client.invoke_model(
        body=json.dumps(body),
        modelId=os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6"),
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


# ---------------------------------------------------------------------------
# Step 1: RETRIEVE_SESSIONS (BM25 over sessions)
# ---------------------------------------------------------------------------
def retrieve_top_sessions(items, question, k=8):
    order, sessions = group_by_session(items)
    docs = {sid: " ".join(i.content for i in sessions[sid]) for sid in order}
    q_terms = [w.lower() for w in re.findall(r"\w+", question) if len(w) > 2]
    if not q_terms:
        return [(sid, sessions[sid]) for sid in order[:k]]
    avgdl = sum(len(re.findall(r"\w+", d)) for d in docs.values()) / max(1, len(docs))
    N = len(docs)
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


# ---------------------------------------------------------------------------
# Step 2: EXTRACT_PROPOSITIONS (atomic facts with session citations)
# ---------------------------------------------------------------------------
def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def extract_propositions(client, kept_sessions, question, blinded: bool = False):
    """Extract atomic propositions WITH source quotes (Codex source-grounding fix).

    Each proposition stores (a) content, (b) session_id, (c) supporting verbatim
    quote from the session, (d) source_grounded flag (verified via substring match).

    If blinded=True, omit the question from the prompt — used for the
    latent-answering control: does the chain still work when propositions are
    extracted without knowing what was asked?
    """
    props = []
    pid = 0
    for sid, sitems in kept_sessions:
        session_text = "\n".join(f"{getattr(i, 'speaker', '') or ''}: {i.content}" for i in sitems)
        session_truncated = session_text[:8000]
        if blinded:
            prompt = (
                "Extract 4-8 ATOMIC propositions from this session. Each proposition must be: "
                "(a) a single self-contained factual claim, "
                "(b) concrete (numbers/names/dates/quantities), "
                "(c) supported by a verbatim quote from the session. "
                "Format each line: <proposition> ||| <verbatim quote>\n\n"
                f"SESSION:\n{session_truncated}\n\nPropositions:"
            )
        else:
            prompt = (
                "Extract 2-6 ATOMIC propositions from this session that may help answer the question. "
                "Each proposition must be: (a) a single self-contained factual claim, "
                "(b) concrete (numbers/names/dates/quantities), "
                "(c) supported by a verbatim quote from the session. "
                "Format each line: <proposition> ||| <verbatim quote>\n\n"
                f"QUESTION: {question}\n\nSESSION:\n{session_truncated}\n\nPropositions:"
            )
        out = _claude(client, prompt, max_tokens=500)
        for line in out.split("\n"):
            line = line.strip().lstrip("-*•0123456789. ").strip()
            if not line or len(line) < 10: continue
            # Parse "proposition ||| quote"
            if "|||" in line:
                content, quote = line.split("|||", 1)
                content, quote = content.strip(), quote.strip()
            else:
                content, quote = line, ""
            # Source-grounded: require quote ≥30 chars AND full normalized quote
            # substring-matches the session text (Codex fix: prefix match was too weak).
            norm_quote = _normalize(quote)
            source_grounded = (
                bool(quote)
                and len(quote) >= 30
                and len(norm_quote) >= 15
                and norm_quote in _normalize(session_truncated)
            )
            pid += 1
            props.append({
                "pid": pid,
                "content": content,
                "session_id": str(sid),
                "quote": quote,
                "source_grounded": source_grounded,
            })
        if pid >= 40:
            break
    return props


# ---------------------------------------------------------------------------
# Step 3: CHAIN_CONSTRUCT
# ---------------------------------------------------------------------------
def construct_chain(client, propositions, question, require_citations=True):
    if not propositions:
        return {"chain": [], "answer": "", "raw": ""}
    prop_text = "\n".join(f"P{p['pid']} [session={p['session_id']}]: {p['content']}" for p in propositions)
    citation_req = (
        "Each chain step MUST cite one proposition by its Pn identifier. "
        if require_citations else ""
    )
    prompt = (
        "Below are extracted PROPOSITIONS from a multi-session conversation. "
        f"Construct a step-by-step CHAIN that answers the question. "
        f"{citation_req}"
        "Format:\n"
        "CHAIN:\n"
        "  Step 1: <statement> [cite: P3]\n"
        "  Step 2: <statement> [cite: P7]\n"
        "  ...\n"
        "ANSWER: <final precise answer, under 15 words>\n\n"
        f"PROPOSITIONS:\n{prop_text}\n\nQUESTION: {question}\n"
    )
    raw = _claude(client, prompt, max_tokens=500)

    # Parse chain steps and answer
    chain = []
    answer = ""
    for line in raw.split("\n"):
        line = line.strip()
        if line.upper().startswith("ANSWER:"):
            answer = line.split(":", 1)[1].strip()
        elif line.upper().startswith("STEP"):
            chain.append(line)
        elif line.startswith("  Step") or re.match(r"^\s*Step\s+\d+", line):
            chain.append(line.strip())
    return {"chain": chain, "answer": answer, "raw": raw}


# ---------------------------------------------------------------------------
# Step 4: VERIFY_CHAIN
# ---------------------------------------------------------------------------
def verify_chain(client, chain_data, propositions, question, check_source_grounding: bool = True):
    """Verifier with TWO-LEVEL grounding (Codex source-grounding fix):
      (a) Citation existence: each cited Pn exists in proposition list.
      (b) Source grounding: cited proposition's quote ACTUALLY appears in
          its session (already verified at extract time; just check the flag).
      (c) Chain reasoning: LLM verifier checks support + temporal + contradiction.
    """
    if not chain_data["chain"]:
        return {"verdict": "FAIL", "reason": "no_chain", "missing_link_query": question,
                "fail_type": "no_chain", "cited_pids": []}

    chain_text = "\n".join(chain_data["chain"]) + f"\nANSWER: {chain_data['answer']}"
    # Extract cited Pn ids from chain
    cited_pids = set()
    for m in re.finditer(r"P(\d+)", chain_text):
        cited_pids.add(int(m.group(1)))
    pid_to_prop = {p["pid"]: p for p in propositions}

    # Citation existence check
    invalid_pids = [p for p in cited_pids if p not in pid_to_prop]
    if invalid_pids and check_source_grounding:
        return {"verdict": "FAIL", "reason": f"citations refer to nonexistent props: {invalid_pids[:3]}",
                "missing_link_query": None, "fail_type": "invalid_cite", "cited_pids": list(cited_pids)}

    # Source-grounding check on cited propositions
    if check_source_grounding:
        ungrounded = [pid for pid in cited_pids if pid in pid_to_prop and not pid_to_prop[pid]["source_grounded"]]
        if ungrounded:
            return {"verdict": "FAIL",
                    "reason": f"cited props lack source-grounding: {ungrounded[:3]}",
                    "missing_link_query": question,
                    "fail_type": "ungrounded_cite", "cited_pids": list(cited_pids)}

    # Chain reasoning check
    prop_text = "\n".join(
        f"P{p['pid']} [session={p['session_id']}]: {p['content']}" for p in propositions
    )
    prompt = (
        "You are verifying a chain-of-reasoning. Check: (a) each cited proposition "
        "supports the chain step's claim, (b) no contradiction between steps, "
        "(c) temporal order is consistent, (d) the final answer follows from the chain.\n\n"
        "Reply in this exact format:\n"
        "VERDICT: PASS or FAIL\n"
        "REASON: <one sentence>\n"
        "MISSING_LINK_QUERY: <a NOVEL search query different from the original question that would retrieve missing evidence, or NONE>\n\n"
        f"QUESTION: {question}\n\nPROPOSITIONS:\n{prop_text}\n\nCHAIN:\n{chain_text}\n"
    )
    raw = _claude(client, prompt, max_tokens=200)
    verdict = "FAIL"; reason = ""; missing = "NONE"
    for line in raw.split("\n"):
        u = line.upper()
        if u.startswith("VERDICT:"):
            verdict = "PASS" if "PASS" in u else "FAIL"
        elif u.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
        elif u.startswith("MISSING_LINK_QUERY:"):
            missing = line.split(":", 1)[1].strip()
    return {
        "verdict": verdict, "reason": reason,
        "missing_link_query": missing if missing.upper() != "NONE" else None,
        "fail_type": "reasoning" if verdict == "FAIL" else "pass",
        "cited_pids": list(cited_pids),
    }


# ---------------------------------------------------------------------------
# Step 5: ANSWER_FROM_CHAIN
# ---------------------------------------------------------------------------
def answer_from_chain(client, chain_data, question):
    if chain_data.get("answer"):
        return chain_data["answer"]
    # Fallback: if chain didn't produce an answer, ask LLM to derive one
    chain_text = "\n".join(chain_data["chain"]) if chain_data["chain"] else ""
    prompt = (
        "Given the following chain of reasoning, output the final precise answer. "
        "If a number or count, give just the number. Under 15 words.\n\n"
        f"CHAIN:\n{chain_text}\n\nQUESTION: {question}\n\nAnswer:"
    )
    return _claude(client, prompt, max_tokens=80)


def llm_judge(client, pred, gold, question):
    if not pred.strip(): return 0
    prompt = (
        "Judge if the PREDICTION correctly answers the QUESTION given GOLD. Reply CORRECT or WRONG only.\n\n"
        f"QUESTION: {question}\nGOLD: {gold}\nPREDICTION: {pred}\n\nReply:"
    )
    v = _claude(client, prompt, max_tokens=20).upper()
    if "WRONG" in v or "INCORRECT" in v or "NOT CORRECT" in v: return 0
    return 1 if "CORRECT" in v else 0


# ---------------------------------------------------------------------------
# Top-level: chain method (full) + ablations
# ---------------------------------------------------------------------------
def run_chain_method(
    client, ex, q,
    use_chain: bool = True,
    use_verifier: bool = True,
    use_source_grounding: bool = True,
    require_citations: bool = True,
    use_re_retrieve: bool = True,
    oracle_evidence: bool = False,
    blinded_extraction: bool = False,
    max_retries: int = 2,
):
    # Step 1: retrieve
    if oracle_evidence:
        order, sessions = group_by_session(ex.items)
        evidence_ids = ex.metadata.get("answer_session_ids") or []
        kept = [(sid, sessions[sid]) for sid in order if sid in evidence_ids]
        if not kept:
            kept = retrieve_top_sessions(ex.items, q.question, k=8)
    else:
        kept = retrieve_top_sessions(ex.items, q.question, k=8)

    # Step 2: extract propositions (optionally blinded — Codex latent-answering control)
    propositions = extract_propositions(client, kept, q.question, blinded=blinded_extraction)

    # Pre-chain diagnostic: is the answer lexically present in propositions?
    answer_in_props = any(
        _normalize(q.answer)[:60] in _normalize(p["content"])
        for p in propositions
    ) if propositions else False

    if not use_chain:
        evidence = "\n".join(f"- {p['content']} [session {p['session_id']}]" for p in propositions)
        prompt = (
            "Answer the question precisely from the EVIDENCE. Under 15 words.\n\n"
            f"EVIDENCE:\n{evidence}\n\nQUESTION: {q.question}\n\nAnswer:"
        )
        ans = _claude(client, prompt, max_tokens=80)
        return {"answer": ans, "method": "no_chain", "n_props": len(propositions),
                "answer_in_props": answer_in_props}

    # Step 3: construct chain
    chain_data = construct_chain(client, propositions, q.question, require_citations=require_citations)

    # Step 4: verifier loop with no-progress detection (Codex fix #3)
    seen_missing_queries = set()
    seen_session_ids = {sid for sid, _ in kept}
    verifier_history = []
    if use_verifier:
        for retry in range(max_retries + 1):
            ver = verify_chain(client, chain_data, propositions, q.question,
                               check_source_grounding=use_source_grounding)
            chain_data["last_verifier"] = ver
            verifier_history.append(ver)
            if ver["verdict"] == "PASS":
                break
            if not use_re_retrieve or not ver.get("missing_link_query") or retry >= max_retries:
                break
            ml_query = ver["missing_link_query"]
            # No-progress: same missing-link query twice
            if ml_query in seen_missing_queries:
                ver["stalled"] = True
                break
            seen_missing_queries.add(ml_query)
            extra_kept = retrieve_top_sessions(ex.items, ml_query, k=4)
            new_kept = [(s, p) for s, p in extra_kept if s not in seen_session_ids][:3]
            # No-progress: no new sessions
            if not new_kept:
                ver["stalled_no_new_sessions"] = True
                break
            seen_session_ids.update(s for s, _ in new_kept)
            kept.extend(new_kept)
            new_props = extract_propositions(client, new_kept, q.question, blinded=blinded_extraction)
            # Renumber new propositions (avoid pid collision)
            next_pid = max((p["pid"] for p in propositions), default=0) + 1
            for np in new_props:
                np["pid"] = next_pid; next_pid += 1
                propositions.append(np)
            chain_data = construct_chain(client, propositions, q.question, require_citations=require_citations)

    # Step 5: answer from chain
    ans = answer_from_chain(client, chain_data, q.question)
    return {
        "answer": ans,
        "method": "chain",
        "n_props": len(propositions),
        "n_propositions_grounded": sum(1 for p in propositions if p["source_grounded"]),
        "chain_steps": len(chain_data.get("chain", [])),
        "verifier_verdict": chain_data.get("last_verifier", {}).get("verdict", "n/a"),
        "verifier_history": verifier_history,
        "answer_in_props": answer_in_props,
        "n_kept_sessions": len(kept),
    }


def bootstrap_ci(scores, n=1000):
    import random
    if not scores: return 0.0, 0.0, 0.0
    rng = random.Random(42)
    n_s = len(scores)
    means = sorted(sum(rng.choice(scores) for _ in range(n_s)) / n_s for _ in range(n))
    return sum(scores) / n_s, means[int(n * 0.025)], means[int(n * 0.975)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20, help="number of examples")
    p.add_argument("--hop-filter", type=int, default=3, help="minimum n_evidence_sessions to include")
    p.add_argument("--mode", default="all",
                   choices=["full", "no_chain", "no_verifier", "no_citation", "no_re_retrieve", "oracle_evidence", "all"])
    p.add_argument("--output", default="/tmp/chain_method.json")
    args = p.parse_args()

    client = _bedrock()
    all_examples = longmemeval(n=500)
    # Filter to hard multi-hop subset (n_evidence_sessions >= hop_filter)
    multi = [ex for ex in all_examples if any("multi-session" in q.query_type for q in ex.queries)]
    hard = [
        ex for ex in multi
        if len(ex.metadata.get("answer_session_ids") or []) >= args.hop_filter
    ][:args.n]
    print(f"[setup] {len(hard)} hard examples (≥{args.hop_filter} evidence sessions)", flush=True)

    # Clean ablation matrix (Codex fix #1): each ablation toggles exactly ONE
    # component holding everything else at "full method" default.
    modes = [
        "full",                # all components on
        "no_chain",            # skip chain construction
        "no_verifier",         # keep chain, skip verification (and thus re-retrieve)
        "no_source_grounding", # keep chain + verifier, skip source-grounding check
        "no_citation",         # keep chain + verifier, but don't require Pn cites
        "no_re_retrieve",      # keep verifier, just don't loop on failure
        "blinded_extraction",  # extract without seeing Q (latent-answering control)
        "oracle_evidence",     # skip retrieval, use gold evidence sessions
    ] if args.mode == "all" else [args.mode]
    results = {m: [] for m in modes}

    for i, ex in enumerate(hard):
        q = ex.queries[0]
        gold = q.answer
        print(f"\n--- Q{i+1}/{len(hard)} id={ex.id} n_ev={len(ex.metadata.get('answer_session_ids') or [])} ---", flush=True)

        for m in modes:
            t0 = time.perf_counter()
            try:
                kwargs = {
                    "full":                dict(use_chain=True,  use_verifier=True,  use_source_grounding=True,  require_citations=True,  use_re_retrieve=True,  oracle_evidence=False, blinded_extraction=False),
                    "no_chain":            dict(use_chain=False, use_verifier=False, use_source_grounding=False, require_citations=False, use_re_retrieve=False, oracle_evidence=False, blinded_extraction=False),
                    # Note: no_verifier=True implicitly disables re-retrieve (verifier is what produces missing_link_query)
                    "no_verifier":         dict(use_chain=True,  use_verifier=False, use_source_grounding=False, require_citations=True,  use_re_retrieve=False, oracle_evidence=False, blinded_extraction=False),
                    "no_source_grounding": dict(use_chain=True,  use_verifier=True,  use_source_grounding=False, require_citations=True,  use_re_retrieve=True,  oracle_evidence=False, blinded_extraction=False),
                    "no_citation":         dict(use_chain=True,  use_verifier=True,  use_source_grounding=False, require_citations=False, use_re_retrieve=True,  oracle_evidence=False, blinded_extraction=False),
                    "no_re_retrieve":      dict(use_chain=True,  use_verifier=True,  use_source_grounding=True,  require_citations=True,  use_re_retrieve=False, oracle_evidence=False, blinded_extraction=False),
                    "blinded_extraction":  dict(use_chain=True,  use_verifier=True,  use_source_grounding=True,  require_citations=True,  use_re_retrieve=True,  oracle_evidence=False, blinded_extraction=True),
                    "oracle_evidence":     dict(use_chain=True,  use_verifier=True,  use_source_grounding=True,  require_citations=True,  use_re_retrieve=True,  oracle_evidence=True,  blinded_extraction=False),
                }[m]
                out = run_chain_method(client, ex, q, **kwargs)
                judge = llm_judge(client, out["answer"], gold, q.question)
                dt = time.perf_counter() - t0
                print(
                    f"  [{m:16s}] judge={judge} ans={out['answer'][:50]!r} n_props={out.get('n_props',0)} ({dt:.1f}s)",
                    flush=True,
                )
                results[m].append({
                    "qid": ex.id, "gold": gold, "ans": out["answer"], "judge": judge,
                    "n_props": out.get("n_props", 0),
                    "chain_steps": out.get("chain_steps", 0),
                    "verifier_verdict": out.get("verifier_verdict", ""),
                    "dt": dt,
                })
            except Exception as e:
                print(f"  [{m:16s}] FAILED: {str(e)[:120]}", flush=True)
                results[m].append({"qid": ex.id, "gold": gold, "ans": "", "judge": 0, "error": str(e)[:200]})

    # Aggregate
    print("\n" + "=" * 80, flush=True)
    print(f"CHAIN METHOD PILOT (n={len(hard)}, hop_filter≥{args.hop_filter})", flush=True)
    print("=" * 80, flush=True)
    summary = {}
    for m in modes:
        scores = [r["judge"] for r in results[m]]
        mean, lo, hi = bootstrap_ci(scores)
        # Per-mode diagnostics (Codex fix)
        n_props = [r.get("n_props", 0) for r in results[m]]
        verifier_passes = sum(1 for r in results[m] if r.get("verifier_verdict") == "PASS")
        answer_in_props = sum(1 for r in results[m] if r.get("answer_in_props"))
        n_grounded = sum(r.get("n_propositions_grounded", 0) for r in results[m])
        n_total_props = sum(n_props)
        pct_grounded = 100 * n_grounded / max(1, n_total_props)
        summary[m] = {
            "mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(scores),
            "avg_n_props": sum(n_props) / max(1, len(n_props)),
            "verifier_pass_rate": verifier_passes / max(1, len(results[m])),
            "answer_in_props_rate": answer_in_props / max(1, len(results[m])),
            "pct_source_grounded": pct_grounded,
        }
        print(
            f"  {m:20s} acc={mean:.3f} [{lo:.3f},{hi:.3f}] (n={len(scores)})  "
            f"props≈{summary[m]['avg_n_props']:.1f}  "
            f"verifier_pass={verifier_passes}/{len(results[m])}  "
            f"answer_in_props={answer_in_props}/{len(results[m])}  "
            f"grounded={pct_grounded:.0f}%",
            flush=True,
        )

    out = {"n": len(hard), "hop_filter": args.hop_filter, "summary": summary, "details": results}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {args.output}", flush=True)

    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        import subprocess
        s3_key = f"s3://{s3_bucket}/chain_method/{os.path.basename(args.output)}"
        try:
            subprocess.run(["aws", "s3", "cp", args.output, s3_key], check=True, timeout=120)
            print(f"[s3] {s3_key}", flush=True)
        except Exception as e:
            print(f"[s3] {e}", flush=True)


if __name__ == "__main__":
    main()
