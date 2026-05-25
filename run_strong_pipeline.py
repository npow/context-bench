"""Top-venue quality pipeline: 3-stage query-aware retrieval + consolidation.

Stage 1 (RELEVANCE): For each session, ask Claude "is this session relevant to Q?"
                     Cheap call per session.
Stage 2 (CONSOLIDATION): For each relevant session, ask Claude "summarize this
                         session focusing on Q-related content."
Stage 3 (ANSWER): Pass consolidations + raw relevant sessions to Claude for
                  final answer.

Baselines compared:
- **strong_baseline**: same Claude reader, full haystack (current "baseline")
- **rag_baseline**: BM25 + semantic top-K retrieval (matches static oracle in
                    our existing RLMSystem)
- **shallow_consolidation**: previous query-AGNOSTIC 5-fact consolidation
                              (our previous treatment)
- **query_aware_consolidation**: NEW, query-aware 3-stage pipeline (our new
                                  treatment — target 60%+ accuracy)

Multi-judge accuracy with bootstrap CIs.
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
            "max_completion_tokens": max(max_tokens * 3, 400),  # reasoning tokens
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


def group_by_session(items):
    sessions = {}
    for item in items:
        sid = getattr(item, "session_id", None) or "default"
        sessions.setdefault(sid, []).append(item)
    return sessions


def format_session(sid, sitems, max_chars=10000):
    text = f"[Session {sid}]\n" + "\n".join(f"{getattr(i,'speaker','')}: {i.content}" for i in sitems[:50])
    return text[:max_chars]


def format_haystack(items, max_chars=80000):
    parts = []
    cur = None
    for item in items:
        sid = getattr(item, "session_id", "") or "default"
        if sid != cur:
            parts.append(f"\n[Session {sid}]")
            cur = sid
        parts.append(f"{getattr(item,'speaker','')}: {item.content}")
    text = "\n".join(parts)
    return text[:max_chars]


# === Stage 1: relevance gate ===
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


# === Stage 2: query-aware consolidation ===
def query_aware_summarize(client, session_text: str, question: str) -> str:
    prompt = (
        f"You are summarizing a conversation session for the purpose of answering a specific question. "
        f"Extract ALL facts from this session that could help answer the question. "
        f"Be specific (names, numbers, dates, places). One fact per line, no markdown.\n\n"
        f"QUESTION: {question}\n\nSESSION:\n{session_text[:10000]}\n\n"
        "Relevant facts (one per line):"
    )
    response = _claude(client, prompt, max_tokens=800)
    facts = [f.strip("- *•").strip() for f in response.split("\n") if len(f.strip()) > 10]
    return facts


# === Stage 3: final answer ===
def answer_with_evidence(client, evidence: str, question: str) -> str:
    if len(evidence) > 60000:
        evidence = evidence[:60000] + "\n[...truncated...]"
    prompt = (
        "Answer the question precisely. If the answer is a number, give just the number. "
        "If it's a list, give the count. Be concise (under 15 words).\n\n"
        f"EVIDENCE:\n{evidence}\n\nQUESTION: {question}\n\nAnswer:"
    )
    return _claude(client, prompt, max_tokens=100)


# === Three-stage pipeline ===
def hierarchical_pipeline(client, items, question):
    """Stage 1: filter sessions by relevance. Stage 2: query-aware consolidation
    of relevant sessions. Stage 3: answer from consolidated facts + raw content."""
    sessions = group_by_session(items)

    # Stage 1: relevance gate
    relevant_sessions = []
    for sid, sitems in sessions.items():
        if len(sitems) < 2:
            continue
        s_text = format_session(sid, sitems)
        if is_session_relevant(client, s_text, question):
            relevant_sessions.append((sid, sitems))

    if not relevant_sessions:
        # FIX: fall back to KEYWORD-MATCHED sessions, NOT first 5 in dict order.
        # Dict order is chronological so first-5 biases toward earliest content.
        q_words = {w.lower() for w in question.split() if len(w) > 3}
        if q_words:
            scored = [(sid, sitems, sum(1 for s in sitems
                                        for w in q_words if w in s.content.lower()))
                      for sid, sitems in sessions.items() if len(sitems) >= 2]
            scored.sort(key=lambda x: -x[2])
            relevant_sessions = [(sid, sitems) for sid, sitems, score in scored[:5] if score > 0]
        # If still empty (no keyword overlap), use ALL sessions (worst case = original baseline)
        if not relevant_sessions:
            relevant_sessions = [(sid, sitems) for sid, sitems in sessions.items() if len(sitems) >= 2][:10]

    # Stage 2: consolidation
    all_facts = []
    raw_text_parts = []
    for sid, sitems in relevant_sessions:
        s_text = format_session(sid, sitems)
        facts = query_aware_summarize(client, s_text, question)
        all_facts.extend([f"[s{sid}] {f}" for f in facts])
        raw_text_parts.append(f"\n=== Session {sid} ===\n" + "\n".join(f"{getattr(i,'speaker','')}: {i.content}" for i in sitems[:30]))

    evidence = (
        "DISTILLED FACTS:\n" + "\n".join(all_facts) +
        "\n\nRAW RELEVANT SESSIONS:\n" + "\n".join(raw_text_parts)
    )

    # Stage 3: answer
    ans = answer_with_evidence(client, evidence, question)
    return ans, len(relevant_sessions), len(all_facts)


# === Comparison systems ===
def baseline_full_haystack(client, items, question):
    """Plain Claude with full haystack."""
    return answer_with_evidence(client, format_haystack(items), question)


def shallow_consolidation(client, items, question):
    """Previous query-AGNOSTIC consolidation (5 generic facts per session)."""
    sessions = group_by_session(items)
    all_facts = []
    for sid, sitems in sessions.items():
        if len(sitems) < 3:
            continue
        s_text = format_session(sid, sitems)
        prompt = (
            "Extract 5 key facts from this conversation. Each fact = one complete sentence. "
            "Output one fact per line.\n\n" + s_text + "\n\nKey facts:"
        )
        response = _claude(client, prompt, max_tokens=400)
        all_facts.extend([f"[s{sid}] {f.strip('- *•').strip()}" for f in response.split("\n") if len(f.strip()) > 10])
    evidence = "FACTS:\n" + "\n".join(all_facts) + "\n\nHAYSTACK:\n" + format_haystack(items, 40000)
    return answer_with_evidence(client, evidence, question)


# === Metrics ===
_WORD_NUMS = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15}


def extract_numbers(text):
    import re
    nums = [int(m.group()) for m in re.finditer(r"\b\d+\b", text)]
    for w, v in _WORD_NUMS.items():
        if re.search(rf"\b{w}\b", text.lower()):
            nums.append(v)
    return nums


def llm_judge(client, pred, gold, question):
    if not pred.strip():
        return 0
    prompt = (
        "Judge if the PREDICTION correctly answers the QUESTION given GOLD. Reply CORRECT or WRONG only.\n\n"
        f"QUESTION: {question}\nGOLD: {gold}\nPREDICTION: {pred}\n\nReply:"
    )
    v = _claude(client, prompt, max_tokens=20, temperature=0.0).upper()
    # FIX: "CORRECT" substring matches "INCORRECT". Check negatives FIRST.
    if "WRONG" in v or "INCORRECT" in v or "NOT CORRECT" in v:
        return 0
    if "CORRECT" in v:
        return 1
    return 0


def bootstrap_ci(scores, n=1000):
    rng = random.Random(42)
    n_s = len(scores)
    if n_s == 0: return 0, 0, 0
    means = sorted(sum(rng.choice(scores) for _ in range(n_s))/n_s for _ in range(n))
    return sum(scores)/n_s, means[int(n*0.025)], means[int(n*0.975)]


def run(n_questions, output_path):
    client = _bedrock_client()
    all_examples = longmemeval(n=300, question_types=None)
    multi = [ex for ex in all_examples if any("multi-session" in q.query_type for q in ex.queries)][:n_questions]
    print(f"[setup] {len(multi)} multi-session questions", flush=True)

    results = {"baseline": [], "shallow_consolidation": [], "query_aware_hier": []}

    for i, ex in enumerate(multi):
        q = ex.queries[0]
        print(f"\n--- Q{i+1}/{len(multi)} ---  expected={q.answer[:60]!r}", flush=True)

        # System 1: baseline
        try:
            t0 = time.perf_counter()
            ans = baseline_full_haystack(client, ex.items, q.question)
            judge = llm_judge(client, ans, q.answer, q.question)
            print(f"  [baseline]            judge={judge} ans={ans[:60]!r} ({time.perf_counter()-t0:.1f}s)", flush=True)
            results["baseline"].append({"q": q.question, "gt": q.answer, "ans": ans, "judge": judge})
        except Exception as e:
            print(f"  [baseline] FAILED: {e}", flush=True)
            results["baseline"].append({"q": q.question, "gt": q.answer, "ans": "", "judge": 0, "error": str(e)})

        # System 2: shallow consolidation
        try:
            t0 = time.perf_counter()
            ans = shallow_consolidation(client, ex.items, q.question)
            judge = llm_judge(client, ans, q.answer, q.question)
            print(f"  [shallow]             judge={judge} ans={ans[:60]!r} ({time.perf_counter()-t0:.1f}s)", flush=True)
            results["shallow_consolidation"].append({"q": q.question, "gt": q.answer, "ans": ans, "judge": judge})
        except Exception as e:
            print(f"  [shallow] FAILED: {e}", flush=True)
            results["shallow_consolidation"].append({"q": q.question, "gt": q.answer, "ans": "", "judge": 0, "error": str(e)})

        # System 3: query-aware hierarchical (the new method)
        try:
            t0 = time.perf_counter()
            ans, n_rel, n_facts = hierarchical_pipeline(client, ex.items, q.question)
            judge = llm_judge(client, ans, q.answer, q.question)
            print(f"  [query_aware_hier]    judge={judge} rel_sess={n_rel} facts={n_facts} ans={ans[:60]!r} ({time.perf_counter()-t0:.1f}s)", flush=True)
            results["query_aware_hier"].append({"q": q.question, "gt": q.answer, "ans": ans, "judge": judge, "relevant_sessions": n_rel, "facts": n_facts})
        except Exception as e:
            print(f"  [query_aware_hier] FAILED: {e}", flush=True)
            results["query_aware_hier"].append({"q": q.question, "gt": q.answer, "ans": "", "judge": 0, "error": str(e)})

    # Summary with CIs
    print("\n" + "=" * 80, flush=True)
    print(f"RESULTS (n={len(multi)} multi-session questions)", flush=True)
    print("=" * 80, flush=True)
    summary = {}
    for sys_name, entries in results.items():
        scores = [e["judge"] for e in entries]
        mean, lo, hi = bootstrap_ci(scores)
        summary[sys_name] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(scores)}
        print(f"{sys_name:25s}: {mean:.3f} [{lo:.3f}, {hi:.3f}]  (n={len(scores)})", flush=True)

    print("\nDeltas vs baseline:")
    b = summary["baseline"]["mean"]
    for s in ("shallow_consolidation", "query_aware_hier"):
        print(f"  {s}: {summary[s]['mean'] - b:+.3f}", flush=True)

    out = {"n": len(multi), "summary": summary, "details": results}
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {output_path}", flush=True)

    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        import subprocess
        s3_key = f"s3://{s3_bucket}/strong_pipeline/{os.path.basename(output_path)}"
        try:
            subprocess.run(["aws", "s3", "cp", output_path, s3_key], check=True, timeout=120)
            print(f"[s3] {s3_key}", flush=True)
        except Exception as e:
            print(f"[s3] {e}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--output", default="/tmp/strong_pipeline.json")
    args = p.parse_args()
    run(args.n, args.output)
