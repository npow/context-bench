"""Multi-session LongMemEval: full-context oracle vs full-context + consolidation.

Simpler experiment that avoids LanceDB/embedding overhead and works in
isolated Mako/Titus containers via AWS Bedrock (TitusContainerRole IAM).

Two systems:
1. Baseline: Claude Sonnet receives full session haystack in context, answers.
2. Treatment: Claude Sonnet first consolidates each session into key facts,
   then answers using the consolidated facts as additional context.

Hypothesis: consolidation gives Claude a clearer signal of cross-session
relevant facts, improving F1 on multi-session questions where the answer
is buried in long histories.
"""
from __future__ import annotations
import json
import os
import sys
import time

sys.path.insert(0, "src")

from context_bench.datasets.memory.longmemeval import longmemeval


def _bedrock_client():
    import boto3
    return boto3.client("bedrock-runtime", region_name="us-east-1")


def _claude_call(client, prompt: str, max_tokens: int = 200) -> str:
    """Single Claude call via Bedrock."""
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    })
    response = client.invoke_model(
        body=body,
        modelId=os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-sonnet-20241022-v2:0"),
        accept="application/json",
        contentType="application/json",
    )
    resp_body = json.loads(response["body"].read())
    return resp_body["content"][0]["text"].strip()


def _f1(prediction: str, gold: str) -> float:
    import re
    def normalize(s):
        s = re.sub(r"[^\w\s]", "", s.lower()).strip()
        return [w for w in s.split() if w not in {"a", "an", "the", "is", "was", "are", "were"}]

    p_toks = set(normalize(prediction))
    g_toks = set(normalize(gold))
    if not p_toks or not g_toks:
        return 0.0
    common = p_toks & g_toks
    if not common:
        return 0.0
    prec = len(common) / len(p_toks)
    rec = len(common) / len(g_toks)
    return 2 * prec * rec / (prec + rec)


def format_haystack(items, max_chars: int = 80000) -> str:
    """Format conversation items as a single context string."""
    parts = []
    cur_session = None
    for item in items:
        sid = getattr(item, "session_id", "") or "default"
        if sid != cur_session:
            parts.append(f"\n[Session {sid}]")
            cur_session = sid
        speaker = getattr(item, "speaker", "")
        parts.append(f"{speaker}: {item.content}")
    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[...truncated...]"
    return text


def consolidate_session(client, session_items) -> list[str]:
    """Ask Claude to extract key facts from a session."""
    session_text = "\n".join([f"{i.speaker}: {i.content}" for i in session_items[:50]])
    if len(session_text) > 12000:
        session_text = session_text[:12000]

    prompt = (
        "Extract 3-7 key facts from this conversation session as short standalone "
        "statements. Each fact should be ONE complete sentence. Include specific "
        "names, numbers, dates, places mentioned. Output one fact per line, no "
        "numbering or markdown.\n\n"
        f"Session:\n{session_text}\n\nKey facts:"
    )
    response = _claude_call(client, prompt, max_tokens=500)
    facts = [f.strip("- *•").strip() for f in response.split("\n")]
    return [f for f in facts if len(f) > 10]


def answer_question(client, context: str, question: str) -> str:
    """Ask Claude to answer a question given context."""
    if len(context) > 80000:
        context = context[:80000] + "\n[...truncated...]"
    prompt = (
        "Answer this question in ONE short sentence or phrase (under 15 words). "
        "Just the direct answer, no explanations.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nShort answer:"
    )
    return _claude_call(client, prompt, max_tokens=100)


def run_experiment(n_questions: int = 20, output_path: str = "/tmp/multisession_v3_results.json"):
    client = _bedrock_client()
    print(f"[setup] Loading LongMemEval multi-session questions (target n={n_questions})", flush=True)

    all_examples = longmemeval(n=200, question_types=None)
    multi = [ex for ex in all_examples if any("multi-session" in q.query_type for q in ex.queries)][:n_questions]
    print(f"[setup] Loaded {len(multi)} multi-session questions", flush=True)

    results = {"baseline": [], "treatment": []}

    for i, ex in enumerate(multi):
        q = ex.queries[0]
        print(f"\n--- Q{i+1}/{len(multi)} ---", flush=True)
        print(f"  Question: {q.question[:90]!r}", flush=True)
        print(f"  Expected: {q.answer!r}", flush=True)

        # Group items by session
        sessions = {}
        for item in ex.items:
            sid = getattr(item, "session_id", None) or "default"
            sessions.setdefault(sid, []).append(item)
        print(f"  Sessions: {len(sessions)} (total items: {len(ex.items)})", flush=True)

        # === BASELINE ===
        try:
            t0 = time.perf_counter()
            haystack = format_haystack(ex.items)
            ans = answer_question(client, haystack, q.question)
            f1 = _f1(ans, q.answer)
            dt = time.perf_counter() - t0
            print(f"  [baseline]  f1={f1:.2f} ({dt:.1f}s) ans_full={ans!r}", flush=True)
            results["baseline"].append({"q": q.question, "gt": q.answer, "ans": ans, "f1": f1})
        except Exception as e:
            print(f"  [baseline] FAILED: {e}", flush=True)
            results["baseline"].append({"q": q.question, "gt": q.answer, "ans": "", "f1": 0.0, "error": str(e)})

        # === TREATMENT: consolidate + answer ===
        try:
            t0 = time.perf_counter()
            consolidated_facts = []
            for sid, sitems in sessions.items():
                if len(sitems) < 3:
                    continue
                facts = consolidate_session(client, sitems)
                consolidated_facts.extend([f"[from session {sid}] {f}" for f in facts])
            consolidated_text = "\n".join(consolidated_facts) + "\n\n" + format_haystack(ex.items, max_chars=40000)
            ans = answer_question(client, consolidated_text, q.question)
            f1 = _f1(ans, q.answer)
            dt = time.perf_counter() - t0
            print(f"  [treatment] f1={f1:.2f} ({dt:.1f}s) writes={len(consolidated_facts)} ans_full={ans!r}", flush=True)
            results["treatment"].append({
                "q": q.question, "gt": q.answer, "ans": ans, "f1": f1,
                "n_consolidated_facts": len(consolidated_facts),
            })
        except Exception as e:
            print(f"  [treatment] FAILED: {e}", flush=True)
            results["treatment"].append({"q": q.question, "gt": q.answer, "ans": "", "f1": 0.0, "error": str(e)})

    # === Summary ===
    n = len(results["baseline"])
    b_f1s = [r["f1"] for r in results["baseline"]]
    t_f1s = [r["f1"] for r in results["treatment"]]

    print("\n" + "=" * 60, flush=True)
    print(f"RESULTS (n={n} multi-session questions):", flush=True)
    print(f"  Baseline (full-context oracle)    : F1 = {sum(b_f1s)/n:.3f}", flush=True)
    print(f"  Treatment (oracle + consolidation): F1 = {sum(t_f1s)/n:.3f}", flush=True)
    print(f"  Delta                             : {(sum(t_f1s)/n - sum(b_f1s)/n):+.3f}", flush=True)
    won_treat = sum(1 for b, t in zip(b_f1s, t_f1s) if t > b)
    won_base = sum(1 for b, t in zip(b_f1s, t_f1s) if b > t)
    print(f"  Won by treatment / tied / baseline: {won_treat}/{n - won_treat - won_base}/{won_base}", flush=True)

    out = {
        "n": n,
        "baseline_mean_f1": sum(b_f1s) / n if n else 0,
        "treatment_mean_f1": sum(t_f1s) / n if n else 0,
        "delta_f1": (sum(t_f1s) / n - sum(b_f1s) / n) if n else 0,
        "won_by_treatment": won_treat,
        "won_by_baseline": won_base,
        "details": results,
    }
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {output_path}", flush=True)

    # Upload to S3 so results survive container exit (set S3_BUCKET env var)
    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        import subprocess
        s3_key = f"s3://{s3_bucket}/multisession_v3/{os.path.basename(output_path)}"
        try:
            subprocess.run(["aws", "s3", "cp", output_path, s3_key], check=True, timeout=60)
            print(f"[s3] Uploaded to {s3_key}", flush=True)
        except Exception as e:
            print(f"[s3] Upload failed: {e}", flush=True)
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--output", default="/tmp/multisession_v3_results.json")
    args = p.parse_args()
    run_experiment(n_questions=args.n, output_path=args.output)
