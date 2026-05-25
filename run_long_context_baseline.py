"""1M-context baseline: dump entire haystack into a long-context model.

If frontier long-context models solve multi-session memory directly, our
consolidation pipeline is unnecessary. This script tests that hypothesis.

Three baselines:
1. Claude Sonnet 4.6 — 200K (standard), no truncation
2. Claude Sonnet 4.6 — 1M (beta long-context header, if Bedrock supports)
3. Claude Opus 4.7 — 200K (smartest available on Bedrock)
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


def _claude_call(client, prompt: str, model_id: str, max_tokens: int = 200,
                 use_1m_beta: bool = False) -> str:
    """Call Anthropic Claude on Bedrock with optional 1M beta header."""
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    if use_1m_beta:
        body["anthropic_beta"] = ["long-context-2024-10-01"]

    r = client.invoke_model(
        body=json.dumps(body),
        modelId=model_id,
        accept="application/json", contentType="application/json",
    )
    return json.loads(r["body"].read())["content"][0]["text"].strip()


def format_full_haystack(items, max_chars: int = 900_000) -> str:
    """Full haystack — only truncate if absolutely necessary (1M chars ≈ 250K tokens)."""
    parts = []
    cur = None
    for item in items:
        sid = getattr(item, "session_id", "") or "default"
        if sid != cur:
            parts.append(f"\n[Session {sid}]")
            cur = sid
        speaker = getattr(item, "speaker", "")
        parts.append(f"{speaker}: {item.content}")
    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[...truncated...]"
    return text


def answer(client, items, question: str, model_id: str, use_1m_beta: bool, max_chars: int) -> str:
    haystack = format_full_haystack(items, max_chars=max_chars)
    prompt = (
        "Answer the question precisely. If the answer is a number or count, give just the number. "
        "Be concise (under 15 words). Search the FULL conversation history carefully — the answer is in there.\n\n"
        f"CONVERSATION HISTORY:\n{haystack}\n\nQUESTION: {question}\n\nAnswer:"
    )
    return _claude_call(client, prompt, model_id, max_tokens=100, use_1m_beta=use_1m_beta)


def llm_judge(client, pred, gold, question) -> int:
    if not pred.strip():
        return 0
    prompt = (
        "Judge if the PREDICTION correctly answers the QUESTION given GOLD. Reply CORRECT or WRONG only.\n\n"
        f"QUESTION: {question}\nGOLD: {gold}\nPREDICTION: {pred}\n\nReply:"
    )
    v = _claude_call(client, prompt, "us.anthropic.claude-sonnet-4-6", max_tokens=20).upper()
    # FIX: check negatives first
    if "WRONG" in v or "INCORRECT" in v or "NOT CORRECT" in v:
        return 0
    return 1 if "CORRECT" in v else 0


def bootstrap_ci(scores, n=1000):
    rng = random.Random(42)
    n_s = len(scores)
    if n_s == 0:
        return 0.0, 0.0, 0.0
    means = sorted(sum(rng.choice(scores) for _ in range(n_s)) / n_s for _ in range(n))
    return sum(scores) / n_s, means[int(n * 0.025)], means[int(n * 0.975)]


def run(n_questions, output_path):
    client = _bedrock_client()
    all_examples = longmemeval(n=300, question_types=None)
    multi = [ex for ex in all_examples if any("multi-session" in q.query_type for q in ex.queries)][:n_questions]
    print(f"[setup] {len(multi)} multi-session questions", flush=True)

    systems = [
        ("sonnet_200k", "us.anthropic.claude-sonnet-4-6", False, 600_000),  # standard 200K
        ("sonnet_1m_beta", "us.anthropic.claude-sonnet-4-6", True, 900_000),  # 1M beta header
        ("opus_47", "us.anthropic.claude-opus-4-7", False, 600_000),  # smartest
    ]

    results = {name: [] for name, _, _, _ in systems}

    for i, ex in enumerate(multi):
        q = ex.queries[0]
        gold = q.answer
        print(f"\n--- Q{i+1}/{len(multi)} ---  expected={gold[:60]!r}", flush=True)

        for sys_name, model_id, use_1m, max_chars in systems:
            try:
                t0 = time.perf_counter()
                ans = answer(client, ex.items, q.question, model_id, use_1m, max_chars)
                judge = llm_judge(client, ans, gold, q.question)
                dt = time.perf_counter() - t0
                print(f"  [{sys_name:15s}] judge={judge} ans={ans[:60]!r} ({dt:.1f}s)", flush=True)
                results[sys_name].append({"q": q.question, "gt": gold, "ans": ans, "judge": judge})
            except Exception as e:
                err_msg = str(e)[:200]
                print(f"  [{sys_name:15s}] FAILED: {err_msg}", flush=True)
                results[sys_name].append({"q": q.question, "gt": gold, "ans": "", "judge": 0, "error": err_msg})

    print("\n" + "=" * 80, flush=True)
    print(f"LONG-CONTEXT BASELINE RESULTS (n={len(multi)})", flush=True)
    print("=" * 80, flush=True)
    summary = {}
    for sys_name in results:
        scores = [r["judge"] for r in results[sys_name]]
        mean, lo, hi = bootstrap_ci(scores)
        summary[sys_name] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(scores)}
        print(f"{sys_name:20s}: {mean:.3f} [{lo:.3f}, {hi:.3f}]  (n={len(scores)})", flush=True)

    out = {"n": len(multi), "summary": summary, "details": results}
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {output_path}", flush=True)

    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        import subprocess
        s3_key = f"s3://{s3_bucket}/long_context/{os.path.basename(output_path)}"
        try:
            subprocess.run(["aws", "s3", "cp", output_path, s3_key], check=True, timeout=120)
            print(f"[s3] {s3_key}", flush=True)
        except Exception as e:
            print(f"[s3] {e}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--output", default="/tmp/long_context.json")
    args = p.parse_args()
    run(args.n, args.output)
