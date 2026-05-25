"""Long-context-strong (CoT, structured search) prompt evaluated across reader families.

Codex demand: don't ship "compression beats long context" without testing a STRONG
long-context prompt on every reader.

The prompt asks the reader to:
1. List ALL sessions that could contain relevant info.
2. Extract candidate evidence quotes from each.
3. Synthesize an answer with citations.

Run on:
  - Claude Sonnet 4.6
  - Claude Haiku 4.5
  - GPT-OSS-120B (via Bedrock)
  - Gemma-3-27B (via Bedrock)

For each reader, on n multi-session questions from LongMemEval.
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


READERS = [
    ("sonnet_4_6", "us.anthropic.claude-sonnet-4-6", "anthropic"),
    ("haiku_4_5",  "us.anthropic.claude-haiku-4-5-20251001-v1:0", "anthropic"),
    ("gpt_oss_120b", "openai.gpt-oss-120b-1:0", "openai-chat"),
    ("gemma_3_27b", "google.gemma-3-27b-it", "openai-chat"),
]


def _bedrock_client():
    import boto3
    return boto3.client("bedrock-runtime", region_name="us-east-1")


def _call_anthropic(client, prompt, model_id, max_tokens=500):
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


def _call_openai_chat(client, prompt, model_id, max_tokens=500):
    """OpenAI/Chat-format API for gpt-oss / gemma on Bedrock."""
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
        "temperature": 0.0,
    }
    r = client.invoke_model(
        body=json.dumps(body), modelId=model_id,
        accept="application/json", contentType="application/json",
    )
    resp = json.loads(r["body"].read())
    return resp["choices"][0]["message"]["content"].strip()


def call_reader(client, prompt, reader_name, model_id, family, max_tokens=500):
    if family == "anthropic":
        return _call_anthropic(client, prompt, model_id, max_tokens)
    else:
        return _call_openai_chat(client, prompt, model_id, max_tokens)


def format_haystack(items, max_chars: int = 900_000) -> tuple[str, bool]:
    """Returns (text, was_truncated)."""
    parts = []
    cur = None
    for item in items:
        sid = getattr(item, "session_id", "") or "default"
        if sid != cur:
            parts.append(f"\n[Session {sid}]")
            cur = sid
        sp = getattr(item, "speaker", "") or ""
        parts.append(f"{sp}: {item.content}")
    text = "\n".join(parts)
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars] + "\n[...truncated...]"
    return text, truncated


STRONG_LC_PROMPT = (
    "You are answering a question about a long multi-session conversation. "
    "Follow this procedure EXACTLY:\n\n"
    "Step 1: SEARCH. Read through ALL sessions. Identify EVERY session that might "
    "contain information relevant to the question. List the session IDs.\n\n"
    "Step 2: EXTRACT. For each relevant session, quote the specific sentences that "
    "address the question.\n\n"
    "Step 3: ANSWER. Synthesize a precise answer using the extracted evidence. "
    "If the answer is a number or count, give just the number. Be concise.\n\n"
    "Format:\n"
    "RELEVANT SESSIONS: [list]\n"
    "EVIDENCE:\n[quoted excerpts with session IDs]\n"
    "ANSWER: [final precise answer, under 15 words]\n\n"
    "===========\n"
    "CONVERSATION HISTORY:\n{haystack}\n\n"
    "QUESTION: {question}\n"
)


def parse_answer(response: str) -> str:
    """Extract the ANSWER: line. Fallback to last line."""
    for line in response.split("\n"):
        if line.strip().upper().startswith("ANSWER:"):
            return line.split(":", 1)[1].strip()
    # Fallback
    return response.split("\n")[-1].strip()


def _judge_call(client, prompt, max_tokens=20):
    return _call_anthropic(client, prompt, "us.anthropic.claude-sonnet-4-6", max_tokens)


def llm_judge(client, pred, gold, question):
    if not pred.strip(): return 0
    prompt = (
        "Judge if the PREDICTION correctly answers the QUESTION given GOLD. Reply CORRECT or WRONG only.\n\n"
        f"QUESTION: {question}\nGOLD: {gold}\nPREDICTION: {pred}\n\nReply:"
    )
    v = _judge_call(client, prompt).upper()
    if "WRONG" in v or "INCORRECT" in v or "NOT CORRECT" in v: return 0
    return 1 if "CORRECT" in v else 0


def bootstrap_ci(scores, n=1000):
    if not scores: return 0.0, 0.0, 0.0
    rng = random.Random(42)
    n_s = len(scores)
    means = sorted(sum(rng.choice(scores) for _ in range(n_s)) / n_s for _ in range(n))
    return sum(scores) / n_s, means[int(n * 0.025)], means[int(n * 0.975)]


def run(n_questions, output_path):
    client = _bedrock_client()
    all_examples = longmemeval(n=300, question_types=None)
    multi = [ex for ex in all_examples if any("multi-session" in q.query_type for q in ex.queries)][:n_questions]
    print(f"[setup] {len(multi)} multi-session questions × {len(READERS)} readers", flush=True)

    # ===== Reader API dry-run (Codex fix #5) =====
    # Verify each reader API works with a trivial prompt BEFORE the main loop.
    print("\n[dry-run] verifying reader APIs...", flush=True)
    working_readers = []
    for reader_name, model_id, family in READERS:
        try:
            r = call_reader(client, "Reply with the single word READY.", reader_name, model_id, family, max_tokens=20)
            if r:
                print(f"  [{reader_name:14s}] OK: {r[:40]!r}", flush=True)
                working_readers.append((reader_name, model_id, family))
            else:
                print(f"  [{reader_name:14s}] EMPTY RESPONSE — skipping", flush=True)
        except Exception as e:
            print(f"  [{reader_name:14s}] FAILED dry-run: {str(e)[:150]} — skipping", flush=True)
    if not working_readers:
        print("[fatal] no working readers", flush=True)
        return

    results = {name: [] for name, _, _ in working_readers}

    for i, ex in enumerate(multi):
        q = ex.queries[0]
        gold = q.answer
        question = q.question
        haystack, truncated = format_haystack(ex.items, max_chars=900_000)
        prompt = STRONG_LC_PROMPT.format(haystack=haystack, question=question)
        print(f"\n--- Q{i+1}/{len(multi)} id={ex.id} haystack_chars={len(haystack)} truncated={truncated} ---", flush=True)

        for reader_name, model_id, family in working_readers:
            try:
                t0 = time.perf_counter()
                # Codex fix #7: bump to 1500 to fit RELEVANT/EVIDENCE/ANSWER sections
                response = call_reader(client, prompt, reader_name, model_id, family, max_tokens=1500)
                ans = parse_answer(response)
                # Robust fallback: if ANSWER: not found, take last non-empty line
                if not ans.strip():
                    lines = [l for l in response.split("\n") if l.strip()]
                    ans = lines[-1] if lines else ""
                judge = llm_judge(client, ans, gold, question)
                dt = time.perf_counter() - t0
                print(f"  [{reader_name:14s}] judge={judge} ans={ans[:60]!r} ({dt:.1f}s)", flush=True)
                results[reader_name].append({
                    "qid": ex.id, "question": question, "gold": gold,
                    "ans": ans, "judge": judge, "response_chars": len(response),
                    "haystack_truncated": truncated,
                    "dt": dt,
                })
            except Exception as e:
                err = str(e)[:200]
                print(f"  [{reader_name:14s}] FAILED: {err}", flush=True)
                results[reader_name].append({
                    "qid": ex.id, "question": question, "gold": gold,
                    "ans": "", "judge": 0, "error": err,
                    "haystack_truncated": truncated,
                })

    # ===== Aggregate =====
    print("\n" + "=" * 80, flush=True)
    print(f"LONG-CONTEXT-STRONG ACROSS READER FAMILIES (n={len(multi)})", flush=True)
    print("=" * 80, flush=True)
    summary = {}
    for reader_name in results:
        scores = [r["judge"] for r in results[reader_name]]
        mean, lo, hi = bootstrap_ci(scores)
        n_truncated = sum(1 for r in results[reader_name] if r.get("haystack_truncated"))
        summary[reader_name] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(scores), "n_truncated": n_truncated}
        print(f"  {reader_name:18s}  {mean:.3f} [{lo:.3f}, {hi:.3f}]  n={len(scores)}  (truncated={n_truncated})", flush=True)

    out = {"n": len(multi), "summary": summary, "details": results}
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {output_path}", flush=True)

    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        import subprocess
        s3_key = f"s3://{s3_bucket}/long_ctx_strong/{os.path.basename(output_path)}"
        try:
            subprocess.run(["aws", "s3", "cp", output_path, s3_key], check=True, timeout=120)
            print(f"[s3] {s3_key}", flush=True)
        except Exception as e:
            print(f"[s3] {e}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--output", default="/tmp/long_ctx_strong_xfam.json")
    args = p.parse_args()
    run(args.n, args.output)
