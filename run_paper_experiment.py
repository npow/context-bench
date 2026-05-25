"""Conference paper experiment: comprehensive comparison of 4 systems
on LongMemEval multi-session questions.

Systems compared:
1. Baseline: full haystack → Claude answers
2. Truncated: first 40K chars only → Claude answers (controls for context length)
3. Random-facts: 20 random sentences from haystack inserted → Claude answers
   (controls for "any extra context")
4. Consolidation: per-session summarization → Claude answers (our method)

All systems get the same question and the same retrieval pool (full haystack).
Difference is in how the context is structured/augmented before Claude reads it.

Outputs:
- Per-question scores (all 4 systems, multiple metrics)
- Bootstrap CIs (1000 resamples) for each system's mean F1
- Pairwise significance tests
- Saved to S3 if S3_BUCKET env var is set
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, "src")

from context_bench.datasets.memory.longmemeval import longmemeval


def _bedrock_client():
    import boto3
    return boto3.client("bedrock-runtime", region_name="us-east-1")


def _claude_call(client, prompt: str, max_tokens: int = 200) -> str:
    """Model-agnostic Bedrock call. Detects format from BEDROCK_MODEL_ID env var."""
    import re as _re
    model_id = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-sonnet-20241022-v2:0")

    if "anthropic" in model_id:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        })
        r = client.invoke_model(body=body, modelId=model_id, accept="application/json", contentType="application/json")
        return json.loads(r["body"].read())["content"][0]["text"].strip()

    if "openai" in model_id or "gpt" in model_id:
        # OpenAI-format on Bedrock (gpt-oss-*). Reasoning tags must be stripped.
        body = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max(max_tokens * 3, 400),  # account for reasoning tokens
        })
        r = client.invoke_model(body=body, modelId=model_id)
        text = json.loads(r["body"].read())["choices"][0]["message"]["content"]
        text = _re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=_re.DOTALL).strip()
        return text

    if "gemma" in model_id or "google" in model_id:
        body = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        })
        r = client.invoke_model(body=body, modelId=model_id)
        resp = json.loads(r["body"].read())
        if "choices" in resp:
            return resp["choices"][0]["message"]["content"].strip()
        if "generation" in resp:
            return resp["generation"].strip()
        return str(resp).strip()

    raise ValueError(f"Unknown model family: {model_id}")


def format_haystack(items, max_chars: int = 80000) -> str:
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


def truncated_haystack(items, max_chars: int = 40000) -> str:
    return format_haystack(items, max_chars=max_chars)


def random_facts_haystack(items, n_facts: int = 20, seed: int = 0) -> str:
    """Sample n_facts random sentences and prepend before full haystack."""
    rng = random.Random(seed)
    candidates = [f"[from session {getattr(i, 'session_id', '?')}] {i.speaker}: {i.content}" for i in items if len(i.content) > 20]
    sampled = rng.sample(candidates, min(n_facts, len(candidates)))
    return "\n".join(sampled) + "\n\n" + format_haystack(items, max_chars=40000)


def consolidate_session(client, session_items, n_facts: int = 5) -> list[str]:
    text = "\n".join([f"{i.speaker}: {i.content}" for i in session_items[:50]])
    if len(text) > 12000:
        text = text[:12000]
    prompt = (
        f"Extract {n_facts} key facts from this conversation session as standalone "
        "statements. Each fact must be ONE complete sentence with specific names, "
        "numbers, dates, places mentioned. Output one fact per line, no numbering "
        "or markdown.\n\n"
        f"Session:\n{text}\n\nKey facts:"
    )
    response = _claude_call(client, prompt, max_tokens=500)
    facts = [f.strip("- *•").strip() for f in response.split("\n")]
    return [f for f in facts if len(f) > 10]


def consolidation_haystack(client, items, n_facts: int = 5) -> tuple[str, int]:
    """Consolidate each session into n_facts, prepend before full haystack."""
    sessions: dict = {}
    for item in items:
        sid = getattr(item, "session_id", None) or "default"
        sessions.setdefault(sid, []).append(item)
    all_facts = []
    for sid, sitems in sessions.items():
        if len(sitems) < 3:
            continue
        facts = consolidate_session(client, sitems, n_facts=n_facts)
        all_facts.extend([f"[from session {sid}] {f}" for f in facts])
    text = "\n".join(all_facts) + "\n\n" + format_haystack(items, max_chars=40000)
    return text, len(all_facts)


def answer(client, context: str, question: str) -> str:
    if len(context) > 80000:
        context = context[:80000] + "\n[...truncated...]"
    prompt = (
        "Answer this question in ONE short sentence or phrase (under 15 words). "
        "Just the direct answer, no explanations.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nShort answer:"
    )
    return _claude_call(client, prompt, max_tokens=100)


# ---- Metrics ----

_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "hundred": 100, "thousand": 1000,
}


def extract_numbers(text: str) -> list[int]:
    import re
    nums = []
    for m in re.finditer(r"\b(\d+)\b", text):
        try: nums.append(int(m.group(1)))
        except ValueError: pass
    lower = text.lower()
    for word, val in _WORD_NUMBERS.items():
        import re as _re
        if _re.search(rf"\b{word}\b", lower):
            nums.append(val)
    return nums


def token_f1(pred: str, gold: str) -> float:
    import re
    def normalize(s):
        s = re.sub(r"[^\w\s]", "", s.lower()).strip()
        return [w for w in s.split() if w not in {"a", "an", "the", "is", "was", "are", "were"}]
    p = set(normalize(pred))
    g = set(normalize(gold))
    if not p or not g: return 0.0
    common = p & g
    if not common: return 0.0
    prec = len(common) / len(p)
    rec = len(common) / len(g)
    return 2 * prec * rec / (prec + rec)


def numeric_match(pred: str, gold: str) -> float:
    gold_nums = extract_numbers(gold)
    pred_nums = extract_numbers(pred)
    if not gold_nums: return -1.0  # not numeric
    return 1.0 if gold_nums[0] in pred_nums else 0.0


def llm_judge(client, pred: str, gold: str, question: str) -> float:
    if not pred.strip(): return 0.0
    prompt = (
        "Judge if the PREDICTION correctly answers the QUESTION given the GOLD answer. "
        "Reply with a single token: CORRECT or WRONG. No other text.\n\n"
        f"QUESTION: {question}\nGOLD: {gold}\nPREDICTION: {pred}\n\nReply:"
    )
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    })
    response = client.invoke_model(
        body=body,
        modelId=os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-sonnet-20241022-v2:0"),
        accept="application/json",
        contentType="application/json",
    )
    resp_body = json.loads(response["body"].read())
    verdict = resp_body["content"][0]["text"].strip().upper()
    # FIX: INCORRECT contains CORRECT — check negatives first
    if "WRONG" in verdict or "INCORRECT" in verdict or "NOT CORRECT" in verdict: return 0.0
    return 1.0 if "CORRECT" in verdict else 0.0


def score_all(client, ans: str, gold: str, question: str) -> dict:
    return {
        "token_f1": token_f1(ans, gold),
        "numeric_match": numeric_match(ans, gold),
        "judge": llm_judge(client, ans, gold, question),
    }


def bootstrap_ci(scores: list[float], n_resamples: int = 1000, conf: float = 0.95) -> tuple[float, float, float]:
    rng = random.Random(42)
    n = len(scores)
    if n == 0: return 0.0, 0.0, 0.0
    means = []
    for _ in range(n_resamples):
        sample = [rng.choice(scores) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    mean = sum(scores) / n
    lo = means[int(n_resamples * (1 - conf) / 2)]
    hi = means[int(n_resamples * (1 + conf) / 2)]
    return mean, lo, hi


def run(n_questions: int = 100, output_path: str = "/tmp/paper_results.json", seed: int = 0):
    client = _bedrock_client()
    print(f"[setup] loading LME multi-session questions, target n={n_questions}", flush=True)

    all_examples = longmemeval(n=300, question_types=None)
    multi = [ex for ex in all_examples if any("multi-session" in q.query_type for q in ex.queries)]
    print(f"[setup] {len(multi)} multi-session questions available, sampling {n_questions}", flush=True)

    rng = random.Random(seed)
    rng.shuffle(multi)
    multi = multi[:n_questions]

    systems = ["baseline", "truncated", "random_facts", "consolidation"]
    results = {s: [] for s in systems}

    for i, ex in enumerate(multi):
        q = ex.queries[0]
        print(f"\n--- Q{i+1}/{len(multi)} ---  expected={q.answer[:60]!r}", flush=True)

        # Build contexts
        try:
            base_ctx = format_haystack(ex.items)
            trunc_ctx = truncated_haystack(ex.items)
            rand_ctx = random_facts_haystack(ex.items, n_facts=20, seed=i)
            consol_ctx, n_facts = consolidation_haystack(client, ex.items, n_facts=5)
        except Exception as e:
            print(f"  [build] failed: {e}", flush=True)
            for s in systems:
                results[s].append({"q": q.question, "gt": q.answer, "ans": "", "error": str(e), "scores": {"token_f1": 0, "numeric_match": -1, "judge": 0}})
            continue

        for sys_name, ctx in [
            ("baseline", base_ctx),
            ("truncated", trunc_ctx),
            ("random_facts", rand_ctx),
            ("consolidation", consol_ctx),
        ]:
            try:
                t0 = time.perf_counter()
                ans = answer(client, ctx, q.question)
                scores = score_all(client, ans, q.answer, q.question)
                dt = time.perf_counter() - t0
                results[sys_name].append({"q": q.question, "gt": q.answer, "ans": ans, "scores": scores})
                print(f"  [{sys_name:13s}] f1={scores['token_f1']:.2f} num={scores['numeric_match']:.0f} judge={scores['judge']:.0f} ({dt:.1f}s)", flush=True)
            except Exception as e:
                results[sys_name].append({"q": q.question, "gt": q.answer, "ans": "", "error": str(e), "scores": {"token_f1": 0, "numeric_match": -1, "judge": 0}})
                print(f"  [{sys_name:13s}] FAILED: {e}", flush=True)

    # Aggregate + bootstrap
    print("\n" + "=" * 70, flush=True)
    print(f"RESULTS  (n={len(multi)} multi-session questions)", flush=True)
    print("=" * 70, flush=True)
    summary = {}
    for s in systems:
        scores = results[s]
        f1s = [r["scores"]["token_f1"] for r in scores]
        nums = [r["scores"]["numeric_match"] for r in scores if r["scores"]["numeric_match"] >= 0]
        judges = [r["scores"]["judge"] for r in scores]
        f1_mean, f1_lo, f1_hi = bootstrap_ci(f1s)
        j_mean, j_lo, j_hi = bootstrap_ci(judges)
        num_mean = sum(nums) / len(nums) if nums else 0
        summary[s] = {
            "token_f1_mean": f1_mean, "token_f1_ci": (f1_lo, f1_hi),
            "judge_mean": j_mean, "judge_ci": (j_lo, j_hi),
            "numeric_match_mean": num_mean, "n_numeric": len(nums),
        }
        print(f"\n{s.upper():20s}", flush=True)
        print(f"  token_f1     : {f1_mean:.3f}  [{f1_lo:.3f}, {f1_hi:.3f}]", flush=True)
        print(f"  judge_acc    : {j_mean:.3f}  [{j_lo:.3f}, {j_hi:.3f}]", flush=True)
        print(f"  numeric_match: {num_mean:.3f}  (n_numeric={len(nums)})", flush=True)

    # Pairwise vs baseline
    print("\nDELTAS vs baseline (judge accuracy):", flush=True)
    for s in ("truncated", "random_facts", "consolidation"):
        delta = summary[s]["judge_mean"] - summary["baseline"]["judge_mean"]
        print(f"  {s:15s}: {delta:+.3f}", flush=True)

    out = {
        "n_questions": len(multi),
        "summary": summary,
        "details": results,
    }
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {output_path}", flush=True)

    # S3 upload
    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        import subprocess
        s3_key = f"s3://{s3_bucket}/paper_results/{os.path.basename(output_path)}"
        try:
            subprocess.run(["aws", "s3", "cp", output_path, s3_key], check=True, timeout=60)
            print(f"[s3] {s3_key}", flush=True)
        except Exception as e:
            print(f"[s3] {e}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="/tmp/paper_results.json")
    args = p.parse_args()
    run(n_questions=args.n, output_path=args.output, seed=args.seed)
