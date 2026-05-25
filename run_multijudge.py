"""Multi-judge rescoring of baselines-n133 results.

Loads the saved baselines S3 results and rescores each (question, answer, gold)
tuple with a second judge (GPT-OSS-120B) to address reviewer concern about
judge-reader confound when Sonnet 4.6 judges its own answers.

Reports:
- Per-condition accuracy under Sonnet judge (original)
- Per-condition accuracy under GPT-OSS-120B judge (second)
- Correlation between judges
- Cohen's kappa
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import boto3
import random

sys.path.insert(0, "src")


def _bedrock():
    return boto3.client("bedrock-runtime", region_name="us-east-1")


def _claude(client, prompt, model_id, max_tokens=20):
    if model_id.startswith("openai.") or "gpt" in model_id:
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_tokens,
            "temperature": 0.0,
        }
    else:
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
    resp = json.loads(r["body"].read())
    if "content" in resp:
        return resp["content"][0]["text"].strip()
    return resp["choices"][0]["message"]["content"].strip()


def judge(client, pred, gold, question, model_id):
    if not pred.strip():
        return 0
    prompt = (
        "Judge if the PREDICTION correctly answers the QUESTION given GOLD. Reply CORRECT or WRONG only.\n\n"
        f"QUESTION: {question}\nGOLD: {gold}\nPREDICTION: {pred}\n\nReply:"
    )
    v = _claude(client, prompt, model_id, max_tokens=20).upper()
    if "WRONG" in v or "INCORRECT" in v or "NOT CORRECT" in v:
        return 0
    return 1 if "CORRECT" in v else 0


def cohen_kappa(a, b):
    n = len(a)
    if n == 0: return 0.0
    p_agree = sum(1 for x, y in zip(a, b) if x == y) / n
    p_a1 = sum(a) / n; p_b1 = sum(b) / n
    p_expected = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)
    return (p_agree - p_expected) / max(1e-9, 1 - p_expected)


def bootstrap_ci(scores, n=1000):
    if not scores: return 0.0, 0.0, 0.0
    rng = random.Random(42)
    n_s = len(scores)
    means = sorted(sum(rng.choice(scores) for _ in range(n_s)) / n_s for _ in range(n))
    return sum(scores) / n_s, means[int(n * 0.025)], means[int(n * 0.975)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-s3", default="s3://netflix.pi.prod/users/npow/rlm_grpo/baselines_n133_results.json",
                   help="S3 path to baselines results JSON; or use --sample-q for inline sampling")
    p.add_argument("--n-sample", type=int, default=50, help="max Q to rescore per condition (cost control)")
    p.add_argument("--second-judge", default="openai.gpt-oss-120b-1:0")
    p.add_argument("--output", default="/tmp/multijudge_results.json")
    args = p.parse_args()

    client = _bedrock()
    first_judge_id = "us.anthropic.claude-sonnet-4-6"
    second_judge_id = args.second_judge

    # Try to load baselines results from S3
    print(f"[setup] loading baselines results...", flush=True)
    try:
        import subprocess
        result = subprocess.run(["aws", "s3", "cp", args.input_s3, "/tmp/baselines_input.json"],
                                capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"S3 download failed: {result.stderr[:200]}")
        with open("/tmp/baselines_input.json") as f:
            data = json.load(f)
        print(f"[setup] loaded from S3: {len(data.get('details', {}))} items", flush=True)
    except Exception as e:
        print(f"[setup] S3 load failed: {e}. Generating inline sample from LongMemEval...", flush=True)
        # Fallback: generate fresh Q&A pairs for rescoring
        from context_bench.datasets.memory.longmemeval import longmemeval
        from run_baselines_ablations import facts_only, long_context_full
        examples = longmemeval(n=200)
        multi = [ex for ex in examples if any("multi-session" in q.query_type for q in ex.queries)][:args.n_sample]

        conditions_sample = {}
        for i, ex in enumerate(multi):
            q = ex.queries[0]
            for cond_name, fn in [("facts_only", facts_only), ("long_context_full", long_context_full)]:
                try:
                    ans = fn(client, ex.items, q.question)
                except Exception:
                    ans = ""
                s1 = judge(client, ans, q.answer, q.question, first_judge_id)
                conditions_sample.setdefault(cond_name, []).append({
                    "q": q.question, "gold": q.answer, "ans": ans, "judge_sonnet": s1
                })
            if i % 5 == 0:
                print(f"  inline sample {i+1}/{len(multi)}", flush=True)
        data = {"details": conditions_sample}

    details = data.get("details", {})
    summary = {}

    for cond_name, records in details.items():
        if not records: continue
        # Sample for cost control
        sample = records[:args.n_sample] if len(records) > args.n_sample else records
        rng = random.Random(42)
        if len(records) > args.n_sample:
            sample = rng.sample(records, args.n_sample)

        sonnet_scores = []
        gpt_scores = []
        for r in sample:
            q_text = r.get("q", "")
            gold = r.get("gt", r.get("gold", ""))
            ans = r.get("ans", "")
            # Sonnet judge (may already be in data)
            s1 = r.get("judge", r.get("judge_sonnet"))
            if s1 is None:
                s1 = judge(client, ans, gold, q_text, first_judge_id)
            sonnet_scores.append(s1)
            # GPT-OSS judge
            s2 = judge(client, ans, gold, q_text, second_judge_id)
            gpt_scores.append(s2)

        m1, l1, h1 = bootstrap_ci(sonnet_scores)
        m2, l2, h2 = bootstrap_ci(gpt_scores)
        kappa = cohen_kappa(sonnet_scores, gpt_scores)
        summary[cond_name] = {
            "sonnet_judge": {"mean": m1, "ci_lo": l1, "ci_hi": h1, "n": len(sonnet_scores)},
            "gpt_judge": {"mean": m2, "ci_lo": l2, "ci_hi": h2, "n": len(gpt_scores)},
            "cohen_kappa": kappa,
        }
        print(
            f"  {cond_name:22s} sonnet={m1:.3f}[{l1:.3f},{h1:.3f}]  "
            f"gpt={m2:.3f}[{l2:.3f},{h2:.3f}]  kappa={kappa:.3f}",
            flush=True,
        )

    out = {"second_judge": second_judge_id, "summary": summary}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {args.output}", flush=True)

    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        import subprocess
        s3_key = f"s3://{s3_bucket}/multijudge/{os.path.basename(args.output)}"
        try:
            subprocess.run(["aws", "s3", "cp", args.output, s3_key], check=True, timeout=120)
            print(f"[s3] {s3_key}", flush=True)
        except Exception as e:
            print(f"[s3] {e}", flush=True)


if __name__ == "__main__":
    main()
