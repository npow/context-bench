"""Multi-judge rescorer: Claude + GPT-OSS + Gemma vote on each prediction.

For academic credibility, judging with a DIFFERENT model family than the
generator avoids self-preference bias. We run THREE judges:

1. anthropic (us.anthropic.claude-sonnet-4-6): same family as generator
2. openai (openai.gpt-oss-120b-1:0): independent (OpenAI architecture)
3. google (google.gemma-3-27b-it): independent (Google architecture)

Final accuracy = majority vote across all three. Disagreement is logged
for failure-mode analysis.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time


def claude_judge(client, pred: str, gold: str, question: str) -> int:
    if not pred.strip(): return 0
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
    r = client.invoke_model(
        body=body,
        modelId="us.anthropic.claude-sonnet-4-6",
        accept="application/json", contentType="application/json",
    )
    text = json.loads(r["body"].read())["content"][0]["text"].strip().upper()
    # FIX: "INCORRECT" contains "CORRECT" — check negatives first
    if "WRONG" in text or "INCORRECT" in text or "NOT CORRECT" in text:
        return 0
    return 1 if "CORRECT" in text else 0


def openai_judge(client, pred: str, gold: str, question: str) -> int:
    if not pred.strip(): return 0
    prompt = (
        "Judge if the PREDICTION correctly answers the QUESTION given the GOLD. "
        "Output a JSON object: {\"verdict\": \"CORRECT\"} or {\"verdict\": \"WRONG\"}. "
        "No other text outside the JSON.\n\n"
        f"QUESTION: {question}\nGOLD: {gold}\nPREDICTION: {pred}"
    )
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 200,
    })
    r = client.invoke_model(body=body, modelId="openai.gpt-oss-120b-1:0")
    text = json.loads(r["body"].read())["choices"][0]["message"]["content"]
    # Strip reasoning tags BEFORE uppercasing
    text_raw = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL).strip()
    # JSON regex on original case (verdict key is lowercase)
    m = re.search(r'"verdict"\s*:\s*"(CORRECT|WRONG)"', text_raw, re.IGNORECASE)
    if m:
        return 1 if m.group(1).upper() == "CORRECT" else 0
    # FIX: "INCORRECT" contains "CORRECT" — check negatives first
    text = text_raw.upper()
    if "WRONG" in text or "INCORRECT" in text or "NOT CORRECT" in text:
        return 0
    return 1 if "CORRECT" in text else 0


def gemma_judge(client, pred: str, gold: str, question: str) -> int:
    if not pred.strip(): return 0
    prompt = (
        "Judge if the PREDICTION correctly answers the QUESTION given the GOLD answer. "
        "Reply with exactly one word: CORRECT or WRONG.\n\n"
        f"QUESTION: {question}\nGOLD: {gold}\nPREDICTION: {pred}\n\nVerdict:"
    )
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "temperature": 0.0,
    })
    r = client.invoke_model(body=body, modelId="google.gemma-3-27b-it")
    text = json.loads(r["body"].read())
    # Format varies; try common shapes
    if "choices" in text:
        out = text["choices"][0]["message"]["content"]
    elif "generation" in text:
        out = text["generation"]
    elif "outputs" in text:
        out = text["outputs"][0].get("text", "")
    else:
        out = json.dumps(text)
    out = out.strip().upper()
    # FIX: "INCORRECT" contains "CORRECT" — check negatives first
    if "WRONG" in out or "INCORRECT" in out or "NOT CORRECT" in out:
        return 0
    return 1 if "CORRECT" in out else 0


_JUDGES = {
    "claude": claude_judge,
    "openai": openai_judge,
    "gemma": gemma_judge,
}


def bootstrap_ci(scores: list[float], n_resamples: int = 1000, conf: float = 0.95) -> tuple[float, float, float]:
    import random
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


def rescore(results_path: str, output_path: str, judges: list[str]):
    import boto3
    client = boto3.client("bedrock-runtime", region_name="us-east-1")

    data = json.loads(open(results_path).read())
    # Auto-detect schema (multi-system from run_paper_experiment.py)
    if "details" in data and isinstance(data["details"], dict):
        details = data["details"]
        system_names = list(details.keys())
    else:
        # Single-system (older format)
        details = {"system": data.get("details", [])}
        system_names = ["system"]

    n = max(len(details[s]) for s in system_names)
    print(f"[rescore] {len(system_names)} systems, n={n}, judges={judges}", flush=True)

    for i in range(n):
        question = None
        gold = None
        for s in system_names:
            if i < len(details[s]):
                question = question or details[s][i].get("q", "")
                gold = gold or details[s][i].get("gt", "")
        if not question or not gold:
            continue

        for s in system_names:
            if i >= len(details[s]):
                continue
            entry = details[s][i]
            pred = entry.get("ans", "")
            scores = entry.get("scores", {})
            judge_results = {}
            for j_name in judges:
                t0 = time.perf_counter()
                try:
                    v = _JUDGES[j_name](client, pred, gold, question)
                    judge_results[j_name] = v
                except Exception as e:
                    judge_results[j_name] = -1
                    print(f"  Q{i+1}/{s}/{j_name} err: {str(e)[:60]}", flush=True)
            judge_results["majority"] = 1 if sum(v for v in judge_results.values() if v >= 0) > len(judges)/2 else 0
            scores["multi_judge"] = judge_results
            entry["scores"] = scores
        if (i+1) % 10 == 0:
            print(f"  rescored Q{i+1}/{n}", flush=True)

    # Aggregate per system, per judge
    print("\n" + "=" * 70, flush=True)
    print(f"MULTI-JUDGE RESULTS  (n={n})", flush=True)
    print("=" * 70, flush=True)
    summary = {}
    for s in system_names:
        summary[s] = {}
        for j_name in list(judges) + ["majority"]:
            scores = [e["scores"]["multi_judge"].get(j_name, 0) for e in details[s] if "multi_judge" in e.get("scores", {})]
            if not scores:
                continue
            mean, lo, hi = bootstrap_ci(scores)
            summary[s][j_name] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(scores)}

    for s in system_names:
        print(f"\n{s.upper()}", flush=True)
        for j in list(judges) + ["majority"]:
            if j in summary.get(s, {}):
                v = summary[s][j]
                print(f"  {j:10s}: {v['mean']:.3f}  [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]  (n={v['n']})", flush=True)

    if "baseline" in system_names:
        print("\nDELTAS vs BASELINE (majority vote):", flush=True)
        b_maj = summary["baseline"]["majority"]["mean"]
        for s in system_names:
            if s == "baseline":
                continue
            delta = summary[s]["majority"]["mean"] - b_maj
            print(f"  {s:15s}: {delta:+.3f}", flush=True)

    data["multi_judge_summary"] = summary
    data["details"] = details
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[saved] {output_path}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--judges", default="claude,openai,gemma",
                   help="Comma-separated: claude, openai, gemma")
    args = p.parse_args()
    output = args.output or args.input.replace(".json", "_multijudge.json")
    judges = [j.strip() for j in args.judges.split(",")]
    rescore(args.input, output, judges)
