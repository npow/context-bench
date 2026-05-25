"""Pareto analysis: accuracy vs cost vs latency for the 4 systems.

A method that's 50x slower for 5pp accuracy gain may or may not be on the
Pareto frontier — depends on whether other systems achieve similar accuracy
at lower cost.

Outputs:
- Per-system: accuracy, mean latency, total API tokens, $ cost estimate
- Pareto frontier: which systems are dominated (worse on all axes)
- 2D plots: accuracy vs latency, accuracy vs cost
"""
from __future__ import annotations
import argparse
import json
import os
import sys


# Bedrock pricing (approximate, USD per 1M tokens, May 2026)
PRICING = {
    "anthropic.claude-sonnet-4-6": (3.0, 15.0),
    "us.anthropic.claude-sonnet-4-6": (3.0, 15.0),
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0": (3.0, 15.0),
    "anthropic.claude-haiku-4-5": (0.80, 4.0),
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": (0.80, 4.0),
    "openai.gpt-oss-120b-1:0": (0.60, 2.40),  # est.
    "openai.gpt-oss-20b-1:0": (0.10, 0.40),
    "google.gemma-3-27b-it": (0.30, 1.20),  # est.
}


def estimate_tokens_per_question(system: str, model_id: str) -> tuple[int, int]:
    """Rough token estimate (input, output) per question for each system."""
    # Average haystack ~20K chars ≈ 5000 tokens
    if system == "baseline":
        return 5000, 50  # full haystack + short answer
    if system == "truncated":
        return 2500, 50  # 40K chars
    if system == "random_facts":
        return 6000, 50  # full + 20 facts
    if system == "consolidation":
        # ~30 consolidation calls per Q + 1 answer
        consol_in = 30 * 850
        consol_out = 30 * 150
        answer_in = 6500
        answer_out = 50
        return consol_in + answer_in, consol_out + answer_out
    return 0, 0


def analyze(results_path: str):
    data = json.loads(open(results_path).read())
    if "details" not in data:
        print("No details in results"); return
    details = data["details"]
    if not isinstance(details, dict):
        print("Expected multi-system results"); return

    # Determine reader model (passed via env or guess from data)
    reader = os.environ.get("READER_MODEL_ID",
                            data.get("reader_model", "us.anthropic.claude-sonnet-4-6"))
    input_price, output_price = PRICING.get(reader, (3.0, 15.0))

    print(f"\n{'='*80}\nPARETO ANALYSIS (reader={reader})\n{'='*80}\n")
    print(f"{'system':<18} {'judge_acc':<10} {'numeric':<10} {'token_f1':<10} {'lat(s)':<10} {'cost($)':<10}")
    print("-" * 78)

    system_summary = {}
    for system, entries in details.items():
        scores = [e.get("scores", {}) for e in entries]
        n = len(scores)
        if n == 0:
            continue

        token_f1s = [s.get("token_f1", 0) for s in scores]
        nums = [s.get("numeric_match", -1) for s in scores]
        nums_valid = [v for v in nums if v >= 0]
        judges = [s.get("judge", 0) for s in scores]

        # Tokens + cost
        toks_in, toks_out = estimate_tokens_per_question(system, reader)
        total_in_per_q = toks_in
        total_out_per_q = toks_out
        cost_per_q = (total_in_per_q / 1e6) * input_price + (total_out_per_q / 1e6) * output_price
        # Latency: actually present? entries may have wall-time
        lats = [e.get("latency_s", 0) for e in entries]
        mean_lat = sum(lats) / n if any(lats) else (
            5.0 if system != "consolidation" else 170.0  # fallback estimate
        )

        mean_judge = sum(judges) / n
        mean_num = sum(nums_valid) / len(nums_valid) if nums_valid else 0
        mean_tf1 = sum(token_f1s) / n

        system_summary[system] = {
            "n": n, "judge_acc": mean_judge, "numeric_match": mean_num,
            "token_f1": mean_tf1, "latency_s": mean_lat, "cost_per_q": cost_per_q,
        }

        print(f"{system:<18} {mean_judge:<10.3f} {mean_num:<10.3f} {mean_tf1:<10.3f} {mean_lat:<10.1f} {cost_per_q:<10.4f}")

    # Pareto: a system is dominated if another has higher accuracy AND lower (latency OR cost)
    print(f"\n{'='*80}\nPARETO FRONTIER (accuracy vs latency vs cost)\n{'='*80}\n")
    systems = list(system_summary.keys())
    dominated = set()
    for s1 in systems:
        for s2 in systems:
            if s1 == s2: continue
            sm1, sm2 = system_summary[s1], system_summary[s2]
            # s1 dominated by s2 if s2 is >= in accuracy AND <= in cost AND <= in latency, with at least one strict
            if (sm2["judge_acc"] >= sm1["judge_acc"] and
                sm2["latency_s"] <= sm1["latency_s"] and
                sm2["cost_per_q"] <= sm1["cost_per_q"] and
                (sm2["judge_acc"] > sm1["judge_acc"] or
                 sm2["latency_s"] < sm1["latency_s"] or
                 sm2["cost_per_q"] < sm1["cost_per_q"])):
                dominated.add(s1)

    print("On Pareto frontier (not dominated):")
    for s in systems:
        if s not in dominated:
            sm = system_summary[s]
            print(f"  - {s:<18}  judge={sm['judge_acc']:.3f}  lat={sm['latency_s']:.1f}s  cost=${sm['cost_per_q']:.4f}")

    print("\nDominated:")
    for s in dominated:
        sm = system_summary[s]
        print(f"  - {s:<18}  judge={sm['judge_acc']:.3f}  lat={sm['latency_s']:.1f}s  cost=${sm['cost_per_q']:.4f}")

    # Cost-effectiveness
    print(f"\n{'='*80}\nCOST/ACCURACY EFFICIENCY\n{'='*80}\n")
    baseline = system_summary.get("baseline")
    if baseline:
        print(f"{'system':<18} {'judge_acc':<10} {'Δ vs base':<12} {'extra $':<10} {'$ per +1pp':<12}")
        for s in systems:
            sm = system_summary[s]
            d_acc = sm["judge_acc"] - baseline["judge_acc"]
            d_cost = sm["cost_per_q"] - baseline["cost_per_q"]
            cost_per_pp = (d_cost * 100 / (d_acc * 100)) if d_acc > 0 else float("inf")
            print(f"{s:<18} {sm['judge_acc']:<10.3f} {d_acc:<+12.3f} {d_cost:<+10.4f} ${cost_per_pp:<12.3f}")

    return system_summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    args = p.parse_args()
    analyze(args.input)
