"""LoCoMo cross-dataset validation of the multihop retrieval finding.

Same 7 conditions as run_multihop_retrieval.py, applied to LoCoMo multi_hop
questions (the hardest category, requiring reasoning across sessions).

Main purpose: show the query-decomposition retrieval benefit generalizes
beyond LongMemEval-S to a second independent benchmark.

LoCoMo (Maharana et al., ACL 2024): 50 long-term conversations,
~9K QA pairs spanning single_hop, multi_hop, temporal, open_domain, adversarial.
We use multi_hop only (n≈200-400 questions).
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, "src")

# Reuse all condition logic from run_multihop_retrieval
from run_multihop_retrieval import (
    CONDITIONS, run_example, bootstrap_ci, mcnemar_test, _bedrock,
)
from context_bench.datasets.memory.locomo import locomo
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=None, help="max dialogues (default: all 50)")
    p.add_argument("--qa-types", default="multi_hop", help="comma-sep question types to include")
    p.add_argument("--conditions", default="all")
    p.add_argument("--shard", type=int, default=0, help="shard index (0-based)")
    p.add_argument("--n-shards", type=int, default=1, help="total number of shards")
    p.add_argument("--output", default="/tmp/locomo_multihop.json")
    args = p.parse_args()

    qa_types = args.qa_types.split(",")
    examples = locomo(n=args.n, qa_types=qa_types)
    # Flatten to (ex, q) pairs
    pairs = [(ex, q) for ex in examples for q in ex.queries]
    # Shard
    if args.n_shards > 1:
        pairs = pairs[args.shard::args.n_shards]
    print(f"[setup] {len(pairs)} {args.qa_types} questions (shard {args.shard}/{args.n_shards})", flush=True)

    conditions = CONDITIONS if args.conditions == "all" else args.conditions.split(",")
    client = _bedrock()
    all_results = []

    for i, (ex, q) in enumerate(pairs):
        print(f"\n--- Q{i+1}/{len(pairs)} id={ex.id} ---", flush=True)
        cond_results = run_example(client, ex, q, conditions)
        for cond, r in cond_results.items():
            print(f"  [{cond:16s}] judge={r['judge']} ans={r['ans'][:50]!r}", flush=True)
        all_results.append({
            "qid": ex.id, "question": q.question, "gold": q.answer,
            "query_type": q.query_type, "conditions": cond_results,
        })

    print("\n" + "=" * 80, flush=True)
    print(f"LOCOMO MULTI-HOP RESULTS (n={len(all_results)})", flush=True)
    print("=" * 80, flush=True)
    summary = {}
    all_judges = {c: [r["conditions"].get(c, {}).get("judge", 0) for r in all_results] for c in conditions}
    base = "BM25_single"
    for cond in conditions:
        scores = all_judges[cond]
        mean, lo, hi = bootstrap_ci(scores)
        p_val = mcnemar_test(all_judges[cond], all_judges[base]) if cond != base else None
        summary[cond] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(scores), "mcnemar_vs_bm25": p_val}
        sig = f"  p={p_val:.3f}{'*' if p_val and p_val < 0.05 else ''}" if p_val is not None else ""
        print(f"  {cond:18s} {mean:.3f} [{lo:.3f},{hi:.3f}] (n={len(scores)}){sig}", flush=True)

    out = {"n": len(all_results), "qa_types": qa_types, "summary": summary, "details": all_results}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {args.output}", flush=True)

    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        import subprocess
        s3_key = f"s3://{s3_bucket}/locomo_multihop/{os.path.basename(args.output)}"
        try:
            subprocess.run(["aws", "s3", "cp", args.output, s3_key], check=True, timeout=120)
            print(f"[s3] {s3_key}", flush=True)
        except Exception as e:
            print(f"[s3] {e}", flush=True)


if __name__ == "__main__":
    main()
