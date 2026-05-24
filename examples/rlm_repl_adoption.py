#!/usr/bin/env python3
"""Measure pretrained-model write/consolidate adoption on a small LoCoMo sample.

This is the Layer 1 experiment from the memory-first architecture research plan:
  "Test whether a pretrained model uses memory_write/consolidate without RL
   (probably not reliably)."

Run:
  python examples/rlm_repl_adoption.py \
    --relay http://localhost:18082 \
    --model claude-haiku-4-5-20251001 \
    --n 3

Prints adoption rates at the end:
  - write_adoption_rate: fraction of queries where model called memory_write()
  - consolidate_adoption_rate: fraction where model called consolidate()
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from context_bench.datasets.memory.locomo import locomo
from context_bench.systems.rlm_repl import RLMSystemRepl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--relay", default="http://localhost:18082")
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
    p.add_argument("--n", type=int, default=2, help="Number of conversations")
    p.add_argument("--queries-per-conv", type=int, default=5)
    args = p.parse_args()

    system = RLMSystemRepl(
        base_url=args.relay,
        model=args.model,
        max_iterations=4,
    )

    examples = locomo(n=args.n, split="validation")
    total_q = 0

    for ex in examples:
        turns = ex["turns"]
        qa_pairs = ex["qa_pairs"][: args.queries_per_conv]

        print(f"\n[Conv {ex['id']}] ingesting {len(turns)} turns, {len(qa_pairs)} queries")
        system.ingest(turns)

        for qa in qa_pairs:
            q = qa["question"]
            result = system.query(q)
            total_q += 1
            w = result.details.get("writes", 0)
            c = result.details.get("consolidations", 0)
            print(
                f"  Q{total_q}: writes={w} consolidate={c} "
                f"iters={result.details.get('iterations', '?')} | {q[:60]}"
            )

    stats = system.usage_stats()
    print("\n" + "=" * 60)
    print("ADOPTION STATS (pretrained baseline — no RL training)")
    print("=" * 60)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.1%}")
        else:
            print(f"  {k}: {v}")
    print()
    print("Research plan hypothesis: write_adoption_rate ≈ 0% without RL.")
    print("Run with --n 10 for a more reliable estimate.")


if __name__ == "__main__":
    main()
