#!/usr/bin/env python3
"""Submit final results to Codex for conference paper quality signoff.

Usage:
    # After training completes:
    python codex_review.py \\
        --baseline /data/tmp/baseline_results.json \\
        --trained ~/rlm_grpo/final_eval.json \\
        --static /data/tmp/static_baseline_results.json

    # Check current state (before training):
    python codex_review.py --partial \\
        --baseline /data/tmp/baseline_results.json \\
        --static /data/tmp/static_baseline_results.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


CODEX_REVIEW_PROMPT = """
Review these experimental results for a research paper on memory-first architectures
for Recursive Language Models. The paper claims:

1. TWO-POLICY GAP: Pretrained LLMs naturally learn the access policy (reads)
   but NOT the management policy (writes). Empirical evidence required.

2. RL TRAINING CLOSES THE GAP: GRPO training teaches the management policy.
   The trained model should show:
   - Write adoption rate >> pretrained baseline
   - Within-session F1 (LoCoMo) >= static retrieval baseline (no regression)
   - Cross-session F1 (LongMemEval) > static retrieval baseline (improvement)

CLAIMS TO VERIFY:
1. Pretrained write_adoption_rate < 25% (gap confirmed)
2. Trained write_adoption_rate > 40% (gap closed by RL)
3. LoCoMo F1 delta (trained - static) > -10pp (no significant regression)
4. LongMemEval F1 delta (trained - static) > +5pp (cross-session improves)

Reply APPROVED if all four claims are supported by the data, or list which
claims fail and what threshold they fell short of.
"""


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def format_results(baseline: dict, trained: dict | None, static: dict | None) -> str:
    lines = ["=" * 60, "EXPERIMENTAL RESULTS", "=" * 60, ""]

    # Layer 1: adoption gap
    b_loc = baseline.get("locomo", {})
    lines.append("LAYER 1: Two-Policy Gap (pretrained baseline)")
    lines.append(f"  Model: {baseline.get('model', '?')}")
    lines.append(f"  LoCoMo mean F1: {b_loc.get('mean_f1', 0):.3f}")
    lines.append(f"  LoCoMo write adoption: {b_loc.get('write_adoption_rate', 0):.1%}")
    b_lme = baseline.get("longmemeval", {})
    lines.append(f"  LongMemEval mean F1: {b_lme.get('mean_f1', 0):.3f}")
    lines.append(f"  LongMemEval write adoption: {b_lme.get('write_adoption_rate', 0):.1%}")
    lines.append("")

    # Static oracle
    if static:
        s_loc = static.get("locomo", {})
        s_lme = static.get("longmemeval", {})
        lines.append("STATIC ORACLE (RLMSystem, no REPL)")
        lines.append(f"  Model: {static.get('model', '?')}")
        lines.append(f"  LoCoMo mean F1: {s_loc.get('mean_f1', 0):.3f}")
        lines.append(f"  LongMemEval mean F1: {s_lme.get('mean_f1', 0):.3f}")
        lines.append("")

    # Layer 2: trained
    if trained:
        t_loc = trained.get("locomo", {})
        t_lme = trained.get("longmemeval", {})
        lines.append("LAYER 2: After GRPO Training")
        lines.append(f"  Checkpoint: {trained.get('checkpoint', '?')}")
        lines.append(f"  LoCoMo mean F1: {t_loc.get('mean_f1', 0):.3f}")
        lines.append(f"  LoCoMo write adoption: {t_loc.get('write_adoption_rate', 0):.1%}")
        lines.append(f"  LongMemEval mean F1: {t_lme.get('mean_f1', 0):.3f}")
        lines.append(f"  LongMemEval write adoption: {t_lme.get('write_adoption_rate', 0):.1%}")
        lines.append("")

        # Deltas
        if static:
            s_loc = static.get("locomo", {})
            s_lme = static.get("longmemeval", {})
            loc_delta = t_loc.get("mean_f1", 0) - s_loc.get("mean_f1", 0)
            lme_delta = t_lme.get("mean_f1", 0) - s_lme.get("mean_f1", 0)
            lines.append("DELTAS (trained - static oracle):")
            lines.append(f"  LoCoMo F1 delta: {loc_delta:+.3f}")
            lines.append(f"  LongMemEval F1 delta: {lme_delta:+.3f}")

    lines += [
        "",
        "=" * 60,
        CODEX_REVIEW_PROMPT.strip(),
    ]
    return "\n".join(lines)


def submit_to_codex(content: str, output_file: str = "/tmp/codex_paper_review.txt") -> None:
    """Pipe content to codex exec and save the review."""
    print("[codex] Submitting for review...", flush=True)
    try:
        result = subprocess.run(
            ["codex", "exec",
             "Review the experimental results and reply APPROVED or list failed claims."],
            input=content,
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout + result.stderr
        Path(output_file).write_text(output)
        print(f"[codex] Review saved to {output_file}", flush=True)

        # Check for APPROVED
        if "APPROVED" in output.upper():
            print("[codex] ✅ APPROVED", flush=True)
            return True
        else:
            print("[codex] ❌ Not approved — see issues in review file", flush=True)
            # Print issues
            for line in output.split("\n"):
                if any(kw in line.lower() for kw in ["fail", "issue", "problem", "claim", "not"]):
                    print(f"  {line}", flush=True)
            return False
    except subprocess.TimeoutExpired:
        print("[codex] Timeout — check /tmp/codex_paper_review.txt manually", flush=True)
        return False
    except Exception as e:
        print(f"[codex] Error: {e}", flush=True)
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, help="REPL pretrained results JSON")
    p.add_argument("--trained", help="GRPO trained results JSON")
    p.add_argument("--static", help="Static oracle results JSON")
    p.add_argument("--partial", action="store_true",
                   help="Show partial results (before training completes)")
    p.add_argument("--output", default="/tmp/codex_paper_review.txt")
    args = p.parse_args()

    baseline = load_json(args.baseline)
    trained = load_json(args.trained) if args.trained else None
    static = load_json(args.static) if args.static else None

    content = format_results(baseline, trained, static)
    print(content)

    if args.partial:
        print("\n[partial] Training still running — showing intermediate state only.")
        return

    if not trained:
        print("\n[info] No trained results yet. Run after training completes.")
        return

    # Submit to Codex
    approved = submit_to_codex(content, args.output)
    if not approved:
        sys.exit(1)


if __name__ == "__main__":
    main()
