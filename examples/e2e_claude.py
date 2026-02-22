"""End-to-end quality evaluation using Claude CLI.

Runs examples through `claude -p` with and without context compression,
then compares answer quality. This measures whether compression preserves
the information needed for correct LLM responses.

Usage:
    # Quick test (5 HotpotQA examples)
    python examples/e2e_claude.py

    # Full BFCL eval
    python examples/e2e_claude.py --dataset bfcl -n 50

    # Compare compressed vs uncompressed
    python examples/e2e_claude.py --dataset hotpotqa -n 20

Requires: claude CLI installed (https://claude.ai/claude-code)
"""

from __future__ import annotations

import argparse

from context_bench import ClaudeCLI, evaluate
from context_bench.evaluators import AnswerQuality
from context_bench.metrics import CostOfPass, MeanScore
from context_bench.reporters import to_markdown


class IdentitySystem:
    """Pass-through baseline — no compression."""

    @property
    def name(self) -> str:
        return "No Compression"

    def process(self, example: dict) -> dict:
        return dict(example)


class TruncateSystem:
    """Truncate context to 50%."""

    @property
    def name(self) -> str:
        return "Truncation (50%)"

    def process(self, example: dict) -> dict:
        context = example.get("context", "")
        return {**example, "context": context[: len(context) // 2]}


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end Claude CLI evaluation")
    parser.add_argument("--dataset", default="hotpotqa", help="Dataset to evaluate")
    parser.add_argument("-n", type=int, default=5, help="Number of examples")
    parser.add_argument("--model", default="haiku", help="Claude model")
    args = parser.parse_args()

    # Load dataset
    if args.dataset == "hotpotqa":
        from context_bench.datasets.huggingface import hotpotqa
        examples = hotpotqa(n=args.n)
    elif args.dataset == "bfcl":
        from context_bench.datasets.huggingface import bfcl_simple
        examples = bfcl_simple(n=args.n)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Define systems: compression happens BEFORE the LLM call.
    # Each "system" first compresses, then sends to Claude.
    baseline = ClaudeCLI(name="baseline", model=args.model)

    # For a compression system, you'd compose: compress then send to Claude.
    # Example with truncation:
    class CompressedClaude:
        def __init__(self, compressor, claude, name):
            self._compressor = compressor
            self._claude = claude
            self._name = name

        @property
        def name(self):
            return self._name

        def process(self, example):
            compressed = self._compressor.process(example)
            return self._claude.process(compressed)

    truncated = CompressedClaude(
        TruncateSystem(),
        ClaudeCLI(name="_inner", model=args.model),
        "Truncated + Claude",
    )

    result = evaluate(
        systems=[baseline, truncated],
        dataset=examples,
        evaluators=[AnswerQuality()],
        metrics=[
            MeanScore(score_field="f1"),
            MeanScore(score_field="contains"),
            CostOfPass(threshold=0.5, score_field="f1"),
        ],
        text_fields=["context"],
    )

    print(to_markdown(result))


if __name__ == "__main__":
    main()
