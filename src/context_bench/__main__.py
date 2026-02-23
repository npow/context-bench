"""CLI entry point for context-bench.

Usage:
    context-bench --proxy http://localhost:7878 --dataset hotpotqa -n 50
    python -m context_bench --proxy http://localhost:7878 --dataset hotpotqa -n 50
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Any
from urllib.parse import urlparse

DATASET_LOADERS: dict[str, tuple[str, str]] = {
    "hotpotqa": ("context_bench.datasets.huggingface", "hotpotqa"),
    "gsm8k": ("context_bench.datasets.huggingface", "gsm8k"),
    "bfcl": ("context_bench.datasets.huggingface", "bfcl_simple"),
    "apigen": ("context_bench.datasets.agent_traces", "apigen_mt"),
    "swebench": ("context_bench.datasets.agent_traces", "swebench"),
    "swebench-verified": ("context_bench.datasets.agent_traces", "swebench_verified"),
    "swebench-lite": ("context_bench.datasets.agent_traces", "swebench_lite"),
    # QA datasets
    "natural-questions": ("context_bench.datasets.qa", "natural_questions"),
    "musique": ("context_bench.datasets.qa", "musique"),
    "narrativeqa": ("context_bench.datasets.qa", "narrativeqa"),
    "triviaqa": ("context_bench.datasets.qa", "triviaqa"),
    "frames": ("context_bench.datasets.qa", "frames"),
    "quality": ("context_bench.datasets.qa", "quality"),
    # Long-context benchmarks
    "longbench": ("context_bench.datasets.longcontext", "longbench"),
    "longbench-v2": ("context_bench.datasets.longcontext", "longbench_v2"),
    "infinitebench": ("context_bench.datasets.longcontext", "infinitebench"),
    "nolima": ("context_bench.datasets.longcontext", "nolima"),
    "bbh": ("context_bench.datasets.longcontext", "bbh"),
    "meetingbank": ("context_bench.datasets.longcontext", "meetingbank"),
    "govreport": ("context_bench.datasets.longcontext", "govreport"),
}


CONFIGURABLE_DATASETS: set[str] = {
    "longbench", "infinitebench", "bbh",
}


def _load_dataset(name: str, max_examples: int | None) -> list[dict[str, Any]]:
    """Load a dataset by name or file path.

    If *name* matches a known dataset key, the corresponding loader is
    lazy-imported and called.  If it ends with ``.jsonl``, it's treated
    as a local file path.

    Supports ``name:config`` syntax for multi-config datasets
    (e.g. ``longbench:qasper``, ``bbh:causal_judgement``).
    """
    if name.endswith(".jsonl"):
        from context_bench.datasets.local import load_jsonl

        return load_jsonl(name, n=max_examples)

    # Parse optional :config suffix
    config = None
    if ":" in name:
        name, config = name.split(":", 1)

    if name not in DATASET_LOADERS:
        available = ", ".join(sorted(DATASET_LOADERS))
        raise SystemExit(
            f"Unknown dataset {name!r}. Available: {available}\n"
            "Or pass a path ending in .jsonl."
        )

    module_path, func_name = DATASET_LOADERS[name]
    mod = importlib.import_module(module_path)
    loader = getattr(mod, func_name)

    kwargs: dict[str, Any] = {"n": max_examples}
    if config is not None:
        if name not in CONFIGURABLE_DATASETS:
            raise SystemExit(
                f"Dataset {name!r} does not accept a :config suffix. "
                f"Configurable datasets: {', '.join(sorted(CONFIGURABLE_DATASETS))}"
            )
        kwargs["config"] = config

    return loader(**kwargs)


def _derive_name(url: str) -> str:
    """Derive a short display name from a proxy URL."""
    parsed = urlparse(url)
    host = parsed.hostname or parsed.netloc or url
    return host


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="context-bench",
        description="Benchmark any system that transforms LLM context.",
        epilog=(
            "Examples:\n"
            "  context-bench --proxy http://localhost:7878 --dataset hotpotqa -n 50\n"
            "  context-bench --proxy http://localhost:7878 --proxy http://localhost:8787 "
            "--name kompact --name headroom --dataset hotpotqa --dataset gsm8k\n"
            "  context-bench --proxy http://localhost:7878 --dataset longbench:qasper -n 20\n"
            "  context-bench --proxy http://localhost:7878 --dataset ./my_data.jsonl --output json\n"
            "\n"
            "Multi-config datasets (longbench, infinitebench, bbh) accept a :config\n"
            "suffix, e.g. --dataset bbh:causal_judgement"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--proxy",
        action="append",
        required=True,
        metavar="URL",
        help="OpenAI-compatible proxy URL (repeatable).",
    )
    parser.add_argument(
        "--name",
        action="append",
        default=None,
        metavar="NAME",
        help="Display name for the corresponding --proxy (repeatable). "
        "Auto-derived from URL hostname if omitted.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="NAME_OR_PATH",
        help=(
            f"Dataset to benchmark against (repeatable). "
            f"Known: {', '.join(sorted(DATASET_LOADERS))}. "
            f"Or a path to a .jsonl file."
        ),
    )
    parser.add_argument(
        "--model",
        default="gpt-4",
        help="Model name passed to the proxy (default: gpt-4).",
    )
    parser.add_argument(
        "-n",
        "--max-examples",
        type=int,
        default=None,
        help="Max examples per dataset (default: all).",
    )
    parser.add_argument(
        "--output",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--score-field",
        default="f1",
        help="Score field from AnswerQuality to use as primary score (default: f1).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Pass/fail threshold for PassRate and CostOfPass (default: 0.7).",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # --- Build proxy systems ---
    from context_bench.systems.openai_proxy import OpenAIProxy

    names: list[str] = args.name or []
    systems = []
    for i, url in enumerate(args.proxy):
        name = names[i] if i < len(names) else _derive_name(url)
        systems.append(OpenAIProxy(base_url=url, model=args.model, name=name))

    # --- Load and concatenate datasets ---
    all_examples: list[dict[str, Any]] = []
    for ds_spec in args.dataset:
        examples = _load_dataset(ds_spec, max_examples=args.max_examples)
        # Tag each example with its dataset source (including :config if given)
        for ex in examples:
            ex.setdefault("dataset", ds_spec)
        all_examples.extend(examples)

    if not all_examples:
        raise SystemExit("No examples loaded. Check your --dataset arguments.")

    # --- Set up evaluators and metrics ---
    from context_bench.evaluators.answer_quality import AnswerQuality
    from context_bench.evaluators.rouge import SummarizationQuality
    from context_bench.metrics import CompressionRatio, CostOfPass, MeanScore, PassRate
    from context_bench.runner import evaluate

    # Auto-add ROUGE-L evaluator when summarization datasets are present
    summarization_datasets = {"meetingbank", "govreport"}
    dataset_names = {spec.split(":")[0] for spec in args.dataset}
    evaluators: list[Any] = [AnswerQuality()]
    if dataset_names & summarization_datasets:
        evaluators.append(SummarizationQuality())
    metrics = [
        MeanScore(score_field=args.score_field),
        PassRate(threshold=args.threshold, score_field=args.score_field),
        CompressionRatio(),
        CostOfPass(threshold=args.threshold, score_field=args.score_field),
    ]

    # --- Run evaluation ---
    result = evaluate(
        systems=systems,
        dataset=all_examples,
        evaluators=evaluators,
        metrics=metrics,
        progress=True,
    )

    # --- Output ---
    if args.output == "json":
        from context_bench.reporters.json_out import to_json

        print(to_json(result))
    else:
        from context_bench.reporters.markdown import to_markdown

        print(to_markdown(result))


if __name__ == "__main__":
    main()
