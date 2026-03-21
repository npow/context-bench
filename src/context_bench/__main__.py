"""CLI entry point for context-bench.

Usage:
    context-bench --proxy http://localhost:7878 --dataset hotpotqa -n 50
    python -m context_bench --proxy http://localhost:7878 --dataset hotpotqa -n 50
    context-bench memory --system naive --relay http://localhost:7878 --dataset locomo
"""

from __future__ import annotations

import argparse
import importlib
import os
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
    # Code generation
    "humaneval": ("context_bench.datasets.code", "humaneval"),
    "mbpp": ("context_bench.datasets.code", "mbpp"),
    # Summarization
    "multi-news": ("context_bench.datasets.summarization", "multi_news"),
    "dialogsum": ("context_bench.datasets.summarization", "dialogsum"),
    "qmsum": ("context_bench.datasets.summarization", "qmsum"),
    "summscreenfd": ("context_bench.datasets.summarization", "summscreenfd"),
    # NLI / fact verification
    "contract-nli": ("context_bench.datasets.nli", "contract_nli"),
    "scifact": ("context_bench.datasets.nli", "scifact"),
    # Scientific paper QA (full papers)
    "qasper": ("context_bench.datasets.qa", "qasper"),
    # Knowledge / multiple-choice
    "mmlu": ("context_bench.datasets.knowledge", "mmlu"),
    "arc-challenge": ("context_bench.datasets.knowledge", "arc_challenge"),
    "truthfulqa": ("context_bench.datasets.knowledge", "truthfulqa"),
    "gpqa": ("context_bench.datasets.knowledge", "gpqa"),
    # Reasoning
    "drop": ("context_bench.datasets.reasoning", "drop"),
    "math": ("context_bench.datasets.reasoning", "math_dataset"),
    # Instruction following
    "ifeval": ("context_bench.datasets.ifeval", "ifeval"),
    "alpaca-eval": ("context_bench.datasets.instruction", "alpaca_eval"),
    # Multi-turn
    "mt-bench": ("context_bench.datasets.multi_turn", "mt_bench"),
    # Additional knowledge / MC
    "hellaswag": ("context_bench.datasets.knowledge", "hellaswag"),
    "winogrande": ("context_bench.datasets.knowledge", "winogrande"),
    "mmlu-pro": ("context_bench.datasets.knowledge", "mmlu_pro"),
    # Multilingual reasoning
    "mgsm": ("context_bench.datasets.reasoning", "mgsm"),
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
    "longbench", "infinitebench", "bbh", "mmlu", "mgsm",
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


def build_proxy_parser(parser: argparse.ArgumentParser) -> None:
    """Add the original proxy-benchmark arguments to *parser*."""
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
        default="claude-haiku-4-5-20251001",
        help="Model name passed to the proxy (default: claude-haiku-4-5-20251001).",
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
        choices=["table", "json", "html"],
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
    parser.add_argument(
        "--judge-url",
        default=None,
        metavar="URL",
        help="OpenAI-compatible URL for LLM-as-judge evaluation (optional).",
    )
    parser.add_argument(
        "--judge-model",
        default="claude-haiku-4-5-20251001",
        help="Model name for the LLM judge (default: claude-haiku-4-5-20251001).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        metavar="N",
        help="Max concurrent threads per system (default: sequential).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        metavar="DIR",
        help="Directory for result caching. Enables resume on re-run.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="context-bench",
        description="Benchmark any system that transforms LLM context.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="subcommand")

    # --- memory subcommand ---
    memory_parser = subparsers.add_parser(
        "memory",
        help="Benchmark stateful memory systems on conversation datasets.",
        description="Evaluate memory systems (naive, mem0, zep) on LoCoMo or LongMemEval.",
        epilog=(
            "Examples:\n"
            "  context-bench memory --system naive --relay http://localhost:7878\n"
            "  context-bench memory --system naive --system mem0 \\\n"
            "    --relay http://localhost:7878 --dataset locomo -n 20\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    memory_parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        metavar="DATASET",
        help="Memory dataset to evaluate on: locomo or longmemeval (repeatable, default: locomo).",
    )
    memory_parser.add_argument(
        "--system",
        action="append",
        required=True,
        metavar="SYSTEM",
        dest="systems",
        help="Memory system to evaluate: naive, mem0, or zep (repeatable, at least one required).",
    )
    memory_parser.add_argument(
        "--relay",
        required=True,
        metavar="URL",
        help="OpenAI-compatible relay URL.",
    )
    memory_parser.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="Model name (default: claude-haiku-4-5-20251001).",
    )
    memory_parser.add_argument(
        "--api-key",
        default=None,
        metavar="KEY",
        help="Bearer token (default: OPENAI_API_KEY env var).",
    )
    memory_parser.add_argument(
        "-n",
        type=int,
        default=None,
        metavar="N",
        help="Max conversations to evaluate (default: all).",
    )
    memory_parser.add_argument(
        "--qa-types",
        default=None,
        metavar="TYPES",
        help="Comma-separated QA types to filter, e.g. temporal,multi_hop (default: all).",
    )
    memory_parser.add_argument(
        "--output",
        choices=["table", "json", "html"],
        default="table",
        help="Output format (default: table).",
    )
    memory_parser.add_argument(
        "--score-field",
        default="f1",
        help="Score field to report (default: f1).",
    )
    memory_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits (default: 42).",
    )

    # --- legacy (no subcommand) arguments attached to the root parser ---
    # We add them as a separate group so the original flat invocation still works.
    _legacy = parser.add_argument_group(
        "proxy benchmark (legacy, no subcommand)",
        description=(
            "Run the original proxy benchmark when no subcommand is given.\n"
            "Example: context-bench --proxy http://localhost:7878 --dataset hotpotqa"
        ),
    )
    _legacy.add_argument(
        "--proxy",
        action="append",
        default=None,
        metavar="URL",
        help="OpenAI-compatible proxy URL (repeatable).",
    )
    _legacy.add_argument(
        "--name",
        action="append",
        default=None,
        metavar="NAME",
        help="Display name for the corresponding --proxy. Auto-derived from URL if omitted.",
    )
    _legacy.add_argument(
        "--dataset",
        action="append",
        default=None,
        metavar="NAME_OR_PATH",
        help=(
            f"Dataset to benchmark against (repeatable). "
            f"Known: {', '.join(sorted(DATASET_LOADERS))}. "
            "Or a path to a .jsonl file."
        ),
    )
    _legacy.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="Model name passed to the proxy (default: claude-haiku-4-5-20251001).",
    )
    _legacy.add_argument(
        "-n",
        "--max-examples",
        type=int,
        default=None,
        help="Max examples per dataset (default: all).",
    )
    _legacy.add_argument(
        "--output",
        choices=["table", "json", "html"],
        default="table",
        help="Output format (default: table).",
    )
    _legacy.add_argument(
        "--score-field",
        default="f1",
        help="Score field from AnswerQuality to use as primary score (default: f1).",
    )
    _legacy.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Pass/fail threshold for PassRate and CostOfPass (default: 0.7).",
    )
    _legacy.add_argument(
        "--judge-url",
        default=None,
        metavar="URL",
        help="OpenAI-compatible URL for LLM-as-judge evaluation (optional).",
    )
    _legacy.add_argument(
        "--judge-model",
        default="claude-haiku-4-5-20251001",
        help="Model name for the LLM judge (default: claude-haiku-4-5-20251001).",
    )
    _legacy.add_argument(
        "--max-workers",
        type=int,
        default=None,
        metavar="N",
        help="Max concurrent threads per system (default: sequential).",
    )
    _legacy.add_argument(
        "--cache-dir",
        default=None,
        metavar="DIR",
        help="Directory for result caching. Enables resume on re-run.",
    )

    # --- dspy subcommand ---
    dspy_parser = subparsers.add_parser(
        "dspy",
        help="Benchmark DSPy optimizers across datasets.",
        description=(
            "Run a factorial sweep of DSPy optimizers across context-bench "
            "datasets, tracking compile cost, inference scores, and task "
            "features for optimizer selection meta-learning."
        ),
        epilog=(
            "Examples:\n"
            "  context-bench dspy --optimizer miprov2 --dataset hotpotqa --budget light\n"
            "  context-bench dspy --optimizer gepa --optimizer simba \\\n"
            "    --dataset gsm8k --dataset mmlu --budget medium \\\n"
            "    --seed 42 --seed 123 --model claude-haiku-4-5-20251001\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    dspy_parser.add_argument(
        "--optimizer",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "DSPy optimizer to benchmark (repeatable). "
            "Available: LabeledFewShot, BootstrapFewShot, "
            "BootstrapFewShotWithRandomSearch, COPRO, MIPROv2, SIMBA, GEPA. "
            "Default: all."
        ),
    )
    dspy_parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        metavar="NAME",
        help="Dataset to benchmark on (repeatable). Default: all compatible.",
    )
    dspy_parser.add_argument(
        "--budget",
        action="append",
        default=None,
        choices=["light", "medium", "heavy"],
        help="Budget tier (repeatable). Default: light.",
    )
    dspy_parser.add_argument(
        "--seed",
        action="append",
        type=int,
        default=None,
        help="Random seed for optimizer (repeatable). Default: [42].",
    )
    dspy_parser.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="LM model name for DSPy (default: claude-haiku-4-5-20251001).",
    )
    dspy_parser.add_argument(
        "--prompt-model",
        default=None,
        help="Separate model for instruction generation (MIPROv2/COPRO/SIMBA).",
    )
    dspy_parser.add_argument(
        "--cache-dir",
        default=".dspy_cache",
        metavar="DIR",
        help="Directory for compiled program cache (default: .dspy_cache).",
    )
    dspy_parser.add_argument(
        "--max-train",
        type=int,
        default=500,
        help="Max training examples per dataset (default: 500).",
    )
    dspy_parser.add_argument(
        "--max-val",
        type=int,
        default=200,
        help="Max validation examples per dataset (default: 200).",
    )
    dspy_parser.add_argument(
        "-n",
        "--max-examples",
        type=int,
        default=None,
        help="Max test examples per dataset (default: all).",
    )
    dspy_parser.add_argument(
        "--cost-cap",
        type=float,
        default=None,
        help="Abort if estimated cost exceeds this amount in dollars.",
    )
    dspy_parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile optimizers but skip evaluation.",
    )
    dspy_parser.add_argument(
        "--include-ablations",
        action="store_true",
        help="Include GEPA ablation runs (without feedback metric).",
    )
    dspy_parser.add_argument(
        "--output",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table).",
    )

    return parser


# ---------------------------------------------------------------------------
# memory subcommand helpers
# ---------------------------------------------------------------------------

_MEMORY_DATASETS = {"locomo", "longmemeval"}


def _load_memory_dataset(
    name: str,
    n: int | None,
    qa_types: list[str] | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Load a memory benchmark dataset by name."""
    if name not in _MEMORY_DATASETS:
        raise SystemExit(
            f"Unknown memory dataset {name!r}. Available: {', '.join(sorted(_MEMORY_DATASETS))}"
        )
    from context_bench.datasets.memory import locomo, longmemeval

    if name == "locomo":
        return locomo(n=n, qa_types=qa_types)
    else:
        return longmemeval(n=n, question_types=qa_types)


def _build_memory_system(system_name: str, relay: str, model: str, api_key: str | None) -> Any:
    """Instantiate a memory system by name."""
    if system_name == "naive":
        from context_bench.systems.naive_memory import NaiveMemorySystem
        return NaiveMemorySystem(base_url=relay, model=model, api_key=api_key)
    elif system_name == "mem0":
        try:
            from context_bench.systems.mem0_system import Mem0System
            return Mem0System(relay_url=relay, model=model, api_key=api_key)
        except ImportError as exc:
            raise SystemExit(
                "mem0 system requires the mem0ai package. "
                "Install it with: pip install context-bench[mem0]\n"
                f"(original error: {exc})"
            ) from exc
    elif system_name == "zep":
        try:
            from context_bench.systems.zep_system import ZepSystem
            return ZepSystem(relay_url=relay, model=model, api_key=api_key)
        except ImportError as exc:
            raise SystemExit(
                "zep system requires graphiti-core. "
                "Install it with: pip install context-bench[zep]\n"
                f"(original error: {exc})"
            ) from exc
    else:
        raise SystemExit(
            f"Unknown memory system {system_name!r}. Available: naive, mem0, zep"
        )


def _render_per_qa_type_table(result: Any, score_field: str) -> str:
    """Render a per-QA-type breakdown table in markdown."""
    systems = list(result.summary.keys())
    if not systems:
        return ""

    # Collect per-QA-type keys (those matching {score_field}_{type})
    prefix = f"{score_field}_"
    qa_type_keys: list[str] = []
    for sys_metrics in result.summary.values():
        for key in sys_metrics:
            if key.startswith(prefix) and key != f"{score_field}_mean" and key not in qa_type_keys:
                qa_type_keys.append(key)
    qa_type_keys.sort()

    if not qa_type_keys:
        return ""

    lines: list[str] = []
    lines.append("\n## Per-QA-Type Breakdown\n")
    header = "| System | " + " | ".join(k[len(prefix):] for k in qa_type_keys) + " |"
    sep = "|--------|" + "|".join("-" * (len(k[len(prefix):]) + 2) for k in qa_type_keys) + "|"
    lines.append(header)
    lines.append(sep)
    for system in systems:
        values = []
        for key in qa_type_keys:
            v = result.summary[system].get(key)
            if v is None:
                values.append("—")
            else:
                values.append(f"{v:.4f}")
        lines.append(f"| {system} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def _run_memory_subcommand(args: argparse.Namespace) -> None:
    """Execute the `memory` subcommand."""
    # Resolve api_key
    api_key: str | None = args.api_key or os.environ.get("OPENAI_API_KEY") or None

    # Resolve datasets (default: locomo)
    dataset_names: list[str] = args.dataset or ["locomo"]

    # Parse qa_types
    qa_types: list[str] | None = None
    if args.qa_types:
        qa_types = [t.strip() for t in args.qa_types.split(",") if t.strip()]

    # Load and merge all requested datasets
    all_examples: list[dict[str, Any]] = []
    for ds_name in dataset_names:
        examples = _load_memory_dataset(ds_name, n=args.n, qa_types=qa_types, seed=args.seed)
        all_examples.extend(examples)

    if not all_examples:
        raise SystemExit("No examples loaded. Check your --dataset and --qa-types arguments.")

    # Build systems
    systems = [
        _build_memory_system(s, relay=args.relay, model=args.model, api_key=api_key)
        for s in args.systems
    ]

    # Build evaluators
    from context_bench.evaluators import AnswerQuality, MemoryJudge
    evaluators: list[Any] = [
        AnswerQuality(),
        MemoryJudge(base_url=args.relay, model=args.model),
    ]

    # Build metrics
    from context_bench.metrics.per_qa_type import PerQATypeMetric
    from context_bench.metrics.token_efficiency import TokenEfficiencyMetric
    metrics: list[Any] = [
        PerQATypeMetric(score_field=args.score_field),
        TokenEfficiencyMetric(score_field=args.score_field),
    ]

    # Run evaluation
    from context_bench.memory_runner import evaluate_memory
    result = evaluate_memory(
        systems=systems,
        dataset=all_examples,
        evaluators=evaluators,
        metrics=metrics,
        max_examples=args.n,
        progress=True,
    )

    # Output
    if args.output == "json":
        from context_bench.reporters.json_out import to_json
        print(to_json(result))
    elif args.output == "html":
        from context_bench.reporters.html import to_html
        print(to_html(result))
    else:
        from context_bench.reporters.markdown import to_markdown
        print(to_markdown(result))
        print(_render_per_qa_type_table(result, args.score_field))


# ---------------------------------------------------------------------------
# dspy subcommand
# ---------------------------------------------------------------------------


def _run_dspy_subcommand(args: argparse.Namespace) -> None:
    """Run the DSPy optimizer benchmarking sweep."""
    from context_bench.dspy_bench.splits import (
        DATASET_TASK_TYPE,
        SPLIT_MANIFEST,
        normalize_dataset_name,
    )
    from context_bench.dspy_bench.sweep import SweepConfig, run_sweep

    # Resolve defaults
    all_optimizers = [
        "LabeledFewShot", "BootstrapFewShot",
        "BootstrapFewShotWithRandomSearch", "COPRO",
        "MIPROv2", "SIMBA", "GEPA",
    ]
    optimizers = args.optimizer or all_optimizers
    datasets = [normalize_dataset_name(d) for d in args.dataset] if args.dataset else list(DATASET_TASK_TYPE.keys())
    budgets = args.budget or ["light"]
    seeds = args.seed or [42]

    # Validate dataset names
    for ds in datasets:
        if ds not in SPLIT_MANIFEST:
            raise SystemExit(
                f"Unknown dataset: '{ds}'. "
                f"Available: {', '.join(sorted(SPLIT_MANIFEST.keys()))}"
            )

    config = SweepConfig(
        optimizers=optimizers,
        datasets=datasets,
        budgets=budgets,
        seeds=seeds,
        model=args.model,
        prompt_model=getattr(args, "prompt_model", None),
        cache_dir=args.cache_dir,
        max_train=args.max_train,
        max_val=args.max_val,
        cost_cap=args.cost_cap,
        compile_only=args.compile_only,
        include_ablations=args.include_ablations,
        n_test=args.max_examples,
    )

    results = run_sweep(config)

    # Report results
    completed = [r for r in results if r.status == "completed"]
    failed = [r for r in results if r.status == "failed"]
    skipped = [r for r in results if r.status == "skipped"]

    print(f"\nSweep complete: {len(completed)} completed, {len(failed)} failed, {len(skipped)} skipped")

    if args.output == "json":
        import json
        output = []
        for r in results:
            entry = {
                "run_key": r.run_key,
                "optimizer": r.optimizer,
                "dataset": r.dataset,
                "budget": r.budget,
                "seed": r.seed,
                "program_variant": r.program_variant,
                "status": r.status,
                "error": r.error,
            }
            if r.compile_meta:
                entry["compile_meta"] = r.compile_meta
            output.append(entry)
        print(json.dumps(output, indent=2, default=str))
    else:
        # Table output
        if completed:
            print(f"\n{'Optimizer':<35} {'Dataset':<20} {'Budget':<8} {'Seed':<6} {'Status':<10}")
            print("-" * 80)
            for r in completed:
                print(f"{r.optimizer:<35} {r.dataset:<20} {r.budget:<8} {r.seed:<6} {r.status:<10}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Dispatch to memory subcommand
    if args.subcommand == "memory":
        _run_memory_subcommand(args)
        return

    # Dispatch to dspy subcommand
    if args.subcommand == "dspy":
        _run_dspy_subcommand(args)
        return

    # Legacy proxy-benchmark path (no subcommand given)
    if not args.proxy:
        parser.error(
            "the following arguments are required when no subcommand is given: --proxy\n"
            "Run 'context-bench memory --help' to see the memory subcommand."
        )
    if not args.dataset:
        parser.error(
            "the following arguments are required when no subcommand is given: --dataset"
        )

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
    from context_bench.metrics import CompressionRatio, CostOfPass, Latency, MeanScore, PassRate, PerDatasetBreakdown
    from context_bench.runner import evaluate

    # Auto-add ROUGE-L evaluator when summarization datasets are present
    summarization_datasets = {
        "meetingbank", "govreport", "multi-news", "dialogsum", "qmsum", "summscreenfd",
    }
    dataset_names_set = {spec.split(":")[0] for spec in args.dataset}
    evaluators: list[Any] = [AnswerQuality()]
    if dataset_names_set & summarization_datasets:
        evaluators.append(SummarizationQuality())
    # Auto-add multiple-choice evaluator
    mc_datasets = {"mmlu", "arc-challenge", "gpqa", "hellaswag", "winogrande", "mmlu-pro"}
    if dataset_names_set & mc_datasets:
        from context_bench.evaluators.multiple_choice import MultipleChoiceAccuracy
        evaluators.append(MultipleChoiceAccuracy())
    # Auto-add code execution evaluator
    code_datasets = {"humaneval", "mbpp"}
    if dataset_names_set & code_datasets:
        from context_bench.evaluators.code_execution import CodeExecution
        evaluators.append(CodeExecution())
    # Auto-add IFEval checker
    if "ifeval" in dataset_names_set:
        from context_bench.evaluators.ifeval_checker import IFEvalChecker
        evaluators.append(IFEvalChecker())
    # Auto-add math equivalence evaluator
    math_datasets = {"math", "gsm8k", "mgsm"}
    if dataset_names_set & math_datasets:
        from context_bench.evaluators.math_equivalence import MathEquivalence
        evaluators.append(MathEquivalence())
    # Auto-add NLI label match evaluator
    nli_datasets = {"contract-nli", "scifact"}
    if dataset_names_set & nli_datasets:
        from context_bench.evaluators.nli_label_match import NLILabelMatch
        evaluators.append(NLILabelMatch())
    if args.judge_url:
        from context_bench.evaluators.llm_judge import LLMJudge
        evaluators.append(LLMJudge(base_url=args.judge_url, model=args.judge_model))
    metrics = [
        MeanScore(score_field=args.score_field),
        PassRate(threshold=args.threshold, score_field=args.score_field),
        CompressionRatio(),
        CostOfPass(threshold=args.threshold, score_field=args.score_field),
        Latency(),
    ]
    # Add per-dataset breakdown when multiple datasets are requested
    if len(args.dataset) > 1:
        metrics.append(PerDatasetBreakdown(score_field=args.score_field))

    # --- Run evaluation ---
    result = evaluate(
        systems=systems,
        dataset=all_examples,
        evaluators=evaluators,
        metrics=metrics,
        progress=True,
        max_workers=args.max_workers,
        cache_dir=args.cache_dir,
    )

    # --- Pareto ranking (cross-system, post-evaluation) ---
    if len(systems) > 1:
        from context_bench.metrics.token_stats import ParetoRank
        pareto_ranks = ParetoRank.rank_systems(
            result.summary,
            quality_field="mean_score",
            cost_field="cost_of_pass",
        )
        for sys_name, rank in pareto_ranks.items():
            result.summary[sys_name]["pareto_rank"] = float(rank)

    # --- Output ---
    if args.output == "json":
        from context_bench.reporters.json_out import to_json

        print(to_json(result))
    elif args.output == "html":
        from context_bench.reporters.html import to_html

        print(to_html(result))
    else:
        from context_bench.reporters.markdown import to_markdown

        print(to_markdown(result))


if __name__ == "__main__":
    main()
