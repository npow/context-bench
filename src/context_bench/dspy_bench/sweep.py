"""Sweep orchestrator for DSPy optimizer benchmarking.

Iterates over the full factorial grid of (optimizer, dataset, budget, seed,
program_variant), compiling and evaluating each cell. Supports caching,
cost estimation, and zero-shot-only dataset filtering.
"""

from __future__ import annotations

import json
import logging
import os
import random
import tempfile
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports of registry / splits constants — these may pull in dspy
# ---------------------------------------------------------------------------

try:
    from context_bench.dspy_bench.registry import REGISTRY
    from context_bench.dspy_bench.splits import (
        DATASET_TASK_TYPE,
        SPLIT_MANIFEST,
        normalize_dataset_name,
    )
    from context_bench.dspy_bench.programs import MULTI_PREDICTOR_PROGRAMS
except ImportError:
    # Allow import when dspy is not installed (tests mock these)
    REGISTRY = {}
    SPLIT_MANIFEST = {}
    DATASET_TASK_TYPE = {}
    MULTI_PREDICTOR_PROGRAMS = {}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SweepConfig:
    """Configuration for a full sweep run."""

    optimizers: list[str]
    datasets: list[str]
    budgets: list[str]  # ["light", "medium", "heavy"]
    seeds: list[int]
    model: str
    prompt_model: str | None
    cache_dir: str
    max_train: int
    max_val: int
    cost_cap: float | None
    compile_only: bool
    include_ablations: bool
    n_test: int | None  # limit test set size


@dataclass
class RunResult:
    """Result of a single (optimizer, dataset, budget, seed, variant) cell."""

    run_key: str
    optimizer: str
    dataset: str
    budget: str
    seed: int
    program_variant: str  # "single" or "multi"
    compile_meta: dict | None
    eval_result: Any | None  # EvalResult from context-bench
    status: str  # "completed", "failed", "skipped"
    error: str | None


# ---------------------------------------------------------------------------
# Allowed optimizers for zero-shot-only datasets
# ---------------------------------------------------------------------------

_ZERO_SHOT_ALLOWED = {"none", "LabeledFewShot"}

# GEPA ablation dataset subset
_ABLATION_DATASETS = {"hotpotqa", "gsm8k", "mmlu", "multi_news", "mbpp"}


# ---------------------------------------------------------------------------
# Cache path construction
# ---------------------------------------------------------------------------


def make_cache_path(
    cache_dir: str,
    optimizer: str,
    dataset: str,
    budget: str,
    seed: int,
    program_variant: str,
) -> str:
    """Build deterministic cache path for a compiled program.

    Format: ``{cache_dir}/compiled/{optimizer}_{dataset}_{budget}_{seed}_{variant}/program.json``
    """
    run_key = f"{optimizer}_{dataset}_{budget}_{seed}_{program_variant}"
    return os.path.join(cache_dir, "compiled", run_key, "program.json")


# ---------------------------------------------------------------------------
# Atomic save / load for compiled program state
# ---------------------------------------------------------------------------


def save_compiled(state: dict, path: str) -> None:
    """Atomically save compiled program state as JSON.

    Writes to a temporary file in the same directory, then renames to the
    final path (POSIX atomic rename).
    """
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        os.close(fd)
        with open(tmp_path, "w") as f:
            json.dump(state, f)
        os.rename(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def load_compiled(path: str) -> dict:
    """Load compiled program state from a JSON file."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_dataset_raw(dataset_name: str, split: str) -> list[dict]:
    """Load raw dataset split via context-bench dataset loaders."""
    import importlib

    # Map normalized dataset names to context-bench loader (module, function)
    _LOADER_MAP = {
        "hotpotqa": ("context_bench.datasets.huggingface", "hotpotqa"),
        "gsm8k": ("context_bench.datasets.huggingface", "gsm8k"),
        "natural_questions": ("context_bench.datasets.qa", "natural_questions"),
        "musique": ("context_bench.datasets.qa", "musique"),
        "narrativeqa": ("context_bench.datasets.qa", "narrativeqa"),
        "triviaqa": ("context_bench.datasets.qa", "triviaqa"),
        "frames": ("context_bench.datasets.qa", "frames"),
        "quality": ("context_bench.datasets.qa", "quality"),
        "qasper": ("context_bench.datasets.qa", "qasper"),
        "truthfulqa": ("context_bench.datasets.knowledge", "truthfulqa"),
        "gpqa": ("context_bench.datasets.knowledge", "gpqa"),
        "mmlu": ("context_bench.datasets.knowledge", "mmlu"),
        "arc_challenge": ("context_bench.datasets.knowledge", "arc_challenge"),
        "hellaswag": ("context_bench.datasets.knowledge", "hellaswag"),
        "winogrande": ("context_bench.datasets.knowledge", "winogrande"),
        "math": ("context_bench.datasets.reasoning", "math_dataset"),
        "mgsm": ("context_bench.datasets.reasoning", "mgsm"),
        "drop": ("context_bench.datasets.reasoning", "drop"),
        "humaneval": ("context_bench.datasets.code", "humaneval"),
        "mbpp": ("context_bench.datasets.code", "mbpp"),
        "multi_news": ("context_bench.datasets.summarization", "multi_news"),
        "dialogsum": ("context_bench.datasets.summarization", "dialogsum"),
        "qmsum": ("context_bench.datasets.summarization", "qmsum"),
        "summscreenfd": ("context_bench.datasets.summarization", "summscreenfd"),
        "meetingbank": ("context_bench.datasets.summarization", "meetingbank"),
        "govreport": ("context_bench.datasets.summarization", "govreport"),
    }

    if dataset_name not in _LOADER_MAP:
        raise ValueError(f"No loader for dataset '{dataset_name}'")

    module_path, func_name = _LOADER_MAP[dataset_name]
    mod = importlib.import_module(module_path)
    loader = getattr(mod, func_name)
    return list(loader(split=split))


def load_splits_for_dataset(
    dataset_name: str,
    max_train: int,
    max_val: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load train/val/test splits using the manifest.

    Returns ``(train, val, test)`` as lists of dicts.

    For zero_shot_only datasets, train and val are empty lists.
    When ``val_from == "train_split"``, the training data is split 75/25
    with a fixed seed for reproducibility.
    """
    manifest = SPLIT_MANIFEST[dataset_name]

    if manifest.get("zero_shot_only"):
        test = _load_dataset_raw(dataset_name, manifest["test_src"])
        return [], [], test

    test = _load_dataset_raw(dataset_name, manifest["test_src"])
    train_raw = _load_dataset_raw(dataset_name, manifest["train_src"])

    if manifest["val_from"] == "train_split":
        n = len(train_raw)
        cut = int(n * 0.75)
        indices = list(range(n))
        random.Random(42).shuffle(indices)
        train = [train_raw[i] for i in indices[:cut]]
        val = [train_raw[i] for i in indices[cut:]]
    else:
        val = _load_dataset_raw(dataset_name, manifest["val_from"])
        train = list(train_raw)

    if len(train) > max_train:
        train = random.Random(42).sample(train, max_train)
    if len(val) > max_val:
        val = random.Random(42).sample(val, max_val)

    return train, val, test


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


def estimate_sweep_cost(config: SweepConfig) -> dict:
    """Estimate total API calls and approximate cost before launching.

    Returns a dict with ``total_calls`` (int) and ``per_optimizer`` (dict
    mapping optimizer name to estimated calls).

    Uses each registry entry's ``cost_heuristic`` callable.
    """
    per_optimizer: dict[str, int] = {}
    total = 0

    for optimizer in config.optimizers:
        entry = REGISTRY.get(optimizer)
        if entry is None:
            per_optimizer[optimizer] = 0
            continue
        heuristic = entry["cost_heuristic"]
        opt_total = 0

        for dataset in config.datasets:
            ds = normalize_dataset_name(dataset) if callable(normalize_dataset_name) else dataset
            manifest = SPLIT_MANIFEST.get(ds, {})

            # Skip non-baseline optimizers for zero_shot_only datasets
            if manifest.get("zero_shot_only") and optimizer not in _ZERO_SHOT_ALLOWED:
                continue

            for budget in config.budgets:
                for _seed in config.seeds:
                    try:
                        calls = heuristic(budget, config.max_train, config.max_val)
                    except Exception:
                        calls = 0
                    opt_total += calls

        per_optimizer[optimizer] = opt_total
        total += opt_total

    return {
        "total_calls": total,
        "per_optimizer": per_optimizer,
    }


# ---------------------------------------------------------------------------
# Single cell compile + eval (internal helper)
# ---------------------------------------------------------------------------


def _get_evaluator_for_dataset(dataset_name: str):
    """Return the appropriate context-bench evaluator for a dataset."""
    from context_bench.evaluators.answer_quality import AnswerQuality

    _EVALUATOR_MAP = {
        "math": "math_equivalence",
        "gsm8k": "math_equivalence",
        "mgsm": "math_equivalence",
        "drop": "math_equivalence",
        "mmlu": "multiple_choice",
        "arc_challenge": "multiple_choice",
        "gpqa": "multiple_choice",
        "hellaswag": "multiple_choice",
        "winogrande": "multiple_choice",
        "humaneval": "code_execution",
        "mbpp": "code_execution",
        "multi_news": "summarization",
        "dialogsum": "summarization",
        "qmsum": "summarization",
        "summscreenfd": "summarization",
        "meetingbank": "summarization",
        "govreport": "summarization",
    }

    eval_type = _EVALUATOR_MAP.get(dataset_name, "answer_quality")

    if eval_type == "math_equivalence":
        from context_bench.evaluators.math_equivalence import MathEquivalence
        return MathEquivalence()
    elif eval_type == "multiple_choice":
        from context_bench.evaluators.multiple_choice import MultipleChoiceAccuracy
        return MultipleChoiceAccuracy()
    elif eval_type == "code_execution":
        from context_bench.evaluators.code_execution import CodeExecution
        return CodeExecution()
    elif eval_type == "summarization":
        from context_bench.evaluators.rouge import SummarizationQuality
        return SummarizationQuality()
    else:
        return AnswerQuality()


def _compile_and_eval_single(
    *,
    optimizer: str,
    dataset: str,
    budget: str,
    seed: int,
    program_variant: str,
    config: SweepConfig,
    train: list[dict],
    val: list[dict],
    test: list[dict],
) -> RunResult:
    """Compile and evaluate a single cell.

    1. Check cache for compiled program
    2. If not cached: compile via DSPy optimizer, cache result
    3. If not compile_only: evaluate via context-bench runner
    """
    import copy
    import time
    import traceback

    import dspy

    from context_bench.dspy_bench.compile_health import assess_compile_health
    from context_bench.dspy_bench.metrics_bridge import build_metric
    from context_bench.dspy_bench.programs import MULTI_PREDICTOR_PROGRAMS, TASK_PROGRAMS
    from context_bench.dspy_bench.splits import convert_to_dspy_example
    from context_bench.dspy_bench.system import DspyOptimizerSystem

    run_key = f"{optimizer}_{dataset}_{budget}_{seed}_{program_variant}"
    task_type = DATASET_TASK_TYPE[dataset]
    cache_path = make_cache_path(
        config.cache_dir, optimizer, dataset, budget, seed, program_variant,
    )

    compile_meta: dict | None = None

    # --- Compile phase ---
    if os.path.exists(cache_path):
        log.info(f"Cache hit: {run_key}")
        compiled_state = load_compiled(cache_path)
        compile_meta = compiled_state.get("_meta", {})
    else:
        log.info(f"Compiling: {run_key}")
        # Build program
        if program_variant == "multi" and dataset in MULTI_PREDICTOR_PROGRAMS:
            program = MULTI_PREDICTOR_PROGRAMS[dataset]()
        else:
            program = TASK_PROGRAMS[task_type]()

        initial_program = copy.deepcopy(program)

        # Build evaluator and metric
        evaluator = _get_evaluator_for_dataset(dataset)
        metric = build_metric(optimizer, evaluator, task_type)

        # Build optimizer
        entry = REGISTRY[optimizer]
        init_kwargs = dict(entry["init_kwargs"][budget])

        if entry.get("prompt_model") == "use_prompt_model" and config.prompt_model:
            init_kwargs["prompt_model"] = dspy.LM(model=config.prompt_model)

        if entry["accepts_metric"]:
            opt_instance = entry["class"](metric=metric, **init_kwargs)
        else:
            opt_instance = entry["class"](**init_kwargs)

        # Convert train/val to DSPy examples
        train_ex = [convert_to_dspy_example(d, task_type) for d in train]
        val_ex = [convert_to_dspy_example(d, task_type) for d in val]

        compile_kwargs = entry["build_compile_kwargs"](train_ex, val_ex, seed)

        # Compile with state isolation
        lm = dspy.LM(model=config.model)
        start_time = time.time()
        try:
            with dspy.context(lm=lm, trace=[]):
                compiled = opt_instance.compile(student=program, **compile_kwargs)
            elapsed = time.time() - start_time
        except Exception as e:
            log.error(f"Compile failed: {run_key}: {e}")
            return RunResult(
                run_key=run_key,
                optimizer=optimizer,
                dataset=dataset,
                budget=budget,
                seed=seed,
                program_variant=program_variant,
                compile_meta=None,
                eval_result=None,
                status="failed",
                error=f"Compile error: {e}\n{traceback.format_exc()}",
            )

        # Health check
        health_config = type("C", (), {
            "optimizer": optimizer,
            "health_checks": entry.get("health_checks", []),
        })()
        health = assess_compile_health(initial_program, compiled, health_config)

        compile_meta = {
            "compile_time": elapsed,
            "optimizer": optimizer,
            "compile_health": health,
            "actual_metric_calls": getattr(metric, "_call_count", 0),
        }

        # Save compiled state atomically
        try:
            import tempfile as _tf
            tmp_dir = _tf.mkdtemp()
            tmp_prog_path = os.path.join(tmp_dir, "program.json")
            compiled.save(tmp_prog_path, save_program=False)
            with open(tmp_prog_path) as f:
                state_data = json.load(f)
            state_data["_meta"] = compile_meta
            save_compiled(state_data, cache_path)
        except Exception as e:
            log.warning(f"Cache save failed for {run_key}: {e}")

    # --- Evaluation phase ---
    if config.compile_only:
        return RunResult(
            run_key=run_key,
            optimizer=optimizer,
            dataset=dataset,
            budget=budget,
            seed=seed,
            program_variant=program_variant,
            compile_meta=compile_meta,
            eval_result=None,
            status="completed",
            error=None,
        )

    # Reconstruct program from cached state for evaluation
    if program_variant == "multi" and dataset in MULTI_PREDICTOR_PROGRAMS:
        eval_program = MULTI_PREDICTOR_PROGRAMS[dataset]()
    else:
        eval_program = TASK_PROGRAMS[task_type]()

    # Load state into program if we have cached state
    if os.path.exists(cache_path):
        cached = load_compiled(cache_path)
        # Remove meta before loading into program
        cached.pop("_meta", None)
        tmp_dir2 = tempfile.mkdtemp()
        tmp_path2 = os.path.join(tmp_dir2, "state.json")
        with open(tmp_path2, "w") as f:
            json.dump(cached, f)
        eval_program.load(tmp_path2)

    # Wrap as context-bench System
    system_name = f"{optimizer}_{budget}_s{seed}_{program_variant}"
    system = DspyOptimizerSystem(eval_program, task_type, system_name)

    # Run evaluation
    try:
        from context_bench.runner import evaluate as cb_evaluate

        evaluator = _get_evaluator_for_dataset(dataset)
        lm = dspy.LM(model=config.model)

        with dspy.context(lm=lm, trace=[]):
            eval_result = cb_evaluate(
                systems=[system],
                dataset=test,
                evaluators=[evaluator],
            )
    except Exception as e:
        log.error(f"Eval failed: {run_key}: {e}")
        return RunResult(
            run_key=run_key,
            optimizer=optimizer,
            dataset=dataset,
            budget=budget,
            seed=seed,
            program_variant=program_variant,
            compile_meta=compile_meta,
            eval_result=None,
            status="failed",
            error=f"Eval error: {e}\n{traceback.format_exc()}",
        )

    return RunResult(
        run_key=run_key,
        optimizer=optimizer,
        dataset=dataset,
        budget=budget,
        seed=seed,
        program_variant=program_variant,
        compile_meta=compile_meta,
        eval_result=eval_result,
        status="completed",
        error=None,
    )


# ---------------------------------------------------------------------------
# Main sweep orchestrator
# ---------------------------------------------------------------------------


def run_sweep(config: SweepConfig) -> list[RunResult]:
    """Main sweep orchestrator. Iterates over the full grid.

    For each (dataset, program_variant, optimizer, budget, seed):
    - Skips zero_shot_only datasets for non-baseline optimizers
    - Skips GEPA_ablation for datasets not in the ablation subset
    - Runs both single and multi variants for datasets in MULTI_PREDICTOR_PROGRAMS
    """
    results: list[RunResult] = []

    # Determine active optimizers
    active_optimizers = list(config.optimizers)
    if config.include_ablations and "GEPA_ablation" not in active_optimizers:
        active_optimizers.append("GEPA_ablation")

    for dataset in config.datasets:
        ds = normalize_dataset_name(dataset) if callable(normalize_dataset_name) else dataset
        manifest = SPLIT_MANIFEST.get(ds, {})

        train, val, test = load_splits_for_dataset(ds, config.max_train, config.max_val)

        if config.n_test is not None and len(test) > config.n_test:
            test = test[: config.n_test]

        # Determine program variants
        program_variants = ["single"]
        if ds in MULTI_PREDICTOR_PROGRAMS:
            program_variants.append("multi")

        for program_variant in program_variants:
            for optimizer in active_optimizers:
                # Skip GEPA_ablation for non-ablation datasets
                if optimizer == "GEPA_ablation" and ds not in _ABLATION_DATASETS:
                    continue

                # Skip non-baseline optimizers for zero_shot_only datasets
                if manifest.get("zero_shot_only") and optimizer not in _ZERO_SHOT_ALLOWED:
                    continue

                for budget in config.budgets:
                    for seed in config.seeds:
                        run_key = f"{optimizer}_{ds}_{budget}_{seed}_{program_variant}"
                        log.info(f"Running: {run_key}")

                        try:
                            result = _compile_and_eval_single(
                                optimizer=optimizer,
                                dataset=ds,
                                budget=budget,
                                seed=seed,
                                program_variant=program_variant,
                                config=config,
                                train=train,
                                val=val,
                                test=test,
                            )
                        except Exception as e:
                            log.error(f"Failed: {run_key}: {e}")
                            result = RunResult(
                                run_key=run_key,
                                optimizer=optimizer,
                                dataset=ds,
                                budget=budget,
                                seed=seed,
                                program_variant=program_variant,
                                compile_meta=None,
                                eval_result=None,
                                status="failed",
                                error=str(e),
                            )

                        results.append(result)

    return results
