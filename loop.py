#!/usr/bin/env python3
# loop.py
# ===================================================================
# Autoresearch loop for context-bench.
#
# Runs overnight to discover better context engineering strategies by
# repeatedly mutating context_pipeline.py, evaluating each candidate,
# and keeping changes that improve the token_efficiency score.
#
# Usage:
#   python loop.py \
#     --relay http://localhost:18082 \
#     --model claude-haiku-4-5-20251001 \
#     --dataset locomo \
#     --iterations 200 \
#     --eval-n 30 \
#     --seed 42 \
#     --output-dir ./loop_results
#
# After all iterations the loop runs a final held-out evaluation
# comparing the best discovered pipeline against NaiveMemorySystem.
# ===================================================================

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any, Type

# Make the src layout importable when running loop.py directly from the repo
# root without `pip install -e .`.
_SRC = Path(__file__).parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Dynamic pipeline loading
# ---------------------------------------------------------------------------

def load_pipeline_from_code(code: str) -> Type[Any]:
    """Write code to a temp file, import it, return the ContextPipeline class.

    Raises ImportError (or any other exception the code raises on import) if
    the module cannot be loaded.  The temp file is deleted after import so
    that the filesystem stays clean even over 200 iterations.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".py", delete=False, mode="w", encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        spec = importlib.util.spec_from_file_location(
            "context_pipeline_candidate", tmp_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    finally:
        # Always clean up the temp file, even if the import fails.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not hasattr(module, "ContextPipeline"):
        raise ImportError("Module does not define a 'ContextPipeline' class")

    return module.ContextPipeline


# ---------------------------------------------------------------------------
# One-line description of what changed between two versions of the code
# ---------------------------------------------------------------------------

def _summarise_mutation(old_code: str, new_code: str) -> str:
    """Return a short human-readable description of the diff.

    Scans for changes to the STRATEGY dict and the class body, producing a
    compact string suitable for logging.  Falls back to a generic label if
    the change cannot be characterised neatly.
    """
    import re

    # Look for changed STRATEGY values
    old_strategy = re.search(r"STRATEGY\s*=\s*\{(.+?)\}", old_code, re.DOTALL)
    new_strategy = re.search(r"STRATEGY\s*=\s*\{(.+?)\}", new_code, re.DOTALL)

    if old_strategy and new_strategy:
        old_text = old_strategy.group(1)
        new_text = new_strategy.group(1)
        if old_text != new_text:
            # Find first differing line
            old_lines = [l.strip() for l in old_text.splitlines() if l.strip()]
            new_lines = [l.strip() for l in new_text.splitlines() if l.strip()]
            for ol, nl in zip(old_lines, new_lines):
                if ol != nl:
                    return f"STRATEGY: {ol!r} -> {nl!r}"
            # Different number of lines
            if len(old_lines) != len(new_lines):
                return "STRATEGY: added/removed keys"

    # Check for code-level changes outside the STRATEGY dict
    old_non_strategy = re.sub(r"STRATEGY\s*=\s*\{.+?\}", "", old_code, flags=re.DOTALL)
    new_non_strategy = re.sub(r"STRATEGY\s*=\s*\{.+?\}", "", new_code, flags=re.DOTALL)
    if old_non_strategy != new_non_strategy:
        return "code-level change (method or logic modified)"

    # Identical — the agent returned the same file
    return "no visible change"


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_pipeline(
    pipeline_class: Type[Any],
    examples: list[dict],
    relay_url: str,
    model: str,
    api_key: str,
) -> dict[str, float]:
    """Run score_pipeline() and return the results dict.

    Retries on transient relay errors (503/429) up to 5 times with backoff.
    Any non-transient exception is re-raised so the caller can treat it as a
    rejection.
    """
    import time as _time
    from context_bench.pipeline.scoring import score_pipeline

    for attempt in range(5):
        try:
            return score_pipeline(
                pipeline_class=pipeline_class,
                examples=examples,
                relay_url=relay_url,
                model=model,
                api_key=api_key,
            )
        except Exception as exc:
            msg = str(exc)
            if any(code in msg for code in ("503", "429", "502", "504")) and attempt < 4:
                wait = 30 * (2 ** attempt)  # 30s, 60s, 120s, 240s
                print(f"  [retry {attempt+1}/4] Transient error ({msg[:60]}), waiting {wait}s...", flush=True)
                _time.sleep(wait)
                continue
            raise


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(
    output_dir: Path,
    best_code: str,
    log: list[dict],
) -> None:
    """Persist the current best pipeline and full log to disk.

    Called after every accepted mutation so the loop can be safely
    interrupted (Ctrl-C) without losing work.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    best_path = output_dir / "best_pipeline.py"
    best_path.write_text(best_code, encoding="utf-8")

    log_path = output_dir / "loop_log.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        for entry in log:
            f.write(json.dumps(entry) + "\n")


def load_checkpoint(output_dir: Path) -> tuple[str | None, list[dict]]:
    """Load a previous run's best pipeline and log, if they exist.

    Returns (best_code_or_None, log_list).
    """
    best_path = output_dir / "best_pipeline.py"
    log_path = output_dir / "loop_log.jsonl"

    best_code = None
    if best_path.exists():
        best_code = best_path.read_text(encoding="utf-8")

    log: list[dict] = []
    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        log.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    return best_code, log


# ---------------------------------------------------------------------------
# Final head-to-head evaluation
# ---------------------------------------------------------------------------

def run_final_eval(
    best_code: str,
    test_examples: list[dict],
    relay_url: str,
    model: str,
    api_key: str,
    output_dir: Path,
) -> None:
    """Evaluate the best discovered pipeline vs NaiveMemorySystem on the held-out test set.

    Prints a comparison table and saves results to output-dir/final_results.json.
    """
    from context_bench.pipeline.scoring import score_pipeline
    from context_bench.systems.naive_memory import NaiveMemorySystem

    print("\n" + "=" * 65)
    print("FINAL EVALUATION — held-out test set")
    print("=" * 65)

    # Evaluate best discovered pipeline
    try:
        best_class = load_pipeline_from_code(best_code)
        best_results = score_pipeline(
            pipeline_class=best_class,
            examples=test_examples,
            relay_url=relay_url,
            model=model,
            api_key=api_key,
        )
    except Exception as exc:
        print(f"ERROR evaluating best pipeline: {exc}")
        best_results = {"score": 0.0, "f1": 0.0, "token_efficiency": 0.0,
                        "mean_input_tokens": 0.0, "n": 0}

    # Evaluate naive baseline
    # NaiveMemorySystem takes base_url, not relay_url — wrap in a thin adapter
    class _NaiveAdapter:
        """Thin wrapper so NaiveMemorySystem matches the score_pipeline() protocol."""

        def __init__(self, relay_url: str, model: str, api_key: str | None = None):
            self._inner = NaiveMemorySystem(
                base_url=relay_url,
                model=model,
                api_key=api_key,
            )

        @property
        def name(self) -> str:
            return self._inner.name

        @property
        def last_context_tokens(self) -> int:
            return self._inner.last_context_tokens

        def reset(self) -> None:
            self._inner.reset()

        def ingest(self, turns):
            self._inner.ingest(turns)

        def query(self, question: str) -> str:
            return self._inner.query(question)

    try:
        naive_results = score_pipeline(
            pipeline_class=_NaiveAdapter,
            examples=test_examples,
            relay_url=relay_url,
            model=model,
            api_key=api_key,
        )
    except Exception as exc:
        print(f"ERROR evaluating NaiveMemorySystem: {exc}")
        naive_results = {"score": 0.0, "f1": 0.0, "token_efficiency": 0.0,
                         "mean_input_tokens": 0.0, "n": 0}

    # Print comparison table
    col_w = 22
    print(f"\n{'Metric':<25} {'BestPipeline':>{col_w}} {'NaiveBaseline':>{col_w}}")
    print("-" * (25 + col_w * 2 + 2))

    metrics = [
        ("token_efficiency (score)", "score"),
        ("F1", "f1"),
        ("mean_input_tokens", "mean_input_tokens"),
        ("n (QA pairs)", "n"),
    ]
    for label, key in metrics:
        bv = best_results.get(key, 0.0)
        nv = naive_results.get(key, 0.0)
        # Format integers cleanly
        if key == "n":
            print(f"{label:<25} {int(bv):>{col_w}} {int(nv):>{col_w}}")
        else:
            print(f"{label:<25} {bv:>{col_w}.4f} {nv:>{col_w}.4f}")

    # Delta row for the primary metric
    delta = best_results.get("score", 0.0) - naive_results.get("score", 0.0)
    sign = "+" if delta >= 0 else ""
    print(f"\n  vs. naive: {sign}{delta:.4f} token_efficiency")

    # Save to disk
    final = {
        "best_pipeline": best_results,
        "naive_baseline": naive_results,
        "delta": delta,
        "test_n": len(test_examples),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "final_results.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )
    print(f"\n  Saved to {output_dir / 'final_results.json'}")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(name: str) -> list[dict]:
    """Load a named dataset and return a flat list of examples."""
    if name == "locomo":
        from context_bench.datasets.memory import locomo
        return locomo()
    elif name.startswith("longmemeval"):
        from context_bench.datasets.memory import longmemeval
        # Support "longmemeval_s", "longmemeval_m", "longmemeval_oracle"
        variant = name.split("_", 1)[1] if "_" in name else "s"
        return longmemeval(variant=variant)
    else:
        raise ValueError(
            f"Unknown dataset: {name!r}. "
            "Supported: 'locomo', 'longmemeval_s', 'longmemeval_m', 'longmemeval_oracle'"
        )


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autoresearch loop: overnight search over context strategies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python loop.py --relay http://localhost:18082 --dataset locomo
              python loop.py --relay http://localhost:18082 --iterations 50 --eval-n 20
        """),
    )
    parser.add_argument(
        "--relay",
        default="http://localhost:18082",
        metavar="URL",
        help="OpenAI-compatible relay URL (default: http://localhost:18082)",
    )
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        metavar="MODEL",
        help="Model for eval/answer calls (default: claude-haiku-4-5-20251001)",
    )
    parser.add_argument(
        "--mutation-model",
        default="claude-sonnet-4-6",
        metavar="MODEL",
        help="Model for mutation proposals — use a stronger model here (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--dataset",
        default="locomo",
        choices=["locomo", "longmemeval_s", "longmemeval_m", "longmemeval_oracle"],
        help="Which benchmark dataset to use (default: locomo)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        metavar="N",
        help="Number of mutation/eval iterations (default: 200)",
    )
    parser.add_argument(
        "--eval-n",
        type=int,
        default=30,
        metavar="N",
        help="Number of examples sampled per evaluation (default: 30)",
    )
    parser.add_argument(
        "--max-qa-per-conv",
        type=int,
        default=10,
        metavar="N",
        help="Max QA pairs per conversation in eval (default: 10, avoids huge LoCoMo evals)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="SEED",
        help="RNG seed for the train/test split (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        default="./loop_results",
        metavar="DIR",
        help="Directory for checkpoints and results (default: ./loop_results)",
    )
    parser.add_argument(
        "--api-key",
        default="",
        metavar="KEY",
        help="Bearer token for the relay. Falls back to OPENAI_API_KEY env var.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a previous checkpoint in --output-dir (if present)",
    )
    parser.add_argument(
        "--pipeline-path",
        default=None,
        metavar="PATH",
        help="Path to the starting context_pipeline.py (default: repo src copy)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    relay_url: str = args.relay
    model: str = args.model
    mutation_model: str = args.mutation_model
    api_key: str = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    iterations: int = args.iterations
    eval_n: int = args.eval_n
    max_qa_per_conv: int = args.max_qa_per_conv
    seed: int = args.seed
    output_dir = Path(args.output_dir)

    # ------------------------------------------------------------------
    # 1. Load dataset and split into train / test
    # ------------------------------------------------------------------
    print(f"Loading dataset: {args.dataset} ...", flush=True)
    all_examples = load_dataset(args.dataset)
    print(f"  Loaded {len(all_examples)} conversations", flush=True)

    from context_bench.datasets.memory import split_conversations, sample_search_pool

    train_examples, test_examples = split_conversations(
        all_examples, train_frac=0.8, seed=seed
    )
    print(
        f"  Split: {len(train_examples)} train / {len(test_examples)} test "
        f"(seed={seed})",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 2. Load the starting context_pipeline.py
    # ------------------------------------------------------------------
    if args.pipeline_path:
        pipeline_source_path = Path(args.pipeline_path)
    else:
        # Default: the canonical copy inside the installed package
        pipeline_source_path = (
            Path(__file__).parent
            / "src"
            / "context_bench"
            / "pipeline"
            / "context_pipeline.py"
        )

    if not pipeline_source_path.exists():
        print(f"ERROR: pipeline source not found at {pipeline_source_path}", file=sys.stderr)
        sys.exit(1)

    starting_code = pipeline_source_path.read_text(encoding="utf-8")
    print(f"  Pipeline source: {pipeline_source_path}", flush=True)

    # ------------------------------------------------------------------
    # 3. Initialise or resume from checkpoint
    # ------------------------------------------------------------------
    log: list[dict] = []
    best_code = starting_code

    if args.resume:
        saved_code, saved_log = load_checkpoint(output_dir)
        if saved_code is not None:
            best_code = saved_code
            log = saved_log
            print(
                f"  Resumed checkpoint: {len(log)} iterations already logged",
                flush=True,
            )
        else:
            print("  No checkpoint found — starting fresh", flush=True)

    current_code = best_code  # the code we evolve from iteration to iteration

    # ------------------------------------------------------------------
    # 4. Evaluate baseline score on a FIXED sample from train.
    #    We re-use this same sample for every candidate evaluation so
    #    accept/reject decisions are apples-to-apples comparisons on
    #    identical data.  Using a fresh random sample per iteration would
    #    make the comparison noisy: a worse pipeline could "win" simply by
    #    drawing an easier subset.
    # ------------------------------------------------------------------
    print("\nEvaluating baseline ...", flush=True)
    eval_sample = sample_search_pool(train_examples, n=eval_n, seed=seed)

    # Cap QA pairs per conversation so large datasets (e.g. LoCoMo with 200 QA/conv)
    # don't make each evaluation take hours.
    if max_qa_per_conv and max_qa_per_conv > 0:
        eval_sample = [
            {**ex, "qa_pairs": ex["qa_pairs"][:max_qa_per_conv]}
            for ex in eval_sample
        ]


    # Cache baseline so restarts don't re-evaluate from scratch.
    baseline_cache_path = output_dir / "baseline_score.json"
    if args.resume and baseline_cache_path.exists():
        import json as _json
        _cached = _json.loads(baseline_cache_path.read_text())
        baseline_results = _cached
        best_score = baseline_results["score"]
        print(f"  (resumed cached baseline)", flush=True)
    else:
        try:
            baseline_class = load_pipeline_from_code(current_code)
            baseline_results = evaluate_pipeline(
                baseline_class, eval_sample, relay_url, model, api_key
            )
            best_score = baseline_results["score"]
            output_dir.mkdir(parents=True, exist_ok=True)
            import json as _json
            baseline_cache_path.write_text(_json.dumps(baseline_results))
        except Exception as exc:
            print(f"  ERROR during baseline evaluation: {exc}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)

    print(
        f"  Baseline score: {best_score:.4f}  "
        f"(f1={baseline_results['f1']:.4f}, "
        f"mean_tokens={baseline_results['mean_input_tokens']:.0f}, "
        f"n={baseline_results['n']})",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 5. Instantiate the mutator
    # ------------------------------------------------------------------
    from context_bench.loop import PipelineMutator

    mutator = PipelineMutator(
        relay_url=relay_url,
        model=mutation_model,
        api_key=api_key,
        timeout=600.0,  # 10 min — architectural rewrites take time
    )

    # ------------------------------------------------------------------
    # 6. Main iteration loop
    # ------------------------------------------------------------------
    print(f"\nStarting search: {iterations} iterations, eval_n={eval_n}\n", flush=True)

    start_iteration = len(log)  # allows resuming mid-run

    for iteration in range(start_iteration, start_iteration + iterations):
        iter_start = time.monotonic()

        # ----------------------------------------------------------------
        # a. Ask the LLM to propose a mutation
        # ----------------------------------------------------------------
        try:
            candidate_code = mutator.propose_mutation(
                current_code=current_code,
                score_history=log,
            )
        except Exception as exc:
            print(
                f"Iter {iteration + 1}/{start_iteration + iterations} | "
                f"MUTATION ERROR: {exc} — skipping",
                flush=True,
            )
            log.append({
                "iteration": iteration,
                "score": best_score,
                "delta": 0.0,
                "mutation": f"mutation_error: {exc}",
                "accepted": False,
                "error": str(exc),
            })
            continue

        # Characterise what changed for the log
        mutation_summary = _summarise_mutation(current_code, candidate_code)

        # ----------------------------------------------------------------
        # b. Try to import the candidate
        # ----------------------------------------------------------------
        try:
            candidate_class = load_pipeline_from_code(candidate_code)
        except Exception as exc:
            print(
                f"Iter {iteration + 1}/{start_iteration + iterations} | "
                f"IMPORT ERROR: {exc} — rejected",
                flush=True,
            )
            log.append({
                "iteration": iteration,
                "score": best_score,
                "delta": 0.0,
                "mutation": mutation_summary,
                "accepted": False,
                "error": f"import_error: {exc}",
            })
            continue

        # ----------------------------------------------------------------
        # c. Evaluate on the SAME fixed sample used for the baseline.
        #    This keeps accept/reject decisions comparable across iterations.
        # ----------------------------------------------------------------

        try:
            results = evaluate_pipeline(
                candidate_class, eval_sample, relay_url, model, api_key
            )
            candidate_score = results["score"]
            eval_error = None
        except Exception as exc:
            candidate_score = 0.0
            eval_error = str(exc)
            results = {}

        # ----------------------------------------------------------------
        # d. Accept or reject
        # ----------------------------------------------------------------
        delta = candidate_score - best_score
        accepted = candidate_score > best_score

        if accepted:
            best_score = candidate_score
            best_code = candidate_code
            current_code = candidate_code
        # If rejected: current_code stays as-is (we keep mutating from best)

        # ----------------------------------------------------------------
        # e. Log the iteration
        # ----------------------------------------------------------------
        elapsed = time.monotonic() - iter_start
        log_entry: dict[str, Any] = {
            "iteration": iteration,
            "score": candidate_score,
            "best_score": best_score,
            "delta": delta,
            "mutation": mutation_summary,
            "accepted": accepted,
            "elapsed_s": round(elapsed, 1),
        }
        if eval_error:
            log_entry["error"] = eval_error
        if results:
            log_entry["f1"] = results.get("f1", 0.0)
            log_entry["mean_input_tokens"] = results.get("mean_input_tokens", 0.0)
            log_entry["n"] = results.get("n", 0)
        log.append(log_entry)

        # ----------------------------------------------------------------
        # f. Print progress line
        # ----------------------------------------------------------------
        sign = "+" if delta >= 0 else ""
        status_icon = "accepted" if accepted else "rejected"
        print(
            f"Iter {iteration + 1:>3}/{start_iteration + iterations} | "
            f"score={candidate_score:.4f} ({sign}{delta:.4f}) "
            f"{status_icon} | "
            f"best={best_score:.4f} | "
            f"{elapsed:.1f}s | "
            f"{mutation_summary}",
            flush=True,
        )

        # ----------------------------------------------------------------
        # g. Checkpoint after every iteration (not just accepted) so that
        #    restarts with --resume can skip the baseline and replay history.
        # ----------------------------------------------------------------
        save_checkpoint(output_dir, best_code, log)

    # ------------------------------------------------------------------
    # 7. Final checkpoint after all iterations complete
    # ------------------------------------------------------------------
    save_checkpoint(output_dir, best_code, log)
    print(f"\nLoop complete.  Best score: {best_score:.4f}", flush=True)
    print(f"Best pipeline saved to: {output_dir / 'best_pipeline.py'}", flush=True)
    print(f"Full log saved to:      {output_dir / 'loop_log.jsonl'}", flush=True)

    # ------------------------------------------------------------------
    # 8. Final held-out evaluation: best discovered pipeline vs naive
    # ------------------------------------------------------------------
    run_final_eval(
        best_code=best_code,
        test_examples=test_examples,
        relay_url=relay_url,
        model=model,
        api_key=api_key,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
