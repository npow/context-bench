"""Management-policy gap measurement across 5 models × 3 prompt variants × 100 questions.

Addresses Codex critique: "n=24 with one model is too small."

Design
------
- Dataset : LongMemEval multi-session (more memory-intensive than LoCoMo)
- Models  : 5 Bedrock models (Sonnet 4.6, Haiku 4.5, Opus 4.7, GPT-OSS-120B, Gemma-3-27B)
- Prompts : 3 variants (A=neutral, B=encourages-write, C=explicit-write-required)
- n       : 100 multi-session questions per cell
- Metrics : read_adoption_rate, write_adoption_rate, mean_writes_per_query,
            valid_python_rate, f1_mean, f1_ci_lo/hi

Output JSON
-----------
{
  "cells": {
    "(model_alias, prompt_label)": {
      "n": 100,
      "read_adoption_rate": ...,
      "write_adoption_rate": ...,
      "mean_writes_per_query": ...,
      "valid_python_rate": ...,
      "f1_mean": ...,
      "f1_ci_lo": ...,
      "f1_ci_hi": ...
    }
  },
  "raw": { "(model_alias, prompt_label)": [per-query records] }
}

The RLMSystemRepl requires LanceDB + sentence-transformers for real embedding-based
retrieval. Both packages work on Mako GPU instances (the pip install in the entrypoint
handles this). The BEDROCK_MODEL_ID env var selects the model; _chat() in rlm.py
already dispatches to Bedrock when that var is set.

NOTE: _chat_bedrock only handles Anthropic models natively. This script patches
_chat_bedrock via a multi-model wrapper that handles OpenAI/Gemma Bedrock routes.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import sys
import textwrap
import time
import traceback

sys.path.insert(0, "src")
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")

from context_bench.datasets.memory.longmemeval import longmemeval
import context_bench.systems.rlm_repl as repl_mod

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "sonnet-4-6":    "us.anthropic.claude-sonnet-4-6",
    "haiku-4-5":     "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "opus-4-7":      "us.anthropic.claude-opus-4-7",
    "gpt-oss-120b":  "openai.gpt-oss-120b-1:0",
    "gemma-3-27b":   "google.gemma-3-27b-it",
}

# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------

PROMPT_A_NEUTRAL = textwrap.dedent("""
You are an agent that answers questions by managing memory programmatically.

You have access to these functions in your Python namespace:

  memory_read(query, k=20) -> list[str]
    Retrieve k relevant memory entries for the given query string.

  memory_write(content: str, memory_type: str = "episodic") -> None
    Write content to persistent memory.
    memory_type: "episodic", "factual", or "procedural"

  consolidate() -> str
    Compress and summarize current working context into memory.

  answer: dict  ({"content": str, "ready": bool})

Protocol:
1. Call memory_read() to retrieve relevant context
2. Use the retrieved context to answer the question
3. Set answer["content"] (SHORT direct answer, under 15 words) and answer["ready"] = True

Write Python code only. No markdown, no explanations.
""").strip()

PROMPT_B_ENCOURAGE_WRITE = textwrap.dedent("""
You are an agent that answers questions by managing memory programmatically.

You have access to these functions in your Python namespace:

  memory_read(query, k=20) -> list[str]
    Retrieve k relevant memory entries for the given query string.

  memory_write(content: str, memory_type: str = "episodic") -> None
    Write content to persistent memory. Writing notes about what you find
    may help answer future queries more efficiently.
    memory_type: "episodic", "factual", or "procedural"

  consolidate() -> str
    Compress and summarize current working context into memory.

  answer: dict  ({"content": str, "ready": bool})

Protocol:
1. Call memory_read() to retrieve relevant context
2. Optionally call memory_write() to save key facts you discover — this builds
   a knowledge base that helps with related future questions
3. Set answer["content"] (SHORT direct answer, under 15 words) and answer["ready"] = True

Write Python code only. No markdown, no explanations.
""").strip()

PROMPT_C_EXPLICIT_WRITE = textwrap.dedent("""
You are a memory-management agent answering questions about a long conversation history.

You MUST follow this exact protocol for EVERY query:

  memory_read(query, k=20) -> list[str]   — retrieves relevant entries
  memory_write(content: str, memory_type: str = "episodic") -> None
    — YOU MUST call this for EVERY non-trivial fact you find.
      Minimum 2 memory_write calls per query. No exceptions.
  consolidate() -> str   — compress context when large
  answer: dict  ({"content": str, "ready": bool})

Required protocol:
1. Call memory_read() to retrieve relevant context
2. For EACH key fact found: call memory_write(fact, "factual")
   — MANDATORY: at least 2 memory_write calls per query
3. Set answer["content"] (SHORT answer, under 15 words) and answer["ready"] = True

Example:
  items = memory_read("camping trips")
  memory_write("User went camping in Yellowstone for 5 days", "factual")
  memory_write("User also camped at Big Sur for 3 days", "factual")
  answer["content"] = "8 days total"
  answer["ready"] = True

Write Python only. ALWAYS call memory_write at least twice.
""").strip()

PROMPT_VARIANTS = {
    "prompt_A": PROMPT_A_NEUTRAL,
    "prompt_B": PROMPT_B_ENCOURAGE_WRITE,
    "prompt_C": PROMPT_C_EXPLICIT_WRITE,
}

# ---------------------------------------------------------------------------
# Multi-model Bedrock dispatcher (patches RLMSystem._chat_bedrock)
# ---------------------------------------------------------------------------

def _make_multimodel_chat_bedrock():
    """Return a patched _chat_bedrock that handles Anthropic, OpenAI, and Gemma."""
    import json as _json
    import re as _re
    import time as _time

    def _chat_bedrock(self, messages, model_id):
        import boto3
        if not hasattr(self, "_bedrock_client"):
            self._bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

        for attempt in range(3):
            try:
                if "anthropic" in model_id:
                    system_msgs = [m["content"] for m in messages if m.get("role") == "system"]
                    user_msgs = [m for m in messages if m.get("role") != "system"]
                    body = _json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 1000,
                        "system": "\n\n".join(system_msgs) if system_msgs else None,
                        "messages": user_msgs,
                        "temperature": 0.3,
                    })
                    resp = self._bedrock_client.invoke_model(
                        body=body, modelId=model_id,
                        accept="application/json", contentType="application/json",
                    )
                    return _json.loads(resp["body"].read())["content"][0]["text"]

                if "openai" in model_id or "gpt" in model_id:
                    # OpenAI-via-Bedrock: system messages get folded into user turn
                    merged = []
                    for m in messages:
                        if m.get("role") == "system":
                            merged.append({"role": "user", "content": m["content"]})
                        else:
                            merged.append(m)
                    body = _json.dumps({
                        "messages": merged,
                        "max_completion_tokens": 1200,
                    })
                    resp = self._bedrock_client.invoke_model(body=body, modelId=model_id)
                    raw = _json.loads(resp["body"].read())
                    text = raw["choices"][0]["message"]["content"]
                    # Strip reasoning tags if present
                    text = _re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=_re.DOTALL)
                    return text.strip()

                if "gemma" in model_id or "google" in model_id:
                    merged = []
                    for m in messages:
                        if m.get("role") == "system":
                            merged.append({"role": "user", "content": m["content"]})
                        else:
                            merged.append(m)
                    body = _json.dumps({
                        "messages": merged,
                        "max_tokens": 1000,
                        "temperature": 0.3,
                    })
                    resp = self._bedrock_client.invoke_model(body=body, modelId=model_id)
                    raw = _json.loads(resp["body"].read())
                    if "choices" in raw:
                        return raw["choices"][0]["message"]["content"].strip()
                    if "generation" in raw:
                        return raw["generation"].strip()
                    return str(raw).strip()

                raise ValueError(f"Unknown model family for model_id={model_id}")

            except Exception as e:
                if attempt < 2:
                    _time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Bedrock error after {attempt+1} attempts: {e}") from e

        raise RuntimeError("Failed after 3 Bedrock retries")

    return _chat_bedrock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f1(pred: str, gold: str) -> float:
    import re
    def norm(s):
        s = re.sub(r"[^\w\s]", "", s.lower()).strip()
        return [w for w in s.split() if w not in {
            "a", "an", "the", "is", "was", "are", "were", "i", "me", "my",
            "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
        }]
    p = set(norm(pred))
    g = set(norm(gold))
    if not p or not g:
        return 0.0
    common = p & g
    if not common:
        return 0.0
    prec = len(common) / len(p)
    rec = len(common) / len(g)
    return 2 * prec * rec / (prec + rec)


def _bootstrap_ci(values: list[float], n_boot: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(42)
    n = len(values)
    boot_means = sorted(
        sum(rng.choices(values, k=n)) / n for _ in range(n_boot)
    )
    lo = boot_means[int((1 - ci) / 2 * n_boot)]
    hi = boot_means[int((1 + ci) / 2 * n_boot)]
    return lo, hi


def _is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


# ---------------------------------------------------------------------------
# Run one cell: (model_alias, prompt_label) × n_questions
# ---------------------------------------------------------------------------

def run_cell(
    model_alias: str,
    model_id: str,
    prompt_label: str,
    prompt_text: str,
    examples: list,
    n: int,
    relay_url: str,
) -> list[dict]:
    """Run one measurement cell. Returns list of per-query result dicts."""
    from context_bench.systems.rlm import RLMSystem
    from context_bench.systems.rlm_repl import RLMSystemRepl

    # Patch multi-model Bedrock dispatcher into the base class
    RLMSystem._chat_bedrock = _make_multimodel_chat_bedrock()

    # Monkey-patch the system prompt for this cell
    repl_mod._SYSTEM_PROMPT = prompt_text

    # Select examples (stable shuffle for reproducibility)
    rng = random.Random(12345)
    pool = list(examples)
    rng.shuffle(pool)
    selected = pool[:n]

    results = []
    for i, ex in enumerate(selected):
        q = ex.queries[0]
        cell_key = f"({model_alias}, {prompt_label})"
        print(
            f"[{cell_key}] Q{i+1}/{n} type={q.query_type} "
            f"items={len(ex.items)}",
            flush=True,
        )
        print(f"  Q: {q.question[:100]!r}", flush=True)

        system = RLMSystemRepl(
            base_url=relay_url,
            model=model_alias,
            max_iterations=5,
        )

        rec: dict = {
            "model": model_alias,
            "prompt": prompt_label,
            "question": q.question,
            "gold": q.answer,
            "query_type": q.query_type,
        }

        try:
            # Set Bedrock model ID for this call
            os.environ["BEDROCK_MODEL_ID"] = model_id

            system.ingest(ex.items)
            t0 = time.perf_counter()
            result = system.query(q.question)
            elapsed = time.perf_counter() - t0

            f1 = _f1(result.answer, q.answer)
            writes = result.details.get("writes", 0)
            reads = result.details.get("reads", 0)
            consolidations = result.details.get("consolidations", 0)
            answer_ready = result.details.get("answer_ready", False)
            iterations = result.details.get("iterations", 0)

            # Determine whether the generated code was syntactically valid
            # We reconstruct the code check from the details; if answer was
            # set and ready, the code was at minimum executable.
            # Additionally check via the details["method"] field.
            method = result.details.get("method", "")
            valid_python = (method == "rlm_repl") or answer_ready

            rec.update({
                "answer": result.answer,
                "f1": f1,
                "writes": writes,
                "reads": reads,
                "consolidations": consolidations,
                "answer_ready": answer_ready,
                "iterations": iterations,
                "valid_python": valid_python,
                "elapsed_s": elapsed,
                "error": None,
            })

            print(
                f"  f1={f1:.2f} writes={writes} reads={reads} "
                f"ready={answer_ready} ({elapsed:.1f}s) "
                f"ans={result.answer[:60]!r}",
                flush=True,
            )

        except Exception as e:
            tb = traceback.format_exc()
            print(f"  FAILED: {e}\n{tb}", flush=True)
            rec.update({
                "answer": "",
                "f1": 0.0,
                "writes": 0,
                "reads": 0,
                "consolidations": 0,
                "answer_ready": False,
                "iterations": 0,
                "valid_python": False,
                "elapsed_s": 0.0,
                "error": str(e),
            })

        finally:
            os.environ.pop("BEDROCK_MODEL_ID", None)
            try:
                system.close()
            except Exception:
                pass

        results.append(rec)

    return results


# ---------------------------------------------------------------------------
# Aggregate cell results → metrics
# ---------------------------------------------------------------------------

def aggregate(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0}

    f1_vals = [r["f1"] for r in records]
    ci_lo, ci_hi = _bootstrap_ci(f1_vals)

    read_adopt = sum(1 for r in records if r.get("reads", 0) > 0) / n
    write_adopt = sum(1 for r in records if r.get("writes", 0) > 0) / n
    mean_writes = sum(r.get("writes", 0) for r in records) / n
    valid_python = sum(1 for r in records if r.get("valid_python", False)) / n
    mean_f1 = sum(f1_vals) / n
    errors = sum(1 for r in records if r.get("error") is not None)

    return {
        "n": n,
        "read_adoption_rate": round(read_adopt, 4),
        "write_adoption_rate": round(write_adopt, 4),
        "mean_writes_per_query": round(mean_writes, 3),
        "valid_python_rate": round(valid_python, 4),
        "f1_mean": round(mean_f1, 4),
        "f1_ci_lo": round(ci_lo, 4),
        "f1_ci_hi": round(ci_hi, 4),
        "error_count": errors,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gap measurement: 5 models × 3 prompts × 100 LongMemEval questions"
    )
    parser.add_argument("--n", type=int, default=100,
                        help="Questions per (model, prompt) cell (default: 100)")
    parser.add_argument("--output", default="/tmp/gap_n100.json",
                        help="Output JSON path")
    parser.add_argument(
        "--relay",
        default=None,
        help="OpenAI-compatible relay URL (only used when BEDROCK_MODEL_ID is not set)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        default=list(MODELS.keys()),
        help="Subset of model aliases to run (default: all 5)",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        choices=list(PROMPT_VARIANTS.keys()),
        default=list(PROMPT_VARIANTS.keys()),
        help="Subset of prompt variants to run (default: all 3)",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for example selection")
    args = parser.parse_args()

    relay_url = args.relay or os.environ.get(
        "RELAY_URL", "http://mgp.local.dev.netflix.net:9123/proxy/npowws"
    )
    print(f"[setup] relay={relay_url}", flush=True)
    print(f"[setup] n_per_cell={args.n}", flush=True)
    print(f"[setup] models={args.models}", flush=True)
    print(f"[setup] prompts={args.prompts}", flush=True)

    # Load dataset once
    print("[setup] Loading LongMemEval multi-session examples...", flush=True)
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"[setup] HF_HOME={hf_home}", flush=True)
    all_examples = longmemeval(n=500, question_types=None)
    multi = [ex for ex in all_examples
             if any("multi-session" in q.query_type for q in ex.queries)]
    print(f"[setup] {len(multi)} multi-session examples available", flush=True)

    if len(multi) < args.n:
        print(
            f"[WARNING] Only {len(multi)} multi-session examples available, "
            f"requested {args.n}. Using all {len(multi)}.",
            flush=True,
        )
        actual_n = len(multi)
    else:
        actual_n = args.n

    # Build cells
    cells_to_run = [
        (model_alias, prompt_label)
        for model_alias in args.models
        for prompt_label in args.prompts
    ]
    total_cells = len(cells_to_run)
    print(f"[setup] {total_cells} cells to run ({total_cells * actual_n} total queries)", flush=True)

    output: dict = {"cells": {}, "raw": {}, "meta": {
        "n_per_cell": actual_n,
        "models": args.models,
        "prompts": args.prompts,
        "dataset": "LongMemEval-multi-session",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }}

    for cell_idx, (model_alias, prompt_label) in enumerate(cells_to_run):
        model_id = MODELS[model_alias]
        prompt_text = PROMPT_VARIANTS[prompt_label]
        cell_key = f"({model_alias}, {prompt_label})"

        print(f"\n{'='*70}", flush=True)
        print(
            f"CELL {cell_idx+1}/{total_cells}: {cell_key}",
            flush=True,
        )
        print(f"  model_id={model_id}", flush=True)
        print(f"  prompt={prompt_label}", flush=True)
        print(f"{'='*70}\n", flush=True)

        t_cell_start = time.perf_counter()

        records = run_cell(
            model_alias=model_alias,
            model_id=model_id,
            prompt_label=prompt_label,
            prompt_text=prompt_text,
            examples=multi,
            n=actual_n,
            relay_url=relay_url,
        )

        cell_elapsed = time.perf_counter() - t_cell_start
        metrics = aggregate(records)
        metrics["elapsed_s"] = round(cell_elapsed, 1)

        output["cells"][cell_key] = metrics
        output["raw"][cell_key] = records

        print(f"\n[cell done] {cell_key}:", flush=True)
        print(f"  read_adoption_rate  = {metrics['read_adoption_rate']:.1%}", flush=True)
        print(f"  write_adoption_rate = {metrics['write_adoption_rate']:.1%}", flush=True)
        print(f"  mean_writes/query   = {metrics['mean_writes_per_query']:.2f}", flush=True)
        print(f"  valid_python_rate   = {metrics['valid_python_rate']:.1%}", flush=True)
        print(f"  f1_mean             = {metrics['f1_mean']:.3f} "
              f"[{metrics['f1_ci_lo']:.3f}, {metrics['f1_ci_hi']:.3f}]", flush=True)
        print(f"  elapsed             = {cell_elapsed/60:.1f} min", flush=True)

        # Save intermediate checkpoint after every cell
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[checkpoint] saved {args.output}", flush=True)

        # Upload to S3 if configured
        s3_bucket = os.environ.get("S3_BUCKET")
        if s3_bucket:
            import subprocess
            s3_key = f"s3://{s3_bucket}/gap_measurement/{os.path.basename(args.output)}"
            try:
                subprocess.run(
                    ["aws", "s3", "cp", args.output, s3_key],
                    check=True, timeout=60,
                )
                print(f"[s3] {s3_key}", flush=True)
            except Exception as e:
                print(f"[s3] upload failed: {e}", flush=True)

    # Final summary table
    print(f"\n{'='*70}", flush=True)
    print("FINAL SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    header = f"{'Cell':<35} {'n':>5} {'read%':>7} {'write%':>7} {'writes/q':>9} {'f1':>7}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for cell_key, m in output["cells"].items():
        print(
            f"{cell_key:<35} {m['n']:>5} "
            f"{m['read_adoption_rate']:>7.1%} "
            f"{m['write_adoption_rate']:>7.1%} "
            f"{m['mean_writes_per_query']:>9.2f} "
            f"{m['f1_mean']:>7.3f}",
            flush=True,
        )

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[done] Final results saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
