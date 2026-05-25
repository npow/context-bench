"""OMW-Bench (Online Memory-Write Benchmark).

Shifts the memory problem from "retrieval over a static log" to
"learning a memory-formation policy on a streaming log".

Phases per example (one conversation history):
  1. INGEST: agent processes sessions[1..K] in temporal order. After each
     session, may emit zero or more memory_write(content) calls.
     Crucially: the agent does NOT see future questions during ingest.
  2. QA: agent gets each held-out question Q, may call memory_read(query)
     to retrieve from the memory_store. In MEMORY_ONLY setting, no raw
     history is visible at QA time. In REALISTIC setting, last-N sessions
     are also visible.

Conditions:
  B0_strong_rag       - no writes; QA-time hybrid retrieval over raw log
  B1_write_every      - write every utterance
  B2_session_summary  - write a summary of each session
  B3_entity_profile   - write a structured profile per entity
  B4_heuristic        - rule-based salient-fact writer
  C1_q_aware_teacher  - Sonnet, sees Q during ingest. ORACLE upper bound (NOT TRAINED ON).
  C2_q_blind_teacher  - Sonnet, Q-blind. Deployable LLM-writer.
  D_sft_writer        - Qwen-3B SFT'd on C2 trajectories. Q-blind. CONTRIBUTION.
  sanity_no_writer    - spontaneous pretrained behavior (expected ~empty memory).

This script runs the EVAL pipeline for an arbitrary writer; the writer
implementations live in src/context_bench/omw/*.

Source data:
  - LoCoMo (Maharana et al. 2024) — natural multi-Q per dialog, PRIMARY.
  - LongMemEval-S (Wu et al. 2024) — 500 examples, SECONDARY.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, "src")

from context_bench.datasets.memory.longmemeval import longmemeval
from context_bench.datasets.memory.locomo import locomo

from context_bench.omw.framework import (
    OMWBenchmark, MemoryStore, run_writer, run_qa, score_qa, write_quality_metrics,
)
from context_bench.omw import writers, readers


def get_dataset(name: str, n: int | None = None):
    if name == "lme":
        return longmemeval(n=n)
    elif name == "locomo":
        return locomo(n=n)
    else:
        raise ValueError(f"unknown dataset: {name}")


def run_condition(condition: str, examples, args) -> dict:
    """Run a single OMW condition end-to-end on a set of examples.

    Special routing:
      - B0_strong_rag: skip ingest, set rag_mode=True at QA time
      - C1_q_aware_per_q: per-Q oracle (Codex fix #5) — separate trajectory per Q

    Returns aggregate metrics + per-example records.
    """
    is_rag = condition == "B0_strong_rag"
    is_per_q_oracle = condition == "C1_per_q_oracle"
    bench = OMWBenchmark(
        memory_only=args.memory_only,
        last_n_window=args.last_n_window,
        max_writes_per_session=args.max_writes_per_session,
        seed=42,
        rag_mode=is_rag,
    )
    reader = readers.get_reader(args)
    records = []

    for i, ex in enumerate(examples):
        if is_rag:
            # No ingest; memory is empty; QA uses rag_mode (raw log retrieval)
            memory = type("Mem", (), {"entries": [], "total_chars": 0, "write_count": 0, "read": lambda *a, **k: []})()
            ingest_dt = 0.0
        elif is_per_q_oracle:
            # Per-Q oracle: ingest once per question (separate memory per Q)
            ingest_dt = 0.0  # accumulated below per Q
            memory = None  # set per-Q
            per_q_writer_factory = writers.make_writer_q_aware_teacher_per_q(
                writers._client(), args.writer_model
            )
        else:
            writer = writers.get_writer(condition, args)
            t0 = time.perf_counter()
            memory = bench.ingest(writer, ex)
            ingest_dt = time.perf_counter() - t0

        for qi, q in enumerate(ex.queries):
            # Per-Q oracle: build fresh memory for this Q
            if is_per_q_oracle:
                pq_writer = per_q_writer_factory(q)
                t0 = time.perf_counter()
                q_memory = bench.ingest(pq_writer, ex)
                ingest_dt = time.perf_counter() - t0
                current_memory = q_memory
            else:
                current_memory = memory

            qa_t0 = time.perf_counter()
            ans = bench.answer(reader, current_memory, ex, q)
            qa_dt = time.perf_counter() - qa_t0
            judge = score_qa(reader, ans, q)

            # Write_recall via LLM judge (not heuristic)
            if hasattr(current_memory, "entries") and current_memory.entries:
                wq = write_quality_metrics(current_memory, ex, q, reader_module=reader)
            else:
                wq = {"write_recall": 0, "memory_used_for_judge_chars": 0}

            records.append({
                "condition": condition,
                "ex_id": ex.id,
                "q_idx": qi,
                "question": q.question,
                "gold": q.answer,
                "ans": ans,
                "judge": judge,
                "qa_dt": qa_dt,
                "ingest_dt_per_ex": ingest_dt,
                "memory_size_chars": getattr(current_memory, "total_chars", 0),
                "memory_write_count": getattr(current_memory, "write_count", 0),
                "write_recall": wq.get("write_recall"),
            })
        if i % 5 == 0:
            print(f"  [{condition}] ex {i+1}/{len(examples)} qs_done={len(records)}", flush=True)
    return {"records": records}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["lme", "locomo"], default="locomo")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--conditions", default="C2_q_blind_teacher,B0_strong_rag,B2_session_summary")
    p.add_argument("--memory-only", action="store_true", default=True)
    p.add_argument("--last-n-window", type=int, default=0)
    p.add_argument("--max-writes-per-session", type=int, default=10)
    p.add_argument("--reader-model", default="us.anthropic.claude-sonnet-4-6")
    p.add_argument("--writer-model", default="us.anthropic.claude-sonnet-4-6")
    p.add_argument("--student-adapter", default="")
    p.add_argument("--output", default="/tmp/omw_bench.json")
    args = p.parse_args()

    print(f"[setup] dataset={args.dataset} n={args.n} memory_only={args.memory_only}", flush=True)
    examples = get_dataset(args.dataset, n=args.n)
    print(f"[setup] {len(examples)} examples, {sum(len(e.queries) for e in examples)} total queries", flush=True)

    results = {}
    for cond in args.conditions.split(","):
        cond = cond.strip()
        print(f"\n=== running condition: {cond} ===", flush=True)
        results[cond] = run_condition(cond, examples, args)

    # Aggregate accuracy per condition
    print("\n" + "=" * 80, flush=True)
    print(f"OMW-BENCH RESULTS ({args.dataset}, n_ex={len(examples)})", flush=True)
    print("=" * 80, flush=True)
    for cond, data in results.items():
        judges = [r["judge"] for r in data["records"] if r["judge"] is not None]
        mem_sizes = [r["memory_size_chars"] for r in data["records"]]
        w_counts = [r["memory_write_count"] for r in data["records"]]
        acc = sum(judges) / max(1, len(judges))
        print(
            f"  {cond:24s}  acc={acc:.3f} (n_q={len(judges)})  "
            f"mem_chars≈{sum(mem_sizes)//max(1,len(mem_sizes))}  "
            f"writes/ex≈{sum(w_counts)/max(1,len(w_counts)):.1f}",
            flush=True,
        )

    out = {
        "dataset": args.dataset,
        "n_examples": len(examples),
        "conditions": list(results.keys()),
        "results": results,
        "args": vars(args),
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {args.output}", flush=True)

    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        import subprocess
        s3_key = f"s3://{s3_bucket}/omw_bench/{os.path.basename(args.output)}"
        try:
            subprocess.run(["aws", "s3", "cp", args.output, s3_key], check=True, timeout=120)
            print(f"[s3] {s3_key}", flush=True)
        except Exception as e:
            print(f"[s3] {e}", flush=True)


if __name__ == "__main__":
    main()
