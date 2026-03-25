#!/usr/bin/env python3
"""Run OOLONG aggregation benchmark across all 3 systems.

OOLONG tests aggregation reasoning — classify and count items across a large document.
This is where code-generated retrieval (RLM with DuckDB) should beat embedding search
because embeddings can't count.
"""

import sys
import time
from collections import defaultdict

# --- Config ---
KOMPACT = "http://localhost:7878"
KOMPACT_MODEL = "claude-sonnet-4-5-20250929"
KOMPACT_KEY = "your-api-key-1"

RELAY = "http://localhost:18082"
RELAY_MODEL = "sonnet"

N_EXAMPLES = 5
MAX_QUERIES_PER_EXAMPLE = 6  # Limit queries per context window for speed

print("=" * 80, flush=True)
print("OOLONG Aggregation Benchmark", flush=True)
print("=" * 80, flush=True)
print(f"Kompact: {KOMPACT}, Model: {KOMPACT_MODEL}", flush=True)
print(f"Relay (judge): {RELAY}, Model: {RELAY_MODEL}", flush=True)
print(f"N examples: {N_EXAMPLES}, Max queries/example: {MAX_QUERIES_PER_EXAMPLE}", flush=True)
print(flush=True)

# --- Load dataset ---
print("[1/4] Loading OOLONG dataset...", flush=True)
from context_bench.datasets.memory.oolong import oolong

examples = oolong(
    n=N_EXAMPLES,
    variant="synth",
    split="validation",
    max_context_len=4096,  # Keep context manageable
)

# Limit queries per example to keep runtime reasonable
for ex in examples:
    if len(ex.queries) > MAX_QUERIES_PER_EXAMPLE:
        # Keep a mix of query types
        by_type: dict[str, list] = defaultdict(list)
        for q in ex.queries:
            by_type[q.query_type].append(q)
        selected = []
        # Round-robin across types
        type_iters = {t: iter(qs) for t, qs in by_type.items()}
        while len(selected) < MAX_QUERIES_PER_EXAMPLE and type_iters:
            exhausted = []
            for t, it in type_iters.items():
                if len(selected) >= MAX_QUERIES_PER_EXAMPLE:
                    break
                try:
                    selected.append(next(it))
                except StopIteration:
                    exhausted.append(t)
            for t in exhausted:
                del type_iters[t]
        ex.queries[:] = selected

total_queries = sum(len(e.queries) for e in examples)
print(f"  Loaded {len(examples)} examples, {total_queries} queries", flush=True)
for ex in examples:
    types = defaultdict(int)
    for q in ex.queries:
        types[q.query_type] += 1
    print(f"  {ex.id}: {len(ex.items)} chunks, {len(ex.queries)} queries ({dict(types)})", flush=True)
print(flush=True)

# --- Build systems ---
print("[2/4] Initializing systems...", flush=True)

from context_bench.systems.rlm import RLMSystem
from context_bench.systems.embedding import EmbeddingSystem
from context_bench.systems.naive import NaiveSystem

systems = [
    NaiveSystem(base_url=KOMPACT, model=KOMPACT_MODEL, api_key=KOMPACT_KEY, timeout=300.0),
    EmbeddingSystem(base_url=KOMPACT, model=KOMPACT_MODEL, top_k=50, api_key=KOMPACT_KEY, timeout=300.0),
    RLMSystem(base_url=KOMPACT, model=KOMPACT_MODEL, api_key=KOMPACT_KEY, timeout=300.0),
]

for s in systems:
    print(f"  {s.name}", flush=True)
print(flush=True)

# --- Build evaluators ---
print("[3/4] Initializing evaluators...", flush=True)

from context_bench.evaluators.answer_quality import AnswerQuality
from context_bench.evaluators.llm_judge_locomo import LLMJudgeLoCoMo

evaluators = [
    AnswerQuality(),
    LLMJudgeLoCoMo(relay_url=RELAY, model=RELAY_MODEL, api_key="unused"),
]

for e in evaluators:
    print(f"  {e.name}", flush=True)
print(flush=True)

# --- Run evaluation ---
print("[4/4] Running evaluation...", flush=True)
print(f"  {len(examples)} examples x {len(systems)} systems = {len(examples) * len(systems)} system-runs", flush=True)
print(flush=True)

from context_bench.memory_runner import evaluate_memory
from context_bench.results import EvalRow

all_rows: list[EvalRow] = []

t0 = time.time()

for system in systems:
    sys_t0 = time.time()
    print(f"--- Running system: {system.name} ---", flush=True)

    result = evaluate_memory(
        systems=[system],
        dataset=examples,
        evaluators=evaluators,
        metrics=[],
        progress=True,
    )
    all_rows.extend(result.rows)

    # Print per-row progress
    for row in result.rows:
        qa_type = row.metadata.get("qa_type", "?")
        f1 = row.scores.get("f1", 0.0)
        judge = row.scores.get("llm_judge", 0.0)
        print(f"  [{system.name}] {row.example_id} ({qa_type}): f1={f1:.3f} judge={judge:.1f}", flush=True)

    elapsed = time.time() - sys_t0
    print(f"  {system.name} done in {elapsed:.1f}s", flush=True)
    print(flush=True)

total_elapsed = time.time() - t0
print(f"Total time: {total_elapsed:.1f}s", flush=True)
print(flush=True)

# --- Compute summary ---
print("=" * 80, flush=True)
print("RESULTS TABLE", flush=True)
print("=" * 80, flush=True)

system_names = [s.name for s in systems]
qa_types_seen = sorted(set(r.metadata.get("qa_type", "?") for r in all_rows))

for score_field in ["f1", "llm_judge"]:
    print(flush=True)
    print(f"--- {score_field.upper()} ---", flush=True)

    header = f"{'System':<25}"
    for qt in qa_types_seen:
        header += f" {qt:>15}"
    header += f" {'MEAN':>12}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for sname in system_names:
        sys_rows = [r for r in all_rows if r.system == sname]
        buckets: dict[str, list[float]] = defaultdict(list)
        all_vals: list[float] = []
        for r in sys_rows:
            qt = r.metadata.get("qa_type", "?")
            v = r.scores.get(score_field, 0.0)
            buckets[qt].append(v)
            all_vals.append(v)

        line = f"{sname:<25}"
        for qt in qa_types_seen:
            vals = buckets.get(qt, [])
            mean = sum(vals) / len(vals) if vals else 0.0
            line += f" {mean:>15.3f}"
        overall = sum(all_vals) / len(all_vals) if all_vals else 0.0
        line += f" {overall:>12.3f}"
        print(line, flush=True)

# --- Per-example detail ---
print(flush=True)
print("=" * 80, flush=True)
print("PER-EXAMPLE DETAIL", flush=True)
print("=" * 80, flush=True)

for sname in system_names:
    sys_rows = [r for r in all_rows if r.system == sname]
    print(f"\n{sname}:", flush=True)
    for row in sys_rows:
        qa_type = row.metadata.get("qa_type", "?")
        f1 = row.scores.get("f1", 0.0)
        judge = row.scores.get("llm_judge", 0.0)
        em = row.scores.get("exact_match", 0.0)
        print(f"  {row.example_id:40s} ({qa_type:>20s}): f1={f1:.3f} judge={judge:.1f} em={em:.0f}", flush=True)

print(flush=True)
print("Done.", flush=True)
