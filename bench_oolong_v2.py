#!/usr/bin/env python3
"""OOLONG aggregation benchmark v2 - large context sizes.

Tests naive, embedding, and RLM (with aggregation) at increasing context sizes.
"""

import json
import sys
import time
from collections import OrderedDict, defaultdict

sys.path.insert(0, "$(pwd)/src")

from context_bench.datasets.memory.oolong import _normalize_answer, _split_into_chunks
from context_bench.memory_types import BenchmarkExample, BenchmarkQuery, DocumentChunk
from context_bench.evaluators.memory_judge import MemoryJudge

KOMPACT_URL = "http://localhost:7878"
KOMPACT_MODEL = "claude-sonnet-4-5-20250929"
KOMPACT_KEY = "your-api-key-1"
JUDGE_URL = "http://localhost:18082"
JUDGE_MODEL = "sonnet"


def load_oolong_at_size(ctx_size, n):
    import datasets
    print(f"[LOAD] Loading oolong-synth at context_len={ctx_size}...", flush=True)
    ds = datasets.load_dataset("oolongbench/oolong-synth", split="validation", streaming=True)
    windows = OrderedDict()
    rows_seen = 0
    for row in ds:
        rows_seen += 1
        if rows_seen > 200000:
            break
        if row.get("context_len", 0) != ctx_size:
            continue
        cwid = str(row.get("context_window_id", ""))
        if cwid not in windows:
            if len(windows) >= n:
                break
            windows[cwid] = {
                "context_text": row.get("context_window_text", ""),
                "queries": [],
            }
        q = row.get("question", "")
        a = _normalize_answer(row.get("answer", ""))
        t = row.get("task", "").replace("TASK_TYPE.", "")
        if q and a:
            windows[cwid]["queries"].append(BenchmarkQuery(
                question=q, answer=a, query_type=t,
            ))

    examples = []
    for cwid, w in windows.items():
        if not w["queries"]:
            continue
        chunks = _split_into_chunks(w["context_text"], cwid, "synth")
        examples.append(BenchmarkExample(
            id=f"oolong_{cwid}",
            items=chunks,
            queries=w["queries"],
            dataset="oolong_synth",
            metadata={"context_len": ctx_size},
        ))
    total_q = sum(len(e.queries) for e in examples)
    print(f"[LOAD] {len(examples)} examples, {total_q} queries (scanned {rows_seen} rows)", flush=True)
    return examples


def run_system(system, example, judge):
    """Run system on one example, return list of (query_type, score, gold, pred, latency)."""
    system.reset()
    t0 = time.time()
    try:
        system.ingest(example.items)
        print(f"    Ingested {len(example.items)} chunks in {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        print(f"    Ingest FAILED: {e}", flush=True)
        return [(q.query_type, 0.0, q.answer, "", 0) for q in example.queries]

    results = []
    for q in example.queries:
        t0 = time.time()
        try:
            result = system.query(q.question)
            elapsed = time.time() - t0
            score_dict = judge.score(
                {"question": q.question, "answer": q.answer},
                {"response": result.answer},
            )
            score = score_dict.get("memory_judge", 0.0)
            pred = result.answer[:80]
        except Exception as e:
            score = 0.0
            pred = f"ERROR: {e}"
            elapsed = time.time() - t0

        print(f"    [{score:.0f}] {q.query_type}: gold={q.answer} | pred={pred} ({elapsed:.1f}s)", flush=True)
        results.append((q.query_type, score, q.answer, pred, elapsed))
    return results


def make_system(name):
    if name == "naive":
        from context_bench.systems.naive import NaiveSystem
        return NaiveSystem(base_url=KOMPACT_URL, model=KOMPACT_MODEL,
                           api_key=KOMPACT_KEY, timeout=300.0)
    elif name == "embedding":
        from context_bench.systems.embedding import EmbeddingSystem
        return EmbeddingSystem(base_url=KOMPACT_URL, model=KOMPACT_MODEL,
                               api_key=KOMPACT_KEY, timeout=300.0)
    elif name == "rlm":
        from context_bench.systems.rlm import RLMSystem
        return RLMSystem(base_url=KOMPACT_URL, model=KOMPACT_MODEL,
                         api_key=KOMPACT_KEY, timeout=300.0)


def main():
    print("=" * 70, flush=True)
    print("OOLONG LARGE CONTEXT BENCHMARK v2", flush=True)
    print("=" * 70, flush=True)

    judge = MemoryJudge(base_url=JUDGE_URL, model=JUDGE_MODEL, api_key="unused")

    # Test plan: (context_size, systems_to_test, n_examples)
    test_plan = [
        (8192, ["naive", "embedding", "rlm"], 1),
        (32768, ["naive", "embedding", "rlm"], 1),
        (65536, ["embedding", "rlm"], 1),
        (131072, ["rlm"], 1),
    ]

    all_results = {}

    for ctx_size, sys_names, n in test_plan:
        print(f"\n{'=' * 70}", flush=True)
        print(f"CONTEXT SIZE: {ctx_size:,} tokens ({ctx_size//1024}K)", flush=True)
        print(f"{'=' * 70}", flush=True)

        examples = load_oolong_at_size(ctx_size, n)
        if not examples:
            print("  NO EXAMPLES, skipping", flush=True)
            continue

        for ex in examples:
            chars = sum(len(c.content) for c in ex.items)
            print(f"  {ex.id}: {len(ex.items)} chunks, {chars:,} chars, {len(ex.queries)} queries", flush=True)

        size_results = {}
        for sys_name in sys_names:
            print(f"\n--- {sys_name} at {ctx_size:,} tokens ---", flush=True)
            system = make_system(sys_name)
            t0_total = time.time()

            all_query_results = []
            for ex in examples:
                print(f"  Example {ex.id}:", flush=True)
                qr = run_system(system, ex, judge)
                all_query_results.extend(qr)

            total_time = time.time() - t0_total
            scores = [s for _, s, _, _, _ in all_query_results]
            avg = sum(scores) / len(scores) if scores else 0.0

            by_type = defaultdict(list)
            for qt, s, _, _, _ in all_query_results:
                by_type[qt].append(s)

            print(f"\n  {sys_name} SCORE: {avg:.3f} ({len(scores)} queries, {total_time:.0f}s)", flush=True)
            for qt, ss in sorted(by_type.items()):
                print(f"    {qt}: {sum(ss)/len(ss):.3f} ({len(ss)}q)", flush=True)

            size_results[sys_name] = {
                "score": avg,
                "n_queries": len(scores),
                "elapsed_s": total_time,
                "by_type": {qt: sum(ss)/len(ss) for qt, ss in by_type.items()},
            }

        all_results[str(ctx_size)] = size_results

    # Summary table
    print(f"\n\n{'=' * 70}", flush=True)
    print("FINAL SUMMARY", flush=True)
    print(f"{'=' * 70}", flush=True)

    all_sys = set()
    for sr in all_results.values():
        all_sys.update(sr.keys())
    sys_list = sorted(all_sys)

    header = f"{'Size':>10}"
    for s in sys_list:
        header += f" | {s:>10}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for ctx_size, _, _ in test_plan:
        key = str(ctx_size)
        if key not in all_results:
            continue
        sr = all_results[key]
        line = f"{ctx_size:>10,}"
        for s in sys_list:
            if s in sr:
                line += f" | {sr[s]['score']:>10.3f}"
            else:
                line += f" | {'--':>10}"
        print(line, flush=True)

    outpath = "oolong_large_results.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}", flush=True)


if __name__ == "__main__":
    main()
