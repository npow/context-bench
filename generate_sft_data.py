"""Generate SFT training data by running prompted-Claude on RLM REPL.

Saves all (prompt, generated_code) traces to JSONL, then filters to keep only
"good" traces (correct answer + used memory_write).

These filtered traces become the SFT dataset for distilling a small model
to use the RLM management policy.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

sys.path.insert(0, "src")
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")


# Strong prompt that forces memory_write usage
WRITE_FORCING_PROMPT = """
You are a memory-management agent. Answer questions using Python REPL calls.

You have these functions in scope:
  memory_read(query, k=20) -> list[str]
  memory_write(content: str, memory_type: str = "factual") -> None
  consolidate() -> str
  answer: dict({"content": str, "ready": bool})

MANDATORY workflow per query:
1. Call memory_read(query) once
2. For each useful fact in results: call memory_write(fact, "factual")
   Always call memory_write at least 3 times.
3. Set answer["content"] = "<short answer>" and answer["ready"] = True

Example:
items = memory_read("camping trips")
memory_write("User camped at Yellowstone for 5 days", "factual")
memory_write("User camped at Big Sur for 3 days", "factual")
memory_write("Total camping = 8 days", "factual")
answer["content"] = "8 days"
answer["ready"] = True

Write Python only. Use memory_write LIBERALLY (3+ times). Set answer at end.
""".strip()


def _f1(pred: str, gold: str) -> float:
    import re
    def norm(s):
        s = re.sub(r"[^\w\s]", "", s.lower()).strip()
        return [w for w in s.split() if w not in {"a", "an", "the", "is", "was", "are", "were"}]
    p = set(norm(pred)); g = set(norm(gold))
    if not p or not g: return 0.0
    common = p & g
    if not common: return 0.0
    prec = len(common) / len(p); rec = len(common) / len(g)
    return 2 * prec * rec / (prec + rec)


def run_generation(n_questions: int, trace_file: str, summary_path: str):
    """Run prompted RLM on N questions, save traces."""
    from context_bench.datasets.memory.longmemeval import longmemeval
    from context_bench.datasets.memory.locomo import locomo
    from context_bench.systems.rlm_repl import RLMSystemRepl
    import context_bench.systems.rlm_repl as repl_mod

    # Force the strong write-encouraging prompt
    repl_mod._SYSTEM_PROMPT = WRITE_FORCING_PROMPT

    # Set env var so REPL logs traces
    os.environ["SFT_TRACE_FILE"] = trace_file

    # Mix of datasets for diversity
    print(f"[setup] loading datasets...", flush=True)
    all_lme = longmemeval(n=200, question_types=None)
    all_loc = locomo(n=8)
    print(f"[setup] LME: {len(all_lme)} examples; LoCoMo: {len(all_loc)} conversations", flush=True)

    # Build flat list of (example, query) pairs
    pairs = []
    for ex in all_lme[:100]:
        for q in ex.queries[:1]:
            pairs.append((ex, q))
    for ex in all_loc:
        for q in ex.queries[:10]:
            pairs.append((ex, q))
    print(f"[setup] {len(pairs)} (example, query) pairs total", flush=True)

    pairs = pairs[:n_questions]

    relay = os.environ.get("RELAY_URL", "http://localhost:8080")
    bedrock_model = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6")

    results = []
    for i, (ex, q) in enumerate(pairs):
        print(f"\n--- Q{i+1}/{len(pairs)}: {q.question[:80]!r}", flush=True)

        system = RLMSystemRepl(
            base_url=relay,
            model="claude-sonnet-4-6",
            max_iterations=4,
        )
        try:
            system.ingest(ex.items)
            t0 = time.perf_counter()
            r = system.query(q.question)
            elapsed = time.perf_counter() - t0
            f1 = _f1(r.answer, q.answer)
            w = r.details.get("writes", 0)
            rd = r.details.get("reads", 0)
            print(f"  f1={f1:.2f} writes={w} reads={rd} ({elapsed:.1f}s) ans={r.answer[:60]!r}", flush=True)
            results.append({
                "i": i, "question": q.question, "gold": q.answer, "answer": r.answer,
                "f1": f1, "writes": w, "reads": rd, "elapsed_s": elapsed,
            })
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            results.append({"i": i, "question": q.question, "gold": q.answer, "error": str(e), "writes": 0, "f1": 0})
        finally:
            try: system.close()
            except: pass

    # Save summary
    n = len(results)
    write_adopt = sum(1 for r in results if r.get("writes", 0) > 0) / n
    mean_f1 = sum(r.get("f1", 0) for r in results) / n
    summary = {
        "n": n,
        "write_adoption_rate": write_adopt,
        "mean_f1": mean_f1,
        "traces_path": trace_file,
        "details": results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[summary] n={n} write_adopt={write_adopt:.1%} mean_f1={mean_f1:.3f}", flush=True)
    print(f"[summary] saved to {summary_path}", flush=True)
    print(f"[traces] saved to {trace_file}", flush=True)

    # S3 upload
    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        import subprocess
        for path in [trace_file, summary_path]:
            s3_key = f"s3://{s3_bucket}/sft_data/{os.path.basename(path)}"
            try:
                subprocess.run(["aws", "s3", "cp", path, s3_key], check=True, timeout=120)
                print(f"[s3] {s3_key}", flush=True)
            except Exception as e:
                print(f"[s3] {e}", flush=True)


def filter_traces(trace_file: str, summary_path: str, output_file: str, min_writes: int = 1, min_f1: float = 0.0):
    """Keep only traces from queries where the model wrote AND answered well."""
    summary = json.loads(open(summary_path).read())
    keep_indices = set()
    for r in summary["details"]:
        if r.get("writes", 0) >= min_writes and r.get("f1", 0) >= min_f1:
            keep_indices.add(r["i"])
    print(f"[filter] keeping {len(keep_indices)}/{summary['n']} queries (writes>={min_writes}, f1>={min_f1})", flush=True)

    # Read traces and keep ones from selected queries
    # Each query may have multiple iterations; we need to map traces to query indices
    # Trace file is appended sequentially: iteration 0 of Q0, iteration 1 of Q0, ..., iteration 0 of Q1, ...
    # We'll keep ALL traces from successful queries
    # But trace file doesn't have query index — need to track by line order
    raw_traces = []
    with open(trace_file) as f:
        for line in f:
            try:
                raw_traces.append(json.loads(line))
            except: continue
    print(f"[filter] {len(raw_traces)} raw traces", flush=True)

    # Group traces by question (using first user message content as key)
    by_question: dict = {}
    for t in raw_traces:
        q_key = next((m["content"] for m in t["messages"] if m.get("role") == "user"), None)
        if not q_key:
            continue
        by_question.setdefault(q_key, []).append(t)

    # Now match question text to summary entries
    kept = []
    for r in summary["details"]:
        if r["i"] not in keep_indices:
            continue
        q_text = f"Question: {r['question']}"
        traces = by_question.get(q_text, [])
        kept.extend(traces)

    # Write filtered SFT data in trl SFTTrainer-friendly format
    with open(output_file, "w") as f:
        for t in kept:
            sft_record = {
                "messages": t["messages"] + [{"role": "assistant", "content": t["code"]}]
            }
            f.write(json.dumps(sft_record) + "\n")
    print(f"[filter] {len(kept)} filtered traces saved to {output_file}", flush=True)

    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        import subprocess
        s3_key = f"s3://{s3_bucket}/sft_data/{os.path.basename(output_file)}"
        try:
            subprocess.run(["aws", "s3", "cp", output_file, s3_key], check=True, timeout=120)
            print(f"[s3] {s3_key}", flush=True)
        except Exception as e:
            print(f"[s3] {e}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["generate", "filter"], required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--trace-file", default="/tmp/sft_traces.jsonl")
    p.add_argument("--summary", default="/tmp/sft_summary.json")
    p.add_argument("--output", default="/tmp/sft_filtered.jsonl")
    p.add_argument("--min-writes", type=int, default=1)
    p.add_argument("--min-f1", type=float, default=0.0)
    args = p.parse_args()
    if args.mode == "generate":
        run_generation(args.n, args.trace_file, args.summary)
    else:
        filter_traces(args.trace_file, args.summary, args.output, args.min_writes, args.min_f1)
