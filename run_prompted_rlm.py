"""Approach 3: Prompt-engineered RLM with explicit memory_write instruction.

Tests whether Claude, given a strong system prompt explicitly instructing it
to call memory_write() after processing each session's content, will actually
use the write API. Bridges the gap between:

- Pretrained RLM (0% write adoption — Approach 1 baseline)
- Oracle consolidation (perfect writes, but hardcoded — Approach 2)

If prompted-Claude achieves >50% write adoption AND beats baseline on
multi-session F1, that's evidence that the management policy can be steered
via prompting (not just trained from scratch).

For SFT distillation, the per-query REPL transcripts are saved for later use
as supervised training data.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, "src")
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")

from context_bench.datasets.memory.longmemeval import longmemeval
from context_bench.systems.rlm_repl import RLMSystemRepl


# Strong system prompt that explicitly demands memory_write usage
PROMPT_AGGRESSIVE_WRITE = """
You are a memory-management agent answering questions about a long conversation history.

You have a persistent memory store. After EVERY memory_read() that returns useful
information, you MUST call memory_write(content="<key fact>", memory_type="factual")
to save the fact for future queries. ALWAYS write at least 3 facts per query.

Your protocol:
1. Call memory_read(query) to retrieve relevant context
2. For each interesting fact found: call memory_write(fact, "factual")
3. Then: set answer["content"] = "<short answer>" and answer["ready"] = True

Example:
  items = memory_read("camping trips")
  memory_write("User went camping in Yellowstone for 5 days", "factual")
  memory_write("User also camped at Big Sur for 3 days", "factual")
  answer["content"] = "8 days total"
  answer["ready"] = True

Write Python only. Use memory_write LIBERALLY — at least 3 calls per query.
Last lines MUST set answer["content"] (short string) and answer["ready"] = True.
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
    return 2 * (len(common)/len(p)) * (len(common)/len(g)) / ((len(common)/len(p)) + (len(common)/len(g)))


def run(n_questions: int = 20, output_path: str = "/tmp/prompted_rlm.json", relay: str = None):
    relay = relay or os.environ.get("RELAY_URL", "http://mgp.local.dev.netflix.net:9123/proxy/npowws")
    print(f"[setup] relay={relay}", flush=True)

    all_examples = longmemeval(n=300, question_types=None)
    multi = [ex for ex in all_examples if any("multi-session" in q.query_type for q in ex.queries)][:n_questions]
    print(f"[setup] {len(multi)} multi-session questions", flush=True)

    # Monkey-patch the system prompt
    import context_bench.systems.rlm_repl as repl_mod
    repl_mod._SYSTEM_PROMPT = PROMPT_AGGRESSIVE_WRITE

    results = []
    for i, ex in enumerate(multi):
        q = ex.queries[0]
        print(f"\n--- Q{i+1}/{len(multi)} type={q.query_type} ---", flush=True)
        print(f"  Question: {q.question[:90]!r}", flush=True)
        print(f"  Expected: {q.answer!r}", flush=True)

        system = RLMSystemRepl(
            base_url=relay,
            model="claude-sonnet-4-6",
            max_iterations=5,
        )
        try:
            system.ingest(ex.items)
            t0 = time.perf_counter()
            result = system.query(q.question)
            elapsed = time.perf_counter() - t0
            f1 = _f1(result.answer, q.answer)
            writes = result.details.get("writes", 0)
            reads = result.details.get("reads", 0)
            print(f"  f1={f1:.2f} writes={writes} reads={reads} ({elapsed:.1f}s) ans={result.answer[:80]!r}", flush=True)
            results.append({
                "q": q.question, "gt": q.answer, "ans": result.answer,
                "f1": f1, "writes": writes, "reads": reads, "elapsed_s": elapsed,
            })
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            results.append({"q": q.question, "gt": q.answer, "ans": "", "f1": 0.0, "writes": 0, "reads": 0, "error": str(e)})
        finally:
            try: system.close()
            except: pass

    n = len(results)
    write_adopt = sum(1 for r in results if r.get("writes", 0) > 0) / n
    mean_writes = sum(r.get("writes", 0) for r in results) / n
    mean_f1 = sum(r["f1"] for r in results) / n

    print("\n" + "=" * 60, flush=True)
    print(f"PROMPTED RLM RESULTS (n={n} multi-session)", flush=True)
    print(f"  write_adoption_rate: {write_adopt:.1%}", flush=True)
    print(f"  mean_writes_per_q:   {mean_writes:.2f}", flush=True)
    print(f"  mean_f1:             {mean_f1:.3f}", flush=True)

    out = {"n": n, "write_adoption_rate": write_adopt, "mean_writes_per_q": mean_writes, "mean_f1": mean_f1, "details": results}
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {output_path}", flush=True)

    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        import subprocess
        s3_key = f"s3://{s3_bucket}/prompted_rlm/{os.path.basename(output_path)}"
        try:
            subprocess.run(["aws", "s3", "cp", output_path, s3_key], check=True, timeout=60)
            print(f"[s3] {s3_key}", flush=True)
        except Exception as e:
            print(f"[s3] {e}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--output", default="/tmp/prompted_rlm.json")
    p.add_argument("--relay", default=None)
    args = p.parse_args()
    run(n_questions=args.n, output_path=args.output, relay=args.relay)
