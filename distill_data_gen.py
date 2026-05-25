"""SFT distillation data generation: run 3-stage strong pipeline on 300 questions.

For each question, logs every (prompt, response) pair from all 3 stages to JSONL,
tagged by stage. Also records the final answer + LLM-judge score.

Train split: seed=42, 300 questions (indices 0..299 from multi-session pool)
Eval split:  seed=137, 100 questions (reserved, non-overlapping)

Output JSONL schema per line:
  {
    "question_id": str,
    "question": str,
    "gold": str,
    "stage": "relevance" | "consolidate" | "answer",
    "prompt": str,
    "response": str,
    "session_id": str | null,   # populated for relevance + consolidate stages
    "final_answer": str,        # populated only on "answer" stage record
    "final_judge": 0 | 1        # populated only on "answer" stage record
  }
"""
from __future__ import annotations
import argparse
import json
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, "src")

S3_BUCKET = os.environ.get("S3_BUCKET", "netflix.pi.prod/users/npow/rlm_grpo")
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6")

# Reproducible seeds — train set must not overlap with eval set
TRAIN_SEED = 42
EVAL_SEED = 137
N_TRAIN = 300
N_EVAL = 100


def _bedrock_client():
    import boto3
    return boto3.client("bedrock-runtime", region_name="us-east-1")


def _claude(client, prompt: str, max_tokens: int = 200, temperature: float = 0.3) -> str:
    model_id = BEDROCK_MODEL_ID
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    })
    r = client.invoke_model(
        body=body, modelId=model_id,
        accept="application/json", contentType="application/json",
    )
    return json.loads(r["body"].read())["content"][0]["text"].strip()


def group_by_session(items):
    sessions: dict = {}
    for item in items:
        sid = getattr(item, "session_id", None) or "default"
        sessions.setdefault(sid, []).append(item)
    return sessions


def format_session(sid, sitems, max_chars=10000):
    text = f"[Session {sid}]\n" + "\n".join(
        f"{getattr(i, 'speaker', '') or getattr(i, 'role', '')}: {i.content}"
        for i in sitems[:50]
    )
    return text[:max_chars]


def format_haystack(items, max_chars=80000):
    parts = []
    cur = None
    for item in items:
        sid = getattr(item, "session_id", "") or "default"
        if sid != cur:
            parts.append(f"\n[Session {sid}]")
            cur = sid
        parts.append(f"{getattr(item, 'speaker', '') or getattr(item, 'role', '')}: {item.content}")
    return "\n".join(parts)[:max_chars]


# ── Stage 1 prompt builder ───────────────────────────────────────────────────
def relevance_prompt(session_text: str, question: str) -> str:
    return (
        "Determine whether the session below contains ANY information that could help answer "
        f"the question. Respond with exactly one word: YES or NO.\n\n"
        f"QUESTION: {question}\n\n"
        f"SESSION:\n{session_text[:8000]}\n\n"
        "Respond YES or NO:"
    )


# ── Stage 2 prompt builder ───────────────────────────────────────────────────
def consolidation_prompt(session_text: str, question: str) -> str:
    return (
        f"You are summarizing a conversation session for the purpose of answering a specific "
        f"question. Extract ALL facts from this session that could help answer the question. "
        f"Be specific (names, numbers, dates, places). One fact per line, no markdown.\n\n"
        f"QUESTION: {question}\n\nSESSION:\n{session_text[:10000]}\n\n"
        "Relevant facts (one per line):"
    )


# ── Stage 3 prompt builder ───────────────────────────────────────────────────
def answer_prompt(evidence: str, question: str) -> str:
    if len(evidence) > 60000:
        evidence = evidence[:60000] + "\n[...truncated...]"
    return (
        "Answer the question precisely. If the answer is a number, give just the number. "
        "If it's a list, give the count. Be concise (under 15 words).\n\n"
        f"EVIDENCE:\n{evidence}\n\nQUESTION: {question}\n\nAnswer:"
    )


# ── Judge prompt ─────────────────────────────────────────────────────────────
def judge_prompt(pred: str, gold: str, question: str) -> str:
    return (
        "Judge if the PREDICTION correctly answers the QUESTION given GOLD. "
        "Reply CORRECT or WRONG only.\n\n"
        f"QUESTION: {question}\nGOLD: {gold}\nPREDICTION: {pred}\n\nReply:"
    )


def llm_judge(client, pred: str, gold: str, question: str) -> int:
    if not pred.strip():
        return 0
    v = _claude(client, judge_prompt(pred, gold, question), max_tokens=10, temperature=0.0).upper()
    # FIX: INCORRECT contains CORRECT — check negatives first
    if "WRONG" in v or "INCORRECT" in v or "NOT CORRECT" in v: return 0
    return 1 if "CORRECT" in v else 0


# ── Core: run all 3 stages, log every (prompt, response) pair ────────────────
def run_and_log(client, ex, question: str, gold: str, q_id: str, writer):
    """Run 3-stage pipeline on one example; write one JSONL line per LLM call.

    Returns (final_answer, final_judge).
    """
    sessions = group_by_session(ex.items)

    # ── Stage 1: relevance gate ──────────────────────────────────────────────
    relevant_sessions = []
    for sid, sitems in sessions.items():
        if len(sitems) < 2:
            continue
        s_text = format_session(sid, sitems)
        prompt = relevance_prompt(s_text, question)
        response = _claude(client, prompt, max_tokens=10)
        writer.write(json.dumps({
            "question_id": q_id,
            "question": question,
            "gold": gold,
            "stage": "relevance",
            "session_id": sid,
            "prompt": prompt,
            "response": response,
            "final_answer": None,
            "final_judge": None,
        }) + "\n")
        if "YES" in response.strip().upper():
            relevant_sessions.append((sid, sitems))

    if not relevant_sessions:
        # Fallback: keep top 5 sessions so we always have something to consolidate
        # FIX: keyword-matched fallback (chronological-first biases evidence)
        q_words = {w.lower() for w in question.split() if len(w) > 3}
        if q_words:
            scored = [(sid, sitems, sum(1 for s in sitems
                                        for w in q_words if w in s.content.lower()))
                      for sid, sitems in sessions.items() if len(sitems) >= 2]
            scored.sort(key=lambda x: -x[2])
            relevant_sessions = [(sid, sitems) for sid, sitems, score in scored[:5] if score > 0]
        if not relevant_sessions:
            relevant_sessions = [(sid, sitems) for sid, sitems in sessions.items() if len(sitems) >= 2][:10]

    # ── Stage 2: query-aware consolidation ──────────────────────────────────
    all_facts = []
    raw_text_parts = []
    for sid, sitems in relevant_sessions:
        s_text = format_session(sid, sitems)
        prompt = consolidation_prompt(s_text, question)
        response = _claude(client, prompt, max_tokens=800)
        writer.write(json.dumps({
            "question_id": q_id,
            "question": question,
            "gold": gold,
            "stage": "consolidate",
            "session_id": sid,
            "prompt": prompt,
            "response": response,
            "final_answer": None,
            "final_judge": None,
        }) + "\n")
        facts = [f.strip("- *•").strip() for f in response.split("\n") if len(f.strip()) > 10]
        all_facts.extend([f"[s{sid}] {f}" for f in facts])
        raw_text_parts.append(
            f"\n=== Session {sid} ===\n"
            + "\n".join(
                f"{getattr(i, 'speaker', '') or getattr(i, 'role', '')}: {i.content}"
                for i in sitems[:30]
            )
        )

    # ── Stage 3: final answer ────────────────────────────────────────────────
    evidence = (
        "DISTILLED FACTS:\n" + "\n".join(all_facts)
        + "\n\nRAW RELEVANT SESSIONS:\n" + "\n".join(raw_text_parts)
    )
    prompt = answer_prompt(evidence, question)
    final_answer = _claude(client, prompt, max_tokens=100)

    # Judge
    judge_score = llm_judge(client, final_answer, gold, question)

    writer.write(json.dumps({
        "question_id": q_id,
        "question": question,
        "gold": gold,
        "stage": "answer",
        "session_id": None,
        "prompt": prompt,
        "response": final_answer,
        "final_answer": final_answer,
        "final_judge": judge_score,
    }) + "\n")
    writer.flush()

    return final_answer, judge_score


def load_multi_session_examples(seed: int, n: int, exclude_ids: set | None = None):
    """Load multi-session LongMemEval examples deterministically."""
    from context_bench.datasets.memory.longmemeval import longmemeval
    all_examples = longmemeval(n=500, question_types=None)
    multi = [
        ex for ex in all_examples
        if any("multi-session" in q.query_type for q in ex.queries)
    ]
    rng = random.Random(seed)
    rng.shuffle(multi)
    if exclude_ids:
        multi = [ex for ex in multi if ex.id not in exclude_ids]
    return multi[:n]


def run(n_questions: int, output_path: str, eval_reserve: bool = True):
    # Build eval ID set first so train set never overlaps
    exclude_ids: set = set()
    if eval_reserve:
        eval_examples = load_multi_session_examples(EVAL_SEED, N_EVAL)
        exclude_ids = {ex.id for ex in eval_examples}
        print(f"[setup] reserved {len(exclude_ids)} eval IDs (seed={EVAL_SEED})", flush=True)

    train_examples = load_multi_session_examples(TRAIN_SEED, n_questions, exclude_ids)
    print(f"[setup] {len(train_examples)} train questions (seed={TRAIN_SEED})", flush=True)

    client = _bedrock_client()

    n_correct = 0
    with open(output_path, "w") as f:
        for i, ex in enumerate(train_examples):
            q = ex.queries[0]
            print(f"\n--- Q{i+1}/{len(train_examples)} id={ex.id} expected={q.answer[:60]!r}", flush=True)
            t0 = time.perf_counter()
            try:
                ans, judge = run_and_log(client, ex, q.question, q.answer, ex.id, f)
                elapsed = time.perf_counter() - t0
                n_correct += judge
                print(
                    f"  judge={judge} ans={ans[:60]!r} ({elapsed:.1f}s) "
                    f"running_acc={n_correct/(i+1):.3f}",
                    flush=True,
                )
            except Exception as e:
                print(f"  FAILED: {e}", flush=True)
                import traceback; traceback.print_exc()

    acc = n_correct / len(train_examples) if train_examples else 0
    print(f"\n[done] teacher_acc={acc:.3f} ({n_correct}/{len(train_examples)})", flush=True)
    print(f"[done] traces saved to {output_path}", flush=True)

    # Upload to S3
    s3_key = f"s3://{S3_BUCKET}/distill/raw_traces.jsonl"
    try:
        subprocess.run(["aws", "s3", "cp", output_path, s3_key], check=True, timeout=300)
        print(f"[s3] uploaded to {s3_key}", flush=True)
    except Exception as e:
        print(f"[s3] upload failed: {e}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--output", default="/tmp/distill_raw_traces.jsonl")
    p.add_argument("--no-eval-reserve", action="store_true",
                   help="Skip reserving eval IDs (for debugging)")
    args = p.parse_args()
    run(args.n, args.output, eval_reserve=not args.no_eval_reserve)
