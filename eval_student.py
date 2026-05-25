"""Evaluate distilled Qwen-Coder-7B student on held-out 100 multi-session questions.

The student model replaces the 3 Bedrock Claude calls in the pipeline.
Judging uses Bedrock Claude (same as teacher) for consistent evaluation.

Metrics reported:
  - student_acc      (LLM-judge, same as teacher eval)
  - teacher_acc      (re-run teacher on same 100 Qs, or loaded from cache)
  - latency_ratio    student / teacher (seconds per question)
  - cost_ratio       student / teacher (estimated; teacher = Bedrock $/token)

Pipeline:
  1. Download LoRA adapter from S3
  2. Load Qwen-Coder-7B-Instruct + merge LoRA
  3. For each held-out question: run the 3-stage pipeline using LOCAL student model
  4. Judge each answer with Bedrock Sonnet
  5. Print comparison table

S3 paths:
  LoRA adapter:  s3://<S3_BUCKET>/sft_models/distill_qwen/
  Output JSON:   s3://<S3_BUCKET>/distill/eval_results.json
"""
from __future__ import annotations
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

S3_BUCKET = os.environ.get("S3_BUCKET", "netflix.pi.prod/users/npow/rlm_grpo")
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6")

# Must match distill_data_gen.py
EVAL_SEED = 137
N_EVAL = 100
TRAIN_SEED = 42
N_TRAIN_APPROX = 300   # used to compute cost ratio denominator

# Rough Bedrock Sonnet pricing (us-east-1, on-demand, 2025)
BEDROCK_INPUT_COST_PER_1K  = 0.003   # $/1K input tokens
BEDROCK_OUTPUT_COST_PER_1K = 0.015   # $/1K output tokens

# Approximate tokens per question for the 3-stage teacher pipeline
AVG_INPUT_TOKENS_TEACHER  = 12000
AVG_OUTPUT_TOKENS_TEACHER = 1200


def _install_deps():
    pkgs = ["trl>=0.12.0", "peft>=0.14.0", "accelerate>=1.3.0",
            "bitsandbytes>=0.45.0", "datasets", "boto3"]
    missing = []
    for spec in pkgs:
        name = spec.split(">=")[0].split("==")[0]
        try:
            __import__(name.replace("-", "_"))
        except ImportError:
            missing.append(spec)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)


_install_deps()

import torch


# ─── Bedrock helpers (for teacher + judge) ───────────────────────────────────

def _bedrock_client():
    import boto3
    return boto3.client("bedrock-runtime", region_name="us-east-1")


def _claude(client, prompt: str, max_tokens: int = 200, temperature: float = 0.3) -> str:
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    })
    r = client.invoke_model(
        body=body, modelId=BEDROCK_MODEL_ID,
        accept="application/json", contentType="application/json",
    )
    return json.loads(r["body"].read())["content"][0]["text"].strip()


def llm_judge(bedrock_client, pred: str, gold: str, question: str) -> int:
    if not pred.strip():
        return 0
    prompt = (
        "Judge if the PREDICTION correctly answers the QUESTION given GOLD. "
        "Reply CORRECT or WRONG only.\n\n"
        f"QUESTION: {question}\nGOLD: {gold}\nPREDICTION: {pred}\n\nReply:"
    )
    v = _claude(bedrock_client, prompt, max_tokens=10, temperature=0.0).upper()
    # FIX: INCORRECT contains CORRECT — check negatives first
    if "WRONG" in v or "INCORRECT" in v or "NOT CORRECT" in v: return 0
    return 1 if "CORRECT" in v else 0


# ─── Student model helpers ───────────────────────────────────────────────────

class StudentModel:
    """Wraps Qwen-Coder-7B-Instruct + merged LoRA for local inference."""

    def __init__(self, base_model_id: str, adapter_dir: str):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        print(f"[student] loading base model {base_model_id} (4-bit)...", flush=True)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"[student] loading LoRA adapter from {adapter_dir}...", flush=True)
        self.model = PeftModel.from_pretrained(base, adapter_dir)
        self.model.eval()
        print("[student] model ready.", flush=True)

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 200, max_input_tokens: int = 6000) -> str:
        msgs = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        # FIX: cap input length to fit on T4 14.5 GiB with 4-bit Qwen 3B
        # 6K input + 200 output tokens fits comfortably
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_input_tokens
        ).to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# ─── Pipeline helpers (shared geometry with distill_data_gen.py) ─────────────

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


STAGE_LABELS = {
    "relevance": "STAGE: Relevance gate. Decide YES/NO if a session is relevant.",
    "consolidate": "STAGE: Query-aware consolidation. Extract relevant facts.",
    "answer": "STAGE: Final answer. Answer the question from evidence.",
}


def run_student_pipeline(student: StudentModel, ex, question: str) -> str:
    """3-stage pipeline using the LOCAL student model for all three calls."""
    sessions = group_by_session(ex.items)

    # Stage 1: relevance gate
    relevant_sessions = []
    for sid, sitems in sessions.items():
        if len(sitems) < 2:
            continue
        s_text = format_session(sid, sitems)
        prompt = (
            f"{STAGE_LABELS['relevance']}\n\n"
            "Determine whether the session below contains ANY information that could help answer "
            f"the question. Respond with exactly one word: YES or NO.\n\n"
            f"QUESTION: {question}\n\nSESSION:\n{s_text[:8000]}\n\nRespond YES or NO:"
        )
        resp = student.generate(prompt, max_new_tokens=10)
        if "YES" in resp.strip().upper():
            relevant_sessions.append((sid, sitems))

    if not relevant_sessions:
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

    # Stage 2: consolidation
    all_facts = []
    raw_text_parts = []
    for sid, sitems in relevant_sessions:
        s_text = format_session(sid, sitems)
        prompt = (
            f"{STAGE_LABELS['consolidate']}\n\n"
            "Extract ALL facts from this session that could help answer the question. "
            "Be specific (names, numbers, dates, places). One fact per line, no markdown.\n\n"
            f"QUESTION: {question}\n\nSESSION:\n{s_text[:10000]}\n\nRelevant facts (one per line):"
        )
        resp = student.generate(prompt, max_new_tokens=800)
        facts = [f.strip("- *•").strip() for f in resp.split("\n") if len(f.strip()) > 10]
        all_facts.extend([f"[s{sid}] {f}" for f in facts])
        raw_text_parts.append(
            f"\n=== Session {sid} ===\n"
            + "\n".join(
                f"{getattr(i, 'speaker', '') or getattr(i, 'role', '')}: {i.content}"
                for i in sitems[:30]
            )
        )

    # Stage 3: answer
    evidence = (
        "DISTILLED FACTS:\n" + "\n".join(all_facts)
        + "\n\nRAW RELEVANT SESSIONS:\n" + "\n".join(raw_text_parts)
    )
    if len(evidence) > 60000:
        evidence = evidence[:60000] + "\n[...truncated...]"
    prompt = (
        f"{STAGE_LABELS['answer']}\n\n"
        "Answer the question precisely. If the answer is a number, give just the number. "
        "If it's a list, give the count. Be concise (under 15 words).\n\n"
        f"EVIDENCE:\n{evidence}\n\nQUESTION: {question}\n\nAnswer:"
    )
    return student.generate(prompt, max_new_tokens=100)


def run_teacher_pipeline(bedrock_client, ex, question: str) -> str:
    """Re-run teacher on same question for latency/cost comparison."""
    from context_bench.datasets.memory.longmemeval import longmemeval  # noqa

    sessions = group_by_session(ex.items)

    relevant_sessions = []
    for sid, sitems in sessions.items():
        if len(sitems) < 2:
            continue
        s_text = format_session(sid, sitems)
        prompt = (
            "Determine whether the session below contains ANY information that could help answer "
            f"the question. Respond with exactly one word: YES or NO.\n\n"
            f"QUESTION: {question}\n\nSESSION:\n{s_text[:8000]}\n\nRespond YES or NO:"
        )
        resp = _claude(bedrock_client, prompt, max_tokens=10)
        if "YES" in resp.strip().upper():
            relevant_sessions.append((sid, sitems))

    if not relevant_sessions:
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

    all_facts = []
    raw_text_parts = []
    for sid, sitems in relevant_sessions:
        s_text = format_session(sid, sitems)
        prompt = (
            "You are summarizing a conversation session for the purpose of answering a specific "
            "question. Extract ALL facts from this session that could help answer the question. "
            "Be specific (names, numbers, dates, places). One fact per line, no markdown.\n\n"
            f"QUESTION: {question}\n\nSESSION:\n{s_text[:10000]}\n\nRelevant facts (one per line):"
        )
        resp = _claude(bedrock_client, prompt, max_tokens=800)
        facts = [f.strip("- *•").strip() for f in resp.split("\n") if len(f.strip()) > 10]
        all_facts.extend([f"[s{sid}] {f}" for f in facts])
        raw_text_parts.append(
            f"\n=== Session {sid} ===\n"
            + "\n".join(
                f"{getattr(i, 'speaker', '') or getattr(i, 'role', '')}: {i.content}"
                for i in sitems[:30]
            )
        )

    evidence = (
        "DISTILLED FACTS:\n" + "\n".join(all_facts)
        + "\n\nRAW RELEVANT SESSIONS:\n" + "\n".join(raw_text_parts)
    )
    if len(evidence) > 60000:
        evidence = evidence[:60000] + "\n[...truncated...]"
    prompt = (
        "Answer the question precisely. If the answer is a number, give just the number. "
        "If it's a list, give the count. Be concise (under 15 words).\n\n"
        f"EVIDENCE:\n{evidence}\n\nQUESTION: {question}\n\nAnswer:"
    )
    return _claude(bedrock_client, prompt, max_tokens=100)


# ─── Dataset loading (mirrors distill_data_gen.py) ───────────────────────────

def load_eval_examples() -> list:
    from context_bench.datasets.memory.longmemeval import longmemeval
    all_examples = longmemeval(n=500, question_types=None)
    multi = [
        ex for ex in all_examples
        if any("multi-session" in q.query_type for q in ex.queries)
    ]
    rng = random.Random(EVAL_SEED)
    rng.shuffle(multi)
    return multi[:N_EVAL]


# ─── Main evaluation loop ─────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-dir", default="/tmp/distill_adapter",
                   help="Local path for LoRA adapter (downloaded from S3 if absent)")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    p.add_argument("--output", default="/tmp/distill_eval_results.json")
    p.add_argument("--s3-bucket", default=None)
    p.add_argument("--skip-teacher", action="store_true",
                   help="Skip teacher re-run (uses 0.0 teacher acc for ratio reporting)")
    args = p.parse_args()

    s3_bucket = args.s3_bucket or S3_BUCKET
    s3_adapter = f"s3://{s3_bucket}/sft_models/distill_qwen/"

    # Download LoRA adapter
    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.exists() or not any(adapter_dir.iterdir()):
        adapter_dir.mkdir(parents=True, exist_ok=True)
        print(f"[setup] downloading LoRA adapter from {s3_adapter}...", flush=True)
        subprocess.run(
            ["aws", "s3", "cp", s3_adapter, str(adapter_dir), "--recursive"],
            check=True, timeout=600,
        )
    else:
        print(f"[setup] using cached adapter at {adapter_dir}", flush=True)

    # Load models
    student = StudentModel(args.model_id, str(adapter_dir))
    bedrock = _bedrock_client()

    # Load eval examples
    eval_examples = load_eval_examples()
    print(f"[eval] {len(eval_examples)} held-out questions (seed={EVAL_SEED})", flush=True)

    student_results = []
    teacher_results = []

    for i, ex in enumerate(eval_examples):
        q = ex.queries[0]
        print(f"\n--- Q{i+1}/{len(eval_examples)} id={ex.id}", flush=True)

        # Student
        t0 = time.perf_counter()
        try:
            student_ans = run_student_pipeline(student, ex, q.question)
        except Exception as e:
            print(f"  [student] FAILED: {e}", flush=True)
            student_ans = ""
        student_latency = time.perf_counter() - t0
        student_judge = llm_judge(bedrock, student_ans, q.answer, q.question)
        print(f"  [student] judge={student_judge} ans={student_ans[:60]!r} ({student_latency:.1f}s)", flush=True)
        student_results.append({
            "id": ex.id, "question": q.question, "gold": q.answer,
            "answer": student_ans, "judge": student_judge, "latency_s": student_latency,
        })

        # Teacher (optional — Bedrock cost, but makes paper comparison valid)
        if not args.skip_teacher:
            t0 = time.perf_counter()
            try:
                teacher_ans = run_teacher_pipeline(bedrock, ex, q.question)
            except Exception as e:
                print(f"  [teacher] FAILED: {e}", flush=True)
                teacher_ans = ""
            teacher_latency = time.perf_counter() - t0
            teacher_judge = llm_judge(bedrock, teacher_ans, q.answer, q.question)
            print(f"  [teacher] judge={teacher_judge} ans={teacher_ans[:60]!r} ({teacher_latency:.1f}s)", flush=True)
            teacher_results.append({
                "id": ex.id, "question": q.question, "gold": q.answer,
                "answer": teacher_ans, "judge": teacher_judge, "latency_s": teacher_latency,
            })

    # ── Compute metrics ───────────────────────────────────────────────────────
    n = len(student_results)
    student_acc = sum(r["judge"] for r in student_results) / n if n else 0
    student_avg_latency = sum(r["latency_s"] for r in student_results) / n if n else 0

    if teacher_results:
        teacher_acc = sum(r["judge"] for r in teacher_results) / len(teacher_results)
        teacher_avg_latency = sum(r["latency_s"] for r in teacher_results) / len(teacher_results)
    else:
        teacher_acc = None
        teacher_avg_latency = None

    # Cost estimation
    # Teacher: ~AVG_INPUT_TOKENS_TEACHER input + AVG_OUTPUT_TOKENS_TEACHER output per question
    teacher_cost_per_q = (
        AVG_INPUT_TOKENS_TEACHER  / 1000 * BEDROCK_INPUT_COST_PER_1K +
        AVG_OUTPUT_TOKENS_TEACHER / 1000 * BEDROCK_OUTPUT_COST_PER_1K
    )
    # Student: GPU compute (A10G spot ~$1.50/h), ~5s/q => ~0.000208 $/q
    # (versus Bedrock ~$0.054/q for 12K input + 1.2K output at Sonnet pricing)
    STUDENT_GPU_COST_PER_HOUR = 1.50  # USD, approx A10G spot
    student_cost_per_q = (student_avg_latency / 3600) * STUDENT_GPU_COST_PER_HOUR

    cost_ratio = student_cost_per_q / teacher_cost_per_q if teacher_cost_per_q > 0 else None
    latency_ratio = (student_avg_latency / teacher_avg_latency
                     if teacher_avg_latency and teacher_avg_latency > 0 else None)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("DISTILLATION EVAL RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(f"  n_questions    : {n}", flush=True)
    print(f"  student_acc    : {student_acc:.3f}", flush=True)
    if teacher_acc is not None:
        print(f"  teacher_acc    : {teacher_acc:.3f}", flush=True)
        print(f"  acc_gap        : {student_acc - teacher_acc:+.3f}", flush=True)
    if latency_ratio is not None:
        print(f"  latency_ratio  : {latency_ratio:.2f}x (student/teacher)", flush=True)
    if cost_ratio is not None:
        print(f"  cost_ratio     : {cost_ratio:.3f}x (student/teacher, est.)", flush=True)
        print(f"  cost_reduction : {1/cost_ratio:.0f}x cheaper than teacher", flush=True)
    print(f"  student_cost/q : ${student_cost_per_q:.5f}", flush=True)
    print(f"  teacher_cost/q : ${teacher_cost_per_q:.5f}", flush=True)
    print("=" * 70, flush=True)

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "n": n,
        "student_acc": student_acc,
        "teacher_acc": teacher_acc,
        "acc_gap": (student_acc - teacher_acc) if teacher_acc is not None else None,
        "student_avg_latency_s": student_avg_latency,
        "teacher_avg_latency_s": teacher_avg_latency,
        "latency_ratio": latency_ratio,
        "student_cost_per_q_usd": student_cost_per_q,
        "teacher_cost_per_q_usd": teacher_cost_per_q,
        "cost_ratio": cost_ratio,
        "cost_reduction_factor": 1/cost_ratio if cost_ratio else None,
        "student_details": student_results,
        "teacher_details": teacher_results,
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] saved to {args.output}", flush=True)

    # Upload to S3
    s3_out = f"s3://{s3_bucket}/distill/eval_results.json"
    try:
        subprocess.run(["aws", "s3", "cp", args.output, s3_out], check=True, timeout=120)
        print(f"[s3] uploaded to {s3_out}", flush=True)
    except Exception as e:
        print(f"[s3] upload failed: {e}", flush=True)


if __name__ == "__main__":
    main()
