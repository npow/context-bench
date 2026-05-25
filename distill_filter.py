"""Filter distillation trajectories: keep only questions where final_judge == 1.

Reads the raw JSONL from distill_data_gen.py, groups records by question_id,
drops trajectories where the final answer was judged incorrect, then writes
one SFT record (trl SFTTrainer messages format) per surviving (prompt, response)
pair.

Input JSONL (distill_raw_traces.jsonl):
  One record per LLM call, tagged by stage. See distill_data_gen.py for schema.

Output JSONL (distill_filtered_sft.jsonl):
  One record per surviving LLM call, in trl SFTTrainer format:
  {"messages": [{"role": "user", "content": <prompt>},
                {"role": "assistant", "content": <response>}]}

S3:
  Downloads: s3://<S3_BUCKET>/distill/raw_traces.jsonl
  Uploads:   s3://<S3_BUCKET>/distill/filtered_sft.jsonl
             s3://<S3_BUCKET>/distill/filter_stats.json
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys

S3_BUCKET = os.environ.get("S3_BUCKET", "netflix.pi.prod/users/npow/rlm_grpo")


def download_from_s3(s3_path: str, local_path: str) -> bool:
    try:
        subprocess.run(["aws", "s3", "cp", s3_path, local_path], check=True, timeout=300)
        print(f"[s3] downloaded {s3_path} -> {local_path}", flush=True)
        return True
    except Exception as e:
        print(f"[s3] download failed: {e}", flush=True)
        return False


def upload_to_s3(local_path: str, s3_path: str):
    try:
        subprocess.run(["aws", "s3", "cp", local_path, s3_path], check=True, timeout=300)
        print(f"[s3] uploaded {local_path} -> {s3_path}", flush=True)
    except Exception as e:
        print(f"[s3] upload failed: {e}", flush=True)


def filter_trajectories(input_path: str, output_path: str, stats_path: str):
    # ── Load all records ─────────────────────────────────────────────────────
    records: list[dict] = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] bad JSON line: {e}", flush=True)
    print(f"[filter] loaded {len(records)} raw records", flush=True)

    # ── Group by question_id ─────────────────────────────────────────────────
    by_qid: dict[str, list[dict]] = {}
    for rec in records:
        qid = rec.get("question_id", "unknown")
        by_qid.setdefault(qid, []).append(rec)

    n_total_qs = len(by_qid)
    print(f"[filter] {n_total_qs} unique questions", flush=True)

    # ── Find which questions have final_judge == 1 ───────────────────────────
    correct_qids: set[str] = set()
    for qid, recs in by_qid.items():
        for rec in recs:
            if rec.get("stage") == "answer" and rec.get("final_judge") == 1:
                correct_qids.add(qid)
                break

    n_correct = len(correct_qids)
    print(
        f"[filter] {n_correct}/{n_total_qs} questions had correct final answer "
        f"({n_correct/n_total_qs:.1%})",
        flush=True,
    )

    # ── Write filtered SFT records ───────────────────────────────────────────
    # For each correct question, emit one SFT record per (prompt, response) pair
    # across all 3 stages. The stage tag is appended to the system context so the
    # student learns which stage it is performing.
    stage_labels = {
        "relevance": "STAGE: Relevance gate. Decide YES/NO if a session is relevant.",
        "consolidate": "STAGE: Query-aware consolidation. Extract relevant facts.",
        "answer": "STAGE: Final answer. Answer the question from evidence.",
    }

    kept_records = 0
    stage_counts: dict[str, int] = {"relevance": 0, "consolidate": 0, "answer": 0}

    with open(output_path, "w") as out:
        for qid in sorted(correct_qids):
            for rec in by_qid[qid]:
                stage = rec.get("stage", "unknown")
                prompt = rec.get("prompt", "").strip()
                response = rec.get("response", "").strip()
                if not prompt or not response:
                    continue

                # Prepend a brief stage context line so the model knows what role it plays
                stage_hint = stage_labels.get(stage, f"STAGE: {stage}")
                full_prompt = f"{stage_hint}\n\n{prompt}"

                sft_record = {
                    "messages": [
                        {"role": "user", "content": full_prompt},
                        {"role": "assistant", "content": response},
                    ]
                }
                out.write(json.dumps(sft_record) + "\n")
                kept_records += 1
                if stage in stage_counts:
                    stage_counts[stage] += 1

    print(f"[filter] {kept_records} SFT records written to {output_path}", flush=True)
    print(f"[filter] stage breakdown: {stage_counts}", flush=True)

    # ── Stats ────────────────────────────────────────────────────────────────
    stats = {
        "n_total_questions": n_total_qs,
        "n_correct_questions": n_correct,
        "teacher_accuracy": n_correct / n_total_qs if n_total_qs else 0,
        "n_sft_records": kept_records,
        "stage_counts": stage_counts,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[filter] stats saved to {stats_path}", flush=True)
    print(f"[filter] teacher_accuracy={stats['teacher_accuracy']:.3f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=None, help="Local raw traces JSONL (downloads from S3 if absent)")
    p.add_argument("--output", default="/tmp/distill_filtered_sft.jsonl")
    p.add_argument("--stats", default="/tmp/distill_filter_stats.json")
    p.add_argument("--s3-bucket", default=None)
    args = p.parse_args()

    s3_bucket = args.s3_bucket or S3_BUCKET

    # Download input if not already local
    input_path = args.input or "/tmp/distill_raw_traces.jsonl"
    if not os.path.exists(input_path):
        s3_src = f"s3://{s3_bucket}/distill/raw_traces.jsonl"
        ok = download_from_s3(s3_src, input_path)
        if not ok:
            print(f"[error] Could not obtain input traces. Aborting.", flush=True)
            sys.exit(1)

    filter_trajectories(input_path, args.output, args.stats)

    # Upload outputs
    upload_to_s3(args.output, f"s3://{s3_bucket}/distill/filtered_sft.jsonl")
    upload_to_s3(args.stats, f"s3://{s3_bucket}/distill/filter_stats.json")


if __name__ == "__main__":
    main()
