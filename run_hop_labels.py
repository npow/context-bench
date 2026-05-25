"""Hop-stratified labeler for LongMemEval-S.

For every question, label:
  - n_evidence_sessions  (from dataset metadata, but verified by LLM)
  - inferred_hop_count   (LLM: how many independent facts must be combined to answer?)
  - span_width           (computed from session positions)
  - intervening_distractors (computed)
  - temporal_flag        (LLM: does Q require temporal reasoning?)
  - update_flag          (LLM: does Q involve updated/contradicted info?)
  - question_class       (single_fact / lookup / count / multi_hop / temporal / abstain)

Output JSONL. Use Sonnet 4.6 for labeling.
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


def _bedrock():
    import boto3
    return boto3.client("bedrock-runtime", region_name="us-east-1")


def _claude(client, prompt, max_tokens=200):
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    r = client.invoke_model(
        body=json.dumps(body),
        modelId="us.anthropic.claude-sonnet-4-6",
        accept="application/json", contentType="application/json",
    )
    return json.loads(r["body"].read())["content"][0]["text"].strip()


def label_question(client, ex, q) -> dict:
    evidence_ids = ex.metadata.get("answer_session_ids") or []

    # Build session order (haystack order)
    seen = []
    for it in ex.items:
        sid = getattr(it, "session_id", None) or "default"
        if sid not in seen:
            seen.append(sid)
    n_sessions = len(seen)

    # Compute structural metrics from metadata
    if evidence_ids:
        idxs = sorted(seen.index(sid) for sid in evidence_ids if sid in seen)
        if idxs:
            earliest, latest = idxs[0], idxs[-1]
            span_width = (latest - earliest) / max(1, n_sessions - 1)
            intervening_distractors = sum(1 for sid in seen[earliest+1:latest] if sid not in evidence_ids)
        else:
            span_width = None
            intervening_distractors = None
    else:
        span_width = None
        intervening_distractors = None

    # LLM labeling: hop count, temporal flag, update flag, class
    prompt = (
        "Analyze the QUESTION and GOLD ANSWER. Output JSON with these fields:\n"
        "  - hop_count (int): how many independent facts must be retrieved and combined to answer? "
        "1 = single lookup; 2 = combine two facts; 3+ = synthesize multiple facts.\n"
        "  - is_temporal (bool): does answering require reasoning about time, dates, durations, "
        "or recency comparisons?\n"
        "  - is_update (bool): does the question involve updated, changed, or contradicted info "
        "across sessions (e.g., 'most recent X', 'current X')?\n"
        "  - question_class (str): one of {single_fact, count, comparison, multi_hop, temporal, abstain, other}.\n"
        "  - rationale (str, max 40 chars): brief justification.\n\n"
        f"QUESTION: {q.question}\nGOLD ANSWER: {q.answer}\nQUERY_TYPE: {q.query_type}\n\n"
        "Output ONLY a JSON object on a single line."
    )
    raw = _claude(client, prompt, max_tokens=150)
    # Extract JSON
    try:
        # Find first { and last }
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            labels = json.loads(raw[start:end])
        else:
            labels = {}
    except Exception:
        labels = {"parse_error": True, "raw": raw[:100]}

    return {
        "qid": ex.id,
        "question": q.question,
        "gold": q.answer,
        "query_type": q.query_type,
        "n_evidence_sessions": len(evidence_ids),
        "n_sessions_total": n_sessions,
        "span_width": span_width,
        "intervening_distractors": intervening_distractors,
        "evidence_ids": evidence_ids,
        **labels,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--output", default="/tmp/hop_labels.jsonl")
    args = p.parse_args()

    client = _bedrock()
    examples = longmemeval(n=args.n)
    print(f"[setup] labeling {len(examples)} examples", flush=True)

    out_f = open(args.output, "w")
    n_labeled = 0
    by_hop = defaultdict(int)
    by_class = defaultdict(int)

    for i, ex in enumerate(examples):
        for q in ex.queries:
            t0 = time.perf_counter()
            try:
                lab = label_question(client, ex, q)
            except Exception as e:
                lab = {"qid": ex.id, "error": str(e)[:200]}
            dt = time.perf_counter() - t0
            out_f.write(json.dumps(lab) + "\n")
            out_f.flush()
            n_labeled += 1
            if "hop_count" in lab:
                by_hop[lab["hop_count"]] += 1
            if "question_class" in lab:
                by_class[lab["question_class"]] += 1
            if n_labeled % 25 == 0:
                print(
                    f"  [{n_labeled}/{len(examples)}] hop={lab.get('hop_count','?')} "
                    f"cls={lab.get('question_class','?')} ({dt:.1f}s)",
                    flush=True,
                )
    out_f.close()

    print("\n" + "=" * 80, flush=True)
    print(f"HOP LABELS SUMMARY (n={n_labeled})", flush=True)
    print("=" * 80, flush=True)
    print("By hop_count:", dict(sorted(by_hop.items())), flush=True)
    print("By question_class:", dict(by_class), flush=True)

    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        import subprocess
        s3_key = f"s3://{s3_bucket}/hop_labels/{os.path.basename(args.output)}"
        try:
            subprocess.run(["aws", "s3", "cp", args.output, s3_key], check=True, timeout=120)
            print(f"[s3] {s3_key}", flush=True)
        except Exception as e:
            print(f"[s3] {e}", flush=True)


if __name__ == "__main__":
    main()
