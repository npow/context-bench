"""Diagnostic: WHY does long context fail?

Uses LongMemEval ground-truth `answer_session_ids` (in BenchmarkExample.metadata)
to identify evidence sessions — NOT heuristic keyword match.

Probes (all per-question):
1. Truncation curve. Haystack truncated at [5K, 15K, 30K, 60K, 200K] chars.
   For each length, record:
     - judge: model accuracy
     - evidence_present: are ALL evidence sessions wholly contained in the
       truncated haystack? (controls for "answer deleted" confound)
   Report accuracy SEPARATELY for evidence_present=True vs False, so the
   curve distinguishes "model can't attend" from "answer absent".

2. Gold-position bin. position = min(evidence_session_idx) / (n_sessions - 1).
   Bins: first_third, middle_third, last_third. Full-length accuracy only.

3. Distractor density. # total sessions binned <=10, 11-20, 21-30, >30.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, "src")

from context_bench.datasets.memory.longmemeval import longmemeval


def _bedrock_client():
    import boto3
    return boto3.client("bedrock-runtime", region_name="us-east-1")


def _claude_call(client, prompt: str, max_tokens: int = 200) -> str:
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


def format_haystack_get_session_spans(items):
    """Return (full_text, session_spans) where session_spans maps session_id -> (start_char, end_char) in the formatted text."""
    parts = []
    cur = None
    char_idx = 0
    session_spans = {}  # session_id -> [start_char, end_char]
    for item in items:
        sid = getattr(item, "session_id", "") or "default"
        if sid != cur:
            if cur is not None and cur in session_spans:
                session_spans[cur][1] = char_idx
            header = f"\n[Session {sid}]"
            parts.append(header)
            char_idx += len(header)
            session_spans.setdefault(sid, [char_idx, char_idx])
            cur = sid
        speaker = getattr(item, "speaker", "") or ""
        line = f"\n{speaker}: {item.content}"
        parts.append(line)
        char_idx += len(line)
    if cur is not None and cur in session_spans:
        session_spans[cur][1] = char_idx
    return "".join(parts), session_spans


def truncate(text: str, max_chars: int) -> str:
    if len(text) > max_chars:
        return text[:max_chars] + "\n[...truncated...]"
    return text


def answer_long_context(client, haystack: str, question: str) -> str:
    prompt = (
        "Answer the question precisely. If the answer is a number or count, give just the number. "
        "Be concise (under 15 words). Search the FULL conversation history carefully — the answer is in there.\n\n"
        f"CONVERSATION HISTORY:\n{haystack}\n\nQUESTION: {question}\n\nAnswer:"
    )
    return _claude_call(client, prompt, max_tokens=100)


def llm_judge(client, pred, gold, question) -> int:
    if not pred.strip():
        return 0
    prompt = (
        "Judge if the PREDICTION correctly answers the QUESTION given GOLD. Reply CORRECT or WRONG only.\n\n"
        f"QUESTION: {question}\nGOLD: {gold}\nPREDICTION: {pred}\n\nReply:"
    )
    v = _claude_call(client, prompt, max_tokens=20).upper()
    if "WRONG" in v or "INCORRECT" in v or "NOT CORRECT" in v:
        return 0
    return 1 if "CORRECT" in v else 0


def bootstrap_ci(scores, n=1000):
    if not scores:
        return 0.0, 0.0, 0.0
    rng = random.Random(42)
    n_s = len(scores)
    means = sorted(sum(rng.choice(scores) for _ in range(n_s)) / n_s for _ in range(n))
    return sum(scores) / n_s, means[int(n * 0.025)], means[int(n * 0.975)]


def evidence_fully_present(session_spans: dict, evidence_ids: list[str], max_chars: int) -> bool:
    """All evidence sessions must end at or before max_chars in the formatted haystack."""
    if not evidence_ids:
        return False
    for sid in evidence_ids:
        span = session_spans.get(sid)
        if span is None:
            return False  # session id not found in this example's haystack
        if span[1] > max_chars:
            return False
    return True


def distractor_metrics(session_order: list[str], evidence_ids: list[str], session_spans: dict) -> dict:
    """Distractor density between earliest and latest evidence sessions.
    Codex demand: we need to separate "evidence is at position X" from
    "evidence is surrounded by N distractors with M tokens."

    Returns:
      - intervening_distractors: # non-evidence sessions between earliest and latest evidence
      - evidence_token_ratio: chars in evidence sessions / chars in all kept sessions
    """
    if not evidence_ids:
        return {}
    idxs = sorted(session_order.index(sid) for sid in evidence_ids if sid in session_order)
    if not idxs:
        return {}
    earliest, latest = idxs[0], idxs[-1]
    intervening = sum(1 for sid in session_order[earliest+1:latest] if sid not in evidence_ids)

    # Total chars in evidence vs all
    evidence_chars = 0
    total_chars = 0
    for sid, span in session_spans.items():
        ln = span[1] - span[0]
        total_chars += ln
        if sid in evidence_ids:
            evidence_chars += ln
    ratio = evidence_chars / max(1, total_chars)
    return {
        "intervening_distractors": intervening,
        "evidence_token_ratio": ratio,
        "evidence_chars": evidence_chars,
        "total_session_chars": total_chars,
    }


def gold_position_metrics(session_order: list[str], evidence_ids: list[str]) -> dict:
    """Compute position metrics for evidence sessions in haystack order.

    Returns dict with:
      - position_bin: bin of earliest position (first/middle/last_third)
      - max_position_bin: bin of latest evidence position
      - span_width: max_pos - min_pos (in normalized 0-1 units)
      - earliest_position, latest_position: normalized [0,1]
    Codex flagged using only "earliest" as the bin — incomplete for multi-evidence.
    """
    if not evidence_ids:
        return {"position_bin": "unknown_no_metadata"}
    idxs = sorted(session_order.index(sid) for sid in evidence_ids if sid in session_order)
    if not idxs:
        return {"position_bin": "unknown_id_mismatch"}
    denom = max(1, len(session_order) - 1)
    earliest, latest = idxs[0], idxs[-1]
    earliest_p, latest_p = earliest / denom, latest / denom

    def _bin(p):
        if p < 1/3: return "first_third"
        if p < 2/3: return "middle_third"
        return "last_third"

    return {
        "position_bin": _bin(earliest_p),
        "max_position_bin": _bin(latest_p),
        "earliest_position": earliest_p,
        "latest_position": latest_p,
        "span_width": latest_p - earliest_p,
    }


def run(n_questions, output_path):
    client = _bedrock_client()
    all_examples = longmemeval(n=300, question_types=None)
    multi = [ex for ex in all_examples if any("multi-session" in q.query_type for q in ex.queries)][:n_questions]
    print(f"[setup] {len(multi)} multi-session questions", flush=True)

    # Lengths in chars. ~4 chars/token, so these correspond roughly to
    # 5K, 15K, 30K, 60K, 200K tokens — covering the full haystack at the
    # highest setting (max LME-S haystack ~115K tokens ~460K chars).
    lengths = [20_000, 60_000, 120_000, 240_000, 800_000]

    results = []

    for i, ex in enumerate(multi):
        q = ex.queries[0]
        gold = q.answer
        question = q.question

        full_text, session_spans = format_haystack_get_session_spans(ex.items)
        full_len = len(full_text)

        # Session order (in haystack order) — used for position bin
        seen = []
        for it in ex.items:
            sid = getattr(it, "session_id", "") or "default"
            if sid not in seen:
                seen.append(sid)

        evidence_ids = ex.metadata.get("answer_session_ids") or []
        n_sessions = len(seen)
        pos_metrics = gold_position_metrics(seen, evidence_ids)
        dist_metrics = distractor_metrics(seen, evidence_ids, session_spans)

        print(
            f"\n--- Q{i+1}/{len(multi)} id={ex.id} qtype={q.query_type} "
            f"full_len={full_len} pos_bin={pos_metrics.get('position_bin')} "
            f"span={pos_metrics.get('span_width', 0):.2f} "
            f"n_evidence={len(evidence_ids)}/{n_sessions} ---",
            flush=True,
        )

        record = {
            "qid": ex.id,
            "question": question,
            "gold": gold,
            "query_type": q.query_type,
            "full_len_chars": full_len,
            "evidence_ids": evidence_ids,
            "n_sessions": n_sessions,
            "n_evidence_sessions": len(evidence_ids),
            "gold_position_bin": pos_metrics.get("position_bin"),
            "gold_max_position_bin": pos_metrics.get("max_position_bin"),
            "evidence_span_width": pos_metrics.get("span_width"),
            "earliest_position": pos_metrics.get("earliest_position"),
            "latest_position": pos_metrics.get("latest_position"),
            "intervening_distractors": dist_metrics.get("intervening_distractors"),
            "evidence_token_ratio": dist_metrics.get("evidence_token_ratio"),
            "judgments": {},
        }

        for L in lengths:
            try:
                t0 = time.perf_counter()
                hay = truncate(full_text, L)
                evidence_present = evidence_fully_present(session_spans, evidence_ids, L) if evidence_ids else None
                ans = answer_long_context(client, hay, question)
                judge = llm_judge(client, ans, gold, question)
                dt = time.perf_counter() - t0
                print(
                    f"  L={L:>7d}  evidence_present={evidence_present}  judge={judge}  "
                    f"ans={ans[:50]!r}  ({dt:.1f}s)",
                    flush=True,
                )
                record["judgments"][str(L)] = {
                    "judge": judge,
                    "evidence_present": evidence_present,
                    "ans": ans,
                    "dt": dt,
                    "api_failed": False,
                }
            except Exception as e:
                # Codex flagged: API failures should track separately, not count as judge=0
                print(f"  L={L:>7d}  FAILED: {str(e)[:120]}", flush=True)
                record["judgments"][str(L)] = {
                    "judge": None, "ans": "", "error": str(e)[:200], "api_failed": True
                }

        results.append(record)

    # ===== Aggregate =====
    print("\n" + "=" * 80, flush=True)
    print(f"DIAGNOSTIC RESULTS (n={len(results)})", flush=True)
    print("=" * 80, flush=True)

    # Truncation curve — overall + split by evidence_present + STABLE SUBSET
    # Stable subset (Codex selection-bias fix): only include questions where
    # evidence_present=True at EVERY length L. This is the apples-to-apples
    # comparison for "model attention" across truncation lengths.
    print("\nAccuracy vs truncation length (overall):")
    overall = {str(L): [] for L in lengths}
    ev_present = {str(L): [] for L in lengths}
    ev_absent = {str(L): [] for L in lengths}
    api_failures = {str(L): 0 for L in lengths}
    for r in results:
        for L_str, jd in r["judgments"].items():
            if jd.get("api_failed"):
                api_failures[L_str] += 1
                continue
            j = jd.get("judge", 0)
            overall[L_str].append(j)
            ep = jd.get("evidence_present")
            if ep is True: ev_present[L_str].append(j)
            elif ep is False: ev_absent[L_str].append(j)

    # Stable subset: questions with evidence_present True at all lengths
    stable_qids = [
        r["qid"] for r in results
        if all(r["judgments"].get(str(L), {}).get("evidence_present") is True for L in lengths)
    ]
    print(f"  [stable subset: n={len(stable_qids)} questions with evidence present at ALL lengths]", flush=True)

    summary = {
        "truncation_overall": {}, "truncation_evidence_present": {},
        "truncation_evidence_absent": {}, "truncation_stable_subset": {},
        "api_failures_per_length": api_failures,
    }
    for L_str in [str(L) for L in lengths]:
        for label, bucket in [
            ("truncation_overall", overall),
            ("truncation_evidence_present", ev_present),
            ("truncation_evidence_absent", ev_absent),
        ]:
            scores = bucket[L_str]
            if scores:
                mean, lo, hi = bootstrap_ci(scores)
                summary[label][L_str] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(scores)}
        # Stable subset
        stable_scores = [
            r["judgments"][L_str]["judge"]
            for r in results
            if r["qid"] in set(stable_qids) and not r["judgments"][L_str].get("api_failed")
        ]
        if stable_scores:
            mean, lo, hi = bootstrap_ci(stable_scores)
            summary["truncation_stable_subset"][L_str] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(stable_scores)}

        m_ov = summary["truncation_overall"].get(L_str, {"mean": 0, "n": 0})
        m_ep = summary["truncation_evidence_present"].get(L_str, {"mean": 0, "n": 0})
        m_ea = summary["truncation_evidence_absent"].get(L_str, {"mean": 0, "n": 0})
        m_st = summary["truncation_stable_subset"].get(L_str, {"mean": 0, "n": 0})
        af = api_failures[L_str]
        print(
            f"  L={L_str:>7s}  overall={m_ov['mean']:.3f}(n={m_ov['n']})  "
            f"present={m_ep['mean']:.3f}(n={m_ep['n']})  "
            f"absent={m_ea['mean']:.3f}(n={m_ea['n']})  "
            f"stable={m_st['mean']:.3f}(n={m_st['n']})  "
            f"fail={af}",
            flush=True,
        )

    # Position bin — restrict to examples where evidence fully fits at max-L
    # (Otherwise it conflates "lost in middle" with "evidence past max-L").
    # Report BOTH earliest-position bin AND latest-position bin (Codex fix).
    print("\nAccuracy vs gold EARLIEST position (evidence_present=True at max L):")
    by_pos_min = defaultdict(list)
    by_pos_max = defaultdict(list)
    for r in results:
        jd = r["judgments"].get(str(lengths[-1]), {})
        if jd.get("evidence_present") is not True or jd.get("api_failed"):
            continue
        j = jd.get("judge", 0)
        if r.get("gold_position_bin"):
            by_pos_min[r["gold_position_bin"]].append(j)
        if r.get("gold_max_position_bin"):
            by_pos_max[r["gold_max_position_bin"]].append(j)
    summary["position_earliest"] = {}
    summary["position_latest"] = {}
    for k in ["first_third", "middle_third", "last_third", "unknown_no_metadata", "unknown_id_mismatch"]:
        scores = by_pos_min.get(k, [])
        if scores:
            mean, lo, hi = bootstrap_ci(scores)
            summary["position_earliest"][k] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(scores)}
            print(f"  earliest_{k:18s}  {mean:.3f} [{lo:.3f}, {hi:.3f}]  (n={len(scores)})", flush=True)
    print("\nAccuracy vs gold LATEST position (evidence_present=True at max L):")
    for k in ["first_third", "middle_third", "last_third"]:
        scores = by_pos_max.get(k, [])
        if scores:
            mean, lo, hi = bootstrap_ci(scores)
            summary["position_latest"][k] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(scores)}
            print(f"  latest_{k:18s}  {mean:.3f} [{lo:.3f}, {hi:.3f}]  (n={len(scores)})", flush=True)

    # Distractor density — restrict to evidence_present=True at max L
    print("\nAccuracy vs distractor density (evidence_present=True at max L only):")
    dist_bins = defaultdict(list)
    for r in results:
        jd_200k = r["judgments"].get(str(lengths[-1]), {})
        if jd_200k.get("evidence_present") is not True or jd_200k.get("api_failed"):
            continue
        full_judge = jd_200k.get("judge", 0)
        ns = r["n_sessions"]
        if ns <= 10: dist_bins["<=10"].append(full_judge)
        elif ns <= 20: dist_bins["11-20"].append(full_judge)
        elif ns <= 30: dist_bins["21-30"].append(full_judge)
        else: dist_bins[">30"].append(full_judge)
    summary["distractor"] = {}
    for k in ["<=10", "11-20", "21-30", ">30"]:
        scores = dist_bins.get(k, [])
        if scores:
            mean, lo, hi = bootstrap_ci(scores)
            summary["distractor"][k] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(scores)}
            print(f"  n_sessions={k:>8s}  {mean:.3f} [{lo:.3f}, {hi:.3f}]  (n={len(scores)})", flush=True)

    # Span-width analysis — does spread of evidence hurt accuracy?
    # Codex flagged: earliest/latest bins alone don't separate "lost in middle"
    # from "evidence distributed across haystack."
    print("\nAccuracy by evidence span width (evidence_present=True at max L):")
    by_span = defaultdict(list)
    for r in results:
        jd = r["judgments"].get(str(lengths[-1]), {})
        if jd.get("evidence_present") is not True or jd.get("api_failed"):
            continue
        sw = r.get("evidence_span_width")
        if sw is None: continue
        if sw == 0: by_span["0_single_session"].append(jd.get("judge", 0))
        elif sw < 0.25: by_span["0_to_0.25"].append(jd.get("judge", 0))
        elif sw < 0.5: by_span["0.25_to_0.5"].append(jd.get("judge", 0))
        else: by_span["over_0.5"].append(jd.get("judge", 0))
    summary["span_width"] = {}
    for k in ["0_single_session", "0_to_0.25", "0.25_to_0.5", "over_0.5"]:
        scores = by_span.get(k, [])
        if scores:
            mean, lo, hi = bootstrap_ci(scores)
            summary["span_width"][k] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(scores)}
            print(f"  span={k:18s}  {mean:.3f} [{lo:.3f}, {hi:.3f}]  (n={len(scores)})", flush=True)

    # Distractor density (Codex demand): intervening non-evidence sessions
    print("\nAccuracy by intervening_distractors (evidence_present=True at max L):")
    by_int = defaultdict(list)
    for r in results:
        jd = r["judgments"].get(str(lengths[-1]), {})
        if jd.get("evidence_present") is not True or jd.get("api_failed"): continue
        idist = r.get("intervening_distractors")
        if idist is None: continue
        if idist == 0: by_int["0_no_distractors"].append(jd.get("judge", 0))
        elif idist <= 3: by_int["1-3"].append(jd.get("judge", 0))
        elif idist <= 10: by_int["4-10"].append(jd.get("judge", 0))
        else: by_int["over_10"].append(jd.get("judge", 0))
    summary["intervening_distractors"] = {}
    for k in ["0_no_distractors", "1-3", "4-10", "over_10"]:
        scores = by_int.get(k, [])
        if scores:
            mean, lo, hi = bootstrap_ci(scores)
            summary["intervening_distractors"][k] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(scores)}
            print(f"  inter_dist={k:18s}  {mean:.3f} [{lo:.3f}, {hi:.3f}]  (n={len(scores)})", flush=True)

    # Evidence token ratio
    print("\nAccuracy by evidence_token_ratio (evidence_present=True at max L):")
    by_ratio = defaultdict(list)
    for r in results:
        jd = r["judgments"].get(str(lengths[-1]), {})
        if jd.get("evidence_present") is not True or jd.get("api_failed"): continue
        ratio = r.get("evidence_token_ratio")
        if ratio is None: continue
        if ratio < 0.05: by_ratio["under_5pct"].append(jd.get("judge", 0))
        elif ratio < 0.15: by_ratio["5-15pct"].append(jd.get("judge", 0))
        elif ratio < 0.30: by_ratio["15-30pct"].append(jd.get("judge", 0))
        else: by_ratio["over_30pct"].append(jd.get("judge", 0))
    summary["evidence_token_ratio"] = {}
    for k in ["under_5pct", "5-15pct", "15-30pct", "over_30pct"]:
        scores = by_ratio.get(k, [])
        if scores:
            mean, lo, hi = bootstrap_ci(scores)
            summary["evidence_token_ratio"][k] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(scores)}
            print(f"  ev_ratio={k:18s}  {mean:.3f} [{lo:.3f}, {hi:.3f}]  (n={len(scores)})", flush=True)

    # Stratified: position × multi-evidence
    print("\nAccuracy by earliest-position × n_evidence (single=1 vs multi≥2):")
    by_strat = defaultdict(list)
    for r in results:
        jd = r["judgments"].get(str(lengths[-1]), {})
        if jd.get("evidence_present") is not True or jd.get("api_failed"):
            continue
        pos = r.get("gold_position_bin")
        ne = r.get("n_evidence_sessions", 0)
        if not pos or pos.startswith("unknown"): continue
        evtype = "single" if ne == 1 else "multi"
        by_strat[(pos, evtype)].append(jd.get("judge", 0))
    summary["position_x_nevidence"] = {}
    for (pos, evtype), scores in sorted(by_strat.items()):
        if scores:
            mean, lo, hi = bootstrap_ci(scores)
            key = f"{pos}_{evtype}"
            summary["position_x_nevidence"][key] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(scores)}
            print(f"  {key:25s}  {mean:.3f} [{lo:.3f}, {hi:.3f}]  (n={len(scores)})", flush=True)

    # Single vs multi-evidence breakdown — control for question difficulty
    print("\nAccuracy by # evidence sessions (evidence_present=True at max L):")
    by_ev = defaultdict(list)
    for r in results:
        jd_200k = r["judgments"].get(str(lengths[-1]), {})
        if jd_200k.get("evidence_present") is not True or jd_200k.get("api_failed"):
            continue
        n_ev = r["n_evidence_sessions"]
        if n_ev <= 1: by_ev["single"].append(jd_200k.get("judge", 0))
        elif n_ev == 2: by_ev["two"].append(jd_200k.get("judge", 0))
        else: by_ev["3+"].append(jd_200k.get("judge", 0))
    summary["n_evidence"] = {}
    for k in ["single", "two", "3+"]:
        scores = by_ev.get(k, [])
        if scores:
            mean, lo, hi = bootstrap_ci(scores)
            summary["n_evidence"][k] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(scores)}
            print(f"  n_evidence={k:8s}  {mean:.3f} [{lo:.3f}, {hi:.3f}]  (n={len(scores)})", flush=True)

    out = {"n": len(results), "summary": summary, "details": results}
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {output_path}", flush=True)

    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        import subprocess
        s3_key = f"s3://{s3_bucket}/diagnostic/{os.path.basename(output_path)}"
        try:
            subprocess.run(["aws", "s3", "cp", output_path, s3_key], check=True, timeout=120)
            print(f"[s3] {s3_key}", flush=True)
        except Exception as e:
            print(f"[s3] {e}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--output", default="/tmp/diagnostic.json")
    args = p.parse_args()
    run(args.n, args.output)
