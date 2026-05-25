"""Map-reduce variant grid: isolate WHY facts_only beats map_reduce.

9 variants on n=50 multi-session LongMemEval questions:

  1. mr_no_query          — map_reduce baseline: query-AGNOSTIC "3 key facts" per session, no gate
  2. mr_query_aware       — query-aware facts per session, no gate  (facts_only minus the gate)
  3. mr_gate_no_query     — gate + query-AGNOSTIC "3 key facts"   (gate without query awareness)
  4. mr_gate_query        — gate + query-aware facts               (= facts_only, best system)
  5. mr_shuffled_query    — like (4) but extract facts with a RANDOM question (sanity check)
  6. mr_budget_matched    — facts_only but capped at 3 facts per session (budget-matched to (1))
  7. mr_oracle_relevance  — gold-answer keywords filter sessions, then query-aware facts (upper bound)
  8. mr_random_gate       — random 5 sessions kept, then query-aware facts (negative gate control)
  9. long_context_strong  — full haystack + strong CoT prompt (fixes weak sonnet_200k baseline)

Factors isolated:
  (2) vs (1): query-awareness alone
  (3) vs (1): gate alone
  (4) vs (2): gate contribution on top of query-awareness
  (4) vs (3): query-awareness on top of gate
  (5) vs (4): does shuffled question break query-aware lift? (should drop if lift is real)
  (6) vs (4): is the improvement just from having MORE facts (no budget cap)?
  (7) vs (4): how far are we from an oracle that always finds the right sessions?
  (8) vs (4): does a random gate hurt? (sanity: random gate should underperform learned gate)
  (9) vs long_context_full: is long-context weak because of the prompt, not the architecture?

LLM judge: FIXED — check WRONG/INCORRECT/NOT CORRECT FIRST, then CORRECT.
Bootstrap 95% CIs over n questions.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, "src")

from context_bench.datasets.memory.longmemeval import longmemeval


# ---------------------------------------------------------------------------
# Bedrock plumbing — verbatim from run_strong_pipeline.py
# ---------------------------------------------------------------------------

def _bedrock_client():
    import boto3
    return boto3.client("bedrock-runtime", region_name="us-east-1")


def _claude(client, prompt: str, max_tokens: int = 200, temperature: float = 0.3) -> str:
    """Model-agnostic Bedrock call — dispatches on BEDROCK_MODEL_ID."""
    import re as _re
    model_id = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-sonnet-20241022-v2:0")

    if "anthropic" in model_id:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        })
        r = client.invoke_model(body=body, modelId=model_id, accept="application/json", contentType="application/json")
        return json.loads(r["body"].read())["content"][0]["text"].strip()

    if "openai" in model_id or "gpt" in model_id:
        body = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max(max_tokens * 3, 400),
        })
        r = client.invoke_model(body=body, modelId=model_id)
        text = json.loads(r["body"].read())["choices"][0]["message"]["content"]
        return _re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=_re.DOTALL).strip()

    if "gemma" in model_id or "google" in model_id:
        body = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        r = client.invoke_model(body=body, modelId=model_id)
        resp = json.loads(r["body"].read())
        if "choices" in resp:
            return resp["choices"][0]["message"]["content"].strip()
        if "generation" in resp:
            return resp["generation"].strip()
        return str(resp).strip()

    raise ValueError(f"Unknown model: {model_id}")


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def group_by_session(items):
    sessions = {}
    for item in items:
        sid = getattr(item, "session_id", None) or "default"
        sessions.setdefault(sid, []).append(item)
    return sessions


def format_session(sid, sitems, max_chars=10000):
    text = f"[Session {sid}]\n" + "\n".join(
        f"{getattr(i, 'speaker', '')}: {i.content}" for i in sitems[:50]
    )
    return text[:max_chars]


def format_haystack(items, max_chars=190000):
    parts = []
    cur = None
    for item in items:
        sid = getattr(item, "session_id", "") or "default"
        if sid != cur:
            parts.append(f"\n[Session {sid}]")
            cur = sid
        parts.append(f"{getattr(item, 'speaker', '')}: {item.content}")
    return "\n".join(parts)[:max_chars]


def answer_with_evidence(client, evidence: str, question: str) -> str:
    if len(evidence) > 60000:
        evidence = evidence[:60000] + "\n[...truncated...]"
    prompt = (
        "Answer the question precisely. If the answer is a number, give just the number. "
        "If it's a list, give the count. Be concise (under 15 words).\n\n"
        f"EVIDENCE:\n{evidence}\n\nQUESTION: {question}\n\nAnswer:"
    )
    return _claude(client, prompt, max_tokens=100)


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def is_session_relevant(client, session_text: str, question: str) -> bool:
    """Stage 1 relevance gate — identical to run_strong_pipeline.py."""
    prompt = (
        "Determine whether the session below contains ANY information that could help answer "
        f"the question. Respond with exactly one word: YES or NO.\n\n"
        f"QUESTION: {question}\n\n"
        f"SESSION:\n{session_text[:8000]}\n\n"
        "Respond YES or NO:"
    )
    verdict = _claude(client, prompt, max_tokens=10).strip().upper()
    return "YES" in verdict


def _relevance_gate(client, sessions: dict, question: str):
    """Return list of (sid, sitems) that pass the relevance gate.
    Mirrors the exact fallback logic from run_strong_pipeline.py."""
    relevant = []
    for sid, sitems in sessions.items():
        if len(sitems) < 2:
            continue
        s_text = format_session(sid, sitems)
        if is_session_relevant(client, s_text, question):
            relevant.append((sid, sitems))

    if not relevant:
        q_words = {w.lower() for w in question.split() if len(w) > 3}
        if q_words:
            scored = [
                (sid, sitems, sum(1 for s in sitems for w in q_words if w in s.content.lower()))
                for sid, sitems in sessions.items() if len(sitems) >= 2
            ]
            scored.sort(key=lambda x: -x[2])
            relevant = [(sid, sitems) for sid, sitems, score in scored[:5] if score > 0]
        if not relevant:
            relevant = [(sid, sitems) for sid, sitems in sessions.items() if len(sitems) >= 2][:10]

    return relevant


def query_aware_facts(client, session_text: str, question: str) -> list[str]:
    """Extract facts relevant to the question — query-AWARE."""
    prompt = (
        "You are summarizing a conversation session for the purpose of answering a specific question. "
        "Extract ALL facts from this session that could help answer the question. "
        "Be specific (names, numbers, dates, places). One fact per line, no markdown.\n\n"
        f"QUESTION: {question}\n\nSESSION:\n{session_text[:10000]}\n\n"
        "Relevant facts (one per line):"
    )
    response = _claude(client, prompt, max_tokens=800)
    return [f.strip("- *•").strip() for f in response.split("\n") if len(f.strip()) > 10]


def query_agnostic_facts(client, session_text: str, k: int = 3) -> list[str]:
    """Extract k key facts — query-AGNOSTIC."""
    prompt = (
        f"Extract the {k} most important facts from this conversation session. "
        f"Output one fact per line. Each fact should be a complete sentence.\n\n"
        f"SESSION:\n{session_text[:8000]}\n\nKey facts:"
    )
    response = _claude(client, prompt, max_tokens=300)
    facts = [f.strip("- *•").strip() for f in response.split("\n") if len(f.strip()) > 10]
    return facts[:k]


# ---------------------------------------------------------------------------
# Metrics — FIXED judge (negatives first, per run_strong_pipeline.py)
# ---------------------------------------------------------------------------

def llm_judge(client, pred: str, gold: str, question: str) -> int:
    if not pred.strip():
        return 0
    prompt = (
        "Judge if the PREDICTION correctly answers the QUESTION given GOLD. Reply CORRECT or WRONG only.\n\n"
        f"QUESTION: {question}\nGOLD: {gold}\nPREDICTION: {pred}\n\nReply:"
    )
    v = _claude(client, prompt, max_tokens=20, temperature=0.0).upper()
    # CRITICAL: check negatives first — "CORRECT" is a substring of "INCORRECT"
    if "WRONG" in v or "INCORRECT" in v or "NOT CORRECT" in v:
        return 0
    if "CORRECT" in v:
        return 1
    return 0


def bootstrap_ci(scores, n=1000):
    rng = random.Random(42)
    n_s = len(scores)
    if n_s == 0:
        return 0, 0, 0
    means = sorted(
        sum(rng.choice(scores) for _ in range(n_s)) / n_s for _ in range(n)
    )
    return sum(scores) / n_s, means[int(n * 0.025)], means[int(n * 0.975)]


# ---------------------------------------------------------------------------
# Variant 1: mr_no_query  (= map_reduce baseline)
# Per-session query-AGNOSTIC "give 3 key facts", no gate, concat, answer.
# Matches map_reduce from run_baselines_ablations.py.
# ---------------------------------------------------------------------------

def mr_no_query(client, items, question):
    sessions = group_by_session(items)
    all_facts = []
    for sid, sitems in sessions.items():
        if len(sitems) < 2:
            continue
        s_text = format_session(sid, sitems)
        facts = query_agnostic_facts(client, s_text, k=3)
        all_facts.extend([f"[s{sid}] {f}" for f in facts])
    evidence = "\n".join(all_facts)
    return answer_with_evidence(client, evidence, question)


# ---------------------------------------------------------------------------
# Variant 2: mr_query_aware  (NO gate)
# Per-session query-aware fact extraction, no gate, concat, answer.
# = facts_only minus the gate stage.
# Isolates: query-awareness alone (compare to variant 1).
# ---------------------------------------------------------------------------

def mr_query_aware(client, items, question):
    sessions = group_by_session(items)
    all_facts = []
    for sid, sitems in sessions.items():
        if len(sitems) < 2:
            continue
        s_text = format_session(sid, sitems)
        facts = query_aware_facts(client, s_text, question)
        all_facts.extend([f"[s{sid}] {f}" for f in facts])
    evidence = "DISTILLED FACTS:\n" + "\n".join(all_facts)
    return answer_with_evidence(client, evidence, question)


# ---------------------------------------------------------------------------
# Variant 3: mr_gate_no_query  (gate + query-agnostic facts)
# Stage 1 relevance gate, then per-session "give 3 key facts" (query-agnostic).
# Isolates: gate alone (compare to variant 1).
# ---------------------------------------------------------------------------

def mr_gate_no_query(client, items, question):
    sessions = group_by_session(items)
    relevant = _relevance_gate(client, sessions, question)
    all_facts = []
    for sid, sitems in relevant:
        s_text = format_session(sid, sitems)
        facts = query_agnostic_facts(client, s_text, k=3)
        all_facts.extend([f"[s{sid}] {f}" for f in facts])
    evidence = "\n".join(all_facts)
    return answer_with_evidence(client, evidence, question)


# ---------------------------------------------------------------------------
# Variant 4: mr_gate_query  (= facts_only, the best system)
# Stage 1 gate + per-session query-aware facts, answer from facts only.
# Reproduces facts_only from run_baselines_ablations.py exactly.
# Isolates: full system — both gate AND query-awareness.
# ---------------------------------------------------------------------------

def mr_gate_query(client, items, question):
    sessions = group_by_session(items)
    relevant = _relevance_gate(client, sessions, question)
    all_facts = []
    for sid, sitems in relevant:
        s_text = format_session(sid, sitems)
        facts = query_aware_facts(client, s_text, question)
        all_facts.extend([f"[s{sid}] {f}" for f in facts])
    evidence = "DISTILLED FACTS:\n" + "\n".join(all_facts)
    return answer_with_evidence(client, evidence, question)


# ---------------------------------------------------------------------------
# Variant 5: mr_shuffled_query  (sanity check)
# Like variant 4 but pass a RANDOM question for fact extraction, then answer
# the original question.  If query-awareness is the source of the lift, this
# should score significantly worse than variant 4.
# ---------------------------------------------------------------------------

def mr_shuffled_query(client, items, question, all_questions: list[str]):
    """Pass a random decoy question during fact extraction, answer with the real one."""
    decoy = random.choice([q for q in all_questions if q != question] or all_questions)
    sessions = group_by_session(items)
    relevant = _relevance_gate(client, sessions, question)  # gate with real question
    all_facts = []
    for sid, sitems in relevant:
        s_text = format_session(sid, sitems)
        facts = query_aware_facts(client, s_text, decoy)  # extract with decoy question
        all_facts.extend([f"[s{sid}] {f}" for f in facts])
    evidence = "DISTILLED FACTS:\n" + "\n".join(all_facts)
    return answer_with_evidence(client, evidence, question)  # answer real question


# ---------------------------------------------------------------------------
# Variant 6: mr_budget_matched  (budget control)
# facts_only but cap at 3 facts per session (matches map_reduce's budget).
# Tests whether improvement over map_reduce is simply from extracting MORE facts.
# ---------------------------------------------------------------------------

def mr_budget_matched(client, items, question):
    sessions = group_by_session(items)
    relevant = _relevance_gate(client, sessions, question)
    all_facts = []
    for sid, sitems in relevant:
        s_text = format_session(sid, sitems)
        facts = query_aware_facts(client, s_text, question)
        all_facts.extend([f"[s{sid}] {f}" for f in facts[:3]])  # cap at 3 per session
    evidence = "DISTILLED FACTS:\n" + "\n".join(all_facts)
    return answer_with_evidence(client, evidence, question)


# ---------------------------------------------------------------------------
# Variant 7: mr_oracle_relevance  (upper bound — CHEATS using answer)
# Use gold-answer keywords to filter sessions, then query-aware facts.
# Intentional cheat: shows ceiling accuracy when relevance is perfect.
# ---------------------------------------------------------------------------

def mr_oracle_relevance(client, items, question, gold_answer: str):
    """Filter sessions by gold-answer keyword overlap — oracle relevance gate."""
    import re
    # Tokenise answer into meaningful keywords (length > 2, skip stop words)
    stop = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "of", "to", "in", "at", "by"}
    gold_words = {w.lower() for w in re.findall(r"\w+", gold_answer) if len(w) > 2 and w.lower() not in stop}

    sessions = group_by_session(items)
    relevant = []
    for sid, sitems in sessions.items():
        if len(sitems) < 2:
            continue
        session_text = " ".join(i.content.lower() for i in sitems)
        overlap = sum(1 for w in gold_words if w in session_text)
        if overlap > 0:
            relevant.append((sid, sitems, overlap))

    if not relevant:
        # Fallback: all sessions with at least 2 turns
        relevant = [(sid, sitems, 0) for sid, sitems in sessions.items() if len(sitems) >= 2]

    relevant.sort(key=lambda x: -x[2])
    relevant_pairs = [(sid, sitems) for sid, sitems, _ in relevant[:10]]

    all_facts = []
    for sid, sitems in relevant_pairs:
        s_text = format_session(sid, sitems)
        facts = query_aware_facts(client, s_text, question)
        all_facts.extend([f"[s{sid}] {f}" for f in facts])
    evidence = "DISTILLED FACTS:\n" + "\n".join(all_facts)
    return answer_with_evidence(client, evidence, question)


# ---------------------------------------------------------------------------
# Variant 8: mr_random_gate  (negative control)
# Random 5 sessions kept (no relevance gate), then query-aware facts.
# If the learned gate adds value, this should score worse than variant 4.
# ---------------------------------------------------------------------------

def mr_random_gate(client, items, question, rng: random.Random):
    sessions = group_by_session(items)
    eligible = [(sid, sitems) for sid, sitems in sessions.items() if len(sitems) >= 2]
    chosen = rng.sample(eligible, min(5, len(eligible)))
    all_facts = []
    for sid, sitems in chosen:
        s_text = format_session(sid, sitems)
        facts = query_aware_facts(client, s_text, question)
        all_facts.extend([f"[s{sid}] {f}" for f in facts])
    evidence = "DISTILLED FACTS:\n" + "\n".join(all_facts)
    return answer_with_evidence(client, evidence, question)


# ---------------------------------------------------------------------------
# Variant 9: long_context_strong_prompt  (fixes weak long-context baseline)
# Full haystack (190K chars) + strong CoT prompt that explicitly asks for
# cross-session aggregation and step-by-step reasoning.
# ---------------------------------------------------------------------------

def long_context_strong_prompt(client, items, question):
    """Full haystack with strong CoT prompt — tests if 20% sonnet_200k was prompt-weak."""
    haystack = format_haystack(items, max_chars=190000)
    prompt = (
        "Read this conversation history carefully. "
        "The answer requires aggregating information from multiple sessions. "
        "Think step-by-step: find ALL relevant evidence, count carefully, then give a concise final answer.\n\n"
        f"CONVERSATION HISTORY:\n{haystack}\n\n"
        f"QUESTION: {question}\n\n"
        "Step-by-step reasoning and final answer (be concise, under 15 words for the answer):"
    )
    return _claude(client, prompt, max_tokens=400)


# ---------------------------------------------------------------------------
# System registry
# ---------------------------------------------------------------------------

SYSTEM_DESCRIPTIONS = {
    "mr_no_query":              "map_reduce baseline — query-agnostic 3 facts/session, no gate",
    "mr_query_aware":           "query-aware facts, no gate — isolates query-awareness alone",
    "mr_gate_no_query":         "gate + query-agnostic facts — isolates gate alone",
    "mr_gate_query":            "gate + query-aware facts — full facts_only system (best)",
    "mr_shuffled_query":        "facts_only with DECOY question — sanity: should drop vs (4)",
    "mr_budget_matched":        "facts_only capped at 3 facts/session — budget-matched to (1)",
    "mr_oracle_relevance":      "gold-answer keyword filter sessions — oracle upper bound (cheats)",
    "mr_random_gate":           "random 5 sessions + query-aware facts — negative gate control",
    "long_context_strong_prompt": "full haystack + strong CoT — fixes weak long-context prompt",
}


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(n_questions: int, output_path: str):
    client = _bedrock_client()
    all_examples = longmemeval(n=300, question_types=None)
    multi = [
        ex for ex in all_examples
        if any("multi-session" in q.query_type for q in ex.queries)
    ][:n_questions]
    print(f"[setup] {len(multi)} multi-session questions", flush=True)

    # Pre-collect all questions for the shuffled-query sanity check
    all_questions = [ex.queries[0].question for ex in multi]

    # Deterministic RNG for random-gate variant
    rng = random.Random(42)

    system_names = list(SYSTEM_DESCRIPTIONS.keys())
    results: dict[str, list] = {name: [] for name in system_names}

    for i, ex in enumerate(multi):
        q = ex.queries[0]
        print(f"\n--- Q{i+1}/{len(multi)} ---  expected={q.answer[:60]!r}", flush=True)

        # --- Variant 1: mr_no_query ---
        try:
            t0 = time.perf_counter()
            ans = mr_no_query(client, ex.items, q.question)
            judge = llm_judge(client, ans, q.answer, q.question)
            print(f"  [mr_no_query          ] judge={judge} ans={ans[:60]!r} ({time.perf_counter()-t0:.1f}s)", flush=True)
            results["mr_no_query"].append({"q": q.question, "gt": q.answer, "ans": ans, "judge": judge})
        except Exception as e:
            print(f"  [mr_no_query          ] FAILED: {e}", flush=True)
            results["mr_no_query"].append({"q": q.question, "gt": q.answer, "ans": "", "judge": 0, "error": str(e)})

        # --- Variant 2: mr_query_aware ---
        try:
            t0 = time.perf_counter()
            ans = mr_query_aware(client, ex.items, q.question)
            judge = llm_judge(client, ans, q.answer, q.question)
            print(f"  [mr_query_aware       ] judge={judge} ans={ans[:60]!r} ({time.perf_counter()-t0:.1f}s)", flush=True)
            results["mr_query_aware"].append({"q": q.question, "gt": q.answer, "ans": ans, "judge": judge})
        except Exception as e:
            print(f"  [mr_query_aware       ] FAILED: {e}", flush=True)
            results["mr_query_aware"].append({"q": q.question, "gt": q.answer, "ans": "", "judge": 0, "error": str(e)})

        # --- Variant 3: mr_gate_no_query ---
        try:
            t0 = time.perf_counter()
            ans = mr_gate_no_query(client, ex.items, q.question)
            judge = llm_judge(client, ans, q.answer, q.question)
            print(f"  [mr_gate_no_query     ] judge={judge} ans={ans[:60]!r} ({time.perf_counter()-t0:.1f}s)", flush=True)
            results["mr_gate_no_query"].append({"q": q.question, "gt": q.answer, "ans": ans, "judge": judge})
        except Exception as e:
            print(f"  [mr_gate_no_query     ] FAILED: {e}", flush=True)
            results["mr_gate_no_query"].append({"q": q.question, "gt": q.answer, "ans": "", "judge": 0, "error": str(e)})

        # --- Variant 4: mr_gate_query (= facts_only) ---
        try:
            t0 = time.perf_counter()
            ans = mr_gate_query(client, ex.items, q.question)
            judge = llm_judge(client, ans, q.answer, q.question)
            print(f"  [mr_gate_query        ] judge={judge} ans={ans[:60]!r} ({time.perf_counter()-t0:.1f}s)", flush=True)
            results["mr_gate_query"].append({"q": q.question, "gt": q.answer, "ans": ans, "judge": judge})
        except Exception as e:
            print(f"  [mr_gate_query        ] FAILED: {e}", flush=True)
            results["mr_gate_query"].append({"q": q.question, "gt": q.answer, "ans": "", "judge": 0, "error": str(e)})

        # --- Variant 5: mr_shuffled_query ---
        try:
            t0 = time.perf_counter()
            ans = mr_shuffled_query(client, ex.items, q.question, all_questions)
            judge = llm_judge(client, ans, q.answer, q.question)
            print(f"  [mr_shuffled_query    ] judge={judge} ans={ans[:60]!r} ({time.perf_counter()-t0:.1f}s)", flush=True)
            results["mr_shuffled_query"].append({"q": q.question, "gt": q.answer, "ans": ans, "judge": judge})
        except Exception as e:
            print(f"  [mr_shuffled_query    ] FAILED: {e}", flush=True)
            results["mr_shuffled_query"].append({"q": q.question, "gt": q.answer, "ans": "", "judge": 0, "error": str(e)})

        # --- Variant 6: mr_budget_matched ---
        try:
            t0 = time.perf_counter()
            ans = mr_budget_matched(client, ex.items, q.question)
            judge = llm_judge(client, ans, q.answer, q.question)
            print(f"  [mr_budget_matched    ] judge={judge} ans={ans[:60]!r} ({time.perf_counter()-t0:.1f}s)", flush=True)
            results["mr_budget_matched"].append({"q": q.question, "gt": q.answer, "ans": ans, "judge": judge})
        except Exception as e:
            print(f"  [mr_budget_matched    ] FAILED: {e}", flush=True)
            results["mr_budget_matched"].append({"q": q.question, "gt": q.answer, "ans": "", "judge": 0, "error": str(e)})

        # --- Variant 7: mr_oracle_relevance ---
        try:
            t0 = time.perf_counter()
            ans = mr_oracle_relevance(client, ex.items, q.question, q.answer)
            judge = llm_judge(client, ans, q.answer, q.question)
            print(f"  [mr_oracle_relevance  ] judge={judge} ans={ans[:60]!r} ({time.perf_counter()-t0:.1f}s)", flush=True)
            results["mr_oracle_relevance"].append({"q": q.question, "gt": q.answer, "ans": ans, "judge": judge})
        except Exception as e:
            print(f"  [mr_oracle_relevance  ] FAILED: {e}", flush=True)
            results["mr_oracle_relevance"].append({"q": q.question, "gt": q.answer, "ans": "", "judge": 0, "error": str(e)})

        # --- Variant 8: mr_random_gate ---
        try:
            t0 = time.perf_counter()
            ans = mr_random_gate(client, ex.items, q.question, rng)
            judge = llm_judge(client, ans, q.answer, q.question)
            print(f"  [mr_random_gate       ] judge={judge} ans={ans[:60]!r} ({time.perf_counter()-t0:.1f}s)", flush=True)
            results["mr_random_gate"].append({"q": q.question, "gt": q.answer, "ans": ans, "judge": judge})
        except Exception as e:
            print(f"  [mr_random_gate       ] FAILED: {e}", flush=True)
            results["mr_random_gate"].append({"q": q.question, "gt": q.answer, "ans": "", "judge": 0, "error": str(e)})

        # --- Variant 9: long_context_strong_prompt ---
        try:
            t0 = time.perf_counter()
            ans = long_context_strong_prompt(client, ex.items, q.question)
            judge = llm_judge(client, ans, q.answer, q.question)
            print(f"  [long_ctx_strong      ] judge={judge} ans={ans[:60]!r} ({time.perf_counter()-t0:.1f}s)", flush=True)
            results["long_context_strong_prompt"].append({"q": q.question, "gt": q.answer, "ans": ans, "judge": judge})
        except Exception as e:
            print(f"  [long_ctx_strong      ] FAILED: {e}", flush=True)
            results["long_context_strong_prompt"].append({"q": q.question, "gt": q.answer, "ans": "", "judge": 0, "error": str(e)})

    # -----------------------------------------------------------------------
    # Summary with bootstrap CIs
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print(f"MAP-REDUCE VARIANT GRID  (n={len(multi)} multi-session questions)", flush=True)
    print("=" * 80, flush=True)

    summary = {}
    for name in system_names:
        scores = [e["judge"] for e in results[name]]
        mean, lo, hi = bootstrap_ci(scores)
        summary[name] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(scores)}
        print(f"  {name:30s}: {mean:.3f} [{lo:.3f}, {hi:.3f}]  ({SYSTEM_DESCRIPTIONS[name]})", flush=True)

    print("\nDeltas vs mr_no_query (map_reduce baseline):", flush=True)
    base = summary["mr_no_query"]["mean"]
    for name in system_names:
        if name == "mr_no_query":
            continue
        delta = summary[name]["mean"] - base
        print(f"  {name:30s}: {delta:+.3f}", flush=True)

    print("\nDeltas vs mr_gate_query (facts_only best):", flush=True)
    best = summary["mr_gate_query"]["mean"]
    for name in system_names:
        if name == "mr_gate_query":
            continue
        delta = summary[name]["mean"] - best
        print(f"  {name:30s}: {delta:+.3f}", flush=True)

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    out = {
        "n": len(multi),
        "descriptions": SYSTEM_DESCRIPTIONS,
        "summary": summary,
        "details": results,
    }
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {output_path}", flush=True)

    # Upload to S3 if bucket is set
    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        import subprocess
        s3_key = f"s3://{s3_bucket}/mapreduce_grid/{os.path.basename(output_path)}"
        try:
            subprocess.run(["aws", "s3", "cp", output_path, s3_key], check=True, timeout=120)
            print(f"[s3] uploaded -> {s3_key}", flush=True)
        except Exception as e:
            print(f"[s3] upload failed: {e}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Map-reduce variant grid experiment")
    p.add_argument("--n", type=int, default=50, help="Number of multi-session questions")
    p.add_argument("--output", default="/tmp/mapreduce_grid.json", help="Output JSON path")
    args = p.parse_args()
    run(args.n, args.output)
