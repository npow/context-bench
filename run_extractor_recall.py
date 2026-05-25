"""Extractor recall analysis — does facts_only WIN because it's compression,
or because the extractor is doing latent answering?

Three experiments, all on the same n questions:

E1. Extraction recall.
    For each Q, the pipeline produces extracted facts. Ask an LLM judge:
    "Do these facts contain the information needed to answer Q?"
    Reports the fraction of extractor outputs where the answer is present
    in the facts. High recall + win → compression wins because it surfaces
    the right facts. Low recall + win → extractor is doing latent answering
    (it figures out the answer and emits it as a "fact").

E2. Token-budget-matched raw snippet.
    For each Q, replace facts with a same-char-budget RAW text snippet:
      - first_chars: first N chars of haystack
      - random_chars: random contiguous N chars
      - bm25_top: top BM25 sessions concatenated, truncated to N chars
    Answer step same as facts_only. If facts_only ≫ all three, extractor
    is doing more than just "any compressed view".

E3. Facts + raw.
    Concat facts with the raw kept-session text. Does adding raw help or hurt
    on top of facts? If raw HURTS on top of facts, supports "raw is harmful"
    claim. If facts+raw > facts alone, then facts alone is missing context.
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


def _claude(client, prompt: str, max_tokens: int = 200) -> str:
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


def group_by_session(items):
    sessions = defaultdict(list)
    order = []
    for it in items:
        sid = getattr(it, "session_id", "") or "default"
        if sid not in sessions:
            order.append(sid)
        sessions[sid].append(it)
    return order, sessions


def is_session_relevant(client, session_text, question):
    prompt = (
        f"Could this session contain information that would help answer the question?\n"
        f"QUESTION: {question}\n"
        f"SESSION:\n{session_text[:8000]}\n\nReply YES or NO only."
    )
    v = _claude(client, prompt, max_tokens=10).upper()
    return "YES" in v


def query_aware_summarize(client, session_text, question):
    prompt = (
        "Extract only facts from this session that would help answer the question. "
        "Output 3-8 short bullet sentences with concrete facts (numbers, names, dates, counts). "
        "If nothing relevant, output 'NONE'.\n\n"
        f"QUESTION: {question}\n"
        f"SESSION:\n{session_text[:8000]}\n\nFacts:"
    )
    return _claude(client, prompt, max_tokens=400)


def answer_from_evidence(client, evidence, question):
    prompt = (
        "Answer the question precisely. If the answer is a number or count, give just the number. "
        "Be concise (under 15 words).\n\n"
        f"EVIDENCE:\n{evidence}\n\nQUESTION: {question}\n\nAnswer:"
    )
    return _claude(client, prompt, max_tokens=100)


def llm_judge(client, pred, gold, question):
    if not pred.strip():
        return 0
    prompt = (
        "Judge if the PREDICTION correctly answers the QUESTION given GOLD. Reply CORRECT or WRONG only.\n\n"
        f"QUESTION: {question}\nGOLD: {gold}\nPREDICTION: {pred}\n\nReply:"
    )
    v = _claude(client, prompt, max_tokens=20).upper()
    if "WRONG" in v or "INCORRECT" in v or "NOT CORRECT" in v:
        return 0
    return 1 if "CORRECT" in v else 0


def facts_contain_answer(client, facts: str, gold: str, question: str) -> int:
    """E1 judge: do the facts contain the info needed to answer Q?"""
    if not facts.strip():
        return 0
    prompt = (
        "You are checking if a fact set contains the information needed to answer a question. "
        "Reply PRESENT if the facts contain the gold answer or info equivalent to it. "
        "Reply MISSING if the facts do not contain that information.\n\n"
        f"QUESTION: {question}\nGOLD ANSWER: {gold}\nFACTS:\n{facts}\n\nReply (PRESENT/MISSING):"
    )
    v = _claude(client, prompt, max_tokens=10).upper()
    if "MISSING" in v: return 0
    if "PRESENT" in v: return 1
    return 0


def bm25_local_windows(items, question, budget_chars: int, window: int = 600):
    """Chunk-level BM25 with local windows around matched terms (Codex fix).
    Splits each session into char windows; ranks all windows; concatenates the
    top windows until budget_chars is filled. Better than session-level truncation
    which can hide late-occurring answers.
    """
    import math, re
    def tokenize(s: str):
        return [w.lower() for w in re.findall(r"\w+", s) if len(w) > 2]

    order, sessions = group_by_session(items)
    # Build chunks (window-sized text slices)
    chunks = []  # list of (chunk_text, session_id)
    for sid in order:
        text = " ".join(i.content for i in sessions[sid])
        for start in range(0, len(text), window):
            chunks.append((text[start:start + window], sid))
    if not chunks:
        return ""

    q_terms = tokenize(question)
    if not q_terms:
        return "".join(c for c, _ in chunks)[:budget_chars]

    # BM25
    tokenized_chunks = [tokenize(c) for c, _ in chunks]
    avgdl = sum(len(t) for t in tokenized_chunks) / max(1, len(tokenized_chunks))
    N = len(chunks)
    df = {term: sum(1 for tc in tokenized_chunks if term in tc) for term in q_terms}
    k1, b = 1.5, 0.75
    scores = []
    for tc in tokenized_chunks:
        dl = len(tc)
        s = 0.0
        for term in q_terms:
            if df[term] == 0: continue
            tf = tc.count(term)
            idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
            s += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(1, avgdl)))
        scores.append(s)

    # Take top chunks until budget
    ranked = sorted(zip(scores, chunks), key=lambda x: -x[0])
    out, used = [], 0
    for score, (chunk, sid) in ranked:
        if score <= 0 and used > 0:
            break
        if used + len(chunk) > budget_chars:
            chunk = chunk[: budget_chars - used]
        out.append(f"[Session {sid}] {chunk}")
        used += len(chunk)
        if used >= budget_chars:
            break

    if not out:
        # Fallback when all scores zero — first budget_chars of haystack
        full = " ".join(c for c, _ in chunks)
        return full[:budget_chars]
    return "\n".join(out)


def hierarchical_facts_and_evidence(client, items, question):
    """Run the relevance gate + per-session extraction. Returns (facts_str, kept_raw_text)."""
    order, sessions = group_by_session(items)
    kept = []
    for sid in order:
        sitems = sessions[sid]
        if len(sitems) < 2:
            continue
        session_text = "\n".join(f"{getattr(i, 'speaker', '')}: {i.content}" for i in sitems)
        if is_session_relevant(client, session_text, question):
            kept.append((sid, session_text))
    if not kept:
        # Fallback: keyword overlap top-5
        q_words = {w.lower() for w in question.split() if len(w) > 3}
        scored = []
        for sid in order:
            sitems = sessions[sid]
            if len(sitems) < 2: continue
            text = " ".join(i.content for i in sitems).lower()
            score = sum(1 for w in q_words if w in text)
            if score > 0:
                stext = "\n".join(f"{getattr(i, 'speaker', '')}: {i.content}" for i in sitems)
                scored.append((sid, stext, score))
        scored.sort(key=lambda x: -x[2])
        kept = [(sid, txt) for sid, txt, _ in scored[:5]]
        if not kept:
            kept = [(sid, "\n".join(f"{getattr(i, 'speaker', '')}: {i.content}" for i in sessions[sid]))
                    for sid in order if len(sessions[sid]) >= 2][:10]

    all_facts = []
    all_raw = []
    for sid, stext in kept[:8]:
        facts = query_aware_summarize(client, stext, question)
        if facts.strip() and "NONE" not in facts.upper():
            all_facts.append(f"[Session {sid}]\n{facts}")
        all_raw.append(f"[Session {sid}]\n{stext}")
    facts_str = "\n\n".join(all_facts)
    raw_str = "\n\n".join(all_raw)
    return facts_str, raw_str


def bootstrap_ci(scores, n=1000):
    if not scores: return 0.0, 0.0, 0.0
    rng = random.Random(42)
    n_s = len(scores)
    means = sorted(sum(rng.choice(scores) for _ in range(n_s)) / n_s for _ in range(n))
    return sum(scores) / n_s, means[int(n * 0.025)], means[int(n * 0.975)]


def run(n_questions, output_path):
    client = _bedrock_client()
    all_examples = longmemeval(n=300, question_types=None)
    multi = [ex for ex in all_examples if any("multi-session" in q.query_type for q in ex.queries)][:n_questions]
    print(f"[setup] {len(multi)} multi-session questions", flush=True)

    results = []

    for i, ex in enumerate(multi):
        q = ex.queries[0]
        gold = q.answer
        question = q.question

        # Build full haystack (for E2 random/first slices)
        full_text_parts = []
        for it in ex.items:
            sp = getattr(it, "speaker", "") or ""
            full_text_parts.append(f"{sp}: {it.content}")
        full_text = "\n".join(full_text_parts)

        # Build facts + raw kept text
        t0 = time.perf_counter()
        facts, raw_kept = hierarchical_facts_and_evidence(client, ex.items, question)
        facts_chars = len(facts)
        print(
            f"\n--- Q{i+1}/{len(multi)} id={ex.id} facts_chars={facts_chars} kept_chars={len(raw_kept)} full_chars={len(full_text)} ({time.perf_counter()-t0:.1f}s) ---",
            flush=True,
        )

        record = {
            "qid": ex.id,
            "question": question,
            "gold": gold,
            "facts_chars": facts_chars,
            "kept_chars": len(raw_kept),
            "full_chars": len(full_text),
        }

        # ===== E1: extraction recall + provenance =====
        recall = facts_contain_answer(client, facts, gold, question)
        record["e1_facts_contain_answer"] = recall

        # Provenance — REDESIGNED to address Codex's underlying critique
        # (not bandaided by bumping slice limits).
        #
        # Old design: judge "are ALL facts in this 8K-char blob supported by
        # this 60K-char raw blob?" → same truncation pathology, judge cannot
        # actually check ALL facts.
        #
        # New design: fact-by-fact provenance with targeted raw retrieval.
        # 1. Split facts into individual bullet sentences.
        # 2. Ask LLM to identify which numbered fact contains the gold-answer info.
        # 3. For that fact, retrieve top-K BM25 windows from raw (~3K chars).
        # 4. Judge ONE FACT against the RELEVANT RAW WINDOWS:
        #    e1_answer_provenance = 1 iff that judge says SUPPORTED.
        # 5. Sample 3 other facts; judge each the same way.
        #    e1_all_facts_provenance = AND across sampled facts (or 0 if no facts).
        import re as _re, hashlib as _hashlib

        fact_lines = []
        for line in facts.split("\n"):
            line = line.strip().lstrip("-*•").strip()
            if line and len(line) > 10 and not line.startswith("[Session"):
                fact_lines.append(line)
        record["n_facts_extracted"] = len(fact_lines)

        def retrieve_raw_windows(fact_text, raw_text, budget=3000, window=400):
            import math as _math
            terms = [w.lower() for w in _re.findall(r"\w+", fact_text) if len(w) > 3]
            if not terms or not raw_text:
                return raw_text[:budget]
            chunks = [raw_text[i:i+window] for i in range(0, len(raw_text), window)]
            if not chunks:
                return raw_text[:budget]
            tokenized = [[w.lower() for w in _re.findall(r"\w+", c)] for c in chunks]
            avgdl = sum(len(t) for t in tokenized) / max(1, len(tokenized))
            N = len(chunks)
            df = {t: sum(1 for tc in tokenized if t in tc) for t in terms}
            k1, b = 1.5, 0.75
            scored = []
            for c, tc in zip(chunks, tokenized):
                dl = len(tc); s = 0.0
                for term in terms:
                    if df[term] == 0: continue
                    tf = tc.count(term)
                    idf = _math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
                    s += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(1, avgdl)))
                scored.append((s, c))
            scored.sort(key=lambda x: -x[0])
            out, used = [], 0
            for s, c in scored:
                if used >= budget: break
                if s <= 0 and out: break
                take = min(len(c), budget - used)
                out.append(c[:take]); used += take
            return "\n---\n".join(out) if out else raw_text[:budget]

        def _normalize(s: str) -> str:
            """Lowercase + collapse whitespace + strip punctuation for quote matching."""
            return _re.sub(r"\s+", " ", _re.sub(r"[^\w\s]", " ", s.lower())).strip()

        def judge_fact_provenance(fact, raw_windows, full_raw):
            """Quote-grounded provenance (Codex demand).
            Judge must return a literal supporting quote from RAW TEXT. We then
            VERIFY (substring match in normalized raw) that the quote actually
            exists. Pure LLM 'yes' verdict is too soft.

            Two-pass like before: BM25 narrow first, then FULL raw if needed.
            """
            def ask(raw_text):
                prompt = (
                    "You are checking whether a FACT is supported by a piece of RAW TEXT. "
                    "Respond in two lines:\n"
                    "Line 1: VERDICT: SUPPORTED or UNSUPPORTED\n"
                    "Line 2: QUOTE: <exact verbatim quote from RAW TEXT that supports the fact, or NONE>\n"
                    "Do NOT paraphrase. Copy the supporting sentence verbatim.\n\n"
                    f"FACT: {fact}\n\nRAW TEXT:\n{raw_text}\n"
                )
                return _claude(client, prompt, max_tokens=200)

            def verify(response: str, raw_text: str) -> tuple[int, str]:
                """Return (verdict, quote). verdict=1 only if quote substring-matches raw."""
                lines = response.split("\n")
                verdict_line = next((l for l in lines if "VERDICT" in l.upper()), "")
                quote_line = next((l for l in lines if "QUOTE" in l.upper()), "")
                v_text = verdict_line.upper()
                if "UNSUPPORTED" in v_text or "SUPPORTED" not in v_text:
                    return 0, ""
                # Extract quote text after "QUOTE:"
                if ":" not in quote_line:
                    return 0, ""
                quote = quote_line.split(":", 1)[1].strip()
                if not quote or quote.upper() == "NONE" or len(quote) < 5:
                    return 0, ""
                # Substring check (normalized)
                if _normalize(quote)[:200] in _normalize(raw_text):
                    return 1, quote
                return 0, quote  # claimed quote but it's not in raw — hallucinated support

            r1 = ask(raw_windows)
            v1, q1 = verify(r1, raw_windows)
            if v1 == 1:
                return v1
            # Retry on full raw
            r2 = ask(full_raw)
            v2, q2 = verify(r2, full_raw)
            return v2

        if fact_lines and raw_kept.strip():
            # Identify the answer-supporting fact
            # Codex fix: don't cap at 30 — answer-fact could be past 30.
            # Pass ALL facts (typical max ~50, well within Sonnet's window).
            id_prompt = (
                "Below are extracted facts (numbered). Identify the SINGLE fact number that contains "
                "the information needed to answer the QUESTION with the GOLD ANSWER. "
                "Reply with just the number (1, 2, 3, ...) or NONE if no fact contains the answer info.\n\n"
                f"QUESTION: {question}\nGOLD ANSWER: {gold}\n\n"
                + "\n".join(f"{i+1}. {f}" for i, f in enumerate(fact_lines))
                + "\n\nFact number (or NONE):"
            )
            v = _claude(client, id_prompt, max_tokens=10).strip()
            m = _re.search(r"\d+", v)
            ans_fact_idx = -1
            if m:
                idx = int(m.group()) - 1
                if 0 <= idx < len(fact_lines):
                    ans_fact_idx = idx
            record["answer_supporting_fact_idx"] = ans_fact_idx

            if ans_fact_idx >= 0:
                answer_fact = fact_lines[ans_fact_idx]
                rw = retrieve_raw_windows(answer_fact, raw_kept, budget=3000)
                record["e1_answer_provenance"] = judge_fact_provenance(answer_fact, rw, raw_kept)
            else:
                record["e1_answer_provenance"] = 0

            # Sample 3 other facts — RENAMED to e1_sampled_facts_provenance per
            # Codex: this is NOT "all facts," it's a sample.
            sample_pool = [i for i in range(len(fact_lines)) if i != ans_fact_idx]
            seed_int = int.from_bytes(_hashlib.sha256(("prov" + str(ex.id)).encode()).digest()[:8], "big")
            random.Random(seed_int).shuffle(sample_pool)
            n_to_sample = min(3, len(sample_pool))
            sampled = sample_pool[:n_to_sample]
            verdicts = []
            for fi in sampled:
                fact = fact_lines[fi]
                rw = retrieve_raw_windows(fact, raw_kept, budget=3000)
                verdicts.append(judge_fact_provenance(fact, rw, raw_kept))
            record["e1_sampled_facts_provenance_verdicts"] = verdicts
            record["e1_sampled_facts_provenance_n"] = n_to_sample
            # All-supported AND across sample; honestly labeled as sampled.
            record["e1_sampled_facts_all_supported"] = 1 if verdicts and all(v == 1 for v in verdicts) else 0
        else:
            record["e1_answer_provenance"] = 0
            record["e1_sampled_facts_provenance_verdicts"] = []
            record["e1_sampled_facts_provenance_n"] = 0
            record["e1_sampled_facts_all_supported"] = 0
            record["answer_supporting_fact_idx"] = -1

        # facts_only baseline answer
        ans_facts = answer_from_evidence(client, facts, question)
        record["facts_only_judge"] = llm_judge(client, ans_facts, gold, question)
        record["facts_only_ans"] = ans_facts
        print(
            f"  E1: recall={recall} ans_prov={record['e1_answer_provenance']} "
            f"sampled_all_supp={record['e1_sampled_facts_all_supported']}({record['e1_sampled_facts_provenance_n']}) "
            f"facts_only_judge={record['facts_only_judge']} ans={ans_facts[:50]!r}",
            flush=True,
        )

        # ===== E2: token-budget-matched raw snippets =====
        # Use facts_chars as budget. Compare three slicing strategies.
        budget = max(2000, facts_chars * 2)  # facts are sparse, budget the raw at 2x for fairness

        # first_chars
        first_snippet = full_text[:budget]
        ans_first = answer_from_evidence(client, first_snippet, question)
        record["e2_first_chars_judge"] = llm_judge(client, ans_first, gold, question)

        # random_chars — deterministic seed per question id (Codex fix)
        import hashlib
        seed_int = int.from_bytes(hashlib.sha256(str(ex.id).encode()).digest()[:8], "big")
        rng = random.Random(seed_int)
        if len(full_text) > budget:
            start = rng.randrange(0, max(1, len(full_text) - budget))
            rand_snippet = full_text[start:start + budget]
        else:
            rand_snippet = full_text
        ans_rand = answer_from_evidence(client, rand_snippet, question)
        record["e2_random_chars_judge"] = llm_judge(client, ans_rand, gold, question)

        # bm25 local windows — chunk-level ranking + local windows (Codex fix)
        bm25_snip = bm25_local_windows(ex.items, question, budget_chars=budget, window=600)
        ans_bm25 = answer_from_evidence(client, bm25_snip, question)
        record["e2_bm25_budget_judge"] = llm_judge(client, ans_bm25, gold, question)

        print(
            f"  E2: first={record['e2_first_chars_judge']}  random={record['e2_random_chars_judge']}  bm25_budget={record['e2_bm25_budget_judge']}",
            flush=True,
        )

        # ===== E3: facts + raw =====
        combined = facts + "\n\n=== RAW SESSIONS ===\n" + raw_kept
        ans_combined = answer_from_evidence(client, combined, question)
        record["e3_facts_plus_raw_judge"] = llm_judge(client, ans_combined, gold, question)
        # raw alone (for comparison)
        ans_raw = answer_from_evidence(client, raw_kept, question)
        record["e3_raw_only_judge"] = llm_judge(client, ans_raw, gold, question)
        print(
            f"  E3: facts_plus_raw={record['e3_facts_plus_raw_judge']}  raw_only={record['e3_raw_only_judge']}",
            flush=True,
        )

        results.append(record)

    # ===== Aggregate =====
    print("\n" + "=" * 80, flush=True)
    print(f"EXTRACTOR RECALL ANALYSIS (n={len(results)})", flush=True)
    print("=" * 80, flush=True)

    metrics = {
        "e1_facts_contain_answer": "Extraction recall — do facts contain gold info?",
        "e1_answer_provenance": "Provenance (answer fact) — gold-supporting fact supported by raw?",
        "e1_sampled_facts_all_supported": "Provenance (3 sampled facts) — ALL supported by raw?",
        "facts_only_judge": "Pipeline acc (facts only)",
        "e2_first_chars_judge": "E2: first-N chars (budget-matched raw)",
        "e2_random_chars_judge": "E2: random-N chars (budget-matched raw)",
        "e2_bm25_budget_judge": "E2: BM25 top, truncated to budget",
        "e3_facts_plus_raw_judge": "E3: facts + raw concatenated",
        "e3_raw_only_judge": "E3: raw kept sessions only",
    }
    summary = {}
    for key, desc in metrics.items():
        scores = [r[key] for r in results if key in r]
        if not scores:
            continue
        mean, lo, hi = bootstrap_ci(scores)
        summary[key] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(scores), "desc": desc}
        print(f"  {key:30s}  {mean:.3f} [{lo:.3f}, {hi:.3f}]  n={len(scores)}  — {desc}", flush=True)

    # Joint analysis
    e1_given_correct = [r["e1_facts_contain_answer"] for r in results if r.get("facts_only_judge") == 1]
    e1_given_wrong = [r["e1_facts_contain_answer"] for r in results if r.get("facts_only_judge") == 0]
    print(f"\nP(facts contain answer | facts_only CORRECT) = {sum(e1_given_correct)/max(1,len(e1_given_correct)):.3f}  (n={len(e1_given_correct)})")
    print(f"P(facts contain answer | facts_only WRONG)   = {sum(e1_given_wrong)/max(1,len(e1_given_wrong)):.3f}  (n={len(e1_given_wrong)})")

    # Latent answering signature: recall=1 BUT answer_provenance=0
    latent = [r for r in results if r.get("e1_facts_contain_answer") == 1 and r.get("e1_answer_provenance") == 0]
    print(f"P(LATENT ANSWERING signature: recall=1 & answer_provenance=0) = {len(latent)}/{len(results)} = {100*len(latent)/max(1,len(results)):.1f}%")
    print(f"P(answer_provenance | recall=1) = {sum(1 for r in results if r.get('e1_facts_contain_answer')==1 and r.get('e1_answer_provenance')==1)}/{sum(1 for r in results if r.get('e1_facts_contain_answer')==1)} (high → compression, not latent answering)")

    out = {"n": len(results), "summary": summary, "details": results}
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {output_path}", flush=True)

    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        import subprocess
        s3_key = f"s3://{s3_bucket}/extractor_recall/{os.path.basename(output_path)}"
        try:
            subprocess.run(["aws", "s3", "cp", output_path, s3_key], check=True, timeout=120)
            print(f"[s3] {s3_key}", flush=True)
        except Exception as e:
            print(f"[s3] {e}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--output", default="/tmp/extractor_recall.json")
    args = p.parse_args()
    run(args.n, args.output)
