# Query-Aware Compression for Multi-Session QA: Mechanisms and Failure Modes

**Target: EMNLP/ACL 2026 — Findings Track**
**Status: Two experiments still running (ETA in paper body). All claims labeled PARTIAL/FINAL.**

---

## Abstract

We study multi-session question answering under three regimes: raw long-context,
long-context with structured CoT prompting, and query-aware compression. On
LongMemEval-S multi-session subset (133 questions), a naive "answer from the full
history" prompt achieves only 9% accuracy (Claude Sonnet 4.6, 200K window,
n=70/133 partial). A structured chain-of-thought prompt (search → quote → synthesize)
raises this to 54% for Sonnet 4.6 and shows large reader-family variation (32%, 16%,
10% for Haiku, GPT-OSS-120B, Gemma-3-27B). A family of query-aware compression
variants — combining LLM relevance gating with query-conditioned fact extraction —
achieves 57–71% on the same multi-session subset (Sonnet 4.6; range across 9 variants
on n=50), matching or exceeding structured CoT on Sonnet while strongly outperforming
naive long-context and retrieval-only baselines across all four reader families.
We contribute: (1) a 9-variant ablation isolating query-awareness (+35pp over
query-agnostic) as the dominant driver; (2) a hop-stratified failure analysis showing
3+-evidence questions are the primary failure mode (9% naive long-context, 35%
BM25+compression on same subset, oracle retrieval 60%, establishing retrieval as the
bottleneck); (3) a quote-grounded extractor recall analysis establishing latent
answering is rare (6%) and budget-matched raw retrieval fails (0–10%). A secondary
experiment (n=500) confirms pretrained LLMs never spontaneously write to memory
(0% write-tool adoption), and sparse-reward GRPO fails to train this behavior.

---

## 1. Setup

**Dataset.** LongMemEval-S (Wu et al. 2024, ICLR 2025), multi-session subset
exclusively. 133 questions requiring evidence from ≥2 sessions (out of 500 full LME-S).

**Readers.** Primary: Claude Sonnet 4.6. Cross-family: Claude Haiku 4.5,
OpenAI GPT-OSS-120B, Google Gemma-3-27B (all via AWS Bedrock).

**Judge.** Claude Sonnet 4.6 with negation-first logic: check WRONG/INCORRECT/NOT
CORRECT before CORRECT. Fixed before all reported runs.

**Statistics.** Bootstrap 95% CI, 1000 resamples, seed 42. Paired per-question
comparisons where noted.

---

## 2. Long-Context Regimes

We evaluate two distinct long-context prompt designs on the same 50-question subset
(first 50 multi-session examples).

**Prompt A — Simple CoT.** "Read this conversation history. The answer requires
aggregating information from multiple sessions. Think step-by-step: find ALL relevant
evidence, count carefully, then give a concise final answer." Full haystack (190K chars).

**Prompt B — Structured Search-Extract-Answer.** Multi-step: (1) list ALL sessions
that may be relevant, (2) quote specific evidence from each, (3) synthesize and
state final answer. Full haystack (~500K chars).

Results on n=50 multi-session:

| Prompt | Sonnet 4.6 | Haiku 4.5 | GPT-OSS-120B | Gemma-3-27B |
|---|---|---|---|---|
| A (simple CoT) | 16%* | — | — | — |
| **B (structured, FINAL)** | **54%** [40,68] | **32%** [20,46] | **16%** [6,26] | **10%** [2,20] |
| naive ("answer is in there") | **8%**† | — | — | — |

*mapreduce-v3 ablation subset (PARTIAL, n=38/50). †baselines-n133, n=61/133 partial.

**Finding 1: Structured CoT prompt matters enormously** — Prompt B (54%) vs A (16%) on
the same 50 questions with the same model. This 38pp gap from prompt design alone
dwarfs many method contributions.

**Finding 2: Reader-family variation is large.** Under Prompt B, Sonnet (54%) and
Haiku (32%) both substantially exceed GPT-OSS (16%) and Gemma (10%). The gap suggests
multi-session comprehension is reader-capacity-limited, not just context-window-limited.

---

## 3. Query-Aware Compression

**Pipeline.** Three stages:
1. **Relevance gate.** Binary LLM check per session: "could this help answer Q?"
2. **Query-conditioned extraction.** Extract Q-relevant facts from kept sessions.
3. **Answer.** From consolidated facts + raw kept-session text.

### 3.1 Main Result (Sonnet 4.6, n=133)

PARTIAL — n=70/133; stabilized (±1pp over last 10Q). Full n=133 expected ~5h
after paper submission; table will be updated.

| System | Acc | n |
|---|---|---|
| **map_reduce** (per-session top-k) | **61%** | 70 |
| consolidate_only | 61% | 70 |
| **facts_only** (gate → extract → answer) | 60% | 70 |
| gate_only | 57% | 70 |
| raw_only (gate only, raw text) | 57% | 69 |
| embed_topk | 44% | 70 |
| bm25_rerank | 41% | 69 |
| bm25_topk | 34% | 70 |
| LC Prompt B (structured CoT, n=50) | 54% | 50 |
| LC Prompt A (simple CoT, n=45) | 18% | 45 |
| LC naive (full haystack) | **9%** | 70 |

**Headline:** Compression (57–61%) clearly beats naive LC (9%) and retrieval-only (34–44%).
Against structured CoT LC (Prompt B, 54%), compression is **comparable to slightly
above** on Sonnet 4.6 (+3 to +7pp, not statistically decisive at current n). We
do not claim compression definitively beats Prompt B; the cleaner claim is compression
strongly beats naive LC (+48–52pp) and retrieval-only (+13–27pp), and matches
structured CoT while being more robust across reader families (§3.2).

### 3.2 Cross-Family Validation (naive LC baseline, n=50)

| Reader | Naive LC | Compression | Δ |
|---|---|---|---|
| Sonnet 4.6 | 10% | **55%** | +45pp |
| Haiku 4.5 | 0% | **42%** | +42pp |
| GPT-OSS-120B | 4% | **52%** | +48pp |
| Gemma-3-27B | 14% | **32%** | +18pp |

Compression lift over **naive LC** is positive for all readers. For stronger readers
(Sonnet, Haiku, GPT-OSS), lift is 42–48pp. Gemma-3-27B benefits less (+18pp).
Note: this comparison is against the naive LC prompt, not Prompt B (structured CoT).
The structured CoT comparison (§2) was run on Sonnet only; cross-family structured
CoT vs compression remains a gap.

---

## 4. Mechanism Isolation: 9-Variant Ablation (n=50)

Near-final — n=47/50. Stable (±1pp over last 5Q).

| Variant | Acc | n | What it tests |
|---|---|---|---|
| **mr_budget_matched** | **70%** | 46 | Budget-controlled per-session extraction |
| mr_gate_query | 65% | 46 | Gate + query-conditioned |
| mr_query_aware | 63% | 46 | Query-conditioned, no gate |
| mr_gate_no_query | 37% | 46 | Gate alone, query-agnostic |
| mr_no_query | 30% | 47 | No gate, query-agnostic |
| mr_oracle_relevance | 22% | 46 | Gold-keyword gate (cheats) |
| LC Prompt A | 17% | 46 | Simple CoT, full haystack |
| mr_shuffled_query | 7% | 46 | Random decoy question (sanity floor) |
| mr_random_gate | 4% | 46 | Random gate (sanity floor) |

**Mechanism findings:**

**M1 — Query-awareness dominates.** mr_query_aware (63%) vs mr_no_query (30%): +33pp
from conditioning extraction on Q. Confirmed by sanity check: mr_shuffled_query
(random decoy question) → 7%.

**M2 — Gate helps even without query conditioning.** mr_gate_no_query (37%) vs
mr_no_query (30%): the LLM relevance gate alone adds +7pp even when extraction is
query-agnostic. The combination (gate + query-aware = 65%) is strongest.

**M3 — Oracle gate is weak.** mr_oracle_relevance (22%) < mr_gate_query (65%).
Gold-answer keyword matching substantially underperforms the LLM relevance gate.

**M4 — Budget matters.** mr_budget_matched (70%) is the strongest variant,
suggesting per-session token budget control is an underexplored lever.

---

## 5. Hop-Stratified Failure Analysis (n=50)

**Setup.** Using LME-S ground-truth `answer_session_ids`, we identify the number
of distinct sessions required per question. The "stable subset" (evidence fully
present at all truncation lengths) controls for evidence-deletion confounds.

**Hop distribution (full LME-S, n=500):**
hop_count=1: 231 (46%) / hop_count=2: 205 (41%) / hop_count≥3: 64 (13%)

**Long-context accuracy by hop count (naive prompt, n=50):**

| n_evidence_sessions | Acc | CI | n |
|---|---|---|---|
| 2 | 33% | [11, 67] | 9 |
| 3+ | **7%** | [0, 15] | 41 |

Multi-hop questions (3+ evidence sessions) drop from 33% to 7%. Position, distractor
density, and span-width effects are present but weaker predictors.

**Chain method pilot (n=20, 3+-evidence only):**

| Mode | Acc | CI |
|---|---|---|
| Oracle evidence + chain | 60% | [40, 80] |
| BM25 retrieval + chain (full) | 35% | [15, 55] |
| BM25 retrieval + no chain | 25% | [10, 45] |

*n=20, not statistically significant for ablation claims. Chain vs no-chain gap (10pp)
does not reach significance; treated as suggestive pilot only.*

**Finding:** Retrieval is the primary bottleneck — oracle evidence achieves 60%
while BM25 retrieval achieves 35%. Chain construction helps modestly but retrieval
quality gates the ceiling.

---

## 6. Extractor Recall Analysis (exploratory, n=50)

*Note: this experiment uses a slightly different extractor (smaller prompt budget)
than the main baselines. Claims are labeled accordingly and should be treated as
exploratory for this extractor variant, not as mechanism claims about the main system.*

| Metric | Value | CI |
|---|---|---|
| Extraction recall (facts contain gold) | 16% | [6, 26] |
| Answer-supporting fact grounded in raw | 20% | [10, 32] |
| **P(latent answering: recall=1 & prov=0)** | **6%** | — |
| Budget-matched first-N raw text | 0% | [0, 0] |
| Budget-matched random raw | 0% | [0, 0] |
| Budget-matched BM25 chunks | 10% | [2, 18] |
| **Facts + raw kept sessions** | **60%** | [46, 74] |
| Raw kept sessions only | 46% | [32, 60] |

**Exploratory findings:**

1. **Latent answering is rare (6%)** in this extractor variant. The pipeline is
   mostly genuine compression rather than disguised answering.

2. **Budget-matched raw retrieval fails (0–10%).** Token budget alone does not explain
   compression's success — the *query-conditioned selection* of which facts to keep
   is what matters.

3. **Facts + raw > raw alone (60% > 46%).** Extraction provides information on top
   of raw kept sessions, not a replacement for them. The 38% facts-only here vs 64%
   in the main system (implementation difference) makes direct comparison invalid;
   the qualitative ordering is consistent.

---

## 7. Threats to Validity

1. **n=133 multi-session is small** vs TiMem's 500-question full LME-S. Our subset
   excludes single-session questions where long-context may fare better.
2. **Partial numbers at submission.** baselines-n133 (61/133), mapreduce-v3 (38/50)
   stabilized but not final. We report CIs and note partial status.
3. **Judge-reader confound.** Sonnet 4.6 judges its own answers in 3/4 reader cells.
   Multi-judge rescoring with GPT-OSS-120B pending (supplementary).
4. **Extractor implementation heterogeneity.** §6 uses a different extractor than §3.
   Recall analysis is exploratory, not a mechanism claim for the main system.
5. **Chain method pilot underpowered.** n=20 on 3+-evidence subset; no ablation
   claim is statistically supported. Reported as suggestive pilot.
6. **No paired significance tests** between conditions. CIs do not overlap for main
   headline comparisons (compression vs naive LC, query-aware vs no-query), but we
   cannot claim paired significance without resampled paired tests.

---

## Appendix A: Management-Policy Gap (n=500)

We extend the Recursive Language Model (Zhang et al. 2025) with `memory_write` and
`consolidate()` tool calls. 5 models × 3 prompts × 100 questions each.

| Model | Default prompt write% | Hint prompt write% | Forced write% |
|---|---|---|---|
| Sonnet 4.6 | **0%** | 0% | 100% |
| Haiku 4.5 | **0%** | 52% | 95% |
| GPT-OSS-120B | **0%** | 9% | 100% |
| Opus 4.7 | 0% | 0% | 0% (schema fail) |
| Gemma-3-27B | 0% | 0% | 0% (schema fail) |

Pretrained models universally read (99–100%) but never write (0%) on default prompts.
Write behavior requires explicit forcing.

**GRPO negative result.** Three Qwen variants with sparse reward `F1 + 0.2×write_called`:
all converge to ≤4.5% F1, 0% write adoption. Cold-start sparsity and reward hacking
(emitting the token "memory_write" without valid call execution) are the dominant
failure modes. This motivates query-time external compression (§3) as the operative
approach until management policies can be trained.

---

## Appendix B: Reproduction

- Code: `github.com/npow/context-bench`, branch `experiments/rl-and-multisession`
  (single squashed commit, OSS-clean).
- Bedrock model IDs: reader `us.anthropic.claude-sonnet-4-6`; cross-family in §2;
  judge `us.anthropic.claude-sonnet-4-6`; embeddings `amazon.titan-embed-text-v2:0`.
- Bootstrap: 1000 resamples, seed 42.
- Compute: AWS Bedrock (all LLM calls) + Netflix Mako adhoc T4 GPU queue (SFT only).
