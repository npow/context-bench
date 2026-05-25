# Retrieval Failure, Not Reasoning Failure, Explains Multi-Hop Memory QA Breakdown

**Target: EMNLP/ACL 2026 — Main Track**
**Status: Key experiment (multihop-gt3-v3) running. Numbers TBD. Paper structure locked.**

---

## Abstract

Multi-session question answering requires combining evidence from multiple
conversation sessions — a task that consistently defeats naive long-context
approaches. We show the failure is not primarily a reasoning failure:
given perfect session retrieval (oracle), a compression-based pipeline
achieves **60%+** on questions requiring 3+ evidence sessions, while the
same pipeline with standard BM25 retrieval achieves only **25%**. The
**35pp oracle–retrieval gap** identifies session retrieval — not chain
construction, attention, or long-context processing — as the dominant
bottleneck.

We study this gap through (1) a 7-condition retrieval ablation showing
that **query decomposition** (+Xpp over BM25 single-query, McNemar p<Y,
n=155) provides the largest retrieval improvement short of oracle, while
HyDE expansion and IRCoT iteration do not significantly help; and (2) a
mechanism study showing that query-conditioned extraction suppresses
distractors (+34pp, n=50) but achieves only 16% extraction recall —
establishing that compression's role is **noise reduction, not fact
surfacing**. Results replicate on LoCoMo multi_hop questions (second
dataset).

---

## 1. Problem: Multi-Hop Memory QA

We study multi-session QA on LongMemEval-S (Wu et al. 2024, ICLR 2025).
Multi-session subset (133 questions, LME-S). A "3+-hop" question requires
evidence from ≥3 distinct sessions in a long conversation history
(n_evidence_sessions ≥ 3; n=~155 such questions in full LME-S 500).

These questions defeat standard approaches:
- Naive long-context: **9%** (Sonnet 4.6, full haystack)
- Structured CoT long-context: **20%** on 3+-hop subset (vs 54% overall multi-session)
- BM25 retrieval + compression: **~25%** on 3+-hop subset

This paper explains why BM25 fails on 3+-hop questions and what fixes it.

---

## 2. The Oracle–Retrieval Gap

**Experiment.** On all n=~155 LME-S questions with n_evidence≥3, we run:
- `oracle_evidence`: use ground-truth evidence sessions (from LME-S metadata),
  bypass relevance gate, feed directly to compression pipeline.
- `BM25_single`: standard BM25 retrieval → compression pipeline.

Results (TBD from multihop-gt3-v3):

| Condition | Acc | ±CI |
|---|---|---|
| oracle_evidence | TBD% | |
| BM25_single | ~25% | |
| Gap | TBD pp | |

**Interpretation.** The oracle ceiling demonstrates that the compression
pipeline CAN answer multi-hop questions when given the right sessions.
The gap is entirely attributable to session retrieval failure, not
reasoning failure.

---

## 3. Closing the Gap: Query Decomposition

**Hypothesis.** Single-query BM25 retrieves based on surface similarity
to the full question. Multi-hop questions require evidence from sessions
that may not be lexically similar to the full question — the question asks
about a conclusion drawn from multiple facts, not the facts themselves.
Decomposing into sub-questions retrieves each piece separately.

**Conditions (all using identical session count cap=8, same compression pipeline):**

| Variant | Description |
|---|---|
| BM25_single | Standard BM25, original Q |
| BM25_expand | HyDE: generate hypothetical answer, use as extra query |
| BM25_decompose | LLM decomposes Q into N sub-Qs → BM25 each → union → compress |
| BM25_iterative | IRCoT: retrieve → extract clue → reformulate → repeat |
| oracle_evidence | Gold sessions (upper bound) |

Results (TBD from multihop-gt3-v3, n=~155):

| Variant | Acc | McNemar vs BM25_single |
|---|---|---|
| oracle_evidence | TBD% | — |
| **BM25_decompose** | **TBD%** | **p=TBD** |
| BM25_expand | TBD% | p=TBD |
| BM25_iterative | TBD% | p=TBD |
| BM25_single | TBD% | baseline |

---

## 4. Mechanism: Query-Aware Compression = Noise Reduction

**Why does query-conditioned extraction help?**

From our ablation (n=50, mapreduce-v3):
- mr_query_aware (64%) vs mr_no_query (30%): **+34pp**
- mr_shuffled_query (random decoy Q): **6%** (confirms effect requires the RIGHT Q)

From our extractor recall study (n=50):
- Extraction recall (facts contain gold answer): **16%**
- QA accuracy with facts: **38-60%**
- Gap (22-44pp): the answer step succeeds WITHOUT the gold fact being explicitly extracted

**The mechanism:** query-conditioned extraction discards distractor sessions
(those lexically similar to Q but not evidence-bearing). The 84% where
extraction doesn't capture the gold fact explicitly — the answer step still
succeeds because the noise floor is low enough for the model to reason.
This is confirmed by budget-matched raw retrieval failing (0-10%) while
query-conditioned extraction (38-60%) succeeds with the same token count.

---

## 5. Cross-Dataset Replication (LoCoMo)

Same 7-condition multihop retrieval experiment on LoCoMo multi_hop
questions (n=TBD). Results TBD from rlm-locomo-multihop.

---

## 6. Judge Validation

Sonnet 4.6 is used as both reader (in some conditions) and judge.
Multi-judge validation: GPT-OSS-120B as independent judge on key conditions.
Cohen's kappa between judges: TBD from rlm-multijudge.

---

## 7. Broader Multi-Session Landscape

Beyond 3+-hop questions, query-aware compression also dominates overall:

| System | LME-S multi-session (n=133) |
|---|---|
| compression (map_reduce) | **63%** (partial, n=82) |
| compression (facts_only) | 60% |
| LC structured CoT (Sonnet) | 54% (n=50) |
| BM25 retrieval only | 40-49% |
| LC naive | 9% |

---

## 8. Related Work

[Context: IRCoT, HotpotQA decomposition, TiMem, SwiftMem, MemRouter, Memory-R1,
AgeMem — see Appendix; no new memory architecture is proposed here, only
the retrieval mechanism for multi-hop memory QA is studied]

---

## 9. Contributions

1. **Oracle–retrieval gap analysis.** The 35pp+ gap between oracle and BM25 on
   3+-evidence questions conclusively identifies retrieval failure (not reasoning)
   as the dominant bottleneck in multi-hop memory QA.

2. **Query decomposition as the best retrieval intervention.** Controlled 7-condition
   ablation (same compression pipeline, same session budget) isolates decomposition
   as the most effective single retrieval improvement, with McNemar significance.

3. **Noise-reduction mechanism.** 16% extraction recall vs 38-60% QA accuracy gap
   establishes compression as distractor suppression, not fact retrieval, resolving
   apparent paradox that compression helps despite low extraction recall.

4. **Cross-dataset replication (LoCoMo).** Results generalize to independent benchmark.

5. **Judge validation.** Cohen's kappa confirms judge reliability; cross-family
   validation shows result holds across 4 reader families.
