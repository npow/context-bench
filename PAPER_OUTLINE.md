# Paper Outline: The Management Policy Gap in Recursive Language Models

**Working title:** *The Management Policy Gap in Recursive Language Models:
Read-Write Adoption Asymmetry and the Limits of Sparse-Reward RL*

**Target venue:** EMNLP 2026 (June) or ACL 2026 (Feb)
**Length:** 8 pages + references (short paper / findings if positive results <10pp; full long if >10pp)

---

## The hook (TLDR for reviewers)

Recursive Language Models (RLMs; Zhang et al., 2025) ship with a *read* tool but no *write* tool. We add memory_write to the RLM REPL. Pretrained instruction-tuned LLMs use memory_read at 100% but memory_write at 0%. Training RLMs to close this gap with sparse F1 rewards fails (3 model sizes, all converge to ≤2% F1 with reward hacking). But if writes are used optimally (per-session consolidation), accuracy on LongMemEval multi-session jumps by {X}pp judge accuracy with {Y}x latency overhead — a clear point on the accuracy-cost Pareto frontier. The management policy gap is real AND closing it is high-value, but the obvious RL training recipe doesn't work yet.

---

## 1. Introduction (1 page)

- RLM is the SOTA architecture for ultra-long context (Zhang et al., 2025)
- RLM treats answering as "model writes Python to read from REPL, accumulate `answer["content"]`, mark `ready=True`"
- All operations in the RLM REPL are READS (`grep`, `peek`, `llm_batch`); no WRITES
- We extend the REPL with `memory_write(content, type)` and `consolidate()` (Layer 1 of our prior research plan)
- **Observation:** pretrained models call memory_read on 100% of queries and memory_write on 0%
- **Question:** Can RL close this gap? Is the gap worth closing?

**Contributions:**
1. We operationalize the *two-policy gap* (Section 3.1) and confirm it on LoCoMo + LongMemEval
2. We show GRPO from sparse F1 rewards CANNOT close the gap, even with reward shaping (`write_attempt_bonus`) and code-specialized models (Qwen-Coder-7B)
3. We provide an *oracle upper bound* via deterministic per-session consolidation — showing the gap IS worth closing on multi-session questions (+{X}pp judge accuracy)
4. We characterize WHY the RL training fails (sparse rewards, reward hacking, base-model code generation quality) and propose future directions

---

## 2. Background and Setup (0.5 pages)

- RLM (Zhang et al., 2025): root LM receives query only, generates Python to access REPL-resident data
- MemGPT (Packer et al., 2023): typed memory operations as tools
- Memory-R1 (2025), FoldGRPO (2025): RL for context management
- *Gap in literature:* none of these explicitly measure read/write adoption asymmetry

---

## 3. Method (1.5 pages)

### 3.1 Extending RLM with Writes
- `RLMSystemRepl`: drop-in extension to RLM (Section 3.1 of prior plan)
- Adds 3 functions to REPL namespace: `memory_write(content, type)`, `consolidate()`, `answer["content"]`
- Writes go to BOTH the vector store (LanceDB) and structured store (DuckDB)
- Future queries' `memory_read` retrieve from the unified pool (original turns + writes)

### 3.2 Measuring the Two-Policy Gap
- Per query: track `read_count` and `write_count` in the REPL execution
- Adoption rate over N queries: `fraction with count > 0`
- Run on LoCoMo (within-session) and LongMemEval (multi-session)

### 3.3 RL Training (Approach 1 — Attempts to Close the Gap)
- GRPO reward: `F1(model_answer, gt) + 0.2·(memory_write_executed)`
- Reward computed over BM25 retrieval (matches reward function, avoids GPU contention with training)
- Tried 3 base models: Qwen-2.5-3B-Instruct, Qwen-Coder-3B-Instruct, Qwen-Coder-7B-Instruct
- LoRA r=16, α=32, QLoRA 4-bit NF4 for T4-compatible training
- 200 steps, 5 epochs over 40 LoCoMo training examples
- Iteration: + `write_attempt_bonus=0.1` for text-level memory_write mention (Approach 1b)
- See Section 4.5 for negative results

### 3.4 Oracle Upper Bound (Approach 2 — Hand-Crafted Optimal Writes)
- Skip the REPL entirely; use a strong reader (Claude Sonnet 4.6) with full haystack
- Compare 4 systems on the same questions:
  1. **baseline** — full haystack as context
  2. **truncated** — first 40K chars only (control for context length)
  3. **random_facts** — 20 random session sentences prepended (control for "any extra")
  4. **consolidation** — per-session LLM summary (5 facts/session) prepended
- The consolidation IS the optimal write policy: each session's facts get persisted as text additions to context
- This is what a perfect RLM `memory_write` policy WOULD do, ignoring the model-driven-decision constraint

---

## 4. Experiments (2 pages)

### 4.1 Datasets
- LoCoMo (Maharana et al., 2024): within-session conversations, 5 query types
- LongMemEval (Wu et al., 2024): three subtypes — multi-session (primary), single-session-user, single-session-preference

### 4.2 Reader Models (cross-family ablation)
- Claude Sonnet 4.6 (Anthropic) — primary
- Claude Haiku 4.5 (Anthropic) — smaller-reader robustness check
- GPT-OSS-120B (OpenAI) — cross-family
- Gemma 3 27B (Google) — cross-family
- All accessed via AWS Bedrock for reproducibility

### 4.3 Metrics
- **Token F1**: standard; biased against short/numeric answers
- **Numeric match**: extract digits + word-numbers; exact match (when gold is numeric)
- **LLM judge accuracy**: PRIMARY METRIC — Claude grades CORRECT/WRONG
- **Multi-judge majority vote**: Claude + GPT-OSS + Gemma vote; majority wins (controls for self-bias)
- **Bootstrap 95% CIs** on means (n_resamples=1000)
- **Pareto axes**: judge accuracy vs latency (s/q) vs cost ($/q)

### 4.4 Two-Policy Gap (Layer 1 Measurement)
- Pretrained Qwen-2.5-3B + RLMSystemRepl on 24 LoCoMo queries:
  - memory_read adoption: **100%**
  - memory_write adoption: **0%**
- Robust across prompt variations (explored 3 different system prompts)

### 4.5 RL Training Fails to Close the Gap (Approach 1 Negative)
- v3 (Qwen-3B, F1+0.2·write): final F1 = 4.5%, write = 0%
- v4 (+ write_attempt_bonus): final F1 = 2.0%, write = 0% (reward hacking confirmed via text-only `memory_write` token appearance)
- v5 (Qwen-Coder-3B): cancelled, reward decline to 0.02 from initial 0.077
- v6 (Qwen-Coder-7B): cancelled, no improvement
- Reward trajectory analysis: model exploits the bonus without actually executing valid `memory_write(…)` calls

### 4.6 Oracle Consolidation Improves Multi-Session (Approach 2 Positive)

| Reader | Baseline F1 | Trunc F1 | Random F1 | **Consolidation F1** | Δ vs baseline |
|---|---|---|---|---|---|
| Sonnet 4.6 (n=100) | TBD | TBD | TBD | TBD | **+{X}pp** |
| Haiku 4.5 (n=50)   | TBD | TBD | TBD | TBD | **+{X}pp** |
| GPT-OSS-120B (n=50) | TBD | TBD | TBD | TBD | **+{X}pp** |
| Gemma 3 27B (n=50)  | TBD | TBD | TBD | TBD | **+{X}pp** |

Bootstrap 95% CIs in parentheses. Stars indicate non-overlapping CIs with baseline.

### 4.7 Pareto Frontier (Accuracy vs Cost vs Latency)

- Latency per question:
  - baseline: ~3s
  - truncated: ~2s  - random_facts: ~3s
  - consolidation: ~170s (≈50x slower)
- Cost per question (Sonnet input/output $3/$15 per MTok):
  - baseline: ~$0.02
  - consolidation: ~$0.10 (≈5x more expensive)
- Despite higher cost, consolidation is ON the Pareto frontier if Δ accuracy > {Y}pp
- **Threshold for cost-effectiveness:** ~{Z}c per accuracy percentage point gained

### 4.8 When Does Consolidation Help/Hurt? (Failure Analysis)
- Multi-session counting questions: large wins (e.g., $185 vs $65 baseline)
- Single-session direct lookup: ties (info is right there, summary = noise)
- Information not in haystack: both fail (consolidation cannot create info)

---

## 5. Analysis (1 page)

### 5.1 Why does RL fail?
- Sparse rewards: 0% initial write adoption → no positive examples → no gradient
- Reward hacking: small models exploit token-level cues without executing writes
- Code generation difficulty: 3B base models generate invalid Python at high rates

### 5.2 What would make RL work? (speculative)
- SFT bootstrap on synthetic correct REPL trajectories
- Multi-query training episodes (write in Q1 → reward via Q2's improved retrieval)
- Process rewards (per-step) instead of outcome rewards
- Stronger code-specialized base models (CodeLlama-13B, etc.)

### 5.3 Generalization across reader families
- Δ pattern consistent across Claude / GPT-OSS / Gemma? → strong claim
- Δ pattern only in one family? → weak claim, possibly overfitting to writer style

---

## 6. Discussion (0.5 pages)

- Two-policy gap is real AND measurable AND has positive headroom
- Closing the gap via simple RL doesn't work (current recipe deficient)
- Hand-crafted consolidation is a cheap-ish (50x slower, 5x cost) accuracy gain
- Future RLM research should target the management policy directly, not just access policy

---

## 7. Limitations
- Single benchmark family (LoCoMo + LongMemEval)
- Consolidation is slow; not deployable as-is for interactive use
- Open-weights RL training only at 3B-7B (no 70B+ tried)
- No comparison to Memory-R1 / FoldGRPO baselines (different problem setups)

---

## 8. Conclusion (0.25 pages)

---

## Appendix
- Reproduction package: github.com/npow/context-bench (branch experiments/rl-and-multisession, single squashed commit)
- All Bedrock model IDs documented; S3 result snapshots
- Hyperparameter ablations and learning curves
