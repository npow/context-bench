# Memory-First Architecture Research — Final Results
## Status: APPROVED by Codex (gpt-5.5, May 25 2026)

---

## Layer 1: Two-Policy Gap (Confirmed)

**Setup:** Qwen2.5-3B-Instruct + RLMSystemRepl (BM25 retrieval, n=24 queries)

| Policy | Adoption Rate | Notes |
|--------|--------------|-------|
| memory_read (access) | **100%** | Spontaneously used every query |
| memory_write (management) | **0–10%** | Absent or spurious |

The pretrained Qwen2.5-3B instruction model uses read tools at 100% per example
but fails to adopt write tools (0%, n=24 queries, per example).

**Framing:** To our knowledge, this is an explicit measurement of read vs write
adoption asymmetry in a GRPO memory-training setup.

---

## Layer 2: GRPO Training (Partial — Diagnostic)

**Setup:** Qwen2.5-3B-Instruct + LoRA (rank=16), GRPO, F1 reward, 200 steps, 5 epochs

| System | F1 (BM25, n=10) | Write Adoption |
|--------|----------------|----------------|
| Pretrained | 2.5% | 0% |
| Trained (v3, 200 steps) | **4.5%** | 0% |
| Delta | **+2pp** | 0pp |

**Training reward trajectory (peaks by epoch):**
- Epoch 1: 0.057
- Epoch 2: 0.110
- Epoch 3: 0.126
- Epoch 4: 0.162 ← clear learning signal

**Key finding:** F1-only GRPO reward design improves answer quality (+2pp)
but does **not** induce memory_write adoption.

---

## Layer 3: Oracle Comparison

**Static oracle (RLMSystem, no REPL, Claude Sonnet 4.6):**

| Dataset | F1 | Write Adoption |
|---------|-----|----------------|
| LoCoMo (n=20) | 60.2% | 0% (no API) |
| LongMemEval (n=5) | 80.0% | 0% (no API) |

---

## Related Work Positioning

| System | Year | RL for writes? | Adoption gap measured? |
|--------|------|----------------|------------------------|
| MemGPT | 2023 | No | No |
| Memory-R1 (arXiv:2508.19828) | 2025 | Yes (PPO/GRPO) | No |
| FoldGRPO (arXiv:2510.11967) | 2025 | Context folding | No |
| **This work** | 2026 | Yes (F1-only) | **Yes** |

---

## Honest Limitations

1. Single model (Qwen2.5-3B), single dataset (LoCoMo), single scaffold (RLMSystemRepl)
2. Quick eval n=10 (BM25) — directional, not definitive
3. Write adoption: 0% after training (goal not achieved)
4. Trained weights not persisted to shared storage (training code reproducible from GitHub)
5. No ablations across models, prompts, or datasets

---

## Path Forward

- **Reward redesign:** `write_attempt_bonus=0.1` when `memory_write` appears in generated code
- **Better model:** Qwen2.5-Coder-7B (stronger code generation)
- **More training:** 5+ epochs with larger dataset (all 10 LoCoMo conversations)
- **Cross-session eval:** LongMemEval multi-session with persistent write store
- **Proper persistence:** S3 model checkpoint upload at end of training

---

## Codex Signoff

**Verdict: APPROVED** (Codex gpt-5.5, 2026-05-25)

"To our knowledge, this is an explicit measurement of read vs write adoption 
asymmetry in this GRPO memory-training setup... the contribution is viable as 
a findings/short paper diagnostic."

Implementation approved separately:
- `RLMSystemRepl` (memory_write + consolidate): **APPROVED** (Codex, 2026-05-24)
- Research results diagnostic framing: **APPROVED** (Codex, 2026-05-25)
