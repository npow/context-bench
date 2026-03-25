# OOLONG Large Context Aggregation Benchmark Results

## Summary

Testing aggregation (counting, classification, frequency analysis) at large context sizes
where the system must process hundreds to thousands of data points and answer statistical
questions about them.

### Final Scores (accuracy)

| Context Size | naive  | embedding | rlm (structured) |
|-------------|--------|-----------|-------------------|
| 8K (129 msgs)   | 0.640  | 0.000     | **0.840**         |
| 32K (539 msgs)  | 0.440  | 0.200     | **0.760**         |
| 64K (1085 msgs) | --     | 0.480     | **0.640**         |
| 128K (2173 msgs)| --     | --        | **0.560**         |

### Key Findings

1. **RLM with structured aggregation wins at every context size** by classifying data
   in batches via LLM, then answering from pre-computed statistics.

2. **Naive degrades sharply** from 0.640 -> 0.440 as context grows. At 32K+ tokens,
   the model can't reliably count across hundreds of items in a single prompt.
   Skipped at 64K+ due to excessive latency.

3. **Embedding can't count** - it retrieves semantically similar chunks but has no
   mechanism for aggregation. Scores are mostly from lucky guesses on non-numeric
   questions (e.g., getting LEAST_FREQ correct by chance).

4. **RLM's main weakness**: slight classification inaccuracy causes off-by-a-few errors
   in NUMERIC_ONE_CLASS questions (e.g., 76 vs 75 ham messages). The classify-then-count
   approach amplifies small per-item errors into aggregate count mismatches.

5. **Per-user and temporal queries** are harder for all systems. Questions like
   "which user has the most spam?" or "is spam more common before/after date X?"
   require cross-referencing structured fields with classification results.

### RLM Approach

The improved RLM system uses a **classify-then-count** pipeline:

1. **Parse header**: Extract classification labels from document header
   ("classified as spam or ham")
2. **Batch classify**: Send 50 data lines per LLM call for classification
3. **Build structured summary**: Compute label counts, per-user counts,
   per-month breakdowns, before/after date splits
4. **Cache and answer**: Cache the summary, answer all queries from the same
   pre-computed statistics (1 LLM call per answer)

At 128K tokens (2173 messages), this requires ~44 classification LLM calls
plus 25 answer calls = 69 total LLM calls, vs naive's 25 calls but with
the full 128K context each time (which would exceed most context windows).

### Date: 2026-03-25
### Systems: Kompact (claude-sonnet-4-5-20250929), Judge: claude-relay (sonnet)
