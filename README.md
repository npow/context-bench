# context-bench

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/npow/context-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/npow/context-bench/actions/workflows/ci.yml)

**Measure whether your LLM context system actually works — in one command.**

You built something that sits between a user and an LLM: a context compressor, a RAG pipeline, a memory system, a reranker. Does it improve answers? Does it make them worse? Is it worth the tokens? context-bench runs your system against 42+ standard datasets and tells you.

---

## What you can do with it

### Compare memory systems on long-conversation QA

Does your memory system actually remember things from a 400-turn conversation? context-bench evaluates this on [LoCoMo](https://huggingface.co/datasets/snap-research/LoCoMo), a benchmark of long conversations with temporal, multi-hop, and adversarial questions.

Here's a real run comparing three built-in systems — full context stuffing (naive), semantic search (embedding), and multi-strategy retrieval + LLM (rlm):

```
$ context-bench memory \
    --system naive --system embedding --system rlm \
    --relay http://localhost:18082 --model sonnet \
    --dataset locomo -n 1

| System         |   F1   | Judge | Token Efficiency |
|----------------|--------|-------|------------------|
| naive          | 0.380  |  0.67 |            0.238 |
| embedding_top50| 0.429  |  0.78 |            0.243 |
| rlm            | 0.433  |  0.78 |            0.251 |
```

The naive system stuffs all 419 turns into the prompt (~14K tokens per query). The embedding system retrieves the top-50 most relevant turns (~1.5K tokens). RLM combines semantic search, keyword matching, and entity lookup. All scored against ground truth with both token-level F1 and an LLM judge.

### Benchmark any OpenAI-compatible endpoint

If your system exposes an OpenAI-compatible API (most do), point context-bench at it:

```
$ context-bench \
    --proxy http://localhost:7878 --name my-system \
    --proxy http://localhost:9091 --name baseline \
    --dataset hotpotqa -n 50

| System    | mean_score | pass_rate | compression_ratio | cost_of_pass |
|-----------|------------|-----------|-------------------|--------------|
| my-system | 0.364      | 0.364     | -0.135            | 2,447        |
| baseline  | 0.293      | 0.293     | -0.326            | 4,291        |
```

Works with any proxy, compressor, or middleware that speaks the OpenAI chat completions API. Evaluators are auto-wired — F1 for QA, code execution for HumanEval, LaTeX-aware matching for math, ROUGE for summarization.

### Let an LLM improve your pipeline automatically

The autoresearch loop uses Claude to iteratively mutate and test a retrieval pipeline:

```bash
uv run python3 loop.py \
  --relay http://localhost:18082 \
  --iterations 50 \
  --output-dir loop_results/ \
  --pipeline-path pipeline/sota_pipeline.py
```

Each iteration: Claude reads the pipeline source + score history, proposes one architectural change (add coreference resolution, try temporal ranking, etc.), the change is evaluated, and improvements are kept. Run it overnight with the watchdog script and come back to a better pipeline.

---

## Getting started

```bash
git clone https://github.com/npow/context-bench.git
cd context-bench
uv sync                       # core deps
uv sync --extra datasets      # + HuggingFace dataset loaders
```

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

### Try it: memory benchmark

The fastest way to see it work — evaluate the naive memory system on LoCoMo. You need an OpenAI-compatible endpoint (any LLM API that speaks `/v1/chat/completions`):

```bash
context-bench memory \
  --system naive \
  --relay http://localhost:18082 \
  --dataset locomo -n 1
```

### Try it: proxy benchmark

Benchmark any endpoint on HotpotQA:

```bash
context-bench \
  --proxy http://localhost:8080 \
  --dataset hotpotqa -n 20
```

### Try it: Python API

```python
from context_bench import evaluate, OpenAIProxy
from context_bench.evaluators import AnswerQuality
from context_bench.metrics import MeanScore, PassRate

result = evaluate(
    systems=[OpenAIProxy("http://localhost:8080", model="gpt-4")],
    dataset=your_data,
    evaluators=[AnswerQuality()],
    metrics=[MeanScore(score_field="f1"), PassRate(score_field="f1")],
)
print(result.summary)
```

---

## How it works

Every evaluation follows four steps:

```
Dataset → System → Evaluator → Metric
```

1. **Dataset** — examples with questions, context, and ground-truth answers
2. **System** — your thing: compresses, retrieves, transforms, or just forwards
3. **Evaluator** — scores the output against ground truth (F1, code execution, math equivalence, etc.)
4. **Metric** — aggregates across examples (mean score, pass rate, cost per pass, latency)

All interfaces are `typing.Protocol` — implement the methods, don't subclass. A complete custom system is 5 lines:

```python
class MySystem:
    name = "my-system"
    def process(self, example):
        transformed = my_transform(example["context"])
        return {**example, "context": transformed, "response": transformed}
```

For memory systems, the protocol is stateful:

```python
class MyMemory:
    name = "my-memory"
    def reset(self): ...
    def ingest(self, items: list[Item]) -> IngestResult: ...
    def query(self, question: str, budget: int | None = None) -> QueryResult: ...
```

---

## Datasets (42+)

Pick a dataset name — the right evaluator is wired automatically.

| Category | Datasets | How it's scored |
|----------|----------|-----------------|
| **QA** | hotpotqa, natural-questions, musique, narrativeqa, triviaqa, frames, quality, qasper | Token F1 + exact match |
| **Multiple choice** | mmlu, mmlu-pro, arc-challenge, truthfulqa, gpqa, hellaswag, winogrande | Letter accuracy |
| **Math** | gsm8k, math, drop, mgsm, bbh | LaTeX-aware equivalence |
| **Code** | humaneval, mbpp | Execution (pass@1) |
| **Summarization** | multi-news, dialogsum, qmsum, summscreenfd, meetingbank, govreport | ROUGE-L |
| **NLI** | contract-nli, scifact | Label match |
| **Instruction** | ifeval, alpaca-eval | Programmatic checks / LLM judge |
| **Long context** | longbench, longbench-v2, infinitebench, nolima | F1 |
| **Memory** | locomo, longmemeval | LLM judge + F1 |
| **Agent traces** | bfcl, apigen, swebench, swebench-verified, swebench-lite | Contains match |

Some are configurable: `mmlu:anatomy`, `mgsm:de`, `bbh:causal_judgement`, `longbench:qasper`.

Local files work too: `--dataset ./my_data.jsonl` (needs `"id"` and `"context"` keys).

---

## Built-in systems

### Context systems (for proxy benchmarking)

| System | What it does |
|--------|-------------|
| `OpenAIProxy` | Forwards to any OpenAI-compatible endpoint |
| `ClaudeCLI` | Routes through the `claude` CLI |
| `NaiveSystem` | Baseline — sends full context unmodified |

### Memory systems (for `context-bench memory`)

| System | Strategy | Install |
|--------|----------|---------|
| `naive` | Stuff all turns into the prompt | — |
| `embedding` | Semantic search via sentence-transformers | — |
| `rlm` | Multi-strategy retrieval (semantic + keyword + entity) + LLM | `pip install lancedb duckdb` |
| `mem0` | Mem0 managed memory | `uv sync --extra mem0` |
| `zep` | Zep/Graphiti temporal knowledge graph | `uv sync --extra zep` |

---

## Memory evaluation details

The memory benchmark tests whether a system can answer questions about conversations it ingested. Each example is a real multi-hundred-turn conversation from LoCoMo or LongMemEval.

**Item types** the system receives during ingestion:

| Type | Example |
|------|---------|
| `ConversationTurn` | Chat messages with timestamps, speakers, and session IDs |
| `DocumentChunk` | RAG document chunks with positions |
| `PlatformEvent` | Slack messages, Git commits, Linear tickets |
| `Declaration` | Explicit user preferences and facts |

**Evaluators:**

| Evaluator | What it checks |
|-----------|---------------|
| `AnswerQuality` | Token F1, exact match, recall against ground truth |
| `LLMJudgeLoCoMo` | LLM rates answer quality with evidence matching |
| `FalseMemoryRate` | Detects hallucinated/fabricated memories |

Run with QA type filtering to focus on specific capabilities:

```bash
context-bench memory \
  --system rlm --relay http://localhost:18082 \
  --dataset locomo -n 3 --qa-types temporal,multi_hop
```

---

## Autoresearch loop details

The loop (`loop.py`) evolves a retrieval pipeline using LLM-proposed mutations.

**Cycle:** read pipeline source + scores → Claude proposes one change → evaluate on held-out data → keep if better → repeat.

```bash
uv run python3 loop.py \
  --relay http://localhost:18082 \
  --model sonnet \
  --dataset locomo \
  --iterations 50 \
  --eval-n 2 \
  --max-qa-per-conv 5 \
  --seed 42 \
  --output-dir loop_results/ \
  --pipeline-path pipeline/sota_pipeline.py \
  --resume   # pick up from last checkpoint
```

**Starting pipelines:**

- `sota_pipeline.py` — strong baseline: coreference resolution, entity-relation triples, temporal ranking, query decomposition, multi-hop retrieval, embedding fallback
- `entity_pipeline.py` — simpler entity extraction (~19% F1)

**Overnight operation:**

```bash
bash watchdog6.sh   # checks every 10 min, restarts on crash or >150 min stall
```

**Output:**

```
loop_results/
├── loop_log.jsonl       # every iteration: score, mutation description, accepted?
├── context_pipeline.py  # current best pipeline (evolves over time)
├── baseline_score.json  # cached baseline score
├── run.log              # full stdout
└── watchdog.log         # watchdog heartbeat + restart events
```

---

## CLI reference

```bash
context-bench [options]           # benchmark context proxies
context-bench memory [options]    # benchmark memory systems
```

**Proxy benchmark:**

| Flag | Default | Description |
|------|---------|-------------|
| `--proxy URL` | *(required)* | Endpoint URL (repeatable for comparison) |
| `--name NAME` | hostname | Display name (paired with `--proxy`) |
| `--dataset NAME` | *(required)* | Dataset name or `.jsonl` path (repeatable) |
| `--model MODEL` | `gpt-4` | Model name passed to the proxy |
| `-n` | all | Max examples per dataset |
| `--output {table,json,html}` | `table` | Output format |
| `--score-field` | `f1` | Which score to aggregate |
| `--threshold` | `0.7` | Pass/fail cutoff |
| `--judge-url URL` | — | LLM-as-judge endpoint |
| `--max-workers N` | 1 | Concurrent threads |
| `--cache-dir DIR` | — | Enable caching for resume |

**Memory benchmark:**

| Flag | Default | Description |
|------|---------|-------------|
| `--system NAME` | *(required)* | System to evaluate: naive, embedding, rlm, mem0, zep (repeatable) |
| `--relay URL` | *(required)* | OpenAI-compatible LLM endpoint |
| `--dataset NAME` | locomo | locomo or longmemeval (repeatable) |
| `--model MODEL` | haiku | Model name |
| `-n` | all | Max conversations |
| `--qa-types TYPES` | all | Comma-separated: single_hop, temporal, multi_hop, etc. |

---

## Metrics reference

| Metric | What it measures |
|--------|------------------|
| `MeanScore` | Average score across examples |
| `PassRate` | % of examples above a threshold |
| `CompressionRatio` | 1 - (output_tokens / input_tokens) |
| `CostOfPass` | Tokens per successful completion |
| `Latency` | mean, median, p95, p99 |
| `PerDatasetBreakdown` | Score sliced by dataset |
| `ParetoRank` | Quality-vs-cost Pareto frontier |

---

## Installation options

```bash
uv sync                    # core
uv sync --extra datasets   # + HuggingFace dataset loaders
uv sync --extra mem0       # + Mem0 integration
uv sync --extra zep        # + Zep integration
uv sync --extra dspy       # + DSPy optimizer
uv sync --all-extras       # everything
uv sync --group dev        # development (pytest)
```

## Tests

```bash
uv run pytest
```

## Project structure

```
src/context_bench/
├── __main__.py          # CLI (context-bench command)
├── types.py             # Protocol definitions (System, Evaluator, Metric)
├── memory_types.py      # Memory protocol + typed items
├── runner.py            # evaluate() orchestration
├── memory_runner.py     # evaluate_memory() orchestration
├── results.py           # EvalRow / EvalResult
├── cache.py             # JSONL caching for resume
├── systems/             # OpenAIProxy, RLM, Embedding, Mem0, Zep, Naive, ...
├── datasets/            # 42+ loaders (includes memory/locomo, memory/longmemeval)
├── evaluators/          # 11 evaluators (F1, MC, code, math, ROUGE, NLI, LLM judge, ...)
├── metrics/             # 7 metrics
├── reporters/           # Markdown, JSON, HTML output
├── loop/mutator.py      # LLM-driven pipeline mutator
└── dspy_bench/          # DSPy optimizer integration

pipeline/
├── sota_pipeline.py     # Strong baseline for autoresearch
└── entity_pipeline.py   # Simple entity-based baseline
```

## License

[MIT](LICENSE)
