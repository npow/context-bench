# context-bench

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/npow/context-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/npow/context-bench/actions/workflows/ci.yml)

**Measure whether your LLM context system actually works — in one command.**

You built something that sits between a user and an LLM: a context compressor, a RAG pipeline, a memory system, a reranker. Does it help? Does it make them worse? Is it worth the tokens? context-bench runs your system against 42+ standard datasets so you don't have to build your own eval harness.

---

## Results: LoCoMo long-conversation memory

[LoCoMo](https://huggingface.co/datasets/snap-research/LoCoMo) tests whether a system can answer questions about 400+ turn conversations — temporal reasoning, multi-hop lookups, adversarial traps. The original paper's best baseline (GPT-3.5-turbo-16K) scores **37.8% F1**. Human ceiling is **87.9% F1**.

Here's a real run from context-bench, 3 conversations, 70 queries across all question types:

| System | F1 | LLM Judge | Strategy |
|--------|-----|-----------|----------|
| naive | 0.383 | 0.63 | Stuff all 400+ turns into the prompt (~14K tokens/query) |
| embedding | 0.330 | 0.51 | Retrieve top-50 turns by semantic similarity (~1.5K tokens/query) |
| **rlm** | **0.431** | **0.69** | Multi-strategy retrieval (semantic + keyword + entity) → LLM answer |
| bm25_entity_v4 | 0.393 | 0.59 | Auto-evolved via autoresearch loop: BM25 + entity index + fact store |

**How this compares to published systems** (note: most report LLM-as-Judge, not raw F1):

| System | Metric | Score | Source |
|--------|--------|-------|--------|
| GPT-3.5-16K (paper baseline) | F1 | 37.8% | [Maharana et al. 2024](https://arxiv.org/abs/2402.17753) |
| **context-bench RLM** | **F1** | **43.1%** | This repo |
| Mem0 | LLM-Judge | 66.9% | [Mem0 paper](https://arxiv.org/html/2504.19413v1) |
| **context-bench RLM** | **LLM-Judge** | **69%** | This repo |
| Engram | LLM-Judge | 80.0% | [engram.fyi](https://www.engram.fyi/research) |
| Backboard.io | LLM-Judge | 90.1% | [Press release](https://www.einnews.com/pr_news/863886023/) |
| Human | F1 | 87.9% | Paper |

The RLM system outperforms the original paper's best baseline on F1 and is competitive with Mem0 on LLM-Judge — using only open-source components (sentence-transformers, LanceDB, DuckDB).

Per question type breakdown (RLM):

| Type | F1 | LLM Judge | What it tests |
|------|-----|-----------|---------------|
| single_hop | 0.48 | 0.73 | Direct fact lookup |
| multi_hop | 0.60 | 0.87 | Reasoning across multiple turns |
| open_domain | 0.65 | 0.80 | General knowledge from conversation |
| temporal | 0.13 | 0.40 | Time-sensitive questions ("what was X's job last year?") |
| adversarial | 0.02 | 0.07 | Questions about things never discussed |

Adversarial and temporal remain hard for all systems — same pattern the original paper found.

<details>
<summary>Reproduce these results</summary>

```bash
uv sync
# You need an OpenAI-compatible LLM endpoint running
context-bench memory \
  --system naive --system embedding --system rlm \
  --relay http://localhost:8080 \
  --model sonnet \
  --dataset locomo -n 3
```

Or via the Python API:

```python
from context_bench.datasets.memory.locomo import locomo
from context_bench.systems.rlm import RLMSystem
from context_bench.systems.naive import NaiveSystem
from context_bench.systems.embedding import EmbeddingSystem
from context_bench.evaluators.answer_quality import AnswerQuality
from context_bench.evaluators.llm_judge_locomo import LLMJudgeLoCoMo
from context_bench.memory_runner import evaluate_memory

examples = locomo(n=3)
systems = [
    NaiveSystem(base_url="http://localhost:8080", model="sonnet", api_key="unused"),
    EmbeddingSystem(base_url="http://localhost:8080", model="sonnet", top_k=50, api_key="unused"),
    RLMSystem(base_url="http://localhost:8080", model="sonnet", api_key="unused"),
]
evaluators = [AnswerQuality(), LLMJudgeLoCoMo(relay_url="http://localhost:8080", model="sonnet")]

result = evaluate_memory(systems=systems, dataset=examples, evaluators=evaluators)
for row in result.rows:
    print(f"{row.system}: f1={row.scores.get('f1', 0):.3f} judge={row.scores.get('llm_judge', 0):.1f}")
```
</details>

---

## What else can it do?

### Benchmark any OpenAI-compatible endpoint

Point it at any system that speaks `/v1/chat/completions`:

```bash
context-bench \
  --proxy http://localhost:7878 --name my-system \
  --proxy http://localhost:9091 --name baseline \
  --dataset hotpotqa -n 50
```

Evaluators are auto-wired — F1 for QA, code execution for HumanEval, LaTeX-aware matching for math, ROUGE for summarization. 42+ datasets across 10 categories.

### Let an LLM improve your pipeline automatically

The autoresearch loop uses Claude to iteratively mutate and test a retrieval pipeline. The `bm25_entity_v4` pipeline in the results above was produced this way — 40 iterations of propose-evaluate-keep:

```bash
uv run python3 loop.py \
  --relay http://localhost:8080 \
  --iterations 50 \
  --output-dir loop_results/ \
  --pipeline-path pipeline/sota_pipeline.py
```

---

## Getting started

```bash
git clone https://github.com/npow/context-bench.git
cd context-bench
uv sync                       # core deps
uv sync --extra datasets      # + HuggingFace dataset loaders
```

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

You'll need an OpenAI-compatible LLM endpoint for the systems that call an LLM (naive, rlm, and the LLM judge evaluator). Any endpoint that serves `/v1/chat/completions` works — OpenAI, Anthropic via a relay, vLLM, Ollama, etc.

### Memory benchmark (quickest way to see results)

```bash
context-bench memory \
  --system naive --system embedding --system rlm \
  --relay http://localhost:8080 \
  --dataset locomo -n 1
```

### Proxy benchmark

```bash
context-bench \
  --proxy http://localhost:8080 \
  --dataset hotpotqa -n 20
```

### Python API

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
3. **Evaluator** — scores the output against ground truth
4. **Metric** — aggregates across examples (mean score, pass rate, cost per pass, latency)

All interfaces are `typing.Protocol` — implement the methods, don't subclass. A complete custom system is 5 lines:

```python
class MySystem:
    name = "my-system"
    def process(self, example):
        transformed = my_transform(example["context"])
        return {**example, "context": transformed, "response": transformed}
```

For stateful memory systems:

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

Local JSONL files work too: `--dataset ./my_data.jsonl`

---

## Built-in systems

### Context systems (for `context-bench`)

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
| `rlm` | Semantic + keyword + entity retrieval → LLM answer | `pip install lancedb duckdb` |
| `mem0` | Mem0 managed memory | `uv sync --extra mem0` |
| `zep` | Zep/Graphiti temporal knowledge graph | `uv sync --extra zep` |

---

## Memory evaluation details

The memory benchmark tests whether a system can answer questions about conversations it ingested.

**Item types** the system receives during ingestion:

| Type | Example |
|------|---------|
| `ConversationTurn` | Chat messages with timestamps, speakers, session IDs |
| `DocumentChunk` | RAG document chunks |
| `PlatformEvent` | Slack messages, Git commits, etc. |
| `Declaration` | Explicit user preferences and facts |

**Evaluators:**

| Evaluator | What it checks |
|-----------|---------------|
| `AnswerQuality` | Token F1, exact match, recall against ground truth |
| `LLMJudgeLoCoMo` | LLM rates answer quality with evidence matching |
| `FalseMemoryRate` | Detects hallucinated/fabricated memories |

Filter by QA type to focus on specific capabilities:

```bash
context-bench memory \
  --system rlm --relay http://localhost:8080 \
  --dataset locomo -n 3 --qa-types temporal,multi_hop
```

---

## Autoresearch loop details

The loop (`loop.py`) evolves a retrieval pipeline by having Claude propose mutations.

**Cycle:** read pipeline source + scores → Claude proposes one change → evaluate → keep if better → repeat.

```bash
uv run python3 loop.py \
  --relay http://localhost:8080 \
  --model sonnet \
  --dataset locomo \
  --iterations 50 \
  --eval-n 2 \
  --seed 42 \
  --output-dir loop_results/ \
  --pipeline-path pipeline/sota_pipeline.py \
  --resume
```

**Starting pipelines:**

- `sota_pipeline.py` — coreference resolution, entity-relation triples, temporal ranking, query decomposition, multi-hop retrieval, embedding fallback
- `entity_pipeline.py` — simpler entity extraction baseline

**Overnight operation:**

```bash
bash watchdog6.sh   # checks every 10 min, restarts on crash/stall
```

**Output:**

```
loop_results/
├── loop_log.jsonl       # every iteration: score, mutation, accepted?
├── best_pipeline.py     # current best pipeline
├── baseline_score.json  # cached baseline
├── run.log              # full stdout
└── watchdog.log         # restart events
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
| `--name NAME` | hostname | Display name |
| `--dataset NAME` | *(required)* | Dataset or `.jsonl` path (repeatable) |
| `--model MODEL` | `gpt-4` | Model name |
| `-n` | all | Max examples |
| `--output {table,json,html}` | `table` | Output format |
| `--score-field` | `f1` | Score to aggregate |
| `--threshold` | `0.7` | Pass/fail cutoff |
| `--judge-url URL` | — | LLM-as-judge endpoint |
| `--max-workers N` | 1 | Concurrent threads |
| `--cache-dir DIR` | — | Cache for resume |

**Memory benchmark:**

| Flag | Default | Description |
|------|---------|-------------|
| `--system NAME` | *(required)* | naive, embedding, rlm, mem0, zep (repeatable) |
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
| `PassRate` | % above threshold |
| `CompressionRatio` | 1 - (output / input tokens) |
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
├── evaluators/          # 11 evaluators
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
