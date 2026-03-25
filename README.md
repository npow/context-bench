# context-bench

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/npow/context-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/npow/context-bench/actions/workflows/ci.yml)

**Measure whether your LLM context system actually works — in one command.**

You built something that sits between a user and an LLM: a context compressor, a RAG pipeline, a memory system, a reranker. Does it help? Does it hurt? Is it worth the tokens? context-bench runs your system against 42+ standard datasets so you don't have to build your own eval harness.

---

## Results: LoCoMo long-conversation memory

[LoCoMo](https://huggingface.co/datasets/snap-research/LoCoMo) tests whether a system can answer questions about 400–700 turn conversations — temporal reasoning, multi-hop lookups, adversarial traps. The original paper's best baseline (GPT-3.5-turbo-16K) scores **37.8% F1**. Human ceiling is **87.9% F1**.

Full benchmark — all 10 conversations, all 1,986 questions, scored with both token-level F1 and an LLM judge (Claude Sonnet as judge):

| System | F1 | LLM Judge | N | Strategy |
|--------|-----|-----------|---|----------|
| naive | 6.7% | 10.7% | 1,986 | Stuff all turns into the prompt |
| Mem0 | 6.6% | 27.6% | 199 | Extract-then-retrieve (Mem0 OSS) |
| embedding | 15.1% | 21.2% | 1,986 | Top-50 turns by semantic similarity |
| **rlm** | **37.4%** | **53.3%** | **1,986** | Multi-strategy retrieval (semantic + keyword + entity) → LLM |

Mem0 was run via `uv sync --extra mem0` with local HuggingFace embeddings and the same Claude Sonnet backend as all other systems. Its extraction-based approach (LLM calls during ingest) loses significant detail from long conversations — the extract step takes ~90 min for one 419-turn conversation.

Per question type (RLM system):

| Type | F1 | LLM Judge | N | What it tests |
|------|-----|-----------|---|---------------|
| single_hop | 35.7% | 55.3% | 282 | Direct fact lookup |
| multi_hop | 41.6% | 60.4% | 321 | Reasoning across multiple turns |
| open_domain | 49.9% | 69.1% | 841 | General knowledge from conversation |
| temporal | 23.2% | 55.2% | 96 | Time-sensitive questions |
| adversarial | 14.9% | 16.8% | 446 | Questions about things never discussed |

**How this compares to published systems:**

| System | Metric | Score | Source |
|--------|--------|-------|--------|
| context-bench Mem0 | F1 | 6.6% | This repo (N=199, ran ourselves) |
| GPT-3.5-16K (paper best) | F1 | 37.8% | [Maharana et al. 2024](https://arxiv.org/abs/2402.17753) |
| **context-bench RLM** | **F1** | **37.4%** | This repo (N=1,986, ran ourselves) |
| context-bench Mem0 | LLM-Judge | 27.6% | This repo (N=199, ran ourselves) |
| MemGPT | LLM-Judge | ~40-50% | [Letta blog](https://www.letta.com/blog/benchmarking-ai-agent-memory) |
| **context-bench RLM** | **LLM-Judge** | **53.3%** | This repo (N=1,986, ran ourselves) |
| Mem0 (published) | LLM-Judge | 66.9% | [Mem0 paper](https://arxiv.org/html/2504.19413v1) |
| Letta | LLM-Judge | 74.0% | [Letta blog](https://www.letta.com/blog/benchmarking-ai-agent-memory) |
| Engram | LLM-Judge | 80.0% | [engram.fyi](https://www.engram.fyi/research) |
| Human | F1 | 87.9% | Paper |

The RLM system matches the original paper's best baseline on raw F1 using only open-source components (sentence-transformers, LanceDB, DuckDB). Mem0's published 66.9% LLM-Judge score uses their managed cloud platform with GPT-4 as judge — our local OSS run with Claude Sonnet as judge scored 27.6%.

> **Note on metrics:** The original paper uses token-level F1. Most 2025+ systems report LLM-as-Judge scores, which are much more lenient (e.g. MemMachine reports 91.7% judge but only ~25% F1). Numbers across different metrics are not directly comparable. Judge model choice also matters — our results use Claude Sonnet, others use GPT-4/4o.

<details>
<summary>Reproduce these results</summary>

```bash
git clone https://github.com/npow/context-bench.git
cd context-bench
uv sync

# You need an OpenAI-compatible LLM endpoint (OpenAI, Anthropic relay, vLLM, Ollama, etc.)
# Full run (~7h for 3 systems x 10 conversations x 1,986 queries):
uv run python3 run_full_locomo.py

# Quick version (~15 min, 1 conversation):
context-bench memory \
  --system naive --system embedding --system rlm \
  --relay http://localhost:8080 \
  --dataset locomo -n 1
```

Or via Python:

```python
from context_bench.datasets.memory.locomo import locomo
from context_bench.systems.rlm import RLMSystem
from context_bench.systems.naive import NaiveSystem
from context_bench.systems.embedding import EmbeddingSystem
from context_bench.evaluators.answer_quality import AnswerQuality
from context_bench.evaluators.llm_judge_locomo import LLMJudgeLoCoMo
from context_bench.memory_runner import evaluate_memory

examples = locomo()  # all 10 conversations
systems = [
    NaiveSystem(base_url="http://localhost:8080", model="gpt-4", api_key="..."),
    EmbeddingSystem(base_url="http://localhost:8080", model="gpt-4", top_k=50, api_key="..."),
    RLMSystem(base_url="http://localhost:8080", model="gpt-4", api_key="..."),
]
evaluators = [AnswerQuality(), LLMJudgeLoCoMo(relay_url="http://localhost:8080", model="gpt-4")]

result = evaluate_memory(systems=systems, dataset=examples, evaluators=evaluators)
```
</details>

---

## Results: LongMemEval-M (ultra-long conversations)

[LongMemEval](https://huggingface.co/datasets/LongMemEval) tests memory over ~5,000-turn conversations (~776K words each) — much longer than LoCoMo. Six question types: knowledge updates, multi-session recall, single-session facts, preferences, and temporal reasoning.

30 examples (5 per question type), Claude Sonnet as both answerer and judge:

| System | F1 | LLM Judge | N | Strategy |
|--------|-----|-----------|---|----------|
| naive | 0.0% | 0.0% | 30 | Context too large — fails completely |
| rlm | 41.9% | 53.3% | 30 | Multi-strategy retrieval → LLM |
| **embedding** | **43.4%** | **56.7%** | 30 | Top-50 turns by semantic similarity |

Per question type (embedding system):

| Type | F1 | Judge | What it tests |
|------|-----|-------|---------------|
| single-session-assistant | 91.7% | 100% | What did the assistant say? |
| single-session-user | 65.7% | 80.0% | What did the user say? |
| knowledge-update | 37.7% | 40.0% | Info that changed over time |
| temporal-reasoning | 28.2% | 20.0% | When did something happen? |
| multi-session | 23.5% | 40.0% | Info spanning multiple sessions |
| single-session-preference | 13.4% | 60.0% | User preferences |

Context stuffing (naive) completely fails on 5,000-turn conversations. Embedding and RLM both handle it — embedding wins on this benchmark because pure semantic similarity works well for direct recall, while RLM's entity/keyword strategies add less value when conversations are this long and diverse.

<details>
<summary>Reproduce these results</summary>

```bash
uv run python3 run_full_longmemeval.py
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

The autoresearch loop uses Claude to iteratively mutate and test a retrieval pipeline:

```bash
uv run python3 loop.py \
  --relay http://localhost:8080 \
  --iterations 50 \
  --output-dir loop_results/ \
  --pipeline-path pipeline/sota_pipeline.py
```

Each iteration: Claude reads the pipeline source + score history, proposes one architectural change, evaluates it, keeps improvements. Run it overnight with the watchdog script.

---

## Getting started

```bash
git clone https://github.com/npow/context-bench.git
cd context-bench
uv sync                       # core deps
uv sync --extra datasets      # + HuggingFace dataset loaders
```

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

You need an OpenAI-compatible LLM endpoint for systems that call an LLM. Any endpoint that serves `/v1/chat/completions` works — OpenAI, Anthropic via a proxy, vLLM, Ollama, etc.

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

Filter by QA type:

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
