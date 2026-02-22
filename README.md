# context-bench

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-brightgreen.svg)](#running-tests)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Benchmark any system that transforms LLM context.**

Prompt compressors, memory managers, context stuffers, RAG rerankers — if it touches the context window before an LLM sees it, context-bench measures how well it works and what it costs.

---

## Why context-bench?

You built (or bought) something that modifies LLM context. Now you need to answer:

- **Does compression destroy information?** Measure quality with F1, exact match, and pass rate against ground-truth QA datasets.
- **Is the cost worth it?** Track compression ratio and cost-per-successful-completion side by side.
- **Which approach wins?** Run multiple systems on the same dataset in one call and get a comparison table.

context-bench gives you a **single `evaluate()` call** that runs your system against a dataset, scores every example, and aggregates the results — no boilerplate, no framework lock-in.

## Quick start

```bash
pip install -e .
```

Benchmark an OpenAI-compatible proxy in **3 lines**:

```python
from context_bench import OpenAIProxy, evaluate
from context_bench.metrics import MeanScore, PassRate

proxy = OpenAIProxy("http://localhost:8080", model="gpt-3.5-turbo")
result = evaluate(
    systems=[proxy],
    dataset=your_dataset,
    evaluators=[your_evaluator],
    metrics=[MeanScore(), PassRate()],
    text_fields=["response"],   # count only the proxy output tokens
)
print(result.summary)
```

Or define a **custom system** — just `.name` and `.process()`:

```python
from context_bench import evaluate
from context_bench.metrics import CompressionRatio, MeanScore

# 1. Define your system — just .name and .process()
class MyProxy:
    name = "my_proxy"

    def process(self, example):
        result = dict(example)
        # Your compression / transformation logic here
        result["context"] = your_compressor(example["context"])
        return result

# 2. Define how to score — just .name and .score()
class QAEvaluator:
    name = "qa_accuracy"

    def score(self, original, processed):
        # Compare processed output against ground truth
        expected = original["answer"]
        actual = run_qa(processed["context"], original["question"])
        return {"score": 1.0 if actual == expected else 0.0}

# 3. Run it
result = evaluate(
    systems=[MyProxy()],
    dataset=your_dataset,         # any list of dicts with "id" and "context"
    evaluators=[QAEvaluator()],
    metrics=[MeanScore(), CompressionRatio()],
)

print(result.summary)
# {'my_proxy': {'mean_score': 0.85, 'compression_ratio': 0.42, ...}}
```

## How it works

```
Dataset (dicts)
    │
    ▼
┌─────────┐     ┌───────────┐     ┌────────┐
│ System   │────▶│ Evaluator │────▶│ Metric │
│.process()│     │ .score()  │     │.compute│
└─────────┘     └───────────┘     └────────┘
    │                 │                │
    ▼                 ▼                ▼
 output dict    scores dict     summary dict
```

1. **Dataset** — any `Iterable[dict]`. Must have `"id"` and `"context"` keys.
2. **System** — implements `.name` and `.process(example) -> dict`. This is the thing you're benchmarking.
3. **Evaluator** — implements `.name` and `.score(original, processed) -> dict[str, float]`. Compares before/after.
4. **Metric** — implements `.name` and `.compute(rows) -> dict[str, float]`. Aggregates scores across examples.

All interfaces are [typing.Protocol](https://docs.python.org/3/library/typing.html#typing.Protocol) — implement the methods, don't subclass anything.

## Benchmark a proxy

The built-in `OpenAIProxy` system wraps any OpenAI-compatible endpoint. Point at a URL, get quality and cost metrics back — no HTTP boilerplate needed.

```python
from context_bench import OpenAIProxy, evaluate
from context_bench.metrics import CostOfPass, MeanScore, PassRate
from context_bench.metrics.quality import f1_score

proxy = OpenAIProxy(
    base_url="http://localhost:8080",
    model="gpt-3.5-turbo",
    # api_key="sk-...",              # or set OPENAI_API_KEY env var
    # system_prompt="Be concise.",   # prepended as system message
    # extra_body={"temperature": 0}, # any additional request params
)

class QAEvaluator:
    name = "qa_f1"
    def score(self, original, processed):
        return {"score": f1_score(processed.get("response", ""),
                                   original.get("answer", ""))}

result = evaluate(
    systems=[proxy],
    dataset=your_dataset,
    evaluators=[QAEvaluator()],
    metrics=[MeanScore(), PassRate(), CostOfPass()],
    text_fields=["response"],  # count only proxy output, not echoed context
)
```

> **`text_fields=["response"]`** — By default the runner counts all string fields for token stats, which would double-count context in output tokens. Pass `text_fields=["response"]` so only the proxy's actual output is measured.

### Custom message builders

Override how examples are mapped to chat messages:

```python
def my_messages(example):
    return [
        {"role": "system", "content": "You are a QA bot."},
        {"role": "user", "content": f"Context: {example['context']}\n\nQ: {example['question']}"},
    ]

proxy = OpenAIProxy("http://localhost:8080", build_messages=my_messages)
```

### Compare multiple proxies

```python
result = evaluate(
    systems=[
        OpenAIProxy("http://localhost:8080", model="gpt-3.5-turbo", name="gpt35"),
        OpenAIProxy("http://localhost:8080", model="gpt-4", name="gpt4"),
        OpenAIProxy("http://localhost:9090", name="my_custom_proxy"),
    ],
    dataset=dataset,
    evaluators=[QAEvaluator()],
    metrics=[MeanScore(), PassRate(), CostOfPass()],
    text_fields=["response"],
)
```

## Benchmarking your own proxy

Wrap any context transformation system in 5 minutes:

```python
class YourProxy:
    name = "my_compressor_v2"

    def process(self, example):
        result = dict(example)
        # Call your compressor, API, or library
        result["context"] = my_compressor.compress(example["context"])
        return result
```

That's it. No base class, no registration, no config file. Any object with `.name` and `.process()` works.

### Compare multiple systems head-to-head

```python
from context_bench import evaluate
from context_bench.metrics import CompressionRatio, CostOfPass, MeanScore, PassRate
from context_bench.reporters.markdown import to_markdown

result = evaluate(
    systems=[
        IdentityBaseline(),    # no-op baseline
        YourProxyV1(),         # your current approach
        YourProxyV2(),         # the new thing you're testing
    ],
    dataset=dataset,
    evaluators=[your_evaluator],
    metrics=[MeanScore(), PassRate(), CompressionRatio(), CostOfPass()],
)

print(to_markdown(result))
```

Output:

```
# Evaluation Results

| System          | mean_score | pass_rate | compression_ratio | cost_of_pass |
|-----------------|------------|-----------|-------------------|--------------|
| identity        | 1.0000     | 1.0000    | 0.0000            | 258.0000     |
| your_proxy_v1   | 0.9200     | 0.9000    | 0.3500            | 185.5556     |
| your_proxy_v2   | 0.8800     | 0.8500    | 0.5200            | 145.4118     |
```

### Export results

```python
result.to_json()          # JSON string
result.to_dataframe()     # pandas DataFrame (requires pandas)
result.filter(system="your_proxy_v2")  # filter to one system
```

## Built-in datasets

| Dataset | Domain | Loader | Install |
|---------|--------|--------|---------|
| [HotpotQA](https://hotpotqa.github.io/) | Multi-hop QA | `datasets.huggingface.hotpotqa()` | `pip install -e ".[datasets]"` |
| [GSM8K](https://github.com/openai/grade-school-math) | Math reasoning | `datasets.huggingface.gsm8k()` | `pip install -e ".[datasets]"` |
| [BFCL v3](https://gorilla.cs.berkeley.edu/leaderboard.html) | Function calling | `datasets.huggingface.bfcl_simple()` | `pip install -e ".[datasets]"` |
| [APIGen](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | Multi-turn tool use | `datasets.agent_traces.apigen_mt()` | `pip install -e ".[datasets]"` |
| [SWE-agent](https://swe-agent.com/) | Coding agent traces | `datasets.agent_traces.swe_agent_traces()` | `pip install -e ".[datasets]"` |
| Local JSONL | Any | `datasets.local.load_jsonl(path)` | Core |

Or bring your own — any `list[dict]` with `"id"` and `"context"` keys works.

## Built-in metrics

| Metric | What it measures |
|--------|------------------|
| `MeanScore` | Average score across all examples |
| `PassRate(threshold)` | Fraction of examples scoring above threshold |
| `CompressionRatio` | `1 - (output_tokens / input_tokens)` |
| `CostOfPass(threshold)` | Tokens spent per successful completion ([arXiv:2504.13359](https://arxiv.org/abs/2504.13359)) |
| `ParetoRank` | Rank on the quality-vs-cost Pareto frontier |
| `f1_score`, `exact_match`, `recall_score` | SQuAD-standard text comparison utilities |

## Installation

```bash
# Core (just tiktoken)
pip install -e .

# With HuggingFace dataset loaders
pip install -e ".[datasets]"

# Everything
pip install -e ".[all]"

# Development
pip install -e ".[dev]"
```

Requires **Python 3.10+**.

## Running tests

```bash
pytest
```

## Project structure

```
src/context_bench/
├── __init__.py          # Public API: evaluate, EvalResult, EvalRow, OpenAIProxy
├── types.py             # Protocol definitions (System, Evaluator, Metric)
├── runner.py            # Core evaluate() orchestration
├── results.py           # EvalRow / EvalResult dataclasses
├── registry.py          # Plugin system for named components
├── systems/             # Built-in systems (OpenAIProxy)
├── datasets/            # Built-in dataset loaders
├── metrics/             # MeanScore, PassRate, CompressionRatio, CostOfPass, ParetoRank
├── reporters/           # Markdown and JSON output formatters
└── utils/tokens.py      # Pluggable tokenizer (default: tiktoken cl100k_base)
```

## CI/CD

This project uses GitHub Actions for continuous integration:

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: pytest
```

## License

[MIT](LICENSE)
