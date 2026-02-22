# Architecture

## Data flow

```
Dataset (iterable of dicts)
    │
    ▼
┌─────────┐     ┌───────────┐     ┌────────┐
│ System   │────▶│ Evaluator │────▶│ Metric │
│ .process │     │ .score    │     │.compute│
└─────────┘     └───────────┘     └────────┘
    │                 │                │
    ▼                 ▼                ▼
  output dict    scores dict     summary dict
```

1. **Dataset** yields `dict[str, Any]` examples
2. **System.process()** transforms each example, returns a new dict
3. **Evaluator.score()** compares original and processed, returns metric scores
4. Each (system, example) pair becomes an **EvalRow**
5. **Metric.compute()** aggregates rows into summary statistics
6. Everything is collected into an **EvalResult**

## Registry

Datasets, metrics, and reporters can be registered by name for config-driven usage:

```python
from context_bench.registry import registry
registry.register("dataset", "my_data", my_loader_fn)
```

## Tokenization

Token counting uses tiktoken by default (cl100k_base). Override by passing a custom
callable to the token counting utilities in `context_bench.utils.tokens`.
