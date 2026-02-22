# context-bench

Benchmark any system that transforms LLM context (compressors, prompt optimizers, memory managers).

## Quick orientation

- **Core loop**: `src/context_bench/runner.py` — `evaluate()` runs systems over datasets, scores with evaluators, aggregates with metrics.
- **Types**: `src/context_bench/types.py` — `System`, `Evaluator`, `Metric` are `Protocol` classes. Users implement these, never subclass.
- **Results**: `src/context_bench/results.py` — `EvalRow` (one system+example) and `EvalResult` (collection with summary).
- **Registry**: `src/context_bench/registry.py` — pluggable registration for datasets, metrics, reporters.

## Key design decisions

1. **Protocols, not inheritance** — all interfaces are `typing.Protocol`
2. **Dicts everywhere** — examples are `dict[str, Any]`, no domain models in core
3. **Flat results** — `EvalRow` is a flat dataclass, easy to convert to DataFrame/JSON
4. **Pluggable tokenizer** — default tiktoken, swap via `src/context_bench/utils/tokens.py`

## Docs

- [Architecture](docs/architecture.md) — system design and data flow
- [Adding datasets](docs/adding-datasets.md) — how to add new dataset loaders

## Testing

```bash
pip install -e ".[dev]"
pytest tests/
```

## Examples

```bash
python examples/minimal.py        # 20-line eval of a string truncator
python examples/compressor_eval.py # integration with compressor project
```
