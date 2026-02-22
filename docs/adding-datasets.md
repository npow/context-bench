# Adding datasets

## Contract

A dataset is any `Iterable[dict[str, Any]]`. Each dict must have at minimum:

- `"id"` — unique example identifier (str or int)
- `"context"` — the text context to be transformed

Additional fields (e.g. `"question"`, `"answer"`, `"tools"`) are domain-specific
and passed through to evaluators.

## Using the registry

```python
from context_bench.registry import registry

def load_my_dataset(n: int | None = None) -> list[dict]:
    examples = [...]  # load your data
    return examples[:n] if n else examples

registry.register("dataset", "my_dataset", load_my_dataset)
```

## HuggingFace datasets

See `src/context_bench/datasets/huggingface.py` for examples of wrapping
HuggingFace datasets into the expected format.

## Local JSONL

```python
from context_bench.datasets.local import load_jsonl

dataset = load_jsonl("path/to/data.jsonl", n=100)
```

Each line must be valid JSON with at least `"id"` and `"context"` fields.
