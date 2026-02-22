"""Minimal example: evaluate a string truncator.

Run with: python examples/minimal.py
"""

from context_bench import evaluate
from context_bench.metrics import CompressionRatio, MeanScore


class Truncator:
    name = "truncate_50pct"

    def process(self, example):
        result = dict(example)
        ctx = example["context"]
        result["context"] = ctx[: len(ctx) // 2]
        return result


class LengthScore:
    name = "length_preservation"

    def score(self, original, processed):
        orig = len(original.get("context", ""))
        proc = len(processed.get("context", ""))
        return {"score": proc / orig if orig else 1.0}


dataset = [
    {"id": i, "context": f"This is example number {i}. " * 20}
    for i in range(10)
]

result = evaluate(
    systems=[Truncator()],
    dataset=dataset,
    evaluators=[LengthScore()],
    metrics=[MeanScore(), CompressionRatio()],
    progress=False,
)

print(f"Mean score: {result.summary['truncate_50pct']['mean_score']:.3f}")
print(f"Compression: {result.summary['truncate_50pct']['compression_ratio']:.3f}")
print(f"\nJSON output:\n{result.to_json()[:500]}...")
