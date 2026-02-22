"""Integration example: evaluate a context compressor.

Run with: python examples/compressor_eval.py

This example shows how to wrap an existing compressor as a context-bench System
and evaluate it against a simple baseline.
"""

from __future__ import annotations

from context_bench import evaluate
from context_bench.metrics import CompressionRatio, CostOfPass, MeanScore, PassRate
from context_bench.reporters.markdown import to_markdown


# --- Systems ---

class IdentityBaseline:
    """Passes context through unchanged."""
    name = "identity"

    def process(self, example: dict) -> dict:
        return dict(example)


class SimpleCompressor:
    """Demo compressor: removes duplicate lines and trims whitespace."""
    name = "dedup_compressor"

    def process(self, example: dict) -> dict:
        result = dict(example)
        lines = example.get("context", "").split("\n")
        seen: set[str] = set()
        unique: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                unique.append(stripped)
        result["context"] = "\n".join(unique)
        return result


# --- Evaluator ---

class ContextPreservationEvaluator:
    """Scores how well key information is preserved after compression."""
    name = "preservation"

    def score(self, original: dict, processed: dict) -> dict[str, float]:
        orig_ctx = original.get("context", "")
        proc_ctx = processed.get("context", "")

        if not orig_ctx:
            return {"score": 1.0}

        # Simple heuristic: check what fraction of unique words are preserved
        orig_words = set(orig_ctx.lower().split())
        proc_words = set(proc_ctx.lower().split())

        if not orig_words:
            return {"score": 1.0}

        preserved = len(orig_words & proc_words) / len(orig_words)
        return {"score": preserved}


# --- Dataset ---

def make_dataset(n: int = 20) -> list[dict]:
    """Generate synthetic examples with some redundancy."""
    examples = []
    for i in range(n):
        # Create context with intentional duplication
        base_lines = [
            f"The system processes request {i} for user authentication.",
            f"Database query returned {i * 10} results.",
            f"Cache hit ratio: {0.5 + (i % 5) * 0.1:.1f}",
            "Processing complete. Status: OK.",
        ]
        # Add duplicates
        lines = base_lines + base_lines[:2] + [""] + base_lines[1:]
        examples.append({
            "id": i,
            "context": "\n".join(lines),
            "question": f"What was the cache hit ratio for request {i}?",
            "answer": f"{0.5 + (i % 5) * 0.1:.1f}",
        })
    return examples


if __name__ == "__main__":
    dataset = make_dataset(20)

    result = evaluate(
        systems=[IdentityBaseline(), SimpleCompressor()],
        dataset=dataset,
        evaluators=[ContextPreservationEvaluator()],
        metrics=[
            MeanScore(),
            PassRate(threshold=0.7),
            CompressionRatio(),
            CostOfPass(threshold=0.7),
        ],
        progress=False,
    )

    print(to_markdown(result))
    print()

    for sys_name, metrics in result.summary.items():
        print(f"\n{sys_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
