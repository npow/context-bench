"""Benchmark Headroom context compression proxy.

Prerequisites:
    pip install "headroom-ai[proxy]"
    headroom proxy --port 8787

Run with:
    python examples/proxy_benchmark.py
"""

from context_bench import evaluate
from context_bench.metrics import CompressionRatio, CostOfPass, MeanScore, PassRate
from context_bench.metrics.quality import f1_score
from context_bench.reporters.markdown import to_markdown
from context_bench.systems import OpenAIProxy

# ---------------------------------------------------------------------------
# 1. Point at your proxy (Headroom, or any OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

headroom = OpenAIProxy(
    base_url="http://localhost:8787",
    model="claude-sonnet-4-5-20250929",
    name="headroom",
    # api_key="sk-...",          # or set OPENAI_API_KEY env var
    # extra_body={"temperature": 0, "max_tokens": 256},
)

# Optional: compare against a baseline without compression
baseline = OpenAIProxy(
    base_url="http://localhost:8080",
    model="claude-sonnet-4-5-20250929",
    name="baseline",
)

# ---------------------------------------------------------------------------
# 2. Build a small QA dataset
# ---------------------------------------------------------------------------

dataset = [
    {
        "id": 0,
        "context": "The Eiffel Tower is located in Paris, France. It was built in 1889.",
        "question": "Where is the Eiffel Tower?",
        "answer": "Paris",
    },
    {
        "id": 1,
        "context": "Python was created by Guido van Rossum and first released in 1991.",
        "question": "Who created Python?",
        "answer": "Guido van Rossum",
    },
    {
        "id": 2,
        "context": "The speed of light is approximately 299,792,458 meters per second.",
        "question": "What is the speed of light?",
        "answer": "299,792,458 meters per second",
    },
]

# ---------------------------------------------------------------------------
# 3. Define a simple evaluator
# ---------------------------------------------------------------------------


class QAEvaluator:
    name = "qa_f1"

    def score(self, original, processed):
        expected = original.get("answer", "")
        actual = processed.get("response", "")
        return {"score": f1_score(actual, expected)}


# ---------------------------------------------------------------------------
# 4. Run the benchmark
# ---------------------------------------------------------------------------

result = evaluate(
    systems=[headroom, baseline],
    dataset=dataset,
    evaluators=[QAEvaluator()],
    metrics=[MeanScore(), PassRate(threshold=0.5), CompressionRatio(), CostOfPass()],
    text_fields=["response"],  # count only proxy output, not the echoed context
)

print(to_markdown(result))
