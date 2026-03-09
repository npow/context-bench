"""Pipeline package for the autoresearch loop.

The autoresearch agent mutates context_pipeline.py to discover better
context strategies. Use score_pipeline() to evaluate any change.

Quick start::

    from context_bench.pipeline import ContextPipeline, score_pipeline

    examples = [...]  # list of evaluate_memory()-compatible dicts
    result = score_pipeline(
        ContextPipeline,
        examples,
        relay_url="http://localhost:7878",
        model="claude-haiku-4-5-20251001",
    )
    print(result["score"])  # token_efficiency — the autoresearch target
"""

from context_bench.pipeline.context_pipeline import STRATEGY, ContextPipeline
from context_bench.pipeline.scoring import score_pipeline

__all__ = [
    "ContextPipeline",
    "STRATEGY",
    "score_pipeline",
]
