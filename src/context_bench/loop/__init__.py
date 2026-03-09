"""Autoresearch loop package for context-bench.

The loop autonomously mutates context_pipeline.py overnight, searching for
context engineering strategies that maximise token_efficiency (F1 per 1 000
input words) on long-conversation QA benchmarks.

Typical usage::

    # From the repo root:
    python loop.py \\
        --relay http://localhost:7878 \\
        --model claude-haiku-4-5-20251001 \\
        --dataset locomo \\
        --iterations 200 \\
        --eval-n 30 \\
        --seed 42 \\
        --output-dir ./loop_results

Public API::

    from context_bench.loop import PipelineMutator

    mutator = PipelineMutator(relay_url="http://localhost:7878", model="claude-haiku-4-5-20251001")
    new_code = mutator.propose_mutation(current_code, score_history)
"""

from context_bench.loop.mutator import PipelineMutator

__all__ = [
    "PipelineMutator",
]
