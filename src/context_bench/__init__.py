"""context-bench: Benchmark any system that transforms LLM context."""

from context_bench.results import EvalResult, EvalRow
from context_bench.runner import evaluate
from context_bench.systems.openai_proxy import OpenAIProxy

__all__ = ["evaluate", "EvalResult", "EvalRow", "OpenAIProxy"]
