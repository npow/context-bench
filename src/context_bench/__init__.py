"""context-bench: Benchmark any system that transforms LLM context."""

from context_bench.results import EvalResult, EvalRow
from context_bench.runner import evaluate
from context_bench.memory_runner import evaluate_memory
from context_bench.systems.claude_cli import ClaudeCLI
from context_bench.systems.openai_proxy import OpenAIProxy
from context_bench.types import MemorySystem

__all__ = [
    "evaluate",
    "evaluate_memory",
    "EvalResult",
    "EvalRow",
    "ClaudeCLI",
    "OpenAIProxy",
    "MemorySystem",
]
