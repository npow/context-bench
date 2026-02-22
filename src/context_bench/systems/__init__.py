"""Built-in systems for context-bench."""

from context_bench.systems.claude_cli import ClaudeCLI
from context_bench.systems.openai_proxy import OpenAIProxy

__all__ = ["ClaudeCLI", "OpenAIProxy"]
