"""Claude CLI system for context-bench.

Uses `claude -p` to run prompts through Claude Code in non-interactive mode.
Useful for end-to-end quality evaluation of context compression.
"""

from __future__ import annotations

import subprocess
from typing import Any


class ClaudeCLI:
    """System that sends examples through the Claude CLI.

    Args:
        name: Display name for benchmark results.
        model: Model to use (e.g. "sonnet", "opus", "haiku").
        max_turns: Maximum agentic turns.
        allowed_tools: Tools to allow (e.g. "Bash,Read,Edit").
        system_prompt: Custom system prompt.
        timeout: Subprocess timeout in seconds.
        build_prompt: Custom function to construct the prompt from an example dict.
            Receives the example and returns a string. If None, uses default
            logic (context + question).
    """

    def __init__(
        self,
        name: str = "claude",
        model: str | None = None,
        max_turns: int | None = None,
        allowed_tools: str | None = None,
        system_prompt: str | None = None,
        timeout: float = 120.0,
        build_prompt: Any = None,
    ) -> None:
        self._name = name
        self._model = model
        self._max_turns = max_turns
        self._allowed_tools = allowed_tools
        self._system_prompt = system_prompt
        self._timeout = timeout
        self._build_prompt = build_prompt

    @property
    def name(self) -> str:
        return self._name

    def process(self, example: dict[str, Any]) -> dict[str, Any]:
        """Run the example through claude -p and return the response."""
        prompt = self._make_prompt(example)
        response = self._run(prompt)
        return {**example, "response": response}

    def _make_prompt(self, example: dict[str, Any]) -> str:
        if self._build_prompt is not None:
            return self._build_prompt(example)

        context = example.get("context", "")
        question = example.get("question", "")

        if question:
            return f"{context}\n\nQuestion: {question}\n\nAnswer concisely."
        return context

    def _run(self, prompt: str) -> str:
        cmd = ["claude", "-p", "--print"]
        if self._model:
            cmd.extend(["--model", self._model])
        if self._system_prompt:
            cmd.extend(["--system-prompt", self._system_prompt])
        if self._allowed_tools:
            cmd.extend(["--allowedTools", self._allowed_tools])

        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            return "[TIMEOUT]"
        except FileNotFoundError:
            raise RuntimeError(
                "claude CLI not found. Install Claude Code: https://claude.ai/claude-code"
            )
