"""Code execution evaluator (pass@1).

Executes generated code against test cases in a subprocess with a timeout.
"""

from __future__ import annotations

import subprocess
import tempfile
from typing import Any


class CodeExecution:
    """Execute generated code against test cases and measure pass@1.

    Expects original dict to have ``test`` (HumanEval) or ``test_list``
    (MBPP) fields with test code, and optionally ``entry_point``.
    """

    def __init__(self, timeout: float = 10.0) -> None:
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "code_execution"

    def score(
        self, original: dict[str, Any], processed: dict[str, Any],
    ) -> dict[str, float]:
        response = str(processed.get("response", ""))
        if not response.strip():
            return {"pass_at_1": 0.0}

        # Build the test code
        test_code = self._build_test_code(original, response)
        if not test_code:
            return {"pass_at_1": 0.0}

        # Execute in subprocess
        passed = self._execute(test_code)
        return {"pass_at_1": 1.0 if passed else 0.0}

    def _build_test_code(
        self, original: dict[str, Any], response: str,
    ) -> str:
        """Combine prompt, response, and tests into executable code."""
        prompt = str(original.get("context", ""))
        entry_point = original.get("entry_point", "")

        # HumanEval style: prompt + response + check function
        test = original.get("test", "")
        if test:
            code = f"{prompt}{response}\n\n{test}\n"
            if entry_point:
                code += f"\ncheck({entry_point})\n"
            return code

        # MBPP style: response + test_list assertions
        test_list = original.get("test_list", [])
        if test_list:
            setup = original.get("test_setup_code", "")
            parts = [response]
            if setup:
                parts.append(setup)
            parts.extend(test_list)
            return "\n".join(parts) + "\n"

        return ""

    def _execute(self, code: str) -> bool:
        """Run code in a subprocess, return True if exit code == 0."""
        try:
            result = subprocess.run(
                ["python3", "-c", code],
                timeout=self._timeout,
                capture_output=True,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False
