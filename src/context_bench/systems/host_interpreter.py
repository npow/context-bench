"""Host-process Python interpreter for DSPy RLM.

Executes code directly in the host Python process (not Pyodide/WASM),
giving full access to native extensions like lancedb and duckdb.

Implements the CodeInterpreter protocol from dspy.primitives.code_interpreter.
"""

from __future__ import annotations

import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Callable

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput


class HostInterpreter:
    """Execute Python code in the host process with persistent namespace.

    This interpreter maintains a persistent namespace across execute() calls,
    allowing variables defined in one call to be used in subsequent calls.

    Tools (like llm_query) are injected into the namespace as callable functions.
    A SUBMIT() function is provided that signals completion by returning FinalOutput.
    """

    def __init__(
        self,
        tools: dict[str, Callable[..., str]] | None = None,
        output_fields: list[dict] | None = None,
    ):
        self.tools: dict[str, Callable[..., str]] = dict(tools or {})
        self.output_fields = output_fields or []
        self._namespace: dict[str, Any] = {}
        self._started = False
        self._shutdown_flag = False
        self._submit_called = False
        self._submit_value: Any = None

    def start(self) -> None:
        """Initialize the interpreter namespace."""
        if self._started:
            return
        self._namespace = {"__builtins__": __builtins__}
        self._started = True

    def _make_submit(self) -> Callable:
        """Create a SUBMIT function that captures the output."""
        # Build typed signature from output_fields
        field_names = [f["name"] for f in self.output_fields] if self.output_fields else []

        def SUBMIT(**kwargs: Any) -> None:
            # Also support positional for single-field case
            if not kwargs and field_names:
                raise ValueError(
                    f"SUBMIT requires keyword arguments: SUBMIT({', '.join(f'{n}=...' for n in field_names)})"
                )
            self._submit_called = True
            self._submit_value = kwargs

        # Also support SUBMIT(answer="...") style
        def SUBMIT_flexible(*args: Any, **kwargs: Any) -> None:
            if args and not kwargs and len(field_names) == 1:
                # Single positional arg maps to single output field
                kwargs = {field_names[0]: args[0]}
            elif args and not kwargs:
                raise ValueError(
                    f"SUBMIT requires keyword arguments: SUBMIT({', '.join(f'{n}=...' for n in field_names)})"
                )
            self._submit_called = True
            self._submit_value = kwargs

        return SUBMIT_flexible

    def execute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        """Execute Python code in the host process.

        Args:
            code: Python code to execute
            variables: Variables to inject into the namespace

        Returns:
            FinalOutput if SUBMIT() was called, else captured stdout as string

        Raises:
            CodeInterpreterError: On runtime errors
            SyntaxError: On invalid Python syntax
        """
        if self._shutdown_flag:
            raise CodeInterpreterError("Interpreter has been shut down")

        if not self._started:
            self.start()

        # Reset submit state
        self._submit_called = False
        self._submit_value = None

        # Inject variables
        if variables:
            self._namespace.update(variables)

        # Inject tools
        for name, func in self.tools.items():
            self._namespace[name] = func

        # Inject SUBMIT
        self._namespace["SUBMIT"] = self._make_submit()

        # Capture stdout
        stdout_capture = io.StringIO()

        try:
            # Compile first to get SyntaxError separately
            compiled = compile(code, "<rlm>", "exec")

            with redirect_stdout(stdout_capture):
                exec(compiled, self._namespace)

        except SyntaxError:
            raise  # Let SyntaxError pass through
        except SystemExit:
            raise CodeInterpreterError("SystemExit called in code")
        except Exception as e:
            tb = traceback.format_exc()
            # Include stdout captured before error
            pre_output = stdout_capture.getvalue()
            error_msg = f"{type(e).__name__}: {e}"
            if pre_output:
                error_msg = f"{pre_output}\n{error_msg}"
            raise CodeInterpreterError(error_msg) from e

        # Check if SUBMIT was called
        if self._submit_called:
            return FinalOutput(self._submit_value)

        # Return captured stdout
        output = stdout_capture.getvalue()
        return output if output else None

    def shutdown(self) -> None:
        """Clean up the interpreter."""
        self._shutdown_flag = True
        self._namespace.clear()

    def reset(self) -> None:
        """Reset for reuse."""
        self._namespace = {"__builtins__": __builtins__}
        self._submit_called = False
        self._submit_value = None
        self._shutdown_flag = False
        self._started = False
