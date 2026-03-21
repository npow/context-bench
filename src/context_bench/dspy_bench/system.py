"""DSPy optimizer system wrapper for context-bench.

Wraps a compiled DSPy program as a context-bench System, providing
thread-safe inference via thread-local deepcopy and minimal output
dicts to prevent token count inflation.
"""

import copy
import threading

class DspyOptimizerSystem:
    """Wraps a compiled DSPy program as a context-bench System.

    The compiled program is stored as a template; each thread gets its own
    deepcopy to avoid shared mutable state. Per-Predict LM references are
    cleared on construction so all Predicts fall through to ``dspy.settings.lm``.
    """

    def __init__(self, compiled_program, task_type: str, name: str):
        self._compiled = compiled_program
        self._task_type = task_type
        self._name = name
        self._local = threading.local()
        # Clear per-Predict LM references so all Predicts fall through to settings.lm
        for pred in self._compiled.predictors():
            pred.lm = None

    @property
    def name(self) -> str:
        return self._name

    def _get_program(self):
        """Return a thread-local deepcopy of the compiled program."""
        if not hasattr(self._local, "program"):
            self._local.program = copy.deepcopy(self._compiled)
        return self._local.program

    def process(self, example: dict) -> dict:
        """Run the compiled DSPy program on an example.

        Returns minimal dict: ``{id, dataset, response}`` to avoid
        token count inflation from spreading input fields.
        """
        from context_bench.dspy_bench.splits import TASK_INPUT_FIELDS, format_choices

        kwargs = {
            k: example[k] for k in TASK_INPUT_FIELDS[self._task_type] if k in example
        }
        if self._task_type == "mc" and isinstance(kwargs.get("choices"), list):
            kwargs["choices"] = format_choices(kwargs["choices"])
        try:
            pred = self._get_program()(**kwargs)
            response = pred.answer
        except Exception:
            response = ""
        return {
            "id": example.get("id", ""),
            "dataset": example.get("dataset", ""),
            "response": response,
        }
