"""Compile health detection for DSPy optimizer benchmarking.

Detects whether compilation produced a meaningful change by comparing
the initial and compiled programs based on configurable health checks.

See spec section 13: Compile Health Detection.
"""

from __future__ import annotations

from typing import Any


def assess_compile_health(
    initial_program: Any,
    compiled_program: Any,
    config: Any,
) -> dict:
    """Detect whether compilation produced a meaningful change.

    Args:
        initial_program: The DSPy Module before compilation.
        compiled_program: The DSPy Module after compilation.
        config: Object with .optimizer (str) and .health_checks (list[str]).

    Returns:
        Dictionary with keys:
            status: "healthy" | "degenerate" | "partial"
            reason: Human-readable explanation.
            demos_added: Number of demos added across all predictors.
            instruction_changed: Whether any predictor's instructions changed.
    """
    checks = config.health_checks

    # Compute metrics regardless of which checks are active.
    initial_demos = sum(len(p.demos) for p in initial_program.predictors())
    compiled_demos = sum(len(p.demos) for p in compiled_program.predictors())
    demos_added = compiled_demos - initial_demos

    initial_instrs = [p.signature.instructions for p in initial_program.predictors()]
    compiled_instrs = [p.signature.instructions for p in compiled_program.predictors()]
    instruction_changed = initial_instrs != compiled_instrs

    # Evaluate each active check: True = pass, False = fail.
    check_results: dict[str, bool] = {}

    if "demos_added" in checks:
        check_results["demos_added"] = demos_added > 0

    if "instruction_changed" in checks:
        # Passes if instructions changed OR demos were added (the program
        # was still meaningfully modified even if instructions stayed the same).
        check_results["instruction_changed"] = instruction_changed or demos_added > 0

    # Determine status from check results.
    reasons: list[str] = []

    if not check_results:
        # No checks configured → healthy by default.
        status = "healthy"
    elif all(check_results.values()):
        status = "healthy"
    elif not any(check_results.values()):
        status = "degenerate"
        if not check_results.get("demos_added", True):
            reasons.append("Zero demos added")
        if not check_results.get("instruction_changed", True):
            reasons.append("Instructions unchanged and no demos added")
    else:
        status = "partial"
        if not check_results.get("demos_added", True):
            reasons.append("Zero demos added")
        if not check_results.get("instruction_changed", True):
            reasons.append("Instructions unchanged")

    return {
        "status": status,
        "reason": "; ".join(reasons) or "OK",
        "demos_added": demos_added,
        "instruction_changed": instruction_changed,
    }
