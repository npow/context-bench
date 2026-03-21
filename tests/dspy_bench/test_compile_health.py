"""Tests for compile health detection module.

Tests are written first (TDD) to validate the assess_compile_health function
against spec section 13 requirements.
"""

import pytest

from context_bench.dspy_bench.compile_health import assess_compile_health


# ---------------------------------------------------------------------------
# Mock helpers — avoid importing dspy
# ---------------------------------------------------------------------------

class MockPredictor:
    def __init__(self, demos=None, instructions="default"):
        self.demos = demos or []
        self.signature = type("Sig", (), {"instructions": instructions})()


class MockProgram:
    def __init__(self, predictors):
        self._predictors = predictors

    def predictors(self):
        return self._predictors


class MockConfig:
    def __init__(self, optimizer="test_optimizer", health_checks=None):
        self.optimizer = optimizer
        self.health_checks = health_checks or []


# ---------------------------------------------------------------------------
# Tests: demos_added check
# ---------------------------------------------------------------------------

class TestDemosAdded:
    def test_more_demos_in_compiled_program_is_healthy(self):
        initial = MockProgram([MockPredictor(demos=[])])
        compiled = MockProgram([MockPredictor(demos=["d1", "d2"])])
        config = MockConfig(health_checks=["demos_added"])

        result = assess_compile_health(initial, compiled, config)

        assert result["status"] == "healthy"
        assert result["demos_added"] == 2

    def test_zero_demos_added_is_degenerate(self):
        initial = MockProgram([MockPredictor(demos=[])])
        compiled = MockProgram([MockPredictor(demos=[])])
        config = MockConfig(health_checks=["demos_added"])

        result = assess_compile_health(initial, compiled, config)

        assert result["status"] == "degenerate"
        assert result["demos_added"] == 0

    def test_same_nonzero_demos_is_degenerate(self):
        """Even if both have demos, zero *added* demos triggers degenerate."""
        initial = MockProgram([MockPredictor(demos=["d1"])])
        compiled = MockProgram([MockPredictor(demos=["d1"])])
        config = MockConfig(health_checks=["demos_added"])

        result = assess_compile_health(initial, compiled, config)

        assert result["status"] == "degenerate"
        assert result["demos_added"] == 0

    def test_multiple_predictors_demos_summed(self):
        initial = MockProgram([
            MockPredictor(demos=[]),
            MockPredictor(demos=[]),
        ])
        compiled = MockProgram([
            MockPredictor(demos=["d1"]),
            MockPredictor(demos=["d2", "d3"]),
        ])
        config = MockConfig(health_checks=["demos_added"])

        result = assess_compile_health(initial, compiled, config)

        assert result["status"] == "healthy"
        assert result["demos_added"] == 3


# ---------------------------------------------------------------------------
# Tests: instruction_changed check
# ---------------------------------------------------------------------------

class TestInstructionChanged:
    def test_changed_instructions_is_healthy(self):
        initial = MockProgram([MockPredictor(instructions="original")])
        compiled = MockProgram([MockPredictor(instructions="optimized")])
        config = MockConfig(health_checks=["instruction_changed"])

        result = assess_compile_health(initial, compiled, config)

        assert result["status"] == "healthy"
        assert result["instruction_changed"] is True

    def test_unchanged_instructions_no_demos_is_degenerate(self):
        initial = MockProgram([MockPredictor(instructions="same")])
        compiled = MockProgram([MockPredictor(instructions="same")])
        config = MockConfig(health_checks=["instruction_changed"])

        result = assess_compile_health(initial, compiled, config)

        assert result["status"] == "degenerate"
        assert result["instruction_changed"] is False

    def test_unchanged_instructions_but_demos_added_is_healthy(self):
        """If instructions didn't change but demos were added, not degenerate."""
        initial = MockProgram([MockPredictor(demos=[], instructions="same")])
        compiled = MockProgram([MockPredictor(demos=["d1"], instructions="same")])
        config = MockConfig(health_checks=["instruction_changed"])

        result = assess_compile_health(initial, compiled, config)

        assert result["status"] == "healthy"
        assert result["instruction_changed"] is False
        assert result["demos_added"] == 1


# ---------------------------------------------------------------------------
# Tests: config-driven checks (not hardcoded)
# ---------------------------------------------------------------------------

class TestConfigDriven:
    def test_empty_health_checks_always_healthy(self):
        """With no checks configured, status should be healthy."""
        initial = MockProgram([MockPredictor(demos=[])])
        compiled = MockProgram([MockPredictor(demos=[])])
        config = MockConfig(health_checks=[])

        result = assess_compile_health(initial, compiled, config)

        assert result["status"] == "healthy"
        assert result["reason"] == "OK"

    def test_demos_check_not_listed_means_no_degenerate(self):
        """Zero demos added should not trigger degenerate if check not listed."""
        initial = MockProgram([MockPredictor(demos=[])])
        compiled = MockProgram([MockPredictor(demos=[])])
        config = MockConfig(health_checks=["instruction_changed"])

        # instruction_changed check alone: instructions are the same AND no demos
        # → degenerate (unchanged instructions + no demos)
        result = assess_compile_health(initial, compiled, config)
        assert result["status"] == "degenerate"

        # But if we have NO checks at all, it's healthy
        config2 = MockConfig(health_checks=[])
        result2 = assess_compile_health(initial, compiled, config2)
        assert result2["status"] == "healthy"

    def test_instruction_check_not_listed_means_no_degenerate(self):
        """Unchanged instructions should not trigger degenerate if check not listed."""
        initial = MockProgram([MockPredictor(demos=["d1"], instructions="same")])
        compiled = MockProgram([MockPredictor(demos=["d1"], instructions="same")])
        # Only demos_added check — no demos added → degenerate
        config = MockConfig(health_checks=["demos_added"])
        result = assess_compile_health(initial, compiled, config)
        assert result["status"] == "degenerate"


# ---------------------------------------------------------------------------
# Tests: partial status
# ---------------------------------------------------------------------------

class TestPartialStatus:
    def test_demos_added_rescues_instruction_check(self):
        """Both checks active: demos added (pass) and instruction_changed check
        also passes because demos were added (spec: instruction_changed only
        fails when instructions unchanged AND no demos added). → healthy."""
        initial = MockProgram([MockPredictor(demos=[], instructions="same")])
        compiled = MockProgram([MockPredictor(demos=["d1"], instructions="same")])
        config = MockConfig(health_checks=["demos_added", "instruction_changed"])

        result = assess_compile_health(initial, compiled, config)

        assert result["status"] == "healthy"
        assert result["demos_added"] == 1
        assert result["instruction_changed"] is False

    def test_partial_when_instructions_changed_but_no_demos_added(self):
        """Both checks active: instructions changed (pass) but no demos added (fail)
        → partial."""
        initial = MockProgram([MockPredictor(demos=[], instructions="old")])
        compiled = MockProgram([MockPredictor(demos=[], instructions="new")])
        config = MockConfig(health_checks=["demos_added", "instruction_changed"])

        result = assess_compile_health(initial, compiled, config)

        assert result["status"] == "partial"
        assert result["demos_added"] == 0
        assert result["instruction_changed"] is True

    def test_all_checks_pass_is_healthy(self):
        """Both checks active and both pass → healthy."""
        initial = MockProgram([MockPredictor(demos=[], instructions="old")])
        compiled = MockProgram([MockPredictor(demos=["d1"], instructions="new")])
        config = MockConfig(health_checks=["demos_added", "instruction_changed"])

        result = assess_compile_health(initial, compiled, config)

        assert result["status"] == "healthy"
        assert result["demos_added"] == 1
        assert result["instruction_changed"] is True

    def test_all_checks_fail_is_degenerate(self):
        """Both checks active and both fail → degenerate."""
        initial = MockProgram([MockPredictor(demos=[], instructions="same")])
        compiled = MockProgram([MockPredictor(demos=[], instructions="same")])
        config = MockConfig(health_checks=["demos_added", "instruction_changed"])

        result = assess_compile_health(initial, compiled, config)

        assert result["status"] == "degenerate"
        assert result["demos_added"] == 0
        assert result["instruction_changed"] is False


# ---------------------------------------------------------------------------
# Tests: return value structure
# ---------------------------------------------------------------------------

class TestReturnStructure:
    def test_return_keys(self):
        initial = MockProgram([MockPredictor()])
        compiled = MockProgram([MockPredictor()])
        config = MockConfig(health_checks=[])

        result = assess_compile_health(initial, compiled, config)

        assert set(result.keys()) == {"status", "reason", "demos_added", "instruction_changed"}

    def test_healthy_reason_is_ok(self):
        initial = MockProgram([MockPredictor(demos=[])])
        compiled = MockProgram([MockPredictor(demos=["d1"])])
        config = MockConfig(health_checks=["demos_added"])

        result = assess_compile_health(initial, compiled, config)

        assert result["reason"] == "OK"

    def test_degenerate_has_nonempty_reason(self):
        initial = MockProgram([MockPredictor(demos=[])])
        compiled = MockProgram([MockPredictor(demos=[])])
        config = MockConfig(health_checks=["demos_added"])

        result = assess_compile_health(initial, compiled, config)

        assert result["reason"] != "OK"
        assert len(result["reason"]) > 0
