"""Tests for the optimizer registry module."""

import pytest

dspy = pytest.importorskip("dspy")

from context_bench.dspy_bench.registry import REGISTRY, validate_registry


EXPECTED_OPTIMIZERS = [
    "LabeledFewShot",
    "BootstrapFewShot",
    "BootstrapFewShotWithRandomSearch",
    "COPRO",
    "MIPROv2",
    "SIMBA",
    "GEPA",
    "GEPA_ablation",
]

REQUIRED_KEYS = {
    "class",
    "accepts_metric",
    "init_kwargs",
    "build_compile_kwargs",
    "metric_factory",
    "health_checks",
    "prompt_model",
    "cost_heuristic",
}


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------


class TestRegistryCompleteness:
    def test_has_all_8_optimizers(self):
        for name in EXPECTED_OPTIMIZERS:
            assert name in REGISTRY, f"Missing optimizer: {name}"

    def test_no_extra_optimizers(self):
        assert set(REGISTRY.keys()) == set(EXPECTED_OPTIMIZERS)

    @pytest.mark.parametrize("name", EXPECTED_OPTIMIZERS)
    def test_entry_has_all_required_keys(self, name):
        entry = REGISTRY[name]
        missing = REQUIRED_KEYS - set(entry.keys())
        assert not missing, f"{name} missing keys: {missing}"


# ---------------------------------------------------------------------------
# accepts_metric
# ---------------------------------------------------------------------------


class TestAcceptsMetric:
    def test_labeled_few_shot_does_not_accept_metric(self):
        assert REGISTRY["LabeledFewShot"]["accepts_metric"] is False

    @pytest.mark.parametrize(
        "name",
        [n for n in EXPECTED_OPTIMIZERS if n != "LabeledFewShot"],
    )
    def test_other_optimizers_accept_metric(self, name):
        assert REGISTRY[name]["accepts_metric"] is True


# ---------------------------------------------------------------------------
# build_compile_kwargs
# ---------------------------------------------------------------------------


class TestBuildCompileKwargs:
    """Test that build_compile_kwargs returns the expected keys per optimizer."""

    def _make_examples(self, n):
        return [dspy.Example(question=f"q{i}", answer=f"a{i}") for i in range(n)]

    def test_labeled_few_shot(self):
        train = self._make_examples(3)
        val = self._make_examples(2)
        result = REGISTRY["LabeledFewShot"]["build_compile_kwargs"](train, val, 42)
        assert "trainset" in result
        assert "sample" in result
        # trainset should be train + val concatenated
        assert len(result["trainset"]) == 5

    def test_bootstrap_few_shot(self):
        train = self._make_examples(3)
        val = self._make_examples(2)
        result = REGISTRY["BootstrapFewShot"]["build_compile_kwargs"](train, val, 42)
        assert "trainset" in result
        assert len(result["trainset"]) == 5

    def test_bootstrap_few_shot_with_random_search(self):
        train = self._make_examples(3)
        val = self._make_examples(2)
        result = REGISTRY["BootstrapFewShotWithRandomSearch"]["build_compile_kwargs"](
            train, val, 42
        )
        assert "trainset" in result
        assert "valset" in result
        assert result["trainset"] is train
        assert result["valset"] is val

    def test_copro(self):
        train = self._make_examples(3)
        val = self._make_examples(2)
        result = REGISTRY["COPRO"]["build_compile_kwargs"](train, val, 42)
        assert "trainset" in result
        assert "eval_kwargs" in result
        # COPRO uses val as trainset
        assert result["trainset"] is val

    def test_miprov2(self):
        train = self._make_examples(3)
        val = self._make_examples(2)
        result = REGISTRY["MIPROv2"]["build_compile_kwargs"](train, val, 42)
        assert "trainset" in result
        assert "valset" in result
        assert "seed" in result
        assert result["seed"] == 42

    def test_simba(self):
        train = self._make_examples(3)
        val = self._make_examples(2)
        result = REGISTRY["SIMBA"]["build_compile_kwargs"](train, val, 42)
        assert "trainset" in result
        assert "seed" in result
        assert len(result["trainset"]) == 5
        assert result["seed"] == 42

    def test_gepa(self):
        train = self._make_examples(3)
        val = self._make_examples(2)
        result = REGISTRY["GEPA"]["build_compile_kwargs"](train, val, 42)
        assert "trainset" in result
        assert "valset" in result
        assert result["trainset"] is train
        assert result["valset"] is val

    def test_gepa_ablation(self):
        train = self._make_examples(3)
        val = self._make_examples(2)
        result = REGISTRY["GEPA_ablation"]["build_compile_kwargs"](train, val, 42)
        assert "trainset" in result
        assert "valset" in result


# ---------------------------------------------------------------------------
# metric_factory
# ---------------------------------------------------------------------------


class TestMetricFactory:
    def test_gepa_uses_gepa_feedback(self):
        assert REGISTRY["GEPA"]["metric_factory"] == "gepa_feedback"

    def test_gepa_ablation_uses_standard(self):
        assert REGISTRY["GEPA_ablation"]["metric_factory"] == "standard"

    @pytest.mark.parametrize(
        "name",
        [n for n in EXPECTED_OPTIMIZERS if n not in ("GEPA", "GEPA_ablation")],
    )
    def test_others_use_standard(self, name):
        assert REGISTRY[name]["metric_factory"] == "standard"


# ---------------------------------------------------------------------------
# validate_registry
# ---------------------------------------------------------------------------


class TestValidateRegistry:
    def test_validate_registry_runs_without_error(self):
        validate_registry()


# ---------------------------------------------------------------------------
# cost_heuristic
# ---------------------------------------------------------------------------


class TestCostHeuristic:
    @pytest.mark.parametrize("name", EXPECTED_OPTIMIZERS)
    def test_cost_heuristic_is_callable(self, name):
        assert callable(REGISTRY[name]["cost_heuristic"])

    @pytest.mark.parametrize("name", EXPECTED_OPTIMIZERS)
    def test_cost_heuristic_returns_number(self, name):
        result = REGISTRY[name]["cost_heuristic"]("light", 50, 20)
        assert isinstance(result, (int, float))
