"""Tests for the sweep orchestrator module."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from context_bench.dspy_bench.sweep import (
    RunResult,
    SweepConfig,
    estimate_sweep_cost,
    load_compiled,
    load_splits_for_dataset,
    make_cache_path,
    run_sweep,
    save_compiled,
)


# ---------------------------------------------------------------------------
# SweepConfig and RunResult dataclass instantiation
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_sweep_config_instantiation(self):
        cfg = SweepConfig(
            optimizers=["LabeledFewShot"],
            datasets=["hotpotqa"],
            budgets=["light"],
            seeds=[0, 1],
            model="gpt-4o-mini",
            prompt_model=None,
            cache_dir="/tmp/cache",
            max_train=100,
            max_val=50,
            cost_cap=10.0,
            compile_only=False,
            include_ablations=False,
            n_test=None,
        )
        assert cfg.optimizers == ["LabeledFewShot"]
        assert cfg.model == "gpt-4o-mini"
        assert cfg.cost_cap == 10.0

    def test_sweep_config_defaults(self):
        cfg = SweepConfig(
            optimizers=["BootstrapFewShot"],
            datasets=["gsm8k"],
            budgets=["medium", "heavy"],
            seeds=[42],
            model="gpt-4o",
            prompt_model="gpt-4o-mini",
            cache_dir="/tmp/c",
            max_train=200,
            max_val=100,
            cost_cap=None,
            compile_only=True,
            include_ablations=True,
            n_test=50,
        )
        assert cfg.compile_only is True
        assert cfg.n_test == 50
        assert cfg.prompt_model == "gpt-4o-mini"

    def test_run_result_instantiation(self):
        rr = RunResult(
            run_key="LabeledFewShot_hotpotqa_light_0_single",
            optimizer="LabeledFewShot",
            dataset="hotpotqa",
            budget="light",
            seed=0,
            program_variant="single",
            compile_meta={"compile_time": 1.5},
            eval_result=None,
            status="completed",
            error=None,
        )
        assert rr.status == "completed"
        assert rr.error is None
        assert rr.program_variant == "single"

    def test_run_result_failed(self):
        rr = RunResult(
            run_key="COPRO_gsm8k_heavy_3_single",
            optimizer="COPRO",
            dataset="gsm8k",
            budget="heavy",
            seed=3,
            program_variant="single",
            compile_meta=None,
            eval_result=None,
            status="failed",
            error="Compile timeout",
        )
        assert rr.status == "failed"
        assert rr.error == "Compile timeout"


# ---------------------------------------------------------------------------
# make_cache_path
# ---------------------------------------------------------------------------


class TestMakeCachePath:
    def test_deterministic(self):
        p1 = make_cache_path("/cache", "MIPROv2", "hotpotqa", "light", 42, "single")
        p2 = make_cache_path("/cache", "MIPROv2", "hotpotqa", "light", 42, "single")
        assert p1 == p2

    def test_unique_for_different_optimizers(self):
        p1 = make_cache_path("/cache", "MIPROv2", "hotpotqa", "light", 42, "single")
        p2 = make_cache_path("/cache", "COPRO", "hotpotqa", "light", 42, "single")
        assert p1 != p2

    def test_unique_for_different_datasets(self):
        p1 = make_cache_path("/cache", "MIPROv2", "hotpotqa", "light", 42, "single")
        p2 = make_cache_path("/cache", "MIPROv2", "gsm8k", "light", 42, "single")
        assert p1 != p2

    def test_unique_for_different_budgets(self):
        p1 = make_cache_path("/cache", "MIPROv2", "hotpotqa", "light", 42, "single")
        p2 = make_cache_path("/cache", "MIPROv2", "hotpotqa", "heavy", 42, "single")
        assert p1 != p2

    def test_unique_for_different_seeds(self):
        p1 = make_cache_path("/cache", "MIPROv2", "hotpotqa", "light", 0, "single")
        p2 = make_cache_path("/cache", "MIPROv2", "hotpotqa", "light", 1, "single")
        assert p1 != p2

    def test_unique_for_different_variants(self):
        p1 = make_cache_path("/cache", "MIPROv2", "hotpotqa", "light", 42, "single")
        p2 = make_cache_path("/cache", "MIPROv2", "hotpotqa", "light", 42, "multi")
        assert p1 != p2

    def test_includes_all_components_in_path(self):
        path = make_cache_path("/cache", "MIPROv2", "hotpotqa", "heavy", 99, "multi")
        assert "MIPROv2" in path
        assert "hotpotqa" in path
        assert "heavy" in path
        assert "99" in path
        assert "multi" in path

    def test_rooted_in_cache_dir(self):
        path = make_cache_path("/my/cache", "opt", "ds", "light", 0, "single")
        assert path.startswith("/my/cache/")


# ---------------------------------------------------------------------------
# save_compiled / load_compiled — atomic write
# ---------------------------------------------------------------------------


class TestSaveLoadCompiled:
    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "program.json")
            program = MagicMock()
            program.save = MagicMock()
            # save_compiled writes state dict as JSON
            state = {"demos": [{"q": "hello", "a": "world"}]}
            save_compiled(state, path)
            assert os.path.exists(path)

    def test_save_atomic_no_partial_on_disk(self):
        """After save_compiled, no .tmp files remain."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "program.json")
            state = {"key": "value"}
            save_compiled(state, path)
            files = os.listdir(td)
            tmp_files = [f for f in files if f.endswith(".tmp")]
            assert len(tmp_files) == 0

    def test_save_content_is_valid_json(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "program.json")
            state = {"demos": [1, 2, 3], "instructions": "test"}
            save_compiled(state, path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded == state

    def test_load_compiled_returns_state(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "program.json")
            state = {"demos": [{"q": "x"}], "instructions": "solve it"}
            save_compiled(state, path)
            loaded = load_compiled(path)
            assert loaded == state

    def test_load_compiled_nonexistent_raises(self):
        with pytest.raises((FileNotFoundError, OSError)):
            load_compiled("/nonexistent/path/program.json")


# ---------------------------------------------------------------------------
# estimate_sweep_cost
# ---------------------------------------------------------------------------


class TestEstimateSweepCost:
    @patch("context_bench.dspy_bench.sweep.REGISTRY", new={
        "LabeledFewShot": {
            "cost_heuristic": lambda budget, n_train, n_val: 0,
        },
        "BootstrapFewShot": {
            "cost_heuristic": lambda budget, n_train, n_val: n_train + n_val,
        },
    })
    @patch("context_bench.dspy_bench.sweep.SPLIT_MANIFEST", new={
        "hotpotqa": {"zero_shot_only": False},
        "gsm8k": {"zero_shot_only": False},
    })
    @patch("context_bench.dspy_bench.sweep.DATASET_TASK_TYPE", new={
        "hotpotqa": "qa",
        "gsm8k": "math",
    })
    def test_returns_dict_with_total_calls(self):
        cfg = SweepConfig(
            optimizers=["LabeledFewShot", "BootstrapFewShot"],
            datasets=["hotpotqa", "gsm8k"],
            budgets=["light"],
            seeds=[0],
            model="gpt-4o-mini",
            prompt_model=None,
            cache_dir="/tmp/cache",
            max_train=100,
            max_val=50,
            cost_cap=None,
            compile_only=False,
            include_ablations=False,
            n_test=None,
        )
        result = estimate_sweep_cost(cfg)
        assert "total_calls" in result
        assert "per_optimizer" in result
        assert isinstance(result["total_calls"], (int, float))

    @patch("context_bench.dspy_bench.sweep.REGISTRY", new={
        "LabeledFewShot": {
            "cost_heuristic": lambda budget, n_train, n_val: 0,
        },
    })
    @patch("context_bench.dspy_bench.sweep.SPLIT_MANIFEST", new={
        "hotpotqa": {"zero_shot_only": False},
    })
    @patch("context_bench.dspy_bench.sweep.DATASET_TASK_TYPE", new={
        "hotpotqa": "qa",
    })
    def test_per_optimizer_breakdown(self):
        cfg = SweepConfig(
            optimizers=["LabeledFewShot"],
            datasets=["hotpotqa"],
            budgets=["light", "medium"],
            seeds=[0, 1, 2],
            model="gpt-4o-mini",
            prompt_model=None,
            cache_dir="/tmp/cache",
            max_train=100,
            max_val=50,
            cost_cap=None,
            compile_only=False,
            include_ablations=False,
            n_test=None,
        )
        result = estimate_sweep_cost(cfg)
        assert "LabeledFewShot" in result["per_optimizer"]

    @patch("context_bench.dspy_bench.sweep.REGISTRY", new={
        "LabeledFewShot": {
            "cost_heuristic": lambda budget, n_train, n_val: 0,
        },
        "COPRO": {
            "cost_heuristic": lambda budget, n_train, n_val: 50,
        },
    })
    @patch("context_bench.dspy_bench.sweep.SPLIT_MANIFEST", new={
        "truthfulqa": {"zero_shot_only": True},
    })
    @patch("context_bench.dspy_bench.sweep.DATASET_TASK_TYPE", new={
        "truthfulqa": "qa",
    })
    def test_zero_shot_only_excludes_non_baseline(self):
        """For zero_shot_only datasets, only 'none' and 'LabeledFewShot' should be counted."""
        cfg = SweepConfig(
            optimizers=["LabeledFewShot", "COPRO"],
            datasets=["truthfulqa"],
            budgets=["light"],
            seeds=[0],
            model="gpt-4o-mini",
            prompt_model=None,
            cache_dir="/tmp/cache",
            max_train=100,
            max_val=50,
            cost_cap=None,
            compile_only=False,
            include_ablations=False,
            n_test=None,
        )
        result = estimate_sweep_cost(cfg)
        # COPRO should have 0 calls for zero_shot_only dataset
        assert result["per_optimizer"].get("COPRO", 0) == 0


# ---------------------------------------------------------------------------
# run_sweep — zero_shot_only skipping
# ---------------------------------------------------------------------------


class TestRunSweepZeroShotSkipping:
    @patch("context_bench.dspy_bench.sweep.REGISTRY", new={
        "none": {
            "cost_heuristic": lambda budget, n_train, n_val: 0,
        },
        "LabeledFewShot": {
            "cost_heuristic": lambda budget, n_train, n_val: 0,
            "class": MagicMock(),
            "accepts_metric": False,
            "init_kwargs": {"light": {"k": 4}},
            "build_compile_kwargs": lambda train, val, seed: {"trainset": train + val},
            "metric_factory": "standard",
            "health_checks": ["demos_added"],
            "prompt_model": None,
        },
        "MIPROv2": {
            "cost_heuristic": lambda budget, n_train, n_val: 100,
            "class": MagicMock(),
            "accepts_metric": True,
            "init_kwargs": {"light": {"auto": "light"}},
            "build_compile_kwargs": lambda train, val, seed: {"trainset": train, "valset": val},
            "metric_factory": "standard",
            "health_checks": ["instruction_changed", "demos_added"],
            "prompt_model": "use_prompt_model",
        },
    })
    @patch("context_bench.dspy_bench.sweep.SPLIT_MANIFEST", new={
        "truthfulqa": {"zero_shot_only": True, "test_src": "validation", "train_src": None, "val_from": None},
    })
    @patch("context_bench.dspy_bench.sweep.DATASET_TASK_TYPE", new={
        "truthfulqa": "qa",
    })
    @patch("context_bench.dspy_bench.sweep.MULTI_PREDICTOR_PROGRAMS", new={})
    @patch("context_bench.dspy_bench.sweep.load_splits_for_dataset")
    @patch("context_bench.dspy_bench.sweep._compile_and_eval_single")
    def test_skips_mipro_on_zero_shot_dataset(self, mock_compile_eval, mock_load_splits):
        mock_load_splits.return_value = ([], [], [{"id": 1, "question": "q", "answer": "a"}])
        mock_compile_eval.return_value = RunResult(
            run_key="test",
            optimizer="LabeledFewShot",
            dataset="truthfulqa",
            budget="light",
            seed=0,
            program_variant="single",
            compile_meta={},
            eval_result=None,
            status="completed",
            error=None,
        )
        cfg = SweepConfig(
            optimizers=["LabeledFewShot", "MIPROv2"],
            datasets=["truthfulqa"],
            budgets=["light"],
            seeds=[0],
            model="gpt-4o-mini",
            prompt_model=None,
            cache_dir="/tmp/cache",
            max_train=100,
            max_val=50,
            cost_cap=None,
            compile_only=False,
            include_ablations=False,
            n_test=None,
        )
        results = run_sweep(cfg)
        # MIPROv2 should be skipped, only LabeledFewShot runs
        optimizers_run = [r.optimizer for r in results]
        assert "MIPROv2" not in optimizers_run
        # LabeledFewShot should have been called
        assert "LabeledFewShot" in optimizers_run


# ---------------------------------------------------------------------------
# run_sweep — multi-predictor variants
# ---------------------------------------------------------------------------


class TestRunSweepMultiVariants:
    @patch("context_bench.dspy_bench.sweep.REGISTRY", new={
        "LabeledFewShot": {
            "cost_heuristic": lambda budget, n_train, n_val: 0,
            "class": MagicMock(),
            "accepts_metric": False,
            "init_kwargs": {"light": {"k": 4}},
            "build_compile_kwargs": lambda train, val, seed: {"trainset": train + val},
            "metric_factory": "standard",
            "health_checks": ["demos_added"],
            "prompt_model": None,
        },
    })
    @patch("context_bench.dspy_bench.sweep.SPLIT_MANIFEST", new={
        "hotpotqa": {"zero_shot_only": False, "train_src": "train", "test_src": "validation", "val_from": "train_split"},
    })
    @patch("context_bench.dspy_bench.sweep.DATASET_TASK_TYPE", new={
        "hotpotqa": "qa",
    })
    @patch("context_bench.dspy_bench.sweep.MULTI_PREDICTOR_PROGRAMS", new={
        "hotpotqa": MagicMock,
    })
    @patch("context_bench.dspy_bench.sweep.load_splits_for_dataset")
    @patch("context_bench.dspy_bench.sweep._compile_and_eval_single")
    def test_runs_both_single_and_multi(self, mock_compile_eval, mock_load_splits):
        mock_load_splits.return_value = (
            [{"id": i} for i in range(10)],
            [{"id": i} for i in range(10, 15)],
            [{"id": i} for i in range(15, 20)],
        )

        def fake_compile_eval(*, optimizer, dataset, budget, seed, program_variant, **kw):
            return RunResult(
                run_key=f"{optimizer}_{dataset}_{budget}_{seed}_{program_variant}",
                optimizer=optimizer,
                dataset=dataset,
                budget=budget,
                seed=seed,
                program_variant=program_variant,
                compile_meta={},
                eval_result=None,
                status="completed",
                error=None,
            )

        mock_compile_eval.side_effect = fake_compile_eval
        cfg = SweepConfig(
            optimizers=["LabeledFewShot"],
            datasets=["hotpotqa"],
            budgets=["light"],
            seeds=[0],
            model="gpt-4o-mini",
            prompt_model=None,
            cache_dir="/tmp/cache",
            max_train=100,
            max_val=50,
            cost_cap=None,
            compile_only=False,
            include_ablations=False,
            n_test=None,
        )
        results = run_sweep(cfg)
        variants = {r.program_variant for r in results}
        assert "single" in variants
        assert "multi" in variants


# ---------------------------------------------------------------------------
# load_splits_for_dataset
# ---------------------------------------------------------------------------


class TestLoadSplitsForDataset:
    @patch("context_bench.dspy_bench.sweep.SPLIT_MANIFEST", new={
        "truthfulqa": {"zero_shot_only": True, "test_src": "validation", "train_src": None, "val_from": None},
    })
    @patch("context_bench.dspy_bench.sweep._load_dataset_raw")
    def test_zero_shot_returns_empty_train_val(self, mock_loader):
        mock_loader.return_value = [{"id": 1}, {"id": 2}]
        train, val, test = load_splits_for_dataset("truthfulqa", max_train=100, max_val=50)
        assert train == []
        assert val == []
        assert len(test) == 2

    @patch("context_bench.dspy_bench.sweep.SPLIT_MANIFEST", new={
        "hotpotqa": {"zero_shot_only": False, "train_src": "train", "test_src": "validation", "val_from": "train_split"},
    })
    @patch("context_bench.dspy_bench.sweep._load_dataset_raw")
    def test_train_split_divides_train_data(self, mock_loader):
        """When val_from == 'train_split', train data is split 75/25."""
        train_data = [{"id": i} for i in range(100)]
        test_data = [{"id": i} for i in range(100, 120)]

        def loader(dataset_name, split):
            if split == "train":
                return list(train_data)
            elif split == "validation":
                return list(test_data)
            return []

        mock_loader.side_effect = loader
        train, val, test = load_splits_for_dataset("hotpotqa", max_train=500, max_val=200)
        assert len(train) + len(val) == 100
        assert len(test) == 20
        # 75/25 split
        assert len(train) == 75
        assert len(val) == 25

    @patch("context_bench.dspy_bench.sweep.SPLIT_MANIFEST", new={
        "hotpotqa": {"zero_shot_only": False, "train_src": "train", "test_src": "validation", "val_from": "train_split"},
    })
    @patch("context_bench.dspy_bench.sweep._load_dataset_raw")
    def test_max_train_caps_training_set(self, mock_loader):
        train_data = [{"id": i} for i in range(1000)]
        test_data = [{"id": i} for i in range(1000, 1050)]

        def loader(dataset_name, split):
            if split == "train":
                return list(train_data)
            elif split == "validation":
                return list(test_data)
            return []

        mock_loader.side_effect = loader
        train, val, test = load_splits_for_dataset("hotpotqa", max_train=50, max_val=200)
        assert len(train) <= 50
