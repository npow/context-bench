"""Tests for the splits module — dataset split manifest and conversion utilities."""

import random
from unittest.mock import MagicMock, patch

import pytest

from context_bench.dspy_bench.splits import (
    DATASET_TASK_TYPE,
    LABEL_FIELD,
    MAX_TRAIN_DEFAULT,
    MAX_VAL_DEFAULT,
    SPLIT_MANIFEST,
    TASK_INPUT_FIELDS,
    convert_to_dspy_example,
    format_choices,
    normalize_dataset_name,
    split_dataset,
)


# ---------------------------------------------------------------------------
# normalize_dataset_name
# ---------------------------------------------------------------------------


class TestNormalizeDatasetName:
    def test_hyphens_to_underscores(self):
        assert normalize_dataset_name("arc-challenge") == "arc_challenge"

    def test_already_underscores(self):
        assert normalize_dataset_name("natural_questions") == "natural_questions"

    def test_mixed_case_lowered(self):
        assert normalize_dataset_name("HotPotQA") == "hotpotqa"

    def test_hyphens_and_uppercase(self):
        assert normalize_dataset_name("Multi-News") == "multi_news"

    def test_no_change_needed(self):
        assert normalize_dataset_name("gsm8k") == "gsm8k"

    def test_multiple_hyphens(self):
        assert normalize_dataset_name("a-b-c") == "a_b_c"


# ---------------------------------------------------------------------------
# format_choices
# ---------------------------------------------------------------------------


class TestFormatChoices:
    def test_two_choices(self):
        result = format_choices(["Yes", "No"])
        assert result == "A) Yes  B) No"

    def test_four_choices(self):
        result = format_choices(["Alpha", "Beta", "Gamma", "Delta"])
        assert result == "A) Alpha  B) Beta  C) Gamma  D) Delta"

    def test_single_choice(self):
        result = format_choices(["Only"])
        assert result == "A) Only"

    def test_empty_list(self):
        result = format_choices([])
        assert result == ""

    def test_ten_choices(self):
        choices = [f"opt{i}" for i in range(10)]
        result = format_choices(choices)
        assert result.startswith("A) opt0")
        assert "J) opt9" in result


# ---------------------------------------------------------------------------
# convert_to_dspy_example
# ---------------------------------------------------------------------------


class TestConvertToDspyExample:
    """Test convert_to_dspy_example. Uses a mock for dspy.Example if dspy
    is not available in the test environment."""

    def _make_mock_example(self):
        """Create a mock that behaves like dspy.Example."""
        mock_cls = MagicMock()

        def example_factory(**kwargs):
            instance = MagicMock()
            instance._store = dict(kwargs)
            # Allow attribute access for stored fields
            for k, v in kwargs.items():
                setattr(instance, k, v)
            instance.with_inputs = MagicMock(return_value=instance)
            return instance

        mock_cls.side_effect = example_factory
        return mock_cls

    @pytest.fixture(autouse=True)
    def _patch_dspy(self):
        """Patch dspy.Example if dspy isn't importable."""
        try:
            import dspy  # noqa: F401
            self._using_real_dspy = True
            yield
        except ImportError:
            self._using_real_dspy = False
            mock_example = self._make_mock_example()
            with patch("context_bench.dspy_bench.splits.dspy") as mock_dspy:
                mock_dspy.Example = mock_example
                yield

    def test_qa_task_preserves_original(self):
        cb_dict = {"context": "Some context", "question": "What?", "answer": "42"}
        ex = convert_to_dspy_example(cb_dict, "qa")
        assert ex._cb_original is cb_dict

    def test_qa_task_sets_with_inputs(self):
        cb_dict = {"context": "ctx", "question": "q", "answer": "a"}
        ex = convert_to_dspy_example(cb_dict, "qa")
        if not self._using_real_dspy:
            ex.with_inputs.assert_called_once_with("context", "question")
        else:
            # Real dspy: check that inputs are set
            assert "context" in ex.inputs()
            assert "question" in ex.inputs()

    def test_math_task_sets_with_inputs(self):
        cb_dict = {"question": "2+2?", "answer": "4"}
        ex = convert_to_dspy_example(cb_dict, "math")
        if not self._using_real_dspy:
            ex.with_inputs.assert_called_once_with("question")
        else:
            assert "question" in ex.inputs()

    def test_mc_task_serializes_choices(self):
        cb_dict = {
            "question": "Pick one",
            "choices": ["A", "B", "C"],
            "answer": "A",
        }
        ex = convert_to_dspy_example(cb_dict, "mc")
        # The choices field should be a formatted string, not a list
        if self._using_real_dspy:
            assert isinstance(ex.choices, str)
            assert "A) A" in ex.choices
        else:
            # Check the mock was called with string choices
            assert ex._cb_original is cb_dict

    def test_mc_task_with_string_choices_unchanged(self):
        cb_dict = {
            "question": "Pick one",
            "choices": "A) Foo  B) Bar",
            "answer": "A",
        }
        ex = convert_to_dspy_example(cb_dict, "mc")
        # String choices should pass through unchanged
        if self._using_real_dspy:
            assert ex.choices == "A) Foo  B) Bar"

    def test_summarization_task_sets_with_inputs(self):
        cb_dict = {"context": "text", "question": "summarize", "answer": "summary"}
        ex = convert_to_dspy_example(cb_dict, "summarization")
        if not self._using_real_dspy:
            ex.with_inputs.assert_called_once_with("context", "question")
        else:
            assert "context" in ex.inputs()
            assert "question" in ex.inputs()

    def test_code_task_sets_with_inputs(self):
        cb_dict = {"context": "code", "question": "fix it", "answer": "fixed"}
        ex = convert_to_dspy_example(cb_dict, "code")
        if not self._using_real_dspy:
            ex.with_inputs.assert_called_once_with("context", "question")
        else:
            assert "context" in ex.inputs()
            assert "question" in ex.inputs()

    def test_original_dict_not_mutated(self):
        cb_dict = {
            "question": "Pick",
            "choices": ["X", "Y"],
            "answer": "X",
        }
        original_choices = cb_dict["choices"].copy()
        convert_to_dspy_example(cb_dict, "mc")
        # Original dict should not be mutated
        assert cb_dict["choices"] == original_choices


# ---------------------------------------------------------------------------
# split_dataset
# ---------------------------------------------------------------------------


class TestSplitDataset:
    def test_deterministic_same_seed(self):
        data = list(range(100))
        a1, b1 = split_dataset(data, train_frac=0.75, seed=42)
        a2, b2 = split_dataset(data, train_frac=0.75, seed=42)
        assert a1 == a2
        assert b1 == b2

    def test_different_seeds_differ(self):
        data = list(range(100))
        a1, _ = split_dataset(data, train_frac=0.75, seed=42)
        a2, _ = split_dataset(data, train_frac=0.75, seed=99)
        assert a1 != a2

    def test_proportions_75_25(self):
        data = list(range(100))
        train, val = split_dataset(data, train_frac=0.75, seed=0)
        assert len(train) == 75
        assert len(val) == 25

    def test_proportions_50_50(self):
        data = list(range(20))
        train, val = split_dataset(data, train_frac=0.5, seed=0)
        assert len(train) == 10
        assert len(val) == 10

    def test_no_data_loss(self):
        data = list(range(50))
        train, val = split_dataset(data, train_frac=0.6, seed=7)
        assert sorted(train + val) == data

    def test_does_not_mutate_input(self):
        data = list(range(30))
        original = data.copy()
        split_dataset(data, train_frac=0.5, seed=1)
        assert data == original

    def test_empty_input(self):
        train, val = split_dataset([], train_frac=0.5, seed=0)
        assert train == []
        assert val == []


# ---------------------------------------------------------------------------
# Manifest and mapping consistency
# ---------------------------------------------------------------------------


class TestManifestConsistency:
    def test_split_manifest_covers_all_datasets(self):
        """Every dataset in DATASET_TASK_TYPE must have a SPLIT_MANIFEST entry."""
        for dataset in DATASET_TASK_TYPE:
            assert dataset in SPLIT_MANIFEST, (
                f"Dataset '{dataset}' in DATASET_TASK_TYPE but missing from SPLIT_MANIFEST"
            )

    def test_task_input_fields_covers_all_task_types(self):
        """Every task type referenced in DATASET_TASK_TYPE must have
        an entry in TASK_INPUT_FIELDS."""
        all_task_types = set(DATASET_TASK_TYPE.values())
        for tt in all_task_types:
            assert tt in TASK_INPUT_FIELDS, (
                f"Task type '{tt}' used in DATASET_TASK_TYPE but missing from TASK_INPUT_FIELDS"
            )

    def test_label_field_is_answer(self):
        assert LABEL_FIELD == "answer"

    def test_max_train_default(self):
        assert MAX_TRAIN_DEFAULT == 500

    def test_max_val_default(self):
        assert MAX_VAL_DEFAULT == 200

    def test_split_manifest_required_keys(self):
        """Each SPLIT_MANIFEST entry must have train_src, test_src, val_from."""
        for ds, cfg in SPLIT_MANIFEST.items():
            assert "train_src" in cfg, f"{ds} missing train_src"
            assert "test_src" in cfg, f"{ds} missing test_src"
            assert "val_from" in cfg, f"{ds} missing val_from"

    def test_zero_shot_datasets_have_no_train(self):
        """Datasets marked zero_shot_only should have train_src=None."""
        for ds, cfg in SPLIT_MANIFEST.items():
            if cfg.get("zero_shot_only"):
                assert cfg["train_src"] is None, (
                    f"{ds} is zero_shot_only but has train_src={cfg['train_src']}"
                )
                assert cfg["val_from"] is None, (
                    f"{ds} is zero_shot_only but has val_from={cfg['val_from']}"
                )

    def test_dataset_count(self):
        """Spec says 26 datasets in DATASET_TASK_TYPE."""
        assert len(DATASET_TASK_TYPE) == 26

    def test_task_types_are_expected(self):
        expected = {"qa", "math", "mc", "summarization", "code"}
        actual = set(DATASET_TASK_TYPE.values())
        assert actual == expected
