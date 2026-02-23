"""Tests for gap-closing dataset loaders and evaluators."""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_dataset(items: list[dict[str, Any]]):
    """Create a fake dataset object that supports iteration and len."""

    class FakeDataset:
        def __init__(self, data):
            self._data = data

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

    return FakeDataset(items)


def _mock_ds_module(items: list[dict[str, Any]]):
    """Return a mock datasets module whose load_dataset returns items."""
    mock_ds = mock.MagicMock()
    mock_ds.load_dataset.return_value = _fake_dataset(items)
    return mock_ds


# ===========================================================================
# Knowledge loaders (knowledge.py)
# ===========================================================================


class TestMMLU:
    def test_basic_load(self):
        items = [
            {
                "question": "What is the capital of France?",
                "choices": ["Berlin", "Paris", "London", "Madrid"],
                "answer": 1,
            },
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.knowledge._require_datasets", return_value=mock_ds):
            from context_bench.datasets.knowledge import mmlu
            result = mmlu()

        assert len(result) == 1
        assert result[0]["answer"] == "Paris"
        assert result[0]["correct_letter"] == "B"
        assert result[0]["choices"] == items[0]["choices"]
        assert "A)" in result[0]["context"]
        assert "B)" in result[0]["context"]

    def test_n_limits_output(self):
        items = [
            {"question": f"Q{i}", "choices": ["A", "B", "C", "D"], "answer": 0}
            for i in range(10)
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.knowledge._require_datasets", return_value=mock_ds):
            from context_bench.datasets.knowledge import mmlu
            result = mmlu(n=3)

        assert len(result) == 3

    def test_config_passed_to_load_dataset(self):
        items = [
            {"question": "Q", "choices": ["A", "B"], "answer": 0},
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.knowledge._require_datasets", return_value=mock_ds):
            from context_bench.datasets.knowledge import mmlu
            mmlu(config="anatomy")

        mock_ds.load_dataset.assert_called_once_with("cais/mmlu", "anatomy", split="test")


class TestARCChallenge:
    def test_basic_load(self):
        items = [
            {
                "id": "arc_0",
                "question": "What causes rain?",
                "choices": {"label": ["A", "B", "C", "D"], "text": ["Sun", "Evaporation", "Wind", "Moon"]},
                "answerKey": "B",
            },
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.knowledge._require_datasets", return_value=mock_ds):
            from context_bench.datasets.knowledge import arc_challenge
            result = arc_challenge()

        assert len(result) == 1
        assert result[0]["answer"] == "Evaporation"
        assert result[0]["correct_letter"] == "B"
        assert result[0]["id"] == "arc_0"

    def test_skips_invalid_answer_key(self):
        items = [
            {
                "id": "arc_bad",
                "question": "Q?",
                "choices": {"label": ["A", "B"], "text": ["X", "Y"]},
                "answerKey": "Z",
            },
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.knowledge._require_datasets", return_value=mock_ds):
            from context_bench.datasets.knowledge import arc_challenge
            result = arc_challenge()

        assert len(result) == 0


class TestTruthfulQA:
    def test_basic_load(self):
        items = [
            {"question": "Can you teach an old dog new tricks?", "best_answer": "Yes"},
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.knowledge._require_datasets", return_value=mock_ds):
            from context_bench.datasets.knowledge import truthfulqa
            result = truthfulqa()

        assert len(result) == 1
        assert result[0]["answer"] == "Yes"
        assert result[0]["context"] == result[0]["question"]


class TestGPQA:
    def test_basic_load(self):
        items = [
            {
                "Question": "What is the Hamiltonian?",
                "Correct Answer": "H = T + V",
                "Incorrect Answer 1": "H = T - V",
                "Incorrect Answer 2": "H = T * V",
                "Incorrect Answer 3": "H = T / V",
            },
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.knowledge._require_datasets", return_value=mock_ds):
            from context_bench.datasets.knowledge import gpqa
            result = gpqa()

        assert len(result) == 1
        assert result[0]["answer"] == "H = T + V"
        assert result[0]["correct_letter"] == "A"
        assert len(result[0]["choices"]) == 4


# ===========================================================================
# Reasoning loaders (reasoning.py)
# ===========================================================================


class TestDROP:
    def test_basic_load(self):
        items = [
            {
                "query_id": "drop_0",
                "passage": "The game was played on Sunday.",
                "question": "When was the game played?",
                "answers_spans": {"spans": ["Sunday"], "types": ["span"]},
            },
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.reasoning._require_datasets", return_value=mock_ds):
            from context_bench.datasets.reasoning import drop
            result = drop()

        assert len(result) == 1
        assert result[0]["answer"] == "Sunday"
        assert result[0]["context"] == "The game was played on Sunday."

    def test_empty_spans(self):
        items = [
            {
                "query_id": "drop_1",
                "passage": "P",
                "question": "Q",
                "answers_spans": {"spans": [], "types": []},
            },
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.reasoning._require_datasets", return_value=mock_ds):
            from context_bench.datasets.reasoning import drop
            result = drop()

        assert result[0]["answer"] == ""


class TestMathDataset:
    def test_basic_load_boxed(self):
        items = [
            {
                "problem": "Solve 2+2.",
                "solution": "We compute 2+2=4. The answer is $\\boxed{4}$.",
            },
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.reasoning._require_datasets", return_value=mock_ds):
            from context_bench.datasets.reasoning import math_dataset
            result = math_dataset()

        assert len(result) == 1
        assert result[0]["answer"] == "4"

    def test_nested_boxed(self):
        items = [
            {
                "problem": "P",
                "solution": "Answer: $\\boxed{\\frac{1}{2}}$.",
            },
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.reasoning._require_datasets", return_value=mock_ds):
            from context_bench.datasets.reasoning import math_dataset
            result = math_dataset()

        assert result[0]["answer"] == "\\frac{1}{2}"

    def test_no_boxed_fallback(self):
        items = [
            {"problem": "P", "solution": "The answer is 42."},
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.reasoning._require_datasets", return_value=mock_ds):
            from context_bench.datasets.reasoning import math_dataset
            result = math_dataset()

        assert result[0]["answer"] == "The answer is 42."


class TestExtractBoxed:
    def test_simple(self):
        from context_bench.datasets.reasoning import _extract_boxed
        assert _extract_boxed(r"\boxed{42}") == "42"

    def test_nested_braces(self):
        from context_bench.datasets.reasoning import _extract_boxed
        assert _extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"

    def test_no_match(self):
        from context_bench.datasets.reasoning import _extract_boxed
        assert _extract_boxed("no boxed here") == ""

    def test_deeply_nested(self):
        from context_bench.datasets.reasoning import _extract_boxed
        assert _extract_boxed(r"\boxed{a{b{c}}}") == "a{b{c}}"


# ===========================================================================
# Summarization: SummScreenFD
# ===========================================================================


class TestSummScreenFD:
    def test_basic_load(self):
        items = [
            {
                "id": "ss_0",
                "input": "NARRATOR: Previously on...\nCHARACTER: Hello!",
                "output": "Characters greet each other.",
            },
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.summarization._require_datasets", return_value=mock_ds):
            from context_bench.datasets.summarization import summscreenfd
            result = summscreenfd()

        assert len(result) == 1
        assert result[0]["context"] == items[0]["input"]
        assert result[0]["answer"] == "Characters greet each other."
        assert result[0]["question"] == "Summarize the TV episode transcript."
        mock_ds.load_dataset.assert_called_once_with(
            "tau/scrolls", "summ_screen_fd", split="validation", trust_remote_code=True,
        )


# ===========================================================================
# IFEval loader
# ===========================================================================


class TestIFEvalLoader:
    def test_basic_load(self):
        items = [
            {
                "prompt": "Write a poem about cats.",
                "instruction_id_list": ["keywords:existence", "length_constraints:number_words"],
                "kwargs": [{"keywords": ["cat"]}, {"num_words": 50, "relation": "at least"}],
            },
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.ifeval._require_datasets", return_value=mock_ds):
            from context_bench.datasets.ifeval import ifeval
            result = ifeval()

        assert len(result) == 1
        assert result[0]["context"] == "Write a poem about cats."
        assert result[0]["answer"] == ""
        assert result[0]["instruction_id_list"] == items[0]["instruction_id_list"]
        assert result[0]["kwargs"] == items[0]["kwargs"]


# ===========================================================================
# Code loader modifications (test/entry_point fields)
# ===========================================================================


class TestHumanEvalFields:
    def test_includes_test_and_entry_point(self):
        items = [
            {
                "task_id": "HumanEval/0",
                "prompt": "def foo():\n",
                "canonical_solution": "    pass\n",
                "test": "def check(candidate):\n    assert candidate() is None\n",
                "entry_point": "foo",
            },
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.code._require_datasets", return_value=mock_ds):
            from context_bench.datasets.code import humaneval
            result = humaneval()

        assert result[0]["test"] == items[0]["test"]
        assert result[0]["entry_point"] == "foo"


class TestMBPPFields:
    def test_includes_test_list(self):
        items = [
            {
                "task_id": 601,
                "prompt": "Write a function.",
                "code": "def f(): pass\n",
                "test_list": ["assert f() is None"],
                "test_setup_code": "",
            },
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.code._require_datasets", return_value=mock_ds):
            from context_bench.datasets.code import mbpp
            result = mbpp()

        assert result[0]["test_list"] == ["assert f() is None"]
        assert result[0]["test_setup_code"] == ""


# ===========================================================================
# MultipleChoiceAccuracy evaluator
# ===========================================================================


class TestMultipleChoiceAccuracy:
    def test_correct_direct_letter(self):
        from context_bench.evaluators.multiple_choice import MultipleChoiceAccuracy
        ev = MultipleChoiceAccuracy()
        scores = ev.score({"correct_letter": "B"}, {"response": "B"})
        assert scores["mc_accuracy"] == 1.0

    def test_incorrect_letter(self):
        from context_bench.evaluators.multiple_choice import MultipleChoiceAccuracy
        ev = MultipleChoiceAccuracy()
        scores = ev.score({"correct_letter": "A"}, {"response": "B"})
        assert scores["mc_accuracy"] == 0.0

    def test_extracts_from_sentence(self):
        from context_bench.evaluators.multiple_choice import MultipleChoiceAccuracy
        ev = MultipleChoiceAccuracy()
        scores = ev.score(
            {"correct_letter": "C"},
            {"response": "The answer is C because of reasons."},
        )
        assert scores["mc_accuracy"] == 1.0

    def test_extracts_parenthesized(self):
        from context_bench.evaluators.multiple_choice import MultipleChoiceAccuracy
        ev = MultipleChoiceAccuracy()
        scores = ev.score(
            {"correct_letter": "D"},
            {"response": "I would choose (D) as my answer."},
        )
        assert scores["mc_accuracy"] == 1.0

    def test_empty_response(self):
        from context_bench.evaluators.multiple_choice import MultipleChoiceAccuracy
        ev = MultipleChoiceAccuracy()
        scores = ev.score({"correct_letter": "A"}, {"response": ""})
        assert scores["mc_accuracy"] == 0.0

    def test_no_correct_letter(self):
        from context_bench.evaluators.multiple_choice import MultipleChoiceAccuracy
        ev = MultipleChoiceAccuracy()
        scores = ev.score({}, {"response": "A"})
        assert scores["mc_accuracy"] == 0.0

    def test_name_property(self):
        from context_bench.evaluators.multiple_choice import MultipleChoiceAccuracy
        assert MultipleChoiceAccuracy().name == "multiple_choice"


# ===========================================================================
# CodeExecution evaluator
# ===========================================================================


class TestCodeExecution:
    def test_passing_code(self):
        from context_bench.evaluators.code_execution import CodeExecution
        ev = CodeExecution(timeout=5.0)
        scores = ev.score(
            {"context": "", "test_list": ["assert 1 + 1 == 2"]},
            {"response": "x = 1"},
        )
        assert scores["pass_at_1"] == 1.0

    def test_failing_code(self):
        from context_bench.evaluators.code_execution import CodeExecution
        ev = CodeExecution(timeout=5.0)
        scores = ev.score(
            {"context": "", "test_list": ["assert 1 + 1 == 3"]},
            {"response": "x = 1"},
        )
        assert scores["pass_at_1"] == 0.0

    def test_timeout_returns_zero(self):
        from context_bench.evaluators.code_execution import CodeExecution
        ev = CodeExecution(timeout=1.0)
        scores = ev.score(
            {"context": "", "test_list": ["import time; time.sleep(10)"]},
            {"response": "x = 1"},
        )
        assert scores["pass_at_1"] == 0.0

    def test_empty_response(self):
        from context_bench.evaluators.code_execution import CodeExecution
        ev = CodeExecution()
        scores = ev.score({"context": ""}, {"response": ""})
        assert scores["pass_at_1"] == 0.0

    def test_humaneval_style(self):
        from context_bench.evaluators.code_execution import CodeExecution
        ev = CodeExecution(timeout=5.0)
        scores = ev.score(
            {
                "context": "def add(a, b):\n",
                "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
                "entry_point": "add",
            },
            {"response": "    return a + b\n"},
        )
        assert scores["pass_at_1"] == 1.0

    def test_name_property(self):
        from context_bench.evaluators.code_execution import CodeExecution
        assert CodeExecution().name == "code_execution"


# ===========================================================================
# IFEvalChecker evaluator
# ===========================================================================


class TestIFEvalChecker:
    def test_keyword_existence_pass(self):
        from context_bench.evaluators.ifeval_checker import IFEvalChecker
        ev = IFEvalChecker()
        scores = ev.score(
            {"instruction_id_list": ["keywords:existence"], "kwargs": [{"keywords": ["hello"]}]},
            {"response": "Hello, world!"},
        )
        assert scores["ifeval_strict"] == 1.0
        assert scores["ifeval_loose"] == 1.0

    def test_keyword_existence_fail(self):
        from context_bench.evaluators.ifeval_checker import IFEvalChecker
        ev = IFEvalChecker()
        scores = ev.score(
            {"instruction_id_list": ["keywords:existence"], "kwargs": [{"keywords": ["goodbye"]}]},
            {"response": "Hello, world!"},
        )
        assert scores["ifeval_strict"] == 0.0

    def test_forbidden_words_pass(self):
        from context_bench.evaluators.ifeval_checker import IFEvalChecker
        ev = IFEvalChecker()
        scores = ev.score(
            {"instruction_id_list": ["keywords:forbidden_words"], "kwargs": [{"forbidden_words": ["bad"]}]},
            {"response": "Good response."},
        )
        assert scores["ifeval_strict"] == 1.0

    def test_forbidden_words_fail(self):
        from context_bench.evaluators.ifeval_checker import IFEvalChecker
        ev = IFEvalChecker()
        scores = ev.score(
            {"instruction_id_list": ["keywords:forbidden_words"], "kwargs": [{"forbidden_words": ["bad"]}]},
            {"response": "This is bad."},
        )
        assert scores["ifeval_strict"] == 0.0

    def test_no_comma_pass(self):
        from context_bench.evaluators.ifeval_checker import IFEvalChecker
        ev = IFEvalChecker()
        scores = ev.score(
            {"instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]},
            {"response": "No commas here."},
        )
        assert scores["ifeval_strict"] == 1.0

    def test_no_comma_fail(self):
        from context_bench.evaluators.ifeval_checker import IFEvalChecker
        ev = IFEvalChecker()
        scores = ev.score(
            {"instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]},
            {"response": "Has, a comma."},
        )
        assert scores["ifeval_strict"] == 0.0

    def test_json_format_pass(self):
        from context_bench.evaluators.ifeval_checker import IFEvalChecker
        ev = IFEvalChecker()
        scores = ev.score(
            {"instruction_id_list": ["detectable_format:json_format"], "kwargs": [{}]},
            {"response": '{"key": "value"}'},
        )
        assert scores["ifeval_strict"] == 1.0

    def test_json_format_fail(self):
        from context_bench.evaluators.ifeval_checker import IFEvalChecker
        ev = IFEvalChecker()
        scores = ev.score(
            {"instruction_id_list": ["detectable_format:json_format"], "kwargs": [{}]},
            {"response": "not json"},
        )
        assert scores["ifeval_strict"] == 0.0

    def test_number_words(self):
        from context_bench.evaluators.ifeval_checker import IFEvalChecker
        ev = IFEvalChecker()
        scores = ev.score(
            {"instruction_id_list": ["length_constraints:number_words"], "kwargs": [{"num_words": 3, "relation": "at least"}]},
            {"response": "one two three four"},
        )
        assert scores["ifeval_strict"] == 1.0

    def test_end_checker(self):
        from context_bench.evaluators.ifeval_checker import IFEvalChecker
        ev = IFEvalChecker()
        scores = ev.score(
            {"instruction_id_list": ["startend:end_checker"], "kwargs": [{"end_phrase": "THE END"}]},
            {"response": "Some text THE END"},
        )
        assert scores["ifeval_strict"] == 1.0

    def test_two_responses(self):
        from context_bench.evaluators.ifeval_checker import IFEvalChecker
        ev = IFEvalChecker()
        scores = ev.score(
            {"instruction_id_list": ["combination:two_responses"], "kwargs": [{}]},
            {"response": "First response\n******\nSecond response"},
        )
        assert scores["ifeval_strict"] == 1.0

    def test_title_format(self):
        from context_bench.evaluators.ifeval_checker import IFEvalChecker
        ev = IFEvalChecker()
        scores = ev.score(
            {"instruction_id_list": ["detectable_format:title"], "kwargs": [{}]},
            {"response": "<<My Title>>\nSome content."},
        )
        assert scores["ifeval_strict"] == 1.0

    def test_postscript(self):
        from context_bench.evaluators.ifeval_checker import IFEvalChecker
        ev = IFEvalChecker()
        scores = ev.score(
            {"instruction_id_list": ["detectable_content:postscript"], "kwargs": [{"postscript_marker": "P.S."}]},
            {"response": "Main text.\n\nP.S. Don't forget!"},
        )
        assert scores["ifeval_strict"] == 1.0

    def test_loose_scoring(self):
        from context_bench.evaluators.ifeval_checker import IFEvalChecker
        ev = IFEvalChecker()
        scores = ev.score(
            {
                "instruction_id_list": ["punctuation:no_comma", "keywords:existence"],
                "kwargs": [{}, {"keywords": ["hello"]}],
            },
            {"response": "hello, world"},
        )
        # no_comma fails, keyword passes -> strict=0, loose=0.5
        assert scores["ifeval_strict"] == 0.0
        assert scores["ifeval_loose"] == pytest.approx(0.5)

    def test_empty_instructions(self):
        from context_bench.evaluators.ifeval_checker import IFEvalChecker
        ev = IFEvalChecker()
        scores = ev.score(
            {"instruction_id_list": [], "kwargs": []},
            {"response": "anything"},
        )
        assert scores["ifeval_strict"] == 1.0
        assert scores["ifeval_loose"] == 1.0

    def test_unknown_instruction_passes(self):
        from context_bench.evaluators.ifeval_checker import IFEvalChecker
        ev = IFEvalChecker()
        scores = ev.score(
            {"instruction_id_list": ["unknown:future_instruction"], "kwargs": [{}]},
            {"response": "anything"},
        )
        assert scores["ifeval_strict"] == 1.0

    def test_english_lowercase(self):
        from context_bench.evaluators.ifeval_checker import IFEvalChecker
        ev = IFEvalChecker()
        scores = ev.score(
            {"instruction_id_list": ["change_case:english_lowercase"], "kwargs": [{}]},
            {"response": "all lowercase here"},
        )
        assert scores["ifeval_strict"] == 1.0

    def test_repeat_prompt(self):
        from context_bench.evaluators.ifeval_checker import IFEvalChecker
        ev = IFEvalChecker()
        scores = ev.score(
            {"instruction_id_list": ["combination:repeat_prompt"], "kwargs": [{"prompt_to_repeat": "Original prompt"}]},
            {"response": "Original prompt and then my answer."},
        )
        assert scores["ifeval_strict"] == 1.0

    def test_name_property(self):
        from context_bench.evaluators.ifeval_checker import IFEvalChecker
        assert IFEvalChecker().name == "ifeval"


# ===========================================================================
# Evaluator exports
# ===========================================================================


class TestEvaluatorExports:
    def test_all_evaluators_importable(self):
        from context_bench.evaluators import (
            AnswerQuality,
            CodeExecution,
            IFEvalChecker,
            LLMJudge,
            MultipleChoiceAccuracy,
            SummarizationQuality,
        )
        assert AnswerQuality is not None
        assert CodeExecution is not None
        assert IFEvalChecker is not None
        assert LLMJudge is not None
        assert MultipleChoiceAccuracy is not None
        assert SummarizationQuality is not None
