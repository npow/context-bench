"""Tests for multi-turn and remaining dataset loaders."""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_dataset(items: list[dict[str, Any]]):
    class FakeDataset:
        def __init__(self, data):
            self._data = data
        def __iter__(self):
            return iter(self._data)
        def __len__(self):
            return len(self._data)
    return FakeDataset(items)


def _mock_ds_module(items: list[dict[str, Any]]):
    mock_ds = mock.MagicMock()
    mock_ds.load_dataset.return_value = _fake_dataset(items)
    return mock_ds


# ===========================================================================
# MT-Bench (multi_turn.py)
# ===========================================================================


class TestMTBench:
    def test_basic_load(self):
        items = [
            {
                "prompt_id": "101",
                "category": "writing",
                "turns": [
                    "Write a poem about spring.",
                    "Now make it rhyme.",
                ],
            },
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.multi_turn._require_datasets", return_value=mock_ds):
            from context_bench.datasets.multi_turn import mt_bench
            result = mt_bench()

        assert len(result) == 1
        assert result[0]["multi_turn"] is True
        assert len(result[0]["turns"]) == 2
        assert result[0]["turns"][0]["role"] == "user"
        assert result[0]["turns"][0]["content"] == "Write a poem about spring."
        assert result[0]["turns"][1]["content"] == "Now make it rhyme."
        assert result[0]["answer"] == ""
        assert result[0]["category"] == "writing"
        assert result[0]["id"] == "101"

    def test_n_limits(self):
        items = [
            {"prompt_id": str(i), "category": "test", "turns": [f"Turn {i}"]}
            for i in range(10)
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.multi_turn._require_datasets", return_value=mock_ds):
            from context_bench.datasets.multi_turn import mt_bench
            result = mt_bench(n=3)

        assert len(result) == 3

    def test_context_is_first_turn(self):
        items = [
            {"prompt_id": "1", "category": "math", "turns": ["Solve x+1=2", "Explain your reasoning"]},
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.multi_turn._require_datasets", return_value=mock_ds):
            from context_bench.datasets.multi_turn import mt_bench
            result = mt_bench()

        assert result[0]["context"] == "Solve x+1=2"
        assert result[0]["question"] == "Explain your reasoning"


# ===========================================================================
# AlpacaEval (instruction.py)
# ===========================================================================


class TestAlpacaEval:
    def test_basic_load(self):
        items = [
            {
                "instruction": "What is the capital of France?",
                "output": "The capital of France is Paris.",
                "generator": "text-davinci-003",
            },
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.instruction._require_datasets", return_value=mock_ds):
            from context_bench.datasets.instruction import alpaca_eval
            result = alpaca_eval()

        assert len(result) == 1
        assert result[0]["context"] == "What is the capital of France?"
        assert result[0]["question"] == "What is the capital of France?"
        assert result[0]["answer"] == "The capital of France is Paris."

    def test_n_limits(self):
        items = [
            {"instruction": f"Q{i}", "output": f"A{i}", "generator": "test"}
            for i in range(10)
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.instruction._require_datasets", return_value=mock_ds):
            from context_bench.datasets.instruction import alpaca_eval
            result = alpaca_eval(n=4)

        assert len(result) == 4


# ===========================================================================
# HellaSwag (knowledge.py)
# ===========================================================================


class TestHellaSwag:
    def test_basic_load(self):
        items = [
            {
                "ind": 42,
                "ctx": "A woman is outside with a bucket and walks toward a car.",
                "endings": [
                    "She washes the car.",
                    "She eats the bucket.",
                    "She throws the bucket away.",
                    "She dances with the car.",
                ],
                "label": "0",
            },
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.knowledge._require_datasets", return_value=mock_ds):
            from context_bench.datasets.knowledge import hellaswag
            result = hellaswag()

        assert len(result) == 1
        assert result[0]["id"] == 42
        assert result[0]["answer"] == "She washes the car."
        assert result[0]["correct_letter"] == "A"
        assert len(result[0]["choices"]) == 4
        assert "A)" in result[0]["context"]

    def test_label_index_3(self):
        items = [
            {
                "ind": 1,
                "ctx": "ctx",
                "endings": ["e0", "e1", "e2", "e3"],
                "label": "3",
            },
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.knowledge._require_datasets", return_value=mock_ds):
            from context_bench.datasets.knowledge import hellaswag
            result = hellaswag()

        assert result[0]["correct_letter"] == "D"
        assert result[0]["answer"] == "e3"


# ===========================================================================
# WinoGrande (knowledge.py)
# ===========================================================================


class TestWinoGrande:
    def test_basic_load(self):
        items = [
            {
                "sentence": "Ian volunteered to eat Dennis's dessert, because _ was on a diet.",
                "option1": "Ian",
                "option2": "Dennis",
                "answer": "2",
            },
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.knowledge._require_datasets", return_value=mock_ds):
            from context_bench.datasets.knowledge import winogrande
            result = winogrande()

        assert len(result) == 1
        assert result[0]["answer"] == "Dennis"
        assert result[0]["correct_letter"] == "B"
        assert result[0]["choices"] == ["Ian", "Dennis"]
        mock_ds.load_dataset.assert_called_once_with(
            "allenai/winogrande", "winogrande_xl", split="validation",
        )

    def test_answer_1(self):
        items = [
            {
                "sentence": "S",
                "option1": "X",
                "option2": "Y",
                "answer": "1",
            },
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.knowledge._require_datasets", return_value=mock_ds):
            from context_bench.datasets.knowledge import winogrande
            result = winogrande()

        assert result[0]["correct_letter"] == "A"
        assert result[0]["answer"] == "X"


# ===========================================================================
# MMLU-Pro (knowledge.py)
# ===========================================================================


class TestMMLUPro:
    def test_basic_load(self):
        items = [
            {
                "question": "What is the speed of light?",
                "options": [
                    "100 m/s", "1000 m/s", "3e8 m/s", "1e6 m/s", "500 m/s",
                    "2e8 m/s", "4e8 m/s", "5e8 m/s", "1e8 m/s", "6e8 m/s",
                ],
                "answer": "C",
                "answer_index": 2,
            },
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.knowledge._require_datasets", return_value=mock_ds):
            from context_bench.datasets.knowledge import mmlu_pro
            result = mmlu_pro()

        assert len(result) == 1
        assert result[0]["answer"] == "3e8 m/s"
        assert result[0]["correct_letter"] == "C"
        assert len(result[0]["choices"]) == 10
        # Should have all 10 choices formatted
        assert "J)" in result[0]["context"]

    def test_n_limits(self):
        items = [
            {"question": f"Q{i}", "options": ["A", "B"], "answer": "A", "answer_index": 0}
            for i in range(10)
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.knowledge._require_datasets", return_value=mock_ds):
            from context_bench.datasets.knowledge import mmlu_pro
            result = mmlu_pro(n=5)

        assert len(result) == 5


# ===========================================================================
# MGSM (reasoning.py)
# ===========================================================================


class TestMGSM:
    def test_basic_load(self):
        items = [
            {
                "question": "Janet's ducks lay 16 eggs per day.",
                "answer_number": 56,
                "answer": "Step by step... #### 56",
            },
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.reasoning._require_datasets", return_value=mock_ds):
            from context_bench.datasets.reasoning import mgsm
            result = mgsm()

        assert len(result) == 1
        assert result[0]["answer"] == "56"
        assert result[0]["context"] == "Janet's ducks lay 16 eggs per day."

    def test_config_passed_to_load_dataset(self):
        items = [{"question": "Q", "answer_number": 1}]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.reasoning._require_datasets", return_value=mock_ds):
            from context_bench.datasets.reasoning import mgsm
            mgsm(config="de")

        mock_ds.load_dataset.assert_called_once_with("juletxara/mgsm", "de", split="test")

    def test_n_limits(self):
        items = [
            {"question": f"Q{i}", "answer_number": i}
            for i in range(10)
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.reasoning._require_datasets", return_value=mock_ds):
            from context_bench.datasets.reasoning import mgsm
            result = mgsm(n=3)

        assert len(result) == 3


# ===========================================================================
# MultipleChoiceAccuracy — expanded A-J support
# ===========================================================================


class TestMultipleChoiceExpanded:
    def test_extract_letter_e(self):
        from context_bench.evaluators.multiple_choice import MultipleChoiceAccuracy
        ev = MultipleChoiceAccuracy()
        scores = ev.score({"correct_letter": "E"}, {"response": "E"})
        assert scores["mc_accuracy"] == 1.0

    def test_extract_letter_j(self):
        from context_bench.evaluators.multiple_choice import MultipleChoiceAccuracy
        ev = MultipleChoiceAccuracy()
        scores = ev.score({"correct_letter": "J"}, {"response": "The answer is J."})
        assert scores["mc_accuracy"] == 1.0

    def test_extract_letter_h_parenthesized(self):
        from context_bench.evaluators.multiple_choice import MultipleChoiceAccuracy
        ev = MultipleChoiceAccuracy()
        scores = ev.score({"correct_letter": "H"}, {"response": "I pick (H) because..."})
        assert scores["mc_accuracy"] == 1.0

    def test_wrong_letter_with_10_choices(self):
        from context_bench.evaluators.multiple_choice import MultipleChoiceAccuracy
        ev = MultipleChoiceAccuracy()
        scores = ev.score({"correct_letter": "F"}, {"response": "G"})
        assert scores["mc_accuracy"] == 0.0
