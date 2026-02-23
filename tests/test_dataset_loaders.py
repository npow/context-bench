"""Tests for QA and long-context dataset loaders."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
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
    # Expose Features / Value / Sequence for infinitebench
    mock_ds.Features = mock.MagicMock()
    mock_ds.Value = mock.MagicMock()
    mock_ds.Sequence = mock.MagicMock()
    return mock_ds


# ---------------------------------------------------------------------------
# QA loaders (qa.py)
# ---------------------------------------------------------------------------


class TestNaturalQuestions:
    def test_basic_load(self):
        items = [
            {"question": "What is Python?", "answer": ["A programming language", "A snake"]},
            {"question": "Who wrote Hamlet?", "answer": ["Shakespeare"]},
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.qa._require_datasets", return_value=mock_ds):
            from context_bench.datasets.qa import natural_questions
            result = natural_questions()

        assert len(result) == 2
        assert result[0]["question"] == "What is Python?"
        assert result[0]["answer"] == "A programming language"
        assert result[0]["context"] == "What is Python?"
        assert "id" in result[0]

    def test_n_limits_output(self):
        items = [{"question": f"q{i}", "answer": [f"a{i}"]} for i in range(10)]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.qa._require_datasets", return_value=mock_ds):
            from context_bench.datasets.qa import natural_questions
            result = natural_questions(n=3)

        assert len(result) == 3

    def test_empty_answer(self):
        items = [{"question": "q", "answer": []}]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.qa._require_datasets", return_value=mock_ds):
            from context_bench.datasets.qa import natural_questions
            result = natural_questions()

        assert result[0]["answer"] == ""


class TestMusique:
    def test_basic_load(self):
        items = [{
            "id": "abc123",
            "question": "Multi-hop question?",
            "answer": "The answer",
            "paragraphs": [
                {"title": "Doc1", "paragraph_text": "First paragraph."},
                {"title": "Doc2", "paragraph_text": "Second paragraph."},
            ],
        }]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.qa._require_datasets", return_value=mock_ds):
            from context_bench.datasets.qa import musique
            result = musique()

        assert len(result) == 1
        assert result[0]["id"] == "abc123"
        assert "Doc1: First paragraph." in result[0]["context"]
        assert "Doc2: Second paragraph." in result[0]["context"]
        assert result[0]["answer"] == "The answer"

    def test_n_limits(self):
        items = [{"id": i, "question": "q", "answer": "a", "paragraphs": []} for i in range(5)]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.qa._require_datasets", return_value=mock_ds):
            from context_bench.datasets.qa import musique
            result = musique(n=2)

        assert len(result) == 2


class TestNarrativeQA:
    def test_basic_load(self):
        items = [{
            "document": {"summary": {"text": "A long story about adventure."}},
            "question": "What is the story about?",
            "answers": [{"text": "Adventure"}, {"text": "A journey"}],
        }]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.qa._require_datasets", return_value=mock_ds):
            from context_bench.datasets.qa import narrativeqa
            result = narrativeqa()

        assert len(result) == 1
        assert result[0]["context"] == "A long story about adventure."
        assert result[0]["answer"] == "Adventure"

    def test_empty_answers(self):
        items = [{
            "document": {"summary": {"text": "ctx"}},
            "question": "q",
            "answers": [],
        }]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.qa._require_datasets", return_value=mock_ds):
            from context_bench.datasets.qa import narrativeqa
            result = narrativeqa()

        assert result[0]["answer"] == ""


class TestTriviaQA:
    def test_basic_load(self):
        items = [{
            "question_id": "tv_12345",
            "question": "Who directed Jaws?",
            "search_results": {"search_context": ["Steven Spielberg directed Jaws in 1975.", ""]},
            "answer": {"value": "Steven Spielberg"},
        }]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.qa._require_datasets", return_value=mock_ds):
            from context_bench.datasets.qa import triviaqa
            result = triviaqa()

        assert len(result) == 1
        assert result[0]["id"] == "tv_12345"
        assert "Steven Spielberg directed Jaws" in result[0]["context"]
        assert result[0]["answer"] == "Steven Spielberg"

    def test_fallback_to_question_when_no_context(self):
        items = [{
            "question_id": "x",
            "question": "Fallback question",
            "search_results": {"search_context": []},
            "answer": {"value": "ans"},
        }]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.qa._require_datasets", return_value=mock_ds):
            from context_bench.datasets.qa import triviaqa
            result = triviaqa()

        assert result[0]["context"] == "Fallback question"


class TestFrames:
    def test_basic_load(self):
        items = [
            {"Prompt": "What year was X founded and who was president then?", "Answer": "1999, Clinton"},
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.qa._require_datasets", return_value=mock_ds):
            from context_bench.datasets.qa import frames
            result = frames()

        assert len(result) == 1
        assert result[0]["context"] == result[0]["question"]
        assert result[0]["answer"] == "1999, Clinton"


class TestQuality:
    def test_basic_load_resolves_gold_label(self):
        items = [{
            "article": "A long article about science...",
            "question": "What is the main topic?",
            "options": ["History", "Science", "Art", "Music"],
            "gold_label": 2,  # 1-indexed -> "Science"
        }]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.qa._require_datasets", return_value=mock_ds):
            from context_bench.datasets.qa import quality
            result = quality()

        assert len(result) == 1
        assert result[0]["answer"] == "Science"
        assert result[0]["context"] == "A long article about science..."

    def test_gold_label_boundary(self):
        items = [{
            "article": "ctx",
            "question": "q",
            "options": ["A", "B", "C", "D"],
            "gold_label": 4,  # 1-indexed -> index 3 -> "D"
        }]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.qa._require_datasets", return_value=mock_ds):
            from context_bench.datasets.qa import quality
            result = quality()

        assert result[0]["answer"] == "D"

    def test_gold_label_zero(self):
        items = [{
            "article": "ctx",
            "question": "q",
            "options": ["A", "B"],
            "gold_label": 0,
        }]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.qa._require_datasets", return_value=mock_ds):
            from context_bench.datasets.qa import quality
            result = quality()

        # gold_label 0 -> idx -1 which is clamped to 0 -> "A"
        assert result[0]["answer"] == "A"


# ---------------------------------------------------------------------------
# Long-context loaders (longcontext.py)
# ---------------------------------------------------------------------------


class TestLongBench:
    def test_basic_load(self):
        items = [{
            "context": "A long passage...",
            "input": "What is discussed?",
            "answers": ["The topic of discussion"],
        }]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.longcontext._require_datasets", return_value=mock_ds):
            from context_bench.datasets.longcontext import longbench
            result = longbench()

        assert len(result) == 1
        assert result[0]["context"] == "A long passage..."
        assert result[0]["question"] == "What is discussed?"
        assert result[0]["answer"] == "The topic of discussion"

    def test_empty_answers(self):
        items = [{"context": "ctx", "input": "q", "answers": []}]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.longcontext._require_datasets", return_value=mock_ds):
            from context_bench.datasets.longcontext import longbench
            result = longbench()

        assert result[0]["answer"] == ""


class TestLongBenchV2:
    def test_resolves_choice(self):
        items = [{
            "context": "Long context...",
            "question": "What is X?",
            "answer": "B",
            "choice_A": "Option A",
            "choice_B": "Option B",
            "choice_C": "Option C",
            "choice_D": "Option D",
        }]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.longcontext._require_datasets", return_value=mock_ds):
            from context_bench.datasets.longcontext import longbench_v2
            result = longbench_v2()

        assert len(result) == 1
        assert result[0]["answer"] == "Option B"

    def test_empty_answer_letter(self):
        items = [{"context": "ctx", "question": "q", "answer": ""}]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.longcontext._require_datasets", return_value=mock_ds):
            from context_bench.datasets.longcontext import longbench_v2
            result = longbench_v2()

        assert result[0]["answer"] == ""


class TestInfiniteBench:
    def test_basic_load(self):
        items = [{
            "id": 0,
            "context": "A very long book...",
            "input": "What happens in chapter 1?",
            "answer": ["The hero begins the journey"],
            "options": [],
        }]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.longcontext._require_datasets", return_value=mock_ds):
            from context_bench.datasets.longcontext import infinitebench
            result = infinitebench()

        assert len(result) == 1
        assert result[0]["answer"] == "The hero begins the journey"
        # Verify Features was constructed
        mock_ds.Features.assert_called_once()

    def test_empty_answer(self):
        items = [{"id": 0, "context": "ctx", "input": "q", "answer": [], "options": []}]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.longcontext._require_datasets", return_value=mock_ds):
            from context_bench.datasets.longcontext import infinitebench
            result = infinitebench()

        assert result[0]["answer"] == ""


def _make_nolima_repo(tmpdir: str, needles: list[dict], haystack_text: str = "Line one.\nLine two.\nLine three.") -> None:
    """Set up a fake NoLiMa repo structure in tmpdir."""
    d = Path(tmpdir)
    (d / "needlesets").mkdir()
    (d / "haystack" / "rand_shuffle").mkdir(parents=True)
    with open(d / "needlesets" / "needle_set.json", "w") as f:
        json.dump(needles, f)
    with open(d / "haystack" / "rand_shuffle" / "rand_book_1.txt", "w") as f:
        f.write(haystack_text)


class TestNoLiMa:
    def test_basic_load(self):
        needles = [{
            "id": "0401",
            "needle": "Actually, {CHAR} lives next to {1}.",
            "questions": {"onehop": "Which character has been to {2}?"},
            "character_set": ["Yuki", "Stuart"],
            "tests": {
                "T17_C02": {"input_args": ["the Kiasma museum", "Helsinki", "Uusimaa"]},
                "T15_C02": {"input_args": ["the European Central Bank", "Frankfurt", "Germany"]},
            },
        }]
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_nolima_repo(tmpdir, needles)

            with mock.patch("huggingface_hub.snapshot_download", return_value=tmpdir):
                from context_bench.datasets.longcontext import nolima
                result = nolima()

        assert len(result) == 2
        # Check the expanded needle is in context
        assert "Actually, Yuki lives next to the Kiasma museum." in result[0]["context"]
        # Answer is the character name
        assert result[0]["answer"] == "Yuki"
        # Question template expanded: {2} -> args[1] = "Helsinki"
        assert "Helsinki" in result[0]["question"]
        # Haystack text is present
        assert "Line one." in result[0]["context"]

    def test_n_limits(self):
        needles = [{
            "id": f"n{i}",
            "needle": "{CHAR} knows {1}.",
            "questions": {"onehop": "Who knows {1}?"},
            "character_set": ["Alice"],
            "tests": {f"T{j}": {"input_args": [f"fact_{i}_{j}"]} for j in range(5)},
        } for i in range(3)]
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_nolima_repo(tmpdir, needles)

            with mock.patch("huggingface_hub.snapshot_download", return_value=tmpdir):
                from context_bench.datasets.longcontext import nolima
                result = nolima(n=4)

        assert len(result) == 4

    def test_missing_needle_set_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_nolima_repo(tmpdir, [])

            with mock.patch("huggingface_hub.snapshot_download", return_value=tmpdir):
                from context_bench.datasets.longcontext import nolima
                with pytest.raises(FileNotFoundError, match="not found"):
                    nolima(needle_set="nonexistent.json")


class TestBBH:
    def test_basic_load(self):
        items = [
            {"input": "not ( True ) and ( True )", "target": "False"},
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.longcontext._require_datasets", return_value=mock_ds):
            from context_bench.datasets.longcontext import bbh
            result = bbh()

        assert len(result) == 1
        assert result[0]["context"] == "not ( True ) and ( True )"
        assert result[0]["question"] == "not ( True ) and ( True )"
        assert result[0]["answer"] == "False"

    def test_n_limits(self):
        items = [{"input": f"i{i}", "target": f"t{i}"} for i in range(10)]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.longcontext._require_datasets", return_value=mock_ds):
            from context_bench.datasets.longcontext import bbh
            result = bbh(n=4)

        assert len(result) == 4


class TestMeetingBank:
    def test_basic_load(self):
        items = [{
            "transcript": "Speaker A: We need to discuss Q4 results...",
            "summary": "The meeting covered Q4 results.",
        }]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.longcontext._require_datasets", return_value=mock_ds):
            from context_bench.datasets.longcontext import meetingbank
            result = meetingbank()

        assert len(result) == 1
        assert result[0]["context"] == "Speaker A: We need to discuss Q4 results..."
        assert result[0]["question"] == "Summarize the meeting transcript."
        assert result[0]["answer"] == "The meeting covered Q4 results."


class TestGovReport:
    def test_basic_load(self):
        items = [{
            "report": "The Department of Energy released...",
            "summary": "DOE report on energy policy.",
        }]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.longcontext._require_datasets", return_value=mock_ds):
            from context_bench.datasets.longcontext import govreport
            result = govreport()

        assert len(result) == 1
        assert result[0]["context"] == "The Department of Energy released..."
        assert result[0]["question"] == "Summarize the government report."
        assert result[0]["answer"] == "DOE report on energy policy."


# ---------------------------------------------------------------------------
# ROUGE-L evaluator
# ---------------------------------------------------------------------------


class TestRougeL:
    def test_identical_strings(self):
        from context_bench.evaluators.rouge import rouge_l
        scores = rouge_l("the cat sat on the mat", "the cat sat on the mat")
        assert scores["rouge_l_f1"] == pytest.approx(1.0)
        assert scores["rouge_l_precision"] == pytest.approx(1.0)
        assert scores["rouge_l_recall"] == pytest.approx(1.0)

    def test_no_overlap(self):
        from context_bench.evaluators.rouge import rouge_l
        scores = rouge_l("hello world", "foo bar baz")
        assert scores["rouge_l_f1"] == pytest.approx(0.0)

    def test_partial_overlap(self):
        from context_bench.evaluators.rouge import rouge_l
        scores = rouge_l("the cat sat on the mat", "the cat on the mat")
        # LCS = "the cat on the mat" (5 tokens)
        # pred = 6 tokens, ref = 5 tokens
        assert scores["rouge_l_recall"] == pytest.approx(1.0)
        assert scores["rouge_l_precision"] == pytest.approx(5 / 6)
        assert scores["rouge_l_f1"] > 0.9

    def test_empty_prediction(self):
        from context_bench.evaluators.rouge import rouge_l
        scores = rouge_l("", "some reference text")
        assert scores["rouge_l_f1"] == pytest.approx(0.0)

    def test_empty_reference(self):
        from context_bench.evaluators.rouge import rouge_l
        scores = rouge_l("some prediction text", "")
        assert scores["rouge_l_f1"] == pytest.approx(0.0)

    def test_case_insensitive(self):
        from context_bench.evaluators.rouge import rouge_l
        scores = rouge_l("The Cat", "the cat")
        assert scores["rouge_l_f1"] == pytest.approx(1.0)


class TestSummarizationQuality:
    def test_score_returns_rouge_keys(self):
        from context_bench.evaluators.rouge import SummarizationQuality
        evaluator = SummarizationQuality()
        scores = evaluator.score(
            {"answer": "The meeting covered Q4 results."},
            {"response": "The meeting discussed Q4 results and outlook."},
        )
        assert "rouge_l_f1" in scores
        assert "rouge_l_precision" in scores
        assert "rouge_l_recall" in scores
        assert scores["rouge_l_f1"] > 0

    def test_empty_answer(self):
        from context_bench.evaluators.rouge import SummarizationQuality
        evaluator = SummarizationQuality()
        scores = evaluator.score({"answer": ""}, {"response": "something"})
        assert scores["rouge_l_f1"] == pytest.approx(1.0)

    def test_empty_response(self):
        from context_bench.evaluators.rouge import SummarizationQuality
        evaluator = SummarizationQuality()
        scores = evaluator.score({"answer": "something"}, {"response": ""})
        assert scores["rouge_l_f1"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CLI dataset:config parsing
# ---------------------------------------------------------------------------


class TestDatasetConfigParsing:
    def test_config_suffix_parsed(self):
        """dataset:config syntax is parsed and passed to loader."""
        fake_data = [{"id": 0, "context": "ctx", "answer": "ans"}]
        mock_loader = mock.MagicMock(return_value=fake_data)

        with mock.patch("importlib.import_module") as mock_import:
            mock_mod = mock.MagicMock()
            mock_mod.longbench = mock_loader
            mock_import.return_value = mock_mod

            from context_bench.__main__ import _load_dataset
            result = _load_dataset("longbench:qasper", max_examples=5)

        mock_loader.assert_called_once_with(n=5, config="qasper")
        assert result == fake_data

    def test_non_configurable_dataset_rejects_config(self):
        from context_bench.__main__ import _load_dataset
        with pytest.raises(SystemExit, match="does not accept a :config suffix"):
            _load_dataset("hotpotqa:something", max_examples=None)

    def test_plain_dataset_no_config(self):
        """Plain dataset name passes no config kwarg."""
        fake_data = [{"id": 0, "context": "ctx", "answer": "ans"}]
        mock_loader = mock.MagicMock(return_value=fake_data)

        with mock.patch("importlib.import_module") as mock_import:
            mock_mod = mock.MagicMock()
            mock_mod.hotpotqa = mock_loader
            mock_import.return_value = mock_mod

            from context_bench.__main__ import _load_dataset
            _load_dataset("hotpotqa", max_examples=10)

        mock_loader.assert_called_once_with(n=10)


# ---------------------------------------------------------------------------
# Cross-cutting: lazy import verification
# ---------------------------------------------------------------------------


class TestLazyImports:
    def test_qa_module_does_not_import_datasets_at_module_level(self):
        """qa.py should not import 'datasets' at module level."""
        # Remove cached module if present
        mod_name = "context_bench.datasets.qa"
        saved = sys.modules.pop(mod_name, None)
        try:
            # Temporarily remove datasets from sys.modules
            saved_ds = sys.modules.pop("datasets", None)
            try:
                import context_bench.datasets.qa  # noqa: F401
                # If datasets were imported at module level, this would fail
                # when datasets is not installed. Since we're mocking, just
                # verify that 'datasets' is NOT in sys.modules after import.
                assert "datasets" not in sys.modules or saved_ds is not None
            finally:
                if saved_ds is not None:
                    sys.modules["datasets"] = saved_ds
        finally:
            if saved is not None:
                sys.modules[mod_name] = saved

    def test_longcontext_module_does_not_import_datasets_at_module_level(self):
        """longcontext.py should not import 'datasets' at module level."""
        mod_name = "context_bench.datasets.longcontext"
        saved = sys.modules.pop(mod_name, None)
        try:
            saved_ds = sys.modules.pop("datasets", None)
            try:
                import context_bench.datasets.longcontext  # noqa: F401
                assert "datasets" not in sys.modules or saved_ds is not None
            finally:
                if saved_ds is not None:
                    sys.modules["datasets"] = saved_ds
        finally:
            if saved is not None:
                sys.modules[mod_name] = saved
