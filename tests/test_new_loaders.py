"""Tests for new dataset loaders (code, summarization, NLI) and LLM judge."""

from __future__ import annotations

import json
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


# ---------------------------------------------------------------------------
# Code loaders (code.py)
# ---------------------------------------------------------------------------


class TestHumanEval:
    def test_basic_load(self):
        items = [
            {
                "task_id": "HumanEval/0",
                "prompt": "def has_close_elements(numbers, threshold):\n",
                "canonical_solution": "    for i in range(len(numbers)):\n        pass\n",
            },
            {
                "task_id": "HumanEval/1",
                "prompt": "def separate_paren_groups(paren_string):\n",
                "canonical_solution": "    result = []\n    return result\n",
            },
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.code._require_datasets", return_value=mock_ds):
            from context_bench.datasets.code import humaneval
            result = humaneval()

        assert len(result) == 2
        assert result[0]["id"] == "HumanEval/0"
        assert result[0]["context"] == items[0]["prompt"]
        assert result[0]["question"] == items[0]["prompt"]
        assert result[0]["answer"] == items[0]["canonical_solution"]

    def test_n_limits_output(self):
        items = [
            {"task_id": f"HumanEval/{i}", "prompt": f"def f{i}():\n", "canonical_solution": "pass\n"}
            for i in range(10)
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.code._require_datasets", return_value=mock_ds):
            from context_bench.datasets.code import humaneval
            result = humaneval(n=3)

        assert len(result) == 3


class TestMBPP:
    def test_basic_load(self):
        items = [
            {
                "task_id": 601,
                "prompt": "Write a function to find the nth fibonacci number.",
                "code": "def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)\n",
            },
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.code._require_datasets", return_value=mock_ds):
            from context_bench.datasets.code import mbpp
            result = mbpp()

        assert len(result) == 1
        assert result[0]["id"] == 601
        assert result[0]["context"] == items[0]["prompt"]
        assert result[0]["answer"] == items[0]["code"]
        mock_ds.load_dataset.assert_called_once_with(
            "google-research-datasets/mbpp", "sanitized", split="test",
        )

    def test_n_limits_output(self):
        items = [
            {"task_id": i, "prompt": f"Write function {i}.", "code": f"def f{i}(): pass\n"}
            for i in range(10)
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.code._require_datasets", return_value=mock_ds):
            from context_bench.datasets.code import mbpp
            result = mbpp(n=5)

        assert len(result) == 5


# ---------------------------------------------------------------------------
# Summarization loaders (summarization.py)
# ---------------------------------------------------------------------------


class TestMultiNews:
    def test_basic_load(self):
        items = [
            {
                "document": "Article 1 text.|||||Article 2 text.",
                "summary": "Summary of both articles.",
            },
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.summarization._require_datasets", return_value=mock_ds):
            from context_bench.datasets.summarization import multi_news
            result = multi_news()

        assert len(result) == 1
        assert "|||||" in result[0]["context"]
        assert result[0]["question"] == "Summarize the news articles."
        assert result[0]["answer"] == "Summary of both articles."

    def test_document_separator_preserved(self):
        items = [
            {"document": "A|||||B|||||C", "summary": "ABC summary"},
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.summarization._require_datasets", return_value=mock_ds):
            from context_bench.datasets.summarization import multi_news
            result = multi_news()

        assert result[0]["context"].count("|||||") == 2


class TestDialogSum:
    def test_basic_load(self):
        items = [
            {
                "id": "train_42",
                "dialogue": "#Person1#: Hi, how are you?\n#Person2#: Fine, thanks.",
                "summary": "Person1 greets Person2.",
            },
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.summarization._require_datasets", return_value=mock_ds):
            from context_bench.datasets.summarization import dialogsum
            result = dialogsum()

        assert len(result) == 1
        assert result[0]["id"] == "train_42"
        assert result[0]["question"] == "Summarize the dialogue."
        assert result[0]["answer"] == "Person1 greets Person2."


class TestQMSum:
    def test_basic_load_with_query(self):
        items = [
            {
                "id": "qmsum_0",
                "input": "Speaker A: We need to discuss the budget.\nSpeaker B: Sure.\nQuery: What was discussed about the budget?",
                "output": "The speakers discussed the budget.",
            },
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.summarization._require_datasets", return_value=mock_ds):
            from context_bench.datasets.summarization import qmsum
            result = qmsum()

        assert len(result) == 1
        assert "Speaker A:" in result[0]["context"]
        assert result[0]["question"] == "What was discussed about the budget?"
        assert result[0]["answer"] == "The speakers discussed the budget."
        # Verify trust_remote_code is passed
        mock_ds.load_dataset.assert_called_once_with(
            "tau/scrolls", "qmsum", split="validation", trust_remote_code=True,
        )

    def test_no_query_separator(self):
        items = [
            {"id": "qmsum_1", "input": "Just a transcript with no query marker.", "output": "summary"},
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.summarization._require_datasets", return_value=mock_ds):
            from context_bench.datasets.summarization import qmsum
            result = qmsum()

        assert result[0]["context"] == "Just a transcript with no query marker."
        assert result[0]["question"] == "Summarize the meeting transcript."


# ---------------------------------------------------------------------------
# NLI loaders (nli.py)
# ---------------------------------------------------------------------------


class TestContractNLI:
    def test_basic_load(self):
        items = [
            {
                "id": "cnli_0",
                "input": "This agreement shall remain in effect for 5 years.\nHypothesis: The agreement lasts forever.",
                "output": "Contradiction",
            },
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.nli._require_datasets", return_value=mock_ds):
            from context_bench.datasets.nli import contract_nli
            result = contract_nli()

        assert len(result) == 1
        assert result[0]["answer"] == "Contradiction"
        assert "The agreement lasts forever" in result[0]["question"]
        mock_ds.load_dataset.assert_called_once_with(
            "tau/scrolls", "contract_nli", split="validation", trust_remote_code=True,
        )

    def test_answer_labels(self):
        items = [
            {"id": "0", "input": "Contract text.\nHypothesis: Something.", "output": "Entailment"},
            {"id": "1", "input": "Contract text.\nHypothesis: Other.", "output": "Not mentioned"},
        ]
        mock_ds = _mock_ds_module(items)

        with mock.patch("context_bench.datasets.nli._require_datasets", return_value=mock_ds):
            from context_bench.datasets.nli import contract_nli
            result = contract_nli()

        assert result[0]["answer"] == "Entailment"
        assert result[1]["answer"] == "Not mentioned"


class TestSciFact:
    def test_basic_load_with_corpus_join(self):
        claims = [
            {
                "id": 1,
                "claim": "Vitamin C prevents colds.",
                "cited_doc_ids": [10],
                "evidence": {"10": [{"label": "SUPPORTS", "sentences": [0]}]},
            },
        ]
        corpus = [
            {"doc_id": 10, "abstract": ["Vitamin C has been shown", "to reduce cold symptoms."]},
            {"doc_id": 20, "abstract": ["Unrelated document."]},
        ]

        mock_ds = mock.MagicMock()
        mock_ds.load_dataset.side_effect = [
            _fake_dataset(claims),
            _fake_dataset(corpus),
        ]

        with mock.patch("context_bench.datasets.nli._require_datasets", return_value=mock_ds):
            from context_bench.datasets.nli import scifact
            result = scifact()

        assert len(result) == 1
        assert "Vitamin C has been shown to reduce cold symptoms." in result[0]["context"]
        assert result[0]["question"] == "Vitamin C prevents colds."
        assert result[0]["answer"] == "SUPPORTS"

    def test_skips_claims_without_cited_docs(self):
        claims = [
            {"id": 1, "claim": "Orphan claim.", "cited_doc_ids": [999], "evidence": {}},
        ]
        corpus = [
            {"doc_id": 10, "abstract": ["Some abstract."]},
        ]

        mock_ds = mock.MagicMock()
        mock_ds.load_dataset.side_effect = [
            _fake_dataset(claims),
            _fake_dataset(corpus),
        ]

        with mock.patch("context_bench.datasets.nli._require_datasets", return_value=mock_ds):
            from context_bench.datasets.nli import scifact
            result = scifact()

        # Claim with no matching corpus doc should be skipped
        assert len(result) == 0


# ---------------------------------------------------------------------------
# QASPer (qa.py — new loader)
# ---------------------------------------------------------------------------


class TestQasper:
    def test_basic_load_full_paper(self):
        items = [{
            "id": "paper_001",
            "title": "A Great Paper",
            "abstract": "This paper studies X.",
            "full_text": {
                "section_name": ["Introduction", "Methods"],
                "paragraphs": [["Intro paragraph 1."], ["Methods paragraph 1."]],
            },
            "qas": {
                "question": ["What does this paper study?", "What methods are used?"],
                "answers": [
                    {"answer": [{"free_form_answer": "X", "extractive_spans": []}]},
                    {"answer": [{"free_form_answer": "Method Y", "extractive_spans": []}]},
                ],
            },
        }]
        mock_ds = _mock_ds_module(items)

        # Use mock.patch.object on the actual module from sys.modules to avoid
        # stale parent-package attribute refs left by TestLazyImports.
        import context_bench.datasets.qa as qa_mod
        with mock.patch.object(qa_mod, "_require_datasets", return_value=mock_ds):
            result = qa_mod.qasper()

        assert len(result) == 2
        assert "A Great Paper" in result[0]["context"]
        assert "This paper studies X." in result[0]["context"]
        assert "Intro paragraph 1." in result[0]["context"]
        assert result[0]["question"] == "What does this paper study?"
        assert result[0]["answer"] == "X"
        assert result[1]["answer"] == "Method Y"

    def test_n_limits_across_papers(self):
        items = [{
            "id": "p1",
            "title": "Paper 1",
            "abstract": "",
            "full_text": {"section_name": [], "paragraphs": []},
            "qas": {
                "question": [f"Q{i}" for i in range(5)],
                "answers": [
                    {"answer": [{"free_form_answer": f"A{i}", "extractive_spans": []}]}
                    for i in range(5)
                ],
            },
        }]
        mock_ds = _mock_ds_module(items)

        import context_bench.datasets.qa as qa_mod
        with mock.patch.object(qa_mod, "_require_datasets", return_value=mock_ds):
            result = qa_mod.qasper(n=3)

        assert len(result) == 3


# ---------------------------------------------------------------------------
# LLM Judge evaluator
# ---------------------------------------------------------------------------


class TestLLMJudge:
    def test_score_parses_rating(self):
        from context_bench.evaluators.llm_judge import LLMJudge

        judge = LLMJudge(base_url="http://localhost:9999", model="test-model")

        fake_response = {
            "choices": [{"message": {"content": "4"}}],
        }
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = json.dumps(fake_response).encode()
        mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
        mock_resp.__exit__ = mock.Mock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_resp):
            scores = judge.score(
                {"question": "What is 2+2?", "answer": "4"},
                {"response": "The answer is 4."},
            )

        assert "judge_score" in scores
        # Rating 4 -> normalized: (4-1)/4 = 0.75
        assert scores["judge_score"] == pytest.approx(0.75)

    def test_score_perfect_rating(self):
        from context_bench.evaluators.llm_judge import LLMJudge

        judge = LLMJudge(base_url="http://localhost:9999")

        fake_response = {"choices": [{"message": {"content": "5"}}]}
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = json.dumps(fake_response).encode()
        mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
        mock_resp.__exit__ = mock.Mock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_resp):
            scores = judge.score({"answer": "yes"}, {"response": "yes"})

        assert scores["judge_score"] == pytest.approx(1.0)

    def test_empty_response_returns_zero(self):
        from context_bench.evaluators.llm_judge import LLMJudge

        judge = LLMJudge(base_url="http://localhost:9999")
        scores = judge.score({"answer": "something"}, {"response": ""})
        assert scores["judge_score"] == 0.0

    def test_http_error_returns_zero(self):
        from context_bench.evaluators.llm_judge import LLMJudge

        judge = LLMJudge(base_url="http://localhost:9999")

        with mock.patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
            scores = judge.score(
                {"question": "q", "answer": "a"},
                {"response": "response"},
            )

        assert scores["judge_score"] == 0.0

    def test_parse_rating_extracts_number(self):
        from context_bench.evaluators.llm_judge import LLMJudge

        assert LLMJudge._parse_rating("3") == 3
        assert LLMJudge._parse_rating("Rating: 4") == 4
        assert LLMJudge._parse_rating("I would rate this a 5.") == 5
        assert LLMJudge._parse_rating("no number here") == 1

    def test_name_property(self):
        from context_bench.evaluators.llm_judge import LLMJudge

        judge = LLMJudge(base_url="http://localhost:9999")
        assert judge.name == "llm_judge"
