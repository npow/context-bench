"""Tests for task feature extraction module."""

import pytest

from context_bench.dspy_bench.features import (
    _avg_field_len,
    _fraction_numeric,
    _answer_vocab_size,
    extract_task_features,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

QA_TRAIN = [
    {"context": "The capital of France is Paris.", "question": "What is the capital of France?", "answer": "Paris"},
    {"context": "Water boils at 100 degrees.", "question": "At what temperature does water boil?", "answer": "100 degrees"},
    {"context": "The Earth orbits the Sun.", "question": "What does the Earth orbit?", "answer": "The Sun"},
]

QA_VAL = [
    {"context": "Python was created by Guido.", "question": "Who created Python?", "answer": "Guido"},
]

QA_TEST = [
    {"context": "Mars is red.", "question": "What color is Mars?", "answer": "red"},
    {"context": "Ice is frozen water.", "question": "What is ice?", "answer": "frozen water"},
]

MC_EXAMPLES = [
    {"question": "What is 2+2?", "choices": ["3", "4", "5"], "answer": "4"},
    {"question": "Capital of Japan?", "choices": ["Tokyo", "Kyoto", "Osaka"], "answer": "Tokyo"},
]

MATH_EXAMPLES = [
    {"question": "What is 5+3?", "answer": "8"},
    {"question": "What is 10/2?", "answer": "5"},
    {"question": "What is 7*6?", "answer": "42"},
    {"question": "What is 100-1?", "answer": "99"},
]

NO_CONTEXT_EXAMPLES = [
    {"question": "What is 1+1?", "answer": "2"},
    {"question": "What is 3+4?", "answer": "7"},
]


# ---------------------------------------------------------------------------
# _avg_field_len
# ---------------------------------------------------------------------------

class TestAvgFieldLen:
    def test_basic(self):
        examples = [{"text": "hello"}, {"text": "hi"}]
        # "hello" = 5, "hi" = 2, avg = 3.5
        assert _avg_field_len(examples, "text") == 3.5

    def test_missing_field_returns_zero(self):
        examples = [{"other": "value"}, {"other": "x"}]
        assert _avg_field_len(examples, "text") == 0.0

    def test_empty_list(self):
        assert _avg_field_len([], "text") == 0.0

    def test_mixed_presence(self):
        # Only examples that have the field contribute; missing ones count as 0 length
        examples = [{"text": "abcd"}, {"other": "x"}]
        # "abcd" = 4, missing = 0, avg = 2.0
        assert _avg_field_len(examples, "text") == 2.0


# ---------------------------------------------------------------------------
# _fraction_numeric
# ---------------------------------------------------------------------------

class TestFractionNumeric:
    def test_all_numeric(self):
        assert _fraction_numeric(MATH_EXAMPLES) == pytest.approx(1.0)

    def test_no_numeric(self):
        assert _fraction_numeric(QA_TRAIN) == pytest.approx(0.0)

    def test_empty(self):
        assert _fraction_numeric([]) == 0.0

    def test_mixed(self):
        mixed = [{"answer": "42"}, {"answer": "hello"}, {"answer": "3.14"}, {"answer": "world"}]
        assert _fraction_numeric(mixed) == pytest.approx(0.5)

    def test_negative_and_float(self):
        examples = [{"answer": "-5"}, {"answer": "3.14"}]
        assert _fraction_numeric(examples) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _answer_vocab_size
# ---------------------------------------------------------------------------

class TestAnswerVocabSize:
    def test_unique_answers(self):
        assert _answer_vocab_size(QA_TRAIN) == 3

    def test_duplicate_answers(self):
        examples = [{"answer": "yes"}, {"answer": "no"}, {"answer": "yes"}]
        assert _answer_vocab_size(examples) == 2

    def test_empty(self):
        assert _answer_vocab_size([]) == 0


# ---------------------------------------------------------------------------
# extract_task_features — full integration
# ---------------------------------------------------------------------------

class TestExtractTaskFeatures:
    def test_all_keys_present(self):
        features = extract_task_features("squad", "qa", QA_TRAIN, QA_VAL, QA_TEST)
        expected_keys = {
            "dataset", "task_type",
            "n_train", "n_val", "n_test",
            "avg_context_len", "avg_question_len", "avg_answer_len",
            "has_context", "has_choices",
            "answer_is_numeric", "answer_vocab_size",
        }
        assert set(features.keys()) == expected_keys

    def test_counts(self):
        features = extract_task_features("squad", "qa", QA_TRAIN, QA_VAL, QA_TEST)
        assert features["n_train"] == 3
        assert features["n_val"] == 1
        assert features["n_test"] == 2

    def test_dataset_and_task_type(self):
        features = extract_task_features("squad", "qa", QA_TRAIN, QA_VAL, QA_TEST)
        assert features["dataset"] == "squad"
        assert features["task_type"] == "qa"

    def test_has_context_true(self):
        features = extract_task_features("squad", "qa", QA_TRAIN, QA_VAL, QA_TEST)
        assert features["has_context"] is True

    def test_has_context_false_when_no_context_field(self):
        features = extract_task_features("math", "math", NO_CONTEXT_EXAMPLES, [], [])
        assert features["has_context"] is False

    def test_avg_context_len_zero_when_no_context(self):
        features = extract_task_features("math", "math", NO_CONTEXT_EXAMPLES, [], [])
        assert features["avg_context_len"] == 0.0

    def test_has_choices_true(self):
        features = extract_task_features("arc", "mc", MC_EXAMPLES, [], [])
        assert features["has_choices"] is True

    def test_has_choices_false(self):
        features = extract_task_features("squad", "qa", QA_TRAIN, QA_VAL, QA_TEST)
        assert features["has_choices"] is False

    def test_answer_is_numeric_for_math(self):
        features = extract_task_features("gsm8k", "math", MATH_EXAMPLES, [], [])
        assert features["answer_is_numeric"] == pytest.approx(1.0)

    def test_answer_is_numeric_for_qa(self):
        features = extract_task_features("squad", "qa", QA_TRAIN, QA_VAL, QA_TEST)
        assert features["answer_is_numeric"] == pytest.approx(0.0)

    def test_answer_vocab_size(self):
        features = extract_task_features("squad", "qa", QA_TRAIN, QA_VAL, QA_TEST)
        # All examples pooled: Paris, 100 degrees, The Sun, Guido, red, frozen water = 6
        assert features["answer_vocab_size"] == 6

    def test_avg_question_len(self):
        features = extract_task_features("math", "math", MATH_EXAMPLES, [], [])
        lengths = [len(e["question"]) for e in MATH_EXAMPLES]
        expected = sum(lengths) / len(lengths)
        assert features["avg_question_len"] == pytest.approx(expected)

    def test_avg_answer_len(self):
        features = extract_task_features("math", "math", MATH_EXAMPLES, [], [])
        lengths = [len(e["answer"]) for e in MATH_EXAMPLES]
        expected = sum(lengths) / len(lengths)
        assert features["avg_answer_len"] == pytest.approx(expected)
