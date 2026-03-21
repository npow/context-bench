"""Tests for DspyOptimizerSystem wrapper."""

import copy
import threading
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Mock DSPy objects — avoids importing dspy
# ---------------------------------------------------------------------------


class MockPrediction:
    def __init__(self, answer="test answer"):
        self.answer = answer


class MockPredict:
    def __init__(self):
        self.lm = "some_lm"  # will be cleared to None

    def __call__(self, **kwargs):
        return MockPrediction()


class MockProgram:
    def __init__(self):
        self._pred = MockPredict()

    def predictors(self):
        return [self._pred]

    def __call__(self, **kwargs):
        return self._pred(**kwargs)

    def __deepcopy__(self, memo):
        new = MockProgram()
        new._pred = copy.deepcopy(self._pred, memo)
        return new


class FailingProgram(MockProgram):
    """A program whose __call__ raises an exception."""

    def __call__(self, **kwargs):
        raise RuntimeError("inference failed")

    def __deepcopy__(self, memo):
        new = FailingProgram()
        new._pred = copy.deepcopy(self._pred, memo)
        return new


class KwargsCapturingProgram(MockProgram):
    """A program that records the kwargs it was called with."""

    def __init__(self):
        super().__init__()
        self.last_kwargs = None

    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        return super().__call__(**kwargs)

    def __deepcopy__(self, memo):
        new = KwargsCapturingProgram()
        new._pred = copy.deepcopy(self._pred, memo)
        new.last_kwargs = None
        return new


# ---------------------------------------------------------------------------
# Fake TASK_INPUT_FIELDS / format_choices used to patch splits imports
# ---------------------------------------------------------------------------

FAKE_TASK_INPUT_FIELDS = {
    "qa": ["context", "question"],
    "math": ["question"],
    "mc": ["question", "choices"],
    "summarization": ["context", "question"],
    "code": ["context", "question"],
}


def fake_format_choices(choices):
    letters = "ABCDEFGHIJ"
    return "  ".join(f"{letters[i]}) {c}" for i, c in enumerate(choices))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_system(program=None, task_type="qa", name="test-system"):
    """Create a DspyOptimizerSystem with splits imports patched."""
    from context_bench.dspy_bench.system import DspyOptimizerSystem

    if program is None:
        program = MockProgram()
    return DspyOptimizerSystem(program, task_type=task_type, name=name)


@pytest.fixture(autouse=True)
def _patch_splits():
    """Patch the splits constants used by system.process() via lazy import."""
    with patch(
        "context_bench.dspy_bench.splits.TASK_INPUT_FIELDS",
        FAKE_TASK_INPUT_FIELDS,
    ), patch(
        "context_bench.dspy_bench.splits.format_choices",
        fake_format_choices,
    ):
        yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNameProperty:
    def test_returns_configured_name(self):
        system = _make_system(name="my-optimizer")
        assert system.name == "my-optimizer"

    def test_different_names(self):
        s1 = _make_system(name="alpha")
        s2 = _make_system(name="beta")
        assert s1.name == "alpha"
        assert s2.name == "beta"


class TestProcessReturnShape:
    def test_returns_dict_with_exact_keys(self):
        system = _make_system()
        result = system.process(
            {"id": "ex1", "dataset": "hotpotqa", "context": "some ctx", "question": "q?"}
        )
        assert set(result.keys()) == {"id", "dataset", "response"}

    def test_does_not_include_input_fields(self):
        """Output must NOT include context or question — prevents token inflation."""
        system = _make_system(task_type="qa")
        result = system.process(
            {
                "id": "ex1",
                "dataset": "hotpotqa",
                "context": "long context here",
                "question": "what is X?",
            }
        )
        assert "context" not in result
        assert "question" not in result

    def test_id_and_dataset_from_example(self):
        system = _make_system()
        result = system.process(
            {"id": "42", "dataset": "triviaqa", "context": "c", "question": "q"}
        )
        assert result["id"] == "42"
        assert result["dataset"] == "triviaqa"

    def test_missing_id_and_dataset_default_to_empty(self):
        system = _make_system()
        result = system.process({"context": "c", "question": "q"})
        assert result["id"] == ""
        assert result["dataset"] == ""


class TestProcessExceptionHandling:
    def test_catches_exceptions_returns_empty_response(self):
        system = _make_system(program=FailingProgram())
        result = system.process(
            {"id": "1", "dataset": "d", "context": "c", "question": "q"}
        )
        assert result["response"] == ""

    def test_still_returns_id_and_dataset_on_failure(self):
        system = _make_system(program=FailingProgram())
        result = system.process({"id": "fail-1", "dataset": "ds"})
        assert result["id"] == "fail-1"
        assert result["dataset"] == "ds"


class TestProcessInputFields:
    def test_qa_extracts_context_and_question(self):
        prog = KwargsCapturingProgram()
        system = _make_system(program=prog, task_type="qa")
        system.process(
            {"id": "1", "dataset": "d", "context": "ctx", "question": "q?", "extra": "ignored"}
        )
        # The capturing happens on the thread-local copy, retrieve it
        local_prog = system._get_program()
        assert local_prog.last_kwargs == {"context": "ctx", "question": "q?"}

    def test_math_extracts_question_only(self):
        prog = KwargsCapturingProgram()
        system = _make_system(program=prog, task_type="math")
        system.process(
            {"id": "1", "dataset": "d", "question": "2+2", "context": "should be ignored"}
        )
        local_prog = system._get_program()
        assert local_prog.last_kwargs == {"question": "2+2"}

    def test_mc_extracts_question_and_choices(self):
        prog = KwargsCapturingProgram()
        system = _make_system(program=prog, task_type="mc")
        system.process(
            {"id": "1", "dataset": "d", "question": "pick one", "choices": "A) x  B) y"}
        )
        local_prog = system._get_program()
        assert local_prog.last_kwargs == {"question": "pick one", "choices": "A) x  B) y"}


class TestProcessMCChoicesSerialization:
    def test_list_choices_serialized_to_string(self):
        prog = KwargsCapturingProgram()
        system = _make_system(program=prog, task_type="mc")
        system.process(
            {"id": "1", "dataset": "d", "question": "q", "choices": ["cat", "dog", "fish"]}
        )
        local_prog = system._get_program()
        assert local_prog.last_kwargs["choices"] == "A) cat  B) dog  C) fish"

    def test_string_choices_left_as_is(self):
        prog = KwargsCapturingProgram()
        system = _make_system(program=prog, task_type="mc")
        system.process(
            {"id": "1", "dataset": "d", "question": "q", "choices": "already formatted"}
        )
        local_prog = system._get_program()
        assert local_prog.last_kwargs["choices"] == "already formatted"


class TestThreadLocalDeepCopy:
    def test_different_threads_get_different_copies(self):
        system = _make_system()
        programs = {}
        barrier = threading.Barrier(2)

        def get_prog(name):
            prog = system._get_program()
            programs[name] = prog
            barrier.wait(timeout=5)

        t1 = threading.Thread(target=get_prog, args=("t1",))
        t2 = threading.Thread(target=get_prog, args=("t2",))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert "t1" in programs
        assert "t2" in programs
        assert programs["t1"] is not programs["t2"]

    def test_same_thread_gets_same_copy(self):
        system = _make_system()
        p1 = system._get_program()
        p2 = system._get_program()
        assert p1 is p2


class TestPredictorLMCleared:
    def test_pred_lm_set_to_none_on_init(self):
        prog = MockProgram()
        assert prog._pred.lm == "some_lm"
        system = _make_system(program=prog)
        for pred in system._compiled.predictors():
            assert pred.lm is None

    def test_multiple_predictors_all_cleared(self):
        class MultiPredProgram(MockProgram):
            def __init__(self):
                self._preds = [MockPredict(), MockPredict(), MockPredict()]

            def predictors(self):
                return self._preds

            def __call__(self, **kwargs):
                return self._preds[0](**kwargs)

            def __deepcopy__(self, memo):
                new = MultiPredProgram()
                new._preds = [copy.deepcopy(p, memo) for p in self._preds]
                return new

        prog = MultiPredProgram()
        for p in prog.predictors():
            assert p.lm == "some_lm"

        system = _make_system(program=prog)
        for p in system._compiled.predictors():
            assert p.lm is None
