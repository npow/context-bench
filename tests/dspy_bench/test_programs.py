"""Tests for DSPy program definitions per task type."""

import pytest

dspy = pytest.importorskip("dspy")


from context_bench.dspy_bench.programs import (
    CodeGenerator,
    MathDecomposer,
    MathReasoner,
    MultiHopQA,
    MultipleChoice,
    MULTI_PREDICTOR_PROGRAMS,
    SimpleQA,
    Summarizer,
    TASK_PROGRAMS,
)


# ---------------------------------------------------------------------------
# Single-predictor programs
# ---------------------------------------------------------------------------

class TestSinglePredictorInstantiation:
    """Each single-predictor program can be instantiated and has 1 predictor."""

    @pytest.mark.parametrize("cls", [SimpleQA, MathReasoner, MultipleChoice, Summarizer, CodeGenerator])
    def test_instantiation(self, cls):
        program = cls()
        assert isinstance(program, dspy.Module)

    @pytest.mark.parametrize("cls", [SimpleQA, MathReasoner, MultipleChoice, Summarizer, CodeGenerator])
    def test_single_predictor_count(self, cls):
        program = cls()
        assert len(program.predictors()) == 1

    @pytest.mark.parametrize("cls", [SimpleQA, MathReasoner, MultipleChoice, Summarizer, CodeGenerator])
    def test_output_field_is_answer(self, cls):
        program = cls()
        predictor = program.predictors()[0]
        sig = predictor.signature
        assert "answer" in sig.output_fields


class TestTaskPrograms:
    """TASK_PROGRAMS covers all 5 task types with correct mappings."""

    EXPECTED_TASK_TYPES = {"qa", "math", "mc", "summarization", "code"}

    def test_all_task_types_covered(self):
        assert set(TASK_PROGRAMS.keys()) == self.EXPECTED_TASK_TYPES

    def test_qa_maps_to_simple_qa(self):
        assert TASK_PROGRAMS["qa"] is SimpleQA

    def test_math_maps_to_math_reasoner(self):
        assert TASK_PROGRAMS["math"] is MathReasoner

    def test_mc_maps_to_multiple_choice(self):
        assert TASK_PROGRAMS["mc"] is MultipleChoice

    def test_summarization_maps_to_summarizer(self):
        assert TASK_PROGRAMS["summarization"] is Summarizer

    def test_code_maps_to_code_generator(self):
        assert TASK_PROGRAMS["code"] is CodeGenerator


# ---------------------------------------------------------------------------
# Multi-predictor programs
# ---------------------------------------------------------------------------

class TestMathDecomposer:
    """MathDecomposer has exactly 2 predictors: decompose + solve."""

    def test_instantiation(self):
        program = MathDecomposer()
        assert isinstance(program, dspy.Module)

    def test_predictor_count(self):
        program = MathDecomposer()
        assert len(program.predictors()) == 2

    def test_has_decompose_and_solve(self):
        program = MathDecomposer()
        assert hasattr(program, "decompose")
        assert hasattr(program, "solve")


class TestMultiHopQA:
    """MultiHopQA has 3+ predictors and BM25 retrieval works."""

    def test_instantiation(self):
        program = MultiHopQA()
        assert isinstance(program, dspy.Module)

    def test_predictor_count_default(self):
        """Default max_hops=2 gives 2 query predictors + 1 answer = 3."""
        program = MultiHopQA()
        assert len(program.predictors()) >= 3

    def test_predictor_count_exact(self):
        program = MultiHopQA(max_hops=2)
        assert len(program.predictors()) == 3

    def test_has_generate_query_and_answer(self):
        program = MultiHopQA()
        assert hasattr(program, "generate_query")
        assert hasattr(program, "generate_answer")
        assert isinstance(program.generate_query, list)
        assert len(program.generate_query) == 2

    def test_bm25_retrieve_returns_top_k(self):
        """Unit test BM25 retrieval with dummy data."""
        rank_bm25 = pytest.importorskip("rank_bm25")
        program = MultiHopQA(top_k=2)
        passages = [
            "The cat sat on the mat",
            "Dogs are loyal animals",
            "Cats and dogs are common pets",
            "The weather is sunny today",
            "Mathematics is the queen of sciences",
        ]
        results = program._bm25_retrieve("cat pets", passages)
        assert len(results) == 2
        # Results should be strings from the original passages
        for r in results:
            assert r in passages

    def test_bm25_retrieve_empty_passages(self):
        """Empty passage list returns empty results."""
        program = MultiHopQA()
        results = program._bm25_retrieve("query", [])
        assert results == []

    def test_bm25_retrieve_respects_top_k(self):
        """top_k parameter limits the number of results."""
        rank_bm25 = pytest.importorskip("rank_bm25")
        program = MultiHopQA(top_k=1)
        passages = ["alpha beta", "gamma delta", "alpha gamma"]
        results = program._bm25_retrieve("alpha", passages)
        assert len(results) == 1


class TestMultiPredictorPrograms:
    """MULTI_PREDICTOR_PROGRAMS maps datasets to correct classes."""

    def test_hotpotqa_maps_to_multihop(self):
        assert MULTI_PREDICTOR_PROGRAMS["hotpotqa"] is MultiHopQA

    def test_musique_maps_to_multihop(self):
        assert MULTI_PREDICTOR_PROGRAMS["musique"] is MultiHopQA

    def test_math_maps_to_decomposer(self):
        assert MULTI_PREDICTOR_PROGRAMS["math"] is MathDecomposer

    def test_expected_datasets(self):
        assert set(MULTI_PREDICTOR_PROGRAMS.keys()) == {"hotpotqa", "musique", "math"}
