"""DSPy program definitions per task type.

Single-predictor programs (one per task type) and multi-predictor programs
(for datasets where decomposition adds meaningful computation).

All signatures use ``answer`` as the output field to match context-bench's
uniform schema.
"""

from __future__ import annotations

import dspy


# ---------------------------------------------------------------------------
# Single-predictor programs
# ---------------------------------------------------------------------------


class SimpleQA(dspy.Module):
    """Simple question-answering with context."""

    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("context, question -> answer")

    def forward(self, context: str, question: str) -> dspy.Prediction:
        return self.predict(context=context, question=question)


class MathReasoner(dspy.Module):
    """Math reasoning with chain-of-thought."""

    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought("question -> answer")

    def forward(self, question: str) -> dspy.Prediction:
        return self.predict(question=question)


class MultipleChoice(dspy.Module):
    """Multiple-choice question answering."""

    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("question, choices -> answer")

    def forward(self, question: str, choices: str) -> dspy.Prediction:
        return self.predict(question=question, choices=choices)


class Summarizer(dspy.Module):
    """Summarization given context and a question/instruction."""

    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("context, question -> answer")

    def forward(self, context: str, question: str) -> dspy.Prediction:
        return self.predict(context=context, question=question)


class CodeGenerator(dspy.Module):
    """Code generation with chain-of-thought reasoning."""

    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought("context, question -> answer")

    def forward(self, context: str, question: str) -> dspy.Prediction:
        return self.predict(context=context, question=question)


# Task-type to program mapping
TASK_PROGRAMS: dict[str, type[dspy.Module]] = {
    "qa": SimpleQA,
    "math": MathReasoner,
    "mc": MultipleChoice,
    "summarization": Summarizer,
    "code": CodeGenerator,
}


# ---------------------------------------------------------------------------
# Multi-predictor programs
# ---------------------------------------------------------------------------


class MathDecomposer(dspy.Module):
    """Math with decomposition. 2 predictors: decompose + solve.

    Tests whether optimizers can jointly optimize complementary instructions.
    """

    def __init__(self):
        super().__init__()
        self.decompose = dspy.ChainOfThought("question -> subproblems")
        self.solve = dspy.ChainOfThought("question, subproblems -> answer")

    def forward(self, question: str) -> dspy.Prediction:
        decomposition = self.decompose(question=question)
        return self.solve(
            question=question, subproblems=decomposition.subproblems
        )


class MultiHopQA(dspy.Module):
    """Multi-hop QA with BM25-based context retrieval over provided passages.

    Based on DSPy's SimplifiedBaleen but with in-memory BM25 retrieval
    over the dataset's gold context field. The query predictor has a real
    optimization target: better queries retrieve more relevant sentences.

    3 predictors: 2 GenerateSearchQuery (ChainOfThought) + 1 GenerateAnswer
    (ChainOfThought). This exercises MIPROv2's multivariate TPE (6 variables
    instead of 2), GEPA's cross-predictor evolution, and SIMBA's predictor
    selection.
    """

    def __init__(self, max_hops: int = 2, top_k: int = 3):
        super().__init__()
        self.generate_query = [
            dspy.ChainOfThought("context, question -> search_query")
            for _ in range(max_hops)
        ]
        self.generate_answer = dspy.ChainOfThought(
            "context, question -> answer"
        )
        self.max_hops = max_hops
        self.top_k = top_k

    def _bm25_retrieve(self, query: str, passages: list[str]) -> list[str]:
        """Retrieve top-k passages from a list using BM25."""
        if not passages:
            return []
        from rank_bm25 import BM25Okapi

        tokenized_corpus = [p.lower().split() for p in passages]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(query.lower().split())
        top_indices = sorted(
            range(len(scores)), key=lambda i: -scores[i]
        )[: self.top_k]
        return [passages[i] for i in top_indices]

    def forward(self, context: str, question: str) -> dspy.Prediction:
        # Split provided context into sentences for BM25 retrieval
        passages = [
            s.strip()
            for s in context.replace(". ", ".\n").split("\n")
            if len(s.strip()) > 10
        ]
        if not passages:
            passages = [context]

        collected_context = ""
        for hop in range(self.max_hops):
            query = self.generate_query[hop](
                context=collected_context or context, question=question
            ).search_query
            retrieved = self._bm25_retrieve(query, passages)
            collected_context = " ".join(retrieved)

        return self.generate_answer(
            context=collected_context, question=question
        )


# Dataset name to multi-predictor program mapping
MULTI_PREDICTOR_PROGRAMS: dict[str, type[dspy.Module]] = {
    "hotpotqa": MultiHopQA,
    "musique": MultiHopQA,
    "math": MathDecomposer,
}
