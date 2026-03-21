"""QSR (Quorum Sensing Router) benchmark evaluation.

Compares five verification strategies on multi-hop QA datasets:
  1. NoVerification — baseline MultiHopQA, no QSR
  2. AlwaysVerify — always re-sample 3x and majority vote
  3. PerAgentThreshold — fire when ANY single step has high hedge uncertainty
  4. QSR — full quorum sensing router
  5. OracleVerification — verify only when baseline would have been wrong

Usage:
    python -m context_bench.dspy_bench.qsr_eval \
        --dataset musique --dataset frames --n 200 \
        --model claude-haiku-4-5-20251001
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

# Ensure dspy source tree is importable
_DSPY_SRC = os.path.expanduser("~/code/dspy")
if _DSPY_SRC not in sys.path:
    sys.path.insert(0, _DSPY_SRC)

import dspy
from dspy.predict.quorum import (
    AutoinducerEmitter,
    EmissionEvent,
    HedgeExtractor,
    QuorumMedium,
    QuorumRouter,
    SamplingVerification,
    _first_string_field,
)
from rank_bm25 import BM25Okapi

from context_bench.datasets.qa import frames as load_frames
from context_bench.datasets.qa import musique as load_musique
from context_bench.evaluators.answer_quality import AnswerQuality
from context_bench.metrics.quality import exact_match, f1_score, recall_score

# ---------------------------------------------------------------------------
# DSPy programs
# ---------------------------------------------------------------------------

DATASET_LOADERS = {
    "musique": load_musique,
    "frames": load_frames,
}


class BaselineMultiHopQA(dspy.Module):
    """Multi-hop QA with BM25-based context retrieval over provided passages."""

    def __init__(self, max_hops: int = 2, top_k: int = 3):
        super().__init__()
        self.generate_query = [
            dspy.ChainOfThought("context, question -> search_query")
            for _ in range(max_hops)
        ]
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")
        self.max_hops = max_hops
        self.top_k = top_k

    def _bm25_retrieve(self, query: str, passages: list[str]) -> list[str]:
        if not passages:
            return []
        tokenized_corpus = [p.lower().split() for p in passages]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[: self.top_k]
        return [passages[i] for i in top_indices]

    def forward(self, context: str, question: str) -> dspy.Prediction:
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

        return self.generate_answer(context=collected_context, question=question)


class FourStageMultiHopQA(dspy.Module):
    """4-stage multi-hop QA: decompose → retrieve per sub-Q → reason per hop → synthesize.

    This pipeline has 4 distinct LLM stages, each of which can emit uncertainty.
    The baseline version runs all 4 stages without QSR.
    """

    def __init__(self, top_k: int = 3):
        super().__init__()
        self.decompose = dspy.ChainOfThought(
            "question -> sub_question_1, sub_question_2"
        )
        self.reason_hop1 = dspy.ChainOfThought(
            "context, sub_question -> intermediate_answer"
        )
        self.reason_hop2 = dspy.ChainOfThought(
            "context, sub_question, prior_answer -> intermediate_answer"
        )
        self.synthesize = dspy.ChainOfThought(
            "question, intermediate_1, intermediate_2 -> answer"
        )
        self.top_k = top_k

    def _bm25_retrieve(self, query: str, passages: list[str]) -> list[str]:
        if not passages:
            return []
        tokenized_corpus = [p.lower().split() for p in passages]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[: self.top_k]
        return [passages[i] for i in top_indices]

    def _split_passages(self, context: str) -> list[str]:
        passages = [
            s.strip()
            for s in context.replace(". ", ".\n").split("\n")
            if len(s.strip()) > 10
        ]
        return passages if passages else [context]

    def forward(self, context: str, question: str) -> dspy.Prediction:
        passages = self._split_passages(context)

        # Stage 1: Decompose question into sub-questions
        decomp = self.decompose(question=question)
        sq1 = decomp.sub_question_1 if hasattr(decomp, "sub_question_1") else question
        sq2 = decomp.sub_question_2 if hasattr(decomp, "sub_question_2") else question

        # Stage 2: Retrieve for each sub-question
        context_1 = " ".join(self._bm25_retrieve(sq1, passages))
        context_2 = " ".join(self._bm25_retrieve(sq2, passages))

        # Stage 3: Reason per hop
        hop1 = self.reason_hop1(context=context_1, sub_question=sq1)
        hop2 = self.reason_hop2(
            context=context_2, sub_question=sq2,
            prior_answer=hop1.intermediate_answer,
        )

        # Stage 4: Synthesize
        result = self.synthesize(
            question=question,
            intermediate_1=hop1.intermediate_answer,
            intermediate_2=hop2.intermediate_answer,
        )
        return result


class _CrossStageConsistencyExtractor:
    """Emits uncertainty based on cross-stage consistency.

    Instead of running each stage twice (noisy), checks whether the outputs
    of different stages are consistent with each other:
    - Does hop2's answer reference information from hop1?
    - Does the synthesis contradict the intermediates?
    - Do the sub-questions actually decompose the original question?

    This only emits from the SYNTHESIZE stage, using information accumulated
    across all prior stages as context for the consistency check.
    """

    def __init__(self, prior_outputs: dict):
        self._prior_outputs = prior_outputs

    def extract(self, prediction: dspy.Prediction, inputs: dict) -> tuple[float, dict]:
        # Get the final answer
        answer = ""
        for key in prediction.keys():
            val = prediction[key]
            if isinstance(val, str):
                answer = val.strip().lower()
                break

        int1 = self._prior_outputs.get("intermediate_1", "").lower()
        int2 = self._prior_outputs.get("intermediate_2", "").lower()

        # Signal 1: Does the answer contradict the intermediates?
        # If the answer doesn't share any tokens with either intermediate, that's suspicious
        from context_bench.metrics.quality import f1_score as _f1

        overlap_1 = _f1(answer, int1) if answer and int1 else 0.0
        overlap_2 = _f1(answer, int2) if answer and int2 else 0.0
        avg_overlap = (overlap_1 + overlap_2) / 2.0

        # Signal 2: Do the two intermediates contradict each other?
        inter_consistency = _f1(int1, int2) if int1 and int2 else 0.5

        # Signal 3: Is either intermediate a refusal?
        refusal_phrases = ["not provide", "cannot determine", "no information", "not available", "not mentioned", "does not"]
        int1_refuses = any(p in int1 for p in refusal_phrases)
        int2_refuses = any(p in int2 for p in refusal_phrases)
        refusal_signal = 0.5 if (int1_refuses or int2_refuses) else 0.0

        # Combine: low overlap with intermediates + refusals = high uncertainty
        uncertainty = max(0.05, min(0.95,
            0.4 * (1.0 - avg_overlap) +
            0.3 * refusal_signal +
            0.3 * (1.0 - inter_consistency)
        ))

        return uncertainty, {
            "answer_int1_overlap": overlap_1,
            "answer_int2_overlap": overlap_2,
            "inter_consistency": inter_consistency,
            "int1_refuses": int1_refuses,
            "int2_refuses": int2_refuses,
            "refusal_signal": refusal_signal,
        }


class QSRFourStageMultiHopQA(dspy.Module):
    """4-stage multi-hop QA with cross-stage consistency QSR.

    Runs the full pipeline, then checks whether the synthesis is consistent
    with the intermediate answers. Emits a SINGLE signal from cross-stage
    analysis rather than noisy per-stage self-consistency.

    This avoids the noise accumulation problem: instead of 4 independent
    self-consistency checks, there's 1 cross-stage consistency check that
    captures correlated failure across the pipeline.
    """

    def __init__(self, top_k: int = 3, threshold: float = 0.40, decay: float = 0.85):
        super().__init__()
        self.medium = QuorumMedium(threshold=threshold, decay=decay)

        self.decompose = dspy.ChainOfThought(
            "question -> sub_question_1, sub_question_2"
        )
        self.reason_hop1 = dspy.ChainOfThought(
            "context, sub_question -> intermediate_answer"
        )
        self.reason_hop2 = dspy.ChainOfThought(
            "context, sub_question, prior_answer -> intermediate_answer"
        )
        self._synthesize = dspy.ChainOfThought(
            "question, intermediate_1, intermediate_2 -> answer"
        )
        self.top_k = top_k

    def _bm25_retrieve(self, query: str, passages: list[str]) -> list[str]:
        if not passages:
            return []
        tokenized_corpus = [p.lower().split() for p in passages]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[: self.top_k]
        return [passages[i] for i in top_indices]

    def _split_passages(self, context: str) -> list[str]:
        passages = [
            s.strip()
            for s in context.replace(". ", ".\n").split("\n")
            if len(s.strip()) > 10
        ]
        return passages if passages else [context]

    def forward(self, context: str, question: str) -> dspy.Prediction:
        self.medium.reset()
        passages = self._split_passages(context)

        # Stage 1: Decompose
        decomp = self.decompose(question=question)
        sq1 = decomp.sub_question_1 if hasattr(decomp, "sub_question_1") else question
        sq2 = decomp.sub_question_2 if hasattr(decomp, "sub_question_2") else question

        # Stage 2: Retrieve
        context_1 = " ".join(self._bm25_retrieve(sq1, passages))
        context_2 = " ".join(self._bm25_retrieve(sq2, passages))

        # Stage 3: Reason
        hop1 = self.reason_hop1(context=context_1, sub_question=sq1)
        hop2 = self.reason_hop2(
            context=context_2, sub_question=sq2,
            prior_answer=hop1.intermediate_answer if hasattr(hop1, "intermediate_answer") else "",
        )

        int1 = hop1.intermediate_answer if hasattr(hop1, "intermediate_answer") else str(_first_string_field(hop1) or "")
        int2 = hop2.intermediate_answer if hasattr(hop2, "intermediate_answer") else str(_first_string_field(hop2) or "")

        # Stage 4: Synthesize with cross-stage consistency emitter
        prior_outputs = {"intermediate_1": int1, "intermediate_2": int2}
        extractor = _CrossStageConsistencyExtractor(prior_outputs)
        self.synthesize_emitter = AutoinducerEmitter(
            module=self._synthesize,
            medium=self.medium,
            extractor=extractor,
            name="cross_stage_check",
        )

        result = self.synthesize_emitter(
            question=question,
            intermediate_1=int1,
            intermediate_2=int2,
        )

        concentration = self.medium.concentration
        fired = concentration >= self.medium.threshold

        qsr_meta = {
            "fired": fired,
            "concentration": concentration,
        }

        if fired:
            # Cross-stage inconsistency detected — re-synthesize with warning
            # Tell the model the intermediates may be unreliable and to
            # use its own knowledge to cross-check
            cautious_synth = dspy.ChainOfThought(
                "question, intermediate_1, intermediate_2, warning -> answer"
            )
            warning = (
                "WARNING: The intermediate answers above may be inconsistent or unreliable. "
                "Cross-check them against your own knowledge before synthesizing. "
                "If they contradict each other or seem wrong, rely on your knowledge instead."
            )
            result = cautious_synth(
                question=question,
                intermediate_1=int1,
                intermediate_2=int2,
                warning=warning,
            )
            qsr_meta["route"] = "cautious_synthesis"
        else:
            qsr_meta["route"] = "passthrough"

        result._qsr_meta = qsr_meta
        return result


class _SelfConsistencyExtractor:
    """Extracts uncertainty by comparing two answers to the same question.

    Runs the answer module twice. If the answers differ, the model is
    uncertain — regardless of what it claims. This catches reasoning
    failures that assessment-based extractors miss.
    """

    def __init__(self, answer_module: dspy.Module):
        self._answer_module = answer_module

    def extract(self, prediction: dspy.Prediction, inputs: dict) -> tuple[float, dict]:
        # prediction is answer_1 (already generated by the emitter's inner module)
        answer_1 = ""
        for key in prediction.keys():
            val = prediction[key]
            if isinstance(val, str):
                answer_1 = val.strip().lower()
                break

        # Generate answer_2 at temperature=1.0 with ALL original kwargs
        try:
            pred_2 = self._answer_module(
                **inputs,
                config={"temperature": 1.0},
            )
            answer_2 = ""
            for key in pred_2.keys():
                val = pred_2[key]
                if isinstance(val, str):
                    answer_2 = val.strip().lower()
                    break
        except Exception:
            # If second call fails, treat as uncertain
            return 0.8, {"answer_1": answer_1, "answer_2": "ERROR", "match": False}

        # Compare: exact match after normalization
        match = _normalize_for_comparison(answer_1) == _normalize_for_comparison(answer_2)

        # Token-level F1 between the two answers as a softer signal
        from context_bench.metrics.quality import f1_score as _f1
        similarity = _f1(answer_1, answer_2) if answer_1 and answer_2 else 0.0

        # Continuous uncertainty: 1 - similarity
        # Exact match → 0.05, completely different → 0.95
        if match:
            uncertainty = 0.05
        else:
            uncertainty = max(0.1, min(0.95, 1.0 - similarity))

        return uncertainty, {
            "answer_1": answer_1[:100],
            "answer_2": answer_2[:100],
            "match": match,
            "similarity": similarity,
        }


def _normalize_for_comparison(text: str) -> str:
    """Normalize text for answer comparison."""
    import re
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Remove common prefixes
    for prefix in ("the answer is", "answer:", "the answer:", "based on"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text


class QSRMultiHopQA(dspy.Module):
    """Multi-hop QA with self-consistency-based Quorum Sensing Router.

    After retrieving and generating an answer, runs a second answer at
    temperature=1.0. If the two answers diverge, emits high uncertainty.
    When quorum fires: majority-votes from 3 additional samples (5 total).
    When quorum is silent: returns the first answer unchanged.

    Cost: baseline + 1 extra LM call (the consistency check).
    When fired: baseline + 1 + 3 = 4 extra LM calls.
    """

    def __init__(
        self,
        max_hops: int = 2,
        top_k: int = 3,
        threshold: float = 0.35,
        decay: float = 0.85,
    ):
        super().__init__()
        self.generate_query = [
            dspy.ChainOfThought("context, question -> search_query")
            for _ in range(max_hops)
        ]
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")
        self.max_hops = max_hops
        self.top_k = top_k

        # QSR components
        self.medium = QuorumMedium(threshold=threshold, decay=decay)

        # The emitter wraps the answer module — runs it once, then the
        # extractor runs it a second time to check consistency
        self.answer_emitter = AutoinducerEmitter(
            module=self.generate_answer,
            medium=self.medium,
            extractor=_SelfConsistencyExtractor(self.generate_answer),
            name="answer_consistency",
        )

    def _bm25_retrieve(self, query: str, passages: list[str]) -> list[str]:
        if not passages:
            return []
        tokenized_corpus = [p.lower().split() for p in passages]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[: self.top_k]
        return [passages[i] for i in top_indices]

    def forward(self, context: str, question: str) -> dspy.Prediction:
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

        # Reset medium for this example
        self.medium.reset()

        # Generate answer via emitter — this runs answer once, then the
        # extractor runs it again to check self-consistency
        pred = self.answer_emitter(
            context=collected_context, question=question
        )

        fired = self.medium.concentration >= self.medium.threshold
        consistency_meta = getattr(pred, "_quorum_uncertainty", 0.0)

        # Build metadata
        qsr_meta = {
            "fired": fired,
            "concentration": self.medium.concentration,
            "consistency_uncertainty": consistency_meta,
        }

        if fired:
            # Answers diverged — re-retrieve with a different query and re-answer
            # This is the key difference from AlwaysVerify: don't re-sample with
            # the same context, get DIFFERENT context
            ans_1 = pred.answer if hasattr(pred, "answer") else str(_first_string_field(pred) or "")
            answers = [ans_1]

            # Strategy: generate a new query, retrieve different passages, re-answer
            try:
                # Ask for a rephrased query
                new_query_pred = self.generate_query[0](
                    context=f"Previous answer attempt: {ans_1}. This may be wrong.",
                    question=question,
                )
                new_query = new_query_pred.search_query
                # Retrieve with new query — may get different passages
                new_retrieved = self._bm25_retrieve(new_query, passages)
                new_context = " ".join(new_retrieved)

                # Answer with new context
                new_pred = self.generate_answer(
                    context=new_context,
                    question=question,
                )
                ans_2 = new_pred.answer if hasattr(new_pred, "answer") else str(_first_string_field(new_pred) or "")
                answers.append(ans_2)
            except Exception:
                pass

            # Also get one more sample with original context at temp=1
            try:
                extra = self.generate_answer(
                    context=collected_context,
                    question=question,
                    config={"temperature": 1.0},
                )
                ans_3 = extra.answer if hasattr(extra, "answer") else str(_first_string_field(extra) or "")
                answers.append(ans_3)
            except Exception:
                pass

            # Majority vote across all answers (original + re-retrieved + temp sample)
            if len(answers) > 1:
                counter = Counter(answers)
                majority, _ = counter.most_common(1)[0]
                pred = dspy.Prediction(answer=majority)

            qsr_meta["route"] = "re_retrieve_vote"
            qsr_meta["n_samples"] = len(answers)
        else:
            qsr_meta["route"] = "passthrough"

        pred._qsr_meta = qsr_meta
        return pred


# ---------------------------------------------------------------------------
# System wrappers (context-bench System protocol)
# ---------------------------------------------------------------------------


def _make_lm(model: str) -> dspy.LM:
    """Create a DSPy LM configured for the proxy."""
    return dspy.LM(
        model=f"openai/{model}",
        api_base="http://127.0.0.1:8317/v1",
        api_key="your-api-key-1",
        max_tokens=500,
    )


def _run_program(program: dspy.Module, example: dict[str, Any], model: str) -> dspy.Prediction:
    """Run a DSPy program within a configured LM context."""
    lm = _make_lm(model)
    with dspy.context(lm=lm):
        return program(context=example["context"], question=example["question"])


class NoVerificationSystem:
    """Baseline 4-stage MultiHopQA, no verification."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self._model = model
        self._program = FourStageMultiHopQA()

    @property
    def name(self) -> str:
        return "NoVerification"

    def process(self, example: dict[str, Any]) -> dict[str, Any]:
        pred = _run_program(self._program, example, self._model)
        return {
            "response": pred.answer if hasattr(pred, "answer") else str(_first_string_field(pred) or ""),
            "lm_calls": 4,  # decompose + hop1 + hop2 + synthesize
        }


class AlwaysVerifySystem:
    """Always re-sample 3x and majority vote — upper cost bound."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001", n_samples: int = 3):
        self._model = model
        self._program = FourStageMultiHopQA()
        self._n_samples = n_samples

    @property
    def name(self) -> str:
        return "AlwaysVerify"

    def process(self, example: dict[str, Any]) -> dict[str, Any]:
        lm = _make_lm(self._model)
        with dspy.context(lm=lm):
            passages = self._program._split_passages(example["context"])

            # Run decompose + retrieve + reason once
            decomp = self._program.decompose(question=example["question"])
            sq1 = decomp.sub_question_1 if hasattr(decomp, "sub_question_1") else example["question"]
            sq2 = decomp.sub_question_2 if hasattr(decomp, "sub_question_2") else example["question"]

            context_1 = " ".join(self._program._bm25_retrieve(sq1, passages))
            context_2 = " ".join(self._program._bm25_retrieve(sq2, passages))

            hop1 = self._program.reason_hop1(context=context_1, sub_question=sq1)
            hop2 = self._program.reason_hop2(
                context=context_2, sub_question=sq2,
                prior_answer=hop1.intermediate_answer if hasattr(hop1, "intermediate_answer") else "",
            )

            int1 = hop1.intermediate_answer if hasattr(hop1, "intermediate_answer") else str(_first_string_field(hop1) or "")
            int2 = hop2.intermediate_answer if hasattr(hop2, "intermediate_answer") else str(_first_string_field(hop2) or "")

            # Re-sample synthesize step N times
            answers = []
            for _ in range(self._n_samples):
                pred = self._program.synthesize(
                    question=example["question"],
                    intermediate_1=int1,
                    intermediate_2=int2,
                    config={"temperature": 1.0},
                )
                ans = pred.answer if hasattr(pred, "answer") else str(_first_string_field(pred) or "")
                answers.append(ans)

            counter = Counter(answers)
            majority, _ = counter.most_common(1)[0]

        return {
            "response": majority,
            "lm_calls": 3 + self._n_samples,  # decompose + 2 hops + N synth
        }


class PerAgentThresholdSystem:
    """Fire verification when the SYNTHESIZE step shows self-inconsistency.

    Per-agent analog of QSR: checks self-consistency only on the final
    synthesize step (one agent). Does NOT accumulate signals across
    decompose/reason stages. If the two synthesis answers diverge,
    majority-votes from 3 more samples.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        n_samples: int = 3,
    ):
        self._model = model
        self._program = FourStageMultiHopQA()
        self._n_samples = n_samples

    @property
    def name(self) -> str:
        return "PerAgentThreshold"

    def process(self, example: dict[str, Any]) -> dict[str, Any]:
        lm = _make_lm(self._model)
        with dspy.context(lm=lm):
            passages = self._program._split_passages(example["context"])

            # Run full 4-stage pipeline
            decomp = self._program.decompose(question=example["question"])
            sq1 = decomp.sub_question_1 if hasattr(decomp, "sub_question_1") else example["question"]
            sq2 = decomp.sub_question_2 if hasattr(decomp, "sub_question_2") else example["question"]

            context_1 = " ".join(self._program._bm25_retrieve(sq1, passages))
            context_2 = " ".join(self._program._bm25_retrieve(sq2, passages))

            hop1 = self._program.reason_hop1(context=context_1, sub_question=sq1)
            hop2 = self._program.reason_hop2(
                context=context_2, sub_question=sq2,
                prior_answer=hop1.intermediate_answer if hasattr(hop1, "intermediate_answer") else "",
            )

            int1 = hop1.intermediate_answer if hasattr(hop1, "intermediate_answer") else str(_first_string_field(hop1) or "")
            int2 = hop2.intermediate_answer if hasattr(hop2, "intermediate_answer") else str(_first_string_field(hop2) or "")

            lm_calls = 3  # decompose + 2 hops

            # Self-consistency on synthesize step ONLY (per-agent, no accumulation)
            pred_1 = self._program.synthesize(
                question=example["question"],
                intermediate_1=int1, intermediate_2=int2,
            )
            ans_1 = pred_1.answer if hasattr(pred_1, "answer") else str(_first_string_field(pred_1) or "")
            lm_calls += 1

            pred_2 = self._program.synthesize(
                question=example["question"],
                intermediate_1=int1, intermediate_2=int2,
                config={"temperature": 1.0},
            )
            ans_2 = pred_2.answer if hasattr(pred_2, "answer") else str(_first_string_field(pred_2) or "")
            lm_calls += 1

            diverged = _normalize_for_comparison(ans_1) != _normalize_for_comparison(ans_2)

            if diverged:
                answers = [ans_1, ans_2]
                for _ in range(self._n_samples):
                    try:
                        extra = self._program.synthesize(
                            question=example["question"],
                            intermediate_1=int1, intermediate_2=int2,
                            config={"temperature": 1.0},
                        )
                        ans = extra.answer if hasattr(extra, "answer") else str(_first_string_field(extra) or "")
                        answers.append(ans)
                    except Exception:
                        pass
                lm_calls += self._n_samples

                counter = Counter(answers)
                majority, _ = counter.most_common(1)[0]
                return {"response": majority, "lm_calls": lm_calls, "per_agent_fired": True}
            else:
                return {"response": ans_1, "lm_calls": lm_calls, "per_agent_fired": False}


class QSRSystem:
    """Full Quorum Sensing Router with 4-stage pipeline."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        threshold: float = 0.20,
        decay: float = 0.90,
    ):
        self._model = model
        self._program = QSRFourStageMultiHopQA(threshold=threshold, decay=decay)

    @property
    def name(self) -> str:
        return "QSR"

    def process(self, example: dict[str, Any]) -> dict[str, Any]:
        pred = _run_program(self._program, example, self._model)
        answer = pred.answer if hasattr(pred, "answer") else str(_first_string_field(pred) or "")
        qsr_meta = getattr(pred, "_qsr_meta", {})

        # Cost: 4 stages × 2 (one original + one consistency check) + verification samples
        lm_calls = 8  # 4 stages × 2 calls each (original + consistency)
        route = qsr_meta.get("route", "passthrough")
        if route == "majority_vote":
            lm_calls += qsr_meta.get("n_samples", 3) - 1

        return {
            "response": answer,
            "lm_calls": lm_calls,
            "qsr_fired": qsr_meta.get("fired", False),
            "qsr_fired_early": qsr_meta.get("fired_early", False),
            "qsr_concentration": qsr_meta.get("concentration", 0.0),
            "qsr_concentration_before_synth": qsr_meta.get("concentration_before_synth", 0.0),
            "qsr_route": route,
        }


class OracleVerificationSystem:
    """Verify only when the baseline would have gotten it wrong.

    Requires running NoVerification first and passing in the set of
    example IDs that the baseline got wrong.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        wrong_ids: set | None = None,
        n_samples: int = 3,
    ):
        self._model = model
        self._program = FourStageMultiHopQA()
        self._wrong_ids = wrong_ids or set()
        self._n_samples = n_samples

    @property
    def name(self) -> str:
        return "OracleVerification"

    def set_wrong_ids(self, wrong_ids: set) -> None:
        self._wrong_ids = wrong_ids

    def process(self, example: dict[str, Any]) -> dict[str, Any]:
        example_id = example.get("id", "")
        should_verify = example_id in self._wrong_ids

        lm = _make_lm(self._model)
        with dspy.context(lm=lm):
            passages = self._program._split_passages(example["context"])

            decomp = self._program.decompose(question=example["question"])
            sq1 = decomp.sub_question_1 if hasattr(decomp, "sub_question_1") else example["question"]
            sq2 = decomp.sub_question_2 if hasattr(decomp, "sub_question_2") else example["question"]

            context_1 = " ".join(self._program._bm25_retrieve(sq1, passages))
            context_2 = " ".join(self._program._bm25_retrieve(sq2, passages))

            hop1 = self._program.reason_hop1(context=context_1, sub_question=sq1)
            hop2 = self._program.reason_hop2(
                context=context_2, sub_question=sq2,
                prior_answer=hop1.intermediate_answer if hasattr(hop1, "intermediate_answer") else "",
            )

            int1 = hop1.intermediate_answer if hasattr(hop1, "intermediate_answer") else str(_first_string_field(hop1) or "")
            int2 = hop2.intermediate_answer if hasattr(hop2, "intermediate_answer") else str(_first_string_field(hop2) or "")

            lm_calls = 3  # decompose + 2 hops

            if should_verify:
                answers = []
                for _ in range(self._n_samples):
                    pred = self._program.synthesize(
                        question=example["question"],
                        intermediate_1=int1, intermediate_2=int2,
                        config={"temperature": 1.0},
                    )
                    ans = pred.answer if hasattr(pred, "answer") else str(_first_string_field(pred) or "")
                    answers.append(ans)

                counter = Counter(answers)
                majority, _ = counter.most_common(1)[0]
                lm_calls += self._n_samples
                return {"response": majority, "lm_calls": lm_calls, "oracle_verified": True}
            else:
                pred = self._program.synthesize(
                    question=example["question"],
                    intermediate_1=int1, intermediate_2=int2,
                )
                lm_calls += 1
                return {
                    "response": pred.answer if hasattr(pred, "answer") else str(_first_string_field(pred) or ""),
                    "lm_calls": lm_calls,
                    "oracle_verified": False,
                }


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

import random


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Returns (mean, lower, upper).
    """
    if not values:
        return 0.0, 0.0, 0.0

    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(values) for _ in range(n)]
        means.append(sum(sample) / len(sample))

    means.sort()
    alpha = (1.0 - ci) / 2.0
    lower_idx = max(0, int(math.floor(alpha * n_bootstrap)))
    upper_idx = min(n_bootstrap - 1, int(math.ceil((1.0 - alpha) * n_bootstrap)) - 1)

    return sum(values) / n, means[lower_idx], means[upper_idx]


def bootstrap_delta_ci(
    values_a: list[float],
    values_b: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI on the delta (A - B) for paired samples.

    Returns (mean_delta, lower, upper).
    """
    if not values_a or not values_b or len(values_a) != len(values_b):
        return 0.0, 0.0, 0.0

    rng = random.Random(seed)
    n = len(values_a)
    deltas_boot = []
    for _ in range(n_bootstrap):
        indices = [rng.randrange(n) for _ in range(n)]
        mean_a = sum(values_a[i] for i in indices) / n
        mean_b = sum(values_b[i] for i in indices) / n
        deltas_boot.append(mean_a - mean_b)

    deltas_boot.sort()
    alpha = (1.0 - ci) / 2.0
    lower_idx = max(0, int(math.floor(alpha * n_bootstrap)))
    upper_idx = min(n_bootstrap - 1, int(math.ceil((1.0 - ci) / 2.0 * n_bootstrap)))
    # Fix: upper index
    upper_idx = min(n_bootstrap - 1, int(math.ceil((1.0 - alpha) * n_bootstrap)) - 1)

    mean_delta = sum(a - b for a, b in zip(values_a, values_b)) / n
    return mean_delta, deltas_boot[lower_idx], deltas_boot[upper_idx]


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


def run_qsr_benchmark(
    datasets: list[str] | None = None,
    models: list[str] | None = None,
    n_examples: int = 200,
    threshold: float = 0.35,
    decay: float = 0.85,
    output_path: str = "qsr_benchmark_results.json",
) -> dict:
    """Run the full QSR benchmark.

    For each (dataset, model) pair:
    1. Load dataset
    2. Run all 5 systems on the same examples
    3. Score with AnswerQuality evaluator
    4. Collect per-example results including QSR metadata
    5. Compute aggregate metrics + QSR-specific metrics
    """
    if datasets is None:
        datasets = ["musique", "frames"]
    if models is None:
        models = ["claude-haiku-4-5-20251001"]

    evaluator = AnswerQuality()
    all_results: dict[str, Any] = {
        "config": {
            "datasets": datasets,
            "models": models,
            "n_examples": n_examples,
            "threshold": threshold,
            "decay": decay,
        },
        "results": {},
        "summary": {},
    }

    for dataset_name in datasets:
        if dataset_name not in DATASET_LOADERS:
            print(f"[WARN] Unknown dataset: {dataset_name}, skipping", file=sys.stderr)
            continue

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Loading dataset: {dataset_name}", file=sys.stderr)
        examples = DATASET_LOADERS[dataset_name](n=n_examples)
        print(f"  Loaded {len(examples)} examples", file=sys.stderr)

        for model in models:
            run_key = f"{dataset_name}/{model}"
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"Running: {run_key}", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)

            # Build systems
            no_verify = NoVerificationSystem(model=model)
            always_verify = AlwaysVerifySystem(model=model)
            per_agent = PerAgentThresholdSystem(model=model)
            qsr_sys = QSRSystem(model=model, threshold=threshold, decay=decay)
            oracle = OracleVerificationSystem(model=model)

            # Phase 1: Run NoVerification baseline to determine oracle set
            print(f"\n[Phase 1] Running {no_verify.name}...", file=sys.stderr)
            baseline_results = _run_system_on_examples(no_verify, examples, evaluator, "Phase 1")

            # Determine which examples the baseline got wrong (f1 < 0.5)
            wrong_ids = set()
            for row in baseline_results:
                if row["scores"].get("f1", 0.0) < 0.5:
                    wrong_ids.add(row["example_id"])
            oracle.set_wrong_ids(wrong_ids)
            print(f"  Baseline got {len(wrong_ids)}/{len(examples)} wrong (f1 < 0.5)", file=sys.stderr)

            # Phase 2: Run remaining systems
            systems_to_run = [
                ("AlwaysVerify", always_verify),
                ("PerAgentThreshold", per_agent),
                ("QSR", qsr_sys),
                ("OracleVerification", oracle),
            ]

            system_results: dict[str, list[dict]] = {
                no_verify.name: baseline_results,
            }

            for sys_name, system in systems_to_run:
                print(f"\n[Phase 2] Running {sys_name}...", file=sys.stderr)
                results = _run_system_on_examples(system, examples, evaluator, sys_name)
                system_results[sys_name] = results

            # Compute aggregate metrics
            aggregates = _compute_aggregates(system_results, examples)

            all_results["results"][run_key] = {
                "per_example": system_results,
                "aggregates": aggregates,
            }

            # Print summary table
            _print_summary_table(run_key, aggregates)

    # Compute cross-dataset summary
    all_results["summary"] = _compute_summary(all_results["results"])

    # Write output
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults written to {output_path}", file=sys.stderr)

    return all_results


def _run_system_on_examples(
    system: Any,
    examples: list[dict],
    evaluator: AnswerQuality,
    label: str,
) -> list[dict]:
    """Run a system on all examples, collecting results with error handling."""
    results = []
    total = len(examples)

    for i, example in enumerate(examples):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{label}] {i+1}/{total}", file=sys.stderr)

        row: dict[str, Any] = {
            "example_id": example.get("id", i),
            "question": example.get("question", ""),
            "gold_answer": example.get("answer", ""),
        }

        t0 = time.monotonic()
        try:
            processed = system.process(example)
            row["response"] = processed.get("response", "")
            row["lm_calls"] = processed.get("lm_calls", 0)

            # Copy QSR metadata if present
            for key in ("qsr_fired", "qsr_concentration", "qsr_assessment", "qsr_route",
                        "qsr_consistency_uncertainty",
                        "per_agent_fired", "oracle_verified"):
                if key in processed:
                    row[key] = processed[key]

            # Score
            scores = evaluator.score(example, processed)
            row["scores"] = scores

        except Exception as e:
            row["response"] = ""
            row["scores"] = {"f1": 0.0, "exact_match": 0.0, "recall": 0.0, "contains": 0.0}
            row["lm_calls"] = 0
            row["error"] = f"{type(e).__name__}: {e}"
            print(f"    [ERROR] Example {example.get('id', i)}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

        row["latency"] = time.monotonic() - t0
        results.append(row)

    return results


def _compute_aggregates(
    system_results: dict[str, list[dict]],
    examples: list[dict],
) -> dict[str, Any]:
    """Compute aggregate metrics for all systems."""
    aggregates: dict[str, Any] = {}

    for sys_name, rows in system_results.items():
        f1_vals = [r["scores"].get("f1", 0.0) for r in rows]
        em_vals = [r["scores"].get("exact_match", 0.0) for r in rows]
        recall_vals = [r["scores"].get("recall", 0.0) for r in rows]
        contains_vals = [r["scores"].get("contains", 0.0) for r in rows]
        lm_calls = [r.get("lm_calls", 0) for r in rows]

        f1_mean, f1_lo, f1_hi = bootstrap_ci(f1_vals)
        em_mean, em_lo, em_hi = bootstrap_ci(em_vals)

        agg: dict[str, Any] = {
            "n": len(rows),
            "f1_mean": f1_mean,
            "f1_ci_lower": f1_lo,
            "f1_ci_upper": f1_hi,
            "exact_match_mean": em_mean,
            "exact_match_ci_lower": em_lo,
            "exact_match_ci_upper": em_hi,
            "recall_mean": sum(recall_vals) / max(len(recall_vals), 1),
            "contains_mean": sum(contains_vals) / max(len(contains_vals), 1),
            "total_lm_calls": sum(lm_calls),
            "mean_lm_calls": sum(lm_calls) / max(len(lm_calls), 1),
        }

        # QSR-specific metrics
        if sys_name == "QSR":
            fired_rows = [r for r in rows if r.get("qsr_fired", False)]
            not_fired_rows = [r for r in rows if not r.get("qsr_fired", False)]

            agg["qsr_fire_rate"] = len(fired_rows) / max(len(rows), 1)

            if fired_rows:
                fired_f1 = [r["scores"].get("f1", 0.0) for r in fired_rows]
                agg["f1_when_fired"] = sum(fired_f1) / len(fired_f1)
            else:
                agg["f1_when_fired"] = None

            if not_fired_rows:
                not_fired_f1 = [r["scores"].get("f1", 0.0) for r in not_fired_rows]
                agg["f1_when_not_fired"] = sum(not_fired_f1) / len(not_fired_f1)
            else:
                agg["f1_when_not_fired"] = None

            # Route distribution
            route_counts: dict[str, int] = {}
            for r in rows:
                route = r.get("qsr_route", "unknown")
                route_counts[route] = route_counts.get(route, 0) + 1
            agg["route_distribution"] = route_counts

        aggregates[sys_name] = agg

    # Key comparative metric: QSR accuracy on fired examples vs baseline on same examples
    if "QSR" in system_results and "NoVerification" in system_results:
        qsr_rows = system_results["QSR"]
        baseline_rows = system_results["NoVerification"]

        # Build lookup by example_id
        baseline_by_id = {r["example_id"]: r for r in baseline_rows}
        qsr_by_id = {r["example_id"]: r for r in qsr_rows}

        # On fired examples
        fired_ids = [r["example_id"] for r in qsr_rows if r.get("qsr_fired", False)]
        if fired_ids:
            qsr_fired_f1 = [qsr_by_id[eid]["scores"].get("f1", 0.0) for eid in fired_ids if eid in qsr_by_id]
            baseline_fired_f1 = [baseline_by_id[eid]["scores"].get("f1", 0.0) for eid in fired_ids if eid in baseline_by_id]

            if qsr_fired_f1 and baseline_fired_f1 and len(qsr_fired_f1) == len(baseline_fired_f1):
                delta_mean, delta_lo, delta_hi = bootstrap_delta_ci(qsr_fired_f1, baseline_fired_f1)
                aggregates["_comparative"] = {
                    "fired_example_count": len(fired_ids),
                    "qsr_f1_on_fired": sum(qsr_fired_f1) / len(qsr_fired_f1),
                    "baseline_f1_on_fired": sum(baseline_fired_f1) / len(baseline_fired_f1),
                    "delta_f1_mean": delta_mean,
                    "delta_f1_ci_lower": delta_lo,
                    "delta_f1_ci_upper": delta_hi,
                    "significant": delta_lo > 0 or delta_hi < 0,  # CI doesn't cross zero
                }

    return aggregates


def _print_summary_table(run_key: str, aggregates: dict[str, Any]) -> None:
    """Print a formatted summary table to stderr."""
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"  RESULTS: {run_key}", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)

    header = f"{'System':<22} {'F1':>8} {'F1 95%CI':>18} {'EM':>8} {'Recall':>8} {'LM Calls':>10}"
    print(header, file=sys.stderr)
    print("-" * 80, file=sys.stderr)

    system_order = ["NoVerification", "AlwaysVerify", "PerAgentThreshold", "QSR", "OracleVerification"]
    for sys_name in system_order:
        if sys_name not in aggregates:
            continue
        agg = aggregates[sys_name]
        ci = f"[{agg['f1_ci_lower']:.3f}, {agg['f1_ci_upper']:.3f}]"
        line = (
            f"{sys_name:<22} "
            f"{agg['f1_mean']:>8.4f} "
            f"{ci:>18} "
            f"{agg['exact_match_mean']:>8.4f} "
            f"{agg['recall_mean']:>8.4f} "
            f"{agg['total_lm_calls']:>10}"
        )
        print(line, file=sys.stderr)

    # QSR-specific stats
    if "QSR" in aggregates:
        qsr = aggregates["QSR"]
        print(f"\n  QSR Details (self-consistency based):", file=sys.stderr)
        print(f"    Fire rate: {qsr.get('qsr_fire_rate', 0):.1%}", file=sys.stderr)
        if qsr.get("f1_when_fired") is not None:
            print(f"    F1 when fired:     {qsr['f1_when_fired']:.4f}", file=sys.stderr)
        if qsr.get("f1_when_not_fired") is not None:
            print(f"    F1 when not fired: {qsr['f1_when_not_fired']:.4f}", file=sys.stderr)
        if "route_distribution" in qsr:
            print(f"    Routes: {qsr['route_distribution']}", file=sys.stderr)

    # Comparative stats
    if "_comparative" in aggregates:
        comp = aggregates["_comparative"]
        print(f"\n  Key Comparison (QSR vs Baseline on fired examples):", file=sys.stderr)
        print(f"    Fired examples: {comp['fired_example_count']}", file=sys.stderr)
        print(f"    QSR F1:      {comp['qsr_f1_on_fired']:.4f}", file=sys.stderr)
        print(f"    Baseline F1: {comp['baseline_f1_on_fired']:.4f}", file=sys.stderr)
        print(f"    Delta F1:    {comp['delta_f1_mean']:+.4f} [{comp['delta_f1_ci_lower']:+.4f}, {comp['delta_f1_ci_upper']:+.4f}]", file=sys.stderr)
        sig = "YES" if comp["significant"] else "NO"
        print(f"    Significant: {sig}", file=sys.stderr)


def _compute_summary(results: dict[str, Any]) -> dict[str, Any]:
    """Compute cross-dataset summary."""
    summary: dict[str, Any] = {}

    # Aggregate per system across all runs
    system_f1s: dict[str, list[float]] = {}
    system_calls: dict[str, list[int]] = {}

    for run_key, run_data in results.items():
        aggs = run_data.get("aggregates", {})
        for sys_name, agg in aggs.items():
            if sys_name.startswith("_"):
                continue
            system_f1s.setdefault(sys_name, []).append(agg.get("f1_mean", 0.0))
            system_calls.setdefault(sys_name, []).append(agg.get("total_lm_calls", 0))

    for sys_name in system_f1s:
        f1_list = system_f1s[sys_name]
        calls_list = system_calls.get(sys_name, [])
        summary[sys_name] = {
            "avg_f1": sum(f1_list) / max(len(f1_list), 1),
            "avg_total_lm_calls": sum(calls_list) / max(len(calls_list), 1),
            "n_runs": len(f1_list),
        }

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QSR Benchmark: Compare quorum sensing router against baselines on multi-hop QA.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset to evaluate on (repeatable). Available: musique, frames. Default: both.",
    )
    parser.add_argument(
        "-n", "--n-examples",
        type=int,
        default=200,
        help="Number of examples per dataset (default: 200).",
    )
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="Model name (default: claude-haiku-4-5-20251001).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.20,
        help="QSR quorum threshold (default: 0.20).",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.85,
        help="QSR decay factor (default: 0.85).",
    )
    parser.add_argument(
        "--output",
        default="qsr_benchmark_results.json",
        help="Output JSON path (default: qsr_benchmark_results.json).",
    )

    args = parser.parse_args()

    datasets = args.dataset or ["musique", "frames"]
    models = [args.model]

    run_qsr_benchmark(
        datasets=datasets,
        models=models,
        n_examples=args.n_examples,
        threshold=args.threshold,
        decay=args.decay,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
