"""NLI and fact-verification dataset loaders.

Requires: pip install context-bench[datasets]
"""

from __future__ import annotations

from typing import Any


def _require_datasets() -> Any:
    try:
        import datasets
        return datasets
    except ImportError:
        raise ImportError(
            "HuggingFace datasets required. Install with: pip install context-bench[datasets]"
        )


def contract_nli(n: int | None = None, split: str = "validation") -> list[dict[str, Any]]:
    """Load ContractNLI legal NLI dataset (via SCROLLS).

    Each example has: id, context, question, answer.
    Answers are one of: "Entailment", "Contradiction", "Not mentioned".
    """
    ds = _require_datasets()
    dataset = ds.load_dataset(
        "tau/scrolls", "contract_nli", split=split, trust_remote_code=True,
    )

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        raw_input = item.get("input", "")
        # SCROLLS ContractNLI format: contract text followed by
        # "\nHypothesis: <hypothesis>" or "\nStatement: <statement>"
        context = raw_input
        question = "Does the contract entail, contradict, or not mention the hypothesis?"
        for sep in ("\nHypothesis:", "\nStatement:"):
            if sep in raw_input:
                idx = raw_input.rfind(sep)
                context = raw_input[:idx].strip()
                hypothesis = raw_input[idx + len(sep):].strip()
                question = f"Does the contract entail, contradict, or not mention: {hypothesis}"
                break

        examples.append({
            "id": item.get("id", i),
            "context": context,
            "question": question,
            "answer": item.get("output", ""),
        })

    return examples


def scifact(n: int | None = None, split: str = "validation") -> list[dict[str, Any]]:
    """Load SciFact scientific fact verification dataset.

    Joins claims with their cited corpus abstracts.
    Each example has: id, context, question, answer.
    Answers are "SUPPORTS" or "REFUTES".
    """
    ds = _require_datasets()

    # Load claims and corpus
    claims_dataset = ds.load_dataset("allenai/scifact", "claims", split=split)
    corpus_dataset = ds.load_dataset("allenai/scifact", "corpus", split="train")

    # Build corpus lookup: doc_id -> abstract text
    corpus_map: dict[int, str] = {}
    for doc in corpus_dataset:
        doc_id = doc.get("doc_id", doc.get("id"))
        if doc_id is not None:
            abstract = doc.get("abstract", [])
            if isinstance(abstract, list):
                corpus_map[doc_id] = " ".join(abstract)
            else:
                corpus_map[doc_id] = str(abstract)

    examples = []
    for i, claim in enumerate(claims_dataset):
        if n is not None and i >= n:
            break

        claim_text = claim.get("claim", "")
        cited_doc_ids = claim.get("cited_doc_ids", [])
        evidence = claim.get("evidence", {})

        # Collect abstracts from cited documents
        context_parts = []
        label = None
        for doc_id in cited_doc_ids:
            abstract = corpus_map.get(doc_id, "")
            if abstract:
                context_parts.append(abstract)
            # Get evidence label from first available evidence
            if label is None and evidence:
                doc_evidence = evidence.get(str(doc_id), [])
                if doc_evidence:
                    first_ev = doc_evidence[0] if isinstance(doc_evidence, list) else doc_evidence
                    if isinstance(first_ev, dict):
                        label = first_ev.get("label")

        if not context_parts:
            continue

        if label is None:
            label = "SUPPORTS"

        examples.append({
            "id": claim.get("id", i),
            "context": "\n\n".join(context_parts),
            "question": claim_text,
            "answer": label,
        })

    return examples
