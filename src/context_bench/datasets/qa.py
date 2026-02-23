"""QA dataset loaders.

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


def natural_questions(n: int | None = None, split: str = "validation") -> list[dict[str, Any]]:
    """Load Natural Questions open-domain QA dataset.

    Each example has: id, context, question, answer.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("google-research-datasets/nq_open", split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        examples.append({
            "id": i,
            "context": item["question"],
            "question": item["question"],
            "answer": item["answer"][0] if item.get("answer") else "",
        })

    return examples


def musique(n: int | None = None, split: str = "validation") -> list[dict[str, Any]]:
    """Load MuSiQue multi-hop QA dataset (answerable subset).

    Each example has: id, context, question, answer.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("bdsaglam/musique", "answerable", split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        context_parts = []
        for p in item.get("paragraphs", []):
            title = p.get("title", "")
            text = p.get("paragraph_text", "")
            context_parts.append(f"{title}: {text}")

        examples.append({
            "id": item.get("id", i),
            "context": "\n".join(context_parts),
            "question": item["question"],
            "answer": item["answer"],
        })

    return examples


def narrativeqa(n: int | None = None, split: str = "test") -> list[dict[str, Any]]:
    """Load NarrativeQA reading comprehension dataset.

    Uses document summaries as context (not full books).
    Each example has: id, context, question, answer.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("deepmind/narrativeqa", split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        doc = item.get("document", {})
        summary = doc.get("summary", {}) if isinstance(doc, dict) else {}
        context = summary.get("text", "") if isinstance(summary, dict) else ""

        answers = item.get("answers", [])
        answer = answers[0]["text"] if answers and isinstance(answers[0], dict) else ""

        examples.append({
            "id": i,
            "context": context,
            "question": item["question"],
            "answer": answer,
        })

    return examples


def triviaqa(n: int | None = None, split: str = "validation") -> list[dict[str, Any]]:
    """Load TriviaQA reading comprehension dataset.

    Uses search result contexts when available.
    Each example has: id, context, question, answer.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("mandarjoshi/trivia_qa", "rc", split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        search = item.get("search_results", {})
        contexts = search.get("search_context", []) if isinstance(search, dict) else []
        context = "\n".join(c for c in contexts if c) if contexts else item.get("question", "")

        answer_obj = item.get("answer", {})
        answer = answer_obj.get("value", "") if isinstance(answer_obj, dict) else ""

        examples.append({
            "id": item.get("question_id", i),
            "context": context,
            "question": item["question"],
            "answer": answer,
        })

    return examples


def frames(n: int | None = None, split: str = "test") -> list[dict[str, Any]]:
    """Load FRAMES multi-hop factual reasoning benchmark.

    Each example has: id, context, question, answer.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("google/frames-benchmark", split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        examples.append({
            "id": i,
            "context": item.get("Prompt", ""),
            "question": item.get("Prompt", ""),
            "answer": item.get("Answer", ""),
        })

    return examples


def qasper(n: int | None = None, split: str = "validation") -> list[dict[str, Any]]:
    """Load QASPer scientific paper QA dataset (full papers).

    Reconstructs full paper text from title, abstract, and section paragraphs.
    Uses the first answer text from each question's answer list.
    Each example has: id, context, question, answer.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("allenai/qasper", split=split)

    examples = []
    for item in dataset:
        # Build full paper text
        parts = []
        title = item.get("title", "")
        if title:
            parts.append(title)
        abstract = item.get("abstract", "")
        if abstract:
            parts.append(abstract)
        for section in item.get("full_text", {}).get("section_name", []):
            if section:
                parts.append(section)
        for para_list in item.get("full_text", {}).get("paragraphs", []):
            if isinstance(para_list, list):
                for para in para_list:
                    if para:
                        parts.append(para)
            elif para_list:
                parts.append(str(para_list))

        full_text = "\n\n".join(parts)

        # Each paper has multiple QA pairs
        qas = item.get("qas", {})
        questions = qas.get("question", [])
        answers_list = qas.get("answers", [])

        for q_idx, question in enumerate(questions):
            if n is not None and len(examples) >= n:
                break

            # Extract first answer text
            answer = ""
            if q_idx < len(answers_list):
                ans_obj = answers_list[q_idx]
                ans_texts = ans_obj.get("answer", []) if isinstance(ans_obj, dict) else []
                for a in ans_texts:
                    if isinstance(a, dict):
                        text = a.get("free_form_answer", "") or a.get("extractive_spans", [""])[0] if a.get("extractive_spans") else a.get("free_form_answer", "")
                        if text:
                            answer = str(text)
                            break
                    elif a:
                        answer = str(a)
                        break

            examples.append({
                "id": f"{item.get('id', '')}_{q_idx}",
                "context": full_text,
                "question": question,
                "answer": answer,
            })

        if n is not None and len(examples) >= n:
            break

    return examples[:n] if n is not None else examples


def quality(n: int | None = None, split: str = "validation") -> list[dict[str, Any]]:
    """Load QuALITY long-document multiple-choice QA dataset.

    Resolves 1-indexed gold_label to the actual option text.
    Each example has: id, context, question, answer.
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("tasksource/QuALITY", split=split)

    examples = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break
        options = item.get("options", [])
        gold_label = item.get("gold_label", 0)
        # gold_label is 1-indexed
        idx = gold_label - 1 if gold_label >= 1 else 0
        answer = options[idx] if 0 <= idx < len(options) else ""

        examples.append({
            "id": i,
            "context": item.get("article", ""),
            "question": item.get("question", ""),
            "answer": answer,
        })

    return examples
