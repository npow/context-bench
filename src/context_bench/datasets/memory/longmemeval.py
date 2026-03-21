"""LongMemEval dataset loader returning typed BenchmarkExample instances.

500 QA pairs, each with a full multi-session chat history as context.

Reference: Wu et al., ICLR 2025. https://arxiv.org/abs/2410.10813
HuggingFace: xiaowu0162/longmemeval (downloaded via hf_hub_download)

Requires: pip install context-bench[datasets]
"""

from __future__ import annotations

from context_bench.memory_types import BenchmarkExample, BenchmarkQuery, ConversationTurn


def longmemeval(
    n: int | None = None,
    question_types: list[str] | None = None,
    variant: str = "s",
) -> list[BenchmarkExample]:
    """Load the LongMemEval chat assistant memory benchmark.

    500 QA pairs, each with a full multi-session chat history as context.

    Reference: Wu et al., ICLR 2025. https://arxiv.org/abs/2410.10813
    HuggingFace: xiaowu0162/longmemeval (downloaded via hf_hub_download)

    Schema:
        question_id:       str
        question_type:     str (e.g. "single-session-user")
        question:          str
        answer:            str
        haystack_sessions: list of sessions, each session is a list of
                           {"role": str, "content": str} dicts

    Args:
        n: Max number of examples to load.
        question_types: Filter by question type substring. Options include:
            "single-session-user", "single-session-assistant",
            "multi-session", "temporal-reasoning", "knowledge-update",
            "abstention".
            None loads all types.
        variant: "s" for LongMemEval-S (~115K token histories),
                 "m" for LongMemEval-M (~1.5M token histories),
                 "oracle" for oracle-retrieval variant.

    Returns:
        List of BenchmarkExample instances for evaluate_memory().
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub required. Install with: pip install context-bench[datasets]"
        )
    import json

    path = hf_hub_download(
        "xiaowu0162/longmemeval",
        f"longmemeval_{variant}",
        repo_type="dataset",
    )
    with open(path) as f:
        data = json.load(f)

    examples: list[BenchmarkExample] = []
    for i, item in enumerate(data):
        if n is not None and i >= n:
            break

        question_type = item.get("question_type", "")
        if question_types is not None:
            if not any(qt in question_type for qt in question_types):
                continue

        # Flatten all haystack sessions into one turn list in order
        turns: list[ConversationTurn] = []
        for session_idx, session in enumerate(item.get("haystack_sessions", [])):
            # Each session is a list of {"role": ..., "content": ...} dicts
            if isinstance(session, list):
                session_id = str(session_idx)
                for turn in session:
                    if turn.get("content", "").strip():
                        turns.append(ConversationTurn(
                            content=turn["content"].strip(),
                            role=turn.get("role", "user"),
                            session_id=session_id,
                        ))

        question = item.get("question", "").strip()
        answer = item.get("answer", "")
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        answer = str(answer).strip()

        if not turns or not question or not answer:
            continue

        examples.append(BenchmarkExample(
            id=str(item.get("question_id", i)),
            items=turns,
            queries=[BenchmarkQuery(question=question, answer=answer, query_type=question_type)],
            dataset=f"longmemeval_{variant}",
        ))

    return examples
