"""LoCoMo dataset loader returning typed BenchmarkExample instances.

50 multi-session conversations with QA pairs covering:
single_hop, multi_hop, temporal, open_domain, adversarial.

Reference: Maharana et al., ACL 2024. https://arxiv.org/abs/2402.17753
HuggingFace: KhangPTT373/locomo_preprocess (split: test)

Requires: pip install context-bench[datasets]
"""

from __future__ import annotations

from typing import Any

from context_bench.memory_types import BenchmarkExample, BenchmarkQuery, ConversationTurn


def _require_datasets() -> Any:
    try:
        import datasets
        return datasets
    except ImportError:
        raise ImportError(
            "HuggingFace datasets required. Install with: pip install context-bench[datasets]"
        )


# LoCoMo category codes from the paper
_LOCOMO_CATEGORY = {
    1: "single_hop",
    2: "multi_hop",
    3: "temporal",
    4: "open_domain",
    5: "adversarial",
}


def locomo(
    n: int | None = None,
    qa_types: list[str] | None = None,
) -> list[BenchmarkExample]:
    """Load the LoCoMo long-term conversational memory benchmark.

    50 multi-session conversations with QA pairs covering:
    single_hop, multi_hop, temporal, open_domain, adversarial.

    Reference: Maharana et al., ACL 2024. https://arxiv.org/abs/2402.17753
    HuggingFace: KhangPTT373/locomo_preprocess (split: test)

    Schema:
        turns:     list of turn strings "timestamp\\n Speaker: text\\n"
        sessions:  list of session strings (grouped turns)
        questions: list of question strings
        answers:   list of answer strings
        category:  list of int codes (1-5)

    Args:
        n: Max number of conversations to load.
        qa_types: Filter to specific QA types. Options:
            "single_hop", "multi_hop", "temporal", "open_domain", "adversarial".
            None loads all types.

    Returns:
        List of BenchmarkExample instances for evaluate_memory().
    """
    ds = _require_datasets()
    dataset = ds.load_dataset("KhangPTT373/locomo_preprocess", split="test")

    examples: list[BenchmarkExample] = []
    for i, item in enumerate(dataset):
        if n is not None and i >= n:
            break

        # turns is a list of raw strings: "timestamp\n Speaker: text\n"
        # Preserve timestamps and real speaker names — both are critical for
        # LoCoMo temporal questions ("when did X happen?") and entity questions.
        #
        # Session assignment: LoCoMo's `sessions` field is a list of session text
        # blocks. We count speaker utterances in each session block to assign
        # session_id to each ConversationTurn (needed for BM25 session retrieval).
        raw_sessions_blocks = item.get("sessions", [])
        # Count how many speaker lines exist per session block
        turns_per_session = []
        for block in raw_sessions_blocks:
            count = sum(1 for l in block.split("\n")
                        if l.strip().startswith("Speaker_1:") or l.strip().startswith("Speaker_2:"))
            turns_per_session.append(max(1, count))
        # Build a session_id assignment list in flat-turn order
        session_assignment = []
        for sidx, cnt in enumerate(turns_per_session):
            session_assignment.extend([str(sidx)] * cnt)
        raw_turns = item.get("turns", [])
        turns: list[ConversationTurn] = []
        turn_idx = 0
        for raw in raw_turns:
            lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
            if not lines:
                continue

            # First line is typically the timestamp (e.g. "1:56 pm on 8 May, 2023")
            timestamp = ""
            content_lines = lines
            if lines and not any(lines[0].startswith(pfx) for pfx in ("Speaker_1:", "Speaker_2:")):
                # Check if this looks like a timestamp (contains time or date)
                first = lines[0]
                if any(kw in first.lower() for kw in ("am", "pm", "on ", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")):
                    timestamp = first
                    content_lines = lines[1:]

            for line in content_lines:
                # Detect speaker: "Name: text" or "Speaker_N: text"
                speaker = None
                content = ""
                if line.startswith("Speaker_1:"):
                    speaker = "Speaker_1"
                    content = line[len("Speaker_1:"):].strip()
                elif line.startswith("Speaker_2:"):
                    speaker = "Speaker_2"
                    content = line[len("Speaker_2:"):].strip()
                elif ":" in line:
                    parts = line.split(":", 1)
                    # Only treat as speaker if the part before ":" is a name (no digits)
                    candidate = parts[0].strip()
                    if candidate and not any(c.isdigit() for c in candidate):
                        speaker = candidate
                        content = parts[1].strip()

                if speaker and content:
                    role = "user" if len(turns) % 2 == 0 else "assistant"
                    # Prefix content with timestamp and real speaker name
                    prefix = ""
                    if timestamp:
                        prefix += f"[{timestamp}] "
                    if speaker not in ("Speaker_1", "Speaker_2"):
                        prefix += f"{speaker}: "
                    # Assign session_id from sessions field if available
                    sid = session_assignment[turn_idx] if turn_idx < len(session_assignment) else "0"
                    turns.append(ConversationTurn(
                        content=prefix + content,
                        role=role,
                        timestamp=timestamp or None,
                        speaker=speaker,
                        session_id=sid,
                    ))
                    turn_idx += 1

        questions = item.get("questions", [])
        answers = item.get("answers", [])
        categories = item.get("category", [])

        queries: list[BenchmarkQuery] = []
        for q, a, cat in zip(questions, answers, categories):
            qa_type = _LOCOMO_CATEGORY.get(cat, str(cat))
            if qa_types is not None and qa_type not in qa_types:
                continue
            if q and a:
                queries.append(BenchmarkQuery(question=q, answer=a, query_type=qa_type))

        if not turns or not queries:
            continue

        examples.append(BenchmarkExample(
            id=str(item.get("dialogue_id", i)),
            items=turns,
            queries=queries,
            dataset="locomo",
        ))

    return examples
