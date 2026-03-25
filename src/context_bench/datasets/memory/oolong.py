"""OOLONG dataset loader returning typed BenchmarkExample instances.

OOLONG is a challenging aggregation benchmark for long-context models.
Tasks require classifying and counting items across a large document.

Two variants:
- oolong-synth: Synthetic tasks (spam classification, etc.) with controllable context lengths
- oolong-real: Real-world D&D transcript analysis tasks

Task types (synth): MOST_FREQ, LEAST_FREQ, RELATIVE_FREQ, NUMERIC_ONE_CLASS,
    REPRESENTED_N_TIMES, SECOND_MOST_FREQ
Answer types: LABEL, NUMERIC, COMPARISON, USER, DATE, MONTH_YEAR

Reference: Bertsch et al., 2025. https://arxiv.org/abs/2511.02817
HuggingFace: oolongbench/oolong-synth, oolongbench/oolong-real

Requires: pip install context-bench[datasets]
"""

from __future__ import annotations

from typing import Any

from context_bench.memory_types import BenchmarkExample, BenchmarkQuery, DocumentChunk


def _require_datasets() -> Any:
    try:
        import datasets
        return datasets
    except ImportError:
        raise ImportError(
            "HuggingFace datasets required. Install with: pip install context-bench[datasets]"
        )


def _normalize_answer(raw_answer: Any) -> str:
    """Normalize OOLONG answer to a string.

    Answers come as string representations of lists (e.g. "['spam']", "[4]")
    or actual lists. We extract the element and convert to a clean string.
    """
    import ast

    s = str(raw_answer).strip()

    # Try to parse as a Python literal (handles "['spam']", "[4]", etc.)
    if s.startswith("["):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                if len(parsed) == 1:
                    return str(parsed[0])
                return ", ".join(str(x) for x in parsed)
        except (ValueError, SyntaxError):
            pass

    # If it's already a list object
    if isinstance(raw_answer, list):
        if len(raw_answer) == 1:
            return str(raw_answer[0])
        return ", ".join(str(x) for x in raw_answer)

    return s


def oolong(
    n: int | None = None,
    variant: str = "synth",
    split: str = "validation",
    task_types: list[str] | None = None,
    answer_types: list[str] | None = None,
    max_context_len: int | None = None,
    min_context_len: int | None = None,
) -> list[BenchmarkExample]:
    """Load the OOLONG aggregation benchmark.

    Each context window becomes one BenchmarkExample with multiple queries
    (all questions sharing the same context_window_id are grouped together).

    Args:
        n: Max number of examples (context windows) to return.
        variant: "synth" or "real". Default "synth".
        split: Dataset split ("validation" or "test"). Default "validation".
        task_types: Filter to specific task types. Options for synth:
            "MOST_FREQ", "LEAST_FREQ", "RELATIVE_FREQ", "NUMERIC_ONE_CLASS",
            "REPRESENTED_N_TIMES", "SECOND_MOST_FREQ".
            None loads all types. Partial matches work (e.g. "NUMERIC").
        answer_types: Filter to specific answer types. Options:
            "LABEL", "NUMERIC", "COMPARISON", "USER", "DATE", "MONTH_YEAR".
            None loads all types.
        max_context_len: Max context length in tokens to include.
        min_context_len: Min context length in tokens to include.

    Returns:
        List of BenchmarkExample instances for evaluate_memory().
    """
    ds = _require_datasets()

    if variant == "synth":
        hf_name = "oolongbench/oolong-synth"
        config = None
    elif variant == "real":
        hf_name = "oolongbench/oolong-real"
        config = "dnd"
    else:
        raise ValueError(f"Unknown OOLONG variant: {variant!r}. Use 'synth' or 'real'.")

    print(f"[OOLONG] Loading {hf_name} split={split} (streaming)...", flush=True)
    if config:
        dataset = ds.load_dataset(hf_name, config, split=split, streaming=True)
    else:
        dataset = ds.load_dataset(hf_name, split=split, streaming=True)

    # Normalize task_types filter for matching
    task_filter = None
    if task_types is not None:
        task_filter = [t.upper().replace("TASK_TYPE.", "") for t in task_types]

    answer_filter = None
    if answer_types is not None:
        answer_filter = [t.upper().replace("ANSWER_TYPE.", "") for t in answer_types]

    # Group rows by context_window_id
    # Each context window gets one BenchmarkExample with multiple queries
    from collections import OrderedDict
    windows: OrderedDict[str, dict[str, Any]] = OrderedDict()

    # When streaming with n limit, we need a generous max_rows to ensure
    # we find enough unique context windows (each has ~6-25 questions).
    max_rows = (n or 100) * 50 if n is not None else 50000
    rows_seen = 0

    for row in dataset:
        rows_seen += 1
        if rows_seen > max_rows:
            break

        # Apply context length filters
        ctx_len = row.get("context_len", 0)
        if max_context_len is not None and ctx_len > max_context_len:
            continue
        if min_context_len is not None and ctx_len < min_context_len:
            continue

        # Determine task type string (strip prefix)
        raw_task = row.get("task", row.get("question_type", ""))
        task_str = raw_task.replace("TASK_TYPE.", "")

        # Apply task filter
        if task_filter is not None:
            if not any(tf in task_str.upper() for tf in task_filter):
                continue

        # Apply answer type filter
        raw_answer_type = row.get("answer_type", "")
        answer_type_str = raw_answer_type.replace("ANSWER_TYPE.", "")
        if answer_filter is not None:
            if not any(af in answer_type_str.upper() for af in answer_filter):
                continue

        # Get context window ID
        cwid = str(row.get("context_window_id", row.get("id", "")))

        if cwid not in windows:
            # Check if we already have enough context windows
            if n is not None and len(windows) >= n:
                break
            # First time seeing this context window
            context_text = row.get("context_window_text", "")
            windows[cwid] = {
                "context_text": context_text,
                "context_len": ctx_len,
                "source_dataset": row.get("dataset", row.get("campaign", variant)),
                "queries": [],
                "row_id": str(row.get("id", cwid)),
            }

        # Build the query
        question = row.get("question", "")
        answer = _normalize_answer(row.get("answer", ""))

        if question and answer:
            windows[cwid]["queries"].append(BenchmarkQuery(
                question=question,
                answer=answer,
                query_type=task_str,
                metadata={
                    "answer_type": answer_type_str,
                    "raw_task": raw_task,
                    "row_id": str(row.get("id", "")),
                },
            ))

    # Convert to BenchmarkExample list
    examples: list[BenchmarkExample] = []
    for cwid, window in windows.items():
        if not window["queries"]:
            continue

        # Split context into document chunks (by paragraph or fixed size)
        context_text = window["context_text"]
        chunks = _split_into_chunks(context_text, cwid, window["source_dataset"])

        examples.append(BenchmarkExample(
            id=f"oolong_{variant}_{cwid}",
            items=chunks,
            queries=window["queries"],
            dataset=f"oolong_{variant}",
            metadata={
                "context_len": window["context_len"],
                "source_dataset": window["source_dataset"],
                "num_queries": len(window["queries"]),
                "variant": variant,
            },
        ))

        if n is not None and len(examples) >= n:
            break

    print(f"[OOLONG] Loaded {len(examples)} examples with "
          f"{sum(len(e.queries) for e in examples)} total queries", flush=True)
    return examples


def _split_into_chunks(
    text: str,
    document_id: str,
    source: str,
    chunk_size: int = 500,
) -> list[DocumentChunk]:
    """Split context text into DocumentChunk items.

    Splits on double-newlines (paragraphs) first, then on single newlines
    for long blocks. Falls back to fixed-size character splitting.
    """
    if not text.strip():
        return [DocumentChunk(
            content="(empty context)",
            document_id=document_id,
            position=0,
            source=source,
        )]

    # Split by lines (each line in OOLONG synth is typically one data point)
    lines = text.split("\n")
    chunks: list[DocumentChunk] = []
    current_chunk: list[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line)
        if current_len + line_len > chunk_size and current_chunk:
            chunks.append(DocumentChunk(
                content="\n".join(current_chunk),
                document_id=document_id,
                position=len(chunks),
                source=source,
            ))
            current_chunk = []
            current_len = 0
        current_chunk.append(line)
        current_len += line_len + 1  # +1 for newline

    if current_chunk:
        chunks.append(DocumentChunk(
            content="\n".join(current_chunk),
            document_id=document_id,
            position=len(chunks),
            source=source,
        ))

    return chunks
