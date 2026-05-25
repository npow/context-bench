"""RLM REPL extension: adds memory_write() and consolidate() to the query loop.

Layer 1 prototype from the memory-first architecture research plan:
- memory_write(content, memory_type) — typed write during query processing
- consolidate() — compress current recursion frame before closing

The REPL loop follows the RLM protocol:
  answer = {"content": "", "ready": False}
  # model generates code that reads, writes, consolidates, then sets ready=True
  while not answer["ready"] and iterations < max_iterations:
      code = llm_generate(...)
      exec(code, namespace)

Usage tracking records whether the pretrained model spontaneously invokes
memory_write/consolidate — the research plan predicts it won't without RL training.
"""

from __future__ import annotations

import textwrap
import threading
import time
import traceback
from typing import Any

_EXEC_TIMEOUT_S: float = 10.0  # max seconds per REPL exec step

from context_bench.memory_types import Item, IngestResult, QueryResult
from context_bench.systems.rlm import RLMSystem

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]


_SYSTEM_PROMPT = textwrap.dedent("""
You are an agent that answers questions by managing memory programmatically.

You have access to these functions in your Python namespace:

  memory_read(query, k=20) -> list[str]
    Retrieve k relevant memory entries for the given query string.
    Returns a list of text snippets ranked by relevance.

  memory_write(content: str, memory_type: str = "episodic") -> None
    Write new content to persistent memory.
    memory_type must be one of: "episodic", "factual", "procedural"
    - "episodic"   : an event or observation from processing
    - "factual"    : a derived fact worth persisting across queries
    - "procedural" : a reusable strategy or pattern

  consolidate() -> str
    Compress the current working context into a single distilled summary.
    Writes the summary to memory (type="consolidation") and returns it.
    Call this when your retrieved context is large or redundant.

  answer: dict  ({"content": str, "ready": bool})
    Set answer["content"] to a SHORT direct answer string (under 15 words) and
    answer["ready"] = True. Extract ONLY the key fact from retrieved items.
    IMPORTANT: answer["content"] must be a plain string, NOT a list.
    Example: answer["content"] = "May 8, 2023"  # correct
    Example: answer["content"] = items[0]  # WRONG if items[0] is a long paragraph
    Extract the specific answer: dates, names, activities — not retrieved paragraphs.

Protocol:
1. Call memory_read() to retrieve relevant context
2. Optionally call memory_write() to persist derived facts
3. Optionally call consolidate() if context is large
4. Set answer["content"] (SHORT answer) and answer["ready"] = True

Write Python code only. No markdown, no explanations — just executable Python.
""").strip()


class RLMSystemRepl(RLMSystem):
    """RLM system extended with REPL-based memory_write and consolidate.

    Inherits all storage (LanceDB + DuckDB) and retrieval from RLMSystem.
    Adds a REPL query loop where the LLM can call memory_write() and
    consolidate() during query processing.

    Usage stats are tracked per query to measure write/consolidate adoption
    by pretrained models (expected: near-zero without RL training).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._write_count = 0
        self._consolidate_count = 0
        self._read_count = 0
        self._repl_iterations: list[int] = []
        self._writes_per_query: list[int] = []
        self._consolidations_per_query: list[int] = []
        self._reads_per_query: list[int] = []

    @property
    def name(self) -> str:
        return "rlm_repl"

    # ------------------------------------------------------------------
    # REPL query entrypoint
    # ------------------------------------------------------------------

    def query(self, question: str, budget: int | None = None) -> QueryResult:
        t0 = time.perf_counter()
        try:
            result = self._query_repl(question)
        except Exception as exc:
            print(f"[RLMRepl] Query failed: {exc}", flush=True)
            traceback.print_exc()
            result = QueryResult(
                answer="",
                total_latency_ms=0,
                context_tokens=0,
                details={"method": "error", "error": str(exc)},
            )
        result.total_latency_ms = (time.perf_counter() - t0) * 1000
        return result

    def _query_repl(self, question: str) -> QueryResult:
        """Run the RLM REPL loop with memory_write and consolidate in scope."""
        # Mutable counters in a dict so inner closures can mutate them
        # without nonlocal (which breaks when closures are rebuilt per iter).
        _counts: dict[str, int] = {"write": 0, "consolidate": 0, "read": 0}
        retrieved_context: list[str] = []
        iterations = 0
        context_tokens = 0

        import duckdb as _duckdb
        duck_rw = _duckdb.connect(self._duck_path)

        try:
            answer: dict[str, Any] = {"content": "", "ready": False}
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}"},
            ]

            # ---- REPL loop -----------------------------------------------
            for iteration in range(self._max_iterations):
                iterations += 1
                code = self._chat(messages)
                code = _strip_code_fences(code)

                # If SFT_TRACE_FILE env var is set, log the (messages, code)
                # pair for later distillation training.
                import os as _os
                _trace_file = _os.environ.get("SFT_TRACE_FILE")
                if _trace_file:
                    try:
                        import json as _json
                        with open(_trace_file, "a") as _tf:
                            _tf.write(_json.dumps({
                                "messages": messages,
                                "code": code,
                                "iteration": iteration,
                            }) + "\n")
                    except Exception:
                        pass

                # Per-iteration cancellation event — never reused or cleared.
                # The closure captures THIS event at definition time. If this
                # iteration times out and the thread later wakes, it sees its
                # own event (still set), not the next iteration's fresh event.
                _this_cancelled = threading.Event()

                def _make_memory_read(
                    ev: threading.Event = _this_cancelled,
                    duck: Any = duck_rw,
                    ctx: list = retrieved_context,
                    counts: dict = _counts,
                ) -> Any:
                    def memory_read(query: str, k: int = 20) -> list[str]:
                        if ev.is_set():
                            return []
                        results = self._retrieve(query, duck)
                        # Post-blocking check: timeout may have fired during _retrieve
                        if ev.is_set():
                            return []
                        seen: set[str] = set()
                        unique: list[str] = []
                        for r in results:
                            key = r.strip()[:200]
                            if key not in seen:
                                seen.add(key)
                                unique.append(r)
                                ctx.append(r)
                        counts["read"] += 1
                        return unique[:k]
                    return memory_read

                def _make_memory_write(
                    ev: threading.Event = _this_cancelled,
                    duck: Any = duck_rw,
                    counts: dict = _counts,
                ) -> Any:
                    def memory_write(
                        content: str, memory_type: str = "episodic"
                    ) -> None:
                        if ev.is_set():
                            return
                        valid = {"episodic", "factual", "procedural", "consolidation"}
                        if memory_type not in valid:
                            memory_type = "episodic"
                        vec = self._embedder.encode(content).tolist()
                        # Post-blocking check: timeout may have fired during encode
                        if ev.is_set():
                            return
                        ok = False
                        try:
                            self._lance_table.add([{
                                "text": content, "vector": vec,
                                "timestamp": "", "session_id": "",
                                "speaker": f"agent:{memory_type}",
                                "item_type": f"agent_write_{memory_type}",
                            }])
                            ok = True
                        except Exception as e:
                            print(f"[RLMRepl] lance write failed: {e}", flush=True)
                        # Post-lance check before touching DuckDB
                        if ev.is_set():
                            return
                        try:
                            duck.execute(
                                "INSERT INTO turns VALUES (?, ?, ?, ?, ?, ?)",
                                [-1, content, "agent", f"agent:{memory_type}", "", ""],
                            )
                            ok = True
                        except Exception:
                            pass
                        # Post-DuckDB check before incrementing counter
                        if ev.is_set():
                            return
                        if ok:
                            counts["write"] += 1
                    return memory_write

                def _make_consolidate(
                    ev: threading.Event = _this_cancelled,
                    ctx: list = retrieved_context,
                    counts: dict = _counts,
                    mw_fn: Any = None,  # set below after memory_write is built
                ) -> Any:
                    def consolidate() -> str:
                        if ev.is_set() or not ctx:
                            return ""
                        unique_ctx = list(dict.fromkeys(r.strip() for r in ctx))[:30]
                        text = "\n---\n".join(unique_ctx)
                        if len(text) > 8000:
                            text = text[:8000] + "\n[truncated]"
                        summary = self._chat([
                            {
                                "role": "system",
                                "content": (
                                    "Compress the following context into a single "
                                    "dense paragraph that retains all named facts "
                                    "and key details. No headings. Under 200 words."
                                ),
                            },
                            {"role": "user", "content": text},
                        ])
                        # Post-LLM-call check: timeout may have fired during _chat
                        if ev.is_set():
                            return ""
                        if mw_fn is not None:
                            mw_fn(summary, "consolidation")
                        # Post-mw_fn check before incrementing counter
                        if ev.is_set():
                            return ""
                        counts["consolidate"] += 1
                        return summary
                    return consolidate

                _mr = _make_memory_read()
                _mw = _make_memory_write()
                _co = _make_consolidate(mw_fn=_mw)

                iter_namespace: dict[str, Any] = {
                    "__builtins__": __builtins__,
                    "memory_read": _mr,
                    "memory_write": _mw,
                    "consolidate": _co,
                    "answer": answer,
                }

                exec_error: str | None = None
                _exc_box: list[str | None] = [None]

                # exec runs in a daemon thread with a timeout.
                # stdout/stderr are NOT redirected — redirect_stdout is
                # process-global and not safe to use from non-main threads
                # (two concurrent contexts would stomp each other). Output
                # simply goes to the real stdout; only errors are captured.
                def _run_exec(
                    code: str = code,
                    ns: dict = iter_namespace,
                    exc_box: list = _exc_box,
                ) -> None:
                    try:
                        exec(code, ns)  # noqa: S102
                    except Exception as exc:
                        exc_box[0] = f"{type(exc).__name__}: {exc}"

                _t = threading.Thread(target=_run_exec, daemon=True)
                _t.start()
                _t.join(timeout=_EXEC_TIMEOUT_S)
                if _t.is_alive():
                    _this_cancelled.set()  # guards THIS iter's namespace fns
                    exec_error = f"TimeoutError: exec exceeded {_EXEC_TIMEOUT_S}s"
                elif _exc_box[0]:
                    exec_error = _exc_box[0]

                if answer["ready"]:
                    break

                # Feed execution result back to model
                feedback_parts = []
                if exec_error:
                    feedback_parts.append(f"error: {exec_error}")
                if not feedback_parts:
                    feedback_parts.append("(no output — answer not yet set)")

                messages.append({"role": "assistant", "content": code})
                messages.append({
                    "role": "user",
                    "content": "\n".join(feedback_parts)
                    + "\n\nContinue. Set answer['ready']=True when done.",
                })

            # ---- Fallback if REPL loop never set answer["ready"] ---------
            final_answer = answer.get("content", "") or ""
            if not final_answer:
                # Graceful degradation: run the static retrieval path.
                # Reuse the already-open duck_rw connection (read_only would
                # conflict with the open read-write connection on the same file).
                try:
                    ctx_parts = self._retrieve(question, duck_rw)
                except Exception:
                    ctx_parts = []
                seen: set[str] = set()
                unique_parts: list[str] = []
                for p in ctx_parts:
                    key = p.strip()[:200]
                    if key not in seen:
                        seen.add(key)
                        unique_parts.append(p)
                context = "\n---\n".join(unique_parts)[:15000]
                context_tokens = len(context.split())
                final_answer = self._chat([
                    {
                        "role": "system",
                        "content": (
                            "Answer in ONE short sentence (under 15 words). "
                            "Just the direct answer, no explanation."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
                    },
                ])

        finally:
            duck_rw.close()

        context_tokens = context_tokens or len(" ".join(retrieved_context).split())
        write_count = _counts["write"]
        consolidate_count = _counts["consolidate"]
        read_count = _counts["read"]

        # Track per-query stats
        self._write_count += write_count
        self._consolidate_count += consolidate_count
        self._read_count += read_count
        self._repl_iterations.append(iterations)
        self._writes_per_query.append(write_count)
        self._consolidations_per_query.append(consolidate_count)
        self._reads_per_query.append(read_count)

        return QueryResult(
            answer=final_answer,
            total_latency_ms=0,
            context_tokens=context_tokens,
            details={
                "method": "rlm_repl",
                "iterations": iterations,
                "reads": read_count,
                "writes": write_count,
                "consolidations": consolidate_count,
                "answer_ready": answer.get("ready", False),
            },
        )

    # ------------------------------------------------------------------
    # Usage stats
    # ------------------------------------------------------------------

    def usage_stats(self) -> dict[str, Any]:
        """Return aggregate write/consolidate adoption stats.

        The research plan hypothesis: pretrained models will near-zero
        call memory_write/consolidate without RL training.
        """
        n = len(self._writes_per_query)
        if n == 0:
            return {"queries": 0}
        queries_with_reads = sum(1 for r in self._reads_per_query if r > 0)
        queries_with_writes = sum(1 for w in self._writes_per_query if w > 0)
        queries_with_consolidate = sum(1 for c in self._consolidations_per_query if c > 0)
        return {
            "queries": n,
            "total_reads": self._read_count,
            "total_writes": self._write_count,
            "total_consolidations": self._consolidate_count,
            "queries_with_reads": queries_with_reads,
            "queries_with_writes": queries_with_writes,
            "queries_with_consolidations": queries_with_consolidate,
            "read_adoption_rate": queries_with_reads / n,
            "write_adoption_rate": queries_with_writes / n,
            "consolidate_adoption_rate": queries_with_consolidate / n,
            "mean_iterations": sum(self._repl_iterations) / n,
        }


def _strip_code_fences(text: str) -> str:
    """Remove ```python ... ``` or ``` ... ``` fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first line (``` or ```python) and last ``` line
        start = 1
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end = i
                break
        text = "\n".join(lines[start:end])
    return text
