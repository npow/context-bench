"""OMW-Bench framework: MemoryStore, ingest+QA loop, scoring.

Memory_store is a simple list of (timestamp, type, content) tuples.
At QA time, memory_read(query) returns top-K by hybrid BM25+lexical match.
"""
from __future__ import annotations
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class MemoryEntry:
    content: str
    type: str = "fact"      # "fact" | "summary" | "profile" | "raw"
    session_idx: int = -1   # which session it was written from
    timestamp: str | None = None


@dataclass
class MemoryStore:
    entries: list[MemoryEntry] = field(default_factory=list)

    def write(self, content: str, type: str = "fact", session_idx: int = -1, timestamp: str | None = None):
        if content and content.strip():
            self.entries.append(MemoryEntry(
                content=content.strip(), type=type, session_idx=session_idx, timestamp=timestamp,
            ))

    @property
    def total_chars(self) -> int:
        return sum(len(e.content) for e in self.entries)

    @property
    def write_count(self) -> int:
        return len(self.entries)

    def read(self, query: str, k: int = 8) -> list[MemoryEntry]:
        """BM25 over entry content + session_idx + timestamp (Codex fix #4).
        Temporal/session metadata is part of the searchable index so questions
        like 'what did they say in session 3?' or 'last week's plan?' can hit."""
        if not self.entries:
            return []
        if not query:
            return self.entries[:k]
        q_terms = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 2]
        if not q_terms:
            return self.entries[:k]
        # Include session_idx + timestamp in searchable text
        def searchable(e: MemoryEntry) -> str:
            parts = [e.content]
            if e.session_idx >= 0: parts.append(f"session {e.session_idx}")
            if e.timestamp: parts.append(e.timestamp)
            return " ".join(parts)
        docs = [[w.lower() for w in re.findall(r"\w+", searchable(e))] for e in self.entries]
        N = len(docs)
        avgdl = sum(len(d) for d in docs) / max(1, N)
        df = {t: sum(1 for d in docs if t in d) for t in q_terms}
        k1, b = 1.5, 0.75
        scored = []
        for i, d in enumerate(docs):
            dl = len(d); s = 0.0
            for term in q_terms:
                if df[term] == 0: continue
                tf = d.count(term)
                idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
                s += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(1, avgdl)))
            scored.append((s, i))
        scored.sort(key=lambda x: -x[0])
        return [self.entries[i] for s, i in scored[:k] if s > 0]


def group_by_session(items) -> tuple[list[str], dict[str, list]]:
    """Return (session_order, sessions_dict). Each item must have session_id attr."""
    sessions = defaultdict(list)
    order = []
    for it in items:
        sid = getattr(it, "session_id", None) or "default"
        if sid not in sessions:
            order.append(sid)
        sessions[sid].append(it)
    return order, sessions


@dataclass
class OMWBenchmark:
    memory_only: bool = True
    last_n_window: int = 0
    max_writes_per_session: int = 10
    seed: int = 42
    # B0_strong_rag mode: skip ingest writes; QA retrieves from raw log
    rag_mode: bool = False

    def ingest(self, writer: Callable, ex: Any) -> MemoryStore:
        """Run the writer over sessions[1..K] in temporal order.

        writer(session_text, prior_memory, session_idx) → list of writes (each a str).
        """
        order, sessions = group_by_session(ex.items)
        memory = MemoryStore()
        for sidx, sid in enumerate(order):
            sitems = sessions[sid]
            if len(sitems) < 1:
                continue
            session_text = "\n".join(
                f"{getattr(i, 'speaker', '') or ''}: {i.content}" for i in sitems
            )
            try:
                writes = writer(session_text=session_text, prior_memory=memory, session_idx=sidx, ex=ex)
            except Exception as e:
                writes = []
            # Truncate to max_writes_per_session
            writes = (writes or [])[: self.max_writes_per_session]
            for w in writes:
                if isinstance(w, dict):
                    memory.write(w.get("content", ""), type=w.get("type", "fact"), session_idx=sidx)
                else:
                    memory.write(str(w), type="fact", session_idx=sidx)
        return memory

    def answer(self, reader: Callable, memory: MemoryStore, ex: Any, query) -> str:
        """Run the QA reader. Memory + optional last-N session window.
        Renders session_idx + timestamp on each memory entry (Codex fix #4).

        B0 strong RAG mode: skip memory; retrieve top-K windows from raw log.
        """
        if self.rag_mode:
            # Build raw RAG over the FULL conversation log (Codex fix #1)
            memory_text = self._rag_retrieve(ex, query.question, budget_chars=8000)
        else:
            retrieved = memory.read(query.question, k=8)
            def render(e: MemoryEntry) -> str:
                tags = [e.type]
                if e.session_idx >= 0: tags.append(f"session={e.session_idx}")
                if e.timestamp: tags.append(f"t={e.timestamp}")
                return f"- ({', '.join(tags)}) {e.content}"
            memory_text = "\n".join(render(e) for e in retrieved)
        # Optional realistic-setting last-N raw window
        window_text = ""
        if self.last_n_window > 0:
            order, sessions = group_by_session(ex.items)
            recent_ids = order[-self.last_n_window:]
            window_text = "\n\n".join(
                f"[Recent session {sid}]\n"
                + "\n".join(f"{getattr(i, 'speaker', '') or ''}: {i.content}" for i in sessions[sid])
                for sid in recent_ids
            )
        return reader(question=query.question, memory_text=memory_text, window_text=window_text)


    def _rag_retrieve(self, ex: Any, question: str, budget_chars: int = 8000, window: int = 600) -> str:
        """Strong RAG over raw conversation log. BM25 chunk-level + budget allocation.

        Used for B0_strong_rag baseline: no memory writes, hybrid retrieval at
        QA time over the raw log. (Codex fix #1: B0 was a no-op without this.)
        """
        order, sessions = group_by_session(ex.items)
        chunks = []  # (text, session_idx)
        for sidx, sid in enumerate(order):
            sitems = sessions[sid]
            text = "\n".join(f"{getattr(i, 'speaker', '') or ''}: {i.content}" for i in sitems)
            for start in range(0, len(text), window):
                chunks.append((text[start:start + window], sidx, sid))
        if not chunks:
            return ""
        q_terms = [w.lower() for w in re.findall(r"\w+", question) if len(w) > 2]
        if not q_terms:
            return "\n\n".join(f"[Session {sid}]\n{c}" for c, _, sid in chunks)[:budget_chars]

        import math as _math
        tokenized = [[w.lower() for w in re.findall(r"\w+", c)] for c, _, _ in chunks]
        N = len(chunks)
        avgdl = sum(len(t) for t in tokenized) / max(1, N)
        df = {t: sum(1 for tc in tokenized if t in tc) for t in q_terms}
        k1, b = 1.5, 0.75
        scored = []
        for (c, sidx, sid), tc in zip(chunks, tokenized):
            dl = len(tc); s = 0.0
            for term in q_terms:
                if df[term] == 0: continue
                tf = tc.count(term)
                idf = _math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
                s += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(1, avgdl)))
            scored.append((s, sidx, sid, c))
        scored.sort(key=lambda x: -x[0])
        out, used = [], 0
        for s, sidx, sid, c in scored:
            if used >= budget_chars: break
            if s <= 0 and out: break
            take = min(len(c), budget_chars - used)
            out.append(f"[Session {sid} (idx={sidx})]\n{c[:take]}")
            used += take
        return "\n\n---\n\n".join(out) if out else ""


def score_qa(reader_module, ans: str, query) -> int | None:
    """LLM-as-judge with CORRECT/WRONG/INCORRECT logic."""
    if not ans.strip():
        return 0
    return reader_module.judge(ans=ans, gold=query.answer, question=query.question)


def write_quality_metrics_judge(reader_module, memory: MemoryStore, ex, query) -> dict:
    """Judge-based write_recall (Codex fix #3): ask the judge whether the memory
    as a whole contains the information needed to answer Q with gold.

    Avoids token-substring pathology ("$50" vs "fifty dollars"; "5/1/2024" vs "May 1").
    """
    if not memory.entries:
        return {"write_recall": 0, "memory_used_for_judge_chars": 0}
    memory_text = "\n".join(f"- ({e.type}) {e.content}" for e in memory.entries)
    # Judge: does the memory contain the gold answer info?
    if hasattr(reader_module, "judge_recall"):
        recall = reader_module.judge_recall(memory_text=memory_text, gold=query.answer, question=query.question)
    else:
        # Fallback to substring heuristic if reader doesn't implement judge_recall
        gold_tokens = {w.lower() for w in re.findall(r"\w+", query.answer) if len(w) > 2}
        recall = 1 if any(any(t in e.content.lower() for t in gold_tokens) for e in memory.entries) else 0
    return {"write_recall": recall, "memory_used_for_judge_chars": len(memory_text)}


# Backward alias
def write_quality_metrics(memory: MemoryStore, ex, query, reader_module=None) -> dict:
    """Wrapper. If reader_module is provided, uses judge-based scoring."""
    if reader_module is not None:
        return write_quality_metrics_judge(reader_module, memory, ex, query)
    # Heuristic fallback (used only as last resort; kept for backward compat)
    if not memory.entries:
        return {"write_recall": 0, "write_precision": 0.0}
    gold_tokens = {w.lower() for w in re.findall(r"\w+", query.answer) if len(w) > 2}
    if not gold_tokens:
        return {"write_recall": None, "write_precision": None}
    contains = sum(1 for e in memory.entries if any(t in e.content.lower() for t in gold_tokens))
    return {
        "write_recall": 1 if contains > 0 else 0,
        "write_precision": contains / len(memory.entries),
    }


def run_writer(*args, **kwargs):
    """Legacy alias."""
    raise NotImplementedError


def run_qa(*args, **kwargs):
    raise NotImplementedError
