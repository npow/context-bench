"""OMW writers: produce memory_write actions per session.

All writers have signature:
    writer(session_text, prior_memory, session_idx, ex) → list[str | dict]
"""
from __future__ import annotations
import json
import re

from .readers import _client, _claude_call


# ---------------------------------------------------------------------------
# B0 strong_rag: no writes — RAG at QA time
# ---------------------------------------------------------------------------
def writer_no_op(session_text, prior_memory, session_idx, ex):
    return []


# ---------------------------------------------------------------------------
# B1 write_every: every utterance becomes a fact
# ---------------------------------------------------------------------------
def writer_write_every(session_text, prior_memory, session_idx, ex):
    return [line.strip() for line in session_text.split("\n") if line.strip()]


# ---------------------------------------------------------------------------
# B2 session_summary: one summary per session
# ---------------------------------------------------------------------------
def make_writer_session_summary(client, model_id: str):
    def w(session_text, prior_memory, session_idx, ex):
        prompt = (
            "Summarize this conversation session into 1-3 short factual sentences "
            "capturing key information (entities, facts, decisions). Be concise.\n\n"
            f"SESSION:\n{session_text[:6000]}\n\nSummary:"
        )
        out = _claude_call(client, prompt, model_id, max_tokens=200)
        return [{"content": out, "type": "summary"}] if out.strip() else []
    return w


# ---------------------------------------------------------------------------
# B3 entity_profile: extract entities and write per-entity profiles
# ---------------------------------------------------------------------------
def make_writer_entity_profile(client, model_id: str):
    def w(session_text, prior_memory, session_idx, ex):
        prompt = (
            "Identify the entities (people, places, things) mentioned in this session "
            "and extract a one-line factual profile for each. Output one fact per line, "
            "in 'Entity: fact' format. Maximum 6 lines.\n\n"
            f"SESSION:\n{session_text[:6000]}\n\nProfiles:"
        )
        out = _claude_call(client, prompt, model_id, max_tokens=300)
        lines = [l.strip() for l in out.split("\n") if l.strip() and ":" in l]
        return [{"content": l, "type": "profile"} for l in lines[:6]]
    return w


# ---------------------------------------------------------------------------
# B4 heuristic_salient: rule-based salience (NER-lite + numeric/temporal keywords)
# ---------------------------------------------------------------------------
SALIENT_PATTERNS = [
    re.compile(r"\$?\d+[,.]?\d*"),                # numbers / prices
    re.compile(r"\b\d{1,2}[:/]\d{1,2}([:/]\d{2,4})?\b"),  # times/dates
    re.compile(r"\b(yesterday|today|tomorrow|last week|next week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I),
    re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b"),  # proper names (heuristic)
]


def writer_heuristic_salient(session_text, prior_memory, session_idx, ex):
    """Pick utterances containing salient tokens. Cap at 4 per session."""
    lines = [l.strip() for l in session_text.split("\n") if l.strip()]
    salient = []
    for line in lines:
        if any(p.search(line) for p in SALIENT_PATTERNS):
            salient.append(line)
        if len(salient) >= 4:
            break
    return salient


# ---------------------------------------------------------------------------
# C1 q_aware_teacher: ORACLE — sees Q during ingest.
# Used ONLY as upper bound. Student is NOT trained on these trajectories.
# ---------------------------------------------------------------------------
def make_writer_q_aware_teacher(client, model_id: str):
    def w(session_text, prior_memory, session_idx, ex):
        # Concatenate all queries (oracle leakage)
        all_questions = "; ".join(q.question for q in ex.queries)
        prompt = (
            "You are building memory from a conversation. The future questions you "
            "must support are listed below. Extract 0-5 SHORT factual statements from "
            "THIS SESSION ONLY that contain information needed to answer any of the "
            "future questions. Output one fact per line. If nothing relevant, output NOTHING.\n\n"
            f"FUTURE QUESTIONS:\n{all_questions}\n\n"
            f"SESSION:\n{session_text[:6000]}\n\nFacts:"
        )
        out = _claude_call(client, prompt, model_id, max_tokens=400)
        lines = [l.strip().lstrip("-*•").strip() for l in out.split("\n") if l.strip()]
        return [l for l in lines if len(l) > 10][:5]
    return w


# ---------------------------------------------------------------------------
# C2 q_blind_teacher: DEPLOYABLE — does NOT see Q. Decides what's worth writing.
# This is the trajectory source for D (SFT writer).
# ---------------------------------------------------------------------------
def make_writer_q_blind_teacher(client, model_id: str):
    """Q-blind memory writer. Prompt broadened (Codex fix #2): captures
    durable user-state, episodic events, temporal markers, transactional
    facts, and updates/contradictions — not just 'stable personal facts'.
    """
    def w(session_text, prior_memory, session_idx, ex):
        prior_summary = ""
        if prior_memory.entries:
            recent = prior_memory.entries[-5:]
            prior_summary = "Prior memory (last 5):\n" + "\n".join(f"- {e.content}" for e in recent) + "\n\n"
        prompt = (
            "You are maintaining long-term memory across a multi-session conversation. "
            "Extract 0-6 SHORT facts from THIS SESSION that may be useful to recall later. "
            "Include any of:\n"
            "  - Durable user state (preferences, possessions, relationships, traits)\n"
            "  - Episodic events with time/place (what happened when/where)\n"
            "  - Transactional facts (purchases, prices, quantities, schedules)\n"
            "  - Temporal markers (dates, durations, deadlines)\n"
            "  - Updates/changes to previously-mentioned facts\n"
            "  - Notable third-party facts (entities, places mentioned)\n"
            "Output one fact per line. Include concrete details (numbers, names, dates) "
            "rather than vague abstractions. Avoid facts already in prior memory unless "
            "they're being updated/contradicted. If the session contains nothing worth "
            "remembering, output NOTHING.\n\n"
            f"{prior_summary}SESSION:\n{session_text[:6000]}\n\nFacts:"
        )
        out = _claude_call(client, prompt, model_id, max_tokens=500)
        lines = [l.strip().lstrip("-*•").strip() for l in out.split("\n") if l.strip()]
        return [l for l in lines if len(l) > 10 and "NOTHING" not in l.upper()][:6]
    return w


# Per-question C1 oracle (Codex fix #5) — better upper bound: separate trajectory per Q.
def make_writer_q_aware_teacher_per_q(client, model_id: str):
    """For evaluation only — returns a CALLABLE that takes (query) and produces
    a per-Q memory trajectory. Used in eval loop, not via standard writer interface.
    """
    def per_q_writer(query):
        def w(session_text, prior_memory, session_idx, ex):
            prompt = (
                "You are building memory for a single future question. Extract 0-5 SHORT "
                "factual statements from THIS SESSION that contain information needed to "
                "answer that question. If nothing in this session is relevant, output NOTHING.\n\n"
                f"FUTURE QUESTION: {query.question}\n\n"
                f"SESSION:\n{session_text[:6000]}\n\nFacts:"
            )
            out = _claude_call(client, prompt, model_id, max_tokens=400)
            lines = [l.strip().lstrip("-*•").strip() for l in out.split("\n") if l.strip()]
            return [l for l in lines if len(l) > 10 and "NOTHING" not in l.upper()][:5]
        return w
    return per_q_writer


# ---------------------------------------------------------------------------
# D sft_writer: Qwen-3B fine-tuned on C2 trajectories.
# Loaded from local adapter; see sft_writer_train.py.
# ---------------------------------------------------------------------------
def make_writer_sft_student(adapter_path: str):
    """Lazy-load the SFT student model. Raises if adapter not available."""
    raise NotImplementedError("SFT student not yet trained; will be plugged in after sft_writer_train.py runs.")


# ---------------------------------------------------------------------------
# sanity_no_writer: spontaneous pretrained behavior (most models will not write).
# ---------------------------------------------------------------------------
def make_writer_spontaneous(client, model_id: str):
    """Ask the model to optionally write — but no instruction to do so."""
    def w(session_text, prior_memory, session_idx, ex):
        prompt = (
            "You are a helpful assistant. Below is a conversation session. "
            "Optionally, you may save important information to memory by writing "
            "lines starting with 'MEMORY:'. If nothing seems important, output nothing.\n\n"
            f"SESSION:\n{session_text[:6000]}\n\nResponse:"
        )
        out = _claude_call(client, prompt, model_id, max_tokens=300)
        # Extract lines starting with MEMORY:
        lines = [l.strip() for l in out.split("\n") if l.strip().upper().startswith("MEMORY:")]
        return [l.split(":", 1)[1].strip() for l in lines if ":" in l][:5]
    return w


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def get_writer(name: str, args):
    client = _client()
    model = args.writer_model
    if name == "B0_strong_rag":
        return writer_no_op
    if name == "B1_write_every":
        return writer_write_every
    if name == "B2_session_summary":
        return make_writer_session_summary(client, model)
    if name == "B3_entity_profile":
        return make_writer_entity_profile(client, model)
    if name == "B4_heuristic":
        return writer_heuristic_salient
    if name == "C1_q_aware_teacher":
        return make_writer_q_aware_teacher(client, model)
    if name == "C2_q_blind_teacher":
        return make_writer_q_blind_teacher(client, model)
    if name == "D_sft_writer":
        return make_writer_sft_student(args.student_adapter)
    if name == "sanity_no_writer":
        return make_writer_spontaneous(client, model)
    raise ValueError(f"unknown writer condition: {name}")
