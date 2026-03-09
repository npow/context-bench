# mutator.py
# ===================================================================
# LLM agent that proposes mutations to context_pipeline.py.
#
# The PipelineMutator reads the current pipeline code and the score
# history, then asks an LLM for ONE targeted improvement.  It returns
# the complete modified Python file as a string — ready to be written
# to disk and dynamically imported by the autoresearch loop.
#
# Uses urllib only (no third-party HTTP libraries).
# ===================================================================

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


# ---------------------------------------------------------------------------
# Strategy parameter reference — shown to the agent in its system prompt
# so it knows exactly what levers it can pull without reading the source.
# ---------------------------------------------------------------------------

_STRATEGY_REFERENCE = """
STRATEGY dict keys and their allowed values
-------------------------------------------
chunk_by          : "turn" | "session"
                    "turn"    = one retrievable chunk per conversation turn
                    "session" = group session_size consecutive turns per chunk
session_size      : int >= 1  (only meaningful when chunk_by="session")
retrieval_k       : int >= 1  (number of chunks retrieved per query)
retrieval_method  : "bm25" | "embedding" | "hybrid"
                    "bm25"      = keyword overlap, zero API cost
                    "embedding" = semantic vector similarity (requires relay)
                    "hybrid"    = 0.5*bm25 + 0.5*embedding blend
context_order     : "relevant_first" | "chronological" | "reverse_chron"
                    "relevant_first" = highest-scored chunk presented first
                    "chronological"  = earliest turn in history first
                    "reverse_chron"  = most-recent turn first
compression       : "none" | "summarize" | "key_sentences"
                    "none"          = send raw chunk text to the LLM
                    "summarize"     = LLM condenses each chunk to 1-2 sentences
                    "key_sentences" = LLM extracts 1-3 key sentences per chunk
query_expansion   : True | False
                    Rewrite the question via LLM before retrieval
iterative         : True | False
                    Two-pass retrieval: retrieve → LLM refines query → retrieve again
max_context_tokens: int >= 100
                    Hard word-count cap on the context block sent to the final LLM call

The optimisation target is token_efficiency = F1 / (mean_input_words / 1000).
Higher is better: the pipeline should answer accurately while consuming FEWER tokens.
"""


class PipelineMutator:
    """Proposes targeted mutations to context_pipeline.py using an LLM.

    The mutator keeps no state between calls — every call to
    ``propose_mutation`` is a fresh, independent LLM conversation.

    Args:
        relay_url: Base URL of the OpenAI-compatible relay (no trailing slash).
        model:     Chat model name.  The prompt is written for a capable model
                   (e.g. gpt-4o, gpt-4o-mini, claude-3-5-sonnet).
        api_key:   Bearer token.  Falls back to the OPENAI_API_KEY env var.
        timeout:   HTTP request timeout in seconds.
    """

    def __init__(
        self,
        relay_url: str,
        model: str = "claude-sonnet-4-6",
        api_key: str = "",
        timeout: float = 120.0,
    ) -> None:
        self._base_url = relay_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propose_mutation(
        self,
        current_code: str,
        score_history: list[dict],
    ) -> str:
        """Ask the LLM for one targeted improvement to context_pipeline.py.

        Args:
            current_code:  Full text of the current context_pipeline.py.
            score_history: List of dicts describing past iterations.
                           Each dict has the shape:
                           {
                             "iteration": int,
                             "score": float,          # token_efficiency
                             "mutation": str,          # one-line description
                             "accepted": bool,
                           }
                           The list should be in chronological order; the
                           mutator looks at the last 10 entries.

        Returns:
            A string containing the complete modified Python file.  The
            caller is responsible for validating that it is importable
            before accepting it.
        """
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(current_code, score_history)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self._chat(messages)
        return self._extract_code(response)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        return """\
You are a research engineer with one goal: surpass state-of-the-art performance \
on the LoCoMo long-conversation QA benchmark.

CURRENT SOTA BEST: 90% F1 (Hindsight with Gemini-3 Pro + TEMPR, Dec 2025)
YOUR TARGET: exceed 0.90 F1. Current pipeline is far below — every iteration must make \
a meaningful architectural advance, not a minor tweak.

You will be given the full source of a pipeline file and a history of past iterations. \
Propose ONE significant improvement that moves F1 substantially higher.

## What SOTA systems do that naive RAG doesn't:

1. **Coreference resolution**: Resolve "he", "she", "my friend", "they" to canonical \
entity names during ingest. LoCoMo questions ask about specific named people — \
you must find the right person's facts even when the turn uses pronouns.

2. **Temporal fact versioning**: Facts evolve. "Nate used to play CS:GO, now plays \
Valorant." Store facts with turn/session timestamps. When answering "what does X do \
NOW", rank recent facts over old ones. When answering "what did X used to do", rank \
old facts.

3. **Typed relation triples**: Store facts as structured (entity, relation, value, \
turn_idx) not just strings. Enables precise lookup: "find Caroline's hobbies" or \
"find where Nate works". BM25 and even embedding search over raw text misses these.

4. **Multi-hop reasoning**: LoCoMo questions often chain facts. "Who introduced X to Y?" \
requires finding who knows both. Implement: extract entities from the question, retrieve \
facts about each entity, reason across them to form the answer.

5. **Query decomposition**: Break questions like "What sport does Nate play and when \
did he start?" into sub-questions, answer each, synthesise. Dramatically improves \
multi-hop and temporal questions.

6. **Cross-session entity tracking**: Conversations happen across many sessions. \
Track the same entity's state across all sessions. An entity profile aggregates all \
known facts about a person across the whole conversation history.

7. **Confidence-aware answering**: If retrieved facts don't clearly answer the question, \
say "based on available information: [best guess]" rather than hallucinating. \
LoCoMo F1 rewards partial credit.

## Your constraints:
- **Optimise pure F1. Token cost is irrelevant.**
- You have FULL FREEDOM: rewrite any method, add any data structures, make as many \
LLM calls during ingest or query as needed. The relay supports "haiku" (fast/cheap) \
and "sonnet" (powerful) — use haiku for extraction/classification, sonnet for reasoning.
- Make ARCHITECTURAL changes. Do not tweak STRATEGY dict parameters — that space \
is exhausted. Implement new capabilities.
- Avoid changes already marked rejected in the history.
- Return ONLY the complete modified Python file. No explanation, no markdown fences. \
First character must be "#".
"""

    def _build_user_prompt(
        self,
        current_code: str,
        score_history: list[dict],
    ) -> str:
        # Show only the last 10 iterations to keep the prompt focused.
        recent = score_history[-10:] if len(score_history) > 10 else score_history

        # Separate accepted from rejected so the agent can see both clearly.
        accepted = [h for h in recent if h.get("accepted", False)]
        rejected = [h for h in recent if not h.get("accepted", False)]

        history_text = self._format_history(recent)
        rejected_summary = self._format_rejected_summary(rejected)

        # Use the tracked best_score (from accepted iterations) rather than
        # max(h["score"]) which includes rejected candidates and is inflated
        # by sampling noise.
        best_score = max(
            (h.get("best_score", h["score"]) for h in score_history), default=0.0
        )
        current_score = score_history[-1].get("best_score", score_history[-1]["score"]) if score_history else 0.0

        prompt_parts = [
            "## Current context_pipeline.py\n",
            current_code,
            "\n\n## Score history (last 10 iterations)\n",
            history_text,
        ]

        if rejected_summary:
            prompt_parts += [
                "\n\n## Changes that were REJECTED (do NOT repeat these):\n",
                rejected_summary,
            ]

        prompt_parts += [
            f"\n\n## Summary\n",
            f"- Current F1    : {current_score:.4f}\n",
            f"- Best F1       : {best_score:.4f}\n",
            f"- SOTA target   : 0.90 (beat this)\n",
            f"- Gap to SOTA   : {max(0.90 - best_score, 0):.4f}\n",
            f"- Total iters   : {len(score_history)}\n",
            "\nPropose ONE architectural improvement that meaningfully closes the gap to SOTA. "
            "Return ONLY the complete modified Python file.\n",
        ]

        return "".join(prompt_parts)

    def _format_history(self, history: list[dict]) -> str:
        if not history:
            return "  (no history yet — this is the first iteration)"
        lines = []
        for h in history:
            status = "ACCEPTED" if h.get("accepted", False) else "rejected"
            score = h.get("score", 0.0)
            delta = h.get("delta", 0.0)
            mutation = h.get("mutation", "(unknown)")
            iteration = h.get("iteration", "?")
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"  iter {iteration:>3}: score={score:.4f} ({sign}{delta:.4f}) "
                f"[{status}]  — {mutation}"
            )
        return "\n".join(lines)

    def _format_rejected_summary(self, rejected: list[dict]) -> str:
        if not rejected:
            return ""
        lines = [f"  - {h.get('mutation', '(unknown)')}" for h in rejected]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Code extraction — strip markdown fences if the model adds them
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_code(response: str) -> str:
        """Strip any markdown code fences or prose preamble the model may have added."""
        text = response.strip()

        # Handle ```python ... ``` or ``` ... ``` blocks.
        if text.startswith("```"):
            lines = text.splitlines()
            start = 1
            end = len(lines)
            if lines[-1].strip() == "```":
                end -= 1
            text = "\n".join(lines[start:end]).strip()

        # If the text still doesn't start with '#', the model added a prose
        # preamble (e.g. "— Here's my proposal:\n# entity_pipeline.py ...").
        # Find the first line that looks like the start of a Python file.
        if not text.startswith("#"):
            lines = text.splitlines()
            for i, line in enumerate(lines):
                stripped = line.strip()
                # Python file starts with a comment or a ``` fence containing code
                if stripped.startswith("#") or stripped.startswith("```"):
                    if stripped.startswith("```"):
                        # nested fence — recurse one level
                        inner = "\n".join(lines[i:])
                        return PipelineMutator._extract_code(inner)
                    text = "\n".join(lines[i:]).strip()
                    break

        # Sanitize unicode characters that are syntactically invalid in Python
        # but commonly output by LLMs (e.g. → instead of ->, — in comments).
        text = text.replace("\u2192", "->")   # → arrow  (type hints)
        text = text.replace("\u2014", "--")   # — em dash (comments)
        text = text.replace("\u2013", "-")    # – en dash
        text = text.replace("\u201c", '"')    # " left double quote
        text = text.replace("\u201d", '"')    # " right double quote
        text = text.replace("\u2018", "'")    # ' left single quote
        text = text.replace("\u2019", "'")    # ' right single quote

        return text

    # ------------------------------------------------------------------
    # HTTP — uses urllib only, no third-party dependencies
    # ------------------------------------------------------------------

    def _chat(self, messages: list[dict[str, Any]]) -> str:
        """POST to /v1/chat/completions with retry on transient errors.

        Uses temperature=0.7 to encourage diverse exploration of the
        strategy space across iterations.
        """
        url = f"{self._base_url}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.7,
        }
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        for attempt in range(3):
            req = urllib.request.Request(
                url, data=body, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    data = json.loads(resp.read().decode())
                    return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(
                    f"Mutator HTTP {exc.code}: {exc.reason}"
                ) from exc
            except urllib.error.URLError as exc:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(
                    f"Mutator connection error: {exc.reason}"
                ) from exc

        raise RuntimeError("Mutator chat request failed after 3 retries")
