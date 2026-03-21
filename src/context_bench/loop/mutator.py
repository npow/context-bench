# mutator.py
# ===================================================================
# LLM agent that proposes mutations to context_pipeline.py.
#
# The PipelineMutator reads the current pipeline code and the score
# history, then asks an LLM for ONE targeted improvement.  It returns
# the complete modified Python file as a string -- ready to be written
# to disk and dynamically imported by the autoresearch loop.
#
# Uses urllib only (no third-party HTTP libraries).
# ===================================================================

from __future__ import annotations

import json
import re as _re_module
import time
import urllib.error
import urllib.request
from typing import Any


class PipelineMutator:
    """Proposes targeted mutations to context_pipeline.py using an LLM."""

    def __init__(
        self,
        relay_url: str,
        model: str = "sonnet",
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
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(current_code, score_history)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Exception | None = None
        for attempt in range(3):
            if attempt > 0:
                # Fresh single-turn request on retry
                retry_user = (
                    f"Your previous attempt had a syntax error: {last_error}\n\n"
                    "Fix this error and return the COMPLETE Python file. "
                    "ONLY output valid Python code -- no markdown, no explanation.\n\n"
                    f"Here is the code that had the error:\n\n{code}\n"
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": retry_user},
                ]
            response = self._chat(messages)
            code = self._extract_code(response)
            try:
                compile(code, "<proposed_pipeline>", "exec")
                return code
            except SyntaxError as e:
                last_error = e
                import sys
                print(f"  [mutator] attempt {attempt+1} syntax error: {e}", file=sys.stderr, flush=True)
                continue

        raise RuntimeError(f"Generated code has syntax errors after 3 attempts: {last_error}")

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        return """\
You are a research engineer optimising a long-conversation QA pipeline to beat SOTA.

## Benchmark: LongMemEval-S
- 500 examples, each a multi-turn user/assistant conversation (~500 turns avg).
- 6 question types: single-session-user (70), multi-session (133), \
single-session-preference (30), temporal-reasoning (133), knowledge-update (78), \
single-session-assistant (56).
- Turns are flat (role + content only) -- NO timestamps, NO session_id markers.
- SOTA: 94.87% (Mastra Observational Memory + gpt-5-mini), 86% (Emergence RAG).

## SOTA approach (Observational Memory by Mastra):
- During ingest, an LLM "Observer" compresses conversation into dense dated \
observations with 3-6x compression. Every turn is observed -- nothing dropped.
- Three-date temporal model: observation date, referenced date, relative date.
- At query time, ALL observations are packed into context -- no retrieval needed.
- Observations are formatted text with turn indices, dates, priority markers.
- A "Reflector" periodically restructures observations to remove redundancy.
- Result: 95.5% on temporal-reasoning, 96.2% on knowledge-update.

## Current pipeline approach:
Uses LLM (haiku) to compress turns into observations during ingest, then packs \
all observations + single sonnet call for answer. This is the right architecture.

## Key areas to improve:
1. **Observation quality**: The observer prompt determines everything. It must \
capture ALL dates (explicit and relative), ALL facts, ALL knowledge updates, \
with turn indices for temporal reasoning.
2. **Temporal anchoring**: Extract and resolve dates. "last month" near a turn \
mentioning "March 10" -> ~February. Store resolved dates in observations.
3. **Knowledge-update tracking**: When user says "I got a new job at X" after \
previously mentioning "I work at Y", the observation must note the change.
4. **Observation format**: Dense, structured, with priority markers and dates.
5. **Answer prompt engineering**: Question-type-specific hints (temporal math, \
knowledge updates, "not enough info" detection).
6. **Reflection/compression**: If observations are too long, compress further.

## Your constraints:
- **Optimise pure LLM-judge accuracy. Use as many LLM calls as needed.**
- You have FULL FREEDOM to make haiku calls during ingest for observation \
extraction. Batch turns into chunks of 30-80 and make one haiku call per chunk. \
At query time, use sonnet for the final answer (and optionally haiku for query \
analysis or evidence extraction).
- The pipeline class MUST be named ContextPipeline or aliased as such.
- The pipeline MUST have: __init__(relay_url, model, api_key, strategy, timeout), \
reset(), ingest(turns), query(question), last_context_tokens, name.
- Wrap ALL LLM calls in try/except with retries.
- Avoid changes already marked rejected in the history.
- CRITICAL OUTPUT FORMAT: Return ONLY the complete modified Python file. \
NO explanation, NO markdown fences, NO prose. First character must be "#".
"""

    def _build_user_prompt(
        self,
        current_code: str,
        score_history: list[dict],
    ) -> str:
        recent = score_history[-10:] if len(score_history) > 10 else score_history

        accepted = [h for h in recent if h.get("accepted", False)]
        rejected = [h for h in recent if not h.get("accepted", False)]

        history_text = self._format_history(recent)
        rejected_summary = self._format_rejected_summary(rejected)

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
            return "  (no history yet -- this is the first iteration)"
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
                f"[{status}]  -- {mutation}"
            )
        return "\n".join(lines)

    def _format_rejected_summary(self, rejected: list[dict]) -> str:
        if not rejected:
            return ""
        lines = [f"  - {h.get('mutation', '(unknown)')}" for h in rejected]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Code extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_code(response: str) -> str:
        """Strip markdown fences or prose preamble the model may have added."""
        text = response.strip()

        # Strategy 1: Find the LARGEST ```python ... ``` fenced block.
        fence_pattern = _re_module.compile(r"```(?:python|py)?\s*\n(.*?)```", _re_module.DOTALL)
        fences = fence_pattern.findall(text)
        if fences:
            text = max(fences, key=len).strip()
        elif text.startswith("```"):
            lines = text.splitlines()
            start = 1
            end = len(lines)
            if lines[-1].strip() == "```":
                end -= 1
            text = "\n".join(lines[start:end]).strip()

        # Strategy 2: Find "class ContextPipeline" and walk backwards.
        if "class " in text and "Pipeline" in text:
            lines = text.splitlines()
            class_line = None
            for i, line in enumerate(lines):
                if "class " in line and "Pipeline" in line:
                    class_line = i
                    break
            if class_line is not None:
                file_start = class_line
                for i in range(class_line - 1, -1, -1):
                    stripped = lines[i].strip()
                    if (stripped == "" or stripped.startswith("#") or
                        stripped.startswith("from ") or stripped.startswith("import ") or
                        stripped.startswith("@") or stripped.startswith('"""') or
                        stripped.startswith("'''") or stripped.startswith("__") or
                        "=" in stripped):
                        file_start = i
                    else:
                        break
                text = "\n".join(lines[file_start:]).strip()

        # Strategy 3: Find first # comment or import line
        if not text.startswith(("#", "from ", "import ", '"""', "'''")):
            lines = text.splitlines()
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith(("#", "from ", "import ", '"""', "'''")):
                    text = "\n".join(lines[i:]).strip()
                    break

        # Sanitize unicode characters
        text = text.replace("\u2192", "->")   # right arrow
        text = text.replace("\u2014", "--")   # em dash
        text = text.replace("\u2013", "-")    # en dash
        text = text.replace("\u2212", "-")    # minus sign
        text = text.replace("\u201c", '"')    # left double quote
        text = text.replace("\u201d", '"')    # right double quote
        text = text.replace("\u2018", "'")    # left single quote
        text = text.replace("\u2019", "'")    # right single quote
        text = text.replace("\u00b7", "*")    # middle dot
        text = text.replace("\u00d7", "*")    # multiplication sign
        text = text.replace("\u00f7", "/")    # division sign
        text = text.replace("\u00b1", "+-")   # plus-minus
        text = text.replace("\u2264", "<=")   # less-than-or-equal
        text = text.replace("\u2265", ">=")   # greater-than-or-equal
        text = text.replace("\u2260", "!=")   # not-equal
        text = text.replace("\u2026", "...")  # ellipsis

        return text

    # ------------------------------------------------------------------
    # HTTP -- cliproxyapi (Anthropic Messages API)
    # ------------------------------------------------------------------

    _MODEL_MAP = {
        "sonnet": "claude-sonnet-4-5-20250929",
        "haiku": "claude-haiku-4-5-20251001",
        "opus": "claude-opus-4-6",
    }

    def _chat(self, messages: list[dict[str, Any]]) -> str:
        """Call LLM via cliproxyapi (Anthropic Messages API)."""
        model_name = self._MODEL_MAP.get(self._model, self._model)

        # Separate system from conversation messages
        system_text = ""
        api_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                api_messages.append({"role": msg["role"], "content": msg["content"]})
        if not api_messages:
            api_messages = [{"role": "user", "content": "Hello"}]

        url = "http://127.0.0.1:8317/v1/messages"
        payload: dict[str, Any] = {
            "model": model_name,
            "max_tokens": 16384,
            "messages": api_messages,
        }
        if system_text:
            payload["system"] = system_text

        body = json.dumps(payload).encode()
        headers = {
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": "your-api-key-1",
        }

        for attempt in range(3):
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    data = json.loads(resp.read().decode())
                    content = data.get("content", [])
                    if content and isinstance(content, list):
                        return content[0].get("text", "").strip()
                    return ""
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 504, 529) and attempt < 2:
                    time.sleep(5 * (2 ** attempt))
                    continue
                raise RuntimeError(
                    f"Mutator HTTP {exc.code}: {exc.reason}"
                ) from exc
            except urllib.error.URLError as exc:
                if attempt < 2:
                    time.sleep(5 * (2 ** attempt))
                    continue
                raise RuntimeError(
                    f"Mutator connection error: {exc.reason}"
                ) from exc

        raise RuntimeError("Mutator chat request failed after 3 retries")
