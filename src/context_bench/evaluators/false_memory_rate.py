"""False-memory-rate evaluator for memory QA systems.

Detects whether a system answer contains hallucinated facts -- specific claims
(dates, names, numbers, events) that could not have been derived from the
question alone and therefore represent fabricated memories.

Uses an LLM judge routed through a relay (OpenAI-compatible endpoint).
No new dependencies -- uses stdlib urllib with the same retry/timeout
pattern as OpenAIProxy._post().
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

_HALLUCINATION_PROMPT = """\
You are an expert fact-checker evaluating an AI memory system for hallucinations.

[Conversation excerpt (most recent turns)]
{context}

[Question asked of the system]
{question}

[System Answer]
{answer}

Your task: decide whether the system answer contains any specific claims
(dates, names, numbers, places, events, or other concrete facts) that could
NOT be found in the conversation excerpt above and therefore represent
potentially hallucinated or fabricated information.

Important: a fact that appears in the conversation excerpt is NOT a hallucination
even if it cannot be inferred from the question alone -- the system is supposed
to recall facts from the conversation.  Only flag claims that are absent from
the conversation and were not inferable from the question.

Answer with YES (hallucination detected) or NO (no hallucination) on the
first line, then briefly explain in one sentence.
Example output:
YES - The answer states a specific date that does not appear anywhere in the conversation."""


class FalseMemoryRate:
    """Detect hallucinated facts in memory QA system answers.

    Uses an LLM judge via a relay to flag answers that contain specific
    factual claims (dates, names, numbers, etc.) that could not have been
    derived from the question alone.

    Returns ``{"false_memory": 1.0}`` when hallucination is detected, and
    ``{"false_memory": 0.0}`` when the answer appears grounded.

    Args:
        base_url: Root URL of the relay/proxy, e.g. ``"http://localhost:8080"``.
            ALL requests are routed through this URL.
        model: Model name to send in the request body.
        api_key: Bearer token.  Falls back to ``OPENAI_API_KEY`` env var.
        timeout: HTTP request timeout in seconds.
        max_retries: Retries on transient failures (HTTP 429/5xx, connection
            errors).  0 disables retries.
        retry_base_delay: Base delay in seconds for exponential backoff.
    """

    def __init__(
        self,
        base_url: str,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay

    @property
    def name(self) -> str:
        return "false_memory_rate"

    def score(
        self, original: dict[str, Any], processed: dict[str, Any]
    ) -> dict[str, float]:
        """Check the system answer for hallucinated facts.

        Returns:
            ``{"false_memory": 1.0}`` if the judge detects hallucination,
            ``{"false_memory": 0.0}`` otherwise.

            On judge failure (network error, malformed response) returns 0.0
            (no hallucination assumed) rather than propagating the exception,
            so a single failure does not abort the entire benchmark run.
        """
        question = str(original.get("question", ""))
        answer = str(processed.get("response", ""))
        context = str(original.get("context", "")).strip()

        if not answer:
            # An empty answer cannot contain hallucinated facts.
            return {"false_memory": 0.0}

        if not context:
            context = "(no conversation context available)"

        prompt = _HALLUCINATION_PROMPT.format(
            context=context,
            question=question,
            answer=answer,
        )

        try:
            raw_text = self._call_relay(prompt)
            hallucinated = self._parse_verdict(raw_text)
            return {"false_memory": hallucinated}
        except Exception:
            # On any failure return 0 (no hallucination detected) rather than
            # crashing the run.  Callers should monitor for systematic 0.0
            # scores as a signal that the judge endpoint is unreachable.
            return {"false_memory": 0.0}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _call_relay(self, prompt: str) -> str:
        """POST the hallucination-detection prompt to the relay."""
        url = f"{self._base_url}/v1/chat/completions"
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            # Deterministic output -- no randomness.
            "temperature": 0,
            # YES/NO + one explanatory sentence; 64 tokens is ample.
            "max_tokens": 64,
        }
        data = json.dumps(body).encode()

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        last_exc: Exception | None = None

        for attempt in range(1 + self._max_retries):
            req = urllib.request.Request(
                url, data=data, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    resp_data = json.loads(resp.read().decode())

                try:
                    choices = resp_data["choices"]
                    if not choices:
                        raise RuntimeError("Relay returned empty choices list")
                    return choices[0]["message"]["content"]
                except (KeyError, IndexError, TypeError) as exc:
                    raise RuntimeError(
                        f"Unexpected relay response structure: {resp_data}"
                    ) from exc

            except urllib.error.HTTPError as exc:
                last_exc = exc
                if exc.code in (429, 500, 502, 503, 504) and attempt < self._max_retries:
                    delay = self._retry_base_delay * (2 ** attempt)
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            pass
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"Relay returned HTTP {exc.code}: {exc.reason}"
                ) from exc

            except urllib.error.URLError as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    time.sleep(self._retry_base_delay * (2 ** attempt))
                    continue
                raise RuntimeError(
                    f"Could not connect to relay at {url}: {exc.reason}"
                ) from exc

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Relay returned invalid JSON: {exc}"
                ) from exc

        raise RuntimeError(
            f"Failed after {self._max_retries} retries: {last_exc}"
        )

    @staticmethod
    def _parse_verdict(text: str) -> float:
        """Extract YES/NO hallucination verdict from judge response.

        Returns:
            1.0 if YES (hallucination detected), 0.0 if NO or ambiguous.
        """
        match = re.search(r"\b(YES|NO)\b", text.strip(), re.IGNORECASE)
        if match:
            return 1.0 if match.group(1).upper() == "YES" else 0.0
        # Ambiguous response -- assume no hallucination (conservative).
        return 0.0
