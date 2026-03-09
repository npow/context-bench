"""LLM-as-judge evaluator for memory QA tasks.

Uses an external LLM reachable via a relay (OpenAI-compatible endpoint) to
judge whether a system answer correctly responds to a question given a known
gold answer.  Designed for short-answer memory QA where token-overlap metrics
(F1, ROUGE) are unreliable.

No new dependencies -- uses stdlib urllib following the same retry/timeout
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

_JUDGE_PROMPT = """\
You are a strict but fair judge evaluating a memory QA system.

[Question]
{question}

[Gold Answer]
{gold}

[System Answer]
{answer}

Given the question and gold answer above, does the system answer correctly?
A correct answer must convey the same essential information as the gold answer.
Minor wording differences, extra context, or slight rephrasing are acceptable
as long as the core fact is right.

Answer with YES or NO on the first line, then briefly explain in one sentence.
Example output:
YES - The system answer states the correct date."""


class MemoryJudge:
    """Score memory QA responses using an LLM judge via a relay endpoint.

    The judge asks whether the system answer correctly answers the question
    given the known gold answer.  Returns a binary ``memory_judge`` score
    (1.0 = correct, 0.0 = incorrect) and a ``memory_judge_raw`` score in
    [0, 1] derived from the YES/NO decision (currently identical to the
    binary score; reserved for future partial-credit extension).

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
        return "memory_judge"

    def score(
        self, original: dict[str, Any], processed: dict[str, Any]
    ) -> dict[str, float]:
        """Judge whether the system answer is correct.

        Returns:
            ``{"memory_judge": 1.0 or 0.0, "memory_judge_raw": float in [0, 1]}``

            On judge failure (network error, malformed response) returns zeros
            rather than propagating the exception, so a single failure does not
            abort the entire benchmark run.
        """
        question = str(original.get("question", ""))
        gold = str(original.get("answer", ""))
        answer = str(processed.get("response", ""))

        if not answer:
            return {"memory_judge": 0.0, "memory_judge_raw": 0.0}

        # If there is no gold answer we cannot judge; return NaN-equivalent
        # (0.5) so this row neither inflates nor deflates the mean, rather
        # than returning 1.0 which would artificially reward all systems.
        if not gold:
            return {"memory_judge": 0.5, "memory_judge_raw": 0.5}

        prompt = _JUDGE_PROMPT.format(
            question=question,
            gold=gold,
            answer=answer,
        )

        try:
            raw_text = self._call_relay(prompt)
            binary, raw = self._parse_judgment(raw_text)
            return {"memory_judge": binary, "memory_judge_raw": raw}
        except Exception:
            # On any failure return zeros rather than crashing the run.
            return {"memory_judge": 0.0, "memory_judge_raw": 0.0}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _call_relay(self, prompt: str) -> str:
        """POST the judge prompt to the relay and return raw response text."""
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
    def _parse_judgment(text: str) -> tuple[float, float]:
        """Extract YES/NO from the judge response.

        Returns:
            Tuple of (binary_score, raw_score) both in {0.0, 1.0}.
            Defaults to (0.0, 0.0) when the response is ambiguous.
        """
        # Look for YES or NO at the very start of the response (case-insensitive).
        match = re.search(r"\b(YES|NO)\b", text.strip(), re.IGNORECASE)
        if match:
            verdict = match.group(1).upper()
            score = 1.0 if verdict == "YES" else 0.0
            return score, score
        # Ambiguous response -- treat as incorrect.
        return 0.0, 0.0
