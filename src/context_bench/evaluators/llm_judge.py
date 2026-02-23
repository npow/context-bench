"""LLM-as-judge evaluator for open-ended tasks.

Uses an external LLM (GPT-4, Claude, etc.) to score responses on a 1-5 scale.
Follows the standard MT-Bench / AlpacaEval evaluation template.

No new dependencies — uses stdlib urllib like OpenAIProxy._post().
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

_JUDGE_PROMPT = """\
You are an expert judge evaluating the quality of an AI assistant's response.

[Question]
{question}

[Reference Answer]
{reference}

[Assistant's Response]
{response}

Rate the assistant's response on a scale of 1 to 5:
1 - Completely wrong or irrelevant
2 - Partially addresses the question but has major errors
3 - Addresses the question but misses key details
4 - Good response with minor issues
5 - Excellent, fully addresses the question

Output ONLY a single integer (1-5) as your rating."""


class LLMJudge:
    """Score responses using an LLM judge (GPT-4, Claude, etc.).

    Args:
        base_url: OpenAI-compatible API base URL.
        model: Model name for the judge (default: "gpt-4").
        api_key: Bearer token. Falls back to ``OPENAI_API_KEY`` env var.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        model: str = "gpt-4",
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "llm_judge"

    def score(
        self, original: dict[str, Any], processed: dict[str, Any]
    ) -> dict[str, float]:
        """Score a response using the LLM judge.

        Returns {"judge_score": float} normalized to 0-1 range.
        """
        question = str(original.get("question", original.get("context", "")))
        reference = str(original.get("answer", ""))
        response = str(processed.get("response", ""))

        if not response:
            return {"judge_score": 0.0}

        prompt = _JUDGE_PROMPT.format(
            question=question,
            reference=reference,
            response=response,
        )

        try:
            rating_text = self._call_judge(prompt)
            rating = self._parse_rating(rating_text)
            # Normalize 1-5 scale to 0-1
            return {"judge_score": (rating - 1) / 4.0}
        except Exception:
            # On judge failure, return 0 rather than crashing the benchmark
            return {"judge_score": 0.0}

    def _call_judge(self, prompt: str) -> str:
        """Send the judge prompt and return raw response text."""
        url = f"{self._base_url}/v1/chat/completions"
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 16,
        }
        data = json.dumps(body).encode()

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            resp_data = json.loads(resp.read().decode())

        return resp_data["choices"][0]["message"]["content"]

    @staticmethod
    def _parse_rating(text: str) -> int:
        """Extract integer rating (1-5) from judge response text."""
        match = re.search(r"[1-5]", text.strip())
        if match:
            return int(match.group())
        return 1
