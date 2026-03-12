"""LLM judge evaluator for LoCoMo — matches MemMachine/Mem0 evaluation protocol.

Uses the same ACCURACY_PROMPT and scoring logic as MemMachine's llm_judge.py
so our scores are directly comparable to published SOTA numbers.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


ACCURACY_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Respond with JSON: {{"label": "CORRECT"}} or {{"label": "WRONG"}}"""


class LLMJudgeLoCoMo:
    """LLM-as-judge evaluator matching the MemMachine/Mem0 LoCoMo evaluation protocol.

    Uses a haiku model via the relay for fast, cheap judging.
    Returns llm_judge score: 1.0 = CORRECT, 0.0 = WRONG.
    """

    def __init__(
        self,
        relay_url: str,
        model: str = "haiku",
        api_key: str = "",
        timeout: float = 60.0,
    ) -> None:
        self._base_url = relay_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "llm_judge_locomo"

    def score(self, original: dict[str, Any], processed: dict[str, Any]) -> dict[str, float]:
        question = str(original.get("question", ""))
        gold_answer = str(original.get("answer", ""))
        generated_answer = str(processed.get("response", ""))

        if not gold_answer:
            return {"llm_judge": 1.0}
        if not generated_answer:
            return {"llm_judge": 0.0}

        prompt = ACCURACY_PROMPT.format(
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer,
        )

        try:
            response = self._chat([{"role": "user", "content": prompt}])
            # Parse JSON label
            label = self._extract_label(response)
            return {"llm_judge": 1.0 if label == "CORRECT" else 0.0}
        except Exception:
            # Fall back to 0 on error rather than crashing the eval
            return {"llm_judge": 0.0}

    def _extract_label(self, response: str) -> str:
        response = response.strip()
        # Try JSON parse first
        try:
            # Find JSON object in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                return str(data.get("label", "")).upper()
        except Exception:
            pass
        # Fallback: look for CORRECT/WRONG in text
        upper = response.upper()
        if "CORRECT" in upper and "WRONG" not in upper:
            return "CORRECT"
        if "WRONG" in upper and "CORRECT" not in upper:
            return "WRONG"
        return "WRONG"

    def _chat(self, messages: list[dict[str, Any]]) -> str:
        url = f"{self._base_url}/v1/chat/completions"
        body = json.dumps({"model": self._model, "messages": messages}).encode()
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        for attempt in range(3):
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    data = json.loads(resp.read().decode())
                    return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(10 * (2 ** attempt))
                    continue
                raise
            except urllib.error.URLError as e:
                if attempt < 2:
                    time.sleep(10 * (2 ** attempt))
                    continue
                raise

        raise RuntimeError("LLM judge request failed after 3 retries")
