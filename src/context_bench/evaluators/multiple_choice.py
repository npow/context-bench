"""Multiple-choice accuracy evaluator.

Extracts the model's chosen letter from the response and compares to
the correct letter stored in the example.
"""

from __future__ import annotations

import re
from typing import Any


class MultipleChoiceAccuracy:
    """Score multiple-choice responses by extracting the chosen letter.

    Expects original dict to have ``correct_letter`` (e.g. ``"A"``).
    """

    @property
    def name(self) -> str:
        return "multiple_choice"

    def score(
        self, original: dict[str, Any], processed: dict[str, Any],
    ) -> dict[str, float]:
        correct = str(original.get("correct_letter", "")).upper().strip()
        if not correct:
            return {"mc_accuracy": 0.0}

        response = str(processed.get("response", "")).strip()
        if not response:
            return {"mc_accuracy": 0.0}

        chosen = self._extract_letter(response)
        return {"mc_accuracy": 1.0 if chosen == correct else 0.0}

    @staticmethod
    def _extract_letter(response: str) -> str:
        """Extract the chosen letter (A-D) from model response."""
        text = response.strip()

        # Direct single letter
        if len(text) == 1 and text.upper() in "ABCD":
            return text.upper()

        # "The answer is A" or "answer: B" pattern
        m = re.search(r"(?:answer|choice)\s*(?:is|:)\s*([A-Da-d])", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

        # Parenthesized: (A) or (B)
        m = re.search(r"\(([A-Da-d])\)", text)
        if m:
            return m.group(1).upper()

        # Fallback: first A-D letter found
        m = re.search(r"\b([A-Da-d])\b", text)
        if m:
            return m.group(1).upper()

        return ""
