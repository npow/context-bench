"""NLI label match evaluator.

Extracts a classification label from the model response and compares to the
reference label. Supports common NLI/fact-verification label sets.
"""

from __future__ import annotations

import re
from typing import Any

# Canonical label mappings — each maps common variants to a standard form.
_LABEL_ALIASES: dict[str, str] = {
    # 3-class NLI (ContractNLI, MNLI, etc.)
    "entailment": "entailment",
    "entail": "entailment",
    "yes": "entailment",
    "true": "entailment",
    "contradiction": "contradiction",
    "contradict": "contradiction",
    "no": "contradiction",
    "false": "contradiction",
    "not mentioned": "not mentioned",
    "not_mentioned": "not mentioned",
    "neutral": "not mentioned",
    "unknown": "not mentioned",
    "neither": "not mentioned",
    # SciFact 2-class
    "supports": "supports",
    "support": "supports",
    "refutes": "refutes",
    "refute": "refutes",
}


class NLILabelMatch:
    """Score NLI/fact-verification responses by label matching.

    Extracts the predicted label from the response using keyword matching
    and compares to the reference answer. Returns ``nli_accuracy`` (1.0 or 0.0).
    """

    @property
    def name(self) -> str:
        return "nli_label_match"

    def score(
        self, original: dict[str, Any], processed: dict[str, Any],
    ) -> dict[str, float]:
        reference = str(original.get("answer", "")).strip()
        response = str(processed.get("response", "")).strip()

        if not reference:
            return {"nli_accuracy": 1.0}
        if not response:
            return {"nli_accuracy": 0.0}

        ref_label = _normalize_label(reference)
        resp_label = _extract_label(response)

        if ref_label and resp_label and ref_label == resp_label:
            return {"nli_accuracy": 1.0}
        return {"nli_accuracy": 0.0}


def _normalize_label(text: str) -> str:
    """Normalize a label string to its canonical form."""
    key = text.strip().lower()
    return _LABEL_ALIASES.get(key, key)


def _extract_label(response: str) -> str:
    """Extract a classification label from model response text."""
    text = response.strip().lower()

    # If the entire response is a known label
    if text in _LABEL_ALIASES:
        return _LABEL_ALIASES[text]

    # Look for patterns like "the answer is: entailment" or "label: supports"
    for pattern in [
        r"(?:answer|label|verdict|classification|judgment)\s*(?:is|:)\s*[\"']?(\w[\w\s]*?)(?:[\"'.,;!?\n]|$)",
        r"(?:therefore|thus|hence|so)\s*(?:,\s*)?(?:the\s+)?(?:answer|label)?\s*(?:is\s+)?[\"']?(\w[\w\s]*?)(?:[\"'.,;!?\n]|$)",
    ]:
        m = re.search(pattern, text)
        if m:
            candidate = m.group(1).strip()
            normalized = _normalize_label(candidate)
            if normalized in set(_LABEL_ALIASES.values()):
                return normalized

    # Scan for any known label keyword in the response (prefer last occurrence
    # since conclusions tend to appear at the end)
    last_match = ""
    last_pos = -1
    for alias, canonical in _LABEL_ALIASES.items():
        # Only match multi-word aliases or whole words
        if " " in alias:
            pattern = re.escape(alias)
        else:
            pattern = r"\b" + re.escape(alias) + r"\b"
        for m in re.finditer(pattern, text):
            if m.start() > last_pos:
                last_pos = m.start()
                last_match = canonical

    return last_match
