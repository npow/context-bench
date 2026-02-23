"""IFEval instruction-following checker.

Implements programmatic checks for verifiable instruction constraints
from the IFEval benchmark. Each instruction type is dispatched to a
dedicated checker function that returns True/False.
"""

from __future__ import annotations

import json
import re
from typing import Any


class IFEvalChecker:
    """Check instruction-following constraints programmatically.

    Expects original dict to have ``instruction_id_list`` (list of str) and
    ``kwargs`` (list of dict) with per-instruction parameters.

    Returns ``ifeval_strict`` (1.0 if ALL pass, else 0.0) and
    ``ifeval_loose`` (fraction passing).
    """

    @property
    def name(self) -> str:
        return "ifeval"

    def score(
        self, original: dict[str, Any], processed: dict[str, Any],
    ) -> dict[str, float]:
        response = str(processed.get("response", ""))
        instructions = original.get("instruction_id_list", [])
        kwargs_list = original.get("kwargs", [])

        if not instructions:
            return {"ifeval_strict": 1.0, "ifeval_loose": 1.0}

        passed = 0
        total = len(instructions)

        for i, instruction_id in enumerate(instructions):
            kw = kwargs_list[i] if i < len(kwargs_list) else {}
            if kw is None:
                kw = {}
            if self._check_instruction(instruction_id, response, kw):
                passed += 1

        loose = passed / total if total > 0 else 1.0
        strict = 1.0 if passed == total else 0.0
        return {"ifeval_strict": strict, "ifeval_loose": loose}

    def _check_instruction(
        self, instruction_id: str, response: str, kwargs: dict,
    ) -> bool:
        """Dispatch to the appropriate checker."""
        checkers = {
            "keywords:existence": self._check_keyword_existence,
            "keywords:forbidden_words": self._check_forbidden_words,
            "keywords:frequency": self._check_keyword_frequency,
            "keywords:letter_frequency": self._check_letter_frequency,
            "length_constraints:number_words": self._check_number_words,
            "length_constraints:number_sentences": self._check_number_sentences,
            "length_constraints:number_paragraphs": self._check_number_paragraphs,
            "detectable_format:json_format": self._check_json_format,
            "detectable_format:number_bullet_points": self._check_bullet_points,
            "detectable_format:title": self._check_title,
            "detectable_format:constrained_response": self._check_constrained_response,
            "detectable_content:number_placeholders": self._check_placeholders,
            "detectable_content:postscript": self._check_postscript,
            "punctuation:no_comma": self._check_no_comma,
            "change_case:english_capital": self._check_english_capital,
            "change_case:english_lowercase": self._check_english_lowercase,
            "startend:end_checker": self._check_end_checker,
            "combination:repeat_prompt": self._check_repeat_prompt,
            "combination:two_responses": self._check_two_responses,
        }

        checker = checkers.get(instruction_id)
        if checker is None:
            # Unknown instruction — fail open (count as pass)
            return True
        try:
            return checker(response, kwargs)
        except Exception:
            return False

    # ---- keyword checkers ----

    @staticmethod
    def _check_keyword_existence(response: str, kwargs: dict) -> bool:
        keywords = kwargs.get("keywords") or []
        resp_lower = response.lower()
        return all(kw.lower() in resp_lower for kw in keywords)

    @staticmethod
    def _check_forbidden_words(response: str, kwargs: dict) -> bool:
        forbidden = kwargs.get("forbidden_words") or []
        resp_lower = response.lower()
        return all(w.lower() not in resp_lower for w in forbidden)

    @staticmethod
    def _check_keyword_frequency(response: str, kwargs: dict) -> bool:
        keyword = kwargs.get("keyword", "")
        frequency = kwargs.get("frequency", 0)
        relation = kwargs.get("relation", "at least")
        count = response.lower().count(keyword.lower())
        if relation == "at least":
            return count >= frequency
        elif relation == "at most":
            return count <= frequency
        elif relation == "exactly":
            return count == frequency
        return count >= frequency

    @staticmethod
    def _check_letter_frequency(response: str, kwargs: dict) -> bool:
        letter = kwargs.get("letter", "")
        let_frequency = kwargs.get("let_frequency", 0)
        let_relation = kwargs.get("let_relation", "at least")
        count = response.lower().count(letter.lower())
        if let_relation == "at least":
            return count >= let_frequency
        elif let_relation == "at most":
            return count <= let_frequency
        elif let_relation == "exactly":
            return count == let_frequency
        return count >= let_frequency

    # ---- length checkers ----

    @staticmethod
    def _check_number_words(response: str, kwargs: dict) -> bool:
        num_words = kwargs.get("num_words", 0)
        relation = kwargs.get("relation", "at least")
        count = len(response.split())
        if relation == "at least":
            return count >= num_words
        elif relation == "at most":
            return count <= num_words
        elif relation == "exactly":
            return count == num_words
        return count >= num_words

    @staticmethod
    def _check_number_sentences(response: str, kwargs: dict) -> bool:
        num_sentences = kwargs.get("num_sentences", 0)
        relation = kwargs.get("relation", "at least")
        # Simple sentence split on .!?
        sentences = [s.strip() for s in re.split(r"[.!?]+", response) if s.strip()]
        count = len(sentences)
        if relation == "at least":
            return count >= num_sentences
        elif relation == "at most":
            return count <= num_sentences
        elif relation == "exactly":
            return count == num_sentences
        return count >= num_sentences

    @staticmethod
    def _check_number_paragraphs(response: str, kwargs: dict) -> bool:
        num_paragraphs = kwargs.get("num_paragraphs", 0)
        paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
        count = len(paragraphs)
        return count >= num_paragraphs

    # ---- format checkers ----

    @staticmethod
    def _check_json_format(response: str, kwargs: dict) -> bool:
        try:
            json.loads(response.strip())
            return True
        except (json.JSONDecodeError, ValueError):
            return False

    @staticmethod
    def _check_bullet_points(response: str, kwargs: dict) -> bool:
        num_bullets = kwargs.get("num_bullets", 0)
        bullets = re.findall(r"^\s*[\*\-•]\s+", response, re.MULTILINE)
        return len(bullets) >= num_bullets

    @staticmethod
    def _check_title(response: str, kwargs: dict) -> bool:
        return bool(re.search(r"<<[^>]+>>", response))

    @staticmethod
    def _check_constrained_response(response: str, kwargs: dict) -> bool:
        choices = kwargs.get("constrained_response") or []
        resp_stripped = response.strip()
        return any(resp_stripped == c for c in choices)

    # ---- content checkers ----

    @staticmethod
    def _check_placeholders(response: str, kwargs: dict) -> bool:
        num_placeholders = kwargs.get("num_placeholders", 0)
        placeholders = re.findall(r"\[.+?\]", response)
        return len(placeholders) >= num_placeholders

    @staticmethod
    def _check_postscript(response: str, kwargs: dict) -> bool:
        postscript_marker = kwargs.get("postscript_marker", "P.S.")
        return postscript_marker.lower() in response.lower()

    # ---- punctuation ----

    @staticmethod
    def _check_no_comma(response: str, kwargs: dict) -> bool:
        return "," not in response

    # ---- case checkers ----

    @staticmethod
    def _check_english_capital(response: str, kwargs: dict) -> bool:
        words = response.split()
        return all(w[0].isupper() for w in words if w and w[0].isalpha())

    @staticmethod
    def _check_english_lowercase(response: str, kwargs: dict) -> bool:
        return response == response.lower()

    # ---- start/end checkers ----

    @staticmethod
    def _check_end_checker(response: str, kwargs: dict) -> bool:
        end_phrase = kwargs.get("end_phrase", "")
        return response.rstrip().endswith(end_phrase)

    # ---- combination checkers ----

    @staticmethod
    def _check_repeat_prompt(response: str, kwargs: dict) -> bool:
        prompt = kwargs.get("prompt_to_repeat", "")
        return prompt in response

    @staticmethod
    def _check_two_responses(response: str, kwargs: dict) -> bool:
        return "******" in response
