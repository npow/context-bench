"""Math equivalence evaluator.

Compares mathematical answers by normalizing LaTeX notation and attempting
numeric comparison. Pure Python — no external math libraries required.
"""

from __future__ import annotations

import re
from typing import Any


class MathEquivalence:
    """Score math responses by checking semantic equivalence of answers.

    Handles LaTeX normalization (``\\frac{1}{2}`` vs ``0.5``), sign/whitespace
    differences, and common formatting variants.

    Returns ``math_equiv`` (1.0 if equivalent, else 0.0).
    """

    @property
    def name(self) -> str:
        return "math_equivalence"

    def score(
        self, original: dict[str, Any], processed: dict[str, Any],
    ) -> dict[str, float]:
        reference = str(original.get("answer", "")).strip()
        response = str(processed.get("response", "")).strip()

        if not reference:
            return {"math_equiv": 1.0}
        if not response:
            return {"math_equiv": 0.0}

        # Extract boxed answer from response if present
        extracted = _extract_boxed(response)
        if extracted:
            response = extracted

        equiv = _is_equivalent(reference, response)
        return {"math_equiv": 1.0 if equiv else 0.0}


def _extract_boxed(text: str) -> str:
    r"""Extract content from \boxed{...} with brace matching."""
    start = text.find(r"\boxed{")
    if start == -1:
        return ""
    i = start + len(r"\boxed{")
    depth = 1
    result = []
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
            result.append(text[i])
        elif text[i] == "}":
            depth -= 1
            if depth > 0:
                result.append(text[i])
        else:
            result.append(text[i])
        i += 1
    return "".join(result)


def _normalize_latex(text: str) -> str:
    """Normalize LaTeX math notation to a canonical form."""
    s = text.strip()
    # Remove dollar signs
    s = s.strip("$")
    # Remove \text{}, \mathrm{}, \textbf{}, etc.
    s = re.sub(r"\\(?:text|mathrm|textbf|mathbf|mathit)\s*\{([^}]*)\}", r"\1", s)
    # Remove \left and \right
    s = re.sub(r"\\(?:left|right)\s*", "", s)
    # Normalize \frac{a}{b} -> (a)/(b)
    s = _normalize_fractions(s)
    # Remove \, \; \: \! (spacing commands)
    s = re.sub(r"\\[,;:!]", "", s)
    # Remove \cdot -> *
    s = re.sub(r"\\cdot\s*", "*", s)
    # Remove \times -> *
    s = re.sub(r"\\times\s*", "*", s)
    # \pm -> +-
    s = s.replace(r"\pm", "+-")
    # \sqrt{x} -> sqrt(x)
    s = re.sub(r"\\sqrt\s*\{([^}]*)\}", r"sqrt(\1)", s)
    # \pi -> pi
    s = s.replace(r"\pi", "pi")
    # \infty -> inf
    s = s.replace(r"\infty", "inf")
    # Remove remaining backslashes before known commands
    s = re.sub(r"\\(?:displaystyle|phantom|hspace|vspace|quad|qquad)\s*\{?[^}]*\}?", "", s)
    # Collapse whitespace
    s = " ".join(s.split())
    return s


def _normalize_fractions(text: str) -> str:
    r"""Convert \frac{a}{b} to (a)/(b), handling nested braces."""
    result = []
    i = 0
    while i < len(text):
        if text[i:].startswith(r"\frac"):
            i += len(r"\frac")
            # Skip optional whitespace
            while i < len(text) and text[i] == " ":
                i += 1
            num, i = _extract_brace_content(text, i)
            # Skip optional whitespace
            while i < len(text) and text[i] == " ":
                i += 1
            den, i = _extract_brace_content(text, i)
            result.append(f"({num})/({den})")
        else:
            result.append(text[i])
            i += 1
    return "".join(result)


def _extract_brace_content(text: str, pos: int) -> tuple[str, int]:
    """Extract content between { and }, returning (content, new_pos)."""
    if pos >= len(text) or text[pos] != "{":
        # No braces — take single character/token
        if pos < len(text):
            return text[pos], pos + 1
        return "", pos
    pos += 1  # skip {
    depth = 1
    content = []
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
            content.append(text[pos])
        elif text[pos] == "}":
            depth -= 1
            if depth > 0:
                content.append(text[pos])
        else:
            content.append(text[pos])
        pos += 1
    return "".join(content), pos


def _try_parse_number(text: str) -> float | None:
    """Attempt to parse text as a number. Returns None on failure."""
    s = text.strip()
    # Remove commas in numbers (e.g., "1,000")
    s = s.replace(",", "")
    # Handle percentage
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100
        except ValueError:
            pass
    # Try simple fraction a/b
    if "/" in s and "\\" not in s:
        parts = s.split("/")
        if len(parts) == 2:
            try:
                num = float(parts[0].strip().strip("()"))
                den = float(parts[1].strip().strip("()"))
                if den != 0:
                    return num / den
            except ValueError:
                pass
    # Try direct float parse
    try:
        return float(s)
    except ValueError:
        pass
    return None


def _is_equivalent(reference: str, response: str) -> bool:
    """Check if two math expressions are equivalent."""
    # Exact string match (case insensitive, whitespace normalized)
    ref_clean = " ".join(reference.lower().split())
    resp_clean = " ".join(response.lower().split())
    if ref_clean == resp_clean:
        return True

    # Normalize LaTeX and compare strings
    ref_norm = _normalize_latex(reference)
    resp_norm = _normalize_latex(response)
    if ref_norm.lower() == resp_norm.lower():
        return True

    # Numeric comparison
    ref_num = _try_parse_number(ref_norm)
    resp_num = _try_parse_number(resp_norm)
    if ref_num is not None and resp_num is not None:
        if ref_num == 0 and resp_num == 0:
            return True
        if abs(ref_num) > 1e-12:
            return abs(ref_num - resp_num) / abs(ref_num) < 1e-6
        return abs(ref_num - resp_num) < 1e-12

    # Also try parsing numbers from the original (pre-normalization) text
    ref_num_orig = _try_parse_number(reference)
    resp_num_orig = _try_parse_number(response)
    if ref_num_orig is not None and resp_num_orig is not None:
        if ref_num_orig == 0 and resp_num_orig == 0:
            return True
        if abs(ref_num_orig) > 1e-12:
            return abs(ref_num_orig - resp_num_orig) / abs(ref_num_orig) < 1e-6
        return abs(ref_num_orig - resp_num_orig) < 1e-12

    return False
