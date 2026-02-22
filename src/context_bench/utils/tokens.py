"""Pluggable tokenizer utilities. Default: tiktoken cl100k_base."""

from __future__ import annotations

from typing import Callable

import tiktoken

_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str, tokenizer: Callable[[str], int] | None = None) -> int:
    """Count tokens in text.

    Args:
        text: The text to tokenize.
        tokenizer: Optional custom tokenizer function (str -> int).
            Defaults to tiktoken cl100k_base.
    """
    if tokenizer is not None:
        return tokenizer(text)
    return len(_get_encoder().encode(text))


def count_tokens_dict(
    d: dict,
    text_fields: list[str] | None = None,
    tokenizer: Callable[[str], int] | None = None,
) -> int:
    """Count tokens across text fields of a dict.

    If text_fields is None, counts tokens in all string values.
    """
    total = 0
    for key, value in d.items():
        if not isinstance(value, str):
            continue
        if text_fields is not None and key not in text_fields:
            continue
        total += count_tokens(value, tokenizer)
    return total
