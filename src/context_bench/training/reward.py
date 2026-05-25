"""Reward function for GRPO training of the memory management policy.

Executes model-generated REPL code against a BM25 retrieval store built from
the conversation, then scores the extracted answer with F1.

reward = F1(answer, ground_truth) + WRITE_BONUS * (1 if memory_write was called)

BM25 is used instead of LanceDB vectors to avoid GPU/embedding overhead in the
reward function (training runs on the same GPUs as generation).
"""

from __future__ import annotations

import re
import threading
from collections import Counter
from typing import Any


# ---- reward parameters --------------------------------------------------

WRITE_BONUS = 0.3        # additive bonus when memory_write is called AND executed
WRITE_ATTEMPT_BONUS = 0.05  # smaller partial bonus when memory_write appears in code text
# Re-balanced (v5): incentivize actual execution > syntactic appearance to prevent reward hacking
EXEC_TIMEOUT = 8.0       # max seconds to execute generated code


# ---- lightweight BM25 store --------------------------------------------

class BM25Store:
    """In-memory keyword retrieval store for training reward computation."""

    def __init__(self, turns: list[str]) -> None:
        self._turns = turns
        self._tokenized = [self._tokenize(t) for t in turns]
        self._idf = self._compute_idf()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def _compute_idf(self) -> dict[str, float]:
        import math
        N = len(self._tokenized)
        df: dict[str, int] = {}
        for tokens in self._tokenized:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        return {t: math.log((N - n + 0.5) / (n + 0.5) + 1) for t, n in df.items()}

    def retrieve(self, query: str, k: int = 20) -> list[str]:
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return self._turns[:k]
        k1, b, avgdl = 1.5, 0.75, sum(len(t) for t in self._tokenized) / max(1, len(self._tokenized))
        scores: list[tuple[float, str]] = []
        for tokens, turn in zip(self._tokenized, self._turns):
            dl = len(tokens)
            score = 0.0
            tf = Counter(tokens)
            for t in q_tokens:
                if t not in self._idf:
                    continue
                f = tf.get(t, 0)
                score += self._idf[t] * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / max(1, avgdl)))
            scores.append((score, turn))
        scores.sort(key=lambda x: -x[0])
        return [t for _, t in scores[:k]]


# ---- REPL execution -----------------------------------------------------

def _execute_repl(
    code: str,
    store: BM25Store,
    write_store: list[str],
) -> tuple[str, bool]:
    """Execute generated code in a sandboxed REPL namespace.

    Returns (answer_content, wrote_anything).
    The write_store list is mutated in place with any memory_write calls.
    """
    answer: dict[str, Any] = {"content": "", "ready": False}
    wrote = [False]
    cancelled = threading.Event()

    def memory_read(query: str, k: int = 20) -> list[str]:
        if cancelled.is_set():
            return []
        results = store.retrieve(query, k)
        # also search write_store entries
        if write_store:
            ws_store = BM25Store(write_store)
            results = results + ws_store.retrieve(query, k // 2)
        return results[:k]

    def memory_write(content: str, memory_type: str = "episodic") -> None:
        if cancelled.is_set():
            return
        wrote[0] = True
        write_store.append(str(content))

    def consolidate() -> str:
        if cancelled.is_set():
            return ""
        if not write_store:
            return ""
        return " | ".join(write_store[-5:])

    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "memory_read": memory_read,
        "memory_write": memory_write,
        "consolidate": consolidate,
        "answer": answer,
    }

    exc_box: list[str | None] = [None]

    def _run(code: str = code, ns: dict = namespace) -> None:
        try:
            exec(code, ns)  # noqa: S102
        except Exception as exc:
            exc_box[0] = str(exc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=EXEC_TIMEOUT)
    if t.is_alive():
        cancelled.set()

    return answer.get("content", "") or "", wrote[0]


# ---- F1 scoring ---------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return " ".join(text.split())


def token_f1(prediction: str, reference: str) -> float:
    if not reference.strip():
        return 1.0
    pred_tokens = _normalize(prediction).split()
    ref_tokens = _normalize(reference).split()
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ---- Main reward function -----------------------------------------------

def _extract_answer_from_text(code: str) -> str:
    """Fallback: extract answer from code text when exec() fails.

    The 3B model often generates partial/invalid Python but contains the
    answer inline. Try multiple extraction patterns before giving up.
    """
    patterns = [
        # Standard: answer["content"] = "X"
        r'answer\["content"\]\s*=\s*["\']([^"\']*)["\']',
        # Variable assignment: Content = "X" or content = "X"
        r'[Cc]ontent\s*=\s*["\']([^"\']*)["\']',
        r'answer_content\s*=\s*["\']([^"\']*)["\']',
        # Answer: "X" format
        r'[Aa]nswer[:\s]+["\']([^"\']{3,100})["\']',
        # Direct fact statement (first quoted string of 3-50 chars)
        r'"([^"]{3,80})"',
    ]
    import re as _re
    for pat in patterns:
        m = _re.search(pat, code, _re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            if val and len(val) > 1:
                return val
    return ""


def compute_reward(
    code: str,
    turns: list[str],
    ground_truth: str,
    write_store: list[str] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Compute GRPO reward for a single (code, conversation) pair.

    First tries exec() to get answer from answer["content"]. If that fails
    or gives empty answer, falls back to text extraction from the code string.
    This handles the common case where a smaller model generates partially-valid
    code that contains the answer but doesn't set answer["content"] correctly.

    Args:
        code: Model-generated REPL code or natural language
        turns: Conversation turns (pre-split, as plain strings)
        ground_truth: Expected answer
        write_store: Shared mutable list; writes go here (for multi-query episodes)

    Returns:
        (reward, info_dict)
    """
    if write_store is None:
        write_store = []
    store = BM25Store(turns)
    answer, wrote = _execute_repl(code, store, write_store)

    # If exec() produced no answer, fall back to text extraction
    if not answer.strip():
        answer = _extract_answer_from_text(code)

    f1 = token_f1(answer, ground_truth)
    # Full write bonus if exec succeeded and memory_write was called.
    # Partial bonus if memory_write appears in code text (attempt reward).
    write_in_text = "memory_write" in code.lower()
    reward = f1 + WRITE_BONUS * float(wrote) + WRITE_ATTEMPT_BONUS * float(write_in_text and not wrote)
    return reward, {
        "f1": f1,
        "wrote": wrote,
        "write_attempted": write_in_text,
        "answer": answer,
        "ground_truth": ground_truth,
    }


def batch_reward_fn(
    prompts: list[str],
    completions: list[str],
    turns: list[list[str]] | None = None,
    ground_truth: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Reward function for trl.GRPOTrainer (trl ≥ 1.0 interface).

    In trl ≥ 1.0 with RepeatSampler, the dataset is already repeated G times,
    so each argument has length B*G:
        prompts:       list[str]        len = B*G
        completions:   list[str]        len = B*G (decoded text, not dicts)
        turns:         list[list[str]]  len = B*G (dataset column)
        ground_truth:  list[str]        len = B*G (dataset column)

    Returns:
        list[float] of length B*G
    """
    if turns is None or ground_truth is None:
        return [0.0] * len(completions)

    rewards: list[float] = []
    for code, conv_turns, gt in zip(completions, turns, ground_truth):
        code_str = code if isinstance(code, str) else str(code)
        reward, _ = compute_reward(code_str, conv_turns, gt)
        rewards.append(reward)

    return rewards
