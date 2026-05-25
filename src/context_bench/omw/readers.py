"""QA reader + judge. Uses Claude on Bedrock."""
from __future__ import annotations
import json
import os


def _client():
    import boto3
    return boto3.client("bedrock-runtime", region_name="us-east-1")


def _claude_call(client, prompt: str, model_id: str, max_tokens: int = 200) -> str:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    r = client.invoke_model(
        body=json.dumps(body),
        modelId=model_id,
        accept="application/json", contentType="application/json",
    )
    return json.loads(r["body"].read())["content"][0]["text"].strip()


class BedrockReader:
    """A QA reader that produces answers from memory + optional window."""

    def __init__(self, model_id: str):
        self.client = _client()
        self.model_id = model_id

    def __call__(self, question: str, memory_text: str, window_text: str = "") -> str:
        if not memory_text and not window_text:
            return ""
        parts = []
        if memory_text:
            parts.append(f"MEMORY:\n{memory_text}")
        if window_text:
            parts.append(f"RECENT RAW SESSIONS:\n{window_text}")
        evidence = "\n\n".join(parts)
        prompt = (
            "Answer the question precisely. If the answer is a number or count, "
            "give just the number. Be concise (under 15 words). "
            "If the evidence does not support an answer, reply: NO_EVIDENCE.\n\n"
            f"{evidence}\n\nQUESTION: {question}\n\nAnswer:"
        )
        return _claude_call(self.client, prompt, self.model_id, max_tokens=100)

    def judge_recall(self, memory_text: str, gold: str, question: str) -> int:
        """Does memory contain the information needed to answer Q with gold?"""
        if not memory_text.strip():
            return 0
        prompt = (
            "Below is the contents of a memory store and a QUESTION the agent will "
            "need to answer. Reply PRESENT if the memory contains information that "
            "supports the GOLD answer to the question. Reply MISSING otherwise.\n\n"
            f"QUESTION: {question}\nGOLD ANSWER: {gold}\n\nMEMORY:\n{memory_text[:30000]}\n\n"
            "Reply (PRESENT/MISSING):"
        )
        v = _claude_call(self.client, prompt, "us.anthropic.claude-sonnet-4-6", max_tokens=10).upper()
        if "MISSING" in v: return 0
        return 1 if "PRESENT" in v else 0

    def judge(self, ans: str, gold: str, question: str) -> int:
        if not ans.strip():
            return 0
        prompt = (
            "Judge if the PREDICTION correctly answers the QUESTION given GOLD. "
            "Reply CORRECT or WRONG only.\n\n"
            f"QUESTION: {question}\nGOLD: {gold}\nPREDICTION: {ans}\n\nReply:"
        )
        v = _claude_call(self.client, prompt, "us.anthropic.claude-sonnet-4-6", max_tokens=20).upper()
        if "WRONG" in v or "INCORRECT" in v or "NOT CORRECT" in v:
            return 0
        return 1 if "CORRECT" in v else 0


def get_reader(args) -> BedrockReader:
    return BedrockReader(model_id=args.reader_model)
