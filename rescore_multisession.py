"""Rescore multisession results with proper metrics.

Old F1 (token-overlap) was broken for numerical answers:
- Expected: "3", Predicted: "Two items: the Zara boots..." → F1=0 (no overlap)
  Even though prediction is verifiably wrong about the count!
- Expected: "2", Predicted: "Two projects: the Nigeria water project..." → F1=0
  Even though prediction is CORRECT (two = 2)!

New scorer:
1. Extract numbers from both prediction and gold (digits + word-numbers)
2. If gold is numeric: compare numbers directly (exact, 1.0; wrong, 0.0)
3. If gold is long/descriptive: token-overlap F1
4. LLM-as-judge: ask Claude if the prediction is correct (final answer)
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys


_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
}


def extract_numbers(text: str) -> list[int]:
    """Extract all numbers from text (digits + word-numbers)."""
    nums = []
    # Digits
    for m in re.finditer(r"\b(\d+)\b", text):
        try:
            nums.append(int(m.group(1)))
        except ValueError:
            pass
    # Word numbers
    lower = text.lower()
    for word, val in _WORD_NUMBERS.items():
        if re.search(rf"\b{word}\b", lower):
            nums.append(val)
    return nums


def is_numeric_question(gold: str) -> bool:
    """A gold answer is numeric if it's primarily a number/count."""
    g = gold.strip()
    if len(g) < 50 and extract_numbers(g):
        return True
    return False


def numeric_match(pred: str, gold: str) -> tuple[float, str]:
    """Check if pred contains the same number as gold. Returns (score, explanation)."""
    gold_nums = extract_numbers(gold)
    pred_nums = extract_numbers(pred)
    if not gold_nums:
        return 0.0, "no_gold_numbers"
    # Look for the FIRST number in gold (usually the canonical answer)
    target = gold_nums[0]
    if target in pred_nums:
        return 1.0, f"numeric_match({target})"
    return 0.0, f"numeric_mismatch(target={target} pred_nums={pred_nums})"


def token_f1(pred: str, gold: str) -> float:
    """Standard token-overlap F1."""
    def normalize(s):
        s = re.sub(r"[^\w\s]", "", s.lower()).strip()
        return [w for w in s.split() if w not in {"a", "an", "the", "is", "was", "are", "were"}]
    p = set(normalize(pred))
    g = set(normalize(gold))
    if not p or not g:
        return 0.0
    common = p & g
    if not common:
        return 0.0
    prec = len(common) / len(p)
    rec = len(common) / len(g)
    return 2 * prec * rec / (prec + rec)


def llm_judge(client, pred: str, gold: str, question: str) -> tuple[float, str]:
    """Use Claude as judge: is the prediction correct?"""
    import json as _json
    prompt = (
        "You are a strict answer grader. Judge if the PREDICTION matches the GOLD answer "
        "to the QUESTION. Reply with a single token: CORRECT or WRONG. No other text.\n\n"
        f"QUESTION: {question}\n"
        f"GOLD: {gold}\n"
        f"PREDICTION: {pred}\n\n"
        "Reply CORRECT or WRONG:"
    )
    body = _json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    })
    response = client.invoke_model(
        body=body,
        modelId=os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-sonnet-20241022-v2:0"),
        accept="application/json",
        contentType="application/json",
    )
    resp_body = _json.loads(response["body"].read())
    verdict = resp_body["content"][0]["text"].strip().upper()
    if "WRONG" in verdict or "INCORRECT" in verdict or "NOT CORRECT" in verdict:
        return 0.0, f"judge_wrong({verdict[:20]})"
    if "CORRECT" in verdict:
        return 1.0, "judge_correct"
    return 0.0, f"judge_wrong({verdict[:20]})"


def rescore(results_path: str, output_path: str, use_judge: bool = True):
    """Rescore all entries with proper metrics."""
    data = json.loads(open(results_path).read())
    details = data["details"]

    judge_client = None
    if use_judge:
        import boto3
        judge_client = boto3.client("bedrock-runtime", region_name="us-east-1")

    new_baseline = []
    new_treatment = []

    for i in range(len(details["baseline"])):
        b = details["baseline"][i]
        t = details["treatment"][i]
        q = b["q"]
        gold = b["gt"]
        is_numeric = is_numeric_question(gold)

        for arm, entry, out_list in [("baseline", b, new_baseline), ("treatment", t, new_treatment)]:
            ans = entry.get("ans", "")
            scores = {"old_f1": entry.get("f1", 0.0), "old_method": "token_overlap"}

            # Numeric match
            if is_numeric:
                nm, nm_expl = numeric_match(ans, gold)
                scores["numeric_match"] = nm
                scores["numeric_expl"] = nm_expl
            scores["token_f1"] = token_f1(ans, gold)

            # LLM judge
            if judge_client and ans:
                try:
                    j, j_expl = llm_judge(judge_client, ans, gold, q)
                    scores["judge"] = j
                    scores["judge_expl"] = j_expl
                except Exception as e:
                    scores["judge"] = 0.0
                    scores["judge_expl"] = f"error: {e}"

            entry["scores"] = scores
            out_list.append(entry)
        print(f"Q{i+1} numeric={is_numeric} | base.judge={new_baseline[i]['scores'].get('judge','?')} treat.judge={new_treatment[i]['scores'].get('judge','?')}", flush=True)

    # Aggregate
    n = len(new_baseline)
    def agg(arr, key):
        vals = [a["scores"].get(key, 0.0) for a in arr]
        return sum(vals) / n if n else 0.0

    summary = {
        "n": n,
        "baseline": {
            "token_f1": agg(new_baseline, "token_f1"),
            "numeric_match": agg(new_baseline, "numeric_match"),
            "judge_accuracy": agg(new_baseline, "judge"),
        },
        "treatment": {
            "token_f1": agg(new_treatment, "token_f1"),
            "numeric_match": agg(new_treatment, "numeric_match"),
            "judge_accuracy": agg(new_treatment, "judge"),
        },
    }
    summary["deltas"] = {
        "token_f1": summary["treatment"]["token_f1"] - summary["baseline"]["token_f1"],
        "numeric_match": summary["treatment"]["numeric_match"] - summary["baseline"]["numeric_match"],
        "judge_accuracy": summary["treatment"]["judge_accuracy"] - summary["baseline"]["judge_accuracy"],
    }

    print("\n" + "=" * 60)
    print(f"RESCORED RESULTS (n={n})")
    print("=" * 60)
    for arm in ("baseline", "treatment"):
        print(f"\n{arm.upper()}:")
        for k, v in summary[arm].items():
            print(f"  {k:20s}: {v:.3f}")
    print(f"\nDELTAS (treatment - baseline):")
    for k, v in summary["deltas"].items():
        print(f"  {k:20s}: {v:+.3f}")

    data["rescored"] = summary
    data["details"] = {"baseline": new_baseline, "treatment": new_treatment}
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[saved] {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--no-judge", action="store_true", help="Skip LLM judge (faster)")
    args = p.parse_args()
    output = args.output or args.input.replace(".json", "_rescored.json")
    rescore(args.input, output, use_judge=not args.no_judge)
