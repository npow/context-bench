"""Evaluate the management policy experiment: baseline vs RL-trained model.

Runs two experiments:
1. LoCoMo within-session: F1 + write adoption rate
   Hypothesis: trained ≈ baseline F1 (writes don't hurt same-session recall)

2. LongMemEval multi-session: F1 on knowledge_update + multi_session types
   Hypothesis: trained > baseline (writes persist across sessions, improving recall)

Usage:
    # Baseline (pretrained) — via OpenAI-compatible relay
    python -m context_bench.training.eval_management_policy \\
        --mode baseline \\
        --relay http://localhost:8080 \\
        --model claude-sonnet-4-6 \\
        --output baseline_results.json

    # Trained model (local LoRA checkpoint)
    python -m context_bench.training.eval_management_policy \\
        --mode trained \\
        --checkpoint ~/rlm_grpo/final \\
        --output trained_results.json

    # Compare two result files
    python -m context_bench.training.eval_management_policy \\
        --mode compare \\
        --baseline baseline_results.json \\
        --trained trained_results.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


# ---- Evaluation entry points --------------------------------------------

def eval_baseline(
    relay: str,
    model: str,
    n_locomo: int = 2,
    n_longmemeval: int = 20,
    use_static: bool = False,
) -> dict[str, Any]:
    """Evaluate baseline model via OpenAI-compatible relay.

    use_static=True: use RLMSystem (no REPL) for the oracle comparison
    use_static=False: use RLMSystemRepl (pretrained, with REPL) for the REPL baseline
    """
    if use_static:
        from context_bench.systems.rlm import RLMSystem
        from unittest.mock import patch, MagicMock
        import numpy as np

        def _make_embedder():
            emb = MagicMock()
            emb.get_sentence_embedding_dimension.return_value = 384
            rng = np.random.default_rng(42)
            # Use real embedder if available, fall back to random
            try:
                from sentence_transformers import SentenceTransformer
                real = SentenceTransformer("all-MiniLM-L6-v2")
                emb.encode.side_effect = lambda t: real.encode(t)
                emb.get_sentence_embedding_dimension.return_value = real.get_sentence_embedding_dimension()
            except Exception:
                emb.encode.side_effect = lambda t: rng.random(384).astype("float32")
            return emb

        system = RLMSystem(base_url=relay, model=model)
        results: dict[str, Any] = {}
        results["mode"] = "static_baseline"
    else:
        from context_bench.systems.rlm_repl import RLMSystemRepl
        system = RLMSystemRepl(
            base_url=relay,
            model=model,
            max_iterations=3,
        )
        results = {}
        results["mode"] = "repl_baseline"

    # LoCoMo within-session
    print(f"[eval] LoCoMo within-session (n={n_locomo})...", flush=True)
    results["locomo"] = _run_locomo(system, n=n_locomo)

    # LongMemEval multi-session
    print(f"[eval] LongMemEval (n={n_longmemeval})...", flush=True)
    results["longmemeval"] = _run_longmemeval(system, n=n_longmemeval)

    results["model"] = model
    return results


def eval_trained(
    checkpoint: str,
    n_locomo: int = 2,
    n_longmemeval: int = 20,
) -> dict[str, Any]:
    """Evaluate RL-trained LoRA model loaded from a checkpoint directory."""
    system = _load_trained_system(checkpoint)

    results: dict[str, Any] = {}

    print(f"[eval] LoCoMo within-session (n={n_locomo})...", flush=True)
    results["locomo"] = _run_locomo(system, n=n_locomo)

    print(f"[eval] LongMemEval multi-session (n={n_longmemeval})...", flush=True)
    results["longmemeval"] = _run_longmemeval(system, n=n_longmemeval)

    results["checkpoint"] = checkpoint
    results["mode"] = "trained"
    return results


# ---- Dataset runners ----------------------------------------------------

def _run_locomo(system, n: int) -> dict[str, Any]:
    from context_bench.datasets.memory.locomo import locomo

    examples = locomo(n=n)
    rows = []
    for ex in examples:
        try:
            system.reset()
            system.ingest(ex.items)
        except Exception as e:
            print(f"  [skip conv {ex.id}] ingest failed: {e}", flush=True)
            continue
        for q in ex.queries[:10]:
            if q.query_type == "adversarial":
                continue
            try:
                result = system.query(q.question)
            except Exception as e:
                print(f"  [skip q] query failed: {e}", flush=True)
                continue
            f1 = _f1(result.answer, q.answer)
            rows.append({
                "conv_id": ex.id,
                "query_type": q.query_type,
                "f1": f1,
                "answer": result.answer[:100],
                "expected": q.answer,
                "writes": result.details.get("writes", 0),
                "reads": result.details.get("reads", 0),
                "consolidations": result.details.get("consolidations", 0),
            })
            print(
                f"  [{q.query_type}] writes={result.details.get('writes',0)} "
                f"f1={f1:.2f} ans={result.answer[:30]!r} exp={q.answer!r}",
                flush=True,
            )

    return _aggregate(rows)


def _run_longmemeval(system, n: int) -> dict[str, Any]:
    from context_bench.datasets.memory.longmemeval import longmemeval

    examples = longmemeval(n=n, question_types=["multi-session", "knowledge-update", "single-session-user"])
    rows = []
    for ex in examples:
        try:
            system.reset()
            system.ingest(ex.items)
        except Exception as e:
            print(f"  [skip conv {ex.id}] ingest failed: {e}", flush=True)
            continue
        for q in ex.queries[:3]:
            try:
                result = system.query(q.question)
            except Exception as e:
                print(f"  [skip q] query failed: {e}", flush=True)
                continue
            f1 = _f1(result.answer, q.answer)
            rows.append({
                "conv_id": ex.id,
                "query_type": q.query_type,
                "f1": f1,
                "answer": result.answer[:100],
                "expected": q.answer,
                "writes": result.details.get("writes", 0),
                "reads": result.details.get("reads", 0),
            })
            print(
                f"  [{q.query_type}] writes={result.details.get('writes',0)} "
                f"f1={f1:.2f} ans={result.answer[:30]!r} exp={q.answer!r}",
                flush=True,
            )

    return _aggregate(rows)


def _aggregate(rows: list[dict]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    n = len(rows)
    f1_values = [r["f1"] for r in rows]
    writes = [r["writes"] for r in rows]
    by_type: dict[str, list[float]] = {}
    for r in rows:
        qt = r.get("query_type", "unknown")
        by_type.setdefault(qt, []).append(r["f1"])

    return {
        "n": n,
        "mean_f1": sum(f1_values) / n,
        "write_adoption_rate": sum(1 for w in writes if w > 0) / n,
        "mean_writes_per_query": sum(writes) / n,
        "f1_by_type": {qt: sum(vs) / len(vs) for qt, vs in by_type.items()},
    }


# ---- Load trained model as a system ------------------------------------

def _load_trained_system(checkpoint: str):
    """Load a LoRA checkpoint as an RLMSystemRepl-compatible system."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"[load] Loading trained model from {checkpoint}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    # Auto-detect base model from adapter config
    import json as _json
    from pathlib import Path as _Path
    adapter_cfg = _json.loads((_Path(checkpoint) / "adapter_config.json").read_text())
    base_model_id = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-3B-Instruct")
    print(f"[load] Base model: {base_model_id}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, checkpoint)
    model.eval()
    print("[load] Model loaded.", flush=True)

    return _LocalModelSystem(model, tokenizer)


class _LocalModelSystem:
    """Wraps a local Hugging Face model as an RLMSystemRepl-style system."""

    def __init__(self, model, tokenizer) -> None:
        import torch
        from context_bench.systems.rlm_repl import RLMSystemRepl
        from unittest.mock import patch

        # Build an RLMSystemRepl but override _chat to use local model
        with patch(
            "context_bench.systems.rlm.SentenceTransformer",
            return_value=_make_embedder(),
        ):
            self._repl = RLMSystemRepl(
                base_url="http://localhost:9999",  # unused
                model="local",
                max_iterations=3,
            )
        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch
        # Monkey-patch _chat to use local model
        self._repl._chat = self._local_chat

    @property
    def name(self) -> str:
        return "rlm_repl_trained"

    def reset(self) -> None:
        self._repl.reset()

    def ingest(self, items) -> None:
        self._repl.ingest(items)

    def query(self, question: str, budget=None):
        return self._repl.query(question, budget)

    def _local_chat(self, messages: list[dict]) -> str:
        import torch
        tok = self._tokenizer
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tok(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        return tok.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    def usage_stats(self) -> dict:
        return self._repl.usage_stats()


def _make_embedder():
    import numpy as np
    from unittest.mock import MagicMock
    emb = MagicMock()
    emb.get_sentence_embedding_dimension.return_value = 32
    emb.encode.side_effect = lambda text: np.random.default_rng(
        abs(hash(str(text))) % (2**32)
    ).random(32).astype("float32")
    return emb


# ---- Comparison ---------------------------------------------------------

def compare(baseline_path: str, trained_path: str) -> None:
    baseline = json.loads(Path(baseline_path).read_text())
    trained = json.loads(Path(trained_path).read_text())

    print("\n" + "=" * 70)
    print("RESULTS COMPARISON: Baseline vs RL-Trained Management Policy")
    print("=" * 70)

    for dataset in ["locomo", "longmemeval"]:
        b = baseline.get(dataset, {})
        t = trained.get(dataset, {})
        if not b or not t:
            continue

        print(f"\n{dataset.upper()}")
        print(f"  {'Metric':<30} {'Baseline':>12} {'Trained':>12} {'Delta':>10}")
        print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10}")

        for key in ["mean_f1", "write_adoption_rate", "mean_writes_per_query"]:
            bv = b.get(key, float("nan"))
            tv = t.get(key, float("nan"))
            delta = tv - bv if not (bv != bv or tv != tv) else float("nan")
            sign = "+" if delta > 0 else ""
            print(f"  {key:<30} {bv:>12.3f} {tv:>12.3f} {sign+f'{delta:.3f}':>10}")

        # F1 by query type
        b_types = b.get("f1_by_type", {})
        t_types = t.get("f1_by_type", {})
        all_types = sorted(set(b_types) | set(t_types))
        if all_types:
            print(f"\n  F1 by query type:")
            for qt in all_types:
                bv = b_types.get(qt, float("nan"))
                tv = t_types.get(qt, float("nan"))
                delta = tv - bv if not (bv != bv or tv != tv) else float("nan")
                sign = "+" if delta > 0 else ""
                print(f"    {qt:<28} {bv:>12.3f} {tv:>12.3f} {sign+f'{delta:.3f}':>10}")

    print("\n" + "=" * 70)


# ---- F1 helper ----------------------------------------------------------

def _f1(prediction: str, reference: str) -> float:
    import re
    from collections import Counter

    def normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = re.sub(r"[^a-z0-9 ]", "", text)
        return " ".join(text.split())

    p_tokens = normalize(prediction).split()
    r_tokens = normalize(reference).split()
    if not r_tokens:
        return 1.0
    if not p_tokens:
        return 0.0
    common = Counter(p_tokens) & Counter(r_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    prec = num_common / len(p_tokens)
    rec = num_common / len(r_tokens)
    return 2 * prec * rec / (prec + rec)


# ---- CLI ----------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "trained", "compare", "static"], required=True)
    p.add_argument("--static", action="store_true",
                   help="For --mode baseline: use RLMSystem (no REPL) as oracle comparison")
    p.add_argument("--relay", default="http://localhost:8080")
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--checkpoint", help="Path to LoRA checkpoint directory")
    p.add_argument("--output", help="Path to save result JSON")
    p.add_argument("--baseline", help="Baseline result JSON (for --mode compare)")
    p.add_argument("--trained", help="Trained result JSON (for --mode compare)")
    p.add_argument("--n-locomo", type=int, default=2)
    p.add_argument("--n-longmemeval", type=int, default=20)
    args = p.parse_args()

    if args.mode == "compare":
        compare(args.baseline, args.trained)
        return

    if args.mode in ("baseline", "static"):
        results = eval_baseline(
            relay=args.relay,
            model=args.model,
            n_locomo=args.n_locomo,
            n_longmemeval=args.n_longmemeval,
            use_static=(args.mode == "static"),
        )
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint required for --mode trained")
        results = eval_trained(
            checkpoint=args.checkpoint,
            n_locomo=args.n_locomo,
            n_longmemeval=args.n_longmemeval,
        )

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"[eval] Results saved to {args.output}", flush=True)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
