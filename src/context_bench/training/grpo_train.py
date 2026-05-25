#!/usr/bin/env python3
"""GRPO training to learn the memory management policy.

Trains Qwen2.5-7B-Instruct (LoRA) on LoCoMo to call memory_write when
answering questions. Reward = F1(answer, gt) + 0.2 * (memory_write called).

The pretrained baseline has 0% write adoption. After training we expect
>50% write adoption with no F1 regression on within-session tasks, and
improvement on cross-session tasks (LongMemEval multi-session split).

Usage:
    # Local dry-run (checks config, no actual training):
    python grpo_train.py --dry-run

    # Full training on GPU (4 GPUs):
    see gpu_submit.py (or equivalent cloud job script)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


# ---- Install dependencies if missing (Mako job startup) -----------------

def _install_if_missing() -> None:
    missing = []
    for pkg, import_name in [
        ("trl>=0.12.0", "trl"),
        ("peft>=0.14.0", "peft"),
        ("accelerate>=1.3.0", "accelerate"),
        ("bitsandbytes>=0.45.0", "bitsandbytes"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
    if not missing:
        return
    print(f"[setup] Installing: {missing}", flush=True)
    import subprocess
    # Try pip first; fall back to uv pip (Workbench venvs may not have pip)
    for pip_cmd in [
        [sys.executable, "-m", "pip", "install", "-q"],
        ["uv", "pip", "install", "-q"],
    ]:
        try:
            subprocess.check_call(pip_cmd + missing)
            print("[setup] Dependencies installed.", flush=True)
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    raise RuntimeError(f"Could not install {missing}. Install manually.")


_install_if_missing()


# ---- Imports (after dependency installation) ----------------------------

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from trl import GRPOConfig, GRPOTrainer

from context_bench.training.data import build_training_dataset, to_hf_dataset
from context_bench.training.reward import batch_reward_fn


# ---- Config -------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"  # Code-specialized 3B: better Python generation for REPL

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
)

GRPO_DEFAULTS = dict(
    num_generations=4,              # G — rollouts per prompt
    max_completion_length=64,       # short completions → faster; enough for REPL
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # smaller accum → more frequent updates
    num_train_epochs=5,             # more epochs → more signal from small dataset
    warmup_steps=5,                 # FIX: was 50 (> total steps). Now 5 = 10% of ~50 steps.
    beta=0.0,
    logging_steps=5,                # log more frequently
    save_steps=200,
    eval_steps=200,
    bf16=True,
    gradient_checkpointing=True,
    dataloader_num_workers=0,
    report_to="none",
    log_completions=True,
    num_completions_to_print=1,
)


# ---- Custom reward wrapper for trl GRPOTrainer --------------------------

def _make_reward_fn(eval_mode: bool = False):
    """Return a reward function compatible with trl ≥ 1.0 GRPOTrainer.

    The dataset is repeated G times by RepeatSampler, so all args are B*G length.
    Column names from the dataset are passed as kwargs.
    """
    return batch_reward_fn


# ---- Training -----------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Allow CLI override of MODEL_ID (used for v6+ to test different model sizes)
    global MODEL_ID
    if getattr(args, "model_id", None):
        MODEL_ID = args.model_id

    print(f"[train] output_dir={output_dir}", flush=True)
    print(f"[train] model={MODEL_ID}", flush=True)
    print(f"[train] n_train_convs={args.n_train}, n_eval_convs={args.n_eval}", flush=True)

    # ---- Data -----------------------------------------------------------
    print("[data] Building LoCoMo training dataset...", flush=True)
    t0 = time.perf_counter()
    train_examples, eval_examples = build_training_dataset(
        n_train=args.n_train,  # None = auto (all - n_eval)
        n_eval=args.n_eval,
        max_questions_per_conv=args.max_q_per_conv,
    )
    print(
        f"[data] {len(train_examples)} train, {len(eval_examples)} eval "
        f"({time.perf_counter()-t0:.1f}s)",
        flush=True,
    )

    train_ds = to_hf_dataset(train_examples)
    eval_ds = to_hf_dataset(eval_examples)

    if args.dry_run:
        print("[dry-run] Dataset OK. Exiting before model load.", flush=True)
        return

    # ---- Model + tokenizer ---------------------------------------------
    print("[model] Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        padding_side="left",  # for generation
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[model] Loading model (QLoRA 4-bit for T4/A100 compatibility)...", flush=True)
    # QLoRA: quantize base to 4-bit NF4, train LoRA adapters in bfloat16.
    # This halves VRAM for the base model weights (~14GB→~4GB), fitting T4.
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print("[model] QLoRA (4-bit NF4) loaded.", flush=True)
    except Exception as e:
        print(f"[model] QLoRA failed ({e}), falling back to bfloat16.", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    # QLoRA requires prepare_model_for_kbit_training before PEFT
    from peft import prepare_model_for_kbit_training
    try:
        model = prepare_model_for_kbit_training(model)
    except Exception:
        pass
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    # ---- GRPO trainer --------------------------------------------------
    grpo_kwargs = {**GRPO_DEFAULTS}
    grpo_kwargs.update({
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args.batch_size,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
    })

    config = GRPOConfig(**grpo_kwargs)
    reward_fn = _make_reward_fn()

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("[train] Starting GRPO training...", flush=True)
    t_start = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - t_start
    print(f"[train] Training complete in {elapsed/3600:.1f}h", flush=True)

    # ---- Save final checkpoint -----------------------------------------
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"[train] Saved to {final_path}", flush=True)

    # ---- Quick eval stats ----------------------------------------------
    _quick_eval(trainer, eval_examples, output_dir)

    # ---- Upload to S3 for reproducibility (uses cloud IAM role) --
    _save_to_s3(final_path)

    # ---- Full LanceDB eval using OpenAI-compatible relay --------------------------------
    relay = os.environ.get("RELAY_URL", "http://localhost:8080")
    if relay:
        _full_eval(str(final_path), relay, output_dir)


def _quick_eval(
    trainer,
    eval_examples: list[dict],
    output_dir: Path,
    sample_n: int = 50,
) -> None:
    """Run a quick sanity eval: adoption rate + mean F1 on a sample."""
    from context_bench.training.reward import compute_reward, BM25Store

    print(f"[eval] Quick eval on {sample_n} examples...", flush=True)
    import random
    random.seed(42)
    sample = random.sample(eval_examples, min(sample_n, len(eval_examples)))

    model = trainer.model
    tokenizer = trainer.processing_class
    model.eval()

    writes = 0
    f1_total = 0.0

    with torch.no_grad():
        for ex in sample:
            prompt = ex["prompt"]
            turns = ex["turns"]
            gt = ex["ground_truth"]

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(model.device)

            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
            code = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            reward, info = compute_reward(code, turns, gt)
            f1_total += info["f1"]
            if info["wrote"]:
                writes += 1

    n = len(sample)
    results = {
        "write_adoption_rate": writes / n,
        "mean_f1": f1_total / n,
        "n_evaluated": n,
    }
    print(f"[eval] Results: {json.dumps(results, indent=2)}", flush=True)

    out_path = output_dir / "quick_eval_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[eval] Saved to {out_path}", flush=True)
    return results


def _save_to_s3(checkpoint_path: Path) -> None:
    """Upload LoRA adapter to S3 using cloud IAM credentials (boto3)."""
    import subprocess
    s3_key = f"s3://YOUR_S3_BUCKET/rlm_grpo/rlm_grpo/{checkpoint_path.name}/"
    try:
        result = subprocess.run(
            ["aws", "s3", "cp", str(checkpoint_path), s3_key, "--recursive"],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            print(f"[s3] Uploaded to {s3_key}", flush=True)
        else:
            print(f"[s3] Upload failed: {result.stderr[:200]}", flush=True)
    except Exception as e:
        print(f"[s3] Upload error: {e}", flush=True)


def _full_eval(checkpoint: str, relay: str, output_dir: Path) -> None:
    """Run full LanceDB eval on trained model using an OpenAI-compatible relay."""
    print(f"[full_eval] Running on {checkpoint} via {relay}...", flush=True)
    try:
        from context_bench.training.eval_management_policy import (
            _load_trained_system, _run_locomo, _run_longmemeval, _aggregate
        )
        import json
        system = _load_trained_system(checkpoint)
        # Override _chat to use relay (instead of the local model's generate)
        # The _LocalModelSystem uses the loaded LoRA weights for generation

        locomo = _run_locomo(system, n=2)
        lme = _run_longmemeval(system, n=5)
        results = {
            "mode": "repl_trained_full_eval",
            "checkpoint": checkpoint,
            "locomo": locomo,
            "longmemeval": lme,
        }
        print(f"[full_eval] LoCoMo: F1={locomo.get('mean_f1',0):.3f} writes={locomo.get('write_adoption_rate',0):.1%}", flush=True)
        print(f"[full_eval] LME:    F1={lme.get('mean_f1',0):.3f} writes={lme.get('write_adoption_rate',0):.1%}", flush=True)
        out_path = output_dir / "full_eval_results.json"
        out_path.write_text(json.dumps(results, indent=2))
        print(f"[full_eval] Saved to {out_path}", flush=True)
    except Exception as e:
        print(f"[full_eval] Error: {e}", flush=True)
        import traceback
        traceback.print_exc()


# ---- Entry point --------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="/output/rlm_grpo")
    p.add_argument("--n-train", type=int, default=None,
                   help="Training conversations (default: all - n_eval)")
    p.add_argument("--n-eval", type=int, default=2)
    p.add_argument("--max-q-per-conv", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--max-completion-length", type=int, default=256)
    p.add_argument("--model-id", default=None,
                   help="Override MODEL_ID (e.g. Qwen/Qwen2.5-Coder-7B-Instruct)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
