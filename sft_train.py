"""SFT distillation: train small model on Claude REPL traces.

Input: JSONL with `{"messages": [...]}` format (assistant message = code to predict)
Output: LoRA adapter for Qwen-Coder-7B that learned the RLM REPL+write pattern

Uses trl SFTTrainer for simple supervised next-token loss on assistant turns.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path


def _install_if_missing():
    missing = []
    for pkg, name in [
        ("trl>=0.12.0", "trl"),
        ("peft>=0.14.0", "peft"),
        ("accelerate>=1.3.0", "accelerate"),
        ("bitsandbytes>=0.45.0", "bitsandbytes"),
        ("datasets", "datasets"),
    ]:
        try: __import__(name)
        except ImportError: missing.append(pkg)
    if missing:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)


_install_if_missing()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True,
                   help="JSONL with messages (local path; if absent, downloaded from S3)")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--output-dir", default="/tmp/sft_model")
    p.add_argument("--output-name", default="distill_qwen",
                   help="S3 sub-folder name for the uploaded adapter")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--s3-bucket", default=None)
    args = p.parse_args()

    print(f"[sft] model={args.model_id} data={args.data}", flush=True)

    # Download input JSONL from S3 if not already present locally
    import subprocess as _sp
    s3_bucket = args.s3_bucket or os.environ.get("S3_BUCKET")
    if not os.path.exists(args.data) and s3_bucket:
        s3_src = f"s3://{s3_bucket}/distill/filtered_sft.jsonl"
        print(f"[sft] {args.data} not found locally; downloading from {s3_src}", flush=True)
        try:
            _sp.run(["aws", "s3", "cp", s3_src, args.data], check=True, timeout=300)
            print(f"[sft] downloaded to {args.data}", flush=True)
        except Exception as e:
            print(f"[sft] S3 download failed: {e}", flush=True)
            raise

    # Load data
    records = []
    with open(args.data) as f:
        for line in f:
            try: records.append(json.loads(line))
            except: pass
    print(f"[sft] {len(records)} training records", flush=True)

    if len(records) < 10:
        print("[sft] not enough data, aborting", flush=True)
        return

    # Convert to dataset
    ds = Dataset.from_list(records)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model (QLoRA 4-bit)
    print(f"[sft] loading model (4-bit NF4)...", flush=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # SFT config
    config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_seq_len,
        warmup_steps=5,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        save_total_limit=2,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print(f"[sft] starting training...", flush=True)
    t0 = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - t0
    print(f"[sft] complete in {elapsed/3600:.1f}h", flush=True)

    final_path = Path(args.output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"[sft] saved to {final_path}", flush=True)

    # Upload LoRA adapter to S3
    if s3_bucket:
        adapter_name = args.output_name if hasattr(args, "output_name") else Path(args.output_dir).name
        s3_key = f"s3://{s3_bucket}/sft_models/{adapter_name}/"
        try:
            _sp.run(["aws", "s3", "cp", str(final_path), s3_key, "--recursive"],
                    check=True, timeout=600)
            print(f"[s3] uploaded LoRA adapter to {s3_key}", flush=True)
        except Exception as e:
            print(f"[s3] upload failed: {e}", flush=True)


if __name__ == "__main__":
    main()
