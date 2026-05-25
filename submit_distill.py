"""Submit distillation pipeline jobs 2-4 to Mako.

Jobs:
  Job 2 — distill_filter.py   (CPU; lightweight; 1h)
  Job 3 — sft_train.py        (GPU + QLoRA; 12h)
  Job 4 — eval_student.py     (GPU + Bedrock; 4h)

Usage (run after Job 1 completes):
  python submit_distill.py --job 2          # submit filter
  python submit_distill.py --job 3          # submit SFT training
  python submit_distill.py --job 4          # submit eval
  python submit_distill.py --all            # submit all three sequentially (wait between each)
  python submit_distill.py --status <id>    # check job status

S3 artifacts (set via env vars or defaults):
  S3_BUCKET  = netflix.pi.prod/users/npow/rlm_grpo
  BEDROCK_MODEL_ID = us.anthropic.claude-sonnet-4-6

Mako image: mlp/mako_gpu_base_py310_cuda11:stable
"""
from __future__ import annotations
import argparse
import os
import sys

S3_BUCKET = os.environ.get("S3_BUCKET", "netflix.pi.prod/users/npow/rlm_grpo")
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
REPO_URL = "https://github.com/npow/context-bench.git"
BRANCH = "experiments/rl-and-multisession"
IMAGE = "mlp/mako_gpu_base_py310_cuda11:stable"

# Shared pip deps for all jobs
BASE_DEPS = (
    "boto3 huggingface_hub datasets tiktoken"
)

# Extra deps for GPU jobs
GPU_DEPS = (
    "trl>=0.12.0 peft>=0.14.0 accelerate>=1.3.0 "
    "bitsandbytes>=0.45.0 transformers>=4.40.0"
)

# Common env for every job
COMMON_ENV = (
    f"S3_BUCKET={S3_BUCKET} "
    f"BEDROCK_MODEL_ID={BEDROCK_MODEL_ID} "
    "HF_HOME=/tmp/.cache/huggingface "
    "TOKENIZERS_PARALLELISM=false "
    "TRANSFORMERS_OFFLINE=0 "
)

CLONE_CMD = (
    f"git clone --depth=1 --branch {BRANCH} {REPO_URL} repo && "
    "cd repo && "
)


def _client():
    from training_toolkit.mako.client import MakoClient
    return MakoClient(env="prod")


def _submit_job(job):
    from training_toolkit.mako.client import MakoClient
    client = MakoClient(env="prod")
    sid = client.submit(job)
    print(f"[submit] {job.job_name} -> {sid}", flush=True)
    return sid


def job2_filter() -> str:
    """distill_filter.py: download raw traces from S3, filter, upload filtered JSONL."""
    from training_toolkit.mako.model import InstanceSpec, MakoJob, Priority

    entrypoint = (
        f"{CLONE_CMD}"
        f"pip install -q {BASE_DEPS} && "
        f"PYTHONPATH=./src {COMMON_ENV} "
        "python distill_filter.py "
        "--input /tmp/distill_raw_traces.jsonl "
        "--output /tmp/distill_filtered_sft.jsonl "
        "--stats /tmp/distill_filter_stats.json"
    )
    job = MakoJob(
        job_name="distill-filter",
        runtime_timeout="1h",
        priority=Priority.LOW,
        queue="root.mako.adhoc",
        instance_specs=[
            InstanceSpec(
                entrypoint=entrypoint,
                instance_num_gpus=1,
                image=IMAGE,
            )
        ],
    )
    return _submit_job(job)


def job3_sft() -> str:
    """sft_train.py: download filtered JSONL, QLoRA fine-tune, upload adapter."""
    from training_toolkit.mako.model import InstanceSpec, MakoJob, Priority

    entrypoint = (
        f"{CLONE_CMD}"
        f"pip install -q {BASE_DEPS} {GPU_DEPS} && "
        f"PYTHONPATH=./src {COMMON_ENV} "
        # FIX: T4 16GB OOM with 4096 seq + 7B; reduce seq + switch to 3B model
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        "python sft_train.py "
        "--data /tmp/distill_filtered_sft.jsonl "
        "--output-dir /tmp/sft_distill_qwen "
        "--output-name distill_qwen "
        "--model-id Qwen/Qwen2.5-Coder-3B-Instruct "
        "--epochs 3 "
        "--lr 1e-4 "
        "--max-seq-len 2048 "
        "--batch-size 1 "
        "--grad-accum 8"
    )
    job = MakoJob(
        job_name="distill-sft-qwen",
        runtime_timeout="12h",
        queue="root.mako.adhoc",
        instance_specs=[
            InstanceSpec(
                entrypoint=entrypoint,
                instance_num_gpus=1,
                image=IMAGE,
            )
        ],
    )
    return _submit_job(job)


def job4_eval() -> str:
    """eval_student.py: download adapter, run student + teacher on 100 held-out Qs."""
    from training_toolkit.mako.model import InstanceSpec, MakoJob, Priority

    entrypoint = (
        f"{CLONE_CMD}"
        f"pip install -q {BASE_DEPS} {GPU_DEPS} && "
        f"PYTHONPATH=./src {COMMON_ENV} "
        "python eval_student.py "
        "--adapter-dir /tmp/distill_adapter "
        "--model-id Qwen/Qwen2.5-Coder-3B-Instruct "
        "--output /tmp/distill_eval_results.json"
    )
    job = MakoJob(
        job_name="distill-eval-student",
        runtime_timeout="4h",
        queue="root.mako.adhoc",
        instance_specs=[
            InstanceSpec(
                entrypoint=entrypoint,
                instance_num_gpus=1,
                image=IMAGE,
            )
        ],
    )
    return _submit_job(job)


def status(submission_id: str):
    client = _client()
    info = client.get_job_info(submission_id)
    print(f"status       : {info.get('last_status')}", flush=True)
    print(f"submitter_id : {info.get('submitter_id')}", flush=True)
    for name, url in (info.get("platform_links") or {}).items():
        print(f"  {name}: {url}", flush=True)


def wait_for(submission_id: str, timeout: str = "12h") -> str:
    client = _client()
    print(f"[wait] waiting for {submission_id} (timeout={timeout})...", flush=True)
    final = client.wait(submission_id, timeout=timeout, print_logs=True)
    print(f"[wait] final status: {final}", flush=True)
    return str(final)


def submit_all():
    """Submit Jobs 2-4 sequentially, waiting for each to complete before the next."""
    print("[pipeline] submitting Job 2 (filter)...", flush=True)
    sid2 = job2_filter()
    final2 = wait_for(sid2, timeout="2h")
    if "error" in final2.lower() or "failed" in final2.lower():
        print(f"[pipeline] Job 2 failed ({final2}). Aborting pipeline.", flush=True)
        sys.exit(1)

    print("[pipeline] submitting Job 3 (SFT training)...", flush=True)
    sid3 = job3_sft()
    final3 = wait_for(sid3, timeout="14h")
    if "error" in final3.lower() or "failed" in final3.lower():
        print(f"[pipeline] Job 3 failed ({final3}). Aborting pipeline.", flush=True)
        sys.exit(1)

    print("[pipeline] submitting Job 4 (eval)...", flush=True)
    sid4 = job4_eval()
    final4 = wait_for(sid4, timeout="6h")
    print(f"[pipeline] Job 4 final status: {final4}", flush=True)

    print("\n[pipeline] all jobs submitted and completed:", flush=True)
    print(f"  Job 2 (filter):  {sid2}", flush=True)
    print(f"  Job 3 (SFT):     {sid3}", flush=True)
    print(f"  Job 4 (eval):    {sid4}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Submit distillation pipeline jobs 2-4 to Mako")
    p.add_argument("--job", type=int, choices=[2, 3, 4], help="Submit a single job")
    p.add_argument("--all", action="store_true", help="Submit all jobs sequentially")
    p.add_argument("--status", metavar="SUBMISSION_ID", help="Check job status")
    p.add_argument("--wait", metavar="SUBMISSION_ID", help="Wait for a job and stream logs")
    args = p.parse_args()

    if args.status:
        status(args.status)
    elif args.wait:
        wait_for(args.wait)
    elif args.all:
        submit_all()
    elif args.job == 2:
        job2_filter()
    elif args.job == 3:
        job3_sft()
    elif args.job == 4:
        job4_eval()
    else:
        p.print_help()


if __name__ == "__main__":
    main()
