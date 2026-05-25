#!/usr/bin/env python3
"""Submit GRPO training job to Mako adhoc queue.

Uses training_toolkit.mako.MakoClient (Python SDK).

Usage:
    python gpu_submit.py [--dry-run] [--smoke-test]
"""

from __future__ import annotations

import argparse
import os
import sys


def submit(dry_run: bool = False, smoke_test: bool = False) -> str | None:
    from training_toolkit.mako.client import MakoClient
    from training_toolkit.mako.model import InstanceSpec, InstanceType, MakoJob, Priority

    output_dir = "~/rlm_grpo"

    if smoke_test:
        # 1-GPU sanity check: clone from GitHub (public), install deps, dry-run
        job = MakoJob(
            job_name="rlm-grpo-smoke",
            runtime_timeout="30m",
            priority=Priority.LOW,
            queue="root.mako.adhoc",
            instance_specs=[
                InstanceSpec(
                    entrypoint=(
                        "nvidia-smi && "
                        "git clone --depth=1 https://github.com/npow/context-bench.git && "
                        "cd context-bench && "
                        "pip install -q trl peft accelerate datasets tiktoken sentence-transformers lancedb duckdb && "
                        "PYTHONPATH=./src HF_HOME=~/.cache/huggingface "
                        "OPENAI_API_KEY=sk-dummy "
                        "python -m context_bench.training.grpo_train --dry-run && "
                        "echo SMOKE_PASSED"
                    ),
                    instance_num_gpus=1,
                )
            ],
        )
    else:
        # Full 4-GPU GRPO training run
        job = MakoJob(
            job_name="rlm-grpo-management-policy",
            runtime_timeout="12h",
            queue="root.mako.adhoc",
            instance_specs=[
                InstanceSpec(
                    entrypoint=(
                        "git clone --depth=1 https://github.com/npow/context-bench.git && "
                        "cd context-bench && "
                        f"pip install -q trl>=0.12.0 peft>=0.14.0 accelerate>=1.3.0 datasets tiktoken sentence-transformers lancedb duckdb bitsandbytes>=0.45.0 && "
                        f"PYTHONPATH=./src HF_HOME=~/.cache/huggingface "
                        f"TOKENIZERS_PARALLELISM=false OPENAI_API_KEY=sk-dummy "
                        f"CUDA_VISIBLE_DEVICES=0,1,2,3 "
                        f"python -m context_bench.training.grpo_train "
                        f"--output-dir {output_dir} "
                        f"--n-eval 2 "
                        f"--batch-size 2 "
                        f"--num-generations 4 "
                        f"--max-completion-length 256"
                    ),
                    instance_num_gpus=4,
                )
            ],
        )

    client = MakoClient(env="prod")

    print(f"[submit] Job: {job.job_name}", flush=True)
    spec = job.instance_specs[0]
    inst = getattr(spec.instance_type, 'value', 'auto') if spec.instance_type else 'auto'
    print(f"[submit] Instance: {inst} × {spec.instance_num_gpus} GPUs", flush=True)
    print(f"[submit] Output: {output_dir}", flush=True)

    if dry_run:
        print("[dry-run] Would submit. Pass --submit to proceed.", flush=True)
        return None

    submission_id = client.submit(job)
    print(f"[submit] Submitted: {submission_id}", flush=True)
    print(f"[submit] Monitor: python gpu_submit.py --status {submission_id}", flush=True)
    return submission_id


def status(submission_id: str) -> None:
    from training_toolkit.mako.client import MakoClient
    client = MakoClient(env="prod")
    info = client.get_job_info(submission_id)
    print(f"Status: {info.get('last_status')}")
    print(f"Submitter: {info.get('submitter_id')}")
    for name, url in (info.get("platform_links") or {}).items():
        print(f"  {name}: {url}")


def wait_and_print(submission_id: str) -> None:
    from training_toolkit.mako.client import MakoClient
    client = MakoClient(env="prod")
    print(f"[wait] Waiting for {submission_id}...", flush=True)
    final = client.wait(submission_id, timeout="12h", print_logs=True)
    print(f"[wait] Final status: {final}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke-test", action="store_true",
                   help="Submit a 1-GPU smoke test first")
    p.add_argument("--status", metavar="SUBMISSION_ID",
                   help="Check status of an existing job")
    p.add_argument("--wait", metavar="SUBMISSION_ID",
                   help="Wait for and stream logs of an existing job")
    args = p.parse_args()

    if args.status:
        status(args.status)
    elif args.wait:
        wait_and_print(args.wait)
    else:
        submit(dry_run=args.dry_run, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
