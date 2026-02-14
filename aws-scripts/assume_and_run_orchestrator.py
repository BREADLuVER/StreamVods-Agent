#!/usr/bin/env python3
import os
import sys
import json
import subprocess
from typing import Dict


def assume_role_and_export_env() -> None:
    role_arn = os.environ.get("AWS_ROLE_ARN")
    external_id = os.environ.get("AWS_EXTERNAL_ID")
    region = os.environ.get("AWS_REGION", "us-east-1")

    if not role_arn or not external_id:
        return  # assume we already have creds; run as-is

    cmd = [
        "aws",
        "sts",
        "assume-role",
        "--role-arn",
        role_arn,
        "--role-session-name",
        "vast-orchestrator",
        "--external-id",
        external_id,
        "--duration-seconds",
        "3600",
        "--region",
        region,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    creds = data["Credentials"]
    os.environ["AWS_ACCESS_KEY_ID"] = creds["AccessKeyId"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = creds["SecretAccessKey"]
    os.environ["AWS_SESSION_TOKEN"] = creds["SessionToken"]


def main() -> None:
    assume_role_and_export_env()

    clip_q = os.environ.get("CLIP_QUEUE_URL")
    render_q = os.environ.get("RENDER_QUEUE_URL")
    full_q = os.environ.get("FULL_QUEUE_URL")
    region = os.environ.get("AWS_REGION", "us-east-1")
    sleep_seconds = os.environ.get("ORCH_SLEEP_SECONDS", "20")

    if not clip_q or not render_q:
        print("Missing CLIP_QUEUE_URL or RENDER_QUEUE_URL", file=sys.stderr)
        sys.exit(2)

    args = [
        sys.executable,
        "aws-scripts/gpu_orchestrator_daemon.py",
        "--clip-queue-url",
        clip_q,
        "--render-queue-url",
        render_q,
        "--region",
        region,
        "--sleep-seconds",
        str(sleep_seconds),
    ]
    if full_q:
        args.extend(["--full-queue-url", full_q])

    # Stream output directly
    os.execv(sys.executable, args)


if __name__ == "__main__":
    main()


