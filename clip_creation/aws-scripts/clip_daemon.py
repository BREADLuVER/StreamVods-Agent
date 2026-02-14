#!/usr/bin/env python3
"""
Lightweight daemon that polls an SQS clip generation queue and launches the local GPU container
to process clip generation using run_gpu_clip_workflow.py.

Usage:
  python aws-scripts/clip_daemon.py --queue-url <SQS_URL>

Env (optional):
  AWS_REGION (default us-east-1)
  AWS_PROFILE (for local credentials via AWS CLI)
  ECR_REPO / AWS_ACCOUNT_ID (inherited by run_gpu_clip_workflow.py)
"""

import argparse
import json
import os
import subprocess
import sys
import time

import boto3
from botocore.exceptions import ClientError


def receive_one(sqs, queue_url: str):
    try:
        resp = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
            VisibilityTimeout=7200,  # 2 hours for clip generation
        )
        return resp.get('Messages', [])
    except ClientError as e:
        print(f"X SQS receive error: {e}")
        return []


def delete_msg(sqs, queue_url: str, receipt_handle: str):
    try:
        sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
    except ClientError as e:
        print(f"  SQS delete error: {e}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue-url', default=os.getenv('CLIP_QUEUE_URL'), help='SQS clip generation queue URL')
    parser.add_argument('--region', default=os.getenv('AWS_REGION', 'us-east-1'))
    args = parser.parse_args()

    if not args.queue_url:
        print('X Missing --queue-url or CLIP_QUEUE_URL')
        return 1

    sqs = boto3.client('sqs', region_name=args.region)

    try:
        # Ensure UTF-8 output on Windows consoles
        import sys
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print('🔄 Clip generation daemon started')
    print(f'🕸️  Queue: {args.queue_url}')

    while True:
        msgs = receive_one(sqs, args.queue_url)
        if not msgs:
            continue

        for msg in msgs:
            receipt = msg['ReceiptHandle']
            try:
                payload = json.loads(msg['Body'])
            except Exception:
                print('  Invalid message body, deleting')
                delete_msg(sqs, args.queue_url, receipt)
                continue

            vod_id = payload.get('vod_id')
            manifest_uri = payload.get('manifest')
            if not vod_id and not manifest_uri:
                print('  Message missing vod_id/manifest, deleting')
                delete_msg(sqs, args.queue_url, receipt)
                continue

            print(f" Received clip job: vod_id={vod_id} manifest={manifest_uri or ''}")

            cmd = [
                sys.executable,
                'clip_creation/run_gpu_clip_workflow.py',
            ]
            if manifest_uri:
                cmd.extend(['--from-manifest', manifest_uri])
            if vod_id:
                cmd.append(vod_id)

            try:
                subprocess.check_call(cmd)
                print(f" Clip generation completed: {vod_id or manifest_uri}")
                delete_msg(sqs, args.queue_url, receipt)
            except subprocess.CalledProcessError as e:
                print(f"X Clip generation failed: {e}")
                # Check if this is a permanent failure (e.g., missing files)
                # For now, we'll delete failed messages to prevent infinite retries
                # You can modify this logic based on your needs
                print(f"🗑️ Deleting failed message to prevent infinite retries")
                delete_msg(sqs, args.queue_url, receipt)

        # backoff between batches
        time.sleep(1)


if __name__ == '__main__':
    raise SystemExit(main())
