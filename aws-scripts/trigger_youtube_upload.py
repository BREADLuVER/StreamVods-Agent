#!/usr/bin/env python3
"""Manually enqueue a YouTube upload job for a processed VOD."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

from youtube_trigger import (
    build_youtube_trigger_payload,
    send_youtube_trigger,
    sync_metadata_to_s3,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trigger cloud YouTube upload")
    parser.add_argument("vod_id", help="Twitch VOD id that has finished local processing")
    parser.add_argument("--bucket", default=None, help="S3 bucket override (defaults to S3_BUCKET env)")
    parser.add_argument(
        "--queue-url",
        default=None,
        help="YouTube uploader SQS queue URL (defaults to YOUTUBE_UPLOAD_QUEUE_URL env)",
    )
    parser.add_argument("--skip-clips", action="store_true", help="Only queue Director's Cut upload")
    parser.add_argument("--dry-run", action="store_true", help="Print payload without sending to SQS")
    return parser.parse_args()


def resolve_bucket(arg_bucket: str | None) -> str:
    bucket = arg_bucket or os.getenv("S3_BUCKET")
    if bucket:
        return bucket
    print("X Missing S3 bucket (pass --bucket or set S3_BUCKET)")
    sys.exit(1)


def resolve_queue_url(arg_queue: str | None) -> str:
    queue_url = arg_queue or os.getenv("YOUTUBE_UPLOAD_QUEUE_URL")
    if queue_url:
        return queue_url
    print("X Missing queue URL (pass --queue-url or set YOUTUBE_UPLOAD_QUEUE_URL)")
    sys.exit(1)


def ensure_artifacts(payload: Dict[str, object]) -> None:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        print("X No artifacts discovered; ensure metadata files exist before triggering")
        sys.exit(1)


def main() -> None:
    args = parse_args()
    vod_id = args.vod_id
    include_clips = not args.skip_clips

    bucket = resolve_bucket(args.bucket)
    queue_url = resolve_queue_url(args.queue_url)

    metadata_keys = sync_metadata_to_s3(vod_id, bucket, include_clips=include_clips)
    payload = build_youtube_trigger_payload(
        vod_id,
        bucket,
        metadata_keys,
        include_clips=include_clips,
    )
    ensure_artifacts(payload)

    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    response = send_youtube_trigger(queue_url, payload)
    message_id = response.get("MessageId", "<unknown>")
    print(f"✅ Enqueued YouTube upload job for VOD {vod_id} (message {message_id})")


if __name__ == "__main__":
    main()

