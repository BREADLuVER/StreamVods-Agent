#!/usr/bin/env python3
"""
Helpers for enqueuing YouTube upload jobs after local processing.

This module centralises three responsibilities:
1. Synchronising locally generated metadata files to S3 so cloud workers can read them.
2. Building a consistent SQS message payload describing the available artifacts.
3. Publishing that payload to the configured YouTube upload queue.

The helpers are used both by the GPU orchestrator and by the manual
`trigger_youtube_upload.py` script so the hand-off logic lives in one place.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

import boto3

# Add parent directory to path so we can import storage module
sys.path.insert(0, str(Path(__file__).parent.parent))

try:  # pragma: no cover - botocore is optional in local mock testing
    from botocore.exceptions import ClientError
except ImportError:  # pragma: no cover - LocalStack/moto tests may skip botocore
    ClientError = Exception  # type: ignore[assignment]

try:
    from storage import StorageManager
except Exception:  # pragma: no cover - surfaced on use
    StorageManager = None  # type: ignore[assignment]

TRUTHY_VALUES = {"1", "true", "yes", "on"}
FALSY_VALUES = {"0", "false", "no", "off"}


def is_truthy(value: Optional[str], *, default: bool = False) -> bool:
    """Interpret common truthy/falsy strings with a configurable fallback."""

    if value is None:
        return default
    lowered = value.strip().lower()
    if not lowered:
        return default
    if lowered in TRUTHY_VALUES:
        return True
    if lowered in FALSY_VALUES:
        return False
    return default


def sync_metadata_to_s3(
    vod_id: str,
    bucket: str,
    *,
    include_clips: bool = True,
    storage: Optional[StorageManager] = None,
) -> Dict[str, str]:
    """
    Upload locally generated YouTube metadata artefacts to S3 and return their keys.

    Returns a mapping with the keys that were successfully uploaded. Missing files
    are skipped gracefully so callers can decide how to handle partial state.
    """

    if StorageManager is None:
        raise RuntimeError("StorageManager import failed; cannot sync metadata")

    manager = storage or StorageManager()
    uploaded: Dict[str, str] = {}

    ai_data_dir = Path("data/ai_data") / vod_id
    clips_dir = Path("data/clips") / vod_id

    director_meta = ai_data_dir / f"{vod_id}_youtube_metadata.json"
    if director_meta.exists():
        director_key = f"ai_data/{vod_id}/{director_meta.name}"
        s3_uri = f"s3://{bucket}/{director_key}"
        # Idempotent metadata upload
        if not manager.exists(s3_uri):
            manager.upload_file(str(director_meta), s3_uri)
        uploaded["director_metadata_key"] = director_key

    if include_clips:
        clip_index = ai_data_dir / f"{vod_id}_clip_metadata_index.json"
        if clip_index.exists():
            clip_index_key = f"ai_data/{vod_id}/{clip_index.name}"
            s3_uri = f"s3://{bucket}/{clip_index_key}"
            if not manager.exists(s3_uri):
                manager.upload_file(str(clip_index), s3_uri)
            uploaded["clip_metadata_index_key"] = clip_index_key

        clips_manifest = clips_dir / ".clips_manifest.json"
        if clips_manifest.exists():
            manifest_key = f"clips/{vod_id}/.clips_manifest.json"
            s3_uri = f"s3://{bucket}/{manifest_key}"
            if not manager.exists(s3_uri):
                manager.upload_file(str(clips_manifest), s3_uri)
            uploaded["clips_manifest_key"] = manifest_key

    # Always upload core AI data artifacts idempotently
    # This enables caching on retries without regenerating documents
    try:
        for json_file in ai_data_dir.glob(f"{vod_id}_*.json"):
            key = f"ai_data/{vod_id}/{json_file.name}"
            s3_uri = f"s3://{bucket}/{key}"
            if not manager.exists(s3_uri):
                manager.upload_file(str(json_file), s3_uri)
    except Exception:
        # Best-effort; don't fail the overall sync
        pass

        uploaded["clips_prefix"] = f"clips/{vod_id}/"

    # Add canonical video key so callers do not have to recompute it.
    uploaded.setdefault(
        "director_video_key",
        f"videos/{vod_id}/director_cut/{vod_id}_directors_cut.mp4",
    )

    return uploaded


def build_youtube_trigger_payload(
    vod_id: str,
    bucket: str,
    metadata_keys: Dict[str, str],
    *,
    include_clips: bool = True,
) -> Dict[str, object]:
    """Construct the SQS message body describing available upload artefacts."""

    artifacts: Dict[str, object] = {}

    director_metadata_key = metadata_keys.get("director_metadata_key")
    if director_metadata_key:
        artifacts["directorCut"] = {
            "bucket": bucket,
            "videoKey": metadata_keys.get(
                "director_video_key",
                f"videos/{vod_id}/director_cut/{vod_id}_directors_cut.mp4",
            ),
            "metadataKey": director_metadata_key,
        }

    if include_clips:
        clip_metadata_index_key = metadata_keys.get("clip_metadata_index_key")
        if clip_metadata_index_key:
            clip_payload = {
                "bucket": bucket,
                "metadataIndexKey": clip_metadata_index_key,
                "clipsPrefix": metadata_keys.get("clips_prefix", f"clips/{vod_id}/"),
            }
            manifest_key = metadata_keys.get("clips_manifest_key")
            if manifest_key:
                clip_payload["manifestKey"] = manifest_key
            artifacts["clips"] = clip_payload

    requested_at = datetime.utcnow().isoformat() + "Z"
    payload: Dict[str, object] = {
        "version": "2024-09-27",
        "vodId": vod_id,
        "requestedAt": requested_at,
        "artifacts": artifacts,
    }

    return payload


def send_youtube_trigger(
    queue_url: str,
    payload: Dict[str, object],
    *,
    region: Optional[str] = None,
) -> Dict[str, object]:
    """Publish the payload to the configured SQS queue."""

    sqs = boto3.client("sqs", region_name=region or os.getenv("AWS_REGION", "us-east-1"))

    message_kwargs: Dict[str, object] = {}
    if queue_url.endswith(".fifo"):
        vod_id = str(payload.get("vodId") or "unknown")
        message_kwargs["MessageGroupId"] = f"youtube-{vod_id}"
        message_kwargs["MessageDeduplicationId"] = str(uuid4())

    try:
        return sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(payload, ensure_ascii=False),
            **message_kwargs,
        )
    except ClientError as error:
        raise RuntimeError(f"Failed to send YouTube upload trigger: {error}") from error


__all__ = [
    "build_youtube_trigger_payload",
    "is_truthy",
    "send_youtube_trigger",
    "sync_metadata_to_s3",
]

