#!/usr/bin/env python3
"""
Lambda function to publish YouTube upload jobs to SQS when S3 objects are uploaded.
Standardised to the same SQS-based flow consumed by the youtube uploader daemon.
Handles both Director's Cut videos and clips uploads to YouTube.
"""

import json
import os
from typing import Dict, Any

from .youtube_trigger import build_youtube_trigger_payload, send_youtube_trigger

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Process S3 event and enqueue an SQS job for YouTube upload.

    Expected S3 events:
    - Director's Cut: s3://bucket/videos/{vod_id}/director_cut/{vod_id}_directors_cut.mp4
    - Clips: s3://bucket/clips/{vod_id}/*.mp4
    """

    queue_url = os.environ.get('YOUTUBE_UPLOAD_QUEUE_URL', '').strip()
    if not queue_url:
        raise RuntimeError("YOUTUBE_UPLOAD_QUEUE_URL env var is required for S3 trigger Lambda")

    for record in event.get('Records', []):
        try:
            # Parse S3 event
            bucket = record['s3']['bucket']['name']
            key = record['s3']['object']['key']
            
            print(f"Processing S3 event: s3://{bucket}/{key}")
            
            # Determine upload type and extract VOD ID
            vod_id, upload_type = parse_s3_key(key)
            if not vod_id:
                print(f"Could not extract VOD ID from key: {key}")
                continue
                
            print(f"Detected {upload_type} upload for VOD {vod_id}")
            
            # Build minimal payload; uploader daemon will generate metadata if missing
            metadata_keys: Dict[str, str] = {}
            include_clips = False
            if upload_type == 'clips':
                metadata_keys['clips_prefix'] = f"clips/{vod_id}/"
                include_clips = True
            else:
                # Director's Cut single file key
                metadata_keys['director_video_key'] = key

            payload = build_youtube_trigger_payload(
                vod_id=vod_id,
                bucket=bucket,
                metadata_keys=metadata_keys,
                include_clips=include_clips,
            )

            # Send to SQS (region optional)
            send_youtube_trigger(
                queue_url=queue_url,
                payload=payload,
                region=os.environ.get('AWS_REGION', 'us-east-1'),
            )
            print(f"Enqueued YouTube upload job for {upload_type} of VOD {vod_id}")
            
        except Exception as e:
            print(f"Error processing S3 event: {e}")
            continue
    
    return {
        'statusCode': 200,
        'body': json.dumps('YouTube upload jobs enqueued successfully')
    }

def parse_s3_key(key: str) -> tuple[str, str]:
    """
    Parse S3 key to extract VOD ID and upload type.
    
    Returns:
        (vod_id, upload_type) where upload_type is 'director_cut' or 'clips'
    """
    parts = key.split('/')
    
    # Director's Cut pattern: videos/{vod_id}/director_cut/{vod_id}_directors_cut.mp4
    if len(parts) >= 4 and parts[0] == 'videos' and parts[2] == 'director_cut':
        vod_id = parts[1]
        return vod_id, 'director_cut'
    
    # Clips pattern: clips/{vod_id}/*.mp4
    if len(parts) >= 2 and parts[0] == 'clips' and key.endswith('.mp4'):
        vod_id = parts[1]
        return vod_id, 'clips'
    
    return None, None
