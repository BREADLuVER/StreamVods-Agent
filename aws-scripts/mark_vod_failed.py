#!/usr/bin/env python3
"""
Mark a specific VOD as failed to prevent it from being re-queued.
"""

import boto3
import sys

def mark_vod_failed(vod_id: str):
    """Mark a VOD as failed in DynamoDB"""
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('streamsniped_jobs')
    
    try:
        # Update the VOD status to failed
        response = table.put_item(
            Item={
                'vod_id': vod_id,
                'status': 'failed',
                'streamer': 'manual-fix',
                'title': f'VOD {vod_id} (manually marked as failed)',
                'started_at': 0,
                'created_at': '2025-01-01T00:00:00Z',
                'failure_reason': 'Manually marked as failed to prevent infinite retries'
            }
        )
        print(f"✅ Marked VOD {vod_id} as failed")
        return True
    except Exception as e:
        print(f"❌ Error marking VOD as failed: {e}")
        return False

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python mark_vod_failed.py <vod_id>")
        sys.exit(1)
    
    vod_id = sys.argv[1]
    mark_vod_failed(vod_id)
