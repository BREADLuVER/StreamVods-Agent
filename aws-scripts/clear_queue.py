#!/usr/bin/env python3
"""
Clear stuck VODs from the queue by marking them as failed.
Use this when queue is backed up and orchestrator is not processing.

Usage:
    python clear_queue.py                    # Show queued VODs
    python clear_queue.py --mark-failed     # Mark all queued as failed
    python clear_queue.py --mark-failed --vod 2597922237  # Mark specific VOD as failed
"""

import boto3
import argparse
from botocore.exceptions import ClientError
from datetime import datetime

DYNAMODB_TABLE = 'streamsniped_jobs'
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')


def get_queued_vods():
    """Get all VODs in queued status"""
    table = dynamodb.Table(DYNAMODB_TABLE)
    try:
        response = table.scan(
            FilterExpression='#s = :status',
            ExpressionAttributeNames={'#s': 'status'},
            ExpressionAttributeValues={':status': 'queued'}
        )
        return response.get('Items', [])
    except ClientError as e:
        print(f"X Error scanning table: {e}")
        return []


def mark_vod_failed(vod_id: str, reason: str = "Cleared from queue"):
    """Mark a VOD as failed"""
    table = dynamodb.Table(DYNAMODB_TABLE)
    try:
        table.update_item(
            Key={'vod_id': vod_id},
            UpdateExpression='SET #s = :status, #r = :reason, #t = :time',
            ExpressionAttributeNames={
                '#s': 'status',
                '#r': 'failure_reason',
                '#t': 'failed_at'
            },
            ExpressionAttributeValues={
                ':status': 'failed',
                ':reason': reason,
                ':time': int(datetime.now().timestamp())
            }
        )
        return True
    except ClientError as e:
        print(f"X Error marking VOD {vod_id} as failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Clear stuck VODs from queue',
        epilog='Examples:\n  python clear_queue.py\n  python clear_queue.py --mark-failed'
    )
    parser.add_argument('--mark-failed', action='store_true', help='Mark all queued VODs as failed')
    parser.add_argument('--vod', type=str, help='Specific VOD ID to mark as failed')
    
    args = parser.parse_args()
    
    if args.vod:
        # Mark specific VOD
        print(f"Marking VOD {args.vod} as failed...")
        if mark_vod_failed(args.vod):
            print(f"✓ VOD {args.vod} marked as failed")
        else:
            print(f"X Failed to mark VOD {args.vod}")
        return
    
    # Get all queued VODs
    vods = get_queued_vods()
    print(f"\n{'='*80}")
    print(f"📊 QUEUED VODS: {len(vods)} total")
    print(f"{'='*80}\n")
    
    if not vods:
        print("✓ No queued VODs found")
        return
    
    # Group by streamer
    by_streamer = {}
    for vod in vods:
        streamer = vod.get('streamer', 'unknown')
        if streamer not in by_streamer:
            by_streamer[streamer] = []
        by_streamer[streamer].append(vod)
    
    # Display
    for streamer, streamer_vods in sorted(by_streamer.items()):
        print(f"\n🎬 {streamer}: {len(streamer_vods)} VODs")
        for vod in streamer_vods[:3]:  # Show first 3
            vod_id = vod.get('vod_id')
            title = vod.get('title', 'N/A')[:50]
            started = vod.get('started_at', 0)
            age_hours = (int(datetime.now().timestamp()) - started) / 3600 if started else 0
            print(f"   [{vod_id}] {title} (queued {age_hours:.1f}h ago)")
        if len(streamer_vods) > 3:
            print(f"   ... and {len(streamer_vods) - 3} more")
    
    if not args.mark_failed:
        print(f"\n💡 To mark all as failed: python clear_queue.py --mark-failed")
        print(f"💡 To mark one as failed: python clear_queue.py --mark-failed --vod <vod_id>")
        return
    
    # Mark all as failed
    print(f"\n{'='*80}")
    print(f"🗑️  CLEARING QUEUE: Marking {len(vods)} VODs as failed")
    print(f"{'='*80}\n")
    
    failed_count = 0
    for vod in vods:
        vod_id = vod.get('vod_id')
        if mark_vod_failed(vod_id, "Cleared from queue - orchestrator backlog"):
            failed_count += 1
            print(f"✓ Marked {vod_id} as failed")
    
    print(f"\n{'='*80}")
    print(f"✓ CLEARED: {failed_count}/{len(vods)} VODs marked as failed")
    print(f"{'='*80}")
    print(f"\n💡 Next: Restart GPU orchestrator to pick up new VODs")
    print(f"   .\setup_gpu_orchestrator.ps1")


if __name__ == '__main__':
    main()
