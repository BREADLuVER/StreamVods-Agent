#!/usr/bin/env python3
"""
One-time recovery script to backfill missing VODs.
Use this after AWS outages or DynamoDB issues to recover VODs from session gaps.

Usage:
    python recover_missing_vods.py
    python recover_missing_vods.py --streamer masayoshi --max-vods 20
    python recover_missing_vods.py --all
"""

import boto3
import os
import sys
import time
import argparse
from datetime import datetime
from botocore.exceptions import ClientError

# Try to import TwitchMonitor
try:
    from twitch_monitor import TwitchMonitor
except ImportError:
    print("X TwitchMonitor not available. Make sure requests is installed.")
    sys.exit(1)

# Configuration
DYNAMODB_TABLE = os.getenv('DYNAMODB_TABLE', 'streamsniped_jobs')
STREAMERS_STR = os.getenv('STREAMERS', 'disguisedtoast,quarterjade,masayoshi')
STREAMERS_DEFAULT = [s.strip() for s in STREAMERS_STR.split(',')]

dynamodb = boto3.resource('dynamodb')


def get_streamer_id(twitch_monitor, streamer_name: str) -> str:
    """Get Twitch user ID for a streamer"""
    try:
        user_id = twitch_monitor.get_user_id(streamer_name)
        if user_id:
            return user_id
        else:
            print(f"X Could not find Twitch user ID for {streamer_name}")
            return None
    except Exception as e:
        print(f"X Error getting user ID for {streamer_name}: {e}")
        return None


def backfill_missing_vods(user_id: str, streamer_name: str, max_vods: int = 10) -> int:
    """
    Backfill missing VODs for a streamer.
    
    Args:
        user_id: Twitch user ID
        streamer_name: Streamer name for logging
        max_vods: Maximum number of VODs to backfill
    
    Returns:
        Number of VODs backfilled
    """
    print(f"\n{'='*80}")
    print(f"🔄 RECOVERY: Backfilling missing VODs for {streamer_name} (user_id={user_id})")
    print(f"{'='*80}")
    
    try:
        twitch_monitor = TwitchMonitor()
        job_table = dynamodb.Table(DYNAMODB_TABLE)
        backfilled_count = 0
        skipped_count = 0
        
        # Get last 30 VODs from Twitch
        print(f"📊 Fetching recent VODs from Twitch...")
        vods = twitch_monitor.get_recent_vods(user_id, limit=30)
        if not vods:
            print(f"ℹ️  No VODs found for {streamer_name}")
            return 0
        
        print(f"Found {len(vods)} recent VODs from Twitch")
        
        for idx, vod in enumerate(vods, 1):
            if backfilled_count >= max_vods:
                print(f"\n✓ Reached backfill limit ({max_vods})")
                break
            
            vod_id = vod.get('id')
            created_at = vod.get('created_at')
            title = vod.get('title', f'VOD {vod_id}')
            duration = vod.get('duration', 'unknown')
            
            if not vod_id:
                continue
            
            print(f"\n[{idx}] VOD {vod_id}: {title} ({duration})")
            
            # Check if VOD already exists in job table
            try:
                response = job_table.get_item(Key={'vod_id': vod_id})
                if 'Item' in response:
                    status = response['Item'].get('status')
                    print(f"    ⊘ Already in job table (status={status})")
                    skipped_count += 1
                    continue
            except ClientError as e:
                print(f"    X Error checking status: {e}")
                continue
            
            # Check VOD age (skip if too old)
            try:
                created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                age_days = (datetime.now(created_time.tzinfo) - created_time).days
                if age_days > 7:
                    print(f"    ⊘ Too old ({age_days} days). Skipping.")
                    skipped_count += 1
                    continue
                print(f"    📅 Age: {age_days} days (created: {created_at})")
            except Exception as e:
                print(f"    ⚠️  Could not parse date {created_at}: {e}")
            
            # Backfill this VOD
            try:
                job_table.put_item(
                    Item={
                        'vod_id': vod_id,
                        'status': 'queued',
                        'streamer': streamer_name,
                        'title': title,
                        'started_at': int(time.time()),
                        'created_at': created_at,
                        'backfilled': True,
                        'backfill_reason': 'Session gap recovery'
                    }
                )
                print(f"    ✓ Backfilled to job queue")
                backfilled_count += 1
            except ClientError as e:
                print(f"    X Error backfilling: {e}")
        
        print(f"\n{'='*80}")
        print(f"BACKFILL SUMMARY for {streamer_name}:")
        print(f"  ✓ Backfilled: {backfilled_count}")
        print(f"  ⊘ Skipped: {skipped_count}")
        print(f"  📝 Total checked: {len(vods)}")
        print(f"{'='*80}\n")
        
        return backfilled_count
        
    except Exception as e:
        print(f"X Error in backfill_missing_vods: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    parser = argparse.ArgumentParser(
        description='Backfill missing VODs for StreamSniped',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python recover_missing_vods.py                    # All default streamers
  python recover_missing_vods.py --streamer masayoshi --max-vods 20
  python recover_missing_vods.py --all --max-vods 30
        '''
    )
    parser.add_argument(
        '--streamer',
        type=str,
        help='Specific streamer to backfill (e.g., masayoshi)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Backfill all default streamers'
    )
    parser.add_argument(
        '--max-vods',
        type=int,
        default=10,
        help='Maximum VODs to backfill per streamer (default: 10)'
    )
    
    args = parser.parse_args()
    
    twitch_monitor = TwitchMonitor()
    total_backfilled = 0
    
    if args.streamer:
        # Single streamer
        streamers = [args.streamer]
    elif args.all:
        # All streamers
        streamers = STREAMERS_DEFAULT
    else:
        # Default streamers
        streamers = STREAMERS_DEFAULT
    
    print(f"\n{'='*80}")
    print(f"🚀 VOD RECOVERY TOOL")
    print(f"{'='*80}")
    print(f"Streamers to process: {', '.join(streamers)}")
    print(f"Max VODs per streamer: {args.max_vods}")
    print(f"DynamoDB table: {DYNAMODB_TABLE}")
    print(f"{'='*80}\n")
    
    for streamer_name in streamers:
        user_id = get_streamer_id(twitch_monitor, streamer_name)
        if user_id:
            count = backfill_missing_vods(user_id, streamer_name, max_vods=args.max_vods)
            total_backfilled += count
        else:
            print(f"⊘ Skipping {streamer_name} (user ID not found)\n")
    
    print(f"\n{'='*80}")
    print(f"✓ RECOVERY COMPLETE")
    print(f"Total VODs backfilled: {total_backfilled}")
    print(f"{'='*80}")
    print(f"\n💡 Next steps:")
    print(f"   1. Monitor CloudWatch logs for the Twitch Watcher Lambda")
    print(f"   2. Wait for the GPU orchestrator to pick up the backfilled VODs")
    print(f"   3. Check S3 for newly generated clips and archives")
    print(f"\n")


if __name__ == '__main__':
    main()
