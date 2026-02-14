#!/usr/bin/env python3
"""
Clear pending uploads for a specific VOD
Useful for cleaning up stuck retries
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.upload_scheduler import UploadScheduler


def main():
    if len(sys.argv) < 2:
        print("Usage: python clear_vod_pending_uploads.py <vod_id> [--permanent]")
        print("\nOptions:")
        print("  --permanent    Mark as permanently failed (default: just remove)")
        sys.exit(1)
    
    vod_id = sys.argv[1]
    mark_failed = "--permanent" in sys.argv
    
    scheduler = UploadScheduler()
    
    # Show pending uploads for this VOD
    pending = scheduler.get_pending_for_vod(vod_id)
    
    if not pending:
        print(f"No pending uploads found for VOD {vod_id}")
        return
    
    print(f"Found {len(pending)} pending uploads for VOD {vod_id}:")
    for item in pending:
        arc_idx = item.get('arc_index')
        channel = item.get('channel')
        retry_count = item.get('retry_count', 0)
        reason = item.get('reason', 'unknown')[:80]
        print(f"  Arc {arc_idx} on {channel} (retry #{retry_count}): {reason}")
    
    # Confirm
    response = input(f"\nClear all {len(pending)} uploads? (yes/no): ").strip().lower()
    if response not in ('yes', 'y'):
        print("Cancelled")
        return
    
    # Clear
    count = scheduler.clear_vod_pending(vod_id, mark_as_failed=mark_failed)
    
    if mark_failed:
        print(f"✅ Cleared {count} uploads and marked as permanently failed")
    else:
        print(f"✅ Cleared {count} uploads")


if __name__ == "__main__":
    main()


