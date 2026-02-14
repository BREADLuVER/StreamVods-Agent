#!/usr/bin/env python3
"""
Manual VOD Trigger Script for StreamSniped Orchestrator (FULL queue).

Enqueues a manual job to the FULL SQS queue so the GPU orchestrator
processes the VOD end-to-end locally.
"""

import boto3
import json
import os
import sys
from botocore.exceptions import ClientError

class ManualVODTrigger:
    """Handles manual triggering of VOD processing via SQS FULL queue."""

    def __init__(self, region: str = "us-east-1", queue_url: str | None = None):
        self.region = region
        self.sqs_client = boto3.client('sqs', region_name=region)
        self.queue_url = queue_url or os.getenv('FULL_QUEUE_URL')
        if not self.queue_url:
            raise RuntimeError("FULL_QUEUE_URL environment variable not set and no --queue-url provided")
        print(" Manual VOD Trigger (FULL Queue)")
        print(f"   Region: {self.region}")
        print(f"   Queue:  {self.queue_url}")

    def send_to_full_queue(self, vod_id: str, *, force: bool = False) -> bool:
        """Send a minimal message to FULL queue for the given VOD ID.
        If force=True, instruct the orchestrator to run regardless of prior status.
        """
        body = {"vod_id": vod_id, "source": "manual-trigger"}
        if force:
            body["force"] = True
        try:
            self.sqs_client.send_message(QueueUrl=self.queue_url, MessageBody=json.dumps(body))
            print(" Enqueued to FULL queue successfully")
            return True
        except ClientError as e:
            print(f"X AWS Error sending to SQS: {e}")
            return False
        except Exception as e:
            print(f"X Unexpected error: {e}")
            return False

    def trigger_vod_processing(self, vod_id: str, *, force: bool = False) -> bool:
        print(f"\n🚀 Enqueuing FULL job for VOD: {vod_id}")
        if force:
            print("   Force: enabled (bypass orchestrator duplicate/no-retry checks)")
        return self.send_to_full_queue(vod_id, force=force)

def main():
    """Main function to handle command line arguments and trigger processing."""
    
    # Usage: python trigger_manual_vod.py <VOD_ID> [--queue-url Q] [--region R] [--force]
    if len(sys.argv) < 2 or "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python trigger_manual_vod.py <VOD_ID> [--queue-url FULL_QUEUE_URL] [--region us-east-1] [--force]")
        print("Example: python trigger_manual_vod.py 2525805114")
        print("Adds a manual job to the FULL SQS queue for local processing.")
        sys.exit(0 if ("--help" in sys.argv or "-h" in sys.argv) else 1)
    force = "--force" in sys.argv
    
    vod_id = sys.argv[1]
    region = os.getenv('AWS_REGION', 'us-east-1')
    queue_url = None
    if "--region" in sys.argv:
        try:
            region = sys.argv[sys.argv.index("--region") + 1]
        except Exception:
            pass
    if "--queue-url" in sys.argv:
        try:
            queue_url = sys.argv[sys.argv.index("--queue-url") + 1]
        except Exception:
            queue_url = None
    
    # Validate VOD ID format (should be numeric)
    if not vod_id.isdigit():
        print(f"X Invalid VOD ID: {vod_id}. VOD IDs should be numeric.")
        sys.exit(1)
    
    print(" Manual VOD Processing Trigger → FULL Queue")
    print(f"   Target VOD ID: {vod_id}")
    print(f"   Region: {region}")
    if queue_url:
        print(f"   Queue:  {queue_url}")
    if force:
        print("   Force:  enabled")
    
    # Create and run the trigger
    try:
        trigger = ManualVODTrigger(region=region, queue_url=queue_url)
        success = trigger.trigger_vod_processing(vod_id, force=force)
        
        if success:
            print("\n🎉 Enqueued successfully. The GPU orchestrator will pick it up from the FULL queue.")
        else:
            print("\n❌ Manual trigger failed. Check the error messages above.")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error during trigger: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)