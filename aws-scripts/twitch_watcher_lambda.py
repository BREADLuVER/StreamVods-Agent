#!/usr/bin/env python3
"""
AWS Lambda function for watching Twitch VODs.

This function is triggered by an EventBridge rule. It checks for new VODs
from a predefined list of streamers, and for each new VOD, it launches an
ECS task to process it. It uses DynamoDB to track which VODs have already
been processed.
"""

import boto3
import os
import time
import json
from botocore.exceptions import ClientError

def get_aws_account_id():
    """Get AWS account ID"""
    try:
        sts = boto3.client('sts')
        return sts.get_caller_identity()['Account']
    except Exception:
        account_id = os.getenv('AWS_ACCOUNT_ID')
        if not account_id:
            raise RuntimeError("AWS_ACCOUNT_ID environment variable not set")
        return account_id

# Import TwitchMonitor only when needed (avoid requests dependency for manual triggers)
TwitchMonitor = None
try:
    from twitch_monitor import TwitchMonitor
except ImportError:
    # Allow function to work without TwitchMonitor for manual triggers
    TwitchMonitor = None

# Environment variables
ECS_CLUSTER = os.getenv('ECS_CLUSTER', 'streamsniped-dev-cluster')
ECS_TASK_DEFINITION = os.getenv('ECS_TASK_DEFINITION', 'streamsniped-fargate')  # Fargate default
ECS_TASK_DEFINITION_GPU = os.getenv('ECS_TASK_DEFINITION_GPU', 'streamsniped-fargate')  # Same task def
GPU_MODE = os.getenv('GPU_MODE', 'true').lower() == 'true'  # Default: GPU-enabled
ECS_SUBNETS = os.getenv('ECS_SUBNETS', '').split(',')
ECS_SECURITY_GROUPS = os.getenv('ECS_SECURITY_GROUPS', '').split(',')
DYNAMODB_TABLE = os.getenv('DYNAMODB_TABLE', 'streamsniped_jobs')
STREAMERS = os.getenv('STREAMERS', 'disguisedtoast,quarterjade,sydeon,scarra,yvonnie').split(',')
STATE_TABLE_NAME = os.getenv('STATE_TABLE_NAME', 'streamsniped-watcher-state-dev')
SESSION_TABLE_NAME = os.getenv('SESSION_TABLE_NAME', 'streamsniped-sessions-dev')

# GPU System Configuration
GPU_ENABLED = os.getenv('GPU_ENABLED', 'true').lower() == 'true'
USE_GPU_FOR_ALL_STEPS = os.getenv('USE_GPU_FOR_ALL_STEPS', 'true').lower() == 'true'
GPU_PROCESSING_MODE = os.getenv('GPU_PROCESSING_MODE', 'hybrid')  # hybrid, local_only, cloud_only
FULL_QUEUE_URL = os.getenv('FULL_QUEUE_URL', '')

ecs = boto3.client('ecs')
dynamodb = boto3.resource('dynamodb')

# Session state constants
SESSION_STATUS_PENDING = 'PENDING'
SESSION_STATUS_RECORDING = 'RECORDING'
SESSION_STATUS_FINALIZING = 'FINALIZING'
SESSION_STATUS_READY = 'READY'

# Stabilization params (can be tuned via env in future)
FINALIZE_CHECKS_REQUIRED = int(os.getenv('FINALIZE_CHECKS_REQUIRED', '2'))  # consecutive checks with no growth
FINALIZE_GRACE_SECONDS = int(os.getenv('FINALIZE_GRACE_SECONDS', str(25 * 60)))  # 25 min grace after last seen live

def log_gpu_system_status():
    """Log comprehensive GPU system status and configuration"""
    print("=" * 80)
    print("🎮 GPU SYSTEM STATUS & CONFIGURATION")
    print("=" * 80)
    print(f"🔧 GPU Mode Enabled: {GPU_MODE}")
    print(f"🚀 GPU Processing Enabled: {GPU_ENABLED}")
    print(f"⚡ Use GPU for All Steps: {USE_GPU_FOR_ALL_STEPS}")
    print(f"🌐 GPU Processing Mode: {GPU_PROCESSING_MODE}")
    print(f" ECS Task Definition: {ECS_TASK_DEFINITION}")
    print(f" GPU Task Definition: {ECS_TASK_DEFINITION_GPU}")
    print(f"🏗️  Architecture: {'Hybrid (Cloud CPU + Local GPU)' if GPU_PROCESSING_MODE == 'hybrid' else 'Local GPU Only' if GPU_PROCESSING_MODE == 'local_only' else 'Cloud GPU Only'}")
    print("=" * 80)
    
    # Log environment variables
    print("📊 GPU Environment Variables:")
    print(f"   GPU_MODE: {os.getenv('GPU_MODE', 'not_set')}")
    print(f"   GPU_ENABLED: {os.getenv('GPU_ENABLED', 'not_set')}")
    print(f"   USE_GPU_FOR_ALL_STEPS: {os.getenv('USE_GPU_FOR_ALL_STEPS', 'not_set')}")
    print(f"   GPU_PROCESSING_MODE: {os.getenv('GPU_PROCESSING_MODE', 'not_set')}")
    print(f"   NVIDIA_VISIBLE_DEVICES: {os.getenv('NVIDIA_VISIBLE_DEVICES', 'not_set')}")
    print("=" * 80)

def handle_manual_trigger(event, context):
    """
    Handles manual trigger events that force processing of a specific VOD.
    Bypasses all duplicate checks and time constraints.
    """
    detail = event.get("detail", {})
    vod_id = detail.get("vod_id")
    vod_details = detail.get("vod_details", {})
    
    print(f" MANUAL TRIGGER: Processing VOD {vod_id}")
    print(f"   Title: {vod_details.get('title', 'N/A')}")
    print(f"   Creator: {vod_details.get('user_name', 'N/A')}")
    print(f"   Bypassing: duplicates={detail.get('bypass_duplicates')}, time_checks={detail.get('bypass_time_checks')}")
    print(f"   Reason: {detail.get('trigger_reason', 'Manual trigger')}")
    
    # Log GPU system status for manual triggers
    log_gpu_system_status()
    
    if not vod_id:
        error_msg = "X Manual trigger missing vod_id in event detail"
        print(error_msg)
        return {'statusCode': 400, 'body': error_msg}
    
    job_table = dynamodb.Table(DYNAMODB_TABLE)
    streamer_name = vod_details.get('user_name', 'manual-trigger')
    
    # For manual triggers, we force the VOD structure to match what process_vod expects
    manual_vod = {
        'id': vod_id,
        'title': vod_details.get('title', f'Manual VOD {vod_id}'),
        'created_at': vod_details.get('created_at', '2025-01-01T00:00:00Z'),
        'url': vod_details.get('url', f'https://www.twitch.tv/videos/{vod_id}'),
        'user_name': streamer_name
    }
    
    print(f"🚀 Force launching ECS task for VOD {vod_id} (manual trigger)")
    
    # Skip all duplicate checking and directly launch the task
    success = force_process_vod(manual_vod, job_table, streamer_name, manual=True)
    
    if success:
        success_msg = f" Manual trigger successful: VOD {vod_id} processing started"
        print(success_msg)
        return {
            'statusCode': 200, 
            'body': success_msg,
            'vod_id': vod_id,
            'streamer': streamer_name,
            'trigger_type': 'manual',
            'gpu_enabled': GPU_ENABLED,
            'gpu_mode': GPU_MODE,
            'processing_mode': GPU_PROCESSING_MODE
        }
    else:
        error_msg = f"X Manual trigger failed: VOD {vod_id} could not be processed"
        print(error_msg)
        return {
            'statusCode': 500, 
            'body': error_msg,
            'vod_id': vod_id,
            'trigger_type': 'manual',
            'gpu_enabled': GPU_ENABLED,
            'gpu_mode': GPU_MODE
        }

def lambda_handler(event, context):
    """
    Main Lambda handler.
    Supports both scheduled events and manual trigger events.
    """
    print(f"Lambda triggered with event: {json.dumps(event, default=str)}")
    
    # Log GPU system status at startup
    log_gpu_system_status()
    
    # Check if this is a manual trigger event
    if event.get("source") == "manual-trigger" and event.get("detail", {}).get("force_process"):
        return handle_manual_trigger(event, context)
    
    print("Starting stateful VOD check...")
    
    # Check if TwitchMonitor is available
    if TwitchMonitor is None:
        error_msg = "X TwitchMonitor not available (missing requests dependency)"
        print(error_msg)
        return {'statusCode': 500, 'body': error_msg}
    
    twitch_monitor = TwitchMonitor()
    job_table = dynamodb.Table(DYNAMODB_TABLE)
    state_table = dynamodb.Table(STATE_TABLE_NAME)
    session_table = dynamodb.Table(SESSION_TABLE_NAME)

    for streamer_name in STREAMERS:
        print(f"Checking streamer: {streamer_name}")
        user_id = twitch_monitor.get_user_id(streamer_name)
        if not user_id:
            print(f"Could not find user ID for {streamer_name}")
            continue

        # Session-aware flow
        manage_stream_session(
            twitch_monitor=twitch_monitor,
            session_table=session_table,
            job_table=job_table,
            state_table=state_table,
            user_id=user_id,
            streamer_name=streamer_name
        )

    print("VOD check finished.")
    return {'statusCode': 200, 'body': 'Stateful VOD check finished successfully.'}

def manage_stream_session(twitch_monitor, session_table, job_table, state_table, user_id: str, streamer_name: str) -> None:
    """
    Maintain a per-streamer session to avoid triggering on baby VODs.
    Transitions: PENDING/RECORDING -> FINALIZING -> READY -> COMPLETED (delete)
    
    Enhanced with:
    - Session timeout watchdog (prevents FINALIZING lockup)
    - VOD-level deduplication (catches VODs from session gaps)
    - Robust session deletion with TTL fallback
    """
    now_ts = int(time.time())

    # Load current session if exists
    session = get_session(session_table, user_id)

    # Check live status
    stream = twitch_monitor.get_stream(user_id)
    is_live = stream is not None
    stream_id = stream.get('id') if is_live else None

    # If live now
    if is_live:
        # Ensure session exists and reflects current stream
        if not session or session.get('stream_id') != str(stream_id):
            print(f" New live session for {streamer_name} (stream_id={stream_id}). Enter RECORDING.")
            session = {
                'streamer_id': user_id,
                'stream_id': str(stream_id),
                'vod_id': '',
                'status': SESSION_STATUS_RECORDING,
                'created_at': now_ts,
                'last_seen_live_at': now_ts,
                'last_checked_at': now_ts,
                'last_vod_duration_seconds': 0,
                'stable_checks': 0
            }
            put_session(session_table, session)
        else:
            # Update heartbeat and keep recording
            session['status'] = SESSION_STATUS_RECORDING
            session['last_seen_live_at'] = now_ts
            session['last_checked_at'] = now_ts
            put_session(session_table, session)

        # Refresh latest archive VOD and track duration growth
        latest_vod = twitch_monitor.get_latest_archive_video(user_id)
        if latest_vod:
            vod_id = latest_vod.get('id')
            dur_str = latest_vod.get('duration')
            dur_sec = twitch_monitor.parse_duration_to_seconds(dur_str)
            if vod_id and (not session.get('vod_id') or session.get('vod_id') == vod_id):
                if dur_sec > session.get('last_vod_duration_seconds', 0):
                    session['vod_id'] = vod_id
                    session['last_vod_duration_seconds'] = dur_sec
                    session['stable_checks'] = 0
                    put_session(session_table, session)
                    print(f"⏱️ Recording {streamer_name}: VOD {vod_id} grew to {dur_sec}s")
        return

    # Not live now
    if not session:
        print(f"ℹ️ {streamer_name} not live and no session found.")
        return

    # Start finalization window if recording
    if session.get('status') in [SESSION_STATUS_PENDING, SESSION_STATUS_RECORDING]:
        session['status'] = SESSION_STATUS_FINALIZING
        session['last_checked_at'] = now_ts
        put_session(session_table, session)
        print(f"🧵 {streamer_name} moved to FINALIZING.")

    # In FINALIZING, check stabilization
    if session.get('status') == SESSION_STATUS_FINALIZING:
        vod_id = session.get('vod_id')
        
        # WATCHDOG: If stuck in FINALIZING for >1 hour, force to READY
        time_in_finalizing = now_ts - session.get('created_at', now_ts)
        if time_in_finalizing > 3600:
            print(f"⚠️  {streamer_name} stuck in FINALIZING for {time_in_finalizing}s (>1h). Force to READY.")
            session['status'] = SESSION_STATUS_READY
            session['last_checked_at'] = now_ts
            put_session(session_table, session)
            # Continue to trigger below
        elif vod_id:
            vid = twitch_monitor.get_video_by_id(vod_id)
            if vid:
                dur_sec = twitch_monitor.parse_duration_to_seconds(vid.get('duration'))
                if dur_sec > session.get('last_vod_duration_seconds', 0):
                    # Growth detected → back to RECORDING state machine (but streamer is offline → twitch still writing)
                    session['status'] = SESSION_STATUS_RECORDING
                    session['last_vod_duration_seconds'] = dur_sec
                    session['stable_checks'] = 0
                    session['last_checked_at'] = now_ts
                    put_session(session_table, session)
                    print(f"↩️ {streamer_name} VOD still growing ({dur_sec}s). Back to RECORDING.")
                    return
                else:
                    session['stable_checks'] = int(session.get('stable_checks', 0)) + 1
                    session['last_checked_at'] = now_ts
                    put_session(session_table, session)
                    print(f"📏 {streamer_name} stabilization check {session['stable_checks']}/{FINALIZE_CHECKS_REQUIRED}")

            # Evaluate grace window as well
            last_live = int(session.get('last_seen_live_at', now_ts))
            grace_elapsed = now_ts - last_live
            if session.get('stable_checks', 0) >= FINALIZE_CHECKS_REQUIRED or grace_elapsed >= FINALIZE_GRACE_SECONDS:
                session['status'] = SESSION_STATUS_READY
                session['last_checked_at'] = now_ts
                put_session(session_table, session)
                print(f" {streamer_name} session READY (vod_id={session.get('vod_id')})")
            else:
                # Not ready yet, return early
                return

    # When READY, trigger exactly once and clear session
    if session.get('status') == SESSION_STATUS_READY:
        vod_id = session.get('vod_id')
        if vod_id:
            # VOD-LEVEL DEDUP: Check if this VOD has been processed before (cross-session safety)
            try:
                response = job_table.get_item(Key={'vod_id': vod_id})
                if 'Item' in response:
                    status = response['Item'].get('status')
                    if status in ['completed', 'running', 'queued']:
                        print(f"VOD {vod_id} already in job table with status={status}. Skipping trigger.")
                        # Still try to clear session
                        session_cleared = delete_session_safe(session_table, user_id)
                        if session_cleared:
                            update_last_seen_timestamp(state_table, user_id, time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(now_ts)))
                            print(f"Session cleared for {streamer_name} (VOD already queued)")
                        return
            except ClientError as e:
                print(f"Warning: Could not check VOD dedup for {vod_id}: {e}")
                # Continue to process anyway
            
            launched = process_vod({'id': vod_id, 'title': f'VOD {vod_id}', 'created_at': ''}, job_table, streamer_name)
            if launched:
                session_cleared = delete_session_safe(session_table, user_id)
                if session_cleared:
                    # update last-seen watermark so subsequent older VODs aren't reprocessed
                    update_last_seen_timestamp(state_table, user_id, time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(now_ts)))
                    print(f" Enqueued {vod_id} and cleared session for {streamer_name}")
                else:
                    print(f"⚠️  Enqueued {vod_id} but session deletion FAILED for {streamer_name}. Will retry on next run.")
            else:
                print(f"X Failed to launch VOD {vod_id}. Session remains READY for retry.")
        else:
            print(f" READY state but no vod_id for {streamer_name}. Clearing session.")
            delete_session_safe(session_table, user_id)

def get_session(session_table, user_id: str) -> dict:
    try:
        resp = session_table.get_item(Key={'streamer_id': user_id})
        return resp.get('Item') if 'Item' in resp else None
    except ClientError as e:
        print(f"DynamoDB error get_session {user_id}: {e}")
        return None

def put_session(session_table, session: dict) -> None:
    try:
        session_table.put_item(Item=session)
    except ClientError as e:
        print(f"DynamoDB error put_session {session.get('streamer_id')}: {e}")

def delete_session(session_table, user_id: str) -> None:
    """Legacy delete_session (kept for backwards compatibility)"""
    try:
        session_table.delete_item(Key={'streamer_id': user_id})
    except ClientError as e:
        print(f"DynamoDB error delete_session {user_id}: {e}")

def delete_session_safe(session_table, user_id: str) -> bool:
    """
    Delete session with TTL fallback.
    Returns True if successfully deleted or TTL was set.
    Returns False if both operations failed (will retry on next run).
    """
    try:
        session_table.delete_item(Key={'streamer_id': user_id})
        print(f"✓ Session deleted for {user_id}")
        return True
    except ClientError as e:
        print(f"DynamoDB error delete_session {user_id}: {e}")
        # Fallback: Mark with TTL so it auto-expires (3600 sec = 1 hour)
        try:
            session_table.update_item(
                Key={'streamer_id': user_id},
                UpdateExpression='SET #ttl = :ttl',
                ExpressionAttributeNames={'#ttl': 'expire_at'},
                ExpressionAttributeValues={':ttl': int(time.time()) + 3600}
            )
            print(f"✓ Set TTL fallback on session {user_id} (will auto-expire in 1h)")
            return True
        except ClientError as ttl_error:
            print(f"X TTL fallback also failed for {user_id}: {ttl_error}")
            return False

def get_last_seen_timestamp(state_table, user_id: str) -> str:
    """
    Gets the last seen VOD timestamp for a user.
    Defaults to a fixed historical timestamp for the initial run.
    """
    try:
        response = state_table.get_item(Key={'streamer_id': user_id})
        if 'Item' in response:
            return response['Item']['last_seen_vod_timestamp']
    except ClientError as e:
        print(f"DynamoDB error getting state for {user_id}: {e}")
    
    # Default to a fixed UTC timestamp for the initial run for a streamer
    print(f"No timestamp found for {user_id}, using default start time.")
    return "2025-07-26T08:00:00Z"

def update_last_seen_timestamp(state_table, user_id: str, timestamp: str):
    """Updates the last seen VOD timestamp for a user."""
    try:
        print(f"Updating last_seen_timestamp for {user_id} to {timestamp}")
        state_table.put_item(
            Item={
                'streamer_id': user_id,
                'last_seen_vod_timestamp': timestamp
            }
        )
    except ClientError as e:
        print(f"DynamoDB error updating state for {user_id}: {e}")

def force_process_vod(vod: dict, job_table, streamer: str, manual: bool = False) -> bool:
    """
    Forces processing of a VOD without any duplicate checks or constraints.
    Used for manual triggers that bypass all normal validation.
    
    Args:
        vod: VOD details dictionary
        job_table: DynamoDB job table reference
        streamer: Streamer name
        manual: Whether this is a manual trigger (affects logging and tags)
    
    Returns:
        True on success, False on failure.
    """
    vod_id = vod['id']
    
    if manual:
        print(f" FORCE PROCESSING (MANUAL): VOD {vod_id} for {streamer}")
        print(f"   Bypassing all duplicate checks and time constraints")
        print(f"   Title: {vod.get('title', 'N/A')}")
    else:
        print(f"🚀 FORCE PROCESSING: VOD {vod_id} for {streamer}")
    
    try:
        # Add entry to DynamoDB to track this manual job
        trigger_type = 'manual' if manual else 'force'
        
        job_table.put_item(
            Item={
                'vod_id': vod_id,
                'status': 'queued',
                'streamer': streamer,
                'title': vod.get('title', 'N/A'),
                'started_at': int(time.time()),
                'created_at': vod.get('created_at'),
                'trigger_type': trigger_type,
                'forced': True
            }
        )
        
        # Launch ECS task with special manual trigger tag
        ecs_success = run_ecs_task(vod_id, streamer, manual_trigger=manual)
        
        # If ECS task launch failed, mark as failed
        if not ecs_success:
            print(f"X ECS task launch failed for forced VOD {vod_id}. Marking as failed.")
            job_table.put_item(
                Item={
                    'vod_id': vod_id,
                    'status': 'failed',
                    'streamer': streamer,
                    'title': vod.get('title', 'N/A'),
                    'started_at': int(time.time()),
                    'created_at': vod.get('created_at'),
                    'trigger_type': trigger_type,
                    'forced': True,
                    'error': 'ECS task launch failed'
                }
            )
        
        return ecs_success
        
    except ClientError as e:
        print(f"DynamoDB error force processing VOD {vod_id}: {e}")
        return False

def process_vod(vod: dict, job_table, streamer: str) -> bool:
    """
    Checks if a VOD has been processed and, if not, launches an ECS task.
    Returns True on success/skip, False on failure.
    """
    vod_id = vod['id']
    print(f"Processing VOD {vod_id} for {streamer}")

    try:
        # Check if VOD already exists in DynamoDB
        response = job_table.get_item(Key={'vod_id': vod_id})
        if 'Item' in response:
            status = response['Item'].get('status')
            started_at = response['Item'].get('started_at', 0)
            current_time = int(time.time())
            
            # Always skip completed and running
            if status in ['completed', 'running']:
                print(f"VOD {vod_id} already processed or in progress. Status: {status}. Skipping.")
                return True
            
            # For queued status, check if it's been stuck for too long
            if status == 'queued':
                hours_stuck = (current_time - started_at) / 3600
                
                # AUTO-TIMEOUT: Mark as failed if stuck >48 hours (prevents 25-day backlog)
                if hours_stuck > 48:
                    print(f"⚠️  VOD {vod_id} stuck in 'queued' for {hours_stuck:.1f}h (>48h). AUTO-TIMING OUT.")
                    job_table.put_item(
                        Item={
                            'vod_id': vod_id,
                            'status': 'failed',
                            'streamer': streamer,
                            'title': response['Item'].get('title', 'N/A'),
                            'started_at': started_at,
                            'created_at': response['Item'].get('created_at', ''),
                            'failure_reason': f'Auto-timeout: queued for {hours_stuck:.1f} hours',
                            'failed_at': current_time
                        }
                    )
                    # Send alert
                    try:
                        send_timeout_alert(vod_id, streamer, hours_stuck)
                    except:
                        pass
                    return True
                
                if hours_stuck < 2:
                    print(f"VOD {vod_id} already queued {hours_stuck:.1f}h ago. Status: {status}. Skipping.")
                    return True
                else:
                    print(f" VOD {vod_id} stuck in 'queued' for {hours_stuck:.1f}h. Retrying...")
                    # Will continue to re-queue below

        # Check if VOD is too old (more than 7 days) before processing
        vod_created_at = vod.get('created_at', '')
        if vod_created_at:
            try:
                from datetime import datetime
                created_time = datetime.fromisoformat(vod_created_at.replace('Z', '+00:00'))
                age_days = (datetime.now(created_time.tzinfo) - created_time).days
                if age_days > 7:
                    print(f" VOD {vod_id} is {age_days} days old. Skipping to prevent processing old VODs.")
                    return True
            except Exception as e:
                print(f"Warning: Could not parse VOD creation date {vod_created_at}: {e}")

        # VOD is new or failed before, add/update in table and launch task
        print(f"New VOD found: {vod_id}. Queueing for processing.")
        job_table.put_item(
            Item={
                'vod_id': vod_id,
                'status': 'queued',
                'streamer': streamer,
                'title': vod.get('title', 'N/A'),
                'started_at': int(time.time()),
                'created_at': vod.get('created_at')
            }
        )
        
        # Try to launch ECS task
        ecs_success = run_ecs_task(vod_id, streamer)
        
        # If ECS task launch failed, mark as failed to unblock future VODs
        if not ecs_success:
            print(f"X ECS task launch failed for VOD {vod_id}. Marking as failed to unblock queue.")
            job_table.put_item(
                Item={
                    'vod_id': vod_id,
                    'status': 'failed',
                    'streamer': streamer,
                    'title': vod.get('title', 'N/A'),
                    'started_at': int(time.time()),
                    'created_at': vod.get('created_at'),
                    'error': 'ECS task launch failed'
                }
            )
        
        return ecs_success

    except ClientError as e:
        print(f"DynamoDB error processing VOD {vod_id}: {e}")
        return False


def send_timeout_alert(vod_id: str, streamer: str, hours_stuck: float) -> None:
    """Send SQS alert when VOD times out in queue"""
    try:
        sqs = boto3.client('sqs')
        alert_queue = f"https://sqs.us-east-1.amazonaws.com/{get_aws_account_id()}/streamsniped-alerts"
        
        message = {
            'alert_type': 'vod_timeout',
            'vod_id': vod_id,
            'streamer': streamer,
            'hours_stuck': hours_stuck,
            'timestamp': int(time.time())
        }
        
        sqs.send_message(
            QueueUrl=alert_queue,
            MessageBody=json.dumps(message)
        )
        print(f"📢 Alert sent: VOD {vod_id} timeout after {hours_stuck:.1f}h")
    except Exception as e:
        print(f"Warning: Could not send alert: {e}")

def run_ecs_task(vod_id: str, streamer: str, manual_trigger: bool = False) -> bool:
    """
    Launches an ECS task to process the given VOD using Fargate.
    
    Args:
        vod_id: The VOD ID to process
        streamer: The streamer name
        manual_trigger: Whether this is a manual trigger (affects tags and logging)
    
    Returns:
        True on success, False on failure.
    """
    try:
        # Determine task definition based on GPU configuration
        if GPU_ENABLED and USE_GPU_FOR_ALL_STEPS:
            task_definition = ECS_TASK_DEFINITION_GPU
            gpu_mode_value = 'true'
            processing_mode = 'GPU-accelerated'
            print(f"🎮 GPU SYSTEM: Using GPU-accelerated processing")
        else:
            task_definition = ECS_TASK_DEFINITION
            gpu_mode_value = 'false'
            processing_mode = 'CPU-only'
            print(f"💻 GPU SYSTEM: Using CPU-only processing")
        
        trigger_msg = " (MANUAL TRIGGER)" if manual_trigger else ""
        print(f"🚀 Launching Fargate task for VOD: {vod_id}{trigger_msg}")
        print(f" Using task definition: {task_definition}")
        print(f" Quality: 1080p | Transitions: enabled | Mode: {processing_mode}")
        print(f"🔧 GPU Mode: {gpu_mode_value} | GPU Enabled: {GPU_ENABLED} | Use GPU for All Steps: {USE_GPU_FOR_ALL_STEPS}")
        
        # If local_only, enqueue to FULL_QUEUE and return
        if GPU_PROCESSING_MODE == 'local_only' and FULL_QUEUE_URL:
            try:
                sqs = boto3.client('sqs')
                body = json.dumps({"vod_id": vod_id, "streamer": streamer, "source": "lambda"})
                sqs.send_message(QueueUrl=FULL_QUEUE_URL, MessageBody=body)
                print(f"📤 Enqueued FULL local job for {vod_id} → {FULL_QUEUE_URL}")
                return True
            except Exception as e:
                print(f"X Failed to enqueue FULL job: {e}")
                # Fall back to cloud execution

        # Use Fargate Spot with On-Demand fallback for cost optimization
        print(f"🚀 Launch type: Fargate Spot (with On-Demand fallback)")
        run_task_params = {
            'cluster': ECS_CLUSTER,
            'taskDefinition': task_definition,
            'capacityProviderStrategy': [
                {
                    'capacityProvider': 'FARGATE_SPOT',
                    'weight': 4,
                    'base': 0
                },
                {
                    'capacityProvider': 'FARGATE', 
                    'weight': 1,
                    'base': 0  # Fallback only
                }
            ],
            'networkConfiguration': {
                'awsvpcConfiguration': {
                    'subnets': ECS_SUBNETS,
                    'securityGroups': ECS_SECURITY_GROUPS,
                    'assignPublicIp': 'ENABLED'
                }
            }
        }
        print(f"🌐 Network config: awsvpc (Fargate)")
        
        # Enhanced environment variables for GPU processing
        environment_vars = [
            {'name': 'VOD_ID', 'value': vod_id},
            {'name': 'VOD_URL', 'value': f'https://www.twitch.tv/videos/{vod_id}'},
            {'name': 'VIDEO_TITLE', 'value': vod_id},  # Fallback to VOD_ID, will be updated during workflow
            {'name': 'CONTAINER_MODE', 'value': 'true'},
            {'name': 'STORAGE_TYPE', 'value': 's3'},
            {'name': 'GENERATE_SHORTS', 'value': 'true'},
            {'name': 'QUALITY', 'value': '1080p'},
            {'name': 'GPU_MODE', 'value': gpu_mode_value},  # Dynamic based on GPU configuration
            {'name': 'GPU_ENABLED', 'value': str(GPU_ENABLED).lower()},
            {'name': 'USE_GPU_FOR_ALL_STEPS', 'value': str(USE_GPU_FOR_ALL_STEPS).lower()},
            {'name': 'GPU_PROCESSING_MODE', 'value': GPU_PROCESSING_MODE},
            {'name': 'MAX_WORKERS', 'value': '2'},
            {'name': 'ENABLE_TRANSITIONS', 'value': os.getenv('ENABLE_TRANSITIONS', 'false')},
            {'name': 'FARGATE_MODE', 'value': 'true'},  # Enable Fargate optimizations
            {'name': 'MEMORY_LIMIT', 'value': '8192'},  # 8GB for GPU processing
            # Clip generation offloading configuration
            {'name': 'SKIP_CLIP_ENCODING_IN_CLOUD', 'value': 'true'},
            {'name': 'OFFLOAD_CLIPS', 'value': 'true'},
            {'name': 'CLIP_QUEUE_URL', 'value': f'https://sqs.us-east-1.amazonaws.com/{get_aws_account_id()}/streamsniped-clip-queue'},
            {'name': 'CLIP_LAYOUT_SYSTEM', 'value': 'advanced'},
            # Vision API keys for Molmo/Gemini inside the container
            {'name': 'OPENROUTER_API_KEY', 'value': os.getenv('OPENROUTER_API_KEY', '')},
            {'name': 'GEMINI_API_KEY', 'value': os.getenv('GEMINI_API_KEY', '')},
            # Ensure cloud respects local postprocess gating for steps 3c/3d/3e
            {'name': 'LOCAL_POSTPROCESS', 'value': os.getenv('LOCAL_POSTPROCESS', 'true')},
        ]
        
        # Add GPU-specific environment variables if GPU is enabled
        if GPU_ENABLED:
            environment_vars.extend([
                {'name': 'NVIDIA_VISIBLE_DEVICES', 'value': 'all'},
                {'name': 'NVIDIA_DRIVER_CAPABILITIES', 'value': 'compute,video,utility'},
                {'name': 'CUDA_VISIBLE_DEVICES', 'value': '0'},
                {'name': 'GPU_DEVICE', 'value': '0'},
            ])
        
        run_task_params.update({
            'overrides': {
                'containerOverrides': [{
                    'name': 'streamsniped',
                    'environment': environment_vars
                }]
            }
        })
        
        # Add tags for tracking
        tags = [
            {'key': 'Service', 'value': 'StreamSniped'},
            {'key': 'VOD_ID', 'value': vod_id},
            {'key': 'Streamer', 'value': streamer},
            {'key': 'GPU_Enabled', 'value': str(GPU_ENABLED)},
            {'key': 'GPU_Mode', 'value': gpu_mode_value},
            {'key': 'Processing_Mode', 'value': processing_mode},
        ]
        
        if manual_trigger:
            tags.append({'key': 'Trigger_Type', 'value': 'manual'})
        
        run_task_params['tags'] = tags
        
        print(f"🎮 GPU SYSTEM: Launching task with {processing_mode} configuration")
        print(f"📊 Environment variables: {len(environment_vars)} GPU-aware variables set")
        
        response = ecs.run_task(**run_task_params)
        
        if response['failures']:
            print(f"X ECS task launch failed: {response['failures']}")
            return False
        
        task_arn = response['tasks'][0]['taskArn']
        print(f" ECS task launched successfully: {task_arn}")
        print(f"🎮 GPU SYSTEM: Task launched with {processing_mode} processing")
        
        return True
        
    except ClientError as e:
        print(f"X AWS Error launching ECS task: {e}")
        return False
    except Exception as e:
        print(f"X Unexpected error launching ECS task: {e}")
        return False


def backfill_missing_vods(user_id: str, streamer_name: str, max_vods: int = 10) -> int:
    """
    One-time recovery function to backfill missing VODs from a streamer's archive.
    Useful after AWS outages or session gaps.
    
    Args:
        user_id: Twitch user ID
        streamer_name: Streamer name for logging
        max_vods: Maximum number of VODs to backfill (prevents overload)
    
    Returns:
        Number of VODs backfilled
    """
    print(f"\n🔄 RECOVERY: Backfilling missing VODs for {streamer_name}")
    
    if TwitchMonitor is None:
        print("X TwitchMonitor not available")
        return 0
    
    try:
        twitch_monitor = TwitchMonitor()
        job_table = dynamodb.Table(DYNAMODB_TABLE)
        backfilled_count = 0
        
        # Get last 20 VODs from Twitch
        vods = twitch_monitor.get_recent_vods(user_id, limit=20)
        if not vods:
            print(f"No VODs found for {streamer_name}")
            return 0
        
        print(f"Found {len(vods)} recent VODs from Twitch")
        
        for vod in vods:
            if backfilled_count >= max_vods:
                print(f"Reached backfill limit ({max_vods})")
                break
            
            vod_id = vod.get('id')
            created_at = vod.get('created_at')
            title = vod.get('title', f'VOD {vod_id}')
            
            if not vod_id:
                continue
            
            # Check if VOD already exists in job table
            try:
                response = job_table.get_item(Key={'vod_id': vod_id})
                if 'Item' in response:
                    status = response['Item'].get('status')
                    print(f"  VOD {vod_id}: Already exists (status={status})")
                    continue
            except ClientError as e:
                print(f"  VOD {vod_id}: Error checking status: {e}")
                continue
            
            # Check VOD age (skip if too old)
            try:
                from datetime import datetime
                created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                age_days = (datetime.now(created_time.tzinfo) - created_time).days
                if age_days > 7:
                    print(f"  VOD {vod_id}: Too old ({age_days} days). Skipping.")
                    continue
            except Exception as e:
                print(f"  VOD {vod_id}: Could not parse date {created_at}: {e}")
            
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
                print(f"  ✓ Backfilled VOD {vod_id} ({title})")
                backfilled_count += 1
            except ClientError as e:
                print(f"  X Error backfilling VOD {vod_id}: {e}")
        
        print(f"✓ Backfill complete: {backfilled_count} VODs queued")
        return backfilled_count
        
    except Exception as e:
        print(f"X Error in backfill_missing_vods: {e}")
        return 0 