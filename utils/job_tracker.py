#!/usr/bin/env python3
"""
Job tracking system for StreamSniped
Uses DynamoDB to track VOD processing jobs
"""

import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
import logging

# Optional DynamoDB imports
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    DYNAMODB_AVAILABLE = True
except ImportError:
    DYNAMODB_AVAILABLE = False

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStep(Enum):
    """Job step enumeration"""
    INGEST = "ingest"
    TRANSCRIBE = "transcribe"
    CLASSIFY = "classify"
    
    RENDER = "render"
    PUBLISH = "publish"


class JobTracker:
    """DynamoDB-based job tracking system"""
    
    def __init__(self, table_name: str = "streamsniped_jobs", region: str = "us-east-1"):
        self.table_name = table_name
        self.region = region
        self.dynamodb = None
        
        if DYNAMODB_AVAILABLE:
            try:
                self.dynamodb = boto3.resource('dynamodb', region_name=region)
                # Test connectivity
                self.dynamodb.meta.client.list_tables()
                logger.info("DynamoDB client initialized successfully")
            except (NoCredentialsError, ClientError) as e:
                logger.warning(f"DynamoDB not available: {e}")
                self.dynamodb = None
    
    def create_table(self) -> bool:
        """Create DynamoDB table if it doesn't exist"""
        if not self.dynamodb:
            logger.error("DynamoDB not available")
            return False
        
        try:
            table = self.dynamodb.create_table(
                TableName=self.table_name,
                KeySchema=[
                    {
                        'AttributeName': 'vod_id',
                        'KeyType': 'HASH'  # Partition key
                    }
                ],
                AttributeDefinitions=[
                    {
                        'AttributeName': 'vod_id',
                        'AttributeType': 'S'
                    }
                ],
                BillingMode='PAY_PER_REQUEST',
                Tags=[
                    {
                        'Key': 'Project',
                        'Value': 'StreamSniped'
                    }
                ]
            )
            
            # Wait for table to be created
            table.meta.client.get_waiter('table_exists').wait(TableName=self.table_name)
            logger.info(f"Created DynamoDB table: {self.table_name}")
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceInUseException':
                logger.info(f"Table {self.table_name} already exists")
                return True
            else:
                logger.error(f"Failed to create table: {e}")
                return False
    
    def start_job(self, vod_id: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Start a new job or resume existing job"""
        if not self.dynamodb:
            logger.warning("DynamoDB not available, skipping job tracking")
            return True
        
        try:
            table = self.dynamodb.Table(self.table_name)
            
            # Check if job already exists
            response = table.get_item(Key={'vod_id': vod_id})
            
            if 'Item' in response:
                existing_job = response['Item']
                if existing_job.get('status') == JobStatus.COMPLETED.value:
                    logger.info(f"Job {vod_id} already completed, skipping")
                    return True
                elif existing_job.get('status') == JobStatus.RUNNING.value:
                    logger.info(f"Job {vod_id} already running, resuming")
            
            # Create or update job record
            job_data = {
                'vod_id': vod_id,
                'status': JobStatus.RUNNING.value,
                'step': JobStep.INGEST.value,
                'started_at': int(time.time()),
                'started_at_iso': datetime.utcnow().isoformat(),
                'updated_at': int(time.time()),
                'updated_at_iso': datetime.utcnow().isoformat(),
                'audio_minutes': 0,
                'compute_seconds': 0,
                'gpu_seconds': 0,
                'stt_provider': 'whisper',
                'error_message': '',
                'trace_id': str(uuid.uuid4()),
                'final_s3': '',
                'log_group': f"/aws/batch/job/{vod_id}"
            }
            
            if metadata:
                job_data.update(metadata)
            
            table.put_item(Item=job_data)
            logger.info(f"Started job tracking for VOD: {vod_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start job {vod_id}: {e}")
            return False

    def acquire_lock(self, vod_id: str, job_type: str = "full", lease_seconds: int = 900, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Acquire a processing lock for a VOD using a conditional update.

        Succeeds if the item does not exist, is not RUNNING, or the existing RUNNING lease has expired.
        """
        if not self.dynamodb:
            return True
        try:
            now = int(time.time())
            table = self.dynamodb.Table(self.table_name)

            update_attrs: Dict[str, Any] = {
                'status': JobStatus.RUNNING.value,
                'job_type': job_type,
                'lease_expires_at': now + max(30, lease_seconds),
                'heartbeat_at': now,
                'updated_at': now,
                'updated_at_iso': datetime.utcnow().isoformat(),
            }
            if metadata:
                update_attrs.update(metadata)

            update_expr = "SET " + ", ".join([f"#{k} = :{k}" for k in update_attrs.keys()])

            expr_names = {f"#{k}": k for k in update_attrs.keys()}
            expr_values = {f":{k}": v for k, v in update_attrs.items()}
            expr_values[":running"] = JobStatus.RUNNING.value
            expr_names["#status"] = "status"
            expr_names["#lease_expires_at"] = "lease_expires_at"

            cond_expr = (
                "attribute_not_exists(vod_id) OR #status <> :running OR "
                "(#status = :running AND (attribute_not_exists(#lease_expires_at) OR #lease_expires_at < :heartbeat_at))"
            )

            table.update_item(
                Key={'vod_id': vod_id},
                UpdateExpression=update_expr,
                ExpressionAttributeNames=expr_names,
                ExpressionAttributeValues=expr_values,
                ConditionExpression=cond_expr,
            )
            logger.info(f"Acquired job lock for {vod_id} ({job_type})")
            return True
        except Exception as e:
            logger.info(f"Lock not acquired for {vod_id}: {e}")
            return False

    def heartbeat(self, vod_id: str, lease_seconds: int = 900) -> bool:
        """Refresh job heartbeat and extend lease for a RUNNING job."""
        if not self.dynamodb:
            return True
        try:
            now = int(time.time())
            table = self.dynamodb.Table(self.table_name)
            table.update_item(
                Key={'vod_id': vod_id},
                UpdateExpression="SET #heartbeat_at = :now, #lease_expires_at = :lease, #updated_at = :now, #updated_at_iso = :now_iso",
                ExpressionAttributeNames={
                    '#heartbeat_at': 'heartbeat_at',
                    '#lease_expires_at': 'lease_expires_at',
                    '#updated_at': 'updated_at',
                    '#updated_at_iso': 'updated_at_iso',
                    '#status': 'status',
                },
                ExpressionAttributeValues={
                    ':now': now,
                    ':now_iso': datetime.utcnow().isoformat(),
                    ':lease': now + max(30, lease_seconds),
                    ':running': JobStatus.RUNNING.value,
                },
                ConditionExpression="#status = :running",
            )
            return True
        except Exception as e:
            logger.warning(f"Heartbeat failed for {vod_id}: {e}")
            return False

    def release_and_mark(self, vod_id: str, status: JobStatus, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Release the lock and set final status (COMPLETED/FAILED/CANCELLED)."""
        if not self.dynamodb:
            return True
        try:
            now = int(time.time())
            table = self.dynamodb.Table(self.table_name)
            update_data: Dict[str, Any] = {
                'status': status.value,
                'ended_at': now,
                'ended_at_iso': datetime.utcnow().isoformat(),
                'updated_at': now,
                'updated_at_iso': datetime.utcnow().isoformat(),
            }
            if metadata:
                update_data.update(metadata)

            set_expr = ", ".join([f"#{k} = :{k}" for k in update_data.keys()])
            update_expr = f"SET {set_expr} REMOVE #lease_expires_at, #heartbeat_at"
            expr_names = {f"#{k}": k for k in update_data.keys()}
            expr_names['#lease_expires_at'] = 'lease_expires_at'
            expr_names['#heartbeat_at'] = 'heartbeat_at'
            expr_values = {f":{k}": v for k, v in update_data.items()}

            table.update_item(
                Key={'vod_id': vod_id},
                UpdateExpression=update_expr,
                ExpressionAttributeNames=expr_names,
                ExpressionAttributeValues=expr_values,
            )
            logger.info(f"Released lock and marked {vod_id} as {status.value}")
            return True
        except Exception as e:
            logger.error(f"Failed to release/mark job {vod_id}: {e}")
            return False
    
    def update_step(self, vod_id: str, step: JobStep, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update job step and metadata"""
        if not self.dynamodb:
            return True
        
        try:
            table = self.dynamodb.Table(self.table_name)
            
            update_data = {
                'step': step.value,
                'updated_at': int(time.time()),
                'updated_at_iso': datetime.utcnow().isoformat()
            }
            
            if metadata:
                update_data.update(metadata)
            
            # Build update expression
            update_expr = "SET "
            expr_attrs = {}
            expr_names = {}
            
            for key, value in update_data.items():
                update_expr += f"#{key} = :{key}, "
                expr_attrs[f":{key}"] = value
                expr_names[f"#{key}"] = key
            
            update_expr = update_expr.rstrip(", ")
            
            table.update_item(
                Key={'vod_id': vod_id},
                UpdateExpression=update_expr,
                ExpressionAttributeValues=expr_attrs,
                ExpressionAttributeNames=expr_names
            )
            
            logger.info(f"Updated job {vod_id} step to: {step.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update job {vod_id} step: {e}")
            return False
    
    def complete_job(self, vod_id: str, final_s3_uri: str = "", metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Mark job as completed"""
        if not self.dynamodb:
            return True
        
        try:
            table = self.dynamodb.Table(self.table_name)
            
            update_data = {
                'status': JobStatus.COMPLETED.value,
                'step': JobStep.PUBLISH.value,
                'ended_at': int(time.time()),
                'ended_at_iso': datetime.utcnow().isoformat(),
                'updated_at': int(time.time()),
                'updated_at_iso': datetime.utcnow().isoformat(),
                'final_s3': final_s3_uri
            }
            
            if metadata:
                update_data.update(metadata)
            
            # Build update expression
            update_expr = "SET "
            expr_attrs = {}
            expr_names = {}
            
            for key, value in update_data.items():
                update_expr += f"#{key} = :{key}, "
                expr_attrs[f":{key}"] = value
                expr_names[f"#{key}"] = key
            
            update_expr = update_expr.rstrip(", ")
            
            table.update_item(
                Key={'vod_id': vod_id},
                UpdateExpression=update_expr,
                ExpressionAttributeValues=expr_attrs,
                ExpressionAttributeNames=expr_names
            )
            
            logger.info(f"Completed job: {vod_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete job {vod_id}: {e}")
            return False
    
    def fail_job(self, vod_id: str, error_message: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Mark job as failed"""
        if not self.dynamodb:
            return True
        
        try:
            table = self.dynamodb.Table(self.table_name)
            
            update_data = {
                'status': JobStatus.FAILED.value,
                'ended_at': int(time.time()),
                'ended_at_iso': datetime.utcnow().isoformat(),
                'updated_at': int(time.time()),
                'updated_at_iso': datetime.utcnow().isoformat(),
                'error_message': error_message[:500],  # Limit error message length
                'trace_id': str(uuid.uuid4())
            }
            
            if metadata:
                update_data.update(metadata)
            
            # Build update expression
            update_expr = "SET "
            expr_attrs = {}
            expr_names = {}
            
            for key, value in update_data.items():
                update_expr += f"#{key} = :{key}, "
                expr_attrs[f":{key}"] = value
                expr_names[f"#{key}"] = key
            
            update_expr = update_expr.rstrip(", ")
            
            table.update_item(
                Key={'vod_id': vod_id},
                UpdateExpression=update_expr,
                ExpressionAttributeValues=expr_attrs,
                ExpressionAttributeNames=expr_names
            )
            
            logger.error(f"Failed job: {vod_id} - {error_message}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark job {vod_id} as failed: {e}")
            return False
    
    def get_job(self, vod_id: str) -> Optional[Dict[str, Any]]:
        """Get job details"""
        if not self.dynamodb:
            return None
        
        try:
            table = self.dynamodb.Table(self.table_name)
            response = table.get_item(Key={'vod_id': vod_id})
            
            if 'Item' in response:
                return response['Item']
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get job {vod_id}: {e}")
            return None
    
    def list_jobs(self, status: Optional[JobStatus] = None, limit: int = 100) -> list:
        """List jobs with optional status filter"""
        if not self.dynamodb:
            return []
        
        try:
            table = self.dynamodb.Table(self.table_name)
            
            if status:
                response = table.scan(
                    FilterExpression='#status = :status',
                    ExpressionAttributeNames={'#status': 'status'},
                    ExpressionAttributeValues={':status': status.value},
                    Limit=limit
                )
            else:
                response = table.scan(Limit=limit)
            
            return response.get('Items', [])
            
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            return []


# Global job tracker instance
job_tracker = JobTracker()


# Convenience functions
def start_job(vod_id: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Start job tracking"""
    return job_tracker.start_job(vod_id, metadata)


def update_step(vod_id: str, step: JobStep, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Update job step"""
    return job_tracker.update_step(vod_id, step, metadata)


def complete_job(vod_id: str, final_s3_uri: str = "", metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Complete job"""
    return job_tracker.complete_job(vod_id, final_s3_uri, metadata)


def fail_job(vod_id: str, error_message: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Fail job"""
    return job_tracker.fail_job(vod_id, error_message, metadata)


def get_job(vod_id: str) -> Optional[Dict[str, Any]]:
    """Get job details"""
    return job_tracker.get_job(vod_id)


def list_jobs(status: Optional[JobStatus] = None, limit: int = 100) -> list:
    """List jobs"""
    return job_tracker.list_jobs(status, limit) 