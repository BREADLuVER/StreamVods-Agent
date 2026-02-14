#!/usr/bin/env python3
"""
Deploys the StreamSniped Twitch Watcher infrastructure to AWS.

This script creates the necessary AWS resources for the VOD watcher, including:
- An IAM Role and Policy for the Lambda function.
- The Lambda function itself, packaged with its dependencies.
- An EventBridge rule to trigger the Lambda on a schedule.
"""

import boto3
import os
import json
from botocore.exceptions import ClientError
from typing import Optional
import time
import sys
import shutil

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

class WatcherDeployer:
    """Handles the deployment of the Lambda-based VOD watcher."""

    def __init__(self, region: str = "us-east-1"):
        """
        Initializes the deployer.

        Args:
            region: The AWS region to deploy to.
        """
        self.region = region
        self.iam = boto3.client('iam', region_name=region)
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.events_client = boto3.client('events', region_name=region)
        self.logs = boto3.client('logs', region_name=region)
        self.account_id = boto3.client('sts').get_caller_identity().get('Account')

        # Resource names
        self.project_name = "streamsniped-watcher"
        self.environment = os.getenv('ENVIRONMENT', 'dev')
        self.role_name = f"{self.project_name}-{self.environment}-lambda-role"
        self.policy_name = f"{self.project_name}-{self.environment}-lambda-policy"
        self.lambda_function_name = f"{self.project_name}-{self.environment}-function"
        self.event_rule_name = f"{self.project_name}-{self.environment}-rule"
        self.state_table_name = f"streamsniped-watcher-state-{self.environment}"
        self.session_table_name = f"streamsniped-sessions-{self.environment}"
        
        print(f"🚀 Deploying StreamSniped Watcher to {self.region}")
        print(f"   Environment: {self.environment}")
        print(f"   Account ID: {self.account_id}")
        print(f"    Features: GPU processing, Random transitions, 1080p quality")
        print(f"   ⏰ Schedule: Every 30 minutes")

    def create_cloudwatch_log_group(self) -> bool:
        """Create CloudWatch log group for Lambda function"""
        log_group_name = f"/aws/lambda/{self.lambda_function_name}"
        
        try:
            print(f"\n Creating CloudWatch log group: {log_group_name}")
            
            # Check if log group already exists
            try:
                self.logs.describe_log_groups(logGroupNamePrefix=log_group_name)
                print(f" Log group {log_group_name} already exists")
                return True
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    # Log group doesn't exist, create it
                    pass
                else:
                    raise e
            
            # Create the log group
            self.logs.create_log_group(
                logGroupName=log_group_name,
                tags={
                    'Service': 'StreamSniped',
                    'Environment': self.environment,
                    'Function': self.lambda_function_name
                }
            )
            
            print(f" CloudWatch log group created: {log_group_name}")
            return True
            
        except ClientError as e:
            print(f"X Failed to create CloudWatch log group: {e}")
            return False
        except Exception as e:
            print(f"X Unexpected error creating log group: {e}")
            return False

    def create_iam_role_for_lambda(self) -> Optional[str]:
        """
        Creates the IAM Role and Policy for the Lambda function.

        The role grants permissions to write to CloudWatch Logs, interact with
        DynamoDB, and run ECS tasks.

        Returns:
            The ARN of the created role, or None if it failed.
        """
        print(f"\n🔧 Creating IAM Role: {self.role_name}...")

        assume_role_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }
            ]
        }

        try:
            role_response = self.iam.create_role(
                RoleName=self.role_name,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy),
                Description="Role for StreamSniped VOD Watcher Lambda"
            )
            role_arn = role_response['Role']['Arn']
            print(f" IAM Role '{self.role_name}' created.")
        except ClientError as e:
            if e.response['Error']['Code'] == 'EntityAlreadyExists':
                print(f"🔹 IAM Role '{self.role_name}' already exists. Fetching ARN.")
                role_response = self.iam.get_role(RoleName=self.role_name)
                role_arn = role_response['Role']['Arn']
            else:
                print(f"X Error creating IAM role: {e}")
                return None

        # Define and attach the permissions policy
        policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents"
                    ],
                    "Resource": f"arn:aws:logs:{self.region}:{self.account_id}:*"
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "sqs:SendMessage",
                        "sqs:GetQueueAttributes",
                        "sqs:GetQueueUrl"
                    ],
                    "Resource": [
                        f"arn:aws:sqs:{self.region}:{self.account_id}:streamsniped-render-queue",
                        f"arn:aws:sqs:{self.region}:{self.account_id}:streamsniped-clip-queue",
                        f"arn:aws:sqs:{self.region}:{self.account_id}:streamsniped-full-queue",
                        f"arn:aws:sqs:{self.region}:{self.account_id}:streamsniped-alerts"
                    ]
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "dynamodb:GetItem",
                        "dynamodb:PutItem",
                        "dynamodb:UpdateItem",
                        "dynamodb:DeleteItem"
                    ],
                    "Resource": [
                        f"arn:aws:dynamodb:{self.region}:{self.account_id}:table/streamsniped_jobs",
                        f"arn:aws:dynamodb:{self.region}:{self.account_id}:table/{self.state_table_name}",
                        f"arn:aws:dynamodb:{self.region}:{self.account_id}:table/{self.session_table_name}"
                    ]
                },
                {
                    "Effect": "Allow",
                    "Action": "ecs:RunTask",
                    "Resource": [
                        f"arn:aws:ecs:{self.region}:{self.account_id}:task-definition/streamsniped-fargate:*",
                        f"arn:aws:ecs:{self.region}:{self.account_id}:task-definition/streamsniped-download:*",
                        f"arn:aws:ecs:{self.region}:{self.account_id}:task-definition/streamsniped-download-cpu:*"
                    ]
                },
                {
                    "Effect": "Allow",
                    "Action": "ecs:TagResource",
                    "Resource": f"arn:aws:ecs:{self.region}:{self.account_id}:task/streamsniped-{self.environment}-cluster/*"
                },
                {
                    "Effect": "Allow",
                    "Action": "iam:PassRole",
                    "Resource": [
                        f"arn:aws:iam::{self.account_id}:role/*ecs*",
                        f"arn:aws:iam::{self.account_id}:role/*task*"
                    ],
                    "Condition": {"StringLike": {"iam:PassedToService": "ecs-tasks.amazonaws.com"}}
                }
            ]
        }
        
        try:
            print(f" Attaching policy '{self.policy_name}' to role '{self.role_name}'...")
            self.iam.put_role_policy(
                RoleName=self.role_name,
                PolicyName=self.policy_name,
                PolicyDocument=json.dumps(policy_document)
            )
            print(" Policy attached successfully.")
            # Wait for the role to be fully propagated
            self.iam.get_waiter('role_exists').wait(RoleName=self.role_name)
            time.sleep(10) # Additional wait time for IAM propagation
            return role_arn
        except ClientError as e:
            print(f"X Error attaching policy: {e}")
            return None

    def create_state_table(self):
        """Creates the DynamoDB table to store watcher state."""
        print(f"\n🏬 Creating DynamoDB state table: {self.state_table_name}...")
        dynamodb = boto3.client('dynamodb', region_name=self.region)
        try:
            dynamodb.create_table(
                TableName=self.state_table_name,
                AttributeDefinitions=[{'AttributeName': 'streamer_id', 'AttributeType': 'S'}],
                KeySchema=[{'AttributeName': 'streamer_id', 'KeyType': 'HASH'}],
                BillingMode='PAY_PER_REQUEST'
            )
            print(" Waiting for table to become active...")
            waiter = dynamodb.get_waiter('table_exists')
            waiter.wait(TableName=self.state_table_name)
            print(" State table created successfully.")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceInUseException':
                print(f"🔹 State table '{self.state_table_name}' already exists.")
                return True
            print(f"X Error creating state table: {e}")
            return False

    def create_session_table(self):
        """Creates the DynamoDB table to store per-streamer session state."""
        print(f"\n🏬 Creating DynamoDB session table: {self.session_table_name}...")
        dynamodb = boto3.client('dynamodb', region_name=self.region)
        try:
            dynamodb.create_table(
                TableName=self.session_table_name,
                AttributeDefinitions=[{'AttributeName': 'streamer_id', 'AttributeType': 'S'}],
                KeySchema=[{'AttributeName': 'streamer_id', 'KeyType': 'HASH'}],
                BillingMode='PAY_PER_REQUEST'
            )
            print(" Waiting for session table to become active...")
            waiter = dynamodb.get_waiter('table_exists')
            waiter.wait(TableName=self.session_table_name)
            print(" Session table created successfully.")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceInUseException':
                print(f"🔹 Session table '{self.session_table_name}' already exists.")
                return True
            print(f"X Error creating session table: {e}")
            return False

    def initialize_state_timestamps(self):
        """Initialize state table with current UTC timestamps for all streamers."""
        import datetime
        
        print(f"\n🕐 Initializing state timestamps with current UTC time...")
        current_utc = datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
        print(f"Current UTC time: {current_utc}")
        
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from twitch_monitor import TwitchMonitor
        
        dynamodb = boto3.resource('dynamodb', region_name=self.region)
        table = dynamodb.Table(self.state_table_name)
        
        # Get streamers from environment
        streamers = os.getenv('STREAMERS','disguisedtoast,quarterjade,masayoshi').split(',')
        
        # Initialize TwitchMonitor to get user IDs
        try:
            twitch_monitor = TwitchMonitor()
        except Exception as e:
            print(f"X Could not initialize TwitchMonitor: {e}")
            print("🔹 Skipping state timestamp initialization")
            return True
        
        for streamer_name in streamers:
            streamer_name = streamer_name.strip()
            try:
                # Get the numeric user ID for this streamer
                user_id = twitch_monitor.get_user_id(streamer_name)
                if not user_id:
                    print(f"X Could not find user ID for {streamer_name}")
                    continue
                
                print(f"🔍 {streamer_name} → user_id: {user_id}")
                
                # Check if entry already exists (using user_id as key)
                response = table.get_item(Key={'streamer_id': user_id})
                if 'Item' not in response:
                    # Only initialize if no entry exists
                    table.put_item(
                        Item={
                            'streamer_id': user_id,
                            'last_seen_vod_timestamp': current_utc
                        }
                    )
                    print(f" Initialized timestamp for {streamer_name} (ID: {user_id}): {current_utc}")
                else:
                    print(f"🔹 {streamer_name} (ID: {user_id}) already has timestamp: {response['Item'].get('last_seen_vod_timestamp', 'Unknown')}")
            except Exception as e:
                print(f"X Error initializing timestamp for {streamer_name}: {e}")
        
        print(" State timestamp initialization complete.")
        return True

    def package_lambda_code(self) -> bytes:
        """
        Packages the Lambda function code and its dependencies into a zip file.

        Returns:
            The zipped code as a bytes object.
        """
        import zipfile
        import io
        import tempfile
        import subprocess

        print("\n📦 Packaging Lambda function code (including dependencies)...")
        
        # Create a temporary directory to stage files
        with tempfile.TemporaryDirectory() as build_dir:
            # Install 'requests' dependency into the temp directory
            subprocess.check_call([
                sys.executable,
                '-m',
                'pip',
                'install',
                'requests',
                '--upgrade',
                '--target',
                build_dir,
                '--no-cache-dir'
            ])

            # Copy our lambda code into temp dir
            shutil.copy('aws-scripts/twitch_watcher_lambda.py', os.path.join(build_dir, 'twitch_watcher_lambda.py'))
            shutil.copy('aws-scripts/twitch_monitor.py', os.path.join(build_dir, 'twitch_monitor.py'))

            # Zip everything up
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for root, _, files in os.walk(build_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, build_dir)
                        zip_file.write(file_path, arcname)
            zip_buffer.seek(0)
            print(" Lambda package with dependencies created.")
            return zip_buffer.read()

    def deploy_lambda_function(self, role_arn: str, env_vars: dict) -> Optional[str]:
        """
        Deploys the Lambda function to AWS.

        Args:
            role_arn: The ARN of the IAM role for the Lambda.
            env_vars: A dictionary of environment variables for the Lambda.

        Returns:
            The ARN of the created Lambda function, or None if it failed.
        """
        print(f"\n🚀 Deploying Lambda function: {self.lambda_function_name}...")
        
        zip_bytes = self.package_lambda_code()
        
        # Prepare environment variables
        lambda_env = {
            'Variables': {
                'ECS_CLUSTER': env_vars.get('ECS_CLUSTER'),
                'ECS_TASK_DEFINITION': env_vars.get('ECS_TASK_DEFINITION'),
                'ECS_TASK_DEFINITION_GPU': env_vars.get('ECS_TASK_DEFINITION_GPU'),
                'GPU_MODE': env_vars.get('GPU_MODE'),
                'ECS_SUBNETS': env_vars.get('ECS_SUBNETS'),
                'ECS_SECURITY_GROUPS': env_vars.get('ECS_SECURITY_GROUPS'),
                'DYNAMODB_TABLE': env_vars.get('DYNAMODB_TABLE'),
                'STATE_TABLE_NAME': env_vars.get('STATE_TABLE_NAME'),
                'SESSION_TABLE_NAME': env_vars.get('SESSION_TABLE_NAME'),
                'STREAMERS': env_vars.get('STREAMERS'),
                'ENABLE_TRANSITIONS': env_vars.get('ENABLE_TRANSITIONS'),
                'TWITCH_CLIENT_ID': env_vars.get('TWITCH_CLIENT_ID'),
                'TWITCH_CLIENT_SECRET': env_vars.get('TWITCH_CLIENT_SECRET'),
                'YOUTUBE_CLIENT_ID': env_vars.get('YOUTUBE_CLIENT_ID'),
                'YOUTUBE_CLIENT_SECRET': env_vars.get('YOUTUBE_CLIENT_SECRET'),
                'OPENROUTER_API_KEY': env_vars.get('OPENROUTER_API_KEY'),
                'GEMINI_API_KEY': env_vars.get('GEMINI_API_KEY'),
                'S3_BUCKET': env_vars.get('S3_BUCKET'),
                'UPLOAD_VIDEOS': env_vars.get('UPLOAD_VIDEOS'),
                'UPLOAD_YOUTUBE': env_vars.get('UPLOAD_YOUTUBE'),
                'AI_DIRECT_CLIPPER_SCORE_THRESHOLD': env_vars.get('AI_DIRECT_CLIPPER_SCORE_THRESHOLD', '7.5'),
                'CLASSIFICATION_SECTIONS_SCORE_THRESHOLD': env_vars.get('CLASSIFICATION_SECTIONS_SCORE_THRESHOLD', '7.5'),
                'HIGHLIGHT_SCORE_THRESHOLD': env_vars.get('HIGHLIGHT_SCORE_THRESHOLD', '7.5'),
                # Clip generation offloading configuration
                'SKIP_CLIP_ENCODING_IN_CLOUD': env_vars.get('SKIP_CLIP_ENCODING_IN_CLOUD'),
                'OFFLOAD_CLIPS': env_vars.get('OFFLOAD_CLIPS'),
                'CLIP_QUEUE_URL': env_vars.get('CLIP_QUEUE_URL'),
                'CLIP_LAYOUT_SYSTEM': env_vars.get('CLIP_LAYOUT_SYSTEM'),
                'LOCAL_POSTPROCESS': env_vars.get('LOCAL_POSTPROCESS', 'true'),
                # GPU processing configuration
                'GPU_PROCESSING_MODE': env_vars.get('GPU_PROCESSING_MODE'),
                'GPU_ENABLED': env_vars.get('GPU_ENABLED'),
                'USE_GPU_FOR_ALL_STEPS': env_vars.get('USE_GPU_FOR_ALL_STEPS'),
                'FULL_QUEUE_URL': env_vars.get('FULL_QUEUE_URL'),
            }
        }
        
        try:
            response = self.lambda_client.create_function(
                FunctionName=self.lambda_function_name,
                Runtime='python3.9',
                Role=role_arn,
                Handler='twitch_watcher_lambda.lambda_handler',
                Code={'ZipFile': zip_bytes},
                Description='Watches Twitch for new VODs and triggers processing.',
                Timeout=900,  # 15 minutes for video processing tasks
                MemorySize=256,  # More memory for better performance
                Publish=True,
                Environment=lambda_env
            )
            function_arn = response['FunctionArn']
            print(f" Lambda function '{self.lambda_function_name}' created.")
            return function_arn
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceConflictException':
                print(f"🔹 Lambda function '{self.lambda_function_name}' already exists. Updating...")
                response = self.lambda_client.update_function_code(
                    FunctionName=self.lambda_function_name,
                    ZipFile=zip_bytes,
                    Publish=True
                )
                # Wait for the function code update to complete
                print(" Waiting for function code update to complete...")
                waiter = self.lambda_client.get_waiter('function_updated_v2')
                waiter.wait(
                    FunctionName=self.lambda_function_name,
                    WaiterConfig={'Delay': 5, 'MaxAttempts': 60}
                )
                print(" Function code update complete.")

                print("🔹 Updating function configuration...")
                self.lambda_client.update_function_configuration(
                    FunctionName=self.lambda_function_name,
                    Role=role_arn,
                    Environment=lambda_env
                )
                function_arn = response['FunctionArn']
                print(" Lambda function updated.")
                return function_arn
            else:
                print(f"X Error creating/updating Lambda function: {e}")
                return None

    def create_eventbridge_rule(self, lambda_function_arn: str) -> Optional[str]:
        """
        Creates an EventBridge rule to trigger the Lambda function every 30 minutes.

        Args:
            lambda_function_arn: The ARN of the Lambda function to trigger.

        Returns:
            The ARN of the created rule, or None if it failed.
        """
        print(f"\n🗓️  Creating EventBridge rule: {self.event_rule_name}...")

        try:
            rule_response = self.events_client.put_rule(
                Name=self.event_rule_name,
                ScheduleExpression='rate(30 minutes)',
                State='ENABLED',
                Description='Triggers the StreamSniped VOD watcher every 30 minutes with GPU+transition support.'
            )
            rule_arn = rule_response['RuleArn']
            print(f" EventBridge rule '{self.event_rule_name}' created.")
        except ClientError as e:
            print(f"X Error creating EventBridge rule: {e}")
            return None
            
        try:
            print(f"🔗 Linking rule to Lambda function...")
            self.lambda_client.add_permission(
                FunctionName=self.lambda_function_name,
                StatementId=f'{self.project_name}-event-permission',
                Action='lambda:InvokeFunction',
                Principal='events.amazonaws.com',
                SourceArn=rule_arn,
            )
            self.events_client.put_targets(
                Rule=self.event_rule_name,
                Targets=[{'Id': '1', 'Arn': lambda_function_arn}]
            )
            print(" Rule linked to Lambda successfully.")
            return rule_arn
        except ClientError as e:
            # Handle cases where permission already exists
            if e.response['Error']['Code'] != 'ResourceConflictException':
                 print(f"X Error linking rule to Lambda: {e}")
                 return None
            else:
                print("🔹 Permission already exists, skipping.")
                return rule_arn

def get_account_id():
    """Get the AWS account ID."""
    sts = boto3.client('sts')
    try:
        identity = sts.get_caller_identity()
        return identity['Account']
    except Exception as e:
        print(f"X Error getting account ID: {e}")
        sys.exit(1)

def get_latest_task_definition_revision() -> int:
    """Return latest revision number for family 'streamsniped-fargate'.

    Tries describe_task_definition first; if that fails (e.g., not found),
    falls back to list_task_definitions and picks the highest revision.
    Returns 0 if none found so callers can decide what to do (e.g., instruct to create TD).
    """
    ecs = boto3.client('ecs', region_name='us-east-1')
    try:
        resp = ecs.describe_task_definition(taskDefinition='streamsniped-fargate')
        rev = int(resp['taskDefinition']['revision'])
        print(f" Latest task definition revision: {rev}")
        return rev
    except Exception as e:
        print(f"  describe_task_definition failed: {e}")
        try:
            paginator = ecs.get_paginator('list_task_definitions')
            arns: list[str] = []
            for page in paginator.paginate(familyPrefix='streamsniped-fargate', sort='DESC'):
                arns.extend(page.get('taskDefinitionArns', []))
            if not arns:
                print("  No task definitions exist for family 'streamsniped-fargate'.")
                return 0
            # First item is highest when sorted DESC
            latest_arn = arns[0]
            # ARN format: ...:task-definition/streamsniped-fargate:<rev>
            rev_str = latest_arn.split(':')[-1]
            rev = int(rev_str)
            print(f" Latest task definition revision (from list): {rev}")
            return rev
        except Exception as e2:
            print(f"  list_task_definitions failed: {e2}")
            return 0

# Removed broken deploy_watcher() function - use main() instead


def main():
    # Check for help flag
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python deploy_watcher.py")
        print("Deploys the StreamSniped Twitch Watcher infrastructure to AWS")
        print("Required environment variables:")
        print("  ECS_SUBNETS: Comma-separated list of subnet IDs")
        print("  ECS_SECURITY_GROUPS: Comma-separated list of security group IDs")
        print("  TWITCH_CLIENT_ID: Twitch API client ID")
        print("  TWITCH_CLIENT_SECRET: Twitch API client secret")
        print("  YOUTUBE_CLIENT_ID: YouTube API client ID")
        print("  YOUTUBE_CLIENT_SECRET: YouTube API client secret")
        print("  OPENROUTER_API_KEY: OpenRouter API key for AI analysis")
        print("  GEMINI_API_KEY: Gemini API key as fallback")
        print("  S3_BUCKET: S3 bucket name (optional, defaults to streamsniped-dev-videos)")
        print("  UPLOAD_VIDEOS: Enable video upload to S3 (optional, defaults to true)")
        print("  UPLOAD_YOUTUBE: Enable YouTube upload (optional, defaults to true)")
        sys.exit(0)
    
    # Get the latest task definition revision
    latest_revision = get_latest_task_definition_revision()
    if latest_revision <= 0:
        print("X No existing 'streamsniped-fargate' task definition found.")
        print("   Run: python aws-scripts/create_fargate_task_definition.py")
        return False
    
    deployer = WatcherDeployer()
    
    # Define environment variables for the Lambda function
    # In a real scenario, these would come from a config file or secrets manager
    lambda_env_vars = {
        'ECS_CLUSTER': 'streamsniped-dev-cluster',
        'ECS_TASK_DEFINITION': f'streamsniped-fargate:{latest_revision}',  # Use latest revision automatically
        'ECS_TASK_DEFINITION_GPU': f'streamsniped-fargate:{latest_revision}',  # Same for consistency
        'GPU_MODE': os.getenv('GPU_MODE', 'false'),  # Default: CPU-only
        'ENABLE_TRANSITIONS': os.getenv('ENABLE_TRANSITIONS', 'false'),
        # RAG pipeline settings
        'RAG_ENABLED': 'true',
        'EMBEDDING_MODEL_NAME': 'all-MiniLM-L6-v2',
        'SCENE_DURATION': '300',
        'RUN_CLOSURE_ENABLED': 'true',
        'ECS_SUBNETS': os.getenv('ECS_SUBNETS'), # e.g., 'subnet-xxxxxxxx,subnet-yyyyyyyy'
        'ECS_SECURITY_GROUPS': os.getenv('ECS_SECURITY_GROUPS'), # e.g., 'sg-xxxxxxxx'
        'DYNAMODB_TABLE': 'streamsniped_jobs',
        'STATE_TABLE_NAME': deployer.state_table_name,
        'SESSION_TABLE_NAME': deployer.session_table_name,
        'STREAMERS': 'disguisedtoast,quarterjade,masayoshi',
        'TWITCH_CLIENT_ID': os.getenv('TWITCH_CLIENT_ID'),
        'TWITCH_CLIENT_SECRET': os.getenv('TWITCH_CLIENT_SECRET'),
        'YOUTUBE_CLIENT_ID': os.getenv('YOUTUBE_CLIENT_ID'),
        'YOUTUBE_CLIENT_SECRET': os.getenv('YOUTUBE_CLIENT_SECRET'),
        'OPENROUTER_API_KEY': os.getenv('OPENROUTER_API_KEY'),
        'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY'),
        'S3_BUCKET': os.getenv('S3_BUCKET', 'streamsniped-dev-videos'),
        'UPLOAD_VIDEOS': os.getenv('UPLOAD_VIDEOS', 'true'),
        'UPLOAD_YOUTUBE': os.getenv('UPLOAD_YOUTUBE', 'true'),
        'AI_DIRECT_CLIPPER_SCORE_THRESHOLD': os.getenv('AI_DIRECT_CLIPPER_SCORE_THRESHOLD', '7.5'),
        'CLASSIFICATION_SECTIONS_SCORE_THRESHOLD': os.getenv('CLASSIFICATION_SECTIONS_SCORE_THRESHOLD', '7.5'),
        'HIGHLIGHT_SCORE_THRESHOLD': os.getenv('HIGHLIGHT_SCORE_THRESHOLD', '7.5'),
        # Clip generation offloading configuration
        'SKIP_CLIP_ENCODING_IN_CLOUD': 'true',
        'OFFLOAD_CLIPS': 'true',
        'CLIP_QUEUE_URL': f'https://sqs.us-east-1.amazonaws.com/{get_aws_account_id()}/streamsniped-clip-queue',
        'CLIP_LAYOUT_SYSTEM': 'advanced',
        'LOCAL_POSTPROCESS': os.getenv('LOCAL_POSTPROCESS', 'true'),
        # Local full-workflow queue for GPU processing
        'FULL_QUEUE_URL': f'https://sqs.us-east-1.amazonaws.com/{get_aws_account_id()}/streamsniped-full-queue',
        'GPU_PROCESSING_MODE': os.getenv('GPU_PROCESSING_MODE', 'local_only'),
        'GPU_ENABLED': os.getenv('GPU_ENABLED', 'true'),
        'USE_GPU_FOR_ALL_STEPS': os.getenv('USE_GPU_FOR_ALL_STEPS', 'true')
    }

    # Verify all environment variables are present
    if not all(lambda_env_vars.values()):
        print("X Missing one or more required environment variables.")
        print("   Please set: ECS_SUBNETS, ECS_SECURITY_GROUPS, TWITCH_CLIENT_ID, TWITCH_CLIENT_SECRET, YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, OPENROUTER_API_KEY, GEMINI_API_KEY")
        return False
    else:
        if deployer.create_state_table() and deployer.create_session_table():
            # Initialize state timestamps with current UTC time
            if not deployer.initialize_state_timestamps():
                print("X State timestamp initialization failed.")
                return False
            
            # Create CloudWatch log group
            if not deployer.create_cloudwatch_log_group():
                print("X CloudWatch log group creation failed.")
                return False
            
            role_arn = deployer.create_iam_role_for_lambda()
            if role_arn:
                print(f"\n🎉 IAM Role setup complete. Role ARN: {role_arn}")
                function_arn = deployer.deploy_lambda_function(role_arn, lambda_env_vars)
                if function_arn:
                    print(f"\n🎉 Lambda function deployment complete. Function ARN: {function_arn}")
                    rule_arn = deployer.create_eventbridge_rule(function_arn)
                    if rule_arn:
                        print(f"\n🎉 EventBridge rule deployment complete. Rule ARN: {rule_arn}")
                        print("\n✨ Full deployment successful! The watcher is now active.")
                        print(f" Monitoring streamers: {os.getenv('STREAMERS', 'disguisedtoast,quarterjade,masayoshi')}")
                        print(f"⚡ Next check: Within 4 hours")
                        print(f"🎥 Video features: GPU acceleration (NVENC) + Random transitions")
                        print(f"📊 CloudWatch logs: /aws/lambda/{deployer.lambda_function_name}")
                        return True
                    else:
                        print("\nX EventBridge rule deployment failed.")
                        return False
                else:
                    print("\nX Lambda function deployment failed.")
                    return False
            else:
                print("\nX IAM Role setup failed.")
                return False
        else:
            print("\nX State table creation failed.")
            return False

if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1) 