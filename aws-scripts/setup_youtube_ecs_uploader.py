#!/usr/bin/env python3
"""
Setup script for YouTube uploader ECS infrastructure.
Creates task definition, Lambda function, and S3 event configuration.
"""

import json
import boto3
import os
from pathlib import Path

def get_aws_account_id():
    """Get AWS account ID"""
    sts = boto3.client('sts')
    return sts.get_caller_identity()['Account']

def create_task_definition():
    """Register ECS task definition"""
    ecs = boto3.client('ecs')
    
    task_def = {
        "family": "streamsniped-youtube-uploader",
        "networkMode": "awsvpc",
        "requiresCompatibilities": ["FARGATE"],
        "cpu": "512",
        "memory": "1024",
        "executionRoleArn": f"arn:aws:iam::{get_aws_account_id()}:role/ecsTaskExecutionRole",
        "taskRoleArn": f"arn:aws:iam::{get_aws_account_id()}:role/streamsniped-ecs-task-role",
        "containerDefinitions": [
            {
                "name": "youtube-uploader",
                "image": "public.ecr.aws/amazonlinux/amazonlinux:2023",
                "essential": True,
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": "/ecs/streamsniped-youtube-uploader",
                        "awslogs-region": "us-east-1",
                        "awslogs-stream-prefix": "ecs"
                    }
                },
                "environment": [
                    {"name": "AWS_REGION", "value": "us-east-1"},
                    {"name": "S3_BUCKET", "value": "streamsniped-dev-videos"}
                ],
                "secrets": [
                    {
                        "name": "YOUTUBE_CLIENT_ID",
                        "valueFrom": f"arn:aws:secretsmanager:us-east-1:{get_aws_account_id()}:secret:streamsniped/youtube-credentials:client_id::"
                    },
                    {
                        "name": "YOUTUBE_CLIENT_SECRET", 
                        "valueFrom": f"arn:aws:secretsmanager:us-east-1:{get_aws_account_id()}:secret:streamsniped/youtube-credentials:client_secret::"
                    },
                    {
                        "name": "YOUTUBE_REFRESH_TOKEN",
                        "valueFrom": f"arn:aws:secretsmanager:us-east-1:{get_aws_account_id()}:secret:streamsniped/youtube-credentials:refresh_token::"
                    }
                ],
                "command": [
                    "sh", "-c",
                    "yum update -y && yum install -y python3 python3-pip git && "
                    "pip3 install boto3 google-api-python-client google-auth-httplib2 google-auth-oauthlib && "
                    "git clone https://github.com/your-repo/StreamSniped-TwitchAnalyzer.git /app && "
                    "cd /app && "
                    "python3 -c \"import boto3; s3=boto3.client('s3'); s3.download_file('$S3_BUCKET', '$S3_KEY', '/tmp/input.mp4'); print('Downloaded from S3')\" && "
                    "python3 processing-scripts/auto_youtube_upload.py $VOD_ID --public --skip-missing --allow-s3-fallback"
                ]
            }
        ]
    }
    
    response = ecs.register_task_definition(**task_def)
    print(f"✅ Created task definition: {response['taskDefinition']['taskDefinitionArn']}")
    return response['taskDefinition']['taskDefinitionArn']

def create_lambda_function():
    """Create Lambda function for S3 trigger"""
    lambda_client = boto3.client('lambda')
    
    # Read Lambda code
    lambda_code = Path(__file__).parent / 's3_trigger_lambda.py'
    with open(lambda_code, 'r') as f:
        lambda_source = f.read()
    
    # Create deployment package
    import zipfile
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
        with zipfile.ZipFile(tmp.name, 'w') as zf:
            zf.writestr('lambda_function.py', lambda_source)
        
        with open(tmp.name, 'rb') as f:
            zip_content = f.read()
    
    os.unlink(tmp.name)
    
    # Create function
    function_name = 'streamsniped-s3-youtube-trigger'
    
    try:
        response = lambda_client.create_function(
            FunctionName=function_name,
            Runtime='python3.9',
            Role=f'arn:aws:iam::{get_aws_account_id()}:role/streamsniped-lambda-execution-role',
            Handler='lambda_function.lambda_handler',
            Code={'ZipFile': zip_content},
            Description='Trigger ECS tasks for YouTube uploads when S3 objects are created',
            Timeout=60,
            Environment={
                'Variables': {
                    'ECS_CLUSTER_ARN': f'arn:aws:ecs:us-east-1:{get_aws_account_id()}:cluster/streamsniped-dev-cluster',
                    'ECS_TASK_DEFINITION': 'streamsniped-youtube-uploader',
                    'SUBNET_IDS': 'subnet-03e6f7b9b1a11a703,subnet-07ce255e6f3637bb0,subnet-0914a211d2bd3a5fc,subnet-0e00cf81c265e1f5b,subnet-0c993fb8d9e2a1a03,subnet-02cf96d691897193f',
                    'SECURITY_GROUP_IDS': 'sg-0ba5e763404146e8a'
                }
            }
        )
        print(f"✅ Created Lambda function: {response['FunctionArn']}")
        return response['FunctionArn']
    except lambda_client.exceptions.ResourceConflictException:
        print(f"Lambda function {function_name} already exists")
        return f"arn:aws:lambda:us-east-1:{get_aws_account_id()}:function:{function_name}"

def configure_s3_events():
    """Configure S3 bucket notifications"""
    s3 = boto3.client('s3')
    lambda_client = boto3.client('lambda')
    
    bucket_name = 'streamsniped-dev-videos'
    function_arn = f"arn:aws:lambda:us-east-1:{get_aws_account_id()}:function:streamsniped-s3-youtube-trigger"
    
    # Add Lambda permission for S3
    try:
        lambda_client.add_permission(
            FunctionName='streamsniped-s3-youtube-trigger',
            StatementId='s3-trigger-permission',
            Action='lambda:InvokeFunction',
            Principal='s3.amazonaws.com',
            SourceArn=f'arn:aws:s3:::{bucket_name}'
        )
        print("✅ Added S3 permission to Lambda")
    except lambda_client.exceptions.ResourceConflictException:
        print("Lambda permission already exists")
    
    # Configure S3 bucket notification
    notification_config = {
        'LambdaFunctionConfigurations': [
            {
                'Id': 'youtube-upload-trigger',
                'LambdaFunctionArn': function_arn,
                'Events': ['s3:ObjectCreated:*'],
                'Filter': {
                    'Key': {
                        'FilterRules': [
                            {'Name': 'prefix', 'Value': 'videos/'},
                            {'Name': 'suffix', 'Value': '.mp4'}
                        ]
                    }
                }
            },
            {
                'Id': 'youtube-clips-trigger', 
                'LambdaFunctionArn': function_arn,
                'Events': ['s3:ObjectCreated:*'],
                'Filter': {
                    'Key': {
                        'FilterRules': [
                            {'Name': 'prefix', 'Value': 'clips/'},
                            {'Name': 'suffix', 'Value': '.mp4'}
                        ]
                    }
                }
            }
        ]
    }
    
    s3.put_bucket_notification_configuration(
        Bucket=bucket_name,
        NotificationConfiguration=notification_config
    )
    print(f"✅ Configured S3 notifications for bucket {bucket_name}")

def main():
    print("Setting up YouTube uploader ECS infrastructure...")
    
    # Create CloudWatch log group
    logs = boto3.client('logs')
    try:
        logs.create_log_group(logGroupName='/ecs/streamsniped-youtube-uploader')
        print("✅ Created CloudWatch log group")
    except logs.exceptions.ResourceAlreadyExistsException:
        print("CloudWatch log group already exists")
    
    # Create task definition
    task_def_arn = create_task_definition()
    
    # Create Lambda function
    lambda_arn = create_lambda_function()
    
    # Configure S3 events
    configure_s3_events()
    
    print("\n🎉 Setup complete!")
    print(f"Task Definition: {task_def_arn}")
    print(f"Lambda Function: {lambda_arn}")
    print("\nNext steps:")
    print("1. Update subnet and security group IDs in the Lambda environment")
    print("2. Ensure IAM roles have proper permissions")
    print("3. Test by uploading a video to S3")

if __name__ == '__main__':
    main()
