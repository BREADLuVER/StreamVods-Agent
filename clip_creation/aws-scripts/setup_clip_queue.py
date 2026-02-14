#!/usr/bin/env python3
"""
Set up SQS queue for clip generation offloading.

Usage:
  python aws-scripts/setup_clip_queue.py
"""

import boto3
import os
import sys


def main():
    # Get AWS credentials and region
    region = os.getenv('AWS_REGION', 'us-east-1')
    
    try:
        sqs = boto3.client('sqs', region_name=region)
    except Exception as e:
        print(f"X Failed to create SQS client: {e}")
        print("Please ensure AWS credentials are configured")
        sys.exit(1)
    
    # Queue configuration
    queue_name = 'streamsniped-clip-queue'
    aws_account_id = os.getenv('AWS_ACCOUNT_ID')
    if not aws_account_id:
        print("❌ AWS_ACCOUNT_ID environment variable not set")
        print("💡 Please run: .\\setup_env.ps1")
        sys.exit(1)
    queue_url = f"https://sqs.{region}.amazonaws.com/{aws_account_id}/{queue_name}"
    
    # Queue attributes for clip generation
    attributes = {
        'VisibilityTimeout': '7200',  # 2 hours for clip generation
        'MessageRetentionPeriod': '1209600',  # 14 days
        'ReceiveMessageWaitTimeSeconds': '20',  # Long polling
        'DelaySeconds': '0',
        'MaximumMessageSize': '262144',  # 256KB
    }
    
    try:
        # Check if queue already exists
        try:
            sqs.get_queue_attributes(QueueUrl=queue_url, AttributeNames=['All'])
            print(f" Queue already exists: {queue_url}")
            return
        except sqs.exceptions.QueueDoesNotExist:
            pass
        
        # Create the queue
        print(f"🔄 Creating SQS queue: {queue_name}")
        response = sqs.create_queue(
            QueueName=queue_name,
            Attributes=attributes
        )
        
        created_url = response['QueueUrl']
        print(f" Queue created successfully: {created_url}")
        print(f" Queue URL for environment: {created_url}")
        
        # Set up dead letter queue for failed messages
        try:
            dlq_name = f"{queue_name}-dlq"
            dlq_response = sqs.create_queue(
                QueueName=dlq_name,
                Attributes={
                    'MessageRetentionPeriod': '1209600',  # 14 days
                }
            )
            dlq_url = dlq_response['QueueUrl']
            dlq_arn = f"arn:aws:sqs:{region}:{aws_account_id}:{dlq_name}"
            
            # Configure main queue to use DLQ
            redrive_policy = {
                'deadLetterTargetArn': dlq_arn,
                'maxReceiveCount': '3'
            }
            
            sqs.set_queue_attributes(
                QueueUrl=created_url,
                Attributes={
                    'RedrivePolicy': str(redrive_policy).replace("'", '"')
                }
            )
            
            print(f" Dead letter queue created: {dlq_url}")
            
        except Exception as e:
            print(f" Failed to create dead letter queue: {e}")
        
    except Exception as e:
        print(f"X Failed to create queue: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
