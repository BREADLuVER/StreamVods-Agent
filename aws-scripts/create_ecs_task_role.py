#!/usr/bin/env python3
"""
Create IAM role for ECS tasks to access S3 and YouTube.
"""

import boto3
import json

def create_ecs_task_role():
    """Create IAM role for ECS tasks"""
    iam = boto3.client('iam')
    
    role_name = 'streamsniped-ecs-task-role'
    
    # Trust policy for ECS tasks
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "ecs-tasks.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    # Permissions policy for S3 and YouTube API
    permissions_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:ListBucket"
                ],
                "Resource": [
                    "arn:aws:s3:::streamsniped-dev-videos",
                    "arn:aws:s3:::streamsniped-dev-videos/*"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                "Resource": "arn:aws:logs:*:*:*"
            }
        ]
    }
    
    try:
        # Create role
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='Task role for StreamSniped ECS tasks'
        )
        print(f"✅ Created ECS task role: {response['Role']['Arn']}")
        
        # Attach permissions policy
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName='StreamSnipedECSTaskPermissions',
            PolicyDocument=json.dumps(permissions_policy)
        )
        print(f"✅ Attached permissions policy to ECS task role {role_name}")
        
        return response['Role']['Arn']
        
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"ECS task role {role_name} already exists")
        return f"arn:aws:iam::{boto3.client('sts').get_caller_identity()['Account']}:role/{role_name}"
    except Exception as e:
        print(f"Error creating ECS task role: {e}")
        raise

if __name__ == '__main__':
    create_ecs_task_role()
