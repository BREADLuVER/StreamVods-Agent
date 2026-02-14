#!/usr/bin/env python3
"""
Create IAM role for Lambda function to trigger ECS tasks.
"""

import boto3
import json

def create_lambda_execution_role():
    """Create IAM role for Lambda function"""
    iam = boto3.client('iam')
    
    role_name = 'streamsniped-lambda-execution-role'
    
    # Trust policy for Lambda
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "lambda.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    # Permissions policy for ECS and S3
    permissions_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                "Resource": "arn:aws:logs:*:*:*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "ecs:RunTask",
                    "ecs:DescribeTasks",
                    "ecs:DescribeTaskDefinition"
                ],
                "Resource": "*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "iam:PassRole"
                ],
                "Resource": [
                    "arn:aws:iam::*:role/ecsTaskExecutionRole",
                    "arn:aws:iam::*:role/streamsniped-ecs-task-role"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
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
                    "dynamodb:GetItem",
                    "dynamodb:PutItem",
                    "dynamodb:UpdateItem",
                    "dynamodb:DeleteItem",
                    "dynamodb:Query",
                    "dynamodb:Scan"
                ],
                "Resource": [
                    "arn:aws:dynamodb:*:*:table/streamsniped_jobs",
                    "arn:aws:dynamodb:*:*:table/streamsniped-watcher-state-dev",
                    "arn:aws:dynamodb:*:*:table/streamsniped-sessions-dev"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "sqs:SendMessage"
                ],
                "Resource": "arn:aws:sqs:*:*:streamsniped-*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "sts:GetCallerIdentity"
                ],
                "Resource": "*"
            }
        ]
    }
    
    try:
        # Create role
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='Execution role for StreamSniped Lambda functions'
        )
        print(f"✅ Created IAM role: {response['Role']['Arn']}")
        
        # Attach permissions policy
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName='StreamSnipedLambdaPermissions',
            PolicyDocument=json.dumps(permissions_policy)
        )
        print(f"✅ Attached permissions policy to role {role_name}")
        
        return response['Role']['Arn']
        
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"IAM role {role_name} already exists")
        return f"arn:aws:iam::{boto3.client('sts').get_caller_identity()['Account']}:role/{role_name}"
    except Exception as e:
        print(f"Error creating IAM role: {e}")
        raise

if __name__ == '__main__':
    create_lambda_execution_role()
