#!/usr/bin/env python3
"""
Update the Lambda function with the age filter fix to prevent old VODs from being re-queued.
"""

import boto3
import zipfile
import tempfile
import os
from pathlib import Path

def update_lambda_function():
    """Update the Lambda function with the fixed code"""
    lambda_client = boto3.client('lambda')
    
    # Read the updated Lambda code
    lambda_code_path = Path('aws-scripts/twitch_watcher_lambda.py')
    if not lambda_code_path.exists():
        print("❌ Lambda code file not found")
        return False
    
    # Create a temporary zip file
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
        with zipfile.ZipFile(tmp_file.name, 'w') as zip_file:
            # Add the Lambda function code
            zip_file.write(lambda_code_path, 'lambda_function.py')
            
            # Add any dependencies if they exist
            if Path('twitch_monitor.py').exists():
                zip_file.write('twitch_monitor.py', 'twitch_monitor.py')
        
        try:
            # Update the Lambda function
            with open(tmp_file.name, 'rb') as zip_file:
                response = lambda_client.update_function_code(
                    FunctionName='streamsniped-watcher-dev-function',
                    ZipFile=zip_file.read()
                )
            
            print(f"✅ Updated Lambda function: {response['FunctionArn']}")
            print("✅ Age filter fix deployed - VODs older than 7 days will be skipped")
            return True
            
        except Exception as e:
            print(f"❌ Error updating Lambda function: {e}")
            return False
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_file.name)
            except:
                pass

if __name__ == '__main__':
    update_lambda_function()
