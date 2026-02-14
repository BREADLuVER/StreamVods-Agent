#!/usr/bin/env python3
"""
Run GPU clip generation workflow directly on Windows PC using Docker.

This handles the complete clip generation pipeline:
1. Download required files from S3
2. Create clips with advanced layout detection
3. Merge clips and generate metadata
4. Upload to YouTube (optional)

Usage:
  python run_gpu_clip_workflow.py <VOD_ID> [--quality 1080p] [--from-manifest <S3_URI>]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('vod_id', nargs='?', help='Twitch VOD ID (optional if --from-manifest provided)')
    parser.add_argument('--quality', default=os.getenv('QUALITY', '1080p'))
    parser.add_argument('--from-manifest', help='Process a clip generation manifest S3 URI (used by daemon)')
    args = parser.parse_args()
    
    # Validate arguments
    if not args.vod_id and not args.from_manifest:
        print("X VOD ID or --from-manifest required")
        sys.exit(1)

    # Get ECR image
    aws_account_id = os.getenv('AWS_ACCOUNT_ID', '590184039189')
    region = os.getenv('AWS_REGION', 'us-east-1')
    repo_name = os.getenv('ECR_REPO', 'streamsniped-dev')
    image = f"{aws_account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}:gpu"

    # Get AWS credentials from AWS CLI
    try:
        import subprocess
        result = subprocess.run(['aws', 'configure', 'get', 'aws_access_key_id'], 
                              capture_output=True, text=True, check=True)
        aws_access_key = result.stdout.strip()
        
        result = subprocess.run(['aws', 'configure', 'get', 'aws_secret_access_key'], 
                              capture_output=True, text=True, check=True)
        aws_secret_key = result.stdout.strip()
        
        aws_region = os.getenv('AWS_REGION', 'us-east-1')
        
        if not aws_access_key or not aws_secret_key:
            print("X AWS credentials not found. Please run 'aws configure' first")
            sys.exit(1)
            
    except Exception as e:
        print(f"X Error getting AWS credentials: {e}")
        print("Please run 'aws configure' to set up your credentials")
        sys.exit(1)
    
    # Build Docker command - mount the project root based on this file's location, not CWD
    project_root = Path(__file__).resolve().parent.parent
    docker_cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "-e", f"VOD_ID={args.vod_id or ''}",
        "-e", f"CLIP_MANIFEST_URI={args.from_manifest or ''}",
        "-e", f"QUALITY={args.quality}",
        "-e", "CONTAINER_MODE=true",
        "-e", "FARGATE_MODE=false",
        "-e", "GPU_MODE=true",
        "-e", "CLIP_LAYOUT_SYSTEM=advanced",  # Enable new layout system
        # Pass-through optional layout override so cloud runs can force A/B/C/JC deterministically
        "-e", f"LAYOUT_FORCE={os.getenv('LAYOUT_FORCE', '')}",
        "-e", f"S3_BUCKET={os.getenv('S3_BUCKET', 'streamsniped-dev-videos')}",
        "-e", f"UPLOAD_YOUTUBE={os.getenv('UPLOAD_YOUTUBE', 'true')}",
        "-e", f"AWS_ACCESS_KEY_ID={aws_access_key}",
        "-e", f"AWS_SECRET_ACCESS_KEY={aws_secret_key}",
        "-e", f"AWS_REGION={aws_region}",
        "-e", "FFMPEG_PATH=ffmpeg",
        "-e", "FFMPEG_BINARY=/usr/bin/ffmpeg",
        # Mount the entire repo so the container uses the latest local code without rebuilding
        "-v", f"{project_root}:/app",
        "--entrypoint", "/usr/bin/python3",
        image,
        "clip_creation/processing-scripts/run_clip_generation_workflow.py",
    ]

    # Add arguments
    if args.from_manifest:
        docker_cmd.extend(["--from-manifest", args.from_manifest])
    if args.vod_id:
        docker_cmd.append(args.vod_id)

    print(f"🚀 Running GPU clip generation workflow locally on Windows PC")
    print(f"📺 VOD ID: {args.vod_id}")
    print(f"🎥 Quality: {args.quality}")
    print(f" Manifest: {args.from_manifest or 'None'}")
    print(f"🐳 Image: {image}")
    print(f"📁 Data volume: {Path.cwd()}/data → /app/data")
    print("=" * 60)

    try:
        subprocess.run(docker_cmd, check=True)
        print(" Clip generation workflow completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"X Clip generation workflow failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
