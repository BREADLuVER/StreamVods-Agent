#!/usr/bin/env python3
"""
Quick setup script to add YouTube credentials to .env file
"""

import os
from pathlib import Path

def main():
    print("🔐 YouTube API Environment Setup")
    print("=" * 40)
    
    # Get YouTube OAuth credentials from environment
    client_id = os.getenv("YOUTUBE_CLIENT_ID")
    client_secret = os.getenv("YOUTUBE_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        print("❌ YouTube credentials not found in environment variables")
        print("💡 Please set YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET in your environment")
        print("   You can use: .\\setup_env.ps1")
        return
    
    # Check if .env file exists
    env_file = Path(".env")
    env_content = ""
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            env_content = f.read()
        print(f" Found existing .env file")
    else:
        print(f"📝 Creating new .env file")
    
    # Check if YouTube variables already exist
    youtube_vars_exist = "YOUTUBE_CLIENT_ID" in env_content
    
    if youtube_vars_exist:
        print("🔍 YouTube credentials already configured in .env")
        response = input("Do you want to update them? (y/N): ").lower().strip()
        if response != 'y':
            print(" Keeping existing YouTube credentials")
            return
    
    # Add/update YouTube environment variables
    youtube_env_vars = f"""
# YouTube API OAuth Credentials
YOUTUBE_CLIENT_ID={client_id}
YOUTUBE_CLIENT_SECRET={client_secret}
"""
    
    if youtube_vars_exist:
        # Replace existing YouTube variables
        lines = env_content.split('\n')
        new_lines = []
        skip_youtube = False
        
        for line in lines:
            if line.startswith("# YouTube API OAuth Credentials"):
                skip_youtube = True
                continue
            elif line.startswith("YOUTUBE_CLIENT_ID") or line.startswith("YOUTUBE_CLIENT_SECRET"):
                continue
            elif skip_youtube and line.strip() == "":
                skip_youtube = False
                continue
            elif skip_youtube and not line.startswith("YOUTUBE_"):
                skip_youtube = False
                new_lines.append(line)
            elif not skip_youtube:
                new_lines.append(line)
        
        env_content = '\n'.join(new_lines) + youtube_env_vars
    else:
        # Append YouTube variables
        env_content += youtube_env_vars
    
    # Write updated .env file
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print(f" YouTube OAuth credentials added to .env file")
    print(f"📁 File: {env_file.absolute()}")
    print(f"\n Ready for YouTube uploads!")
    print(f"💡 Test with: python processing-scripts/youtube_uploader.py <vod_id>")

if __name__ == "__main__":
    main() 