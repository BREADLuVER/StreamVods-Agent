#!/usr/bin/env python3
"""
YouTube OAuth Setup Script
Sets up YouTube API OAuth 2.0 authentication for StreamSniped
Creates youtube_credentials.json for video uploads
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Import from processing-scripts directory
    processing_scripts_dir = Path(__file__).parent / "processing-scripts"
    sys.path.insert(0, str(processing_scripts_dir))
    from youtube_uploader import YouTubeUploader
except ImportError:
    print("X Failed to import YouTubeUploader")
    print("💡 Make sure you're running this from the project root directory")
    sys.exit(1)


def main():
    print("🔐 YouTube OAuth 2.0 Setup")
    print("=" * 40)
    
    # Get credentials from environment
    client_id = os.getenv("YOUTUBE_CLIENT_ID")
    client_secret = os.getenv("YOUTUBE_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        print("X Missing YouTube OAuth credentials in environment variables")
        print("💡 Make sure you have set:")
        print("   YOUTUBE_CLIENT_ID")
        print("   YOUTUBE_CLIENT_SECRET")
        print("\nExample:")
        print('$env:YOUTUBE_CLIENT_ID="your-client-id"')
        print('$env:YOUTUBE_CLIENT_SECRET="your-client-secret"')
        sys.exit(1)
    
    print(f" Found YouTube client ID: {client_id[:20]}...")
    print(f" Found YouTube client secret: {client_secret[:10]}...")
    
    # Check if credentials file already exists
    credentials_file = "john_credentials.json"
    if os.path.exists(credentials_file):
        print(f"🔍 Found existing credentials file: {credentials_file}")
        response = input("Do you want to re-authenticate? (y/N): ").lower().strip()
        if response != 'y':
            print(" Keeping existing YouTube credentials")
            return
        else:
            print("🔄 Re-authenticating...")
    
    # Initialize YouTube uploader and run OAuth flow
    print("\n🚀 Starting YouTube OAuth setup...")
    print("💡 IMPORTANT: This will request offline access for auto-refresh tokens")
    uploader = YouTubeUploader(client_id, client_secret, credentials_file)
    
    try:
        success = uploader.authenticate()
        if success:
            print("\n🎉 YouTube OAuth setup completed successfully!")
            print(f"💾 Credentials saved to: {credentials_file}")
            print("\n✅ You can now upload videos to YouTube!")
            print("🔄 Tokens will auto-refresh when expired")
            print("💡 Run your workflow again - YouTube upload should work now")
            
            # Check if we have a refresh token
            if os.path.exists(credentials_file):
                try:
                    with open(credentials_file, 'r') as f:
                        creds_data = json.load(f)
                    if not creds_data.get('refresh_token'):
                        print("\n⚠️  WARNING: No refresh token in credentials!")
                        print("💡 You may need to revoke access and re-authenticate:")
                        print("   1. Go to https://myaccount.google.com/permissions")
                        print("   2. Remove 'StreamSniped' or your app name")
                        print("   3. Run authentication again")
                except Exception:
                    pass
        else:
            print("\nX YouTube OAuth setup failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nX Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()