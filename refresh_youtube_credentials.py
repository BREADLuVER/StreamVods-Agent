#!/usr/bin/env python3
"""
Refresh YouTube OAuth credentials
This will get fresh tokens and update youtube_credentials.json
"""

import os
import sys
import json
from pathlib import Path

# Add processing-scripts to path
sys.path.insert(0, str(Path(__file__).parent / "processing-scripts"))

try:
    from youtube_uploader import YouTubeUploader
except ImportError as e:
    print(f"X Failed to import YouTubeUploader: {e}")
    print("💡 Make sure you're running this from the project root directory")
    sys.exit(1)

def main():
    print("🔄 Refreshing YouTube OAuth Credentials")
    print("=" * 50)
    
    # Check if credentials file exists
    creds_file = "youtube_credentials.json"
    if os.path.exists(creds_file):
        print(f"🔍 Found existing credentials file: {creds_file}")
        
        # Load existing credentials to get client_id and client_secret
        try:
            with open(creds_file, 'r') as f:
                existing_creds = json.load(f)
            
            client_id = existing_creds.get('client_id')
            client_secret = existing_creds.get('client_secret')
            
            if not client_id or not client_secret:
                print("X Missing client_id or client_secret in existing credentials")
                print("💡 You need to set up OAuth credentials first")
                print("💡 Run: python utils/setup_youtube_auth.py")
                sys.exit(1)
            
            print(f"✅ Found client credentials: {client_id[:20]}...")
            
        except Exception as e:
            print(f"X Error reading existing credentials: {e}")
            sys.exit(1)
    else:
        print("X No existing credentials file found")
        print("💡 You need to set up OAuth credentials first")
        print("💡 Run: python utils/setup_youtube_auth.py")
        sys.exit(1)
    
    # Initialize uploader with existing client credentials
    uploader = YouTubeUploader(client_id, client_secret, creds_file)
    
    print("\n🚀 Starting OAuth flow to get fresh tokens...")
    print("💡 IMPORTANT: This will request offline access for auto-refresh tokens")
    print("📝 This will open a browser window for authentication")
    print("📝 Complete the OAuth flow to get fresh tokens")
    
    try:
        success = uploader.authenticate()
        if success:
            print("\n🎉 YouTube credentials refreshed successfully!")
            print(f"💾 Fresh credentials saved to: {creds_file}")
            print("\n✅ You can now use YouTube uploads locally and in cloud!")
            print("🔄 Tokens will auto-refresh when expired")
            
            # Check if we have a refresh token
            if os.path.exists(creds_file):
                try:
                    with open(creds_file, 'r') as f:
                        creds_data = json.load(f)
                    if not creds_data.get('refresh_token'):
                        print("\n⚠️  WARNING: No refresh token in credentials!")
                        print("💡 You may need to revoke access and re-authenticate:")
                        print("   1. Go to https://myaccount.google.com/permissions")
                        print("   2. Remove 'StreamSniped' or your app name")
                        print("   3. Run this script again")
                except Exception:
                    pass
        else:
            print("\nX Credential refresh failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n❌ Authentication cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nX Authentication failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
