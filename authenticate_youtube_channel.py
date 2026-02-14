#!/usr/bin/env python3
"""
Authenticate a specific YouTube channel
Gets fresh OAuth tokens and saves them to the channel's credentials file
"""

import os
import sys
import json
import argparse
from pathlib import Path

try:
    from google_auth_oauthlib.flow import InstalledAppFlow
except ImportError as e:
    print(f"❌ Missing required Google API libraries: {e}")
    print("💡 Install with: pip install google-auth google-auth-oauthlib google-api-python-client")
    sys.exit(1)


def load_channels_config():
    """Load youtube_channels.json configuration"""
    config_path = Path("config/youtube_channels.json")
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return {}


def authenticate_channel(channel_name: str, client_id: str = None, client_secret: str = None):
    """Authenticate a YouTube channel and save credentials"""
    print(f"🔄 Authenticating YouTube channel: {channel_name}")
    print("=" * 60)
    
    # Load channel configuration
    config = load_channels_config()
    if not config or 'channels' not in config:
        print("❌ No channels configured in config/youtube_channels.json")
        return False
    
    channels = config.get('channels', {})
    if channel_name not in channels:
        print(f"❌ Channel '{channel_name}' not found in configuration")
        print(f"💡 Available channels: {', '.join(channels.keys())}")
        return False
    
    channel_config = channels[channel_name]
    creds_file = channel_config.get('credentials_file')
    
    if not creds_file:
        print(f"❌ No credentials_file configured for channel '{channel_name}'")
        return False
    
    print(f"📁 Credentials file: {creds_file}")
    
    # Get client credentials
    if not client_id or not client_secret:
        # Try to load from existing credentials file
        if os.path.exists(creds_file):
            print("🔍 Loading client credentials from existing file...")
            try:
                with open(creds_file, 'r') as f:
                    existing_creds = json.load(f)
                
                client_id = existing_creds.get('client_id')
                client_secret = existing_creds.get('client_secret')
            except Exception as e:
                print(f"⚠️  Could not read existing credentials: {e}")
        
        # Try to load from default channel credentials
        if not client_id or not client_secret:
            default_creds_file = channels.get('default', {}).get('credentials_file', 'youtube_credentials/youtube_credentials.json')
            if os.path.exists(default_creds_file):
                print("🔍 Loading client credentials from default channel...")
                try:
                    with open(default_creds_file, 'r') as f:
                        default_creds = json.load(f)
                    
                    client_id = default_creds.get('client_id')
                    client_secret = default_creds.get('client_secret')
                except Exception as e:
                    print(f"⚠️  Could not read default credentials: {e}")
    
    if not client_id or not client_secret:
        print("❌ Missing client_id or client_secret")
        print("💡 Either:")
        print("   1. Provide --client-id and --client-secret arguments")
        print("   2. Have existing credentials with client credentials")
        print("   3. Set up default channel with client credentials first")
        return False
    
    print(f"✅ Using client_id: {client_id[:20]}...")
    
    # Perform OAuth flow
    try:
        flow = InstalledAppFlow.from_client_config(
            {
                "installed": {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [
                        "http://localhost:8080",
                        "http://localhost:8080/",
                    ]
                }
            },
            scopes=[
                'https://www.googleapis.com/auth/youtube.upload',
                'https://www.googleapis.com/auth/youtube.readonly'
            ]
        )
        
        print("\n🚀 Starting OAuth flow...")
        print(f"⚠️  CRITICAL: Log in with the Google account for '{channel_name}' channel")
        print("⚠️  Do NOT use the same Google account for different channels!")
        print("=" * 60)
        
        # Run local server flow on port 8080 (consistent with OAuth config)
        # Don't auto-open browser - user needs to manually open and log in with correct account
        # Request offline access to get a long-lived refresh token
        flow.redirect_uri = 'http://localhost:8080/'
        
        credentials = flow.run_local_server(
            port=8080,
            open_browser=False,
            access_type='offline',
            prompt='consent'
        )
        
        # Ensure we got a refresh token
        if not credentials.refresh_token:
            print("\n⚠️  WARNING: No refresh token received!")
            print("💡 This may happen if you've already authorized this app before.")
            print("💡 To fix: Revoke access at https://myaccount.google.com/permissions")
            print("💡 Then run authentication again to get a fresh refresh token.")
        
        # Ensure directory exists
        creds_path = Path(creds_file)
        creds_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save credentials with client info for token refresh
        creds_dict = json.loads(credentials.to_json())
        creds_dict['client_id'] = client_id
        creds_dict['client_secret'] = client_secret
        
        with open(creds_file, 'w') as f:
            json.dump(creds_dict, f, indent=2)
        
        print(f"\n💾 Saved credentials with refresh token: {bool(credentials.refresh_token)}")
        
        print(f"\n✅ Successfully authenticated channel: {channel_name}")
        print(f"💾 Credentials saved to: {creds_file}")
        
        # Verify by getting channel ID
        try:
            from googleapiclient.discovery import build
            service = build('youtube', 'v3', credentials=credentials)
            response = service.channels().list(part="snippet", mine=True).execute()
            
            if response.get('items'):
                channel_info = response['items'][0]
                authenticated_channel_id = channel_info['id']
                channel_title = channel_info['snippet']['title']
                
                print("\n📺 Authenticated Channel Info:")
                print(f"   Title: {channel_title}")
                print(f"   Channel ID: {authenticated_channel_id}")
                
                # Check if it matches expected channel ID
                expected_channel_id = channel_config.get('channel_id')
                if expected_channel_id and expected_channel_id != authenticated_channel_id:
                    print("\n❌ ERROR: Channel ID mismatch!")
                    print(f"   Expected: {expected_channel_id}")
                    print(f"   Got: {authenticated_channel_id}")
                    print("   You authenticated with the WRONG Google account!")
                    print(f"   Delete {creds_file} and try again with the correct account")
                    return False
                elif expected_channel_id:
                    print("✅ Channel ID matches configuration")
            else:
                print("\n❌ ERROR: Could not get channel info - authentication failed")
                return False
        except Exception as e:
            print(f"\n❌ ERROR: Could not verify channel info: {e}")
            print("   Authentication may have failed")
            return False
        
        return True
        
    except KeyboardInterrupt:
        print("\n❌ Authentication cancelled by user")
        return False
    except Exception as e:
        print(f"\n❌ Authentication failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Authenticate a YouTube channel')
    parser.add_argument('--channel', type=str, required=False,
                       help='Channel name from youtube_channels.json')
    parser.add_argument('--client-id', type=str,
                       help='OAuth client ID (optional, will try to load from existing credentials)')
    parser.add_argument('--client-secret', type=str,
                       help='OAuth client secret (optional, will try to load from existing credentials)')
    parser.add_argument('--list', action='store_true',
                       help='List available channels and exit')
    
    args = parser.parse_args()
    
    # List channels mode
    if args.list:
        config = load_channels_config()
        if config and 'channels' in config:
            print("📋 Available channels:")
            for name, cfg in config['channels'].items():
                creds_file = cfg.get('credentials_file', 'N/A')
                channel_id = cfg.get('channel_id', 'N/A')
                print(f"  • {name}")
                print(f"    Credentials: {creds_file}")
                print(f"    Channel ID: {channel_id}")
        return
    
    # Authenticate channel
    if not args.channel:
        parser.error('--channel is required unless --list is used')
    success = authenticate_channel(args.channel, args.client_id, args.client_secret)
    
    if success:
        print("\n🎉 Authentication complete!")
        print(f"💡 You can now upload videos to the '{args.channel}' channel")
    else:
        print("\n❌ Authentication failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
