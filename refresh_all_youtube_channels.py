#!/usr/bin/env python3
"""
Refresh all YouTube channel credentials
Checks all channels in youtube_channels.json and refreshes expired tokens
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict

# Add processing-scripts to path
sys.path.insert(0, str(Path(__file__).parent / "processing-scripts"))

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
except ImportError as e:
    print(f"❌ Missing required Google API libraries: {e}")
    print("💡 Install with: pip install google-auth google-auth-oauthlib google-api-python-client")
    sys.exit(1)


def load_channels_config() -> Dict:
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


def get_credential_status(creds_file: str) -> Dict:
    """Check credential file status"""
    if not os.path.exists(creds_file):
        return {"exists": False, "error": "File not found"}
    
    try:
        with open(creds_file, 'r') as f:
            cred_data = json.load(f)
        
        # Check required fields
        has_refresh_token = bool(cred_data.get('refresh_token'))
        has_client_id = bool(cred_data.get('client_id'))
        has_client_secret = bool(cred_data.get('client_secret'))
        
        # Check expiry
        expiry_str = cred_data.get('expiry')
        is_expired = False
        expires_in_hours = None
        
        if expiry_str:
            try:
                expiry = datetime.fromisoformat(expiry_str.replace('Z', '+00:00'))
                now = datetime.now(timezone.utc)
                is_expired = expiry < now
                
                if not is_expired:
                    delta = expiry - now
                    expires_in_hours = delta.total_seconds() / 3600
            except Exception:
                pass
        
        return {
            "exists": True,
            "has_refresh_token": has_refresh_token,
            "has_client_id": has_client_id,
            "has_client_secret": has_client_secret,
            "is_expired": is_expired,
            "expires_in_hours": expires_in_hours,
            "expiry": expiry_str,
            "client_id": cred_data.get('client_id', '')[:20] + "...",
        }
    except Exception as e:
        return {"exists": True, "error": str(e)}


def refresh_credential(creds_file: str, channel_name: str, force_reauth: bool = False) -> bool:
    """Refresh a single credential file"""
    print(f"\n{'='*60}")
    print(f"🔄 Processing channel: {channel_name}")
    print(f"📁 Credentials file: {creds_file}")
    
    status = get_credential_status(creds_file)
    
    if not status.get("exists"):
        print("❌ Credentials file not found - needs initial setup")
        return False
    
    if "error" in status and status["error"] != "File not found":
        print(f"❌ Error reading credentials: {status['error']}")
        return False
    
    # Load credentials
    try:
        with open(creds_file, 'r') as f:
            cred_data = json.load(f)
        
        credentials = Credentials.from_authorized_user_file(creds_file)
        
        # Check if we need to refresh
        if not credentials.expired and not force_reauth:
            expires_in = status.get('expires_in_hours')
            if expires_in and expires_in > 1:
                print(f"✅ Token is valid (expires in {expires_in:.1f} hours)")
                return True
        
        # Try to refresh
        if credentials.expired and credentials.refresh_token and not force_reauth:
            print("🔄 Token expired, attempting refresh...")
            try:
                credentials.refresh(Request())
                
                # Save refreshed credentials with client info preserved
                creds_dict = json.loads(credentials.to_json())
                creds_dict['client_id'] = cred_data.get('client_id')
                creds_dict['client_secret'] = cred_data.get('client_secret')
                
                with open(creds_file, 'w') as f:
                    json.dump(creds_dict, f, indent=2)
                
                print(f"✅ Successfully refreshed token for {channel_name}")
                return True
                
            except Exception as refresh_error:
                error_str = str(refresh_error)
                print(f"⚠️ Refresh failed: {refresh_error}")
                
                # Check if it's an invalid_grant error (revoked token)
                if 'invalid_grant' in error_str.lower() or 'revoked' in error_str.lower():
                    print("❌ Refresh token has been revoked - re-authentication required")
                    print(f"💡 Run: python authenticate_youtube_channel.py --channel {channel_name}")
                    return False
                
                print(f"❌ Refresh failed with error: {error_str}")
                return False
        
        # Token is valid but user requested force re-auth
        if force_reauth:
            print("🔄 Force re-authentication requested...")
            client_id = cred_data.get('client_id')
            client_secret = cred_data.get('client_secret')
            
            if not client_id or not client_secret:
                print("❌ Missing client_id or client_secret in credentials")
                return False
            
            # Perform OAuth flow
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
            
            print("🔄 Starting OAuth flow...")
            print(f"⚠️  CRITICAL: Log in with the Google account for '{channel_name}' channel")
            print("⚠️  Do NOT use the same Google account for different channels!")
            print("=" * 60)
            
            try:
                # Use run_local_server which handles everything including printing URL
                # The redirect_uri needs to match OAuth config
                flow.redirect_uri = 'http://localhost:8080/'
                
                # This will print the URL automatically and wait for callback
                credentials = flow.run_local_server(
                    port=8080,
                    open_browser=False,
                    access_type='offline',
                    prompt='consent'
                )
                
                # Verify we got a refresh token
                if not credentials.refresh_token:
                    print("\n⚠️  WARNING: No refresh token received!")
                    print("💡 Revoke access at https://myaccount.google.com/permissions")
                    print("💡 Then run authentication again.")
                
                # Save new credentials with client info for future refreshes
                creds_dict = json.loads(credentials.to_json())
                creds_dict['client_id'] = client_id
                creds_dict['client_secret'] = client_secret
                
                with open(creds_file, 'w') as f:
                    json.dump(creds_dict, f, indent=2)
                
                print(f"✅ Successfully re-authenticated {channel_name}")
                print(f"💾 Saved with refresh token: {bool(credentials.refresh_token)}")
                return True
                
            except Exception as e:
                print(f"❌ Re-authentication failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing credentials: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Refresh all YouTube channel credentials')
    parser.add_argument('--force', action='store_true', 
                       help='Force re-authentication for all channels (even if valid)')
    parser.add_argument('--channel', type=str,
                       help='Only refresh specific channel')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check status, do not refresh')
    
    args = parser.parse_args()
    
    print("🔄 YouTube Channel Credentials Manager")
    print("=" * 60)
    
    # Load channel configuration
    config = load_channels_config()
    if not config or 'channels' not in config:
        print("❌ No channels configured in config/youtube_channels.json")
        sys.exit(1)
    
    channels = config['channels']
    print(f"📋 Found {len(channels)} channel(s) in configuration")
    
    # Check status of all channels
    print("\n" + "=" * 60)
    print("📊 Channel Status Summary")
    print("=" * 60)
    
    channel_statuses = {}
    for channel_name, channel_config in channels.items():
        creds_file = channel_config.get('credentials_file')
        if not creds_file:
            print(f"⚠️  {channel_name}: No credentials_file configured")
            continue
        
        status = get_credential_status(creds_file)
        channel_statuses[channel_name] = status
        
        # Print status
        if not status.get("exists"):
            print(f"❌ {channel_name}: Credentials not found")
        elif "error" in status:
            print(f"⚠️  {channel_name}: Error - {status['error']}")
        elif status.get("is_expired"):
            print(f"❌ {channel_name}: Token EXPIRED (since {status.get('expiry')})")
        elif status.get("expires_in_hours"):
            hours = status.get("expires_in_hours")
            if hours < 24:
                print(f"⚠️  {channel_name}: Expires in {hours:.1f} hours")
            else:
                days = hours / 24
                print(f"✅ {channel_name}: Valid (expires in {days:.1f} days)")
        else:
            print(f"⚠️  {channel_name}: Unknown expiry status")
    
    # Exit if check-only mode
    if args.check_only:
        # Check if any channels are expired or have errors
        has_issues = any(
            not status.get("exists") or 
            status.get("error") or 
            status.get("is_expired")
            for status in channel_statuses.values()
        )
        
        if has_issues:
            print("\n⚠️  Status check complete - issues found")
            sys.exit(1)
        else:
            print("\n✅ Status check complete - all tokens valid")
            sys.exit(0)
    
    # Refresh channels
    print("\n" + "=" * 60)
    print("🔄 Refreshing Credentials")
    print("=" * 60)
    
    success_count = 0
    failed_channels = []
    
    for channel_name, channel_config in channels.items():
        # Skip if specific channel requested and this isn't it
        if args.channel and channel_name != args.channel:
            continue
        
        creds_file = channel_config.get('credentials_file')
        if not creds_file:
            continue
        
        success = refresh_credential(creds_file, channel_name, force_reauth=args.force)
        
        if success:
            success_count += 1
        else:
            failed_channels.append(channel_name)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Refresh Summary")
    print("=" * 60)
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {len(failed_channels)}")
    
    if failed_channels:
        print(f"\n⚠️  Failed channels: {', '.join(failed_channels)}")
        print("\n💡 To manually re-authenticate a channel:")
        print("   python authenticate_youtube_channel.py --channel <channel_name>")
        sys.exit(1)
    else:
        print("\n🎉 All channels refreshed successfully!")


if __name__ == "__main__":
    main()

