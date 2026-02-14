#!/usr/bin/env python3
"""
YouTube Uploader for StreamSniped
Handles YouTube video uploads with OAuth 2.0 authentication
"""

import os
import json
import sys
from typing import Dict, Optional

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload
except ImportError as e:
    print(f"X Missing required Google API libraries: {e}")
    print("💡 Install with: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")
    sys.exit(1)


class YouTubeUploader:
    """YouTube video uploader with OAuth 2.0 authentication"""
    
    def __init__(self, client_id: str = "", client_secret: str = "", credentials_file: str = "youtube_credentials.json"):
        self.client_id = client_id
        self.client_secret = client_secret
        self.credentials_file = credentials_file
        self.service = None
        self.credentials = None
    
    def authenticate(self) -> bool:
        """Authenticate with YouTube API using stored credentials or OAuth flow"""
        try:
            # Try to load existing credentials
            if os.path.exists(self.credentials_file):
                print(f"🔍 Loading existing credentials from {self.credentials_file}")
                
                # Load credentials and client info from JSON
                with open(self.credentials_file, 'r') as f:
                    cred_data = json.load(f)
                
                self.credentials = Credentials.from_authorized_user_file(self.credentials_file)
                
                # If we have client_id/secret in the JSON, use them for refresh
                if not self.client_id and 'client_id' in cred_data:
                    self.client_id = cred_data['client_id']
                if not self.client_secret and 'client_secret' in cred_data:
                    self.client_secret = cred_data['client_secret']
                
                # Refresh if expired
                if self.credentials.expired and self.credentials.refresh_token:
                    print("🔄 Refreshing expired credentials...")
                    try:
                        self.credentials.refresh(Request())
                        
                        # Save refreshed credentials with client info preserved
                        creds_dict = json.loads(self.credentials.to_json())
                        creds_dict['client_id'] = self.client_id
                        creds_dict['client_secret'] = self.client_secret
                        
                        with open(self.credentials_file, 'w') as f:
                            json.dump(creds_dict, f, indent=2)
                        print("💾 Refreshed credentials saved")
                    except Exception as refresh_error:
                        print(f"⚠️ Refresh failed: {refresh_error}")
                        print("🔄 Attempting full re-authentication...")
                        return self._oauth_flow()
                elif self.credentials.expired and not self.credentials.refresh_token:
                    print("⚠️ Credentials expired but no refresh token available!")
                    print("🔄 Re-authentication required...")
                    return self._oauth_flow()
                
                # Build service
                self.service = build('youtube', 'v3', credentials=self.credentials)
                print("✅ YouTube authentication successful")
                return True
            
            # If no credentials file and we have client_id/secret, do OAuth flow
            elif self.client_id and self.client_secret:
                print("🚀 Starting OAuth flow...")
                return self._oauth_flow()
            
            else:
                print("X No credentials found")
                print("💡 Either provide client_id/secret or ensure youtube_credentials.json exists")
                return False
                
        except Exception as e:
            print(f"X Authentication failed: {e}")
            # Try OAuth flow as fallback if we have client credentials
            if self.client_id and self.client_secret:
                print("🔄 Attempting OAuth flow as fallback...")
                return self._oauth_flow()
            return False

    def get_authenticated_channel_id(self) -> Optional[str]:
        """Return the channel ID for the authenticated account, if available."""
        try:
            if not self.service:
                return None
            resp = self.service.channels().list(part="id", mine=True).execute()
            items = (resp or {}).get("items") or []
            if items:
                return (items[0] or {}).get("id")
        except Exception:
            return None
        return None
    
    def _oauth_flow(self) -> bool:
        """Perform OAuth 2.0 flow to get credentials"""
        try:
            # Create OAuth flow with standard redirect URIs
            flow = InstalledAppFlow.from_client_config(
                {
                    "installed": {
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": [
                            "http://localhost:8080",
                            "http://localhost:8080/",
                            "http://localhost:8090",
                            "http://localhost:8090/",
                            "http://127.0.0.1:8090",
                            "http://127.0.0.1:8090/",
                            "urn:ietf:wg:oauth:2.0:oob"
                        ]
                    }
                },
                scopes=[
                    'https://www.googleapis.com/auth/youtube.upload',
                    'https://www.googleapis.com/auth/youtube.readonly'
                ]
            )
            
            # Prefer local server flow without auto-opening a browser
            print("🔄 Starting OAuth flow...")
            try:
                # Always use port 8080 to match OAuth config
                # Request offline access to get long-lived refresh token
                flow.redirect_uri = 'http://localhost:8080/'
                
                self.credentials = flow.run_local_server(
                    port=8080,
                    open_browser=False,
                    access_type='offline',
                    prompt='consent'
                )
            except Exception as _e:
                # Fallback: manual code entry with access_type=offline
                try:
                    auth_url, _ = flow.authorization_url(
                        prompt='consent',
                        access_type='offline'
                    )
                    print("\nOpen this URL in the desired browser/profile:")
                    print(auth_url)
                    code = input("\nPaste the authorization code here: ").strip()
                    flow.fetch_token(code=code)
                    self.credentials = flow.credentials
                except Exception as _e2:
                    print(f"X OAuth interactive flow failed: {_e2}")
                    return False
            
            # Verify we got a refresh token
            if not self.credentials.refresh_token:
                print("\n⚠️  WARNING: No refresh token received!")
                print("💡 This may happen if you've already authorized this app before.")
                print("💡 To fix: Revoke access at https://myaccount.google.com/permissions")
                print("💡 Then run authentication again to get a fresh refresh token.")
            
            # Save credentials with client info for token refresh
            creds_dict = json.loads(self.credentials.to_json())
            creds_dict['client_id'] = self.client_id
            creds_dict['client_secret'] = self.client_secret
            
            with open(self.credentials_file, 'w') as f:
                json.dump(creds_dict, f, indent=2)
            
            print(f"💾 Saved credentials with refresh token: {bool(self.credentials.refresh_token)}")
            
            # Build service
            self.service = build('youtube', 'v3', credentials=self.credentials)
            print("✅ OAuth flow completed successfully")
            return True
            
        except Exception as e:
            print(f"X OAuth flow failed: {e}")
            return False
    
    def _resumable_upload(self, insert_request, media, file_size: int, vod_id: Optional[str] = None):
        """Handle resumable upload with progress monitoring, adaptive chunking, and timeout"""
        import time
        import math
        import random
        import socket

        # Timeouts (tunable via env)
        max_hours_env = os.getenv('YT_MAX_UPLOAD_HOURS', '2')
        max_upload_time = int(max_hours_env) * 3600 if max_hours_env and max_hours_env.isdigit() else 24 * 3600
        progress_timeout_min_env = os.getenv('YT_PROGRESS_TIMEOUT_MIN', '20')
        progress_timeout = int(progress_timeout_min_env) * 60 if progress_timeout_min_env and progress_timeout_min_env.isdigit() else 15 * 60

        # Adaptive chunk sizing
        min_chunk_mb = max(1, int(os.getenv('YT_MIN_CHUNK_MB', '4')))  # conservative floor
        start_chunk_mb = max(min_chunk_mb, int(os.getenv('YT_UPLOAD_CHUNK_MB', '16')))
        max_chunk_mb = max(start_chunk_mb, int(os.getenv('YT_MAX_CHUNK_MB', '32')))
        # Align to 256KB boundaries as recommended
        def _mb_to_aligned_bytes(mb: int) -> int:
            size = mb * 1024 * 1024
            return (size // (256 * 1024)) * (256 * 1024)

        current_chunk_bytes = _mb_to_aligned_bytes(start_chunk_mb)
        media._chunksize = current_chunk_bytes  # type: ignore[attr-defined]

        # Session persistence (best effort)
        session_dir = None
        session_file = None
        if vod_id:
            session_dir = os.path.join('data', 'ai_data', vod_id)
            try:
                os.makedirs(session_dir, exist_ok=True)
                session_file = os.path.join(session_dir, '.yt_session.json')
            except Exception:
                session_file = None

        # Try to resume from previous session URI
        if session_file and os.path.exists(session_file):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    sess = json.load(f)
                uri = sess.get('resumable_uri')
                if uri:
                    insert_request.resumable_uri = uri  # type: ignore[attr-defined]
                    print('🔁 Resuming previous upload session')
            except Exception:
                pass

        response = None
        retry = 0
        max_retries = 8
        start_time = time.time()
        last_progress_time = start_time
        last_progress_bytes = 0
        last_log_time = start_time

        # Throughput tracking for adaptive logic
        stable_minutes_needed = int(os.getenv('YT_ADAPT_STABLE_MIN', '5'))
        last_retry_time = 0.0

        def _log_progress(now_ts: float, uploaded_bytes: int):
            nonlocal last_log_time
            if now_ts - last_log_time < 5:
                return
            mb_up = uploaded_bytes / (1024 * 1024)
            mb_total = file_size / (1024 * 1024)
            elapsed = now_ts - start_time
            speed = (uploaded_bytes / 1024 / 1024) / elapsed if elapsed > 0 else 0.0
            remaining_bytes = max(0, file_size - uploaded_bytes)
            eta_sec = remaining_bytes / (speed * 1024 * 1024) if speed > 0 else float('inf')
            eta_txt = f"{int(eta_sec // 3600)}h {int((eta_sec % 3600) // 60)}m" if math.isfinite(eta_sec) else '∞'
            print(f"📈 {mb_up:.1f}/{mb_total:.1f} MB ({(uploaded_bytes/file_size)*100:.1f}%) @ {speed:.2f} MB/s | ETA {eta_txt} | chunk {media._chunksize // (1024*1024)}MB")  # type: ignore[attr-defined]
            last_log_time = now_ts

        while response is None:
            try:
                # Timeouts
                now = time.time()
                if now - start_time > max_upload_time:
                    print(f"X Upload timeout after {max_upload_time/3600:.1f} hours")
                    return None
                if now - last_progress_time > progress_timeout:
                    print(f"X Upload stalled - no progress for {int(progress_timeout/60)} minutes")
                    # Try reducing chunk size once before giving up
                    if (media._chunksize // (1024*1024)) > min_chunk_mb:  # type: ignore[attr-defined]
                        new_mb = max(min_chunk_mb, (media._chunksize // (1024*1024)) // 2)  # type: ignore[attr-defined]
                        media._chunksize = _mb_to_aligned_bytes(new_mb)  # type: ignore[attr-defined]
                        print(f"↘️ Reduced chunk size to {new_mb}MB due to stall; continuing...")
                        last_progress_time = now
                    else:
                        return None

                status, response = insert_request.next_chunk()

                if status:
                    # status.progress() is fraction [0,1]
                    fraction = float(status.progress())
                    uploaded_bytes = int(fraction * file_size)
                    if uploaded_bytes > last_progress_bytes:
                        last_progress_time = now
                        # Persist session URI + progress
                        try:
                            if session_file:
                                with open(session_file, 'w', encoding='utf-8') as f:
                                    json.dump({
                                        'resumable_uri': getattr(insert_request, 'resumable_uri', None),
                                        'uploaded_bytes': uploaded_bytes,
                                        'chunk_bytes': media._chunksize  # type: ignore[attr-defined]
                                    }, f, indent=2)
                        except Exception:
                            pass
                        last_progress_bytes = uploaded_bytes
                        _log_progress(now, uploaded_bytes)

                        # Adaptive increase: if no retries in stable window, bump once up to max
                        minutes_elapsed = (now - start_time) / 60.0
                        no_recent_retry = (now - last_retry_time) > (stable_minutes_needed * 60)
                        if no_recent_retry and minutes_elapsed >= stable_minutes_needed:
                            cur_mb = media._chunksize // (1024*1024)  # type: ignore[attr-defined]
                            if cur_mb < max_chunk_mb:
                                new_mb = min(max_chunk_mb, cur_mb * 2)
                                if new_mb != cur_mb:
                                    media._chunksize = _mb_to_aligned_bytes(new_mb)  # type: ignore[attr-defined]
                                    print(f"↗️ Increased chunk size to {new_mb}MB (stable {stable_minutes_needed}m)")
                                    # reset stability window
                                    last_retry_time = now

                    retry = 0
                else:
                    # Waiting for first server response
                    pass

            except HttpError as e:
                if getattr(e, 'resp', None) and getattr(e.resp, 'status', None) in [500, 502, 503, 504]:
                    print(f"⚠️ Retriable HTTP {e.resp.status}: {e}")
                else:
                    print(f"X Non-retriable HTTP error: {e}")
                    # Re-raise non-retriable errors so caller can handle them
                    raise
                retry += 1
                last_retry_time = time.time()
                if retry > max_retries:
                    print(f"X Too many retries ({max_retries}), upload failed")
                    return None
                # Exponential backoff with jitter up to 300s
                wait_time = min(2 ** retry, 300) + random.uniform(0, 3)
                print(f"⏳ Waiting {int(wait_time)}s before retry...")
                time.sleep(wait_time)
            except (socket.timeout, ConnectionResetError, TimeoutError, OSError) as e:
                print(f"⚠️ Transient network error: {e}")
                retry += 1
                last_retry_time = time.time()
                if retry > max_retries:
                    print(f"X Too many network retries ({max_retries}), upload failed")
                    return None
                # Reduce chunk size to improve reliability under bad network
                cur_mb = media._chunksize // (1024*1024)  # type: ignore[attr-defined]
                if cur_mb > min_chunk_mb:
                    new_mb = max(min_chunk_mb, cur_mb // 2)
                    media._chunksize = _mb_to_aligned_bytes(new_mb)  # type: ignore[attr-defined]
                    print(f"↘️ Reduced chunk size to {new_mb}MB after network error")
                wait_time = min(2 ** retry, 180) + random.uniform(0, 2)
                time.sleep(wait_time)
            except Exception as e:
                print(f"X Unexpected error during upload: {e}")
                return None

        # Success: cleanup session file
        if session_file and os.path.exists(session_file):
            try:
                os.remove(session_file)
            except Exception:
                pass
        return response

    def set_thumbnail(self, video_id: str, thumbnail_path: str) -> bool:
        """Set a custom thumbnail for an uploaded YouTube video."""
        if not self.service:
            print("X YouTube service not initialized - authenticate first")
            return False
        try:
            if not os.path.exists(thumbnail_path):
                print(f"X Thumbnail not found: {thumbnail_path}")
                return False
            media = MediaFileUpload(thumbnail_path, mimetype='image/jpeg')
            request = self.service.thumbnails().set(videoId=video_id, media_body=media)
            request.execute()
            print(f"🖼️  Thumbnail set: {thumbnail_path}")
            return True
        except HttpError as e:
            print(f"X Failed to set thumbnail: {e}")
            return False
        except Exception as e:
            print(f"X Unexpected error setting thumbnail: {e}")
            return False
    
    def upload_video(self, video_path: str, metadata: Dict) -> Optional[str]:
        """Upload video to YouTube with given metadata"""
        if not self.service:
            print("X YouTube service not initialized - authenticate first")
            return None
        
        # Normalize to string to accept Path-like inputs
        try:
            video_path_str = video_path if isinstance(video_path, str) else str(video_path)
        except Exception:
            video_path_str = str(video_path)

        # Handle S3 URLs by downloading temporarily (only if not local)
        temp_file = None
        if video_path_str.startswith('s3://'):
            # Check if we can find the file locally first
            s3_path = video_path_str
            local_path = None
            
            # Try to extract local path from S3 URL
            if '/director_cut/' in s3_path:
                # Extract vod_id and construct local path
                parts = s3_path.split('/')
                vod_id = parts[-2] if 'director_cut' in s3_path else parts[-3]
                local_path = f"data/chunks/{vod_id}/director_cut/{vod_id}_directors_cut.mp4"
            elif '/videos/' in s3_path:
                # Extract vod_id and chapter from S3 path
                parts = s3_path.split('/')
                try:
                    vod_id = parts[-3]
                    chapter = parts[-2]
                    local_path = f"data/chunks/{vod_id}/{chapter}/{vod_id}_{chapter}.mp4"
                except IndexError:
                    local_path = None
            
            # Check if local file exists
            if local_path and os.path.exists(local_path):
                print(f"✅ Found local file: {local_path}")
                video_path_str = local_path
            else:
                print(f"📥 Local file not found, downloading from S3: {video_path_str}")
                try:
                    from storage import StorageManager
                    storage = StorageManager()
                    
                    # Create temporary file
                    import tempfile
                    temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
                    temp_file.close()
                    
                    print(f"📁 Temporary file: {temp_file.name}")
                    
                    # Download from S3 with timeout
                    print("⏳ Downloading from S3 (this may take a while for large videos)...")
                    storage.download_file(video_path_str, temp_file.name)
                    
                    if not os.path.exists(temp_file.name):
                        print("X Failed to download video from S3")
                        return None
                    
                    # Verify file size
                    downloaded_size = os.path.getsize(temp_file.name)
                    print(f"✅ Downloaded video to temporary file ({downloaded_size / (1024*1024):.1f} MB)")
                    video_path_str = temp_file.name
                    
                except Exception as e:
                    print(f"X Error downloading from S3: {e}")
                    if temp_file and os.path.exists(temp_file.name):
                        try:
                            os.unlink(temp_file.name)
                        except Exception:
                            pass
                    return None
        elif not os.path.exists(video_path_str):
            print(f"X Video file not found: {video_path_str}")
            return None
        
        try:
            print(f"📤 Uploading video: {os.path.basename(video_path_str)}")
            
            # Get file size for progress tracking
            file_size = os.path.getsize(video_path_str)
            print(f"📏 File size: {file_size / (1024*1024):.1f} MB")
            
            # Prepare upload
            body = {
                'snippet': metadata.get('snippet', {}),
                'status': metadata.get('status', {})
            }
            
            # Validate metadata
            if not body['snippet'].get('title'):
                print("⚠️ Warning: No title in metadata")
            if not body['snippet'].get('description'):
                print("⚠️ Warning: No description in metadata")
            
            # Create media upload with adaptive chunk size (default 8MB, aligned)
            def _aligned_bytes(mb: int) -> int:
                size = mb * 1024 * 1024
                return (size // (256 * 1024)) * (256 * 1024)
            default_chunk_mb = max(1, int(os.getenv('YT_UPLOAD_CHUNK_MB', '16')))
            media = MediaFileUpload(
                video_path_str,
                chunksize=_aligned_bytes(default_chunk_mb),
                resumable=True,
                mimetype='video/mp4'
            )
            
            # Execute upload
            request = self.service.videos().insert(
                part=','.join(body.keys()),
                body=body,
                media_body=media
            )
            
            # Extract vod_id from path for session files
            def _extract_vod_id_from_path(p: str) -> Optional[str]:
                try:
                    parts = os.path.normpath(p).split(os.sep)
                    if 'data' in parts and 'chunks' in parts:
                        i = parts.index('chunks')
                        if i + 1 < len(parts):
                            return parts[i+1]
                except Exception:
                    pass
                # Try filename pattern fallback
                try:
                    import re
                    m = re.search(r"(\d{7,})", os.path.basename(p))
                    return m.group(1) if m else None
                except Exception:
                    return None

            vod_id_for_session = _extract_vod_id_from_path(video_path_str)

            # Execute the upload with resumable upload handling
            print("📤 Starting YouTube upload (optimized for large files)...")
            response = self._resumable_upload(request, media, file_size, vod_id_for_session)
            
            if not response:
                print("X Upload failed")
                return None
                
            video_id = response.get('id')
            
            if video_id:
                print("✅ Video uploaded successfully!")
                print(f"🎥 Video ID: {video_id}")
                print(f"🔗 URL: https://www.youtube.com/watch?v={video_id}")
                return video_id
            else:
                print("X Upload failed - no video ID returned")
                return None
                
        except HttpError as e:
            print(f"X YouTube API error: {e}")
            # Re-raise so caller can handle quota and other errors
            raise
        except Exception as e:
            print(f"X Upload failed: {e}")
            # Re-raise so caller can handle errors
            raise
        finally:
            # Clean up temporary file if it was downloaded from S3
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                    print("🗑️ Cleaned up temporary file")
                except Exception:
                    pass


def main():
    """Test the YouTube uploader"""
    if len(sys.argv) < 2:
        print("Usage: python youtube_uploader.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Initialize uploader
    uploader = YouTubeUploader()
    
    # Authenticate
    if not uploader.authenticate():
        print("X Authentication failed")
        sys.exit(1)
    
    # Test metadata
    test_metadata = {
        'snippet': {
            'title': 'Test Upload',
            'description': 'Test video upload',
            'tags': ['test'],
            'categoryId': '20'
        },
        'status': {
            'privacyStatus': 'private'
        }
    }
    
    # Upload
    video_id = uploader.upload_video(video_path, test_metadata)
    if video_id:
        print(f"✅ Test upload successful: {video_id}")
    else:
        print("X Test upload failed")


if __name__ == "__main__":
    main()
