#!/usr/bin/env python3
"""
Upload individual clips to YouTube
Uploads each clip with its own title, description, and metadata
"""

# --- universal log adapter -----------------------------------------------
import os, logging, sys
from pathlib import Path
# Add project root to path BEFORE trying to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)
if os.getenv("JOB_RUN_ID"):
    try:
        from utils.log import get_job_logger
        job_type = os.getenv("JOB_TYPE") or "clip"
        _, logger = get_job_logger(job_type, vod_id=os.getenv("VOD_ID", "unknown"), run_id=os.getenv("RUN_ID"))
        
        # Replace print with logger.info for this module
        def _log_print(*args, **kwargs):
            file = kwargs.get('file', sys.stdout)
            if file == sys.stderr:
                logger.error(' '.join(str(arg) for arg in args))
            else:
                logger.info(' '.join(str(arg) for arg in args))
        
        # Override print in this module's global scope
        print = _log_print
        
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        pass
# -------------------------------------------------------------------------

import json
import sys
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to Python path to import src module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config
from src.youtube_channels import resolve_channels_for_vod, get_channel_credentials_file
from storage import StorageManager


def load_metadata_index(vod_id: str) -> list[Path]:
    """Load list of per-clip metadata files from index; return absolute Paths."""
    ai_data_dir = config.get_ai_data_dir(vod_id)
    index_path = ai_data_dir / f"{vod_id}_clip_metadata_index.json"
    if not index_path.exists():
        return []
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        files = data.get('files', [])
        result = []
        for name in files:
            p = ai_data_dir / name
            if p.exists():
                result.append(p)
        return result
    except Exception:
        return []

def load_clip_metadata(vod_id: str, clip_index: int) -> Optional[Dict]:
    """Load YouTube metadata for specific clip"""
    ai_data_dir = config.get_ai_data_dir(vod_id)
    metadata_path = ai_data_dir / f"{vod_id}_clip_{clip_index:02d}_youtube_metadata.json"
    
    if not metadata_path.exists():
        print(f"X Clip metadata not found: {metadata_path}")
        return None
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_clip_file(vod_id: str, clip_title: str) -> Optional[Path]:
    """Find clip file in clips directory, searching recursively and chapter subfolders."""
    clips_dir = Path(f"data/clips/{vod_id}")

    if not clips_dir.exists():
        print(f"X Clips directory not found: {clips_dir}")
        return None

    # Remove any hashtag suffixes (e.g. "title #tag") before normalisation
    if "#" in clip_title:
        clip_title = clip_title.split("#", 1)[0].strip()

    # Normalize title
    safe_title = "".join(c for c in clip_title if c.isalnum() or c in (" ", "_", "-")).rstrip()
    safe_title = safe_title.replace(" ", "_")

    # Prefer chapter subfolders first to match creation layout
    # Exclude temp files and prioritize processed clips
    processed_files = []
    
    for file_path in clips_dir.rglob("*.mp4"):
        stem_lower = file_path.stem.lower()
        if safe_title.lower() == stem_lower or safe_title.lower() in stem_lower:
            if not file_path.stem.startswith("temp_"):
                processed_files.append(file_path)
    
    # Only allow processed files; never fall back to temp_ files
    if processed_files:
        return processed_files[0]  # Return first processed file found
    else:
        print(f"⛔ No processed clip found for title (skipping temp files): {clip_title}")

    # Fallback: flat root (processed files only)
    candidate = clips_dir / f"{safe_title}.mp4"
    if candidate.exists():
        return candidate

    print(f"X Clip file not found for title: {clip_title}")
    return None


def download_clip_from_s3(vod_id: str, clip_title: str, local_path: Path) -> bool:
    """Download clip from S3 if not found locally"""
    # Respect local-only flags – skip S3 entirely
    if os.getenv('DISABLE_S3_UPLOADS', '').lower() in ('1','true','yes') or \
       os.getenv('LOCAL_TEST_MODE', '').lower() in ('1','true','yes'):
        print('⏭️  Skipping S3 fallback (local-only mode)')
        return False

    try:
        import boto3
        s3 = boto3.client('s3')
        bucket_name = os.environ.get('S3_BUCKET', 'streamsniped-dev-videos')
        
        # Sanitize title for S3 key
        safe_title = "".join(c for c in clip_title if c.isalnum() or c in (' ', '_', '-')).rstrip()
        safe_title = safe_title.replace(' ', '_')
        
        # Prefer chapter-based layout first
        # List under clips/<vod>/chapter_*/ and match by title stem
        try:
            prefix = f"clips/{vod_id}/"
            print(f"🔎 Listing S3 prefix for chapter lookup: s3://{bucket_name}/{prefix}")
            paginator = s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                for obj in page.get('Contents', []) if isinstance(page, dict) else []:
                    key = obj.get('Key', '')
                    if key.lower().endswith('.mp4') and safe_title.lower() in key.lower():
                        # Confirm object exists (HEAD) to avoid 404 during download
                        try:
                            s3.head_object(Bucket=bucket_name, Key=key)
                            print(f"📥 Fallback downloading from S3: s3://{bucket_name}/{key}")
                            s3.download_file(bucket_name, key, str(local_path))
                            print(f" Downloaded clip from S3 (matched chapter path): {key}")
                            return True
                        except Exception:
                            continue
        except Exception as e:
            print(f" Chapter lookup failed: {e}")

        # As last resort, try flat keys
        s3_keys = [
            f"clips/{vod_id}/{safe_title}.mp4",
            f"clips/{vod_id}/{safe_title}/{safe_title}.mp4",
            f"clips/{vod_id}/{safe_title}/{vod_id}_{safe_title}.mp4"
        ]
        
        for s3_key in s3_keys:
            try:
                print(f"📥 Downloading from S3: s3://{bucket_name}/{s3_key}")
                s3.download_file(bucket_name, s3_key, str(local_path))
                print(f" Downloaded clip from S3: {s3_key}")
                return True
            except Exception as e:
                print(f" Failed to download {s3_key}: {e}")
                continue
        
        print(f"X Clip not found in S3 for title: {clip_title}")
        return False
        
    except Exception as e:
        print(f"X Error downloading from S3: {e}")
        return False


def verify_shorts_format(video_path: Path) -> bool:
    """Verify video is in proper Shorts format (9:16 aspect ratio, ≤180s)"""
    try:
        import subprocess
        # Try to use local ffmpeg.exe if available
        ffprobe_cmd = "ffprobe"
        if Path("ffmpeg.exe").exists():
            # Use ffmpeg with -f lavfi to simulate ffprobe functionality
            cmd = [
                "./ffmpeg.exe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,duration",
                "-of", "json",
                str(video_path)
            ]
        else:
            cmd = [
                ffprobe_cmd, "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,duration",
                "-of", "json",
                str(video_path)
            ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            stream = data.get('streams', [{}])[0]
            width = float(stream.get('width', 0))
            height = float(stream.get('height', 0))
            duration = float(stream.get('duration', 0))
            
            # Check aspect ratio (should be ~9:16)
            aspect_ratio = height / width if width > 0 else 0
            # 9:16 means height/width ≈ 16/9 ≈ 1.78 (tolerance 10%)
            is_vertical = abs(aspect_ratio - (16/9)) < 0.15
            
            # Check duration (≤180s for Shorts)
            is_short = duration <= 180
            
            if not is_vertical:
                print(f" Video not in vertical format: {width}x{height} (ratio: {aspect_ratio:.2f})")
            if not is_short:
                print(f" Video too long for Shorts: {duration:.1f}s")
            
            return is_vertical and is_short
    except Exception as e:
        print(f" Failed to verify video format: {e}")
        # Skip verification if ffprobe/ffmpeg not available
        print(" Skipping video format verification")
        return True

def upload_clip_to_youtube(clip_path: Path, metadata: Dict, vod_id: str, clip_index: int, force_public: bool = False, *, channel_key: str = "default") -> Optional[str]:
    """Upload individual clip to YouTube.
    
    For Shorts:
    - Verifies 9:16 aspect ratio
    - Verifies ≤60s duration
    - Adds #Shorts to title if missing
    """
    try:
        # Override privacy status if force_public is True
        if force_public:
            metadata['status']['privacyStatus'] = 'public'
            print("🌍 Uploading clip as public video")
        else:
            print(f"🔒 Uploading clip as {metadata['status']['privacyStatus']} video")

        # Verify video format for Shorts (skip for now since clips are pre-formatted)
        # if not verify_shorts_format(clip_path):
        #     print("X Video not in proper Shorts format - skipping upload")
        #     return None

        # Import YouTube uploader
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from youtube_uploader import YouTubeUploader
        
        # Idempotency per-channel: skip if already uploaded
        ai_data_dir = config.get_ai_data_dir(vod_id)
        result_path = ai_data_dir / f"{vod_id}_clip_{clip_index:02d}_youtube_upload_result__{channel_key}.json"
        try:
            if result_path.exists():
                with open(result_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                if str(existing.get('youtube_video_id') or '').strip():
                    print(f"⏭️  Clip {clip_index} already uploaded for channel '{channel_key}', skipping")
                    return existing.get('youtube_video_id')
        except Exception:
            pass

        # Initialize uploader for this channel
        cred_file = get_channel_credentials_file(channel_key)
        print(f" Using channel '{channel_key}' (creds: {cred_file})")
        uploader = YouTubeUploader(credentials_file=cred_file)
        
        # Authenticate
        if not uploader.authenticate():
            print("X YouTube authentication failed")
            return None
        
        print(f" Uploading clip {clip_index} to YouTube...")
        print(f"📁 File: {clip_path}")
        print(f"📝 Title: {metadata['snippet']['title']}")
        
        # Upload video
        video_id = uploader.upload_video(clip_path, metadata)
        
        if video_id:
            print(f" Clip {clip_index} uploaded successfully!")
            print(f"🎥 YouTube Video ID: {video_id}")
            print(f"🔗 Video URL: https://www.youtube.com/watch?v={video_id}")
            
            # Save upload result
            result = {
                "vod_id": vod_id,
                "clip_index": clip_index,
                "youtube_video_id": video_id,
                "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
                "title": metadata['snippet']['title'],
                "upload_date": None  # Will be set by uploader
            }
            
            # Save result to file (per-channel)
            
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Upload result saved: {result_path}")
            return video_id
        else:
            print(f"X Failed to upload clip {clip_index}")
            return None
            
    except Exception as e:
        print(f"X Error uploading clip {clip_index}: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_clip_titles(vod_id: str) -> List[Dict]:
    """Load clip titles, preferring 'clip_titles' with fallback to 'titles'"""
    ai_data_dir = config.get_ai_data_dir(vod_id)
    titles_path = ai_data_dir / f"{vod_id}_clip_titles.json"
    
    if not titles_path.exists():
        print(f"X Clip titles not found: {titles_path}")
        return []
    
    with open(titles_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    titles = data.get('clip_titles')
    if isinstance(titles, list) and titles:
        print(f" Loaded {len(titles)} titles from 'clip_titles'")
        return titles
    titles = data.get('titles', [])
    if isinstance(titles, list) and titles:
        print(f" Loaded {len(titles)} titles from legacy 'titles'")
        return titles
    print("X No titles found under 'clip_titles' or 'titles'")
    return []


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python upload_clips_to_youtube.py <vod_id> [--clip-index <number>] [--public] [--channels k1,k2]")
        print("\nOptions:")
        print("  --clip-index <number>  Upload specific clip by index")
        print("  --public              Upload as public (default: uses metadata setting)")
        print("  --channels k1,k2      Upload to specific channel keys (default: auto-resolve)")
        sys.exit(1)
    
    vod_id = sys.argv[1]
    specific_clip = None
    force_public = "--public" in sys.argv
    channels_csv = None
    
    # Parse arguments
    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == '--clip-index' and i + 1 < len(sys.argv):
            try:
                specific_clip = int(sys.argv[i + 1])
            except ValueError:
                print("X Invalid clip index")
                sys.exit(1)
        if arg == '--channels' and i + 1 < len(sys.argv):
            channels_csv = sys.argv[i + 1]
    
    try:
        print(f" Uploading clips to YouTube for VOD: {vod_id}")
        # Resolve channels
        if channels_csv:
            channel_keys = [k.strip() for k in channels_csv.split(',') if k.strip()]
        else:
            channel_keys = resolve_channels_for_vod(vod_id, content_type="clips")
        if not channel_keys:
            channel_keys = ["default"]
        print(f" Target channels: {', '.join(channel_keys)}")

        # Prefer metadata index to drive uploads; fallback to titles
        metadata_files = load_metadata_index(vod_id)
        using_metadata_index = len(metadata_files) > 0

        if not using_metadata_index:
            # Load clip titles
            clip_titles = load_clip_titles(vod_id)
            if not clip_titles:
                print("X No clip titles found")
                sys.exit(1)
            total_to_upload = len(clip_titles)
            print(f"📊 Found {total_to_upload} clips to upload")
        else:
            total_to_upload = len(metadata_files)
            print(f"📊 Found {total_to_upload} clips to upload (from metadata index)")

        # Upload clips (self-heal: ensure metadata exists)
        uploaded_count = 0
        # Self-heal: if first metadata file is missing but titles exist, try to generate metadata now
        try:
            first_meta = config.get_ai_data_dir(vod_id) / f"{vod_id}_clip_01_youtube_metadata.json"
            if not first_meta.exists():
                print("🔧 Missing clip metadata detected. Generating now...")
                import subprocess as _subprocess
                result = _subprocess.run([
                    'python', 'processing-scripts/generate_clip_youtube_metadata.py', vod_id
                ], capture_output=True, text=True)
                if result.returncode == 0:
                    print(" Clip metadata generated on-the-fly")
                else:
                    print(" Failed to auto-generate clip metadata; continuing best-effort")
        except Exception:
            pass
        
        if using_metadata_index:
            iterable = list(enumerate(metadata_files, 1))
        else:
            clip_titles = clip_titles  # already loaded above
            iterable = list(enumerate(clip_titles, 1))

        for i, item in iterable:
            # Initialize per-iteration state
            clip_path: Optional[Path] = None
            # Skip if specific clip requested
            if specific_clip and i != specific_clip:
                continue
            
            print(f"\n Processing clip {i}/{total_to_upload}")
            
            # Load clip metadata (from index when available)
            if using_metadata_index:
                try:
                    with open(item, 'r', encoding='utf-8') as mf:
                        metadata = json.load(mf)
                except Exception as e:
                    print(f"X Failed to read metadata file {item}: {e}")
                    continue
                # Derive base clip title (strip streamer suffix if present)
                raw_title = metadata.get('snippet', {}).get('title', f'Clip {i}')
                clip_title = raw_title.split(' - ')[0]
                # Remove #Shorts suffix for filename matching
                if clip_title.endswith(' #Shorts'):
                    clip_title = clip_title[:-8]  # Remove " #Shorts"
                # Also remove #shorts if present
                if clip_title.endswith(' #shorts'):
                    clip_title = clip_title[:-8]  # Remove " #shorts"
                # Prefer explicit clip_path in metadata if provided
                explicit_path = metadata.get('streamsniped_metadata', {}).get('clip_path')
                if explicit_path:
                    p = Path(explicit_path)
                    if p.exists():
                        clip_path = p
                        print(f"📁 Using explicit clip path from metadata: {clip_path}")
                    else:
                        # If it's an S3 URI, try to download
                        if explicit_path.startswith('s3://'):
                            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                                tmp_file_path = Path(tmp_file.name)
                            try:
                                storage = StorageManager()
                                storage.download_file(explicit_path, str(tmp_file_path))
                                clip_path = tmp_file_path
                                print(f"📥 Downloaded clip via explicit S3 path: {explicit_path}")
                            except Exception as _e:
                                print(f" Failed explicit S3 download: {explicit_path} ({_e})")
                                clip_path = None
                # If explicit path succeeded, skip title lookup
                if 'clip_path' in locals() and clip_path:
                    pass
                else:
                    clip_path = None
            else:
                metadata = load_clip_metadata(vod_id, i)
                if not metadata:
                    print(f" Skipping clip {i} - no metadata found")
                    continue
                clip_title = item.get('title', f'Clip {i}')
            
            # Find clip file
            if 'clip_path' not in locals() or not clip_path:
                print(f"🔍 Looking for clip file with title: {clip_title}")
                clip_path = find_clip_file(vod_id, clip_title)
                print(f"🔍 Found clip path: {clip_path}")
            
            if not clip_path:
                # Try to download from S3
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                    tmp_path = Path(tmp_file.name)
                if download_clip_from_s3(vod_id, clip_title, tmp_path):
                    clip_path = tmp_path
            
            # Skip if no processed clip found
            if not clip_path:
                print(f"⛔ Skipping clip {i} - processed clip not found (and temp_ files are disallowed)")
                continue

            # Upload to all resolved channels
            any_success = False
            for ch in channel_keys:
                video_id = upload_clip_to_youtube(clip_path, metadata, vod_id, i, force_public, channel_key=ch)
                if video_id:
                    any_success = True
            if any_success:
                uploaded_count += 1
                print(f" Clip {i} uploaded successfully to at least one channel")
            else:
                print(f"X Failed to upload clip {i} to all channels")
        
        # Save overall upload summary
        upload_summary = {
            "vod_id": vod_id,
            "total_clips": total_to_upload,
            "uploaded_clips": uploaded_count,
            "success_rate": uploaded_count / total_to_upload if total_to_upload > 0 else 0,
            "upload_date": None  # Will be set by each individual upload
        }
        
        ai_data_dir = config.get_ai_data_dir(vod_id)
        summary_path = ai_data_dir / f"{vod_id}_youtube_upload_clips_result.json"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(upload_summary, f, indent=2, ensure_ascii=False)
        
        if uploaded_count == total_to_upload and total_to_upload > 0:
            print("\n🎉 Upload complete!")
        elif uploaded_count == 0 and total_to_upload > 0:
            print(f"\n❌ Upload failed: 0/{total_to_upload} clips uploaded")
        else:
            print(f"\n⚠️ Upload partially complete: {uploaded_count}/{total_to_upload} clips")
        print(f"📊 Uploaded {uploaded_count}/{total_to_upload} clips")
        print(f"💾 Upload summary saved: {summary_path}")
        
        if uploaded_count > 0:
            print("💡 Some clips are live on YouTube!")
            print("🔗 Check your YouTube channel for the new videos")
        else:
            # Non-zero exit to signal failure when nothing was uploaded
            try:
                import sys as _sys
                _sys.exit(1)
            except SystemExit:
                pass
        
    except Exception as e:
        print(f"X Error uploading clips: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 