#!/usr/bin/env python3
"""
Upload Arc videos to YouTube using generated arc metadata.

Usage:
  python processing-scripts/auto_youtube_upload_arch.py <vod_id> [--arc 1] [--public]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _find_arc_video(vod_id: str, arc_index: int) -> Optional[Path]:
    base = Path(f"data/chunks/{vod_id}/arcs")
    if not base.exists():
        return None
    p = base / f"{vod_id}_arc_{arc_index:03d}.mp4"
    return p if p.exists() else None


def _load_arc_indices(vod_id: str, only: Optional[int]) -> List[int]:
    arcs_dir = Path(f"data/vector_stores/{vod_id}/arcs")
    if only is not None:
        mp = arcs_dir / f"arc_{only:03d}_manifest.json"
        return [only] if mp.exists() else []
    # infer from manifests present
    out: List[int] = []
    for p in sorted(arcs_dir.glob("arc_*_manifest.json")):
        try:
            num = int(p.stem.split('_')[1])
            out.append(num)
        except Exception:
            continue
    return out


def upload_arc_with_error_tracking(
    vod_id: str, 
    arc_idx: int, 
    video_path: Path, 
    metadata: dict, 
    channel_keys: List[str],
    get_channel_credentials_file,
    YouTubeUploader
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Upload a single arc to YouTube with error tracking.
    Returns (success, error_type, error_message)
    error_type can be: 'quota_exceeded', 'retriable', 'permanent', None
    """
    from utils.upload_scheduler import is_quota_exceeded_error, is_retriable_error
    
    any_success = False
    last_error_type = None
    last_error_message = None
    
    for ch in channel_keys:
        cred_file = get_channel_credentials_file(ch)
        print(f" Using channel '{ch}' (creds: {cred_file})")
        
        try:
            up = YouTubeUploader(credentials_file=cred_file)
            if not up.authenticate():
                print(f"X Auth failed for channel '{ch}'")
                continue
            
            vid = up.upload_video(str(video_path), metadata)
            
            if vid:
                any_success = True
                # Try to attach thumbnail
                try:
                    thumb_dir = Path(f"data/thumbnails/{vod_id}/arch")
                    pref = thumb_dir / f"arc_{arc_idx:03d}_v1.jpg"
                    thumb_path = pref if pref.exists() else (next(iter(sorted(thumb_dir.glob(f"arc_{arc_idx:03d}_v*.jpg"))), None))
                    if thumb_path and isinstance(thumb_path, Path) and thumb_path.exists():
                        _ = up.set_thumbnail(vid, str(thumb_path))
                except Exception:
                    pass
                print(f"✅ Uploaded to {ch}: https://www.youtube.com/watch?v={vid}")
                
                # Mark as completed in scheduler
                try:
                    from utils.upload_scheduler import UploadScheduler
                    scheduler = UploadScheduler()
                    scheduler.mark_upload_completed(vod_id, arc_idx, ch)
                except Exception:
                    pass
            else:
                print(f"❌ Upload failed for arc {arc_idx:03d} on {ch}")
                # Upload returned None, but we don't have error details
                # Assume retriable for now
                last_error_type = "retriable"
                last_error_message = "Upload returned None"
                
        except Exception as e:
            error_str = str(e)
            print(f"❌ Upload failed for arc {arc_idx:03d} on {ch}: {error_str}")
            
            # Classify error
            if is_quota_exceeded_error(error_str):
                last_error_type = "quota_exceeded"
                last_error_message = error_str
                print("🚫 YouTube quota exceeded - will retry in 24 hours")
            elif is_retriable_error(error_str):
                last_error_type = "retriable"
                last_error_message = error_str
                print("⚠️  Retriable error - will retry later")
            else:
                last_error_type = "permanent"
                last_error_message = error_str
                print("❌ Permanent error - will not retry")
            
            # Add to scheduler if retriable
            if last_error_type in ("quota_exceeded", "retriable"):
                try:
                    from utils.upload_scheduler import UploadScheduler
                    scheduler = UploadScheduler()
                    retry_hours = 24 if last_error_type == "quota_exceeded" else 6
                    scheduler.add_failed_upload(
                        vod_id=vod_id,
                        arc_index=arc_idx,
                        reason=last_error_message,
                        channel=ch,
                        retry_after_hours=retry_hours
                    )
                except Exception as sched_err:
                    print(f"⚠️  Failed to schedule retry: {sched_err}")
    
    return any_success, last_error_type, last_error_message


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python processing-scripts/auto_youtube_upload_arch.py <vod_id> [--arc <index>] [--public]")
        sys.exit(1)

    vod_id = sys.argv[1]
    arc_only: Optional[int] = None
    if "--arc" in sys.argv:
        try:
            i = sys.argv.index("--arc")
            if i + 1 < len(sys.argv):
                arc_only = int(sys.argv[i + 1])
        except Exception:
            print("X Invalid --arc flag usage")
            sys.exit(1)
    make_public = "--public" in sys.argv

    arcs = _load_arc_indices(vod_id, arc_only)
    if not arcs:
        print("X No arc manifests found")
        sys.exit(1)

    from src.config import config
    meta_dir = config.get_ai_data_dir(vod_id)
    from youtube_uploader import YouTubeUploader  # type: ignore
    from src.youtube_channels import resolve_channels_for_vod, get_channel_credentials_file

    channels_arg = None
    if "--channels" in sys.argv:
        try:
            j = sys.argv.index("--channels")
            if j + 1 < len(sys.argv):
                channels_arg = sys.argv[j + 1]
        except Exception:
            channels_arg = None

    if channels_arg:
        channel_keys = [k.strip() for k in channels_arg.split(',') if k.strip()]
    else:
        channel_keys = resolve_channels_for_vod(vod_id, content_type="arcs")
    if not channel_keys:
        channel_keys = ["default"]
    print(f" Target channels: {', '.join(channel_keys)}")

    # Validate per-channel credentials; allow partial success
    missing: List[str] = []
    for ch in list(channel_keys):
        cred = get_channel_credentials_file(ch)
        if not Path(cred).exists() and not (os.getenv('YOUTUBE_CLIENT_ID') and os.getenv('YOUTUBE_CLIENT_SECRET')):
            missing.append(f"{ch} ({cred})")
    if missing:
        if len(missing) == len(channel_keys):
            print(f"X No valid YouTube credentials found for channels: {', '.join(missing)}")
            sys.exit(1)
        else:
            print(f"⚠️  Skipping channels without creds: {', '.join(missing)}")
            channel_keys = [k for k in channel_keys if k not in {m.split(' ')[0] for m in missing}]

    # Process each arc and upload to all channels
    any_success_overall = False
    quota_exceeded = False
    
    for arc_idx in arcs:
        video_path = _find_arc_video(vod_id, arc_idx)
        if not video_path:
            print(f"⏭️  Skipping arc {arc_idx:03d}: video not found")
            continue
        meta_path = meta_dir / f"{vod_id}_arc_{arc_idx:03d}_youtube_metadata.json"
        if not meta_path.exists():
            print(f"⏭️  Skipping arc {arc_idx:03d}: metadata not found ({meta_path})")
            continue
        try:
            metadata = json.loads(meta_path.read_text(encoding='utf-8'))
        except Exception as e:
            print(f"⏭️  Skipping arc {arc_idx:03d}: invalid metadata ({e})")
            continue
        if make_public:
            metadata['status']['privacyStatus'] = 'public'

        print(f"\n🚀 Uploading arc {arc_idx:03d}")
        print(f"📁 {video_path}")
        print(f"📝 {metadata['snippet']['title']}")

        # Upload with error tracking
        success, error_type, error_msg = upload_arc_with_error_tracking(
            vod_id, arc_idx, video_path, metadata, 
            channel_keys, get_channel_credentials_file, YouTubeUploader
        )
        
        if success:
            any_success_overall = True
        else:
            print("❌ Upload failed for this arc on all channels")
            if error_type == "quota_exceeded":
                quota_exceeded = True
                print("🚫 YouTube quota exceeded - remaining uploads will be scheduled for retry")
                # Schedule remaining arcs for later
                for remaining_idx in arcs:
                    if remaining_idx > arc_idx:
                        for ch in channel_keys:
                            try:
                                from utils.upload_scheduler import UploadScheduler
                                scheduler = UploadScheduler()
                                scheduler.add_failed_upload(
                                    vod_id=vod_id,
                                    arc_index=remaining_idx,
                                    reason="Quota exceeded on previous arc",
                                    channel=ch,
                                    retry_after_hours=24
                                )
                            except Exception:
                                pass
                break  # Stop trying more arcs
    
    if not any_success_overall:
        if quota_exceeded:
            print("🚫 Uploads failed due to YouTube quota - will retry in 24 hours")
            sys.exit(2)  # Exit code 2 = quota exceeded
        else:
            print("❌ Uploads failed for all arcs and channels")
            sys.exit(1)  # Exit code 1 = general failure


if __name__ == "__main__":
    main()


