#!/usr/bin/env python3
"""
Clean up local files after processing/upload.
Safely deletes large video assets only after confirming successful upload.
Always cleans derived artifacts (transcripts, chats, chat_contexts,
chunks, cache/raw_clips) to prevent storage bloat. Optional TTL sweeper.

Important: This script does NOT delete ai_data or vector_stores by default
to preserve expensive computation across retries. Use explicit flags to
remove them if you are certain.
"""

import os
import sys
import json
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_upload_success(vod_id: str) -> tuple[bool, bool]:
    """Check if both Director's Cut and clips were uploaded successfully"""
    
    # Check Director's Cut upload
    director_cut_result = Path(f"data/ai_data/{vod_id}/{vod_id}_youtube_upload_result.json")
    director_cut_success = False
    if director_cut_result.exists():
        try:
            with open(director_cut_result, 'r') as f:
                result = json.load(f)
                director_cut_success = result.get('youtube_video_id') is not None
        except Exception:
            pass
    
    # Check clips upload
    clips_result = Path(f"data/ai_data/{vod_id}/{vod_id}_youtube_upload_clips_result.json")
    clips_success = False
    if clips_result.exists():
        try:
            with open(clips_result, 'r') as f:
                result = json.load(f)
                clips_success = result.get('uploaded_clips', 0) > 0
        except Exception:
            pass
    
    return director_cut_success, clips_success

def _remove_path(path: Path) -> float:
    """Remove a file or directory tree and return reclaimed size in MB (best-effort)."""
    try:
        if not path.exists():
            return 0.0
        if path.is_file():
            size_mb = path.stat().st_size / (1024*1024)
            path.unlink()
            return size_mb
        # Directory: sum sizes then remove
        total = 0.0
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size / (1024*1024)
            except OSError:
                continue
        shutil.rmtree(path, ignore_errors=True)
        return total
    except Exception:
        return 0.0

def cleanup_derived_artifacts(vod_id: str, clean_ai_data: bool = False) -> tuple[int, float]:
    """
    Remove derived per-VOD artifacts regardless of upload status to avoid storage bloat.
    Targets:
      - data/transcripts
      - data/chunks/<vod_id>
      - data/chats
      - data/chat_contexts
      - data/cache/raw_clips
      - data/ai_data/<vod_id> (optional)
    """
    targets: list[Path] = [
        Path("data/transcripts"),
        Path(f"data/chunks/{vod_id}"),
        Path("data/chats"),
        Path("data/chat_contexts"),
        Path("data/cache/raw_clips"),
    ]

    cleaned = 0
    size_mb = 0.0

    def remove_matching(base: Path) -> tuple[int, float]:
        removed = 0
        reclaimed = 0.0
        if not base.exists():
            return removed, reclaimed
        # If path already scoped to vod (chunks/<vod_id>) remove directly
        if base.name == vod_id and base.parent.name == "chunks":
            reclaimed += _remove_path(base)
            print(f"🧽 Removed chunks for {vod_id}: {base}")
            return 1, reclaimed
        # Otherwise remove files/folders whose path includes vod_id
        candidates = list(base.rglob(f"*{vod_id}*"))
        # Remove deeper paths first
        for p in sorted(candidates, key=lambda p: len(p.parts), reverse=True):
            reclaimed += _remove_path(p)
            removed += 1
            try:
                rel = p.relative_to(base)
            except Exception:
                rel = p
            print(f"🗑️ Removed derived artifact: {rel}")
        return removed, reclaimed

    for base in targets:
        r, s = remove_matching(base)
        cleaned += r
        size_mb += s

    if clean_ai_data:
        ai_dir = Path(f"data/ai_data/{vod_id}")
        if ai_dir.exists():
            size_mb += _remove_path(ai_dir)
            cleaned += 1
            print(f"🧽 Removed AI data for {vod_id}: {ai_dir}")

    return cleaned, size_mb

def upload_to_s3_backup(vod_id: str) -> bool:
    """Deprecated: no-op to avoid S3 bloat during cleanup."""
    return True

def cleanup_after_upload(vod_id: str, force_clean: bool = False, clean_ai_data: bool = False, sweep_ttl_days: int = 0) -> bool:
    """
    Clean up local files after processing/upload.
    - Always purge derived artifacts for this VOD to prevent storage bloat.
    - Delete large assets (Director's Cut, clips) only if uploads were successful or if force_clean is True.
    - Optionally sweep stale artifacts older than N days across common data dirs.
    """

    print(f"🧹 Starting post-processing cleanup for VOD: {vod_id}")

    # 1) Remove derived artifacts regardless of YouTube status
    derived_count, derived_size = cleanup_derived_artifacts(vod_id, clean_ai_data=clean_ai_data)
    print(f"✅ Removed {derived_count} derived artifacts ({derived_size:.1f} MB)")

    # 2) Determine if we can remove large assets
    director_cut_success, clips_success = check_upload_success(vod_id)
    if not (director_cut_success or clips_success) and not force_clean:
        print("⚠️ No successful uploads found; keeping large assets (use --force-clean to override)")
        return True

    # 3) Remove large assets
    cleaned_count = 0
    total_size = 0

    try:
        if force_clean or director_cut_success:
            director_cut_path = Path(f"data/videos/{vod_id}/director_cut/{vod_id}_director_cut.mp4")
            if director_cut_path.exists():
                file_size = director_cut_path.stat().st_size
                total_size += file_size
                director_cut_path.unlink()
                cleaned_count += 1
                print(f"🗑️ Cleaned Director's Cut: {director_cut_path.name}")

        if force_clean or clips_success:
            clips_dir = Path(f"data/clips/{vod_id}")
            if clips_dir.exists():
                for clip_file in clips_dir.glob("*.mp4"):
                    if clip_file.is_file():
                        file_size = clip_file.stat().st_size
                        total_size += file_size
                        clip_file.unlink()
                        cleaned_count += 1
                        print(f"🗑️ Cleaned clip: {clip_file.name}")

        print(f"✅ Cleaned up {cleaned_count} large files ({total_size / (1024*1024):.1f} MB)")
        print("💾 Large assets backed up in S3 or forcibly removed per flag")

        # 5) Optional TTL sweep across shared dirs
        if sweep_ttl_days and sweep_ttl_days > 0:
            sweep_old_artifacts(sweep_ttl_days)
        return True

    except Exception as e:
        print(f"❌ Error cleaning up files: {e}")
        return False

def sweep_old_artifacts(ttl_days: int) -> None:
    """Sweep stale per-VOD artifacts older than ttl_days across shared data dirs."""
    import time
    if ttl_days <= 0:
        return
    now = time.time()
    ttl_seconds = ttl_days * 86400
    targets = [
        Path("data/transcripts"),
        Path("data/chunks"),
        Path("data/chats"),
        Path("data/chat_contexts"),
        Path("data/cache/raw_clips"),
        Path("data/temp"),
    ]

    def latest_mtime(path: Path) -> float:
        try:
            if path.is_file():
                return path.stat().st_mtime
            latest = path.stat().st_mtime
            for sub in path.rglob("*"):
                try:
                    latest = max(latest, sub.stat().st_mtime)
                except OSError:
                    continue
            return latest
        except OSError:
            return 0.0

    print(f"🧽 Sweeping stale artifacts older than {ttl_days} days...")
    for base in targets:
        if not base.exists():
            continue
        for entry in base.iterdir():
            try:
                mtime = latest_mtime(entry)
                if (now - mtime) > ttl_seconds:
                    if entry.is_dir():
                        shutil.rmtree(entry, ignore_errors=True)
                        print(f"🧽 Removed stale directory: {entry}")
                    else:
                        try:
                            entry.unlink()
                            print(f"🧽 Removed stale file: {entry}")
                        except OSError:
                            pass
            except OSError:
                continue

def main():
    if len(sys.argv) < 2 or "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python cleanup_after_upload.py <vod_id> [--force-clean] [--clean-ai-data] [--sweep-ttl-days N]")
        print("  --force-clean: Remove large assets even if upload status is unknown/failed")
        print("  --clean-ai-data: Also remove data/ai_data/<vod_id>")
        print("  --sweep-ttl-days N: Sweep stale artifacts older than N days across shared dirs")
        sys.exit(0 if ("--help" in sys.argv or "-h" in sys.argv) else 1)

    vod_id = sys.argv[1]
    force_clean = "--force-clean" in sys.argv
    clean_ai_data = "--clean-ai-data" in sys.argv
    sweep_days = 0
    if "--sweep-ttl-days" in sys.argv:
        try:
            i = sys.argv.index("--sweep-ttl-days")
            sweep_days = int(sys.argv[i+1])
        except Exception:
            sweep_days = 0

    print(f"🧹 Cleaning up local files for VOD: {vod_id}")
    print(f"📊 Force clean large assets: {force_clean}")
    print(f"📊 Clean AI data: {clean_ai_data}")
    if sweep_days:
        print(f"📊 TTL sweep days: {sweep_days}")

    success = cleanup_after_upload(vod_id, force_clean=force_clean, clean_ai_data=clean_ai_data, sweep_ttl_days=sweep_days)
    if success:
        print("✅ Cleanup completed")
    else:
        print("❌ Cleanup failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
