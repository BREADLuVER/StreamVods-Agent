#!/usr/bin/env python3
"""
Comprehensive cleanup script for local files after successful S3 upload.
Cleans up video files, chat files, clip files, and chunk files to prevent storage issues.
"""

import sys
import shutil
from pathlib import Path
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB"""
    try:
        return file_path.stat().st_size / (1024 * 1024)
    except OSError:
        return 0.0

def cleanup_directory(dir_path: Path, file_patterns: List[str] = None, keep_dirs: bool = False) -> Tuple[int, float]:
    """
    Clean up files in a directory
    
    Args:
        dir_path: Directory to clean
        file_patterns: List of file patterns to match (e.g., ['*.mp4', '*.json'])
        keep_dirs: Whether to keep empty directories
    
    Returns:
        Tuple of (file_count, total_size_mb)
    """
    if not dir_path.exists():
        return 0, 0.0
    
    cleaned_count = 0
    total_size = 0.0
    
    try:
        if file_patterns:
            # Clean specific file patterns
            for pattern in file_patterns:
                for file_path in dir_path.rglob(pattern):
                    if file_path.is_file():
                        size_mb = get_file_size_mb(file_path)
                        total_size += size_mb
                        file_path.unlink()
                        cleaned_count += 1
                        print(f"🗑️ Cleaned: {file_path.relative_to(dir_path.parent)} ({size_mb:.1f} MB)")
        else:
            # Clean all files in directory
            for file_path in dir_path.rglob("*"):
                if file_path.is_file():
                    size_mb = get_file_size_mb(file_path)
                    total_size += size_mb
                    file_path.unlink()
                    cleaned_count += 1
                    print(f"🗑️ Cleaned: {file_path.relative_to(dir_path.parent)} ({size_mb:.1f} MB)")
        
        # Remove empty directories if not keeping them
        if not keep_dirs:
            # Remove directories bottom-up
            for dir_to_remove in sorted(dir_path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if dir_to_remove.is_dir() and dir_to_remove != dir_path:
                    try:
                        if not any(dir_to_remove.iterdir()):
                            dir_to_remove.rmdir()
                            print(f"📁 Removed empty directory: {dir_to_remove.relative_to(dir_path.parent)}")
                    except OSError:
                        pass  # Directory not empty or permission issue
            
            # Try to remove the main directory if empty
            try:
                if not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    print(f"📁 Removed empty directory: {dir_path}")
            except OSError:
                pass
        
        return cleaned_count, total_size
        
    except Exception as e:
        print(f"❌ Error cleaning {dir_path}: {e}")
        return cleaned_count, total_size

def cleanup_entries_matching_vod_id(base_dir: Path, vod_id: str, file_patterns: List[str] | None = None) -> Tuple[int, float]:
    """
    Remove files and subdirectories within base_dir whose path contains the given vod_id.
    This is used for shared directories that are not organized strictly per-VOD.

    Returns:
        Tuple of (item_count_removed, total_size_mb)
    """
    if not base_dir.exists():
        return 0, 0.0

    removed_count = 0
    total_size_mb = 0.0

    try:
        search_patterns = file_patterns if file_patterns else ["**/*"]
        # Collect candidates first to avoid modifying while iterating
        candidates: List[Path] = []
        for pattern in search_patterns:
            candidates.extend(base_dir.rglob(pattern))

        # Sort by depth descending so child items are removed before parents
        for path in sorted(candidates, key=lambda p: len(p.parts), reverse=True):
            try:
                if vod_id not in str(path):
                    continue
                if path.is_file():
                    total_size_mb += get_file_size_mb(path)
                    path.unlink()
                    removed_count += 1
                    print(f"🗑️ Removed file: {path.relative_to(base_dir)}")
                elif path.is_dir():
                    # Only remove directory if its name or full path includes the vod_id
                    shutil.rmtree(path, ignore_errors=True)
                    removed_count += 1
                    print(f"📁 Removed directory: {path.relative_to(base_dir)}")
            except Exception:
                # Best-effort cleanup
                pass

        return removed_count, total_size_mb
    except Exception as e:
        print(f"❌ Error cleaning entries in {base_dir}: {e}")
        return removed_count, total_size_mb

def cleanup_vod_files(vod_id: str) -> bool:
    """Clean up VOD video files"""
    vod_dir = Path(f"data/vods/{vod_id}")
    if not vod_dir.exists():
        print(f"📁 No VOD directory found for {vod_id}")
        return True
    
    print(f"🧹 Cleaning VOD files for {vod_id}...")
    file_patterns = ['*.mp4', '*.mkv', '*.avi', '*.mov', '*.webm']
    count, size = cleanup_directory(vod_dir, file_patterns)
    print(f"✅ Cleaned {count} VOD files ({size:.1f} MB)")
    return True

def cleanup_chat_files(vod_id: str) -> bool:
    """Clean up chat files"""
    chat_dir = Path(f"data/chats/{vod_id}")
    if not chat_dir.exists():
        print(f"📁 No chat directory found for {vod_id}")
        return True
    
    print(f"🧹 Cleaning chat files for {vod_id}...")
    file_patterns = ['*.json', '*.txt', '*.csv']
    count, size = cleanup_directory(chat_dir, file_patterns)
    print(f"✅ Cleaned {count} chat files ({size:.1f} MB)")
    return True

def cleanup_transcript_files(vod_id: str) -> bool:
    """Clean up transcript files from shared transcripts directory."""
    transcripts_dir = Path("data/transcripts")
    if not transcripts_dir.exists():
        print("📁 No transcripts directory found")
        return True

    print(f"🧹 Cleaning transcript files for {vod_id}...")
    count, size = cleanup_entries_matching_vod_id(transcripts_dir, vod_id, [f"**/*{vod_id}*"])
    print(f"✅ Cleaned {count} transcript items ({size:.1f} MB)")
    return True

def cleanup_chat_contexts(vod_id: str) -> bool:
    """Clean up chat context artifacts from shared directory."""
    ctx_dir = Path("data/chat_contexts")
    if not ctx_dir.exists():
        print("📁 No chat_contexts directory found")
        return True

    print(f"🧹 Cleaning chat context files for {vod_id}...")
    count, size = cleanup_entries_matching_vod_id(ctx_dir, vod_id, [f"**/*{vod_id}*"])
    print(f"✅ Cleaned {count} chat context items ({size:.1f} MB)")
    return True

def cleanup_vector_stores(vod_id: str) -> bool:
    """Clean up vector store artifacts for the VOD."""
    vec_dir = Path("data/vector_stores")
    if not vec_dir.exists():
        print("📁 No vector_stores directory found")
        return True

    print(f"🧹 Cleaning vector store files for {vod_id}...")
    count, size = cleanup_entries_matching_vod_id(vec_dir, vod_id, [f"**/*{vod_id}*"])
    print(f"✅ Cleaned {count} vector store items ({size:.1f} MB)")
    return True

def cleanup_raw_clip_cache(vod_id: str) -> bool:
    """Clean up cached raw clip artifacts for the VOD."""
    cache_dir = Path("data/cache/raw_clips")
    if not cache_dir.exists():
        print("📁 No cache/raw_clips directory found")
        return True

    print(f"🧹 Cleaning raw clip cache for {vod_id}...")
    count, size = cleanup_entries_matching_vod_id(cache_dir, vod_id, [f"**/*{vod_id}*"])
    print(f"✅ Cleaned {count} raw clip cache items ({size:.1f} MB)")
    return True

def cleanup_clip_files(vod_id: str) -> bool:
    """Clean up clip files"""
    clip_dir = Path(f"data/clips/{vod_id}")
    if not clip_dir.exists():
        print(f"📁 No clip directory found for {vod_id}")
        return True
    
    print(f"🧹 Cleaning clip files for {vod_id}...")
    file_patterns = ['*.mp4', '*.mkv', '*.avi', '*.mov', '*.webm']
    count, size = cleanup_directory(clip_dir, file_patterns)
    print(f"✅ Cleaned {count} clip files ({size:.1f} MB)")
    return True

def cleanup_chunk_files(vod_id: str) -> bool:
    """Clean up chunk files (audio chunks, video chunks, etc.)"""
    chunk_dir = Path(f"data/chunks/{vod_id}")
    if not chunk_dir.exists():
        print(f"📁 No chunk directory found for {vod_id}")
        return True
    
    print(f"🧹 Cleaning chunk files for {vod_id}...")
    file_patterns = ['*.mp3', '*.wav', '*.m4a', '*.aac', '*.mp4', '*.mkv']
    count, size = cleanup_directory(chunk_dir, file_patterns)
    print(f"✅ Cleaned {count} chunk files ({size:.1f} MB)")
    return True

def cleanup_temp_files(vod_id: str) -> bool:
    """Clean up temporary files"""
    temp_dir = Path(f"data/temp/{vod_id}")
    if not temp_dir.exists():
        print(f"📁 No temp directory found for {vod_id}")
        return True
    
    print(f"🧹 Cleaning temp files for {vod_id}...")
    count, size = cleanup_directory(temp_dir)  # Clean all files
    print(f"✅ Cleaned {count} temp files ({size:.1f} MB)")
    return True

def cleanup_director_cut_files(vod_id: str) -> bool:
    """Clean up Director's Cut files"""
    # Look for Director's Cut files in various locations
    possible_locations = [
        Path(f"data/vods/{vod_id}"),
        Path(f"data/clips/{vod_id}"),
        Path(".")  # Current directory
    ]
    
    cleaned_count = 0
    total_size = 0.0
    
    for location in possible_locations:
        if location.exists():
            # Look for Director's Cut files
            for file_path in location.glob(f"*{vod_id}*director*"):
                if file_path.is_file():
                    size_mb = get_file_size_mb(file_path)
                    total_size += size_mb
                    file_path.unlink()
                    cleaned_count += 1
                    print(f"🗑️ Cleaned Director's Cut: {file_path} ({size_mb:.1f} MB)")
    
    if cleaned_count > 0:
        print(f"✅ Cleaned {cleaned_count} Director's Cut files ({total_size:.1f} MB)")
    else:
        print("📁 No Director's Cut files found")
    
    return True

def cleanup_all_local_files(vod_id: str, keep_ai_data: bool = True, clips_only: bool = False) -> bool:
    """
    Comprehensive cleanup of all local files for a VOD after successful S3 upload
    
    Args:
        vod_id: VOD ID to clean up
        keep_ai_data: Whether to keep AI data files (default: True)
        clips_only: Whether to only clean clip files (default: False)
    """
    if clips_only:
        print(f"🧹 Starting clip-only cleanup for VOD {vod_id}...")
        cleanup_functions = [cleanup_clip_files]
    else:
        print(f"🧹 Starting comprehensive cleanup for VOD {vod_id}...")
        cleanup_functions = [
            cleanup_vod_files,
            cleanup_chat_files, 
            cleanup_transcript_files,
            cleanup_chat_contexts,
            cleanup_vector_stores,
            cleanup_raw_clip_cache,
            cleanup_clip_files,
            cleanup_chunk_files,
            cleanup_temp_files,
            cleanup_director_cut_files
        ]
    
    # Note: individual cleanup functions log their own results
    
    # Clean up different file types
    for cleanup_func in cleanup_functions:
        try:
            success = cleanup_func(vod_id)
            if not success:
                print(f"⚠️ Warning: {cleanup_func.__name__} had issues")
        except Exception as e:
            print(f"❌ Error in {cleanup_func.__name__}: {e}")
    
    # Optionally clean AI data (usually keep for analysis)
    if not keep_ai_data:
        ai_data_dir = Path(f"data/ai_data/{vod_id}")
        if ai_data_dir.exists():
            print(f"🧹 Cleaning AI data for {vod_id}...")
            count, size = cleanup_directory(ai_data_dir)
            print(f"✅ Cleaned {count} AI data files ({size:.1f} MB)")
    
    print(f"✅ Cleanup completed for VOD {vod_id}")
    return True

def sweep_old_artifacts(ttl_days: int) -> None:
    """
    Best-effort sweeping of stale artifacts older than ttl_days across shared data directories.
    Removes per-VOD folders or files whose latest modification time is older than the TTL.
    """
    if ttl_days <= 0:
        return

    import time
    now = time.time()
    ttl_seconds = ttl_days * 86400

    targets = [
        Path("data/transcripts"),
        Path("data/vector_stores"),
        Path("data/chunks"),
        Path("data/chats"),
        Path("data/chat_contexts"),
        Path("data/cache/raw_clips"),
        Path("data/ai_data"),
        Path("data/temp"),
    ]

    def latest_mtime(path: Path) -> float:
        if path.is_file():
            return path.stat().st_mtime
        latest = path.stat().st_mtime
        for sub in path.rglob("*"):
            try:
                latest = max(latest, sub.stat().st_mtime)
            except OSError:
                continue
        return latest

    print(f"🧽 Sweeping stale artifacts older than {ttl_days} days...")
    for base in targets:
        if not base.exists():
            continue
        # Consider top-level entries (files or directories)
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
        print("Usage: python cleanup_local_files.py <vod_id> [--keep-ai-data] [--clips-only] [--sweep-ttl-days N]")
        print("  --keep-ai-data: Keep AI data files (default: True)")
        print("  --clips-only: Only clean clip files (default: False)")
        print("  --sweep-ttl-days N: Also sweep stale artifacts older than N days")
        sys.exit(0 if "--help" in sys.argv or "-h" in sys.argv else 1)
    
    vod_id = sys.argv[1]
    keep_ai_data = "--keep-ai-data" in sys.argv or "--no-clean-ai" in sys.argv
    clips_only = "--clips-only" in sys.argv
    ttl_days = 0
    if "--sweep-ttl-days" in sys.argv:
        try:
            idx = sys.argv.index("--sweep-ttl-days")
            ttl_days = int(sys.argv[idx + 1])
        except Exception:
            ttl_days = 0
    
    print(f"🧹 Cleanup for VOD: {vod_id}")
    print(f"📊 Keep AI data: {keep_ai_data}")
    print(f"📊 Clips only: {clips_only}")
    
    success = cleanup_all_local_files(vod_id, keep_ai_data=keep_ai_data, clips_only=clips_only)
    if ttl_days > 0:
        sweep_old_artifacts(ttl_days)
    if success:
        print("✅ Local file cleanup completed successfully")
    else:
        print("❌ Local file cleanup had issues")
        sys.exit(1)

if __name__ == "__main__":
    main()
