#!/usr/bin/env python3
"""
Clean up temporary audio chunks after AI data generation.
Removes audio files from data/chunks/<vod_id>/chapter_*_part_* directories.
Also supports sweeping stale chunk folders older than a TTL across data/chunks.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def cleanup_audio_chunks(vod_id: str) -> bool:
    """Clean up temporary audio chunks for a VOD"""
    chunks_dir = Path(f"data/chunks/{vod_id}")
    
    if not chunks_dir.exists():
        print(f"📁 No chunks directory found for VOD {vod_id}")
        return True
    
    cleaned_count = 0
    total_size = 0
    
    try:
        # Find all audio files in chapter subdirectories
        for chapter_dir in chunks_dir.iterdir():
            if chapter_dir.is_dir() and chapter_dir.name.startswith("chapter_"):
                for part_dir in chapter_dir.iterdir():
                    if part_dir.is_dir() and part_dir.name.startswith("part_"):
                        # Look for audio files (mp3, wav, etc.)
                        for audio_file in part_dir.glob("*"):
                            if audio_file.is_file() and audio_file.suffix.lower() in ['.mp3', '.wav', '.m4a', '.aac']:
                                file_size = audio_file.stat().st_size
                                total_size += file_size
                                audio_file.unlink()
                                cleaned_count += 1
                                print(f"🗑️ Cleaned: {audio_file.relative_to(chunks_dir)}")
        
        # Try to remove empty directories (bottom-up approach)
        for chapter_dir in chunks_dir.iterdir():
            if chapter_dir.is_dir() and chapter_dir.name.startswith("chapter_"):
                try:
                    # Remove empty part directories first
                    for part_dir in chapter_dir.iterdir():
                        if part_dir.is_dir() and part_dir.name.startswith("part_"):
                            try:
                                # Check if directory is empty before removing
                                if not any(part_dir.iterdir()):
                                    part_dir.rmdir()
                                    print(f"📁 Removed empty directory: {part_dir.relative_to(chunks_dir)}")
                            except OSError:
                                pass  # Directory not empty or permission issue
                    
                    # Try to remove empty chapter directory
                    try:
                        if not any(chapter_dir.iterdir()):
                            chapter_dir.rmdir()
                            print(f"📁 Removed empty directory: {chapter_dir.relative_to(chunks_dir)}")
                    except OSError:
                        pass  # Directory not empty
                except OSError:
                    pass
        
        # Try to remove the entire chunks directory if empty
        try:
            if not any(chunks_dir.iterdir()):
                chunks_dir.rmdir()
                print(f"📁 Removed empty chunks directory: {chunks_dir}")
        except OSError:
            pass
        
        print(f"✅ Cleaned up {cleaned_count} audio files ({total_size / (1024*1024):.1f} MB)")
        return True
        
    except Exception as e:
        print(f"❌ Error cleaning audio chunks: {e}")
        return False

def sweep_stale_chunks(ttl_days: int) -> None:
    """Sweep chunk directories older than ttl_days under data/chunks."""
    if ttl_days <= 0:
        return
    import time
    base = Path("data/chunks")
    if not base.exists():
        return
    now = time.time()
    ttl_seconds = ttl_days * 86400

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

    print(f"🧽 Sweeping stale chunk directories older than {ttl_days} days...")
    for entry in base.iterdir():
        try:
            mtime = latest_mtime(entry)
            if (now - mtime) > ttl_seconds:
                if entry.is_dir():
                    # Remove directory tree
                    import shutil
                    shutil.rmtree(entry, ignore_errors=True)
                    print(f"🧽 Removed stale chunk dir: {entry}")
        except OSError:
            continue

def main():
    if len(sys.argv) < 2 or "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python cleanup_audio_chunks.py <vod_id> [--sweep-ttl-days N]")
        print("  --sweep-ttl-days N: Also sweep stale chunk dirs older than N days")
        sys.exit(0 if ("--help" in sys.argv or "-h" in sys.argv) else 1)

    vod_id = sys.argv[1]
    ttl_days = 0
    if "--sweep-ttl-days" in sys.argv:
        try:
            idx = sys.argv.index("--sweep-ttl-days")
            ttl_days = int(sys.argv[idx + 1])
        except Exception:
            ttl_days = 0

    print(f"🧹 Cleaning up audio chunks for VOD: {vod_id}")
    success = cleanup_audio_chunks(vod_id)
    if ttl_days > 0:
        sweep_stale_chunks(ttl_days)
    if success:
        print("✅ Audio chunk cleanup completed")
    else:
        print("❌ Audio chunk cleanup failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
