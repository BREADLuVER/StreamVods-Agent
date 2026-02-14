#!/usr/bin/env python3
"""
Merge Chapter Clips - Combine chapter-specific AI clipper results into unified file
This script creates the unified file that downstream processes expect while preserving 
chapter-specific files for backfill integrity.
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
from pathlib import Path
from typing import List, Dict

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.config import config
from storage import StorageManager


def load_chapter_clips(vod_id: str, chapter_name: str) -> List[Dict]:
    """Load clips from a specific chapter"""
    ai_data_dir = config.get_ai_data_dir(vod_id)
    chapter_clips_path = ai_data_dir / f"{vod_id}_{chapter_name}_clips.json"
    # Legacy fallback: if ID-based not found, try cleaned-name file
    if not chapter_clips_path.exists():
        legacy = ai_data_dir / f"{vod_id}_{chapter_name.lower()}_clips.json"
        if legacy.exists():
            chapter_clips_path = legacy
    
    if not chapter_clips_path.exists():
        print(f"  Chapter clips not found: {chapter_clips_path}")
        return []
    
    try:
        with open(chapter_clips_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        clips = data.get('missed_highlights', [])
        print(f" Loaded {len(clips)} clips from {chapter_name}")
        
        # Add chapter information to each clip
        for clip in clips:
            clip['source_chapter'] = chapter_name
        
        return clips
        
    except Exception as e:
        print(f"X Error loading {chapter_name} clips: {e}")
        return []


def find_all_chapter_clips(vod_id: str) -> List[str]:
    """Find all chapter-specific clip files for this VOD"""
    ai_data_dir = config.get_ai_data_dir(vod_id)
    
    if not ai_data_dir.exists():
        return []
    
    # Look for files matching pattern: {vod_id}_{chapter}_clips.json
    chapter_files = []
    for file_path in ai_data_dir.glob(f"{vod_id}_*_clips.json"):
        # Extract chapter name from filename
        filename = file_path.stem
        if filename.startswith(f"{vod_id}_") and filename.endswith("_clips"):
            chapter_name = filename[len(f"{vod_id}_"):-len("_clips")]
            if chapter_name != "direct_ai":  # Skip the unified file itself
                chapter_files.append(chapter_name)
    
    return sorted(chapter_files)


def create_unified_clips_file(vod_id: str) -> bool:
    """Create unified clips file from all chapter-specific files"""
    print(f"🔗 Creating unified clips file for VOD: {vod_id}")
    
    # Find all chapter clip files
    chapter_names = find_all_chapter_clips(vod_id)
    
    if not chapter_names:
        print(f"  No chapter-specific clip files found for VOD {vod_id} (continuing with empty unified file)")
        # Create empty unified file so downstream steps won't 404
        ai_data_dir = config.get_ai_data_dir(vod_id)
        unified_path = ai_data_dir / f"{vod_id}_direct_ai_clips.json"
        try:
            with open(unified_path, 'w', encoding='utf-8') as f:
                json.dump({"missed_highlights": [], "clips_selected": 0, "total_evaluated": 0}, f)
            print(f" Wrote empty unified clips file: {unified_path}")
        except Exception as e:
            print(f"  Could not write empty unified file: {e}")
        return True
    
    print(f" Found chapters: {', '.join(chapter_names)}")
    
    # Load clips from all chapters
    all_clips = []
    total_evaluated = 0
    rejection_summaries = []
    # Also gather per-chapter clip titles to merge later
    merged_titles = []
    
    for chapter_name in chapter_names:
        chapter_clips = load_chapter_clips(vod_id, chapter_name)
        all_clips.extend(chapter_clips)
        
        # Load metadata from chapter file
        try:
            ai_data_dir = config.get_ai_data_dir(vod_id)
            chapter_clips_path = ai_data_dir / f"{vod_id}_{chapter_name}_clips.json"
            
            with open(chapter_clips_path, 'r', encoding='utf-8') as f:
                chapter_data = json.load(f)
            
            total_evaluated += chapter_data.get('total_evaluated', 0)
            
            rejection_summary = chapter_data.get('rejection_summary', '')
            if rejection_summary:
                rejection_summaries.append(f"{chapter_name}: {rejection_summary}")
                
        except Exception as e:
            print(f"  Could not load metadata from {chapter_name}: {e}")

        # Load optional per-chapter titles file if present
        try:
            ai_data_dir = config.get_ai_data_dir(vod_id)
            titles_path = ai_data_dir / f"{vod_id}_{chapter_name}_clip_titles.json"
            if titles_path.exists():
                with open(titles_path, 'r', encoding='utf-8') as tf:
                    tdata = json.load(tf)
                titles = tdata.get('clip_titles', []) or []
                for t in titles:
                    t_copy = dict(t)
                    t_copy['chapter'] = chapter_name
                    merged_titles.append(t_copy)
        except Exception as e:
            print(f"  Could not load titles from {chapter_name}: {e}")
    
    # Sort clips by start time for consistency
    all_clips.sort(key=lambda x: x.get('start_time', 0))
    
    # Create unified data structure
    unified_data = {
        "missed_highlights": all_clips,
        "total_evaluated": total_evaluated,
        "clips_selected": len(all_clips),
        "rejection_summary": " | ".join(rejection_summaries) if rejection_summaries else f"Combined from {len(chapter_names)} chapters",
        "generation_method": "ai_direct_clipper_merged",
        "source_chapters": chapter_names,
        "merge_timestamp": str(int(time.time())) if 'time' in globals() else "unknown"
    }
    
    # Save unified file
    ai_data_dir = config.get_ai_data_dir(vod_id)
    unified_path = ai_data_dir / f"{vod_id}_direct_ai_clips.json"
    
    try:
        with open(unified_path, 'w', encoding='utf-8') as f:
            json.dump(unified_data, f, indent=2, ensure_ascii=False)
        
        print(f" Created unified clips file: {unified_path}")
        print(f"📊 Total clips: {len(all_clips)} from {len(chapter_names)} chapters")
        print(f"🎮 Chapters: {', '.join(chapter_names)}")
        
        # Upload to S3 for cloud compatibility
        try:
            upload_unified_to_s3(vod_id, unified_path)
        except Exception as e:
            print(f"  S3 upload failed: {e}")
        
        # Additionally, if we collected any per-chapter titles, merge into unified clip titles file
        try:
            if merged_titles:
                unified_titles_path = ai_data_dir / f"{vod_id}_clip_titles.json"
                payload = {
                    'vod_id': vod_id,
                    'clip_titles': merged_titles,
                    'metadata': {
                        'total_clips': len(merged_titles),
                        'generation_method': 'merged_per_chapter'
                    }
                }
                storage = StorageManager()
                storage.save_json_with_cloud_backup(
                    local_path=str(unified_titles_path),
                    data=payload,
                    s3_key=f"ai_data/{vod_id}/{vod_id}_clip_titles.json",
                    force_s3=True,
                )
                print(f" Merged per-chapter clip titles: {unified_titles_path}")
        except Exception as e:
            print(f"  Failed to merge clip titles: {e}")

        return True
        
    except Exception as e:
        print(f"X Failed to create unified file: {e}")
        return False


def upload_unified_to_s3(vod_id: str, unified_path: Path) -> None:
    """Upload unified clips file to S3"""
    try:
        import boto3
        s3 = boto3.client('s3')
        bucket_name = "streamsniped-dev-videos"
        s3_key = f"ai_data/{vod_id}/{vod_id}_direct_ai_clips.json"
        
        print(f"📤 Uploading unified file to S3: s3://{bucket_name}/{s3_key}")
        s3.upload_file(str(unified_path), bucket_name, s3_key)
        print(f" Uploaded unified clips to S3")
        
    except Exception as e:
        print(f"X Failed to upload to S3: {e}")
        raise


def main():
    """Main execution"""
    if len(sys.argv) < 2:
        print("Usage: python merge_chapter_clips.py <vod_id>")
        print("Example: python merge_chapter_clips.py 2507943627")
        sys.exit(1)
    
    vod_id = sys.argv[1]
    
    print(f" Merging chapter clips for VOD: {vod_id}")
    
    try:
        success = create_unified_clips_file(vod_id)
        
        if success:
            print(f"\n Chapter clips merged successfully!")
            print(f"📁 Unified file created for downstream processing")
            print(f" Chapter-specific files preserved for backfill integrity")
        else:
            print(f"\nX Failed to merge chapter clips")
            sys.exit(1)
            
    except Exception as e:
        print(f"X Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import time  # Import time module for timestamp
    main()