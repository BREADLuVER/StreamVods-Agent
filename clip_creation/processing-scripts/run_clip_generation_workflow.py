#!/usr/bin/env python3
"""
Complete clip generation workflow for local GPU processing.

This script handles the entire clip generation pipeline:
1. Download required files from S3 (clips JSONs, titles, etc.)
2. Create clips with advanced layout detection
3. Merge clips and generate metadata
4. Upload to YouTube (optional)

Usage:
  python processing-scripts/run_clip_generation_workflow.py <VOD_ID> [--from-manifest <S3_URI>]
"""

# --- universal log adapter -----------------------------------------------
import os, logging, sys
from pathlib import Path
# Add project root to path BEFORE trying to import utils (we're in clip_creation/processing-scripts/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)
if os.getenv("JOB_RUN_ID"):
    try:
        from utils.log import get_job_logger
        job_type = os.getenv("JOB_TYPE") or "clip"
        _, logger = get_job_logger(job_type, vod_id=os.getenv("VOD_ID", "unknown"))
        
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

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path (now we're in clip_creation/processing-scripts/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import config
from storage import StorageManager


def load_manifest(manifest_uri: str) -> Dict:
    """Load clip generation manifest from S3"""
    try:
        storage = StorageManager()
        manifest_data = storage.read_json(manifest_uri)
        print(f" Loaded manifest: {manifest_uri}")
        return manifest_data
    except Exception as e:
        print(f"X Failed to load manifest: {e}")
        raise


def download_required_files(vod_id: str, manifest: Optional[Dict] = None) -> List[str]:
    """Download required files from S3 for clip generation"""
    storage = StorageManager()
    s3_bucket = os.getenv('S3_BUCKET', 'streamsniped-dev-videos')
    
    # Create local directories
    ai_data_dir = config.get_ai_data_dir(vod_id)
    ai_data_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_files = []
    
    # Download core VOD data
    core_files = [
        f"ai_data/{vod_id}/{vod_id}_ai_data.json",
        f"ai_data/{vod_id}/{vod_id}_chapters.json",
        f"ai_data/{vod_id}/{vod_id}_clip_titles.json",
    ]
    
    for s3_key in core_files:
        local_path = ai_data_dir / Path(s3_key).name
        if not local_path.exists():
            try:
                s3_uri = f"s3://{s3_bucket}/{s3_key}"
                storage.download_file(s3_uri, str(local_path))
                downloaded_files.append(s3_key)
                print(f"📥 Downloaded: {s3_key}")
            except Exception as e:
                print(f" Failed to download {s3_key}: {e}")
    
    # Download per-chapter clips JSONs
    if manifest and 'chapters' in manifest:
        chapters = manifest['chapters']
    else:
        # Discover chapters from S3
        try:
            chapters = []
            prefix = f"ai_data/{vod_id}/"
            objects = storage.list_files(f"s3://{s3_bucket}/{prefix}")
            for obj in objects:
                if obj.endswith('_clips.json') and not obj.endswith('_direct_ai_clips.json'):
                    chapter_id = obj.split('/')[-1].replace(f"{vod_id}_", "").replace("_clips.json", "")
                    chapters.append(chapter_id)
        except Exception as e:
            print(f" Failed to discover chapters: {e}")
            chapters = []
    
    for chapter_id in chapters:
        clips_key = f"ai_data/{vod_id}/{vod_id}_{chapter_id}_clips.json"
        local_path = ai_data_dir / f"{vod_id}_{chapter_id}_clips.json"
        
        if not local_path.exists():
            try:
                s3_uri = f"s3://{s3_bucket}/{clips_key}"
                storage.download_file(s3_uri, str(local_path))
                downloaded_files.append(clips_key)
                print(f"📥 Downloaded: {clips_key}")
            except Exception as e:
                print(f" Failed to download {clips_key}: {e}")
    
    return downloaded_files


def run_clip_creation(vod_id: str, quality: str, chapters: List[str]) -> bool:
    """Run clip creation for each chapter"""
    print(f"\n Creating clips for {len(chapters)} chapters")
    
    success_count = 0
    for chapter_id in chapters:
        print(f"\n Processing chapter: {chapter_id}")
        
        # Run clip creation for this chapter
        cmd = [
            'python', 'processing-scripts/create_individual_clips.py',
            vod_id, quality,
            '--clips-file', f"data/ai_data/{vod_id}/{vod_id}_{chapter_id}_clips.json"
        ]
        
        try:
            # Don't capture output - let it go to the log files created by the orchestrator
            result = subprocess.run(cmd, check=True, text=True)
            print(f" Chapter {chapter_id} clips created successfully")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"X Failed to create clips for chapter {chapter_id}: {e}")
            # Output will be in log files, not captured here
    
    print(f"\n📊 Clip creation summary: {success_count}/{len(chapters)} chapters successful")
    return success_count > 0


def run_clip_merging(vod_id: str) -> bool:
    """Run clip merging to create unified files"""
    print(f"\n🔗 Merging clips for VOD: {vod_id}")
    
    cmd = ['python', 'processing-scripts/merge_chapter_clips.py', vod_id]
    
    try:
        # Don't capture output - let it go to the log files created by the orchestrator
        result = subprocess.run(cmd, check=True, text=True)
        print(" Clips merged successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"X Failed to merge clips: {e}")
        # Output will be in log files, not captured here
        return False


def run_metadata_generation(vod_id: str) -> bool:
    """Run clip metadata generation"""
    print(f"\n📝 Generating clip metadata for VOD: {vod_id}")
    
    cmd = ['python', 'processing-scripts/generate_clip_youtube_metadata.py', vod_id]
    
    try:
        # Don't capture output - let it go to the log files created by the orchestrator
        result = subprocess.run(cmd, check=True, text=True)
        print(" Clip metadata generated successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"X Failed to generate metadata: {e}")
        # Output will be in log files, not captured here
        return False


def run_youtube_upload(vod_id: str) -> bool:
    """Run YouTube upload for clips"""
    upload_youtube = os.getenv('UPLOAD_YOUTUBE', 'true').lower() in ['true', '1', 'yes']
    if not upload_youtube:
        print("⏭️ Skipping YouTube upload (UPLOAD_YOUTUBE=false)")
        return True
    
    print(f"\n📤 Uploading clips to YouTube for VOD: {vod_id}")
    
    cmd = ['python', 'processing-scripts/upload_clips_to_youtube.py', vod_id]
    
    try:
        # Don't capture output - let it go to the log files created by the orchestrator
        result = subprocess.run(cmd, check=True, text=True)
        print(" Clips uploaded to YouTube successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"X Failed to upload clips: {e}")
        # Output will be in log files, not captured here
        return False


def upload_results_to_s3(vod_id: str) -> bool:
    """Upload all generated results back to S3"""
    print(f"\n🪣 Uploading results to S3 for VOD: {vod_id}")
    
    storage = StorageManager()
    s3_bucket = os.getenv('S3_BUCKET', 'streamsniped-dev-videos')
    
    # Upload clips directory
    clips_dir = Path(f"data/clips/{vod_id}")
    if clips_dir.exists():
        try:
            for clip_file in clips_dir.rglob("*.mp4"):
                relative_path = clip_file.relative_to(Path("data/clips"))
                s3_key = f"clips/{relative_path}"
                s3_uri = f"s3://{s3_bucket}/{s3_key}"
                storage.upload_file(str(clip_file), s3_uri)
                print(f"📤 Uploaded clip: {s3_key}")
        except Exception as e:
            print(f" Failed to upload clips: {e}")
    
    # Upload AI data files
    ai_data_dir = config.get_ai_data_dir(vod_id)
    if ai_data_dir.exists():
        try:
            for json_file in ai_data_dir.glob("*.json"):
                if "clip" in json_file.name or "youtube" in json_file.name:
                    s3_key = f"ai_data/{vod_id}/{json_file.name}"
                    s3_uri = f"s3://{s3_bucket}/{s3_key}"
                    storage.upload_file(str(json_file), s3_uri)
                    print(f"📤 Uploaded metadata: {s3_key}")
        except Exception as e:
            print(f" Failed to upload metadata: {e}")
    
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('vod_id', help='Twitch VOD ID')
    parser.add_argument('--from-manifest', help='Process from clip generation manifest S3 URI')
    args = parser.parse_args()
    
    vod_id = args.vod_id
    quality = os.getenv('QUALITY', '1080p')
    
    print("=" * 80)
    print(" GPU Clip Generation Workflow")
    print("=" * 80)
    print(f"📺 VOD ID: {vod_id}")
    print(f"🎥 Quality: {quality}")
    print(f" Manifest: {args.from_manifest or 'None'}")
    print(f"🧠 Layout System: {os.getenv('CLIP_LAYOUT_SYSTEM', 'advanced')}")
    print("=" * 80)
    
    try:
        # Step 1: Load manifest and download required files
        manifest = None
        chapters = []
        
        if args.from_manifest:
            manifest = load_manifest(args.from_manifest)
            chapters = manifest.get('chapters', [])
            print(f" Manifest loaded: {len(chapters)} chapters")
        else:
            # Discover chapters from S3
            storage = StorageManager()
            s3_bucket = os.getenv('S3_BUCKET', 'streamsniped-dev-videos')
            prefix = f"ai_data/{vod_id}/"
            try:
                objects = storage.list_files(f"s3://{s3_bucket}/{prefix}")
                for obj in objects:
                    if obj.endswith('_clips.json') and not obj.endswith('_direct_ai_clips.json'):
                        chapter_id = obj.split('/')[-1].replace(f"{vod_id}_", "").replace("_clips.json", "")
                        chapters.append(chapter_id)
                print(f"🔍 Discovered {len(chapters)} chapters from S3")
            except Exception as e:
                print(f" Failed to discover chapters: {e}")
        
        if not chapters:
            print("X No chapters found for clip generation")
            sys.exit(1)
        
        # Download required files
        downloaded = download_required_files(vod_id, manifest)
        print(f"📥 Downloaded {len(downloaded)} files from S3")
        
        # Step 2: Create clips for each chapter
        if not run_clip_creation(vod_id, quality, chapters):
            print("X Clip creation failed")
            sys.exit(1)
        
        # Step 3: Merge clips
        if not run_clip_merging(vod_id):
            print("X Clip merging failed")
            sys.exit(1)
        
        # Step 4: Generate metadata
        if not run_metadata_generation(vod_id):
            print("X Metadata generation failed")
            sys.exit(1)
        
        # Step 5: Upload to YouTube (optional)
        if not run_youtube_upload(vod_id):
            print(" YouTube upload failed (non-critical)")
        
        # Step 6: Upload results to S3
        upload_results_to_s3(vod_id)
        
        print("\n" + "=" * 80)
        print("🎉 Clip generation workflow completed successfully!")
        print("=" * 80)
        print(f"📊 Processed {len(chapters)} chapters")
        print(f"📁 Outputs available in data/clips/{vod_id}/")
        print(f"📝 Metadata available in data/ai_data/{vod_id}/")
        print("=" * 80)
        
    except Exception as e:
        print(f"X Clip generation workflow failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
