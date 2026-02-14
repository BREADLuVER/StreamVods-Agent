#!/usr/bin/env python3
"""
Generate YouTube metadata for individual clips
Creates separate metadata for each clip with unique titles and descriptions
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
from typing import Dict, List, Optional, Tuple
import re
import subprocess
import json

# Add parent directory to Python path to import src module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config
from storage import StorageManager
def _safe_load_collaborators(vod_id: str) -> List[str]:
    """Optionally load a collaborators list from ai_data/<vod_id>/<vod_id>_collaborators.json.

    Expected schema: { "collaborators": ["Name1", "Name2", ...] } or a plain list.
    Returns an empty list if not present.
    """
    try:
        ai_data_dir = config.get_ai_data_dir(vod_id)
        fpath = ai_data_dir / f"{vod_id}_collaborators.json"
        if not fpath.exists():
            return []
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get('collaborators'), list):
            return [str(x).strip() for x in data['collaborators'] if str(x).strip()]
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
        return []
    except Exception:
        return []


def _ai_extract_extra_tags(vod_id: str, streamer: str, clip_title: str, reasoning: str) -> List[str]:
    """Use AI (optional) to extract collaborator/org/game tags from title + reasoning.

    Controlled by ENABLE_AI_TAGS env var. Returns [] if disabled or AI fails.
    """
    if os.getenv('ENABLE_AI_TAGS', 'false').lower() not in ['true', '1', 'yes']:
        return []
    prompt = (
        "You will extract concise tags for a YouTube gaming Short.\n"
        "Return ONLY a JSON array of strings (no preamble).\n"
        "Guidelines: include streamer handles/names involved (besides the main streamer), orgs/teams, event names, game modes/maps, and 1-2 relevant keywords.\n"
        f"Main streamer: {streamer or ''}\n"
        f"Clip title: {clip_title or ''}\n"
        f"Why the clip is good (reasoning): {reasoning or ''}\n"
        "Examples of good tags: ['Squeaks','OTV','Valorant','Team Canada','overtime']\n"
    )
    try:
        from src.ai_client import call_llm
        text = call_llm(prompt, max_tokens=120, temperature=0.2, request_tag="clip_tag_suggestion")
        tags = json.loads(text)
        if isinstance(tags, list):
            return [str(t).strip() for t in tags if str(t).strip()]
    except Exception:
        return []
    return []


def _augment_tags_with_collab_and_ai(
    vod_id: str,
    base_tags: List[str],
    streamer: str,
    clip_title: str,
    reasoning: str,
) -> List[str]:
    """Augment base tags with collaborators (file) and AI suggestions.

    - Adds unique names from collaborators file
    - Optionally adds AI-extracted tags
    """
    result: List[str] = []
    seen = set()

    def add(tag: str):
        t = str(tag).strip()
        if not t:
            return
        if t.lower() == 'unknown':
            return
        if t not in seen:
            seen.add(t)
            result.append(t)

    for t in base_tags:
        add(t)

    # Collaborators from file
    for t in _safe_load_collaborators(vod_id):
        if t != streamer:
            add(t)

    # AI suggestions (optional)
    for t in _ai_extract_extra_tags(vod_id, streamer, clip_title, reasoning):
        if t != streamer:
            add(t)

    return result


def _load_titles_from_manifest(vod_id: str) -> Tuple[List[Dict], bool]:
    """Try to load titles from vector_store clips_manifest.json.

    Returns (titles, used_manifest_flag).
    Each title item is {'title': str} to match legacy shape.
    """
    try:
        manifest_path = Path(f"data/vector_stores/{vod_id}/clips_manifest.json")
        if not manifest_path.exists():
            return ([], False)
        with open(manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        clips = data.get('clips') if isinstance(data, dict) else None
        if not isinstance(clips, list):
            return ([], False)
        out: List[Dict] = []
        for c in clips:
            if not isinstance(c, dict):
                continue
            t = str(c.get('title') or '').strip()
            if t:
                out.append({'title': t})
        if out:
            print(f" Loaded {len(out)} clip titles from clips_manifest.json")
            return (out, True)
        return ([], False)
    except Exception as e:
        print(f" Failed reading clips manifest for titles: {e}")
        return ([], False)


def load_clip_titles(vod_id: str) -> List[Dict]:
    """Load clip titles with new-manifest-first preference.

    Preference order:
    1) data/vector_stores/{vod_id}/clips_manifest.json (title per clip)
    2) Unified {vod_id}_clip_titles.json
    3) Fallback: concatenate all {vod_id}_*_clip_titles.json files
    """
    # 1) Prefer manifest titles
    titles_from_manifest, used_manifest = _load_titles_from_manifest(vod_id)
    if used_manifest and titles_from_manifest:
        return titles_from_manifest
    ai_data_dir = config.get_ai_data_dir(vod_id)
    titles_path = ai_data_dir / f"{vod_id}_clip_titles.json"

    if not titles_path.exists():
        # Default to S3 download if local missing
        try:
            storage = StorageManager()
            bucket = os.environ.get('S3_BUCKET', 'streamsniped-dev-videos')
            s3_uri = f"s3://{bucket}/ai_data/{vod_id}/{vod_id}_clip_titles.json"
            storage.download_file(s3_uri, str(titles_path))
        except Exception:
            pass

    if titles_path.exists():
        try:
            with open(titles_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            titles = data.get('clip_titles')
            if isinstance(titles, list) and titles:
                print(f" Loaded {len(titles)} clip titles from unified file")
                return titles
            titles = data.get('titles', [])
            if isinstance(titles, list) and titles:
                print(f" Loaded {len(titles)} clip titles from legacy 'titles'")
                return titles
        except Exception:
            pass

    # Fallback: load and merge per-chapter title files
    merged: List[Dict] = []
    for p in ai_data_dir.glob(f"{vod_id}_*_clip_titles.json"):
        if p.name == f"{vod_id}_clip_titles.json":
            continue
        try:
            with open(p, 'r', encoding='utf-8') as f:
                d = json.load(f)
            arr = d.get('clip_titles', []) or []
            for t in arr:
                merged.append(t)
        except Exception:
            continue
    if merged:
        print(f" Loaded {len(merged)} clip titles from per-chapter files")
        return merged

    print(f"X Clip titles not found: {titles_path}")
    return []


def load_clips_data(vod_id: str) -> List[Dict]:
    """Load clips data from clips manifest (new pipeline)"""
    try:
        # Try clips manifest first (new pipeline)
        manifest_path = Path(f"data/vector_stores/{vod_id}/clips_manifest.json")
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            clips = data.get('clips', [])
            if clips:
                print(f" Loaded {len(clips)} clips from clips manifest")
                return clips
        
        # Fallback to AI Direct Clipper results
        ai_data_dir = config.get_ai_data_dir(vod_id)
        direct_clips_path = ai_data_dir / f"{vod_id}_direct_ai_clips.json"
        
        if not direct_clips_path.exists():
            try:
                storage = StorageManager()
                bucket = os.environ.get('S3_BUCKET', 'streamsniped-dev-videos')
                s3_uri = f"s3://{bucket}/ai_data/{vod_id}/{vod_id}_direct_ai_clips.json"
                storage.download_file(s3_uri, str(direct_clips_path))
            except Exception:
                pass

        if direct_clips_path.exists():
            with open(direct_clips_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            clips = data.get('missed_highlights', [])
            print(f" Loaded {len(clips)} clips from AI Direct Clipper")
            return clips
        
        # Fallback to classification sections
        focused_path = config.get_focused_dir(vod_id) / f"{vod_id}_sections.json"
        if not focused_path.exists():
            # Also check root ai_data (legacy path), then try S3 focused and root
            root_ai = config.get_ai_data_dir(vod_id) / f"{vod_id}_sections.json"
            if root_ai.exists():
                focused_path = root_ai
            else:
                try:
                    storage = StorageManager()
                    bucket = os.environ.get('S3_BUCKET', 'streamsniped-dev-videos')
                    # Try focused path first
                    s3_uri = f"s3://{bucket}/ai_data/{vod_id}/focused/{vod_id}_sections.json"
                    storage.download_file(s3_uri, str(focused_path))
                except Exception:
                    # Fallback to legacy root path
                    try:
                        s3_uri = f"s3://{bucket}/ai_data/{vod_id}/{vod_id}_sections.json"
                        storage.download_file(s3_uri, str(root_ai))
                        focused_path = root_ai
                    except Exception:
                        pass
        if focused_path.exists():
            with open(focused_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            sections = data.get('sections', [])
            print(f" Loaded {len(sections)} sections from classification")
            return sections
        
        print(f"X No clips data found for VOD {vod_id}")
        return []
    except Exception as e:
        print(f"X Error loading clips data: {e}")
        return []


def get_streamer_info(vod_id: str) -> Dict:
    """Get streamer information; prefer live CLI query, fallback to saved JSON.

    Normalizes into keys: streamer, original_title, duration (seconds when possible).
    """
    try:
        # Prefer centralized stream context for streamer/title/duration
        try:
            sc_path = config.get_ai_data_dir(vod_id) / f"{vod_id}_stream_context.json"
            if sc_path.exists():
                with open(sc_path, 'r', encoding='utf-8') as f:
                    sc = json.load(f)
                streamer = str(sc.get('streamer') or '').strip()
                vod_title = str(sc.get('vod_title') or '').strip()
                duration = int(sc.get('duration') or 0)
                if streamer or vod_title or duration:
                    print(" Using stream_context for streamer/title/duration")
                    return {
                        'streamer': streamer,
                        'original_title': vod_title,
                        'duration': duration,
                    }
        except Exception as _e:
            print(f" stream_context unavailable for clips: {_e}")

        # 1) Try TwitchDownloaderCLI for the most reliable structured info
        try:
            result = subprocess.run(
                ["TwitchDownloaderCLI", "info", "--id", vod_id, "--format", "table"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                check=True,
            )
            lines = result.stdout.strip().split('\n')
            table_info: Dict[str, str] = {}
            for line in lines:
                if '│' in line:
                    parts = line.split('│')
                    if len(parts) >= 3:
                        key = parts[1].strip()
                        value = parts[2].strip()
                        table_info[key] = value
            if table_info:
                streamer_cli = table_info.get('Streamer') or table_info.get('UserName')
                title_cli = table_info.get('Title')
                duration_cli = table_info.get('Duration') or table_info.get('Length')
                duration_seconds = 0
                # Duration might be like '2:20:10' or '2h34m55s'
                if duration_cli:
                    if re.match(r"^\d{1,2}:\d{2}:\d{2}$", duration_cli):
                        h, m, s = [int(x) for x in duration_cli.split(':')]
                        duration_seconds = h * 3600 + m * 60 + s
                    else:
                        total = 0
                        for value, unit in re.findall(r"(\d+)([hms])", duration_cli):
                            v = int(value)
                            total += v * (3600 if unit == 'h' else 60 if unit == 'm' else 1)
                        duration_seconds = total
                return {
                    'streamer': streamer_cli or '',
                    'original_title': title_cli or '',
                    'duration': duration_seconds,
                }
        except Exception:
            pass

        # 2) Fallback to saved JSON file produced earlier in the pipeline
        ai_data_dir = config.get_ai_data_dir(vod_id)
        vod_info_path = ai_data_dir / f"{vod_id}_vod_info.json"

        if not vod_info_path.exists():
            try:
                storage = StorageManager()
                bucket = os.environ.get('S3_BUCKET', 'streamsniped-dev-videos')
                s3_uri = f"s3://{bucket}/ai_data/{vod_id}/{vod_id}_vod_info.json"
                storage.download_file(s3_uri, str(vod_info_path))
            except Exception:
                pass

        if vod_info_path.exists():
            with open(vod_info_path, 'r', encoding='utf-8') as f:
                vod_info = json.load(f)

            streamer = (
                vod_info.get('streamer')
                or vod_info.get('Streamer')
                or vod_info.get('UserName')
                or vod_info.get('user_name')
                or vod_info.get('channel')
                or ''
            )
            original_title = (
                vod_info.get('title')
                or vod_info.get('Title')
                or vod_info.get('original_title')
                or ''
            )
            duration = vod_info.get('duration') or vod_info.get('Duration') or 0

            return {
                'streamer': streamer,
                'original_title': original_title,
                'duration': duration,
            }

        return {'streamer': 'Unknown', 'original_title': 'Unknown', 'duration': 0}

    except Exception as e:
        print(f" Error getting streamer info: {e}")
        return {'streamer': 'Unknown', 'original_title': 'Unknown', 'duration': 0}


def get_game_tags_from_chapters(vod_id: str) -> List[str]:
    """Extract game/category names from chapters to use as tags/hashtags."""
    try:
        ai_data_dir = config.get_ai_data_dir(vod_id)
        chapters_file = ai_data_dir / f"{vod_id}_chapters.json"
        if not chapters_file.exists():
            try:
                storage = StorageManager()
                bucket = os.environ.get('S3_BUCKET', 'streamsniped-dev-videos')
                s3_uri = f"s3://{bucket}/ai_data/{vod_id}/{vod_id}_chapters.json"
                storage.download_file(s3_uri, str(chapters_file))
            except Exception:
                pass
        if not chapters_file.exists():
            return []

        with open(chapters_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        chapters = data
        if isinstance(data, dict) and 'chapters' in data:
            chapters = data['chapters']
        if not isinstance(chapters, list):
            return []

        tags: List[str] = []
        for chapter in chapters:
            if not isinstance(chapter, dict):
                continue
            category = (chapter.get('category') or '').strip()
            if category and category not in tags:
                tags.append(category)
        return tags
    except Exception:
        return []


def create_clip_description(clip_data: Dict, streamer_info: Dict, vod_id: str) -> str:
    """Create minimal description for individual clip with just links and hashtags"""
    streamer = streamer_info.get('streamer', '')
    
    # Twitch links
    twitch_vod_url = f"https://www.twitch.tv/videos/{vod_id}"
    twitch_channel_url = f"https://www.twitch.tv/{str(streamer).lower()}" if streamer else ""

    # Game hashtags
    game_tags = get_game_tags_from_chapters(vod_id)
    hashtags = "#TwitchHighlights #Gaming #Shorts"
    for tag in game_tags:
        clean = re.sub(r"[^a-zA-Z0-9]", "", tag)
        if clean:
            hashtags += f" #{clean}"
    if streamer:
        clean_streamer = re.sub(r"[^a-zA-Z0-9]", "", streamer)
        if clean_streamer:
            hashtags += f" #{clean_streamer}"

    description_parts = []
    description_parts.append(f"Original VOD: {twitch_vod_url}")
    if twitch_channel_url:
        description_parts.append(f"Streamer: {twitch_channel_url}")
    description_parts.append("")
    description_parts.append(hashtags)

    return "\n".join(description_parts)


def create_clip_metadata(vod_id: str, clip_index: int, clip_data: Dict, title_data: Dict) -> Dict:
    """Create YouTube metadata for individual clip.
    
    For Shorts:
    - Adds #Shorts hashtag to title
    - Sets vertical video metadata
    - Ensures title + hashtag under 100 chars
    """
    streamer_info = get_streamer_info(vod_id)
    streamer = streamer_info.get('streamer', 'Unknown')
    
    # Get clip title
    clip_title = title_data.get('title', f'Clip {clip_index}')
    
    # Create description
    description = create_clip_description(clip_data, streamer_info, vod_id)
    
    # Determine if this should be treated as a Short (<=181s by YouTube leniency)
    clip_duration_s = clip_data.get('duration', 0)
    IS_SHORT_MAX = 181  # seconds (3:01) – allow slight buffer before YT treats as video
    is_short = clip_duration_s <= IS_SHORT_MAX

    # Build title: always tag the streamer instead of #Shorts
    streamer_tag = f" #{streamer}" if streamer else ""
    MAX_TITLE_LENGTH = 100 - len(streamer_tag)
    final_title = (clip_title[:MAX_TITLE_LENGTH].strip() + streamer_tag).strip()

    # Tags: include Shorts-focused hashtags only when uploading as Short
    shorts_tags = [
        "shorts", "short", "viralvideos", "viralshort", "funny", 
        "youtubeshorts", "shortsfeed", "TwitchHighlights", "Gaming"
    ] if is_short else []

    base_tags = [streamer] + shorts_tags + get_game_tags_from_chapters(vod_id)
    cleaned_tags = _augment_tags_with_collab_and_ai(
        vod_id=vod_id,
        base_tags=base_tags,
        streamer=streamer,
        clip_title=clip_title,
        reasoning=clip_data.get('rationale', ''),  # Changed from 'reasoning' to 'rationale'
    )

    metadata = {
        "snippet": {
            "title": final_title,
            # Indicate vertical format only for Shorts (YouTube ignores otherwise)
            **({
                "resourceId": {"kind": "youtube#video", "videoId": ""},
                "videoFormat": "vertical",
            } if is_short else {}),
            "description": description,
            # Include streamer, game tags, collaborators for discoverability
            "tags": cleaned_tags,
            "categoryId": "20",  # Gaming category
            "defaultLanguage": "en",
            "defaultAudioLanguage": "en"
        },
        "status": {
            "privacyStatus": "public",  # Upload as public
            "madeForKids": False,
            "selfDeclaredMadeForKids": False
        },
        # Additional metadata for tracking
        "streamsniped_metadata": {
            "vod_id": vod_id,
            "clip_index": clip_index,
            "streamer": streamer,
            "original_title": streamer_info.get('original_title', ''),
            "clip_start_time": clip_data.get('start', 0),  # Changed from 'start_time' to 'start'
            "clip_end_time": clip_data.get('end', 0),     # Changed from 'end_time' to 'end'
            "clip_duration": clip_duration_s,
            "clip_score": clip_data.get('score', 0),
            "processing_date": None  # Will be set when uploading
        }
    }
    
    return metadata


def save_clip_metadata(vod_id: str, clip_index: int, metadata: Dict) -> Path:
    """Save clip metadata to file"""
    ai_data_dir = config.get_ai_data_dir(vod_id)
    output_file = ai_data_dir / f"{vod_id}_clip_{clip_index:02d}_youtube_metadata.json"
    
    storage = StorageManager()
    success = storage.save_json_with_cloud_backup(
        local_path=str(output_file),
        data=metadata,
        s3_key=f"ai_data/{vod_id}/{vod_id}_clip_{clip_index:02d}_youtube_metadata.json",
        force_s3=True
    )
    
    if not success:
        raise RuntimeError(f"Failed to save clip metadata to {output_file}")
    
    return output_file


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python generate_clip_youtube_metadata.py <vod_id>")
        sys.exit(1)
    
    vod_id = sys.argv[1]
    
    try:
        print(f" Generating YouTube metadata for clips from VOD: {vod_id}")
        
        # Load clip data
        clip_titles = load_clip_titles(vod_id)
        clips_data = load_clips_data(vod_id)
        
        if not clip_titles or not clips_data:
            print("X No clip data found")
            sys.exit(1)
        
        # Limit to matching number of clips
        num_clips = min(len(clip_titles), len(clips_data))
        print(f"📊 Generating metadata for {num_clips} clips")
        
        # Generate metadata for each clip
        for i in range(num_clips):
            print(f"\n Generating metadata for clip {i+1}/{num_clips}")
            
            title_data = clip_titles[i]
            clip_data = clips_data[i]
            
            # Create metadata
            metadata = create_clip_metadata(vod_id, i+1, clip_data, title_data)
            
            # Save metadata
            output_file = save_clip_metadata(vod_id, i+1, metadata)
            
            print(f" Clip {i+1} metadata saved: {output_file.name}")
            print(f"📝 Title: {metadata['snippet']['title']}")
            print(f"📊 Privacy: {metadata['status']['privacyStatus']}")
        
        # Write an index file for cache robustness
        try:
            index = {
                "vod_id": vod_id,
                "count": num_clips,
                "files": [f"{vod_id}_clip_{i+1:02d}_youtube_metadata.json" for i in range(num_clips)],
            }
            ai_data_dir = config.get_ai_data_dir(vod_id)
            index_path = ai_data_dir / f"{vod_id}_clip_metadata_index.json"
            storage = StorageManager()
            storage.save_json_with_cloud_backup(
                local_path=str(index_path),
                data=index,
                s3_key=f"ai_data/{vod_id}/{vod_id}_clip_metadata_index.json",
                force_s3=True,
            )
            print(f"📝 Metadata index written: {index_path}")
        except Exception as e:
            print(f" Failed to write metadata index: {e}")

        print(f"\n🎉 Generated metadata for {num_clips} clips!")
        
    except Exception as e:
        print(f"X Error generating clip metadata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 