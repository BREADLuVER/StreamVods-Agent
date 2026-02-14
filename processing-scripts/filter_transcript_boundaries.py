#!/usr/bin/env python3
"""
Filter transcript boundaries — find real start/end of streamer content.
Runs after generate_ai_data_cloud.py, before classification.

AI-only: uses first/last 10 minutes of transcript + chat with the unified AI client
to determine start/end. No heuristic fallbacks.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import re

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.config import config
from src.ai_client import call_llm
from src.chat_utils import chat_utils
from storage import StorageManager


def load_ai_data(vod_id: str) -> List[Dict]:
    """Load AI data from local or S3"""
    storage = StorageManager()
    
    # Try local first
    ai_data_dir = config.get_ai_data_dir(vod_id)
    ai_data_path = ai_data_dir / f"{vod_id}_ai_data.json"
    
    if ai_data_path.exists():
        with open(ai_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("📄 AI data source: raw (ai_data.json)")
        return data.get('segments', [])
    
    # Try S3
    s3_bucket = os.environ.get('S3_BUCKET', 'streamsniped-dev-videos')
    s3_uri = f"s3://{s3_bucket}/ai_data/{vod_id}/{vod_id}_ai_data.json"
    
    try:
        data = storage.read_json(s3_uri)
        print(f"[OK] Loaded AI data from S3: {s3_uri}")
        print("📄 AI data source: raw (ai_data.json from S3)")
        return data.get('segments', [])
    except Exception:
        pass
    
    raise FileNotFoundError(f"AI data not found locally or in S3: {ai_data_path}")


def _extract_transcript_window(
    ai_data: List[Dict],
    window_start: int,
    window_end: int,
) -> List[Dict]:
    """Return a compact list of transcript segments within [window_start, window_end]."""
    window_segments: List[Dict] = []
    for seg in ai_data:
        st = int(seg.get('start_time', 0))
        et = int(seg.get('end_time', 0))
        if st > window_end or et < window_start:
            continue
        text = (seg.get('transcript') or '').strip()
        window_segments.append({
            'start_time': st,
            'end_time': et,
            'duration': max(0, et - st),
            'transcript': text
        })
    return window_segments


def _load_chat_df(vod_id: str):
    """Load and parse chat to DataFrame. Returns (df or None)."""
    chat_dir = config.get_chat_dir(vod_id)
    chat_path = chat_dir / f"{vod_id}_chat.json"
    if not chat_path.exists():
        return None
    try:
        raw = chat_utils.load_chat(chat_path)
        df = chat_utils.parse_chat_messages(raw)
        return df
    except Exception:
        return None


def _compute_chat_activity_baseline(chat_df) -> float:
    """Compute global average chat messages per second (smoothed)."""
    if chat_df is None or chat_df.empty:
        return 0.0
    activity = chat_utils.calculate_chat_activity(chat_df)
    if activity.empty:
        return 0.0
    # Use smoothed_count mean as baseline
    try:
        return float(activity['smoothed_count'].mean())
    except Exception:
        return 0.0


def _get_chat_window_sample(chat_df, start_s: int, end_s: int, max_messages: int = 200, sample_step: int = 3) -> List[Dict]:
    """Return a compact list of chat messages within [start_s, end_s]."""
    if chat_df is None or chat_df.empty:
        return []
    window = chat_df[(chat_df['second'] >= start_s) & (chat_df['second'] <= end_s)]
    if window.empty:
        return []
    # Downsample by time step to keep prompt compact
    if sample_step > 1:
        window = window[window['second'] % sample_step == 0]
    # Cap total messages
    if len(window) > max_messages:
        window = window.iloc[:max_messages]
    messages: List[Dict] = []
    for _, row in window.iterrows():
        messages.append({
            'timestamp': int(row['second']),
            'content': str(row.get('content', ''))[:200]
        })
    return messages


# Heuristic helpers removed — AI-only path


def _extract_boundaries_from_text(raw_text: str) -> Dict:
    """Robustly extract a JSON object with start_time/end_time from arbitrary LLM output.

    Strategy:
    1) Search for JSON-looking substrings and try json.loads on each.
    2) Inspect fenced code blocks (``` ... ```), trim to first/last braces and parse.
    3) Regex fallback for lines like "start_time: 123" and "end_time: 456" (any order).
    """
    text = (raw_text or "").strip()

    def _try_parse_json_snippet(snippet: str) -> Dict:
        snippet = snippet.strip()
        # Trim to the outermost braces if present
        if "{" in snippet and "}" in snippet:
            first = snippet.find("{")
            last = snippet.rfind("}")
            if first != -1 and last != -1 and last > first:
                snippet = snippet[first:last + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict) and "start_time" in obj and "end_time" in obj:
                return obj
        except Exception:
            pass
        return {}

    # 1) Direct attempt on the whole text
    obj = _try_parse_json_snippet(text)
    if obj:
        return obj

    # 2) Split on code fences and try each chunk
    if "```" in text:
        parts = [p.strip() for p in text.split("```") if p.strip()]
        # Try longer parts first
        parts.sort(key=len, reverse=True)
        for p in parts:
            # If language tag present on first line, drop it
            lines = p.splitlines()
            if lines and re.match(r"^[a-zA-Z0-9_+-]+$", lines[0]):
                p = "\n".join(lines[1:])
            obj = _try_parse_json_snippet(p)
            if obj:
                return obj

    # 3) Regex for inline JSON-like object containing both keys
    # Use a single-line alternation; VERBOSE is unnecessary here
    json_like_matches = re.findall(r"\{[^{}]*start_time[^{}]*\}|\{[^{}]*end_time[^{}]*\}", text, flags=re.IGNORECASE | re.DOTALL)
    for m in json_like_matches:
        obj = _try_parse_json_snippet(m)
        if obj:
            return obj

    # 4) Regex fallback for key:value patterns in any order
    # Match start_time first, then end_time
    m = re.search(r"start[_\s-]*time\s*[:=]\s*(-?\d+)[\s\S]*?end[_\s-]*time\s*[:=]\s*(-?\d+)", text, flags=re.IGNORECASE)
    if not m:
        # Or end_time first, then start_time
        m = re.search(r"end[_\s-]*time\s*[:=]\s*(-?\d+)[\s\S]*?start[_\s-]*time\s*[:=]\s*(-?\d+)", text, flags=re.IGNORECASE)
        if m:
            # swap groups to (start, end)
            start_val, end_val = m.group(2), m.group(1)
        else:
            start_val = end_val = None
    else:
        start_val, end_val = m.group(1), m.group(2)

    if start_val is not None and end_val is not None:
        try:
            return {"start_time": int(start_val), "end_time": int(end_val)}
        except Exception:
            pass

    raise ValueError("Could not extract JSON boundaries from LLM output")


def filter_by_transcript_content(ai_data: List[Dict]) -> Tuple[int, int, List[Dict]]:
    """Find real content boundaries via AI using transcript + chat windows."""
    
    if not ai_data:
        print("[WARN] No AI data available for filtering")
        return 0, 0, []
    
    # Check if this is a very short VOD
    total_duration = ai_data[-1]['end_time'] - ai_data[0]['start_time']
    if total_duration < 300:  # Less than 5 minutes
        print(f"[WARN] Very short VOD detected: {total_duration/60:.1f} minutes")
        print("[INFO] Skipping transcript filtering for short content")
        return ai_data[0]['start_time'], ai_data[-1]['end_time'], ai_data
    
    # Use larger windows for better context
    vod_start = int(ai_data[0]['start_time'])
    vod_end = int(ai_data[-1]['end_time'])
    total_seconds = max(0, vod_end - vod_start)
    base_window = 600
    window = int(min(base_window, max(300, total_seconds // 2)))  # at least 5 min, at most 1/2 VOD
    head_start, head_end = vod_start, min(vod_start + window, vod_end)
    tail_start, tail_end = max(vod_start, vod_end - window), vod_end

    # Build compact transcript windows
    head_transcripts = _extract_transcript_window(ai_data, head_start, head_end)
    tail_transcripts = _extract_transcript_window(ai_data, tail_start, tail_end)

    # Load chat once and compute baselines
    vod_id = os.getenv('CURRENT_VOD_ID')  # optional hint if set by caller
    chat_df = None
    if vod_id:
        chat_df = _load_chat_df(vod_id)
    else:
        # Try to infer vod_id from first segment if present
        try:
            # Not always available; safe to ignore
            pass
        except Exception:
            pass

    global_chat_avg = _compute_chat_activity_baseline(chat_df)
    head_chat = _get_chat_window_sample(chat_df, head_start, head_end)
    tail_chat = _get_chat_window_sample(chat_df, tail_start, tail_end)

    # Add some randomization to make the prompt less deterministic
    # (kept minimal; not used directly in the prompt to avoid unstable diffs)
    # import random  # disabled to avoid unused-variable warnings
    # _random_seed = random.randint(1, 1000)
    
    prompt = f"""
Analyze this Twitch VOD transcript to find the real content boundaries.

VOD RANGE: {vod_start}s to {vod_end}s (total: {total_seconds//60} minutes)

TASK: Find where the streamer actually starts and stops talking.

REMOVE:
- Intro screens (no streamer speech)
- Outro screens (no streamer speech)
- Repetitive Whisper hallucinations
- Background music-only segments

KEEP:
- All streamer speech, even greetings and goodbyes
- Varied, meaningful content

START WINDOW (first {window//60} minutes):
{json.dumps(head_transcripts, ensure_ascii=False)}

END WINDOW (last {window//60} minutes):
{json.dumps(tail_transcripts, ensure_ascii=False)}

CHAT CONTEXT:
Global average: {global_chat_avg:.2f} messages/sec
Start chat: {json.dumps(head_chat, ensure_ascii=False)}
End chat: {json.dumps(tail_chat, ensure_ascii=False)}

Respond with ONLY a single minified JSON object, no prose, no markdown, no code fences:
{{"start_time": <int>, "end_time": <int>}}
"""
    
    try:
        # Call unified AI client for analysis
        response = call_llm(prompt, max_tokens=200, temperature=0.1, request_tag="transcript_boundary")
        result = response.strip()
        
        # Debug: show what the AI returned
        print(f"🤖 AI Response: {result[:200]}...")
        
        # Try to extract JSON robustly from arbitrary model output
        boundaries = _extract_boundaries_from_text(result)
        start_time = int(boundaries['start_time'])
        end_time = int(boundaries['end_time'])

        # Clamp to VOD range and sanity check
        start_time = max(vod_start, min(start_time, vod_end))
        end_time = max(vod_start, min(end_time, vod_end))
        if end_time <= start_time:
            raise ValueError("Invalid AI-proposed boundaries (end <= start)")
        
        print(f"[DETECT] OpenRouter detected real content: {start_time}s - {end_time}s")
        
        # Filter segments to only include real content
        filtered_segments = []
        for segment in ai_data:
            if start_time <= segment['start_time'] <= end_time:
                filtered_segments.append(segment)
        
        print(f"[FILTER] Filtered from {len(ai_data)} to {len(filtered_segments)} segments")
        
        return start_time, end_time, filtered_segments
        
    except Exception as e:
        print(f"X Transcript boundary AI failed: {e}")
        raise


def save_filtered_ai_data(ai_data: List[Dict], vod_id: str, start_time: int, end_time: int) -> Path:
    """Save filtered AI data"""
    ai_data_dir = config.get_ai_data_dir(vod_id)
    filtered_path = ai_data_dir / f"{vod_id}_filtered_ai_data.json"
    
    data = {
        'vod_id': vod_id,
        'segments': ai_data,
        'metadata': {
            'original_segments': len(ai_data),
            'filtered_start_time': start_time,
            'filtered_end_time': end_time,
            'filtered_duration': end_time - start_time,
            'source': 'transcript_boundary_filter'
        }
    }
    
    # Use unified storage to save both locally and to S3
    storage = StorageManager()
    success = storage.save_json_with_cloud_backup(
        local_path=str(filtered_path),
        data=data,
        s3_key=f"ai_data/{vod_id}/{vod_id}_filtered_ai_data.json"
    )
    
    if not success:
        raise RuntimeError(f"Failed to save filtered AI data to {filtered_path}")
    
    return filtered_path


def filter_and_save_chat_data(vod_id: str, start_time: int, end_time: int) -> bool:
    """Filter chat data to match filtered transcript boundaries"""
    
    chat_dir = config.get_chat_dir(vod_id)
    chat_path = chat_dir / f"{vod_id}_chat.json"
    filtered_chat_path = chat_dir / f"{vod_id}_filtered_chat.json"
    
    if not chat_path.exists():
        print(f"[WARN] Original chat file not found: {chat_path}")
        return False
    
    try:
        # Load original chat data
        with open(chat_path, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
        
        # Handle different chat formats
        if 'comments' in chat_data:
            messages = chat_data['comments']
        else:
            messages = chat_data
        
        # Filter messages within time boundaries
        filtered_messages = []
        for message in messages:
            # Handle different timestamp formats
            if 'content_offset_seconds' in message:
                msg_time = message['content_offset_seconds']
            elif 'timestamp' in message:
                msg_time = message['timestamp']
            else:
                continue
            
            # Keep messages within filtered boundaries
            if start_time <= msg_time <= end_time:
                filtered_messages.append(message)
        
        # Create filtered chat data structure
        if 'comments' in chat_data:
            filtered_data = chat_data.copy()
            filtered_data['comments'] = filtered_messages
            filtered_data['metadata'] = {
                'original_message_count': len(messages),
                'filtered_message_count': len(filtered_messages),
                'filtered_start_time': start_time,
                'filtered_end_time': end_time,
                'source': 'transcript_boundary_filter'
            }
        else:
            filtered_data = {
                'messages': filtered_messages,
                'metadata': {
                    'original_message_count': len(messages),
                    'filtered_message_count': len(filtered_messages),
                    'filtered_start_time': start_time,
                    'filtered_end_time': end_time,
                    'source': 'transcript_boundary_filter'
                }
            }
        
        # Save filtered chat data
        with open(filtered_chat_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Filtered chat: {len(messages)} → {len(filtered_messages)} messages")
        print(f"[SAVE] Saved filtered chat: {filtered_chat_path}")
        
        return True
        
    except Exception as e:
        print(f"[WARN] Failed to filter chat data: {e}")
        return False


def main():
    """Main function"""
    # Set UTF-8 encoding for console output
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    if len(sys.argv) < 2:
        print("Usage: python filter_transcript_boundaries.py <vod_id>")
        print("Example: python filter_transcript_boundaries.py 2512564693")
        sys.exit(1)
    
    vod_id = sys.argv[1]
    
    # Early exit if filtered AI data already exists
    ai_data_dir = config.get_ai_data_dir(vod_id)
    existing_filtered = ai_data_dir / f"{vod_id}_filtered_ai_data.json"
    if existing_filtered.exists():
        print(f"[SKIP] Filtered AI data already exists: {existing_filtered}")
        print("[SKIP] No work done.")
        return
    
    print(f"[FILTER] Filtering transcript boundaries for VOD: {vod_id}")
    
    try:
        # Load AI data
        print("[LOAD] Loading AI data...")
        ai_data = load_ai_data(vod_id)
        print(f"[OK] Loaded {len(ai_data)} segments")
        
        # Check if we have enough data to filter
        if len(ai_data) < 3:
            print("[WARN] Insufficient segments for filtering (need at least 3)")
            print("[INFO] Using original data without filtering")
            if ai_data:
                start_time = ai_data[0]['start_time']
                end_time = ai_data[-1]['end_time']
                filtered_data = ai_data
            else:
                start_time = 0
                end_time = 0
                filtered_data = []
        else:
            # Filter by transcript content
            print("[ANALYZE] Analyzing transcript boundaries...")
            # Provide VOD id to downstream loaders via env for chat lookup
            os.environ['CURRENT_VOD_ID'] = vod_id
            start_time, end_time, filtered_data = filter_by_transcript_content(ai_data)
        
        # Save filtered data
        print("[SAVE] Saving filtered AI data...")
        output_path = save_filtered_ai_data(filtered_data, vod_id, start_time, end_time)
        
        # Filter and save chat data to match transcript boundaries
        print("[SAVE] Filtering chat data to match transcript boundaries...")
        chat_filtered = filter_and_save_chat_data(vod_id, start_time, end_time)
        
        print("\n[SUCCESS] Transcript and chat filtering complete!")
        print(f"[FILE] AI Data: {output_path}")
        if chat_filtered:
            print(f"[FILE] Chat Data: {config.get_chat_dir(vod_id) / f'{vod_id}_filtered_chat.json'}")
        print(f"[TIME] Real content: {start_time}s - {end_time}s ({(end_time-start_time)/60:.1f} minutes)")
        print(f"[STATS] Segments: {len(filtered_data)} (filtered from {len(ai_data)})")
        
        # Show what was removed (only if we actually filtered)
        if ai_data and len(ai_data) >= 3:
            removed_start = ai_data[0]['start_time']
            removed_end = ai_data[-1]['end_time']
            print(f"[REMOVED] Removed: {removed_start}s - {start_time}s (intro) and {end_time}s - {removed_end}s (outro)")
        
    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 