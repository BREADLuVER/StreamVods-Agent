#!/usr/bin/env python3
"""
Transcript loader utility for title/timestamp generation.

Loads transcript data from _filtered_ai_data.json for specific time ranges.
Does NOT include chat messages - only the spoken transcript text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

# Try to import config for path resolution
try:
    from src.config import config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False


def _get_ai_data_path(vod_id: str) -> Path:
    """Get the path to the filtered AI data file."""
    if HAS_CONFIG:
        return config.get_ai_data_dir(vod_id) / f"{vod_id}_filtered_ai_data.json"
    return Path(f"data/ai_data/{vod_id}/{vod_id}_filtered_ai_data.json")


def _hms_to_seconds(hms: str) -> float:
    """Convert HH:MM:SS or H:MM:SS to seconds."""
    parts = hms.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(hms)


def load_transcript_for_range(
    vod_id: str,
    start_seconds: float,
    end_seconds: float,
    padding_seconds: float = 0.0,
) -> str:
    """Load transcript text from filtered_ai_data.json for the given time range.
    
    Args:
        vod_id: The VOD ID
        start_seconds: Start time in seconds
        end_seconds: End time in seconds
        padding_seconds: Optional padding to extend the range on both sides
        
    Returns:
        Concatenated transcript text for segments in the time range.
        Returns empty string if file doesn't exist or no segments found.
    """
    ai_data_path = _get_ai_data_path(vod_id)
    
    if not ai_data_path.exists():
        return ""
    
    try:
        data = json.loads(ai_data_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    
    # Handle both formats: {"segments": [...]} or just [...]
    segments = data.get("segments", data) if isinstance(data, dict) else data
    if not isinstance(segments, list):
        return ""
    
    # Apply padding
    range_start = start_seconds - padding_seconds
    range_end = end_seconds + padding_seconds
    
    # Filter segments by time range
    transcripts: List[str] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        
        seg_start = seg.get("start_time", 0)
        seg_end = seg.get("end_time", seg_start)
        
        # Include segment if it overlaps with the range
        if seg_start < range_end and seg_end > range_start:
            transcript = str(seg.get("transcript", "")).strip()
            if transcript:
                transcripts.append(transcript)
    
    return " ".join(transcripts)


def load_transcript_for_hms_range(
    vod_id: str,
    start_hms: str,
    end_hms: str,
    padding_seconds: float = 0.0,
) -> str:
    """Load transcript for a time range specified in HH:MM:SS format.
    
    Args:
        vod_id: The VOD ID
        start_hms: Start time as "HH:MM:SS" or "H:MM:SS"
        end_hms: End time as "HH:MM:SS" or "H:MM:SS"
        padding_seconds: Optional padding to extend the range
        
    Returns:
        Concatenated transcript text for segments in the time range.
    """
    start_s = _hms_to_seconds(start_hms)
    end_s = _hms_to_seconds(end_hms)
    return load_transcript_for_range(vod_id, start_s, end_s, padding_seconds)


def load_segments_for_range(
    vod_id: str,
    start_seconds: float,
    end_seconds: float,
    padding_seconds: float = 0.0,
) -> List[dict]:
    """Load raw segment data for the given time range.
    
    Returns list of segment dicts with start_time, end_time, transcript, etc.
    Useful when you need more than just the transcript text.
    """
    ai_data_path = _get_ai_data_path(vod_id)
    
    if not ai_data_path.exists():
        return []
    
    try:
        data = json.loads(ai_data_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    
    segments = data.get("segments", data) if isinstance(data, dict) else data
    if not isinstance(segments, list):
        return []
    
    range_start = start_seconds - padding_seconds
    range_end = end_seconds + padding_seconds
    
    result: List[dict] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        
        seg_start = seg.get("start_time", 0)
        seg_end = seg.get("end_time", seg_start)
        
        if seg_start < range_end and seg_end > range_start:
            result.append(seg)
    
    return result

