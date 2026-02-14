#!/usr/bin/env python3
"""
Chapter metadata attachment and exclusion handling.

Maps windows to chapters and handles excluded content.
"""

import json
from typing import List, Optional
from dataclasses import dataclass
import logging

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logger = logging.getLogger(__name__)


@dataclass
class ChapterInfo:
    """Chapter information from chapters.json"""
    id: str
    category: str
    start_time: float
    end_time: float
    duration: float
    excluded: bool = False


def load_chapters(vod_id: str) -> List[ChapterInfo]:
    """Load chapter information from chapters.json file."""
    try:
        # Prefer unmerged for metadata attachment; fallback to merged
        chapters_unmerged_path = f"data/ai_data/{vod_id}/{vod_id}_chapters_unmerged.json"
        try_paths = [chapters_unmerged_path, f"data/ai_data/{vod_id}/{vod_id}_chapters.json"]
        data = None
        for path in try_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    break
            except FileNotFoundError:
                continue
        if data is None:
            raise FileNotFoundError
        
        chapters = []
        for chapter_data in data.get('chapters', []):
            chapter = ChapterInfo(
                id=chapter_data.get('id', ''),
                category=chapter_data.get('category', ''),
                start_time=chapter_data.get('start_time', 0),
                end_time=chapter_data.get('end_time', 0),
                duration=chapter_data.get('duration', 0),
                excluded=chapter_data.get('excluded', False)
            )
            chapters.append(chapter)
        
        logger.info(f"Loaded {len(chapters)} chapters for VOD {vod_id}")
        return chapters
        
    except FileNotFoundError:
        logger.info(f"No chapters file found for VOD {vod_id}")
        return []
    except Exception as e:
        logger.error(f"Failed to load chapters for VOD {vod_id}: {e}")
        return []


def load_chapters_merged(vod_id: str) -> List[ChapterInfo]:
    """Load merged chapter information from chapters.json only.

    Use when downstream grouping must align with merged chapter IDs, e.g.,
    rate baselines and video creation flows.
    """
    try:
        chapters_path = f"data/ai_data/{vod_id}/{vod_id}_chapters.json"
        with open(chapters_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        chapters: List[ChapterInfo] = []
        for chapter_data in data.get('chapters', []):
            chapters.append(
                ChapterInfo(
                    id=chapter_data.get('id', ''),
                    category=chapter_data.get('category', ''),
                    start_time=chapter_data.get('start_time', 0),
                    end_time=chapter_data.get('end_time', 0),
                    duration=chapter_data.get('duration', 0),
                    excluded=chapter_data.get('excluded', False),
                )
            )
        logger.info(f"Loaded {len(chapters)} merged chapters for VOD {vod_id}")
        return chapters
    except FileNotFoundError:
        logger.info(f"No merged chapters file found for VOD {vod_id}")
        return []
    except Exception as e:
        logger.error(f"Failed to load merged chapters for VOD {vod_id}: {e}")
        return []


def find_enclosing_chapter(window_start: float, window_end: float, 
                          chapters: List[ChapterInfo]) -> Optional[ChapterInfo]:
    """
    Find the chapter that encloses a window's time range.
    
    Args:
        window_start: Window start time
        window_end: Window end time
        chapters: List of available chapters
    
    Returns:
        ChapterInfo if found, None otherwise
    """
    for chapter in chapters:
        # Check if window is fully contained within chapter
        if (chapter.start_time <= window_start and 
            window_end <= chapter.end_time):
            return chapter
        
        # Check for partial overlap (window spans chapter boundary)
        if (window_start < chapter.end_time and 
            window_end > chapter.start_time):
            # Window overlaps with chapter, return the chapter
            return chapter
    
    return None


def attach_chapter_metadata(windows: List, chapters: List[ChapterInfo]) -> List:
    """
    Attach chapter metadata to windows and filter out excluded content.
    
    Args:
        windows: List of Window objects
        chapters: List of ChapterInfo objects
    
    Returns:
        List of windows with chapter metadata attached, excluding excluded chapters
    """
    if not chapters:
        # No chapters available, assign default metadata
        for window in windows:
            window.chapter_id = None
            window.category = None
            window.excluded = False
        return windows
    
    filtered_windows = []
    
    for window in windows:
        # Find enclosing chapter
        chapter = find_enclosing_chapter(window.start_time, window.end_time, chapters)
        
        if chapter:
            window.chapter_id = chapter.id
            window.category = chapter.category
            window.excluded = chapter.excluded
        else:
            # No chapter found, assign defaults
            window.chapter_id = None
            window.category = None
            window.excluded = False
        
        # Include ALL windows - filtering happens at query time, not build time
        filtered_windows.append(window)
    
    logger.info(f"Attached chapter metadata to {len(filtered_windows)} windows")
    return filtered_windows


def map_category_to_mode(category: str) -> str:
    """
    Map chapter category to mode using Twitch's category as truth.
    
    Args:
        category: Chapter category string from Twitch
    
    Returns:
        Mode string: 'jc' if "Just Chatting", 'game' for everything else
    """
    if not category:
        return 'unknown'
    
    # Use Twitch's category as truth - anything with "just_chatting" or "just chatting" is JC
    category_lower = category.strip().lower()
    return 'jc' if ('just_chatting' in category_lower or 'just chatting' in category_lower) else 'game'


def derive_mode_from_talkiness(windows: List, fallback_mode: str = 'unknown') -> str:
    """
    Derive mode from talkiness analysis when category mapping fails.
    
    Args:
        windows: List of windows to analyze
        fallback_mode: Default mode if analysis fails
    
    Returns:
        Mode string: 'jc' or 'game'
    """
    if not windows:
        return fallback_mode
    
    # Calculate words per second for each window
    wps_scores = []
    for window in windows:
        # Combine all transcript text in window
        transcript_text = " ".join(
            seg.get('transcript', '') for seg in window.segments
        )
        word_count = len(transcript_text.split())
        wps = word_count / window.duration if window.duration > 0 else 0
        wps_scores.append(wps)
    
    if not wps_scores:
        return fallback_mode
    
    # Calculate median and threshold
    if HAS_NUMPY:
        median_wps = np.median(wps_scores)
    else:
        sorted_wps = sorted(wps_scores)
        median_wps = sorted_wps[len(sorted_wps) // 2]
    
    # Use 70th percentile as threshold (or knee point if available)
    if HAS_NUMPY:
        threshold_wps = np.percentile(wps_scores, 70)
    else:
        # Fallback without numpy
        sorted_wps = sorted(wps_scores)
        threshold_wps = sorted_wps[int(len(sorted_wps) * 0.7)]
    
    logger.info(f"Talkiness analysis: median_wps={median_wps:.2f}, "
                f"threshold_wps={threshold_wps:.2f}")
    
    # High talkiness = Just Chatting, Low talkiness = Gaming
    if median_wps >= threshold_wps:
        return 'jc'
    else:
        return 'game'


def assign_mode_to_windows(windows: List) -> List:
    """
    Assign mode to windows based on chapter category (Twitch's truth).
    
    Args:
        windows: List of windows with chapter metadata
    
    Returns:
        Windows with mode assigned
    """
    for window in windows:
        # Use Twitch's category as truth
        window.mode = map_category_to_mode(window.category)
    
    # Log mode distribution
    mode_counts = {}
    for window in windows:
        mode_counts[window.mode] = mode_counts.get(window.mode, 0) + 1
    
    logger.info(f"Mode distribution (from Twitch categories): {mode_counts}")
    return windows
