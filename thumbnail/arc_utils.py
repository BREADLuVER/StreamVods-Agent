#!/usr/bin/env python3
"""
Shared utilities for arc thumbnail generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional


def is_just_chatting_arc(vod_id: str, arc_manifest: Dict) -> bool:
    """
    Check if an arc is a "Just Chatting" or "IRL" arc.
    
    Logic:
    1. All ranges must have the same chapter_id
    2. That chapter's file_safe_name must be "just_chatting" or "irl"
    
    Returns:
        True if arc is just chatting/irl, False otherwise (including mixed chapters)
    """
    try:
        ranges = arc_manifest.get("ranges", [])
        if not ranges:
            return False
        
        # Check if all ranges have same chapter_id
        chapter_ids = set()
        for r in ranges:
            ch_id = r.get("chapter_id")
            if ch_id:
                chapter_ids.add(str(ch_id))
        
        if len(chapter_ids) != 1:
            # Mixed chapters or no chapter_id → fall back to crops
            return False
        
        chapter_id = chapter_ids.pop()
        
        # Load chapters file
        chapters_data = _load_chapters(vod_id)
        if not chapters_data:
            return False
        
        # Find the chapter
        for ch in chapters_data.get("chapters", []):
            if str(ch.get("id", "")) == chapter_id:
                file_safe_name = str(ch.get("file_safe_name", "")).lower()
                return file_safe_name in ["just_chatting", "irl"]
        
        return False
    except Exception:
        return False


def _load_chapters(vod_id: str) -> Optional[Dict]:
    """Load chapters JSON for a VOD."""
    # Try both locations
    for name in [f"{vod_id}_chapters.json", f"{vod_id}_chapters_unmerged.json"]:
        path = Path(f"data/ai_data/{vod_id}/{name}")
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
    return None


def get_chapter_name(vod_id: str, chapter_id: str) -> str:
    """Get human-readable chapter name from chapter_id."""
    try:
        chapters_data = _load_chapters(vod_id)
        if not chapters_data:
            return chapter_id
        
        for ch in chapters_data.get("chapters", []):
            if str(ch.get("id", "")) == chapter_id:
                return str(ch.get("file_safe_name", chapter_id)).replace("_", " ").title()
        
        return chapter_id.replace("_", " ").title()
    except Exception:
        return chapter_id

