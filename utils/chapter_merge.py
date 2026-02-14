#!/usr/bin/env python3
"""
Deterministic chapter merging utilities.

- Merge chapters shorter than a threshold into the longer neighbor
- Preserve dominant chapter identity and original category casing
"""

from typing import List, Dict, Optional


def _format_hms(total_seconds: int) -> str:
    """Format seconds into H:MM:SS if hours>0 else M:SS."""
    if total_seconds < 0:
        total_seconds = 0
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def _safe_name_from_category(category: str) -> str:
    """Generate a normalized file-safe name from category (fallback)."""
    if not category:
        return "unknown"
    safe = category.lower()
    for ch in [" ", "+", "&", "-", "/", "\\", "(", ")", "[", "]", "{", "}", ":", ";", ",", ".", "!", "?", "@", "#", "$", "%", "^", "*", "=", "|", '"', "'", "`", "~"]:
        safe = safe.replace(ch, "_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "unknown"


def _pick_dominant(left: Dict, right: Dict) -> Dict:
    left_duration = int(left.get("duration", 0) or 0)
    right_duration = int(right.get("duration", 0) or 0)
    return right if right_duration >= left_duration else left


def _merge_pair(left: Dict, right: Dict) -> Dict:
    """Merge two adjacent chapters into one, preserving dominant identity."""
    if not left:
        return right
    if not right:
        return left

    merged: Dict = left.copy()

    # New temporal bounds
    start_time = int(left.get("start_time", 0) or 0)
    end_time = int(right.get("end_time", left.get("end_time", 0)) or 0)
    merged["start_time"] = start_time
    merged["end_time"] = end_time
    merged["duration"] = max(0, end_time - start_time)

    # Dominant chapter determines identity and labels
    dominant = _pick_dominant(left, right)

    # Preserve dominant ID
    if dominant.get("id"):
        merged["id"] = dominant["id"]

    # Category casing from original_category when available; else category
    dom_original_category: Optional[str] = dominant.get("original_category")
    dom_category: Optional[str] = dominant.get("category")
    merged["category"] = (dom_original_category or dom_category or merged.get("category") or "unknown")

    # Keep normalized file_safe_name (fallback derive if missing)
    if dominant.get("file_safe_name"):
        merged["file_safe_name"] = dominant["file_safe_name"]
    else:
        merged["file_safe_name"] = _safe_name_from_category(merged["category"])  # fallback

    # Propagate type if present on dominant
    if dominant.get("type"):
        merged["type"] = dominant["type"]

    # Update timestamp strings if present
    if left.get("start_timestamp"):
        merged["start_timestamp"] = left.get("start_timestamp")
    if end_time is not None:
        merged["end_timestamp"] = _format_hms(end_time)
        merged["length"] = _format_hms(merged["duration"])

    # Remove exclusion flag if any
    if "excluded" in merged:
        merged.pop("excluded", None)

    return merged


def merge_short_chapters(chapters: List[Dict], min_minutes: int = 40) -> List[Dict]:
    """
    Iteratively merge any chapter shorter than min_minutes into the longer neighbor.

    - End-of-list short chapter merges into previous
    - Tie-breaker prefers the next chapter when available
    - Dominant neighbor identity and labeling are preserved
    """
    if not chapters:
        return []

    working: List[Dict] = list(chapters)
    changed = True
    threshold_seconds = int(min_minutes * 60)

    while changed and len(working) > 1:
        changed = False
        i = 0
        while i < len(working):
            current = working[i]
            cur_duration = int(current.get("duration", 0) or 0)
            if cur_duration < threshold_seconds:
                prev_idx = i - 1
                next_idx = i + 1
                prev_ch = working[prev_idx] if prev_idx >= 0 else None
                next_ch = working[next_idx] if next_idx < len(working) else None

                prev_dur = int(prev_ch.get("duration", 0) or 0) if prev_ch else -1
                next_dur = int(next_ch.get("duration", 0) or 0) if next_ch else -1

                if next_ch is None and prev_ch is None:
                    i += 1
                    continue

                if next_dur >= prev_dur and next_ch is not None:
                    merged = _merge_pair(current, next_ch)
                    working[i:i+2] = [merged]
                    changed = True
                    i = max(i - 1, 0)
                    continue
                elif prev_ch is not None:
                    merged = _merge_pair(prev_ch, current)
                    working[prev_idx:i+1] = [merged]
                    changed = True
                    i = max(prev_idx - 1, 0)
                    continue
            i += 1

    return working


__all__ = ["merge_short_chapters", "_format_hms"]


