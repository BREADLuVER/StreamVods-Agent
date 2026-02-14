#!/usr/bin/env python3
"""
Manifest loader and range normalization for Director's Cut.

- Reads data/vector_stores/<vod_id>/enhanced_director_cut_manifest.json
- Normalizes ranges by merging overlaps, merging small gaps, and fixing micro gaps
"""

from __future__ import annotations

import json
from pathlib import Path
import os
from typing import Any, Dict, List, Optional


def _fix_timing_boundaries(ranges: List[Dict[str, Any]], gap_epsilon_s: float = 0.25) -> List[Dict[str, Any]]:
    if not ranges or len(ranges) <= 1:
        return ranges
    fixed: List[Dict[str, Any]] = []
    for i, current in enumerate(ranges):
        cur = dict(current)
        if i > 0:
            prev = fixed[-1]
            prev_end = float(prev["end"])
            cur_start = float(cur["start"])
            if cur_start > prev_end:
                gap = cur_start - prev_end
                if gap <= gap_epsilon_s:
                    prev["end"] = cur_start
                    prev["duration"] = float(prev["end"]) - float(prev["start"])
        fixed.append(cur)
    return fixed


def _merge_overlaps(ranges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not ranges or len(ranges) <= 1:
        return ranges
    merged: List[Dict[str, Any]] = []
    for i, r in enumerate(ranges):
        cur = dict(r)
        if i > 0:
            prev = merged[-1]
            if float(cur["start"]) < float(prev["end"]):
                prev["end"] = max(float(prev["end"]), float(cur["end"]))
                prev["duration"] = float(prev["end"]) - float(prev["start"])
                # best-effort carry keys
                if prev.get("summary") and cur.get("summary"):
                    prev["summary"] = f"{prev['summary']} → {cur['summary']}"
                if prev.get("burst_ids") and cur.get("burst_ids"):
                    prev["burst_ids"] = list(prev["burst_ids"]) + list(cur["burst_ids"])
                continue
        merged.append(cur)
    return merged


def _merge_close_gaps(ranges: List[Dict[str, Any]], max_gap_s: float = 15.0) -> List[Dict[str, Any]]:
    if not ranges or len(ranges) <= 1:
        return ranges
    # Sort robustly by (chapter_id, start)
    sorted_ranges = sorted(
        ranges,
        key=lambda r: (str(r.get("chapter_id") or ""), float(r.get("start") or 0.0)),
    )
    merged: List[Dict[str, Any]] = []
    for cur in sorted_ranges:
        if not merged:
            merged.append(dict(cur))
            continue
        prev = merged[-1]
        same_chapter = str(prev.get("chapter_id") or "") == str(cur.get("chapter_id") or "")
        gap = float(cur["start"]) - float(prev["end"])
        if same_chapter and 0.0 <= gap <= max_gap_s:
            prev["end"] = max(float(prev["end"]), float(cur["end"]))
            prev["duration"] = float(prev["end"]) - float(prev["start"])
            if prev.get("summary") and cur.get("summary"):
                prev["summary"] = f"{prev['summary']} → {cur['summary']}"
            if prev.get("burst_ids") and cur.get("burst_ids"):
                prev["burst_ids"] = list(prev["burst_ids"]) + list(cur["burst_ids"])
            if not prev.get("anchor_burst_id") and cur.get("anchor_burst_id"):
                prev["anchor_burst_id"] = cur["anchor_burst_id"]
            continue
        merged.append(dict(cur))
    return merged


def load_manifest(vod_id: str) -> Dict[str, Any]:
    """Load and normalize the enhanced director's cut manifest for a VOD."""
    base = Path("data/vector_stores") / vod_id
    path = base / "enhanced_director_cut_manifest.json"
    if not path.exists():
        # Best-effort S3 fallback
        try:
            from storage import StorageManager
            s3_bucket = os.getenv('S3_BUCKET', 'streamsniped-dev-videos')
            s3_uri = f"s3://{s3_bucket}/vector_stores/{vod_id}/enhanced_director_cut_manifest.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            StorageManager().download_file(s3_uri, str(path))
        except Exception:
            pass
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    ranges = data.get("ranges") or []
    # Normalize order and numeric types
    normalized: List[Dict[str, Any]] = []
    for r in ranges:
        try:
            start = float(r.get("start"))
            end = float(r.get("end"))
            if end <= start:
                continue
            item = dict(r)
            item["start"] = start
            item["end"] = end
            item["duration"] = float(item.get("duration") or (end - start))
            normalized.append(item)
        except Exception:
            continue
    normalized = sorted(normalized, key=lambda x: (str(x.get("chapter_id") or ""), float(x.get("start") or 0.0)))
    # Merge overlaps, merge close gaps, fix micro gaps
    merged = _merge_overlaps(normalized)
    merged_close = _merge_close_gaps(merged, max_gap_s=15.0)
    fixed = _fix_timing_boundaries(merged_close, gap_epsilon_s=0.25)
    # Update totals
    total_seconds = sum(float(x["end"]) - float(x["start"]) for x in fixed)
    def _hms(sec: float) -> str:
        s = int(round(sec))
        h = s // 3600
        m = (s % 3600) // 60
        ss = s % 60
        return f"{h:02d}:{m:02d}:{ss:02d}"
    data["ranges"] = fixed
    data["total_duration_seconds"] = round(total_seconds, 3)
    data["total_duration_minutes"] = round(total_seconds / 60.0, 2)
    data["total_duration_hms"] = _hms(total_seconds)
    return data


