#!/usr/bin/env python3
"""
Create story arcs from the enhanced Director's Cut manifest.

This CLI clusters the manifest's merged ranges into smaller, coherent arcs
of 10–30 minutes using semantic similarity (vector store), topic alignment,
and time proximity. It writes per-arc manifests and an arcs index. Optionally,
it can generate per-arc timestamps.

Usage:
  python -m story_archs.create_story_archs <vod_id> \
      --target-min 600 --target-max 1800 --sim-threshold 0.45 \
      [--time-tau 120.0] [--topic-bonus 0.1] \
      [--timestamps] [--no-llm]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------- Data structures -----------------------------

from dataclasses import dataclass


@dataclass
class Range:
    start: float
    end: float
    duration: float
    topic_key: str
    anchor_burst_id: Optional[str]
    burst_ids: List[str]
    raw: Dict[str, Any]


# ----------------------------- Helpers -----------------------------

def _format_hms(sec: float) -> str:
    s = int(round(max(0.0, float(sec))))
    h = s // 3600
    m = (s % 3600) // 60
    s2 = s % 60
    return f"{h:02d}:{m:02d}:{s2:02d}"


def _load_enhanced_manifest(vod_id: str, manifest_path: Optional[str] = None) -> Tuple[List[Range], Dict[str, Any], Path]:
    if manifest_path:
        path = Path(manifest_path)
    else:
        path = Path(f"data/vector_stores/{vod_id}/enhanced_director_cut_manifest.json")
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_ranges = data.get("ranges") or []
    ranges: List[Range] = []
    for item in raw_ranges:
        try:
            s = float(item.get("start", 0.0))
            e = float(item.get("end", 0.0))
            if e <= s:
                continue
            dur = float(item.get("duration") or (e - s))
            tk = str(item.get("topic_key") or "")
            abid = str(item.get("anchor_burst_id") or "").strip() or None
            bids = item.get("burst_ids") if isinstance(item.get("burst_ids"), list) else []
            ranges.append(
                Range(
                    start=s,
                    end=e,
                    duration=dur,
                    topic_key=tk,
                    anchor_burst_id=abid,
                    burst_ids=[str(b) for b in bids if isinstance(b, str)],
                    raw=dict(item),
                )
            )
        except Exception:
            continue
    if not ranges:
        raise RuntimeError("No valid ranges in manifest")
    ranges.sort(key=lambda r: r.start)
    return ranges, data, path


def _id_for_range(r: Range) -> Optional[str]:
    if r.anchor_burst_id:
        return r.anchor_burst_id
    if r.burst_ids:
        return r.burst_ids[0]
    return None


def _load_retriever(vod_id: str):
    try:
        from rag.retrieval import load_retriever  # type: ignore
        return load_retriever(vod_id)
    except Exception:
        class _Dummy:
            have_index = False
            def sim(self, *_a, **_k):
                return -1.0
        return _Dummy()


def _similarity(a: Range, b: Range, retriever, topic_bonus: float, time_tau: float) -> float:
    sem = -1.0
    if retriever and getattr(retriever, "have_index", False):
        a_id = _id_for_range(a)
        b_id = _id_for_range(b)
        if a_id and b_id:
            try:
                sem = float(retriever.sim(a_id, b_id))
            except Exception:
                sem = -1.0
    sem01 = (sem + 1.0) / 2.0 if sem >= -1.0 else 0.0
    t_bonus = topic_bonus if (a.topic_key and b.topic_key and a.topic_key == b.topic_key) else 0.0
    dt = max(0.0, b.start - a.end)
    try:
        import math
        time_p = math.exp(-dt / max(1.0, float(time_tau)))
    except Exception:
        time_p = 0.0
    return 0.7 * sem01 + 0.2 * time_p + 0.1 * t_bonus


def _cluster_ranges(
    ranges: List[Range],
    retriever,
    target_min: float,
    target_max: float,
    sim_threshold: float,
    time_tau: float,
    topic_bonus: float,
) -> List[List[Range]]:
    clusters: List[List[Range]] = []
    current: List[Range] = []
    cur_dur = 0.0
    for r in ranges:
        if not current:
            current = [r]
            cur_dur = r.duration
            continue
        if cur_dur < target_min:
            current.append(r)
            cur_dur += r.duration
            continue
        if cur_dur >= target_max:
            clusters.append(current)
            current = [r]
            cur_dur = r.duration
            continue
        score = _similarity(current[-1], r, retriever, topic_bonus=topic_bonus, time_tau=time_tau)
        if score >= sim_threshold:
            current.append(r)
            cur_dur += r.duration
        else:
            clusters.append(current)
            current = [r]
            cur_dur = r.duration
    if current:
        clusters.append(current)
    return clusters


def _write_arc_manifests(
    vod_id: str,
    clusters: List[List[Range]],
    base_manifest: Dict[str, Any],
    out_root: Optional[Path] = None,
) -> Tuple[Path, List[Path], Path]:
    out_dir = out_root or Path(f"data/vector_stores/{vod_id}/arcs")
    out_dir.mkdir(parents=True, exist_ok=True)

    arc_paths: List[Path] = []
    index: List[Dict[str, Any]] = []

    for idx, block in enumerate(clusters, 1):
        total_dur = sum(max(0.0, r.duration) for r in block)
        start_abs = block[0].start
        end_abs = block[-1].end
        span_seconds = max(0.0, end_abs - start_abs)
        contiguity_ratio = (total_dur / span_seconds) if span_seconds > 0 else 0.0
        # compute longest internal gap and gap fraction
        gaps: List[float] = []
        for i in range(len(block) - 1):
            gaps.append(max(0.0, block[i + 1].start - block[i].end))
        max_internal_gap = max(gaps) if gaps else 0.0
        gap_fraction = ((span_seconds - total_dur) / span_seconds) if span_seconds > 0 else 0.0
        manifest_obj: Dict[str, Any] = {
            "vod_id": vod_id,
            "from_enhanced_manifest": True,
            "arc_index": idx,
            "total_ranges": len(block),
            "total_duration_seconds": round(total_dur, 3),
            "total_duration_hms": _format_hms(total_dur),
            "start_abs": round(start_abs, 3),
            "end_abs": round(end_abs, 3),
            "start_hms": _format_hms(start_abs),
            "end_hms": _format_hms(end_abs),
            "span_seconds": round(span_seconds, 3),
            "contiguity_ratio": round(contiguity_ratio, 3),
            "max_internal_gap": round(max_internal_gap, 3),
            "gap_fraction": round(gap_fraction, 3),
            "ranges": [r.raw for r in block],
        }
        arc_path = out_dir / f"arc_{idx:03d}_manifest.json"
        arc_path.write_text(json.dumps(manifest_obj, indent=2), encoding="utf-8")
        arc_paths.append(arc_path)

        index.append(
            {
                "arc_index": idx,
                "manifest": str(arc_path),
                "total_duration_seconds": manifest_obj["total_duration_seconds"],
                "total_duration_hms": manifest_obj["total_duration_hms"],
                "total_ranges": manifest_obj["total_ranges"],
                "start_abs": manifest_obj["start_abs"],
                "end_abs": manifest_obj["end_abs"],
                "span_seconds": manifest_obj["span_seconds"],
                "contiguity_ratio": manifest_obj["contiguity_ratio"],
                "max_internal_gap": manifest_obj["max_internal_gap"],
                "gap_fraction": manifest_obj["gap_fraction"],
                "start_hms": manifest_obj["start_hms"],
                "end_hms": manifest_obj["end_hms"],
            }
        )

    index_obj = {
        "vod_id": vod_id,
        "num_arcs": len(clusters),
        "arcs": index,
        "source": "enhanced_director_cut_manifest.json",
    }
    index_path = out_dir / "arcs_index.json"
    index_path.write_text(json.dumps(index_obj, indent=2), encoding="utf-8")
    # Write a simple VOD time summary for quick manual verification
    try:
        summary_path = out_dir / "arcs_vod_times.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            for item in index:
                idx = int(item["arc_index"]) if isinstance(item.get("arc_index"), int) else item["arc_index"]
                f.write(f"Arc {idx:03d}: {item['start_hms']} -> {item['end_hms']}  (span={item['span_seconds']}s, contig={item['contiguity_ratio']})\n")
        print(f"🕒 Wrote VOD time summary: {summary_path}")
    except Exception:
        pass

    return out_dir, arc_paths, index_path


def _run_render_for_arc(*_args, **_kwargs) -> bool:
    # Rendering via create_cloud_video is intentionally disabled until it supports
    # a manifest override flag. This placeholder always returns False.
    return False


def _run_timestamps_for_arc(vod_id: str, manifest_path: Path, no_llm: bool, max_chars: int) -> Optional[Path]:
    out_dir = Path(f"data/ai_data/{vod_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Generate a per-arc timestamps file named after the manifest
    arc_name = manifest_path.stem.replace("_manifest", "")
    out_path = out_dir / f"{vod_id}_{arc_name}_timestamps.txt"
    # Call the script path directly because the module name contains a dash.
    script_path = Path("processing-scripts") / "generate_youtube_timestamps.py"
    cmd = [
        "python", str(script_path), str(vod_id),
        "--manifest", str(manifest_path),
        "--vod-time",
        "--merge-gap", os.getenv("TS_MERGE_GAP", "15"),
        "--max-chars", str(max(8, int(max_chars))),
    ]
    if no_llm:
        cmd.append("--no-llm")
    try:
        res = subprocess.run(cmd, capture_output=False, check=False)
        return out_path if res.returncode == 0 else None
    except Exception:
        return None


# ----------------------------- Round-Based Game Detection -----------------------------

def analyze_resolution_quality(
    chapter_ranges: List[Dict],
    resolution_points: List[Dict],
    debug: bool = False,
) -> Dict[str, Any]:
    """Analyze resolution points to determine if they represent real round boundaries.
    
    Uses data-driven approach:
    - Ranks resolutions by score
    - Detects quality tier breaks (gap in score distribution)
    - Validates temporal spacing (not clustered noise)
    
    Returns dict with:
        - high_quality_resolutions: List of validated resolution events
        - is_round_based: bool
        - detection_mode: str (none/high_confidence/temporal_validation)
    """
    if not chapter_ranges or not resolution_points:
        return {'high_quality_resolutions': [], 'is_round_based': False, 'detection_mode': 'none', 'debug_info': []}
    
    chapter_start = min(r['start'] for r in chapter_ranges)
    chapter_end = max(r['end'] for r in chapter_ranges)
    chapter_duration_minutes = (chapter_end - chapter_start) / 60.0
    
    # Get all resolutions in chapter, sorted by score descending
    chapter_resolutions = sorted(
        [r for r in resolution_points if chapter_start <= r['ts'] <= chapter_end],
        key=lambda r: r['score'],
        reverse=True
    )
    
    if not chapter_resolutions:
        return {'high_quality_resolutions': [], 'is_round_based': False, 'detection_mode': 'none', 'debug_info': []}
    
    debug_info = []
    
    # Method 1: Top-K with quality floor
    # Take all resolutions above a quality threshold (score >= 2.0)
    # This is more lenient than gap-based detection
    
    # Strategy: Keep all resolutions with score >= 2.0
    # This captures real round ends regardless of score variance
    high_quality = [r for r in chapter_resolutions if r['score'] >= 2.0]
    
    debug_info.append(f"Initial filter (score >= 2.0): {len(high_quality)}/{len(chapter_resolutions)} kept")
    
    if debug:
        for i, res in enumerate(chapter_resolutions):
            status = "✓ kept" if res in high_quality else "✗ filtered (score < 2.0)"
            debug_info.append(f"  [{i+1}] score={res['score']:.2f} @ {res['ts']/60:.1f}min - {status}")
    
    # Method 2: Temporal validation - resolutions should be spaced out (not clustered noise)
    gaps = []
    
    if len(high_quality) >= 2:
        # Check average gap between resolutions
        timestamps = sorted([r['ts'] for r in high_quality])
        gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        avg_gap_minutes = (sum(gaps) / len(gaps)) / 60.0
        min_gap_minutes = min(gaps) / 60.0
        
        debug_info.append(f"Temporal check: avg_gap={avg_gap_minutes:.1f}min, min_gap={min_gap_minutes:.1f}min")
        
        # Adaptive thresholds based on chapter duration
        # Shorter chapters = shorter expected rounds
        expected_round_duration = chapter_duration_minutes / max(1, len(high_quality))
        min_avg_gap = max(3.0, expected_round_duration * 0.3)  # At least 30% of expected round length
        min_single_gap = 1.5  # Very lenient - just filter obvious false positives
        
        debug_info.append(f"  Expected round duration: {expected_round_duration:.1f}min")
        debug_info.append(f"  Thresholds: min_avg={min_avg_gap:.1f}min, min_single={min_single_gap:.1f}min")
        
        # Only filter if SEVERELY clustered (obvious false positives)
        if avg_gap_minutes < min_avg_gap and min_gap_minutes < min_single_gap:
            # Too clustered - keep only strongest ones
            original_count = len(high_quality)
            high_quality = sorted(high_quality, key=lambda r: r['score'], reverse=True)[:max(2, int(len(high_quality) * 0.5))]
            debug_info.append(f"  ⚠️  Clustered resolutions detected, filtered {original_count} → {len(high_quality)}")
            
            # Recalculate gaps after filtering
            if len(high_quality) >= 2:
                timestamps = sorted([r['ts'] for r in high_quality])
                gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                avg_gap_minutes = (sum(gaps) / len(gaps)) / 60.0
                debug_info.append(f"  After filtering: avg_gap={avg_gap_minutes:.1f}min")
    
    # Decision: Is this round-based?
    is_round_based = len(high_quality) >= 2  # At least 2 validated round boundaries
    
    detection_mode = 'none'
    if is_round_based:
        if len(high_quality) >= 3 and chapter_duration_minutes > 30:
            detection_mode = 'high_confidence'
        else:
            detection_mode = 'temporal_validation'
    
    debug_info.append(f"Final: {len(high_quality)} resolutions → {'ROUND-BASED' if is_round_based else 'CONTINUOUS'} ({detection_mode})")
    
    return {
        'high_quality_resolutions': high_quality,
        'is_round_based': is_round_based,
        'detection_mode': detection_mode,
        'total_resolutions': len(chapter_resolutions),
        'avg_gap_minutes': (sum(gaps) / len(gaps) / 60.0) if gaps else 0,
        'debug_info': debug_info,
    }


def segment_chapter_into_rounds(
    chapter_ranges: List[Dict],
    resolution_points: List[Dict],
) -> List[Dict]:
    """Segment chapter ranges into rounds using resolution tokens.
    
    Resolution tokens mark the END of a round. Each round extends from
    the previous resolution (or chapter start) UP TO AND INCLUDING the next resolution.
    
    Args:
        chapter_ranges: All ranges in the chapter
        resolution_points: Resolution events with 'ts' and 'score'
    
    Returns:
        List of round dicts, each containing its ranges
    """
    if not chapter_ranges:
        return []
    
    chapter_start = min(r['start'] for r in chapter_ranges)
    chapter_end = max(r['end'] for r in chapter_ranges)
    
    # Get strong resolution events (score >= 2.0) in order
    strong_resolutions = sorted(
        [r for r in resolution_points if chapter_start <= r['ts'] <= chapter_end and r['score'] >= 2.0],
        key=lambda r: r['ts']
    )
    
    # Filter out false positives
    filtered_resolutions = []
    
    # 1. Remove suspiciously early resolutions (first resolution <15min into chapter)
    # These are likely from pre-game chat or false positives
    if strong_resolutions:
        chapter_duration = chapter_end - chapter_start
        first_res = strong_resolutions[0]
        time_into_chapter = first_res['ts'] - chapter_start
        
        # If first resolution is suspiciously early AND next resolution is much later, skip first
        # Heuristic: first resolution <35% into chapter AND gap to next >20 minutes = false positive
        if len(strong_resolutions) > 1 and time_into_chapter < chapter_duration * 0.35:
            second_res = strong_resolutions[1]
            gap_to_next = second_res['ts'] - first_res['ts']
            if gap_to_next > 1200:  # 20+ minutes to next resolution
                # Skip first resolution (likely false positive or pre-game)
                strong_resolutions = strong_resolutions[1:]
    
    # 2. Merge resolutions that are too close together (within 5 minutes)
    # Keep the stronger one
    i = 0
    while i < len(strong_resolutions):
        current = strong_resolutions[i]
        
        # Look ahead for nearby resolutions
        j = i + 1
        while j < len(strong_resolutions):
            next_res = strong_resolutions[j]
            time_gap = next_res['ts'] - current['ts']
            
            if time_gap < 300:  # Within 5 minutes
                # Keep the stronger one
                if next_res['score'] > current['score']:
                    current = next_res
                j += 1
            else:
                break
        
        filtered_resolutions.append(current)
        i = j if j > i + 1 else i + 1
    
    res_timestamps = [r['ts'] for r in filtered_resolutions]
    
    if not res_timestamps:
        # No resolutions: treat whole chapter as one round
        return [{
            'round_idx': 0,
            'ranges': chapter_ranges,
            'start': chapter_start,
            'end': chapter_end,
            'duration': sum(r.get('duration', r['end'] - r['start']) for r in chapter_ranges),
        }]
    
    # Build rounds: each round ENDS at a resolution timestamp
    # Assign each range to EXACTLY ONE round (no overlaps)
    rounds = []
    round_start = chapter_start
    used_range_indices = set()
    
    for i, res_ts in enumerate(res_timestamps):
        # This round goes from round_start up to and including the resolution (+ small buffer)
        round_end = res_ts + 30.0
        
        # Collect ranges that belong to this round (exclusive assignment by midpoint)
        round_ranges = []
        for idx, r in enumerate(chapter_ranges):
            if idx in used_range_indices:
                continue  # Already assigned to previous round
            
            # Assign range to round if its midpoint falls within round boundaries
            r_mid = (r['start'] + r['end']) / 2.0
            if round_start <= r_mid <= round_end:
                round_ranges.append(r)
                used_range_indices.add(idx)
        
        if round_ranges:
            rounds.append({
                'round_idx': i,
                'ranges': round_ranges,
                'start': round_ranges[0]['start'],
                'end': round_ranges[-1]['end'],
                'duration': sum(rr.get('duration', rr['end'] - rr['start']) for rr in round_ranges),
                'resolution_ts': res_ts,
            })
        
        # Next round starts after this resolution (plus buffer)
        round_start = round_end
    
    # Handle any remaining content after last resolution
    remaining_ranges = [r for idx, r in enumerate(chapter_ranges) if idx not in used_range_indices]
    if remaining_ranges:
        rounds.append({
            'round_idx': len(res_timestamps),
            'ranges': remaining_ranges,
            'start': remaining_ranges[0]['start'],
            'end': remaining_ranges[-1]['end'],
            'duration': sum(rr.get('duration', rr['end'] - rr['start']) for rr in remaining_ranges),
            'resolution_ts': None,  # No resolution for this round yet
        })
    
    return rounds


def score_round_arc_confidence(
    arc_rounds: List[Dict],
    all_rounds: List[Dict],
    chapter_start: float,
) -> Dict[str, Any]:
    """Score confidence that an arc contains complete, valid rounds.
    
    Detects issues:
    - First round too long (includes startup pollution)
    - Round duration outliers (3x median)
    - Arc starts at chapter beginning (suspicious, likely includes startup)
    
    Returns dict with 'score' (0-1) and 'reasons' (list of issues).
    """
    reasons = []
    penalties = []
    
    if not arc_rounds:
        return {'score': 0.0, 'reasons': ['empty_arc'], 'confidence': 'none'}
    
    # Get round durations for this arc and all rounds in chapter
    arc_durations = [r['duration'] for r in arc_rounds]
    all_durations = [r['duration'] for r in all_rounds if r['duration'] > 60]  # Filter tiny rounds
    
    # Calculate typical round duration from all rounds
    if len(all_durations) >= 2:
        import statistics
        median_round = statistics.median(all_durations)
        
        # Check for duration outliers in this arc
        for i, dur in enumerate(arc_durations):
            if dur > median_round * 2.5:  # 2.5x median is suspicious
                penalties.append(0.3)
                reasons.append(f"round_{i}_outlier:{dur:.0f}s_vs_median_{median_round:.0f}s")
    
    # Check if arc starts at chapter beginning (likely includes startup)
    first_round = arc_rounds[0]
    if abs(first_round['start'] - chapter_start) < 60.0:  # Within 1min of chapter start
        penalties.append(0.5)
        reasons.append('starts_at_chapter_beginning')
        
        # Extra penalty if ALSO suspiciously long (definite startup pollution)
        if arc_durations and arc_durations[0] > 1800:  # 30+ minutes
            penalties.append(0.3)
            reasons.append(f"startup_pollution:{arc_durations[0]:.0f}s")
    
    # Check if first round in arc is suspiciously long (even if not at start)
    elif arc_durations and arc_durations[0] > 2400:  # 40+ minutes
        penalties.append(0.3)
        reasons.append(f"first_round_too_long:{arc_durations[0]:.0f}s")
    
    # Bonus: Multi-round arcs with consistent durations are good
    if len(arc_durations) >= 2:
        import statistics
        if arc_durations:
            try:
                cv = statistics.stdev(arc_durations) / statistics.mean(arc_durations)
                if cv < 0.3:  # Low coefficient of variation = consistent rounds
                    penalties.append(-0.2)  # Bonus
                    reasons.append('consistent_round_durations')
            except (statistics.StatisticsError, ZeroDivisionError):
                pass
    
    # Calculate final score
    base_score = 1.0
    final_score = max(0.0, min(1.0, base_score - sum(penalties)))
    
    # Classify confidence
    if final_score >= 0.7:
        confidence = 'high'
    elif final_score >= 0.4:
        confidence = 'medium'
    else:
        confidence = 'low'
    
    return {
        'score': final_score,
        'confidence': confidence,
        'reasons': reasons,
        'arc_durations': arc_durations,
    }


def cluster_full_rounds(
    rounds: List[Dict],
    target_min: float = 900.0,
    target_max: float = 1800.0,
    chapter_start: float = 0.0,
    min_confidence: float = 0.5,
) -> List[List[Dict]]:
    """Cluster full rounds into arcs (never split a round).
    
    Groups 1-3 consecutive rounds into arcs based on duration targets.
    Ensures each arc respects round boundaries.
    Filters out low-confidence arcs (startup pollution, incomplete rounds).
    
    Args:
        rounds: List of round dicts from segment_chapter_into_rounds()
        target_min: Minimum arc duration (seconds)
        target_max: Maximum arc duration (seconds)
        chapter_start: Chapter start time for pollution detection
        min_confidence: Minimum confidence score to keep arc (0-1)
    
    Returns:
        List of arcs (high-confidence only), each arc is a list of range dicts
    """
    if not rounds:
        return []
    
    arcs_with_rounds = []  # Store (arc_rounds, metadata) tuples
    current_arc_rounds = []
    current_duration = 0.0
    
    for round_dict in rounds:
        round_dur = round_dict['duration']
        
        if not current_arc_rounds:
            # Start new arc
            current_arc_rounds = [round_dict]
            current_duration = round_dur
            continue
        
        would_be_duration = current_duration + round_dur
        
        if would_be_duration <= target_max:
            # Add to current arc
            current_arc_rounds.append(round_dict)
            current_duration = would_be_duration
        
        elif current_duration >= target_min:
            # Current arc is good length, close it and start new one
            arcs_with_rounds.append(list(current_arc_rounds))
            current_arc_rounds = [round_dict]
            current_duration = round_dur
        
        else:
            # Current arc is too short, but adding this round exceeds max
            # Decision: Add it anyway (prefer complete rounds over strict duration)
            current_arc_rounds.append(round_dict)
            current_duration = would_be_duration
            # Close this arc (it's now above target_min)
            arcs_with_rounds.append(list(current_arc_rounds))
            current_arc_rounds = []
            current_duration = 0.0
    
    # Don't forget last arc
    if current_arc_rounds:
        arcs_with_rounds.append(current_arc_rounds)
    
    # Score and filter arcs by confidence
    high_confidence_arcs = []
    filtered_count = 0
    
    for idx, arc_rounds in enumerate(arcs_with_rounds, 1):
        confidence_data = score_round_arc_confidence(arc_rounds, rounds, chapter_start)
        
        # Log confidence details (will be printed if debug enabled)
        arc_start = arc_rounds[0]['start']
        arc_end = arc_rounds[-1]['end']
        arc_dur = sum(r['duration'] for r in arc_rounds)
        
        if confidence_data['score'] >= min_confidence:
            # Flatten: convert round groups to range lists
            arc_ranges = []
            for round_dict in arc_rounds:
                arc_ranges.extend(round_dict['ranges'])
            if arc_ranges:
                high_confidence_arcs.append(arc_ranges)
                # Store confidence metadata on first range for inspection
                arc_ranges[0]['_round_confidence'] = confidence_data['score']
                arc_ranges[0]['_round_confidence_reasons'] = confidence_data.get('reasons', [])
        else:
            filtered_count += 1
            import sys
            # Log filtered arcs to stderr for debugging
            print(
                f"      ⚠️  Filtered arc {idx}: {arc_start/60:.1f}min-{arc_end/60:.1f}min ({arc_dur/60:.1f}min) "
                f"[confidence={confidence_data['score']:.2f}, reasons: {', '.join(confidence_data.get('reasons', []))}]",
                file=sys.stderr
            )
    
    return high_confidence_arcs


# ----------------------------- Resolution Helpers -----------------------------

def _compute_resolution_score(reasons: List[str]) -> float:
    """Compute resolution quality score from token reasons.
    
    Returns score based on token confidence levels:
    - High confidence (0.95): gg, ggs, finished, we won, etc.
    - Medium confidence (0.75): o7, nt, peepoclap, rip, etc.
    - Low confidence (0.45): om, etc.
    - Boosters: chat_z, burst_score add small bonuses
    """
    # Token confidence tiers (matching print_resolutions.py)
    high_confidence = {
        "gg", "ggs", "gg no re", "good game", "ggez", "ggwp",
        "go next", "gonext", "next round", "next game", "new game",
        "queue up", "queue again", "requeue",
        "match found", "match over", "round over", "game over", "game ended",
        "new run", "good run", "goodrun", "total wipeout",
        "back to lobby", "return to lobby", "back to menu",
        "you are the champion", "chicken dinner", "you died", "wasted", "mission failed",
        "terrorists win", "counter terrorists win", "final killcam", "play of the game",
        "we won", "we lost", "victory", "victory royale", "defeat",
        "finish", "finished", "end", "its over", "it's over", "die", "death",
    }
    medium_confidence = {
        "ez", "ezclap", "easyclap", "pepeclap", "peepoclap",
        "goodbye", "bye", "o7", "07", "ff", "ff15",
        "ready up", "q up", "wp", "well played", "nt", "nice try",
        "rip", "press f", "f in chat", "on to the next", "onto the next",
    }
    low_confidence = {"om"}
    
    score = 0.0
    for token in reasons:
        t = str(token).lower().strip()
        if t in high_confidence:
            score += 0.95
        elif t in medium_confidence:
            score += 0.75
        elif t in low_confidence:
            score += 0.45
        elif t in ("chat_z", "burst_score"):
            score += 0.2  # Boosters
    
    return score


def _load_chapters_data(vod_id: str) -> Optional[Dict[str, Any]]:
    """Load chapter data for a VOD."""
    import json
    ch_path = Path(f"data/ai_data/{vod_id}/{vod_id}_chapters.json")
    if not ch_path.exists():
        return None
    try:
        with ch_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _filter_ranges_for_chapter(ranges: List[Range], chapter_start: float, chapter_end: float) -> List[Range]:
    """Filter ranges that overlap with a chapter."""
    filtered = []
    for r in ranges:
        # Keep ranges that have any overlap with chapter
        if r.end > chapter_start and r.start < chapter_end:
            filtered.append(r)
    return filtered


# ----------------------------- Main -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster Director's Cut manifest into story arcs")
    parser.add_argument("vod_id", help="VOD ID")
    parser.add_argument("--manifest", default=None, help="Explicit enhanced manifest path")
    parser.add_argument("--target-min", type=int, default=int(os.getenv("ARC_TARGET_MIN", "600")), help="Target minimum arc duration (seconds)")
    parser.add_argument("--target-max", type=int, default=int(os.getenv("ARC_TARGET_MAX", "1800")), help="Target maximum arc duration (seconds)")
    parser.add_argument("--sim-threshold", type=float, default=float(os.getenv("ARC_SIM_THRESHOLD", "0.45")), help="Similarity threshold to keep adjacent ranges in the same arc")
    parser.add_argument("--time-tau", type=float, default=float(os.getenv("ARC_TIME_TAU", "120.0")), help="Time proximity decay (seconds) for adjacency")
    parser.add_argument("--topic-bonus", type=float, default=float(os.getenv("ARC_TOPIC_BONUS", "0.1")), help="Bonus when topic_key matches")
    parser.add_argument("--render", action="store_true", help="Render each arc into a separate video using the existing renderer")
    parser.add_argument("--render-args", nargs=argparse.REMAINDER, help="Extra args forwarded to the renderer after '--' (e.g. -- --transition-duration 1.0 --audio-crossfade)")
    parser.add_argument("--timestamps", action="store_true", help="Generate YouTube timestamp lines per arc")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM for timestamps (condensed summaries only)")
    parser.add_argument("--max-title-chars", type=int, default=int(os.getenv("TS_TITLE_MAX_CHARS", "50")), help="Max title length for timestamps")
    parser.add_argument("--min-score", type=float, default=float(os.getenv("ARC_MIN_SCORE", "0.70")), help="Minimum composite score to accept an arc")
    parser.add_argument("--emit-debug", action="store_true", help="Emit arc scoring debug to stdout")
    parser.add_argument("--min-contiguity", type=float, default=float(os.getenv("ARC_MIN_CONTIGUITY", "0.6")), help="Minimum contiguity ratio (kept/span) required")
    parser.add_argument("--max-gap-seconds", type=float, default=float(os.getenv("ARC_MAX_GAP_SECONDS", "180.0")), help="Maximum allowed internal gap (seconds)")
    parser.add_argument("--max-gap-fraction", type=float, default=float(os.getenv("ARC_MAX_GAP_FRACTION", "0.4")), help="Maximum allowed total gap fraction of the span")
    parser.add_argument("--max-arcs", type=int, default=1, help="Maximum number of arcs to generate (default: 1)")
    parser.add_argument("--resolution-grace-seconds", type=float, default=float(os.getenv("ARC_RESOLUTION_GRACE_SECONDS", "240.0")), help="Allow bounded overshoot past target-max to capture imminent resolution")
    args = parser.parse_args()

    vod_id = args.vod_id
    # Self-heal: ensure enhanced manifest exists; generate if missing
    try:
        ranges, base_manifest, src_path = _load_enhanced_manifest(vod_id, args.manifest)
    except FileNotFoundError as _e:
        print(f"⚠️  Enhanced manifest missing for {vod_id}; generating via rag.enhanced_director_cut_selector...")
        try:
            cmd = [sys.executable, "-u", "-m", "rag.enhanced_director_cut_selector", vod_id]
        except Exception:
            cmd = ["python", "-m", "rag.enhanced_director_cut_selector", vod_id]
        try:
            _ = subprocess.run(cmd, capture_output=False, check=False)
        except Exception as _ge:
            print(f"❌ Could not generate enhanced manifest: {_ge}")
        # Retry once
        ranges, base_manifest, src_path = _load_enhanced_manifest(vod_id, args.manifest)
    retriever = _load_retriever(vod_id)
    
    # Load chapters for chapter-first processing (merged chapters only)
    chapters_data = _load_chapters_data(vod_id)
    chapters: List[Dict[str, Any]] = []
    if chapters_data is not None:
        raw_chapters = chapters_data.get("chapters", [])
        # Validate and normalize chapters: require start_time/end_time in seconds
        errors: List[str] = []
        def _normalize_label(text: str) -> str:
            try:
                import re
                t = str(text or "").lower()
                t = re.sub(r"[_\s]+", " ", t).strip()
                t = re.sub(r"[^a-z0-9 ]", "", t)
                return t.replace(" ", "")
            except Exception:
                return str(text or "").strip().lower()
        def _mk_chapter_name(ch: Dict[str, Any], idx: int) -> Tuple[str, str]:
            """Return (display_name_for_hashtag, display_base_for_series)."""
            orig = str(ch.get("original_category") or "").strip()
            cat = str(ch.get("category") or "").strip()
            cat_pretty = cat.replace("_", " ").strip()
            n_orig = _normalize_label(orig)
            n_cat = _normalize_label(cat_pretty)
            # Build hashtag display name
            if orig and cat_pretty and n_orig and n_cat and n_orig != n_cat:
                hashtag_name = f"{orig} + {cat_pretty.title()}"
            else:
                hashtag_name = orig or (cat_pretty.title() if cat_pretty else f"Chapter {idx+1}")
            # Build series base display: prefer original formatting if equivalent
            if orig and n_orig and n_orig == n_cat:
                base_display = orig
            elif cat_pretty:
                base_display = cat_pretty.title()
            elif orig:
                base_display = orig
            else:
                base_display = f"Chapter {idx+1}"
            return hashtag_name, base_display
        for i, ch in enumerate(raw_chapters):
            try:
                st = float(ch.get("start_time"))
                et = float(ch.get("end_time"))
            except Exception:
                errors.append(f"chapter {i+1}: missing numeric start_time/end_time")
                continue
            if not (et > st >= 0):
                errors.append(f"chapter {i+1}: invalid bounds start_time={ch.get('start_time')} end_time={ch.get('end_time')}")
                continue
            orig_cat = str(ch.get("original_category") or "").strip()
            cat_slug = str(ch.get("category") or "").strip()
            cat_pretty = cat_slug.replace("_", " ").strip().title() if cat_slug else ""
            hashtag_name, base_display = _mk_chapter_name(ch, i)
            chapters.append({
                "name": hashtag_name,
                "start": float(st),
                "end": float(et),
                "raw": ch,
                "orig_category": orig_cat,
                "category_pretty": cat_pretty,
                "display_base": base_display,
            })
        if errors:
            msg = "\n".join(errors)
            raise SystemExit(
                f"Invalid chapter data in data/ai_data/{vod_id}/{vod_id}_chapters.json:\n{msg}\nRefusing to fall back to whole-VOD; please fix chapters and retry."
            )

    # Detector-first approach for unicorn arcs; strict gating, no fallbacks
    from story_archs.detectors import detect_arcs  # type: ignore
    target_min = max(60, int(args.target_min))
    target_max = max(int(args.target_min) + 60, int(args.target_max))
    sim_th = float(args.sim_threshold)
    time_tau = float(args.time_tau)
    topic_bonus = float(args.topic_bonus)

    # Chapter-first processing: Process each chapter independently
    all_detected_arcs: List[List[Dict[str, Any]]] = []
    
    if chapters:
        print(f"📖 Processing {len(chapters)} chapters independently...")
        for ch_idx, ch in enumerate(chapters):
            ch_name = str(ch.get("name", f"Chapter {ch_idx+1}"))
            ch_start = float(ch.get("start", 0.0))
            ch_end = float(ch.get("end", 0.0))
            ch_duration = ch_end - ch_start
            
            print(f"\n  Chapter {ch_idx+1}: {ch_name} ({ch_start:.0f}s -> {ch_end:.0f}s, {ch_duration/60:.1f}min)")
            
            # Filter ranges for this chapter
            chapter_ranges = _filter_ranges_for_chapter(ranges, ch_start, ch_end)
            if not chapter_ranges:
                print("    ⚠️  No ranges in chapter, skipping")
                continue
            
            raw_ranges_sorted = [r.raw for r in chapter_ranges]
            
            # Extract resolution points for this chapter
            resolution_points = []
            for r in raw_ranges_sorted:
                res_ts = r.get('_resolution_event_ts')
                res_reasons = r.get('_resolution_reasons')
                if res_ts is not None and res_reasons:
                    score = _compute_resolution_score(res_reasons)
                    if score > 0:
                        resolution_points.append({
                            'ts': float(res_ts),
                            'score': score,
                            'reasons': list(res_reasons) if isinstance(res_reasons, list) else [],
                        })
            
            # Analyze resolution quality (data-driven, no hardcoded thresholds)
            resolution_analysis = analyze_resolution_quality(
                raw_ranges_sorted, 
                resolution_points,
                debug=bool(args.emit_debug)
            )
            is_round_based = resolution_analysis['is_round_based']
            high_quality_resolutions = resolution_analysis['high_quality_resolutions']
            
            # Debug: show resolution analysis
            if resolution_points:
                ch_start_time = raw_ranges_sorted[0]['start'] if raw_ranges_sorted else 0
                ch_end_time = raw_ranges_sorted[-1]['end'] if raw_ranges_sorted else 0
                ch_resolutions = [r for r in resolution_points if ch_start_time <= r['ts'] <= ch_end_time]
                if ch_resolutions:
                    all_scores = ", ".join(f"{r['score']:.1f}" for r in sorted(ch_resolutions, key=lambda x: x['ts']))
                    hq_scores = ", ".join(f"{r['score']:.1f}" for r in sorted(high_quality_resolutions, key=lambda x: x['ts']))
                    print(f"    📍 All resolutions: [{all_scores}]")
                    print(f"    ✅ High-quality: [{hq_scores}]")
                    
                    # Print detailed debug info if enabled
                    if args.emit_debug and resolution_analysis.get('debug_info'):
                        print("    🔍 Resolution Analysis:")
                        for line in resolution_analysis['debug_info']:
                            print(f"       {line}")
            
            # Check if this is a Just Chatting chapter
            is_jc_chapter = False
            if ch.get('raw'):
                cat = str(ch['raw'].get('category', '')).lower().replace('_', ' ')
                orig_cat = str(ch['raw'].get('original_category', '')).lower()
                is_jc_chapter = ('just chatting' in cat or 'just chatting' in orig_cat)
            
            # Mark all JC content for later detection
            if is_jc_chapter:
                for r in raw_ranges_sorted:
                    r['_is_jc'] = True
            
            # Special handling for short Just Chatting chapters
            if is_jc_chapter and ch_duration < 3600:  # Under 60 minutes
                # Keep as single arc to avoid fragmentation
                print("    💬 Just Chatting chapter (<60min) → single arc")
                chapter_arcs = [raw_ranges_sorted]  # One arc with all ranges
            
            elif is_round_based:
                # ROUND-BASED: Segment into rounds and cluster full rounds only
                print(f"    🎮 Round-based game detected ({len(high_quality_resolutions)} validated rounds)")
                # Use only high-quality resolutions for segmentation
                rounds = segment_chapter_into_rounds(raw_ranges_sorted, high_quality_resolutions)
                chapter_arcs = cluster_full_rounds(
                    rounds, 
                    target_min=target_min, 
                    target_max=target_max,
                    chapter_start=ch_start,
                    min_confidence=0.5,  # Filter out polluted/incomplete arcs
                )
                print(f"    📊 Segmented into {len(rounds)} rounds → {len(chapter_arcs)} high-confidence arcs")
                
                # Mark as round-based to disable gap filling later
                for arc in chapter_arcs:
                    for r in arc:
                        r['_is_round_based'] = True
            else:
                # CONTINUOUS: Use semantic clustering (current approach)
                print("    🎯 Continuous game detected, using semantic clustering")
                chapter_arcs = detect_arcs(
                    raw_ranges_sorted,
                    retriever,
                    target_min=target_min,
                    target_max=target_max,
                    sim_threshold=sim_th,
                    time_tau=time_tau,
                    topic_bonus=topic_bonus,
                    min_score=float(args.min_score),
                    debug=bool(args.emit_debug),
                    min_contiguity=float(args.min_contiguity),
                    max_gap_seconds=float(args.max_gap_seconds),
                    max_gap_fraction=float(args.max_gap_fraction),
                    resolution_grace_seconds=float(args.resolution_grace_seconds),
                    resolution_points=resolution_points,
                )
            
            # Tag arcs with chapter info
            for arc in chapter_arcs:
                for r in arc:
                    r['_chapter_idx'] = ch_idx
                    r['_chapter_name'] = ch_name
                    # For base title ("Plays X"), use display_base (prefers original formatting when equivalent)
                    base_name = str(ch.get("display_base") or "").strip() or ch_name
                    r['_chapter_base'] = base_name
            
            all_detected_arcs.extend(chapter_arcs)
            print(f"    ✅ Found {len(chapter_arcs)} arcs in {ch_name}")
    else:
        # No chapters: Fall back to whole-VOD processing
        print("📖 No chapters found, processing entire VOD as one segment...")
        raw_ranges_sorted = [r.raw for r in ranges]
        
        # Extract resolution points from manifest for round-based game detection
        resolution_points = []
        for r in raw_ranges_sorted:
            res_ts = r.get('_resolution_event_ts')
            res_reasons = r.get('_resolution_reasons')
            if res_ts is not None and res_reasons:
                score = _compute_resolution_score(res_reasons)
                if score > 0:
                    resolution_points.append({
                        'ts': float(res_ts),
                        'score': score,
                        'reasons': list(res_reasons) if isinstance(res_reasons, list) else [],
                    })
        
        # Analyze resolution quality (data-driven)
        resolution_analysis = analyze_resolution_quality(
            raw_ranges_sorted, 
            resolution_points,
            debug=bool(args.emit_debug)
        )
        is_round_based = resolution_analysis['is_round_based']
        high_quality_resolutions = resolution_analysis['high_quality_resolutions']
        
        # Print debug info if enabled
        if args.emit_debug and resolution_analysis.get('debug_info'):
            print("🔍 Resolution Analysis:")
            for line in resolution_analysis['debug_info']:
                print(f"   {line}")
        
        if is_round_based:
            # ROUND-BASED: Segment into rounds and cluster full rounds only
            print(f"🎮 Round-based game detected ({len(high_quality_resolutions)} validated rounds)")
            rounds = segment_chapter_into_rounds(raw_ranges_sorted, high_quality_resolutions)
            vod_start = raw_ranges_sorted[0]['start'] if raw_ranges_sorted else 0.0
            all_detected_arcs = cluster_full_rounds(
                rounds, 
                target_min=target_min, 
                target_max=target_max,
                chapter_start=vod_start,
                min_confidence=0.5,
            )
            print(f"📊 Segmented into {len(rounds)} rounds → {len(all_detected_arcs)} high-confidence arcs")
        else:
            # CONTINUOUS: Use semantic clustering
            print("🎯 Continuous game detected, using semantic clustering")
            all_detected_arcs = detect_arcs(
                raw_ranges_sorted,
                retriever,
                target_min=target_min,
                target_max=target_max,
                sim_threshold=sim_th,
                time_tau=time_tau,
                topic_bonus=topic_bonus,
                min_score=float(args.min_score),
                debug=bool(args.emit_debug),
                min_contiguity=float(args.min_contiguity),
                max_gap_seconds=float(args.max_gap_seconds),
                max_gap_fraction=float(args.max_gap_fraction),
                resolution_grace_seconds=float(args.resolution_grace_seconds),
                resolution_points=resolution_points,
            )
    
    detected_arcs = all_detected_arcs

    # Note: max_arcs limit disabled for gap filling - we want full coverage
    # The gap filler will create arcs for any missing time spans
    if args.max_arcs > 0 and len(detected_arcs) > args.max_arcs:
        print(f"⚠️  Found {len(detected_arcs)} arcs, but max_arcs={args.max_arcs} is set")
        print("    With gap filling enabled, recommend setting --max-arcs 0 for full coverage")

    # Convert detected arcs to Range blocks (no fallbacks)
    def _to_range(r: Dict[str, Any]) -> Range:
        s = float(r.get("start", 0.0))
        e = float(r.get("end", 0.0))
        d = float(r.get("duration") or (e - s))
        tk = str(r.get("topic_key") or "")
        abid = str(r.get("anchor_burst_id") or "").strip() or None
        bids = r.get("burst_ids") if isinstance(r.get("burst_ids"), list) else []
        return Range(start=s, end=e, duration=d, topic_key=tk, anchor_burst_id=abid, burst_ids=[str(b) for b in bids if isinstance(b, str)], raw=r)

    clusters: List[List[Range]] = []
    for arc in detected_arcs:
        clusters.append([_to_range(r) for r in arc])

    # Gap filling: Create arcs for missing time spans to ensure full coverage
    # Large gaps are chunked to respect target_max (for processing constraints)
    def _chunk_gap(gap_ranges: List[Range], max_duration: float, chapter_info: Optional[Dict[str, Any]] = None) -> List[List[Range]]:
        """Split a gap into multiple arcs if it exceeds max_duration."""
        if not gap_ranges:
            return []
        
        # Tag gap ranges with chapter info if provided
        if chapter_info:
            for r in gap_ranges:
                if '_chapter_idx' not in r.raw:
                    r.raw['_chapter_idx'] = chapter_info.get('idx')
                    r.raw['_chapter_name'] = chapter_info.get('name')
        
        total_dur = sum(r.duration for r in gap_ranges)
        if total_dur <= max_duration:
            return [gap_ranges]
        
        # Split into chunks
        chunks: List[List[Range]] = []
        current_chunk: List[Range] = []
        current_dur = 0.0
        
        for r in gap_ranges:
            if current_dur > 0 and current_dur + r.duration > max_duration:
                # Start new chunk
                chunks.append(current_chunk)
                current_chunk = [r]
                current_dur = r.duration
            else:
                current_chunk.append(r)
                current_dur += r.duration
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    # Chapter-aware gap filling: Fill gaps within each chapter separately
    # Skip gap filling for round-based games (confidence filtering is intentional)
    if chapters and clusters:
        print("\n🔧 Chapter-aware gap filling...")
        clusters.sort(key=lambda c: c[0].start)
        gap_min_duration = 600  # Minimum 10 minutes to create a gap arc
        filled_clusters: List[List[Range]] = []
        
        for ch_idx, ch in enumerate(chapters):
            ch_name = ch.get("name", f"Chapter {ch_idx+1}")
            ch_start = float(ch.get("start", 0.0))
            ch_end = float(ch.get("end", 0.0))
            chapter_info = {'idx': ch_idx, 'name': ch_name}
            
            # Get all arcs for this chapter
            ch_arcs = [c for c in clusters if c[0].raw.get('_chapter_idx') == ch_idx]
            
            # Check if this chapter has round-based arcs
            chapter_has_round_based = any(
                c[0].raw.get('_is_round_based', False) 
                for c in ch_arcs
            )
            
            if chapter_has_round_based:
                # Round-based: skip gap filling (filtered arcs are intentionally excluded)
                print("  🎮 %s: Round-based chapter, skipping gap filling" % ch_name)
                filled_clusters.extend(ch_arcs)
                continue
            if not ch_arcs:
                # No detected arcs in chapter - create one arc for entire chapter if long enough
                ch_duration = ch_end - ch_start
                if ch_duration >= gap_min_duration:
                    ch_ranges = [r for r in ranges if ch_start <= r.start < ch_end]
                    if ch_ranges:
                        gap_chunks = _chunk_gap(ch_ranges, target_max, chapter_info)
                        filled_clusters.extend(gap_chunks)
                        print(f"  📖 {ch_name}: Created {len(gap_chunks)} arc(s) (entire chapter)")
                continue
            
            # Sort chapter arcs by start time
            ch_arcs.sort(key=lambda c: c[0].start)
            
            # Get chapter ranges
            ch_ranges = _filter_ranges_for_chapter(ranges, ch_start, ch_end)
            
            # Fill gap before first arc in chapter
            first_arc_start = ch_arcs[0][0].start
            if first_arc_start - ch_start >= gap_min_duration:
                gap_ranges = [r for r in ch_ranges if ch_start <= r.start < first_arc_start]
                if gap_ranges:
                    gap_chunks = _chunk_gap(gap_ranges, target_max, chapter_info)
                    filled_clusters.extend(gap_chunks)
                    print(f"  📖 {ch_name}: Added {len(gap_chunks)} gap arc(s) before first arc")
            
            # Add first arc
            filled_clusters.append(ch_arcs[0])
            
            # Fill gaps between consecutive arcs in chapter
            for i in range(len(ch_arcs) - 1):
                curr_end = ch_arcs[i][-1].end
                next_start = ch_arcs[i + 1][0].start
                gap_duration = next_start - curr_end
                
                if gap_duration >= gap_min_duration:
                    gap_ranges = [r for r in ch_ranges if curr_end <= r.start < next_start]
                    if gap_ranges:
                        gap_chunks = _chunk_gap(gap_ranges, target_max, chapter_info)
                        filled_clusters.extend(gap_chunks)
                        print(f"  📖 {ch_name}: Added {len(gap_chunks)} gap arc(s) between arcs")
                
                filled_clusters.append(ch_arcs[i + 1])
            
            # Fill gap after last arc in chapter
            last_arc_end = ch_arcs[-1][-1].end
            if ch_end - last_arc_end >= gap_min_duration:
                gap_ranges = [r for r in ch_ranges if last_arc_end <= r.start < ch_end]
                if gap_ranges:
                    gap_chunks = _chunk_gap(gap_ranges, target_max, chapter_info)
                    filled_clusters.extend(gap_chunks)
                    print(f"  📖 {ch_name}: Added {len(gap_chunks)} gap arc(s) after last arc")
        
        clusters = filled_clusters
        clusters.sort(key=lambda c: c[0].start)
    elif clusters:
        # No chapters: Use original whole-VOD gap filling
        # Skip if round-based
        is_round_based_vod = any(
            c[0].raw.get('_is_round_based', False) 
            for c in clusters
        )
        
        if is_round_based_vod:
            print("\n🎮 Round-based VOD, skipping gap filling")
        else:
            print("\n🔧 Whole-VOD gap filling...")
            clusters.sort(key=lambda c: c[0].start)
            vod_start = ranges[0].start if ranges else 0.0
            vod_end = ranges[-1].end if ranges else 0.0
            gap_min_duration = 600
            filled_clusters: List[List[Range]] = []
            
            # Gap before first arc
            if clusters[0][0].start - vod_start >= gap_min_duration:
                gap_ranges = [r for r in ranges if vod_start <= r.start < clusters[0][0].start]
                if gap_ranges:
                    gap_chunks = _chunk_gap(gap_ranges, target_max)
                    filled_clusters.extend(gap_chunks)
                    print(f"  Added {len(gap_chunks)} gap arc(s) before first arc")
            
            filled_clusters.append(clusters[0])
            
            # Gaps between arcs
            for i in range(len(clusters) - 1):
                curr_end = clusters[i][-1].end
                next_start = clusters[i + 1][0].start
                gap_duration = next_start - curr_end
                
                if gap_duration >= gap_min_duration:
                    gap_ranges = [r for r in ranges if curr_end <= r.start < next_start]
                    if gap_ranges:
                        gap_chunks = _chunk_gap(gap_ranges, target_max)
                        filled_clusters.extend(gap_chunks)
                        print(f"  Added {len(gap_chunks)} gap arc(s) between arcs")
                
                filled_clusters.append(clusters[i + 1])
            
            # Gap after last arc
            if vod_end - clusters[-1][-1].end >= gap_min_duration:
                gap_ranges = [r for r in ranges if clusters[-1][-1].end <= r.start <= vod_end]
                if gap_ranges:
                    gap_chunks = _chunk_gap(gap_ranges, target_max)
                    filled_clusters.extend(gap_chunks)
                    print(f"  Added {len(gap_chunks)} gap arc(s) after last arc")
            
            clusters = filled_clusters
            clusters.sort(key=lambda c: c[0].start)

    # Filter/chunk arcs based on duration and content type
    # - Short arcs (<5min): Skip (too short for meaningful content)
    # - Long JC arcs (>90min): Chunk into ~90min pieces (preserve JC content)
    # - Long game arcs (>90min): Skip (likely unusable/incomplete rounds)
    min_arc_duration = 300  # 5 minutes in seconds
    max_arc_duration = 5400  # 90 minutes in seconds
    chunk_target = 5100  # Target 85 minutes per chunk (slightly under max)
    
    def _is_jc_arc(cluster: List[Range]) -> bool:
        """Check if arc is primarily Just Chatting content."""
        # Check if any range has JC marker
        return any(r.raw.get('_is_jc', False) for r in cluster)
    
    def _chunk_long_arc(cluster: List[Range], target_duration: float) -> List[List[Range]]:
        """Split a long arc into chunks of roughly target_duration."""
        chunks: List[List[Range]] = []
        current_chunk: List[Range] = []
        current_duration = 0.0
        
        for r in cluster:
            if current_duration > 0 and current_duration + r.duration > target_duration:
                # Start new chunk
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [r]
                current_duration = r.duration
            else:
                current_chunk.append(r)
                current_duration += r.duration
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    processed_clusters = []
    skipped_short_arcs = []
    skipped_long_arcs = []
    chunked_jc_arcs = []
    
    for cluster in clusters:
        arc_duration = sum(r.duration for r in cluster)
        is_jc = _is_jc_arc(cluster)
        
        if arc_duration < min_arc_duration:
            # Too short - skip
            skipped_short_arcs.append({
                'duration_minutes': arc_duration / 60.0,
                'start': cluster[0].start,
                'end': cluster[-1].end,
                'type': 'JC' if is_jc else 'game',
            })
        elif arc_duration > max_arc_duration:
            if is_jc:
                # Long JC: Chunk it into manageable pieces
                chunks = _chunk_long_arc(cluster, chunk_target)
                processed_clusters.extend(chunks)
                chunked_jc_arcs.append({
                    'original_duration': arc_duration / 60.0,
                    'num_chunks': len(chunks),
                    'start': cluster[0].start,
                    'end': cluster[-1].end,
                })
            else:
                # Long game arc: Skip (likely incomplete/unusable)
                skipped_long_arcs.append({
                    'duration_minutes': arc_duration / 60.0,
                    'start': cluster[0].start,
                    'end': cluster[-1].end,
                    'type': 'game',
                })
        else:
            # Good duration - keep as-is
            processed_clusters.append(cluster)
    
    # Report filtering/chunking
    if skipped_short_arcs or skipped_long_arcs or chunked_jc_arcs:
        print("\n⏭️  Processing arcs:")
        if skipped_short_arcs:
            print(f"   📉 Skipped {len(skipped_short_arcs)} short arc(s) (<{min_arc_duration/60:.0f}min):")
            for info in skipped_short_arcs:
                print(f"      - {info['type']}: {info['duration_minutes']:.1f}min ({_format_hms(info['start'])} → {_format_hms(info['end'])})")
        if skipped_long_arcs:
            print(f"   ⚠️  Skipped {len(skipped_long_arcs)} long game arc(s) (>{max_arc_duration/60:.0f}min):")
            for info in skipped_long_arcs:
                print(f"      - {info['duration_minutes']:.1f}min ({_format_hms(info['start'])} → {_format_hms(info['end'])})")
        if chunked_jc_arcs:
            print(f"   ✂️  Chunked {len(chunked_jc_arcs)} long JC arc(s) into smaller pieces:")
            for info in chunked_jc_arcs:
                print(f"      - {info['original_duration']:.1f}min → {info['num_chunks']} chunks (~{chunk_target/60:.0f}min each)")
        
        total_kept = len(processed_clusters)
        total_jc = sum(1 for c in processed_clusters if _is_jc_arc(c))
        total_game = total_kept - total_jc
        print(f"   ✅ Result: {total_kept} arc(s) ({total_jc} JC, {total_game} game)")
    
    if not processed_clusters:
        print("⚠️ No arcs in usable duration range - no manifests generated")
        print("   Consider adjusting clustering parameters or checking VOD content")
        return
    
    filtered_clusters = processed_clusters
    
    out_dir, arc_paths, index_path = _write_arc_manifests(vod_id, filtered_clusters, base_manifest)
    print(f"✅ Wrote {len(arc_paths)} arc manifests to {out_dir}")
    print(f"📄 Index: {index_path}")

    # Generate titles with format: {general name} (part {num}) #{chapter name}
    try:
        from story_archs.create_arch_title import generate_titles  # type: ignore
        wrote = generate_titles(vod_id, only_arc=None, force=False)
        if wrote == 0:
            print("⚠️ No new titles generated (use --force to regenerate)")
    except Exception as _e:
        print(f"⚠️ Title generation failed: {_e}")

    if args.render:
        print("🎬 Rendering arcs...")
        extra = list(args.render_args or [])
        for arc_idx, arc_manifest in enumerate(arc_paths, 1):
            output_path = out_dir / f"{vod_id}_arc_{arc_idx:03d}.mp4"
            ok = _run_render_for_arc(vod_id, arc_manifest, output_path=output_path, extra_args=extra)
            print(("✅" if ok else "❌"), f"Arc {arc_idx:03d} render -> {output_path if ok else 'failed'}")

    if args.timestamps:
        print("⏱️  Generating timestamps per arc...")
        for arc_idx, arc_manifest in enumerate(arc_paths, 1):
            out = _run_timestamps_for_arc(vod_id, arc_manifest, no_llm=bool(args.no_llm), max_chars=int(args.max_title_chars))
            print(("✅" if out else "❌"), f"Arc {arc_idx:03d} timestamps -> {out if out else 'failed'}")

    # Final tips
    print("\nNext steps:")
    print(f"- To render all arcs: python -m story_archs.create_story_archs {vod_id} --render -- --transition-duration 1.0 --audio-crossfade --use-existing")
    print(f"- To generate timestamps: python -m story_archs.create_story_archs {vod_id} --timestamps --no-llm")


if __name__ == "__main__":
    main()


