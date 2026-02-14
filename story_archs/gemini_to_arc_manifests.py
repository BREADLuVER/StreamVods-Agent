#!/usr/bin/env python3
"""
Convert Gemini arc manifest to individual arc manifests for video rendering.

This script bridges the new Gemini arc detection output to the existing
create_arch_videos.py renderer by:
  1. Reading gemini_arc_manifest.json
  2. Rating arcs with a composite quality score
  3. Selecting the best arcs (dynamic or top-k)
  4. Creating individual arc_XXX_manifest.json files
  5. Creating arcs_index.json for the video renderer

Usage:
  python -m story_archs.gemini_to_arc_manifests <vod_id> [--top-k 5] [--min-rating 50]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _format_hms(sec: float) -> str:
    s = int(round(max(0.0, float(sec))))
    h = s // 3600
    m = (s % 3600) // 60
    s2 = s % 60
    return f"{h:02d}:{m:02d}:{s2:02d}"


def _format_duration_display(sec: float) -> str:
    s = int(round(max(0.0, float(sec))))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        m = s // 60
        s2 = s % 60
        return f"{m}m {s2}s" if s2 > 0 else f"{m}m"
    h = s // 3600
    m = (s % 3600) // 60
    return f"{h}h {m}m" if m > 0 else f"{h}h"


def load_gemini_arc_manifest(vod_id: str, manifest_path: Optional[str] = None) -> Dict[str, Any]:
    """Load the Gemini arc manifest."""
    if manifest_path:
        path = Path(manifest_path)
    else:
        path = Path(f"data/vector_stores/{vod_id}/gemini_arc_manifest.json")
    
    if not path.exists():
        raise FileNotFoundError(f"Gemini arc manifest not found: {path}")
    
    return json.loads(path.read_text(encoding="utf-8"))


# -----------------------------------------------------------------------------
# Arc Rating System
# -----------------------------------------------------------------------------

def compute_arc_rating(arc: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """
    Compute two separate unique scores: HYPE SCORE and YAP SCORE.
    
    Returns: (max_score, breakdown_dict)
    
    Scores:
    1. HYPE SCORE (0-100): Based on Climax Intensity (Chat Rate, Burst Score, Reactions).
       - Used for Gameplay, High Energy moments.
    2. YAP SCORE (0-100): Based on Narrative Interest & Controversy.
       - Used for Parasocial Yap, Drama, Stories, Reactions.
       
    The final rating is simply MAX(HYPE, YAP).
    We drop Confidence and Resolution from the score calculation entirely (they are just noise).
    """
    breakdown = {}
    
    # 1. Hype Score (0-100)
    # ---------------------
    climax = arc.get("climax", {})
    climax_score = float(climax.get("score", 0))
    peak_chat_z = float(climax.get("peak_chat_rate_z", 0))
    total_reactions = int(climax.get("total_reactions", 0))
    
    # Normalize climax score: scores typically range 0-10, exceptional can be 15+
    climax_normalized = min(1.0, climax_score / 8.0)  # 8+ = perfect
    chat_normalized = min(1.0, peak_chat_z / 6.0)      # 6+ z-score = perfect
    reaction_normalized = min(1.0, total_reactions / 30.0)  # 30+ reactions = perfect
    
    hype_score = (
        climax_normalized * 0.5 +
        chat_normalized * 0.3 +
        reaction_normalized * 0.2
    ) * 100.0
    breakdown["hype"] = round(hype_score, 2)
    
    # 2. Yap Score (0-100)
    # --------------------
    # Formerly "Narrative Score"
    # Controversy (1-10) -> x10 -> 0-100
    # Narrative (1-10) -> x10 -> 0-100
    controversy = float(arc.get("controversy_score", 0)) * 10.0
    interest = float(arc.get("narrative_score", 0)) * 10.0
    
    # Bonus for "drama" keywords in summary
    keyword_bonus = 0.0
    summary_lower = arc.get("summary", "").lower()
    keywords = ["drama", "controversy", "ban", "cheater", "scam", "apology", "response", "opinion", "hated", "failed", "beef", "twitter"]
    if any(k in summary_lower for k in keywords):
        keyword_bonus = 15.0
        
    # Yap Score is mostly Interest + Controversy
    yap_score = min(100.0, (controversy * 0.4) + (interest * 0.6) + keyword_bonus)
    breakdown["yap"] = round(yap_score, 2)
    
    # TOTAL SCORE
    # -----------
    # We pick the clip if EITHER score is high.
    # So the representative rating is just the MAX of the two.
    total_rating = max(hype_score, yap_score)
    
    return round(total_rating, 2), breakdown


def rate_and_rank_arcs(arcs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rate all arcs and add rating info, sorted by rating descending."""
    rated_arcs = []
    
    for arc in arcs:
        rating, breakdown = compute_arc_rating(arc)
        arc_copy = dict(arc)
        arc_copy["_rating"] = rating
        arc_copy["_rating_breakdown"] = breakdown
        rated_arcs.append(arc_copy)
    
    # Sort by rating descending
    rated_arcs.sort(key=lambda a: a["_rating"], reverse=True)
    
    return rated_arcs


def select_best_arcs(
    arcs: List[Dict[str, Any]],
    top_k: int = 0,
    min_rating: float = 0.0,
    min_duration: float = 300.0,
    max_duration: float = 2400.0,
) -> List[Dict[str, Any]]:
    """
    Select the best arcs based on rating.
    
    Args:
        arcs: List of arcs (already rated)
        top_k: Maximum number of arcs to select (0 = dynamic based on VOD length)
        min_rating: Minimum rating threshold (0-100)
        min_duration: Minimum arc duration in seconds
        max_duration: Maximum arc duration in seconds
    
    Returns:
        Selected arcs (best first)
    """
    selected = []
    
    for arc in arcs:
        rating = arc.get("_rating", 0)
        duration = float(arc.get("duration", 0))
        
        # Duration filter
        if duration < min_duration:
            print(f"  ⏭️  Arc {arc.get('arc_id')}: skipped (duration {_format_duration_display(duration)} < {_format_duration_display(min_duration)})")
            continue
        if duration > max_duration:
            print(f"  ⏭️  Arc {arc.get('arc_id')}: skipped (duration {_format_duration_display(duration)} > {_format_duration_display(max_duration)})")
            continue
        
        # Rating filter
        if rating < min_rating:
            print(f"  ⏭️  Arc {arc.get('arc_id')}: skipped (rating {rating:.1f} < {min_rating})")
            continue
        
        selected.append(arc)
        
        # Top-k limit
        if top_k > 0 and len(selected) >= top_k:
            break
    
    return selected


def filter_arcs(
    arcs: List[Dict[str, Any]],
    min_duration: float = 300.0,
    max_duration: float = 2400.0,
    min_confidence: float = 0.0,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """Legacy filter function - kept for backwards compatibility."""
    filtered = []
    
    for arc in arcs:
        duration = float(arc.get("duration", 0))
        confidence = float(arc.get("confidence", 0))
        climax = arc.get("climax", {})
        score = float(climax.get("score", 0)) if climax else 0.0
        
        if duration < min_duration:
            continue
        if duration > max_duration:
            continue
        if confidence < min_confidence:
            continue
        if score < min_score:
            continue
        
        filtered.append(arc)
    
    return filtered


def create_arc_manifest(arc: Dict[str, Any], arc_index: int) -> Dict[str, Any]:
    """Convert a Gemini arc to the format expected by create_arch_videos.py."""
    start = float(arc.get("start", 0))
    end = float(arc.get("end", 0))
    duration = end - start
    
    # Build ranges - for now, single continuous range per arc
    ranges = [{
        "start": start,
        "end": end,
        "duration": duration,
    }]
    
    # Extract metadata for the manifest
    manifest = {
        "arc_index": arc_index,
        "vod_id": arc.get("vod_id", ""),
        "start_abs": start,
        "end_abs": end,
        "start_hms": _format_hms(start),
        "end_hms": _format_hms(end),
        "duration": duration,
        "duration_display": _format_duration_display(duration),
        "ranges": ranges,
        
        # Gemini arc metadata (preserved for downstream use)
        "arc_type": arc.get("arc_type", "unknown"),
        "chapter": arc.get("chapter", ""),
        "confidence": arc.get("confidence", 0.0),
        "summary": arc.get("summary", ""),
        "controversy_score": arc.get("controversy_score", 0),
        "narrative_score": arc.get("narrative_score", 0),
        
        # Rating info (from dynamic rating system)
        "rating": arc.get("_rating", 0.0),
        "rating_breakdown": arc.get("_rating_breakdown", {}),
        
        # Structured segments
        "intro": arc.get("intro", {}),
        "climax": arc.get("climax", {}),
        "resolution": arc.get("resolution", {}),
        
        # Source tracking
        "source": "gemini_arc_detection",
    }
    
    return manifest


def write_arc_manifests(
    vod_id: str,
    arcs: List[Dict[str, Any]],
    output_dir: Optional[Path] = None,
) -> tuple[Path, List[Path]]:
    """Write individual arc manifests and index."""
    if output_dir is None:
        output_dir = Path(f"data/vector_stores/{vod_id}/arcs")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    arc_paths: List[Path] = []
    index_entries: List[Dict[str, Any]] = []
    
    for i, arc in enumerate(arcs):
        # Add vod_id to arc for manifest creation
        arc["vod_id"] = vod_id
        
        # Create manifest
        manifest = create_arc_manifest(arc, arc_index=i)
        
        # Write individual manifest
        arc_path = output_dir / f"arc_{i:03d}_manifest.json"
        arc_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        arc_paths.append(arc_path)
        
        # Build index entry
        rating = manifest.get("rating", 0)
        index_entries.append({
            "arc_index": i,
            "start": manifest["start_abs"],
            "end": manifest["end_abs"],
            "start_hms": manifest["start_hms"],
            "end_hms": manifest["end_hms"],
            "duration": manifest["duration"],
            "duration_display": manifest["duration_display"],
            "arc_type": manifest["arc_type"],
            "chapter": manifest["chapter"],
            "confidence": manifest["confidence"],
            "rating": rating,
            "summary": manifest["summary"][:100] + "..." if len(manifest.get("summary", "")) > 100 else manifest.get("summary", ""),
            "manifest_path": str(arc_path.name),
        })
        
        print(f"  ✅ Arc {i:03d}: {manifest['start_hms']} → {manifest['end_hms']} ({manifest['duration_display']}) [rating: {rating:.1f}] - {manifest['arc_type']}")
    
    # Write arcs index
    index_data = {
        "vod_id": vod_id,
        "num_arcs": len(arcs),
        "total_duration": sum(a["duration"] for a in index_entries),
        "total_duration_display": _format_duration_display(sum(a["duration"] for a in index_entries)),
        "source": "gemini_arc_detection",
        "arcs": index_entries,
    }
    
    index_path = output_dir / "arcs_index.json"
    index_path.write_text(json.dumps(index_data, indent=2), encoding="utf-8")
    
    # Write VOD times summary (for quick verification)
    times_path = output_dir / "arcs_vod_times.txt"
    lines = [f"VOD {vod_id} - {len(arcs)} arcs ({_format_duration_display(index_data['total_duration'])} total)", ""]
    for entry in index_entries:
        lines.append(f"Arc {entry['arc_index']:3d}: {entry['start_hms']} -> {entry['end_hms']} ({entry['duration_display']:>10}) [{entry['arc_type']}]")
    times_path.write_text("\n".join(lines), encoding="utf-8")
    
    return index_path, arc_paths


def convert_gemini_to_arc_manifests(
    vod_id: str,
    manifest_path: Optional[str] = None,
    top_k: int = 0,
    min_rating: float = 0.0,
    min_duration: float = 300.0,
    max_duration: float = 2400.0,
) -> tuple[Path, List[Path]]:
    """Main conversion function with dynamic arc rating and selection."""
    print(f"🎬 Converting Gemini arcs to video manifests for VOD {vod_id}")
    
    # Load Gemini manifest
    data = load_gemini_arc_manifest(vod_id, manifest_path)
    arcs = data.get("arcs", [])
    
    if not arcs:
        raise ValueError("No arcs found in Gemini manifest")
    
    print(f"   Found {len(arcs)} arcs in Gemini manifest")
    
    # Rate and rank all arcs
    print("\n⭐ Rating arcs...")
    rated_arcs = rate_and_rank_arcs(arcs)
    
    # Show all ratings
    print("\n   Arc Ratings (sorted by quality):")
    for arc in rated_arcs:
        rating = arc.get("_rating", 0)
        breakdown = arc.get("_rating_breakdown", {})
        print(f"     Arc {arc.get('arc_id'):2d}: {rating:5.1f}/100  "
              f"[hype:{breakdown.get('hype', 0):4.1f} yap:{breakdown.get('yap', 0):4.1f}]  "
              f"({_format_duration_display(arc.get('duration', 0))}) - {arc.get('arc_type', 'unknown')}")
    
    # Select best arcs
    print(f"\n📋 Selecting best arcs (top_k={top_k}, min_rating={min_rating}, "
          f"duration: {_format_duration_display(min_duration)}-{_format_duration_display(max_duration)})...")
    
    selected = select_best_arcs(
        rated_arcs,
        top_k=top_k,
        min_rating=min_rating,
        min_duration=min_duration,
        max_duration=max_duration,
    )
    
    if not selected:
        raise ValueError("No arcs passed selection criteria")
    
    print(f"\n   {len(selected)} arcs selected")
    
    # Write manifests
    print("\n📝 Writing arc manifests...")
    index_path, arc_paths = write_arc_manifests(vod_id, selected)
    
    print(f"\n✅ Done! Created {len(arc_paths)} arc manifests")
    print(f"   Index: {index_path}")
    print(f"   Arcs:  {arc_paths[0].parent}")
    
    return index_path, arc_paths


def main():
    parser = argparse.ArgumentParser(
        description="Convert Gemini arc manifest to individual arc manifests for video rendering"
    )
    parser.add_argument("vod_id", help="VOD ID to process")
    parser.add_argument("--manifest", default=None, help="Path to Gemini arc manifest (default: auto-detect)")
    parser.add_argument("--top-k", type=int, default=0, help="Max number of arcs to select (0 = all that pass filters)")
    parser.add_argument("--min-rating", type=float, default=0.0, help="Minimum arc rating 0-100 (default: 0 = no filter)")
    parser.add_argument("--min-duration", type=float, default=300.0, help="Minimum arc duration in seconds (default: 300 = 5 min)")
    parser.add_argument("--max-duration", type=float, default=2400.0, help="Maximum arc duration in seconds (default: 2400 = 40 min)")
    
    args = parser.parse_args()
    
    try:
        convert_gemini_to_arc_manifests(
            vod_id=args.vod_id,
            manifest_path=args.manifest,
            top_k=args.top_k,
            min_rating=args.min_rating,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        raise SystemExit(1)
    except ValueError as e:
        print(f"❌ Error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

