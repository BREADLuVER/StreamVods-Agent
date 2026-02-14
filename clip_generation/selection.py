"""
Selection and deduplication functions for clip generation.
"""

from typing import List, Tuple

from .types import ClipCandidate, FinalClip
from .config import ClipConfig


def time_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Compute intersection over union for time intervals."""
    a0, a1 = a
    b0, b1 = b
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    if inter <= 0:
        return 0.0
    union = (a1 - a0) + (b1 - b0) - inter
    if union <= 0:
        return 0.0
    return inter / union


def deduplicate_and_select(
    candidates: List[ClipCandidate], 
    top_k: int = 0, 
    iou_thr: float = 0.5, 
    min_spacing: float = 50.0
) -> List[ClipCandidate]:
    """Deduplicate candidates and select top-K by score."""
    if not candidates:
        return []
    
    # Sort by score descending
    cands = sorted(candidates, key=lambda c: c.score, reverse=True)
    selected: List[ClipCandidate] = []
    
    for c in cands:
        keep = True
        for s in selected:
            # Check IoU overlap
            if time_iou((c.start, c.end), (s.start, s.end)) >= iou_thr:
                keep = False
                break
            # Check center spacing
            center_gap = abs(((c.start + c.end) * 0.5) - ((s.start + s.end) * 0.5))
            if center_gap < min_spacing:
                keep = False
                break
        
        if keep:
            selected.append(c)
            if top_k and len(selected) >= top_k:
                break
    
    return selected


def append_sequence_numbers_to_adjacent_titles(
    clips: List[FinalClip], 
    proximity_s: float = 120.0
) -> List[FinalClip]:
    """Add sequence numbers to adjacent clips with similar titles."""
    if not clips:
        return clips
    
    # Work on a start-sorted copy to find adjacent groups by time
    ordered = sorted(clips, key=lambda c: (c.start, c.end))
    groups: List[List[int]] = []  # indices into ordered
    cur: List[int] = [0]
    
    for i in range(1, len(ordered)):
        prev = ordered[i-1]
        cur_end = prev.end
        ns = ordered[i].start
        # Group if overlapping or within proximity_s gap
        if ns <= cur_end + proximity_s:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)

    # Apply numbering only to groups with multiple clips
    for g in groups:
        if len(g) <= 1:
            continue
        # Use the first title in the group as the base; keep first unchanged, later clips get base + index
        base_title = str(ordered[g[0]].title or "Clip").strip()
        for idx_in_group, pos in enumerate(g, start=1):
            if idx_in_group == 1:
                ordered[pos].title = base_title
            else:
                ordered[pos].title = f"{base_title} {idx_in_group}"

    return ordered
