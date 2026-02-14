"""
Windowing functions for dynamic padding and boundary snapping.
"""

import bisect
from typing import List, Tuple

from .types import WindowDoc
from .config import ClipConfig


def unitize(values: List[float]) -> List[float]:
    """Normalize values to 0-1 range."""
    good = [v for v in values if isinstance(v, (int, float))]
    if not good:
        return [0.0 for _ in values]
    vmin, vmax = min(good), max(good)
    span = (vmax - vmin) if (vmax - vmin) > 1e-9 else 1.0
    return [(max(0.0, v - vmin) / span) if isinstance(v, (int, float)) else 0.0 for v in values]


def moving_average(xs: List[float], window: int = 5) -> List[float]:
    """Compute moving average of values."""
    n = len(xs)
    if n == 0:
        return []
    out = []
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        if lo >= hi:
            out.append(xs[i])
        else:
            out.append(sum(xs[lo:hi]) / max(1, (hi - lo)))
    return out


def clamp_window(start: float, end: float, min_dur: float = 30.0, max_dur: float = 180.0) -> Tuple[float, float]:
    """Clamp a [start,end] window to within duration bounds."""
    if end <= start:
        end = start + min_dur
    dur = end - start
    if dur < min_dur:
        end = start + min_dur
    elif dur > max_dur:
        end = start + max_dur
    return (start, end)


def smoothed_positive_chat(docs: List[WindowDoc], chapter_id: str) -> Tuple[List[int], List[float]]:
    """Get smoothed positive chat rate for a chapter."""
    idxs = [i for i, d in enumerate(docs) if d.chapter_id == chapter_id]
    series = [max(0.0, docs[i].chat_rate_z) for i in idxs]
    return idxs, moving_average(series, window=3)


def apply_dynamic_padding(docs: List[WindowDoc], chapter_id: str, start_guess: float, end_guess: float) -> Tuple[float, float]:
    """Apply dynamic padding based on smoothed chat signals."""
    idxs, smooth = smoothed_positive_chat(docs, chapter_id)
    # Map time -> index
    times = [docs[i].start for i in idxs]
    if not times:
        return (start_guess, end_guess)
    
    # Find nearest positions
    def nearest_pos(t: float) -> int:
        pos = bisect.bisect_left(times, t)
        if pos <= 0:
            return 0
        if pos >= len(times):
            return len(times) - 1
        if abs(times[pos] - t) < abs(times[pos - 1] - t):
            return pos
        return pos - 1

    left_pos = nearest_pos(start_guess)
    right_pos = nearest_pos(end_guess)
    local_max = max(smooth[left_pos: min(len(smooth), right_pos + 10)] or [0.0])
    if local_max <= 0:
        return (start_guess, end_guess)

    # Pre-padding: extend further back to compensate for chat latency
    target_left = 0.45 * local_max
    t = start_guess
    while t - start_guess > -90.0:  # allow up to 90s earlier
        pos = nearest_pos(t)
        if smooth[pos] <= target_left:
            break
        t -= 5.0
    # Ensure at least 10s minimum pre-pad even if threshold didn't drop
    padded_start = min(start_guess - 10.0, t)
    # Do not exceed 120s pre-pad cap
    padded_start = max(start_guess - 120.0, padded_start)

    # Post-padding: slightly earlier cutoff
    target_right = 0.35 * local_max
    t = end_guess
    while t - end_guess < 45.0:
        pos = nearest_pos(t)
        if smooth[pos] <= target_right:
            break
        t += 5.0
    padded_end = min(t, end_guess + 45.0)

    # Clamp duration (global bounds 30..180s)
    padded_start, padded_end = clamp_window(padded_start, padded_end, min_dur=30.0, max_dur=180.0)
    return (padded_start, padded_end)


def snap_to_transcript_boundaries(
    vstart: float,
    vend: float,
    ctx_docs: List[WindowDoc],
    win_start: float,
    win_end: float,
    anchor_center: float,
) -> Tuple[float, float]:
    """Snap start/end to nearest transcript boundaries while preserving anchor and constraints."""
    if not ctx_docs:
        return (vstart, vend)
    
    boundaries = sorted({float(d.start) for d in ctx_docs}.union({float(d.end) for d in ctx_docs}))
    boundaries = [b for b in boundaries if (win_start <= b <= win_end)]
    if not boundaries:
        return (vstart, vend)

    # Choose start boundary
    start_candidates = [b for b in boundaries if b <= vstart]
    if start_candidates:
        new_start = max(start_candidates)
    else:
        new_start = min(boundaries)

    # Choose end boundary
    end_candidates = [b for b in boundaries if b >= vend]
    if end_candidates:
        new_end = min(end_candidates)
    else:
        new_end = max(boundaries)

    # Keep anchor inside
    if anchor_center < new_start:
        left_candidates = [b for b in boundaries if b <= anchor_center]
        if left_candidates:
            new_start = max(left_candidates)
    if anchor_center > new_end:
        right_candidates = [b for b in boundaries if b >= anchor_center]
        if right_candidates:
            new_end = min(right_candidates)

    # Enforce window containment
    new_start = max(win_start, min(new_start, win_end))
    new_end = max(win_start, min(new_end, win_end))
    if new_end <= new_start:
        # fallback to original if invalid
        new_start, new_end = vstart, vend

    # Enforce duration bounds
    dur = new_end - new_start
    if dur < 30.0:
        # try to extend right first
        extend = 30.0 - dur
        new_end = min(win_end, new_end + extend)
        if (new_end - new_start) < 30.0:
            new_start = max(win_start, new_start - (30.0 - (new_end - new_start)))
    elif dur > 180.0:
        new_end = min(win_end, new_start + 180.0)

    # Final anchor check; if still outside due to coarse boundaries, fall back
    if not (new_start <= anchor_center <= new_end):
        return (vstart, vend)

    return (new_start, new_end)


def left_pad_to_sentence_start(
    vstart: float,
    vend: float,
    ctx_docs: List[WindowDoc],
    win_start: float,
    max_left_pad: float = 30.0,
) -> Tuple[float, float]:
    """Extend start earlier by up to max_left_pad seconds, snapping to a sentence start."""
    boundaries = sorted({float(d.start) for d in ctx_docs}.union({float(d.end) for d in ctx_docs}))
    dur_current = vend - vstart
    if dur_current <= 0.0:
        return (vstart, vend)
    allowed_by_180 = max(0.0, 180.0 - dur_current)
    if allowed_by_180 <= 0.0:
        return (vstart, vend)
    left_pad_limit = min(max_left_pad, allowed_by_180, max(0.0, vstart - win_start))
    if left_pad_limit <= 0.0:
        return (vstart, vend)
    target_start = vstart - left_pad_limit
    # pick boundary within [win_start, vstart] closest to target_start
    candidates = [b for b in boundaries if (win_start <= b <= vstart)]
    if candidates:
        new_start = min(candidates, key=lambda b: abs(b - target_start))
        # ensure we don't exceed left_pad_limit due to boundary rounding
        if (vstart - new_start) > left_pad_limit:
            new_start = max(win_start, vstart - left_pad_limit)
    else:
        new_start = max(win_start, target_start)
    # no change to vend
    return (new_start, vend)


def apply_final_padding(
    vstart: float,
    vend: float,
    front_pad_s: float,
    back_pad_s: float,
    hard_cap: float = 179.0,
) -> Tuple[float, float]:
    """Apply final front and back padding, respecting hard cap."""
    new_start = vstart - front_pad_s
    new_end = vend + back_pad_s
    
    # Apply hard cap
    current_dur = new_end - new_start
    if current_dur > hard_cap:
        # Trim from the right preferentially
        new_end = new_start + hard_cap
    
    return (new_start, new_end)


def bounds_of_indices(docs: List[WindowDoc], idxs: List[int]) -> Tuple[float, float]:
    """Get start and end bounds for a list of document indices."""
    if not idxs:
        return (0.0, 0.0)
    s = min(docs[i].start for i in idxs)
    e = max(docs[i].end for i in idxs)
    return (s, e)
