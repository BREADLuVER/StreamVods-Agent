#!/usr/bin/env python3
"""
Speech-coherent window detection system.

Groups consecutive transcript segments until inter-segment gaps exceed
a pause threshold derived from gap histogram analysis.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Window:
    """Represents a speech/narrative window plus burst metrics."""
    start_time: float
    end_time: float
    duration: float
    segments: List[Dict]
    pause_threshold: float

    # --- Burst analytics (optional, filled later) ---
    chat_rate: float = 0.0          # msgs per second in window
    chat_rate_z: float = 0.0        # z-score vs chapter baseline
    burst_score: float = 0.0        # prominence vs neighbours
    reaction_hits: Dict[str, int] = None  # gg/ez/o7 counts
    section_context: str = ""       # short text for LLM context

    def __post_init__(self):
        if self.reaction_hits is None:
            self.reaction_hits = {}


def calculate_gaps(segments: List[Dict]) -> List[float]:
    """Calculate gaps between consecutive segments."""
    gaps = []
    for i in range(len(segments) - 1):
        current_end = segments[i].get('end_time', segments[i].get('start_time', 0) + segments[i].get('duration', 0))
        next_start = segments[i + 1].get('start_time', 0)
        gap = next_start - current_end
        gaps.append(gap)
    return gaps


def find_knee_point(values: List[float]) -> float:
    """
    Find knee point in sorted values using the knee detection algorithm.
    This identifies the point where the curve changes most dramatically.
    """
    if len(values) < 3:
        return values[-1] if values else 0.0
    
    # Sort values
    sorted_values = sorted(values)
    
    # Calculate first and second derivatives
    first_deriv = np.diff(sorted_values)
    second_deriv = np.diff(first_deriv)
    
    # Find the point with maximum second derivative (knee point)
    if len(second_deriv) > 0:
        knee_idx = np.argmax(second_deriv) + 2  # +2 because of double diff
        if knee_idx < len(sorted_values):
            return sorted_values[knee_idx]
    
    # Fallback: use 75th percentile
    return np.percentile(sorted_values, 75)


def find_otsu_threshold(values: List[float]) -> float:
    """
    Find threshold using Otsu's method for bimodal distribution.
    Adapted for gap analysis.
    """
    if len(values) < 2:
        return values[0] if values else 0.0
    
    # Create histogram
    hist, bin_edges = np.histogram(values, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate probabilities
    total = len(values)
    prob = hist / total
    
    # Otsu's method
    best_threshold = bin_centers[0]
    best_variance = 0
    
    for i in range(1, len(hist)):
        # Class probabilities
        w0 = np.sum(prob[:i])
        w1 = np.sum(prob[i:])
        
        if w0 == 0 or w1 == 0:
            continue
        
        # Class means
        mu0 = np.sum(prob[:i] * bin_centers[:i]) / w0
        mu1 = np.sum(prob[i:] * bin_centers[i:]) / w1
        
        # Between-class variance
        variance = w0 * w1 * (mu0 - mu1) ** 2
        
        if variance > best_variance:
            best_variance = variance
            best_threshold = bin_centers[i]
    
    return best_threshold


def determine_pause_threshold(gaps: List[float]) -> float:
    """
    Determine pause threshold using data-driven methods.
    Tries knee detection first, falls back to Otsu, then percentiles.
    """
    if not gaps:
        return 2.0  # Default 2 second threshold
    
    # Remove negative gaps (overlapping segments)
    positive_gaps = [g for g in gaps if g > 0]
    
    if not positive_gaps:
        return 2.0
    
    # Try knee detection
    try:
        knee_threshold = find_knee_point(positive_gaps)
        logger.info(f"Knee threshold: {knee_threshold:.2f}s")
    except Exception as e:
        logger.warning(f"Knee detection failed: {e}")
        knee_threshold = None
    
    # Try Otsu's method
    try:
        otsu_threshold = find_otsu_threshold(positive_gaps)
        logger.info(f"Otsu threshold: {otsu_threshold:.2f}s")
    except Exception as e:
        logger.warning(f"Otsu detection failed: {e}")
        otsu_threshold = None
    
    # Fallback to percentiles
    percentile_75 = np.percentile(positive_gaps, 75)
    percentile_90 = np.percentile(positive_gaps, 90)
    
    # Choose the most reasonable threshold
    candidates = [t for t in [knee_threshold, otsu_threshold, percentile_75] if t is not None]
    
    if candidates:
        # Use the median of available methods
        chosen_threshold = np.median(candidates)
    else:
        chosen_threshold = percentile_90
    
    # Ensure reasonable bounds
    chosen_threshold = max(0.5, min(chosen_threshold, 10.0))
    
    logger.info(f"Chosen pause threshold: {chosen_threshold:.2f}s")
    logger.info(f"Gap statistics: min={min(positive_gaps):.2f}s, "
                f"max={max(positive_gaps):.2f}s, "
                f"median={np.median(positive_gaps):.2f}s")
    
    return chosen_threshold


def create_speech_windows(segments: List[Dict], pause_threshold: Optional[float] = None) -> List[Window]:
    """
    Create speech-coherent windows from segments.
    
    Args:
        segments: List of transcript segments
        pause_threshold: Override threshold (if None, will be calculated)
    
    Returns:
        List of Window objects
    """
    if not segments:
        return []
    
    # Calculate gaps and determine threshold
    gaps = calculate_gaps(segments)
    if pause_threshold is None:
        pause_threshold = determine_pause_threshold(gaps)
    
    windows = []
    current_window_segments = [segments[0]]
    
    for i in range(1, len(segments)):
        current_segment = segments[i]
        previous_segment = segments[i - 1]
        
        # Calculate gap
        prev_end = previous_segment.get('end_time', 
                                      previous_segment.get('start_time', 0) + 
                                      previous_segment.get('duration', 0))
        curr_start = current_segment.get('start_time', 0)
        gap = curr_start - prev_end
        
        # If gap is small, add to current window
        if gap <= pause_threshold:
            current_window_segments.append(current_segment)
        else:
            # Gap is large, finalize current window and start new one
            window = create_window_from_segments(current_window_segments, pause_threshold)
            windows.append(window)
            current_window_segments = [current_segment]
    
    # Add final window
    if current_window_segments:
        window = create_window_from_segments(current_window_segments, pause_threshold)
        windows.append(window)
    
    logger.info(f"Created {len(windows)} speech-coherent windows")
    logger.info(f"Window duration stats: "
                f"min={min(w.duration for w in windows):.1f}s, "
                f"max={max(w.duration for w in windows):.1f}s, "
                f"median={np.median([w.duration for w in windows]):.1f}s")
    
    return windows


def create_window_from_segments(segments: List[Dict], pause_threshold: float) -> Window:
    """Create a Window object from a list of segments."""
    if not segments:
        raise ValueError("Cannot create window from empty segments")
    
    start_time = segments[0].get('start_time', 0)
    last_segment = segments[-1]
    end_time = last_segment.get('end_time', 
                               last_segment.get('start_time', 0) + 
                               last_segment.get('duration', 0))
    duration = end_time - start_time
    
    return Window(
        start_time=start_time,
        end_time=end_time,
        duration=duration,
        segments=segments,
        pause_threshold=pause_threshold
    )


def split_window_at_boundaries(window, boundaries: List[float]) -> List:
    """
    Split a window at chapter boundaries.
    
    Args:
        window: Window object to split
        boundaries: List of boundary timestamps
    
    Returns:
        List of split window objects
    """
    if not boundaries:
        return [window]
    
    # Find boundaries within this window
    relevant_boundaries = [b for b in boundaries if window.start_time < b < window.end_time]
    
    if not relevant_boundaries:
        return [window]
    
    # Sort boundaries and add window endpoints
    split_points = sorted([window.start_time] + relevant_boundaries + [window.end_time])
    
    split_windows = []
    for i in range(len(split_points) - 1):
        start = split_points[i]
        end = split_points[i + 1]
        
        # Find segments that fall within this split
        split_segments = []
        for segment in window.segments:
            seg_start = segment.get('start_time', 0)
            seg_end = segment.get('end_time', seg_start + segment.get('duration', 0))
            
            # Check if segment overlaps with this split
            if seg_start < end and seg_end > start:
                split_segments.append(segment)
        
        if split_segments:
            split_window = Window(
                start_time=start,
                end_time=end,
                duration=end - start,
                segments=split_segments,
                pause_threshold=window.pause_threshold
            )
            split_windows.append(split_window)
    
    return split_windows


def split_windows_at_chapter_boundaries(windows: List, chapters: List) -> List:
    """
    Split windows at chapter boundaries to prevent cross-chapter windows.
    
    Args:
        windows: List of Window objects
        chapters: List of ChapterInfo objects
    
    Returns:
        List of split windows
    """
    if not chapters:
        return windows
    
    # Extract all chapter boundaries
    boundaries = set()
    for chapter in chapters:
        boundaries.add(chapter.start_time)
        boundaries.add(chapter.end_time)
    
    boundaries = sorted(boundaries)
    
    split_windows = []
    for window in windows:
        split_parts = split_window_at_boundaries(window, boundaries)
        split_windows.extend(split_parts)
    
    logger.info(f"Split {len(windows)} windows into {len(split_windows)} windows at chapter boundaries")
    return split_windows


def apply_chat_latency_correction(segments: List[Dict], lag_seconds: float = 5.0) -> List[Dict]:
    """
    Shift all chat timestamps earlier by the specified lag.
    
    Args:
        segments: List of segments with chat_messages
        lag_seconds: How many seconds to shift chat earlier (default 5.0)
    
    Returns:
        Segments with corrected chat timestamps
    """
    corrected_segments = []
    
    for segment in segments:
        corrected_segment = segment.copy()
        chat_messages = segment.get('chat_messages', [])
        
        corrected_chat = []
        for msg in chat_messages:
            corrected_msg = msg.copy()
            original_time = msg.get('timestamp', 0)
            corrected_msg['timestamp'] = max(0, original_time - lag_seconds)
            corrected_chat.append(corrected_msg)
        
        corrected_segment['chat_messages'] = corrected_chat
        corrected_segments.append(corrected_segment)
    
    logger.info(f"Applied {lag_seconds}s chat latency correction to {len(segments)} segments")
    return corrected_segments
