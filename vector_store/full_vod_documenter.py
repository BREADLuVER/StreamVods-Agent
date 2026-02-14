#!/usr/bin/env python3
"""
Full VOD Documenter - Documents the entire VOD timeline, not just segments.

This creates a complete timeline representation of the VOD for both
director's cut (full coverage) and highlights (filtered sections).
"""

import sys
import re
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vector_store.window_detector import Window, apply_chat_latency_correction
from vector_store.chapter_metadata import load_chapters_merged, ChapterInfo
from vector_store.document_builder import create_document_from_window
from vector_store.vector_index import VectorIndex
from vector_store.query_system import QuerySystem

logger = logging.getLogger(__name__)


def create_full_vod_timeline(
    vod_id: str,
    segments: List[Dict],
    chapters: List[ChapterInfo],
    *,
    min_window_duration: float = 10.0,
    group_contiguous: bool = True,
) -> List[Window]:
    """Build timeline windows based on *narrative* segment boundaries.

    The previous version used *fixed-size* 30 s windows.  That produced
    arbitrary cuts that ignored how the original transcript is chunked.  We now:

    1.  Sort narrative segments by ``start_time``.
    2.  **Group** consecutive segments whose ``end_time`` exactly matches the next
        ``start_time`` (allowing <0.01 s slack) so we treat uninterrupted
        runs as one window.
    3.  **Fuse short bursts** (``duration < min_window_duration``) with the
        previous window if possible, else with the next, to avoid very tiny
        clips that break narrative flow.

    Args:
        vod_id:           VOD identifier.
        segments:         Raw transcript segments loaded from *_ai_data.json*.
        chapters:         Chapter metadata (unused here but kept for parity).
        min_window_duration:  Minimum length (seconds) for a window after
                              grouping.  Windows shorter than this are
                              merged into neighbours.
        group_contiguous: If *False* we fall back to legacy 30 s slicing.

    Returns:
        List[Window] describing the full content timeline.
    """

    if not segments:
        return []

    # ------------------------------------------------------------------
    # Legacy fallback – keep a 30 s grid if caller disables grouping.
    # ------------------------------------------------------------------
    if not group_contiguous:
        return _create_fixed_windows(segments, window_size_seconds=30.0)

    # ------------------------------------------------------------------
    # 1) Sort segments by start time for deterministic processing.
    # ------------------------------------------------------------------
    segments = sorted(segments, key=lambda s: s.get("start_time", 0.0))

    windows: List[Window] = []
    current_group: List[Dict] = []

    def _flush_group(group: List[Dict]):
        """Helper – push current_group into *windows*."""
        if not group:
            return
        start = group[0]["start_time"]
        end = group[-1].get("end_time", group[-1]["start_time"] + group[-1].get("duration", 0))
        windows.append(
            Window(
                start_time=start,
                end_time=end,
                duration=end - start,
                segments=list(group),
                pause_threshold=0.0,  # not used in grouped mode
            )
        )

    # ------------------------------------------------------------------
    # 2) Group contiguous narrative segments.
    # ------------------------------------------------------------------
    SLACK = 1e-2  # 10 ms tolerance for float rounding
    for seg in segments:
        if not current_group:
            current_group.append(seg)
            continue

        prev_end = current_group[-1].get("end_time", current_group[-1]["start_time"] + current_group[-1].get("duration", 0))
        if abs(prev_end - seg["start_time"]) < SLACK:
            # Seamless continuation – same narrative burst.
            current_group.append(seg)
        else:
            # Gap → flush existing group, start new one.
            _flush_group(current_group)
            current_group = [seg]

    _flush_group(current_group)

    if not windows:
        logger.warning("Grouping produced zero windows – falling back to fixed-size windows.")
        return _create_fixed_windows(segments, window_size_seconds=30.0)

    # ------------------------------------------------------------------
    # 3) Merge very short windows with neighbours.
    # ------------------------------------------------------------------
    merged: List[Window] = []
    for w in windows:
        if w.duration >= min_window_duration or not merged:
            merged.append(w)
            continue

        # Too short – attach to previous window.
        prev = merged[-1]
        prev.end_time = w.end_time
        prev.duration = prev.end_time - prev.start_time
        prev.segments.extend(w.segments)

    logger.info(
        "Created %d narrative windows (after merging <%.1fs bursts) covering %.1f minutes",
        len(merged),
        min_window_duration,
        (merged[-1].end_time - merged[0].start_time) / 60 if merged else 0,
    )

    return merged


# ----------------------------------------------------------------------
# Helper – legacy fixed-size timeline creation (kept for optional use)
# ----------------------------------------------------------------------

def _create_fixed_windows(segments: List[Dict], window_size_seconds: float = 30.0) -> List[Window]:
    """Create fixed-size windows that span from first to last segment."""
    vod_start = min(seg.get("start_time", 0) for seg in segments)
    vod_end = max(seg.get("end_time", seg.get("start_time", 0) + seg.get("duration", 0)) for seg in segments)

    windows: List[Window] = []
    current_time = vod_start
    while current_time < vod_end:
        window_end = min(current_time + window_size_seconds, vod_end)
        window_segments = [
            seg
            for seg in segments
            if seg.get("start_time", 0) < window_end and seg.get("end_time", seg.get("start_time", 0) + seg.get("duration", 0)) > current_time
        ]

        windows.append(
            Window(
                start_time=current_time,
                end_time=window_end,
                duration=window_end - current_time,
                segments=window_segments,
                pause_threshold=window_size_seconds,
            )
        )
        current_time = window_end

    return windows


def attach_chapter_metadata_to_timeline(windows: List[Window], chapters: List[ChapterInfo]) -> List[Window]:
    """
    Attach chapter metadata to timeline windows.
    
    Args:
        windows: List of timeline windows
        chapters: List of chapter information
    
    Returns:
        Windows with chapter metadata attached
    """
    if not chapters:
        # No chapters available, assign default metadata
        for window in windows:
            window.chapter_id = None
            window.category = None
            window.excluded = False
        return windows
    
    for window in windows:
        # Find enclosing chapter
        chapter = None
        for ch in chapters:
            if ch.start_time <= window.start_time < ch.end_time:
                chapter = ch
                break
        
        if chapter:
            window.chapter_id = chapter.id
            window.category = chapter.category
            window.excluded = chapter.excluded
        else:
            # No chapter found, assign defaults
            window.chapter_id = None
            window.category = None
            window.excluded = False
    
    logger.info(f"Attached chapter metadata to {len(windows)} timeline windows")
    return windows


def assign_mode_to_timeline_windows(windows: List[Window]) -> List[Window]:
    """
    Assign mode to timeline windows based on chapter category.
    
    Args:
        windows: List of timeline windows
    
    Returns:
        Windows with mode assigned
    """
    for window in windows:
        # Use Twitch's category as truth
        if window.category:
            category_lower = window.category.strip().lower()
            window.mode = 'jc' if ('just_chatting' in category_lower or 'just chatting' in category_lower) else 'game'
        else:
            window.mode = 'unknown'
    
    # Log mode distribution
    mode_counts = {}
    for window in windows:
        mode_counts[window.mode] = mode_counts.get(window.mode, 0) + 1
    
    logger.info(f"Timeline mode distribution: {mode_counts}")
    return windows


def build_full_vod_vector_store(vod_id: str, index_path: Optional[str] = None,
                               embedding_model: str = "all-MiniLM-L6-v2",
                               window_size_seconds: float = 30.0,
                               chat_lag_seconds: float = 5.0) -> QuerySystem:
    """
    Build a complete VOD vector store covering the entire timeline.
    
    This creates a database for both director's cut (full coverage) and highlights.
    
    Args:
        vod_id: VOD identifier
        index_path: Path to store vector index
        embedding_model: Sentence transformer model name
        window_size_seconds: Size of each timeline window
        chat_lag_seconds: Chat latency correction in seconds
    
    Returns:
        QuerySystem instance
    """
    logger.info(f"Building FULL VOD vector store for VOD: {vod_id}")
    
    # Set up index path
    if index_path is None:
        index_path = f"data/vector_stores/{vod_id}"
    
    # Remove existing store if it exists
    index_path_obj = Path(index_path)
    if index_path_obj.exists():
        logger.info(f"Removing existing vector store at {index_path}")
        import shutil
        shutil.rmtree(index_path)
    
    # Load input data - try filtered first, then fall back to raw
    logger.info("Loading AI data...")
    ai_data_dir = Path(f"data/ai_data/{vod_id}")
    
    # Try filtered data first
    filtered_path = ai_data_dir / f"{vod_id}_filtered_ai_data.json"
    raw_path = ai_data_dir / f"{vod_id}_ai_data.json"
    
    ai_data_path = None
    data_source = None
    
    if filtered_path.exists():
        ai_data_path = filtered_path
        data_source = "filtered"
    elif raw_path.exists():
        ai_data_path = raw_path
        data_source = "raw"
    else:
        raise FileNotFoundError(f"AI data not found: {filtered_path} or {raw_path}")
    
    import json
    with open(ai_data_path, 'r', encoding='utf-8') as f:
        ai_data = json.load(f)
    
    logger.info(f"📄 AI data source: {data_source} ({ai_data_path.name})")
    
    segments = ai_data.get('segments', [])
    if not segments:
        raise ValueError(f"No segments found in AI data for VOD {vod_id}")
    
    logger.info(f"Loaded AI data for VOD {vod_id}: {len(segments)} segments")
    
    # Apply chat latency correction
    logger.info(f"Applying {chat_lag_seconds}s chat latency correction...")
    segments = apply_chat_latency_correction(segments, chat_lag_seconds)
    
    # Load chapters
    logger.info("Loading chapters (merged)…")
    chapters = load_chapters_merged(vod_id)
    
    # Create full timeline
    logger.info("Creating full VOD timeline...")
    windows = create_full_vod_timeline(vod_id, segments, chapters)
    
    if not windows:
        raise ValueError("No timeline windows created")
    
    # Log timeline statistics
    window_durations = [w.duration for w in windows]
    logger.info(f"Created {len(windows)} timeline windows")
    logger.info(f"Window duration stats: "
                f"min={min(window_durations):.1f}s, "
                f"max={max(window_durations):.1f}s, "
                f"median={sorted(window_durations)[len(window_durations)//2]:.1f}s")
    
    # Attach chapter metadata
    logger.info("Attaching chapter metadata...")
    windows = attach_chapter_metadata_to_timeline(windows, chapters)
    
    # Assign modes
    logger.info("Assigning modes...")
    windows = assign_mode_to_timeline_windows(windows)

    # ------------------------------------------------------------
    # PART A: Capture burst metrics (chat rates, reaction hits, etc.)
    # ------------------------------------------------------------
    from collections import defaultdict

    logger.info("Computing burst metrics for timeline windows…")

    # 1) Build baselines for chat rate (msgs+emote-msgs per second)
    #    - Cap emote weight per message (multi-emote message counts as +1 extra, not N)
    #    - Compute baselines per chapter ONLY; fallback to global if chapter too small
    rates_by_chapter = defaultdict(list)
    global_rates: list = []

    for w in windows:
        total_msgs = 0
        emote_msg_count = 0
        for seg in w.segments:
            chat_messages = seg.get("chat_messages", [])
            total_msgs += len(chat_messages)
            for msg in chat_messages:
                emotes = msg.get("emotes", [])
                # Cap per-message emote weight to 1
                if emotes:
                    emote_msg_count += 1

        total_chat_activity = total_msgs + emote_msg_count
        rate = total_chat_activity / w.duration if w.duration > 0 else 0.0

        rates_by_chapter[w.chapter_id].append(rate)
        global_rates.append(rate)

    # Means/stds for each grouping (sample std, fallback to 1.0)
    import math
    def _mean_std(values: list) -> tuple:
        if not values:
            return (0.0, 1.0)
        m = sum(values) / len(values)
        if len(values) <= 1:
            return (m, 1.0)
        var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
        s = math.sqrt(var) or 1.0
        return (m, s)

    def _robust_mean_std(values: list, exclude_outliers: bool = True) -> tuple:
        """Calculate robust mean/std that excludes outliers to prevent early stream setup from skewing normalization."""
        if not values:
            return (0.0, 1.0)
        if len(values) <= 1:
            return (values[0], 1.0)
        
        if not exclude_outliers:
            return _mean_std(values)
        
        # Sort values to find outliers
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        # Use IQR method to identify outliers
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        q1 = sorted_values[q1_idx]
        q3 = sorted_values[q3_idx]
        iqr = q3 - q1
        
        # Define outlier bounds (1.5 * IQR beyond quartiles)
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Filter out outliers
        filtered_values = [v for v in values if lower_bound <= v <= upper_bound]
        
        # If we filtered out too much (>50%), use original values
        if len(filtered_values) < max(1, len(values) // 2):
            filtered_values = values
        
        return _mean_std(filtered_values)

    def _time_aware_robust_mean_std(windows: list, rates: list, early_cutoff_seconds: float = 1800.0) -> tuple:
        """Calculate robust mean/std that specifically excludes early stream setup content."""
        if not rates:
            return (0.0, 1.0)
        if len(rates) <= 1:
            return (rates[0], 1.0)
        
        # Filter out early stream content (first 30 minutes by default)
        filtered_rates = []
        for i, rate in enumerate(rates):
            if i < len(windows) and windows[i].start_time > early_cutoff_seconds:
                filtered_rates.append(rate)
        
        # If we have enough data after filtering, use it; otherwise fall back to robust method
        if len(filtered_rates) >= max(3, len(rates) // 4):
            return _robust_mean_std(filtered_rates)
        else:
            return _robust_mean_std(rates)

    chapter_stats = {cid: _robust_mean_std(rates) for cid, rates in rates_by_chapter.items()}
    global_mean, global_std = _time_aware_robust_mean_std(windows, global_rates)

    # 2) Comprehensive token patterns for reaction hits
    def get_token_patterns():
        patterns = {
        # === GG / EZ family ===
        "gg": r"(?<![A-Za-z0-9_./:@])[gG]{2,}(?![A-Za-z0-9_./:@])",
        "ggs": r"(?<![A-Za-z0-9_./:@])[gG]{2,}[sS]+(?![A-Za-z0-9_./:@])",
        "good game": r"(?<![A-Za-z0-9_])[gG]ood\s*[gG]ame[sS]*(?![A-Za-z0-9_])",
        "gg no re": r"(?<![A-Za-z0-9_])[gG]{2,}\s*[nN]o\s*[rR]e(?![A-Za-z0-9_])",
        "ggez": r"(?<![A-Za-z0-9_./:@])[gG]{2,}[eE][zZ]+(?![A-Za-z0-9_./:@])",
        "ez": r"(?<![A-Za-z0-9_./:@])[eE][zZ]+(?![A-Za-z0-9_./:@])",

        # === Salute ===
        "o7": r"(?<![A-Za-z0-9_./:@])[oO]7(?![A-Za-z0-9_./:@])",
        "07": r"(?<![A-Za-z0-9_./:@-])07(?![A-Za-z0-9_./:@-])",

        # === FF ===
        "ff": r"(?<![A-Za-z0-9_./:@])[fF]{2,}(?![A-Za-z0-9_./:@])",
        "ff15": r"(?<![A-Za-z0-9_./:@])[fF]{2,}\s*@?\s*1[05](?![A-Za-z0-9_./:@])",

        # === Clap ===
        "easyclap": r"(?<![A-Za-z0-9_])[eE]asy\s*[cC]lap(?![A-Za-z0-9_])",
        "ezclap": r"(?<![A-Za-z0-9_])[eE][zZ]+\s*[cC]lap(?![A-Za-z0-9_])",
        "pepeclap": r"(?<![A-Za-z0-9_])[pP]epe\s*[cC]lap(?![A-Za-z0-9_])",
        "peepoclap": r"(?<![A-Za-z0-9_])[pP]eepo\s*[cC]lap(?![A-Za-z0-9_])",

        # === Next / queue ===
        "gonext": r"(?<![A-Za-z0-9_])[gG]o[nN]ext(?![A-Za-z0-9_])",
        "go next": r"(?<![A-Za-z0-9_])[gG]o\s*[nN]ext(?![A-Za-z0-9_])",
        "next game": r"(?<![A-Za-z0-9_])[nN]ext\s*[gG]ame(?![A-Za-z0-9_])",
        "new game": r"(?<![A-Za-z0-9_])[nN]ew\s*[gG]ame(?![A-Za-z0-9_])",
        "ready up": r"(?<![A-Za-z0-9_])[rR]eady\s*[uU]p(?![A-Za-z0-9_])",
        "q up": r"(?<![A-Za-z0-9_])[qQ]\s*[uU]p(?![A-Za-z0-9_])",

        # === Hype ===
        "lets go": r"(?<![A-Za-z0-9_])(?:[lL]ets+|[lL]et['']s+)\s*[gG]o+(?![A-Za-z0-9_])",
        "lesgo": r"(?<![A-Za-z0-9_])[lL]esgo+(?![A-Za-z0-9_])",
        "letsgoo": r"(?<![A-Za-z0-9_])[lL]etsgoo+(?![A-Za-z0-9_])",
        "yaaa": r"(?<![A-Za-z0-9_])[yY]aaa+(?![A-Za-z0-9_])",

        # === Win / loss keywords ===
        "win": r"(?<![A-Za-z0-9_./:@-])[wW]in(?![A-Za-z0-9_./:@-])",
        "victory": r"(?<![A-Za-z0-9_./:@-])[vV]ictory(?![A-Za-z0-9_./:@-])",
        "victory royale": r"(?<![A-Za-z0-9_])[vV]ictory\s*[rR]oyale(?![A-Za-z0-9_])",
        "defeat": r"(?<![A-Za-z0-9_./:@-])[dD]efeat(?![A-Za-z0-9_./:@-])",

        # === Pog family ===
        "pog": r"(?<![A-Za-z0-9_./:@-])[pP]og+(?:gers|champ|u)?(?![A-Za-z0-9_./:@-])",

        # === End / over ===
        "end": r"(?<![A-Za-z0-9_./:@-])[eE]nd(?![A-Za-z0-9_./:@-])",
        "next round": r"(?<![A-Za-z0-9_])[nN]ext\s*[rR]ound(?![A-Za-z0-9_])",
        "round over": r"(?<![A-Za-z0-9_])[rR]ound\s*(?:[oO]ver|[eE]nd(?:ed)?)(?![A-Za-z0-9_])",
        "match over": r"(?<![A-Za-z0-9_])[mM]atch\s*(?:[oO]ver|[eE]nd(?:ed)?)(?![A-Za-z0-9_])",
        "game over": r"(?<![A-Za-z0-9_])[gG]ame\s*[oO]ver(?![A-Za-z0-9_])",
        "game ended": r"(?<![A-Za-z0-9_./:@-])[gG]ame\s*[eE]nded(?![A-Za-z0-9_./:@-])",

        # === Again / requeue ===
        "again": r"(?<![A-Za-z0-9_./:@-])[aA]gain(?![A-Za-z0-9_./:@-])",
        "queue up": r"(?<![A-Za-z0-9_])[qQ]ueue\s*[uU]p(?![A-Za-z0-9_])",
        "queue again": r"(?<![A-Za-z0-9_])[qQ]ueue\s*[aA]gain(?![A-Za-z0-9_])",
        "requeue": r"(?<![A-Za-z0-9_])[rR]e-?queue(?:ing)?(?![A-Za-z0-9_])",

        # === Match / run ===
        "match found": r"(?<![A-Za-z0-9_])[mM]atch\s*[fF]ound(?![A-Za-z0-9_])",
        "new run": r"(?<![A-Za-z0-9_])[nN]ew\s*[rR]un(?![A-Za-z0-9_])",
        "goodrun": r"(?<![A-Za-z0-9_./@-])[gG]ood[rR]un(?![A-Za-z0-9_./@-])",
        "good run": r"(?<![A-Za-z0-9_])[gG]ood\s*[rR]un(?![A-Za-z0-9_])",
        "total wipeout": r"(?<![A-Za-z0-9_])[tT]otal\s*[wW]ipeout(?![A-Za-z0-9_])",

        # === RIP / F ===
        "rip": r"(?<![A-Za-z0-9_./:@-])[rR][iI][pP](?![A-Za-z0-9_./:@-])",
        "press f": r"(?<![A-Za-z0-9_])[pP]ress\s*[fF](?![A-Za-z0-9_])",
        "f in chat": r"(?<![A-Za-z0-9_])[fF](?:'s)?\s*in\s*(?:the\s*)?[cC]hat(?![A-Za-z0-9_])",

        # === Well played ===
        "ggwp": r"(?<![A-Za-z0-9_./:@])[gG]{2,}[wW][pP](?![A-Za-z0-9_./:@])",
        "wp": r"(?<![A-Za-z0-9_./:@])[wW][pP](?![A-Za-z0-9_./:@])",
        "well played": r"(?<![A-Za-z0-9_])[wW]ell\s*[pP]layed(?![A-Za-z0-9_])",
        "nt": r"(?<![A-Za-z0-9_./:@])[nN][tT](?![A-Za-z0-9_./:@])",
        "nice try": r"(?<![A-Za-z0-9_])[nN]ice\s*[tT]ry(?![A-Za-z0-9_])",
        "ope": r"(?<![A-Za-z0-9_./:@-])[oO]pe(?![A-Za-z0-9_./:@-])",

        # === Finish ===
        "finish": r"(?<![A-Za-z0-9_./:@-])[fF]inish(?![A-Za-z0-9_./:@-])",
        "finished": r"(?<![A-Za-z0-9_./:@-])[fF]inished(?![A-Za-z0-9_./:@-])",

        # === We won / lost ===
        "we won": r"(?<![A-Za-z0-9_])[wW]e\s*[wW]on(?![A-Za-z0-9_])",
        "we lost": r"(?<![A-Za-z0-9_])[wW]e\s*[lL]ost(?![A-Za-z0-9_])",

        # === Lobby / menu ===
        "back to lobby": r"(?<![A-Za-z0-9_])[bB]ack\s*to\s*[lL]obby(?![A-Za-z0-9_])",
        "return to lobby": r"(?<![A-Za-z0-9_])[rR]eturn\s*to\s*[lL]obby(?![A-Za-z0-9_])",
        "back to menu": r"(?<![A-Za-z0-9_])(?:[bB]ack|[rR]eturn)\s*to\s*(?:main\s*)?[mM]enu(?![A-Za-z0-9_])",

        # === Game-specific ===
        "you are the champion": r"(?<![A-Za-z0-9_])[yY]ou\s*are\s*the\s*[cC]hampion(?![A-Za-z0-9_])",
        "chicken dinner": r"(?<![A-Za-z0-9_])[cC]hicken\s*[dD]inner(?![A-Za-z0-9_])",
        "you died": r"(?<![A-Za-z0-9_])[yY]ou\s*[dD]ied(?![A-Za-z0-9_])",
        "wasted": r"(?<![A-Za-z0-9_./:@-])[wW]asted(?![A-Za-z0-9_./:@-])",
        "mission failed": r"(?<![A-Za-z0-9_])[mM]ission\s*[fF]ailed(?![A-Za-z0-9_])",

        # === Transition phrases ===
        "on to the next": r"(?<![A-Za-z0-9_])[oO]n\s*to\s*the\s*[nN]ext(?![A-Za-z0-9_])",
        "onto the next": r"(?<![A-Za-z0-9_])[oO]nto\s*the\s*[nN]ext(?![A-Za-z0-9_])",

        # === CS:GO ===
        "terrorists win": r"(?<![A-Za-z0-9_])[tT]errorists\s*[wW]in(?![A-Za-z0-9_])",
        "counter terrorists win": r"(?<![A-Za-z0-9_])[cC]ounter[-\s]*[tT]errorists\s*[wW]in(?![A-Za-z0-9_])",

        # === Overwatch ===
        "final killcam": r"(?<![A-Za-z0-9_])[fF]inal\s*[kK]ill\s*[cC]am(?![A-Za-z0-9_])",
        "play of the game": r"(?<![A-Za-z0-9_])[pP]lay\s*of\s*the\s*[gG]ame(?![A-Za-z0-9_])",

        # === Laughter / reactions ===
        "lmao": r"(?<![A-Za-z0-9_./:@])[lL][mM][aA][oO]+(?![A-Za-z0-9_./:@])",
        "lmfao": r"(?<![A-Za-z0-9_./:@])[lL][mM][fF][aA][oO]+(?![A-Za-z0-9_./:@])",
        "lol": r"(?<![A-Za-z0-9_./:@])[lL][oO][lL]+(?![A-Za-z0-9_./:@])",
        "xd": r"(?<![A-Za-z0-9_./:@])[xX][dD]+(?![A-Za-z0-9_./:@])",
        "no": r"(?<![A-Za-z0-9_./:@])[nN][oO]+(?![A-Za-z0-9_./:@])",

        # === Emotes / memes ===
        "om": r"(?<![A-Za-z0-9_./:@])[oO][mM](?![A-Za-z0-9_./:@])",
        "kekw": r"(?<![A-Za-z0-9_./:@])[kK][eE][kK][wW](?![A-Za-z0-9_./:@])",
        "kek": r"(?<![A-Za-z0-9_./:@])[kK][eE][kK](?![A-Za-z0-9_./:@])",

        # === Single-letter W / hype words ===
        "w": r"(?<![A-Za-z0-9_./:@-])[wW](?![A-Za-z0-9_./:@-])",
        "holy": r"(?<![A-Za-z0-9_./:@-])[hH]oly(?![A-Za-z0-9_./:@-])",
        "awesome": r"(?<![A-Za-z0-9_./:@-])[aA]wesome(?![A-Za-z0-9_./:@-])",

        # === Shock / surprise ===
        "omg": r"(?<![A-Za-z0-9_./:@-])[oO][mM][gG]+(?![A-Za-z0-9_./:@-])",
        "oh noo": r"(?<![A-Za-z0-9_])[oO]h\s*[nN]oo+(?![A-Za-z0-9_])",
        "oh no": r"(?<![A-Za-z0-9_])[oO]h\s*[nN]o+(?![A-Za-z0-9_])",
        "noooo": r"(?<![A-Za-z0-9_./:@-])[nN]oooo+(?![A-Za-z0-9_./:@-])",
        "noway": r"(?<![A-Za-z0-9_./:@-])[nN][oO][wW][aA][yY]+(?![A-Za-z0-9_./:@-])",
        "aintnoway": r"(?<![A-Za-z0-9_./:@-])[aA]int[nN]ow[aA]+[yY]+(?![A-Za-z0-9_./:@-])",
        "thats crazy": r"(?<![A-Za-z0-9_])[tT]hats?\s*[cC]razy+(?![A-Za-z0-9_])",
        "cinema": r"(?<![A-Za-z0-9_./:@-])[cC]inema(?![A-Za-z0-9_./:@-])",

        # === Twitch-y reactions ===
        "sadge": r"(?<![A-Za-z0-9_./:@])[sS]adge(?![A-Za-z0-9_./:@])",
        "monka": r"(?<![A-Za-z0-9_./:@])[mM]onka(?![A-Za-z0-9_./:@])",
        "pepelaugh": r"(?<![A-Za-z0-9_./:@])[pP]epe[lL]augh(?![A-Za-z0-9_./:@])",
        "pausechamp": r"(?<![A-Za-z0-9_./:@])[pP]ause[cC]hamp(?![A-Za-z0-9_./:@])",

        # === GOAT ===
        "goat": r"(?<![A-Za-z0-9_./:@-])[gG]\.?[oO]\.?[aA]\.?[tT]\.?(?![A-Za-z0-9_./:@-])",
        "greatest of all time": r"(?<![A-Za-z0-9_])[gG]reatest\s*of\s*all\s*[tT]ime(?![A-Za-z0-9_])",

        # === Failure / death ===
        "die": r"(?<![A-Za-z0-9_./:@-])[dD]ie(?:d|s|ing)?(?![A-Za-z0-9_./:@-])",
        "death": r"(?<![A-Za-z0-9_./:@-])[dD]eath(?![A-Za-z0-9_./:@-])",
        "its over": r"(?<![A-Za-z0-9_])[iI]t['']?s\s*[oO]ver(?![A-Za-z0-9_])",
        "it's over": r"(?<![A-Za-z0-9_])[iI]t['']?s\s*[oO]ver(?![A-Za-z0-9_])",
        "cooked": r"(?<![A-Za-z0-9_./:@-])[cC]ooked(?![A-Za-z0-9_./:@-])",
        
        "lg": r"(?<![A-Za-z0-9_./:@-])[lL][gG](?![A-Za-z0-9_./:@-])",
        "last game": r"(?<![A-Za-z0-9_])[lL]ast\s*[gG]ame[sS]?(?![A-Za-z0-9_])",
        "last round": r"(?<![A-Za-z0-9_])[lL]ast\s*[rR]ound[sS]?(?![A-Za-z0-9_])",
        "lul": r"(?<![A-Za-z0-9_./:@])[lL][uU][lL]+(?![A-Za-z0-9_./:@])",
        }
        # Compile regex patterns for faster matching and to use .findall
        return {name: re.compile(regex) for name, regex in patterns.items()}
    
    TOKEN_PATTERNS = get_token_patterns()

    # 3) Pass 2 – fill each window
    for idx, w in enumerate(windows):
        # Count messages and cap per-message emote weight to avoid emote spam inflation
        total_msgs = 0
        emote_msg_count = 0
        for seg in w.segments:
            chat_messages = seg.get("chat_messages", [])
            total_msgs += len(chat_messages)
            for msg in chat_messages:
                emotes = msg.get("emotes", [])
                if emotes:
                    emote_msg_count += 1
        
        # Total chat activity = messages + messages-with-emotes (cap = 1 per msg)
        total_chat_activity = total_msgs + emote_msg_count
        w.chat_rate = total_chat_activity / w.duration if w.duration > 0 else 0.0

        # Per-chapter baseline; fallback to global if missing
        mean, std = chapter_stats.get(w.chapter_id, (global_mean, global_std))
        w.chat_rate_z = (w.chat_rate - mean) / (std or 1.0)

        # neighbouring rates for burst_score
        left_rate = windows[idx - 1].chat_rate if idx > 0 else w.chat_rate
        right_rate = windows[idx + 1].chat_rate if idx + 1 < len(windows) else w.chat_rate
        background = 0.5 * (left_rate + right_rate) + 1e-5
        w.burst_score = w.chat_rate / background if background else 0.0

        # reaction hits across transcript and chat using comprehensive patterns
        all_text = []
        for seg in w.segments:
            if seg.get("transcript"):
                all_text.append(seg["transcript"])
            for msg in seg.get("chat_messages", []):
                all_text.append(msg.get("content", ""))
        
        # Count all reaction patterns
        combined_text = " ".join(all_text)
        hits = {}
        for pattern_name, pattern_regex in TOKEN_PATTERNS.items():
            matches = pattern_regex.findall(combined_text)
            if matches:
                hits[pattern_name] = len(matches)
        w.reaction_hits = hits

        # section_context placeholder (transcript snippet)
        if all_text:
            joined = " ".join(all_text)
            w.section_context = (joined[:220] + "…") if len(joined) > 220 else joined

    logger.info("Burst metrics computed.")
    
    # Create documents
    logger.info("Creating documents...")
    documents = []
    for window in windows:
        doc = create_document_from_window(
            window=window,
            vod_id=vod_id,
            chapter_id=getattr(window, 'chapter_id', None),
            category=getattr(window, 'category', None),
            excluded=getattr(window, 'excluded', False),
            mode=getattr(window, 'mode', 'unknown')
        )
        documents.append(doc)
    
    logger.info(f"Created {len(documents)} documents")
    
    # Log document statistics
    if documents:
        modes = {}
        categories = {}
        excluded_count = 0
        for doc in documents:
            modes[doc.mode] = modes.get(doc.mode, 0) + 1
            if doc.category:
                categories[doc.category] = categories.get(doc.category, 0) + 1
            if doc.excluded:
                excluded_count += 1
        
        logger.info(f"Document modes: {modes}")
        logger.info(f"Document categories: {categories}")
        logger.info(f"Excluded documents: {excluded_count}")
        
        # Log duration statistics
        durations = [doc.len_s for doc in documents]
        logger.info(f"Document durations: min={min(durations):.1f}s, "
                   f"max={max(durations):.1f}s, "
                   f"median={sorted(durations)[len(durations)//2]:.1f}s")
    
    # Format embedding texts
    logger.info("Formatting embedding texts...")
    from vector_store.document_builder import format_embedding_text
    embedding_texts = [format_embedding_text(doc) for doc in documents]
    
    # Create vector index
    logger.info("Creating vector index...")
    vector_index = VectorIndex(index_path, embedding_model)
    
    # Add documents to index
    logger.info("Adding documents to vector index...")
    vector_index.add_documents(documents, embedding_texts)
    
    # Create query system
    query_system = QuerySystem(vector_index)
    
    # Log completion statistics
    stats = vector_index.get_stats()
    
    logger.info("=" * 60)
    logger.info(f"FULL VOD vector store build complete for VOD {vod_id}")
    logger.info(f"Documents indexed: {stats.get('document_count', 0)}")
    logger.info(f"Vector count: {stats.get('vector_count', 0)}")
    logger.info(f"Index path: {index_path}")
    logger.info("=" * 60)
    
    return query_system


def main():
    """Main function for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build full VOD vector store")
    parser.add_argument("vod_id", help="VOD ID to process")
    parser.add_argument("--window-size", type=float, default=30.0, help="Window size in seconds")
    parser.add_argument("--chat-lag", type=float, default=5.0, help="Chat lag correction in seconds")
    
    args = parser.parse_args()
    
    try:
        query_system = build_full_vod_vector_store(
            args.vod_id, 
            window_size_seconds=args.window_size,
            chat_lag_seconds=args.chat_lag
        )
        print(f"✅ Full VOD vector store built successfully for {args.vod_id}")
        
        # Test a query
        results = query_system.search("content", k=5)
        print(f"📊 Test query returned {len(results)} results")
        
    except Exception as e:
        print(f"❌ Build failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
