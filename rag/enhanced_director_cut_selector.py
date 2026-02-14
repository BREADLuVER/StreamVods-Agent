#!/usr/bin/env python3
"""
Enhanced Director's Cut Selector with Semantic Grouping

This module applies semantic similarity-based grouping before director cut selection,
preventing conversation fragmentation and maintaining narrative coherence.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rag.retrieval import (
    load_bursts,
    group_by_chapter,
    chapter_stats,
    detect_anchors,
    load_retriever,
    Retriever,
)


def load_atomic_segments(vod_id: str) -> List[Tuple[float, float]]:
    """Load sponsor segments (and other long atomic narrative spans) from narrative_analysis."""
    path = Path(f"data/ai_data/{vod_id}/{vod_id}_narrative_analysis.json")
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    spans: List[Tuple[float, float]] = []
    for ch in data.get("chunks", []):
        for seg in ch.get("sponsor_segments", []):
            # sponsor segments have start/end
            try:
                s = _ts_to_seconds(seg.get("start"))
                e = _ts_to_seconds(seg.get("end"))
                # offset by chunk start
                chunk_offset = ch.get("start_time", 0.0)
                s += chunk_offset
                e += chunk_offset
                spans.append((s, e))
            except Exception:
                continue
    return spans


def _ts_to_seconds(ts: str | float | int) -> float:
    if isinstance(ts, (int, float)):
        return float(ts)
    parts = [int(p) for p in str(ts).strip().split(":")]
    if len(parts) == 2:
        m, s = parts
        h = 0
    elif len(parts) == 3:
        h, m, s = parts
    else:
        return 0.0
    return float(h * 3600 + m * 60 + s)


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    m = sum(values) / len(values)
    if len(values) <= 1:
        return (m, 0.0)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    import math
    s = math.sqrt(max(0.0, var))
    return (m, s)


def _compute_mode_adjusted_chat_z(ch_bursts: List[Dict]) -> None:
    """Add per-mode, per-chapter adjusted chat z to each burst as _chat_rate_z_mode.

    Keeps merged-chapter baselines intact while preventing cross-mode dilution
    by normalizing within mode (jc vs game) inside the chapter.
    """
    vals_by_mode: Dict[str, List[float]] = {"jc": [], "game": []}
    # Collect cz per mode
    for b in ch_bursts:
        mode = str(b.get("mode") or "").lower()
        mkey = "jc" if mode == "jc" else "game"
        cz = float(b.get("chat_rate_z") or 0.0)
        vals_by_mode[mkey].append(cz)

    stats: Dict[str, Tuple[float, float]] = {}
    for k, arr in vals_by_mode.items():
        stats[k] = _mean_std(arr)

    # Write adjusted value
    for b in ch_bursts:
        mode = str(b.get("mode") or "").lower()
        mkey = "jc" if mode == "jc" else "game"
        cz = float(b.get("chat_rate_z") or 0.0)
        mean, std = stats.get(mkey, (0.0, 0.0))
        if std and std > 0.0:
            b["_chat_rate_z_mode"] = (cz - mean) / std
        else:
            # Fallback to original cz if no spread
            b["_chat_rate_z_mode"] = cz

def build_keep_set_gameplay_enhanced(ch_bursts: List[Dict], retriever: Retriever) -> List[bool]:
    """
    Enhanced keep set for gameplay that uses semantic similarity more selectively.
    """
    keep = [False] * len(ch_bursts)
    anchors = detect_anchors(ch_bursts)
    
    # Keep anchors and important roles (original logic)
    for i, b in enumerate(ch_bursts):
        if anchors[i]:
            keep[i] = True
        if b.get("role") in {"peak", "conflict", "resolution"}:
            keep[i] = True
    
    # Loosen anchor selection: include chat/burst-prominent windows as anchors
    for i, b in enumerate(ch_bursts):
        if keep[i]:
            continue
        cz = float(b.get("_chat_rate_z_mode") or b.get("chat_rate_z") or 0.0)
        bs = float(b.get("burst_score") or 0.0)
        if cz >= 0.25 or bs >= 1.2:
            keep[i] = True
    
    # Use semantic similarity to bridge topic threads for continuous content
    if retriever.have_index:
        for i, b in enumerate(ch_bursts):
            if not keep[i]:
                continue
            
            # Look for semantically similar content within reasonable time window
            for j in range(len(ch_bursts)):
                if i == j or keep[j]:
                    continue
                
                # Time proximity check - allow longer gaps for continuous content
                time_gap = abs(ch_bursts[j]["start_time"] - b["start_time"])
                if time_gap > 300.0:  # Within 5 minutes for continuous content
                    continue
                
                similarity = retriever.sim(b["id"], ch_bursts[j]["id"])
                
                # Lower threshold for continuous content like giveaways
                if similarity >= 0.4:  # Lower threshold for continuous content
                    # Additional check: if both are about the same topic, keep them together
                    b_topic = b.get("topic_key", "").lower()
                    j_topic = ch_bursts[j].get("topic_key", "").lower()
                    
                    # If both are giveaway-related, keep them together
                    if ("giveaway" in b_topic and "giveaway" in j_topic) or similarity >= 0.5:
                        keep[j] = True
    
    return keep


def build_keep_set_jc_enhanced(ch_bursts: List[Dict], retriever: Retriever) -> List[bool]:
    """
    Enhanced keep set for Just Chatting that maintains conversation flow.
    """
    drop_roles = {"afk", "tech_issue"}

    def is_lively(b: Dict) -> bool:
        """Return True if chat or reactions suggest active engagement."""
        czm = float(b.get("_chat_rate_z_mode") or b.get("chat_rate_z") or 0.0)
        if czm > 0.3:
            return True
        if (b.get("burst_score", 0.0) or 0.0) > 0.0:
            return True
        if sum((b.get("reaction_hits") or {}).values()) > 0:
            return True
        if (b.get("energy") or "medium") == "high":
            return True
        return False

    keep: List[bool] = []
    for b in ch_bursts:
        role = b.get("role")
        if role in drop_roles and not is_lively(b):
            keep.append(False)
        else:
            keep.append(True)
    
    # Use semantic similarity more conservatively for JC
    if retriever.have_index:
        for i, b in enumerate(ch_bursts):
            if not keep[i]:
                continue
            
            # Only look at nearby segments for conversation flow
            for j in range(len(ch_bursts)):
                if i == j or keep[j]:
                    continue
                
                # Time proximity check first
                time_gap = abs(ch_bursts[j]["start_time"] - b["start_time"])
                if time_gap > 180.0:  # Only within 3 minutes for JC
                    continue
                
                similarity = retriever.sim(b["id"], ch_bursts[j]["id"])
                if similarity >= 0.5:  # Higher threshold for JC too
                    keep[j] = True
    
    return keep


def enhanced_smooth_fill_small_gaps(
    ch_bursts: List[Dict],
    keep: List[bool],
    retriever: Retriever,
    is_game: bool,
    stats: Dict[str, float],
    max_gap_s: float = 30.0,  # Increased from 15s
) -> None:
    """
    Enhanced gap filling that uses semantic similarity for continuous content.
    """
    if is_game:
        max_gap_s = min(30.0, stats.get("dur_q25", 30.0) or 30.0)  # Allow longer gaps
    
    n = len(ch_bursts)
    i = 0
    while i < n:
        if not keep[i]:
            i += 1
            continue
        
        # Find next kept burst
        j = i + 1
        while j < n and not keep[j]:
            j += 1
        
        if j < n:
            gap = max(0.0, ch_bursts[j]["start_time"] - ch_bursts[i]["end_time"])
            if gap <= max_gap_s:
                # Use semantic similarity to fill gaps
                if retriever.have_index:
                    anchor_id = ch_bursts[i]["id"]
                    anchor_topic = ch_bursts[i].get("topic_key", "").lower()
                    
                    for k in range(i + 1, j):
                        b = ch_bursts[k]
                        similarity = retriever.sim(anchor_id, b["id"])
                        b_topic = b.get("topic_key", "").lower()
                        
                        # Lower threshold for continuous content like giveaways
                        if similarity >= 0.3:  # Lower threshold for continuous content
                            # Special handling for giveaway content
                            if ("giveaway" in anchor_topic and "giveaway" in b_topic) or similarity >= 0.4:
                                keep[k] = True
                else:
                    # Fallback to simple gap filling
                    for k in range(i + 1, j):
                        keep[k] = True
        
        i = j


def enhanced_flatten_manifest(ch_bursts: List[Dict], keep: List[bool], 
                             block_ids: List[Optional[str]], is_game: bool,
                             retriever: Retriever) -> List[Dict]:
    """
    Enhanced manifest flattening that uses semantic similarity for merging.
    """
    out: List[Dict] = []
    current: Optional[Dict] = None

    def push_current():
        nonlocal current
        if not current:
            return
        current["duration"] = current["end"] - current["start"]
        out.append(current)
        current = None

    for i, b in enumerate(ch_bursts):
        if not keep[i]:
            continue
        
        bid = block_ids[i]
        start, end = b["start_time"], b["end_time"]
        
        if not current:
            current = {
                "chapter_id": b.get("chapter_id"),
                "peak_block_id": bid,
                "start": start,
                "end": end,
                "burst_ids": [b["id"]],
                "anchor_burst_id": b["id"] if sum((b.get("reaction_hits") or {}).values()) > 0 else None,
                "topic_key_votes": [b.get("topic_key")],
                "energies": [b.get("energy")],
                "summaries": [b.get("summary")],
            }
            continue
        
        # Enhanced merging logic using semantic similarity
        same_block = (current.get("peak_block_id") == bid)
        gap = max(0.0, start - current["end"])
        
        # Use semantic similarity for merging decisions
        should_merge = False
        if retriever.have_index and current.get("burst_ids"):
            # Check similarity to any burst in current segment
            for existing_id in current["burst_ids"]:
                similarity = retriever.sim(existing_id, b["id"])
                if similarity >= 0.4:  # High similarity threshold
                    should_merge = True
                    break
        
        # Fallback to time-based merging
        if not should_merge:
            gap_threshold = 8.0 if is_game else 15.0
            should_merge = same_block and gap <= gap_threshold
        
        if should_merge:
            current["end"] = end
            current["burst_ids"].append(b["id"])
            if (not current.get("anchor_burst_id")) and sum((b.get("reaction_hits") or {}).values()) > 0:
                current["anchor_burst_id"] = b["id"]
            current["topic_key_votes"].append(b.get("topic_key"))
            current["energies"].append(b.get("energy"))
            current["summaries"].append(b.get("summary"))
        else:
            push_current()
            current = {
                "chapter_id": b.get("chapter_id"),
                "peak_block_id": bid,
                "start": start,
                "end": end,
                "burst_ids": [b["id"]],
                "anchor_burst_id": b["id"] if sum((b.get("reaction_hits") or {}).values()) > 0 else None,
                "topic_key_votes": [b.get("topic_key")],
                "energies": [b.get("energy")],
                "summaries": [b.get("summary")],
            }
    
    push_current()

    # Finalize fields
    for m in out:
        # Majority topic_key
        votes = [v for v in m["topic_key_votes"] if v]
        maj = None
        if votes:
            counts: Dict[str, int] = {}
            for v in votes:
                counts[v] = counts.get(v, 0) + 1
            maj = max(counts, key=counts.get)
        m["topic_key"] = maj or ""
        
        # Energy mode or max
        order = {"low": 0, "medium": 1, "high": 2}
        rev = {v: k for k, v in order.items()}
        ranks = [order.get(e, 1) for e in m["energies"]]
        m["energy"] = rev[max(ranks)] if ranks else "medium"
        
        # Stitched summary
        summaries = [s for s in m["summaries"] if s]
        m["summary"] = " → ".join(summaries[:8])
        
        del m["topic_key_votes"]
        del m["energies"]
        del m["summaries"]
    
    return out


# -------------------- Resolution post-pass using reaction hits --------------------

def _get_token_confidence(token: str) -> float:
    """Return confidence score for tokens that suggest a game end or scene change."""
    # Strong end/transition signals
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
        "finish", "finished", "end",
        "its over", "it's over", "die", "death",
    }

    # Moderate indicators
    medium_confidence = {
        "ez", "ezclap", "easyclap", "pepeclap", "peepoclap",
        "goodbye", "bye", "o7", "07", "ff", "ff15",
        "ready up", "q up", "om",
        "wp", "well played", "nt", "nice try",
        "rip", "press f", "f in chat",
        "on to the next", "onto the next",
    }

    # Weak/filler
    low_confidence = {
        "lets go", "lesgo", "letsgoo", "yaaa", "w", "win", "pog", "again",
        "holy", "awesome",
        "lmao", "lmfao", "lol", "xd", "no", "omg", "oh noo", "oh no", "noooo",
        "noway", "aintnoway", "thats crazy", "cinema",
        "kekw", "kek", "sadge", "monka", "pepelaugh", "pausechamp",
        "goat", "greatest of all time",
        "cooked", "lol", "lul", "lmao", "lmfao",
    }

    if token in high_confidence:
        return 0.95
    if token in medium_confidence:
        return 0.75
    if token in low_confidence:
        return 0.01
    return 0.5


def _score_resolution_candidates(ch_bursts: List[Dict]) -> List[Dict]:
    """Produce candidate resolution events from reaction_hits and metrics.

    Returns a list of dicts with keys: ts, score, reasons, chapter_id.
    """
    events: List[Dict] = []
    for b in ch_bursts:
        try:
            start = float(b.get("start_time", 0.0) or 0.0)
            end = float(b.get("end_time", start))
            rh = b.get("reaction_hits") or {}
            cz = float(b.get("chat_rate_z") or b.get("_chat_rate_z_mode") or 0.0)
            bs = float(b.get("burst_score") or 0.0)
            chapter_id = b.get("chapter_id")
            # token score
            token_score = 0.0
            reasons: List[str] = []
            had_conf_token = False  # at least one token (not just boosters)
            if isinstance(rh, dict):
                for name, count in rh.items():
                    try:
                        cnt = float(count or 0.0)
                    except Exception:
                        cnt = 0.0
                    if cnt <= 0:
                        continue
                    conf = _get_token_confidence(str(name))
                    token_score += conf * min(3.0, cnt)
                    if conf >= 0.75:
                        reasons.append(str(name))
                        had_conf_token = True
            # boosters
            if cz >= 0.6:
                token_score += 0.2
                reasons.append("chat_z")
            if bs >= 1.3:
                token_score += 0.2
                reasons.append("burst_score")
            # Require at least one confidence token OR a strong aggregate token_score
            if token_score <= 0.0 or (not had_conf_token and token_score < 1.2):
                continue
            events.append({"ts": end, "score": token_score, "reasons": list(dict.fromkeys(reasons)), "chapter_id": chapter_id})
        except Exception:
            continue

    # Non-maximum suppression over time (chapter-local): keep top events within ±90s
    events.sort(key=lambda e: e["score"], reverse=True)
    kept: List[Dict] = []
    for ev in events:
        # Stricter base threshold; require stronger aggregate signal
        if ev["score"] < 1.4:
            continue
        too_close = False
        replace_idx = -1
        for idx, kv in enumerate(kept):
            if kv.get("chapter_id") != ev.get("chapter_id"):
                continue
            if abs(float(kv["ts"]) - float(ev["ts"])) <= 90.0:
                # Compete: prefer higher score; tie-breaker prefers later timestamp
                if (ev["score"] > kv["score"] + 0.15) or (abs(ev["score"] - kv["score"]) <= 0.15 and float(ev["ts"]) > float(kv["ts"])):
                    replace_idx = idx
                too_close = True
                break
        if not too_close:
            kept.append(ev)
        elif replace_idx >= 0:
            kept[replace_idx] = ev
    kept.sort(key=lambda e: e["ts"])  # chronological
    return kept


def _apply_resolution_events_to_ranges(
    ranges: List[Dict],
    events: List[Dict],
    ch_bursts: List[Dict],
    tail_max_s: float = 120.0,
    tail_frac: float = 0.50,
    cause_lookback_s: float = 150.0,
) -> None:
    """Mutate ranges adding resolution evidence based on detected events.

    Policy: event must fall within the tail of the range; require a preceding
    conflict/peak within a lookback window in the same chapter/thread when possible.
    """
    # Index bursts by time for quick cause lookup
    chapter_map: Dict[str, List[Dict]] = {}
    for b in ch_bursts:
        cid = str(b.get("chapter_id") or "")
        chapter_map.setdefault(cid, []).append(b)
    for cid in chapter_map:
        chapter_map[cid].sort(key=lambda x: float(x.get("start_time", 0.0)))

    # Map each event to the single range that contains it (rs <= ts <= re)
    ts_to_range_idx: Dict[float, int] = {}
    for idx, r in enumerate(ranges):
        try:
            rs = float(r.get("start", 0.0) or 0.0)
            re = float(r.get("end", rs))
        except Exception:
            continue
        for ev in events:
            t = float(ev.get("ts") or -1.0)
            if rs <= t <= re and ev.get("chapter_id") == r.get("chapter_id"):
                # Only keep the closest-to-end mapping if duplicates
                prev_idx = ts_to_range_idx.get(t)
                if prev_idx is None:
                    ts_to_range_idx[t] = idx
                else:
                    prev_re = float(ranges[prev_idx].get("end", 0.0) or 0.0)
                    # prefer the range whose end is nearer to the event
                    if abs(prev_re - t) > abs(re - t):
                        ts_to_range_idx[t] = idx

    for idx, r in enumerate(ranges):
        try:
            rs = float(r.get("start", 0.0) or 0.0)
            re = float(r.get("end", rs))
            dur = max(0.0, re - rs)
            tail_window = min(tail_max_s, tail_frac * dur) if dur > 0 else 0.0
            if tail_window <= 0.0 or dur < 60.0:
                continue
            cid = str(r.get("chapter_id") or "")
            # pick best event in tail that maps to THIS range only
            tail_events = [
                ev for ev in events
                if ev.get("chapter_id") == r.get("chapter_id")
                and (re - tail_window) <= float(ev["ts"]) <= re
                and ts_to_range_idx.get(float(ev["ts"])) == idx
            ]
            if not tail_events:
                # allow slight spillover just after the end (rendering drift)
                tail_events = [
                    ev for ev in events
                    if ev.get("chapter_id") == r.get("chapter_id")
                    and re < float(ev["ts"]) <= (re + 10.0)
                    and ts_to_range_idx.get(float(ev["ts"])) == idx
                ]
            if not tail_events:
                continue
            # Prefer later event among similarly scored; pick strongest otherwise
            tail_events.sort(key=lambda e: (e["score"], e["ts"]))
            best = tail_events[-1]
            # Ensure event is late in the range (>= 60% of duration)
            if float(best["ts"]) < (rs + 0.60 * dur):
                continue
            # cause proximity: look for prior conflict/peak within window
            cause_ok = False
            for b in chapter_map.get(cid, []):
                bs = float(b.get("start_time", 0.0) or 0.0)
                if bs > float(best["ts"]):
                    break
                if bs < (float(best["ts"]) - cause_lookback_s):
                    continue
                role = str(b.get("role") or "").lower()
                if role in {"conflict", "peak"}:
                    cause_ok = True
                    break
            # Apply flags
            r["_resolution_event_ts"] = float(best["ts"])  # for inspection
            r["_resolution_reasons"] = list(best.get("reasons", []))
            has_strong_token = bool(r["_resolution_reasons"])  # only medium/high tokens populate reasons
            if cause_ok and has_strong_token:
                r["_resolution_evidence"] = True
                r["_resolution_reason"] = "hits_tail"
                r["_has_resolution"] = True
                if (r.get("_major_role") or "") != "peak":
                    r["_major_role"] = "resolution"
            else:
                r["_resolution_reason"] = "hits_tail_weak"
                # Mark only as candidate; avoid setting evidence to reduce noise
                if not r.get("_has_resolution"):
                    r["_has_resolution_weak"] = True
        except Exception:
            continue


def update_db_keep_blocks(vod_id: str, chapter_id: str, ch_bursts: List[Dict], 
                         keep: List[bool], block_ids: List[Optional[str]]):
    """Update database with keep flags and block IDs."""
    db_path = Path(f"data/vector_stores/{vod_id}/metadata.db")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    
    # Auto-migrate columns
    cur.execute("PRAGMA table_info(documents)")
    cols = {row[1] for row in cur.fetchall()}
    if "keep_flag" not in cols:
        cur.execute("ALTER TABLE documents ADD COLUMN keep_flag INTEGER DEFAULT 0")
    if "peak_block_id" not in cols:
        cur.execute("ALTER TABLE documents ADD COLUMN peak_block_id TEXT")
    if "anchor_burst_id" not in cols:
        cur.execute("ALTER TABLE documents ADD COLUMN anchor_burst_id TEXT")
    conn.commit()

    for b, k, pb in zip(ch_bursts, keep, block_ids):
        cur.execute("UPDATE documents SET keep_flag=?, peak_block_id=? WHERE id=?", 
                   (1 if k else 0, pb, b["id"]))
    conn.commit()
    conn.close()


def load_chapters(vod_id: str) -> Dict[str, Dict]:
    """Load chapters metadata for strong JC/Game hint from chapters.json."""
    path = Path(f"data/ai_data/{vod_id}/{vod_id}_chapters.json")
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(raw, dict) and isinstance(raw.get("chapters"), list):
        arr = raw["chapters"]
    elif isinstance(raw, list):
        arr = raw
    else:
        arr = []
    out: Dict[str, Dict] = {}
    for ch in arr:
        cid = str(ch.get("id")) if ch.get("id") is not None else None
        if cid:
            out[cid] = ch
    return out


def is_jc_chapter(chapter_id: str, ch_bursts: List[Dict], chapters_map: Dict[str, Dict]) -> bool:
    """Determine if a chapter is Just Chatting.

    Priority:
      1) chapters.json category/original_category contains 'just chatting'
      2) Majority vote of document.mode ('jc' vs 'game') within this chapter
      3) Legacy keyword heuristic on topic/topic_key
    """
    # 1) chapters.json
    ch_meta = chapters_map.get(str(chapter_id) if chapter_id is not None else "")
    if isinstance(ch_meta, dict):
        cats = []
        for k in ("original_category", "category"):
            v = ch_meta.get(k)
            if isinstance(v, str):
                cats.append(v.lower())
        blob = " ".join(cats)
        if "just chatting" in blob or "just_chatting" in blob:
            return True

    # 2) majority document.mode
    mode_counts: Dict[str, int] = {"jc": 0, "game": 0}
    for b in ch_bursts:
        m = str(b.get("mode") or "").lower()
        if m in mode_counts:
            mode_counts[m] += 1
    if (mode_counts["jc"] + mode_counts["game"]) > 0:
        return mode_counts["jc"] >= mode_counts["game"]

    # 3) fallback heuristic
    keys = [(b.get("topic_key") or "").lower() for b in ch_bursts]
    topics = [(b.get("topic") or "").lower() for b in ch_bursts]
    blob2 = " ".join(keys + topics)
    return ("just chatting" in blob2) or ("just_chatting" in blob2) or ("jc" in keys)


# -------------------- Narrative-aware manifest postprocess --------------------

def _aggregate_range_stats(ch_bursts: List[Dict], start: float, end: float) -> Dict[str, object]:
    roles: Dict[str, int] = {}
    chat_vals: List[float] = []
    topic_keys: Dict[str, int] = {}
    threads: Dict[str, int] = {}
    has_peak = False
    has_conflict = False
    has_build = False
    has_resolution = False

    for b in ch_bursts:
        bs = float(b.get("start_time", 0.0))
        be = float(b.get("end_time", 0.0))
        if be <= start or bs >= end:
            continue
        r = (b.get("role") or "").lower()
        roles[r] = roles.get(r, 0) + 1
        if r == "peak":
            has_peak = True
        elif r == "conflict":
            has_conflict = True
        elif r == "build_up":
            has_build = True
        elif r == "resolution":
            has_resolution = True
        czm = float(b.get("_chat_rate_z_mode") or b.get("chat_rate_z") or 0.0)
        chat_vals.append(max(0.0, czm))
        tk = (b.get("topic_key") or "").strip()
        if tk:
            topic_keys[tk] = topic_keys.get(tk, 0) + 1
        th = b.get("topic_thread")
        if th is not None:
            ths = str(th)
            threads[ths] = threads.get(ths, 0) + 1

    def _majority(d: Dict[str, int]) -> str:
        return max(d, key=d.get) if d else ""

    avg_chat = (sum(chat_vals) / len(chat_vals)) if chat_vals else 0.0
    return {
        "avg_chat_z": avg_chat,
        "major_role": _majority(roles),
        "has_peak": has_peak,
        "has_conflict": has_conflict,
        "has_build": has_build,
        "has_resolution": has_resolution,
        "topic_key_major": _majority(topic_keys),
        "topic_thread_major": _majority(threads),
    }


def _find_back_buildup_start(ch_bursts: List[Dict], start: float, max_extend: float, frac: float) -> float:
    """Scan up to max_extend seconds before start and return an earlier start capturing buildup.

    Heuristic: extend to earliest time within window where chat_z <= frac * local_max, or
    to the earliest build_up/conflict burst start if present, but not beyond the window.
    """
    window_start = max(0.0, start - max_extend)
    # Collect bursts in [window_start, start]
    bursts = [b for b in ch_bursts if float(b.get("end_time", 0.0)) > window_start and float(b.get("start_time", 0.0)) < start]
    if not bursts:
        return start
    bursts.sort(key=lambda b: float(b.get("start_time", 0.0)))
    # Build simple time series of midpoints and chat_z
    series: List[tuple] = []
    for b in bursts:
        bs = float(b.get("start_time", 0.0))
        be = float(b.get("end_time", 0.0))
        mid = (bs + be) * 0.5
        cz = max(0.0, float(b.get("chat_rate_z") or 0.0))
        if window_start <= mid <= start:
            series.append((mid, cz, bs))
    if not series:
        return start
    local_max = max(cz for _, cz, _ in series)
    if local_max <= 0:
        # If no signal, prefer earliest build/conflict
        for b in bursts:
            if (b.get("role") or "").lower() in {"build_up", "conflict"}:
                return max(window_start, float(b.get("start_time", 0.0)))
        return start
    thr = frac * local_max
    candidate = start
    for mid, cz, bs in series:
        if cz <= thr:
            candidate = min(candidate, bs)
            break
    # Also consider earliest build/conflict
    for b in bursts:
        if (b.get("role") or "").lower() in {"build_up", "conflict"}:
            candidate = min(candidate, float(b.get("start_time", 0.0)))
            break
    return max(window_start, candidate)


def postprocess_manifest_for_chapter(
    ch_bursts: List[Dict],
    ranges: List[Dict],
    is_game: bool,
    atomic_spans: List[Tuple[float, float]],
    
    min_segment_s: float = 45.0,
    neighbor_gap_keep_small: float = 25.0,
    merge_gap_s: float = 40.0,
    buildup_max_extend: float = 75.0,
    buildup_frac: float = 0.35,
    resolution_guard_window: float = 90.0,
) -> List[Dict]:
    if not ranges:
        return ranges

    # Sort ranges by start
    ranges = sorted(ranges, key=lambda r: float(r.get("start") or 0.0))

    # Enrich ranges with stats
    for r in ranges:
        s = float(r.get("start") or 0.0)
        e = float(r.get("end") or s)
        stats = _aggregate_range_stats(ch_bursts, s, e)
        r["_avg_chat_z"] = stats["avg_chat_z"]
        r["_major_role"] = stats["major_role"]
        r["_has_peak"] = stats["has_peak"]
        r["_has_conflict"] = stats["has_conflict"]
        r["_has_build"] = stats["has_build"]
        r["_has_resolution"] = stats["has_resolution"]
        r.setdefault("topic_key", stats["topic_key_major"])  # keep if existing
        r["_topic_thread"] = stats["topic_thread_major"]

    # 1) Buildup capture for ranges with peak
    out: List[Dict] = []
    for i, r in enumerate(ranges):
        s = float(r.get("start") or 0.0)
        e = float(r.get("end") or s)
        new_start = s
        if is_game and r.get("_has_peak"):
            new_start = _find_back_buildup_start(ch_bursts, s, buildup_max_extend, buildup_frac)
            # do not cross sponsor spans if present: clamp to not enter a sponsor-only island
            for ss, ee in atomic_spans:
                if new_start < ss < s < ee:
                    new_start = ss  # clamp to sponsor boundary
        if new_start < s:
            r["start"] = new_start
            r["duration"] = float(r["end"]) - new_start
        out.append(r)
    ranges = sorted(out, key=lambda r: float(r.get("start") or 0.0))

    # 2) Resolution guard: drop orphan resolution
    guarded: List[Dict] = []
    for i, r in enumerate(ranges):
        if not is_game:
            guarded.append(r)
            continue
        if (r.get("_major_role") or "") != "resolution":
            guarded.append(r)
            continue
        # look back within window for conflict/peak in bursts
        window_start = float(r.get("start") or 0.0) - resolution_guard_window
        found_cause = False
        for b in ch_bursts:
            bs = float(b.get("start_time", 0.0))
            if window_start <= bs <= float(r.get("start") or 0.0):
                role = (b.get("role") or "").lower()
                if role in {"conflict", "peak"}:
                    found_cause = True
                    break
        if found_cause:
            guarded.append(r)
        else:
            # attempt left expansion to include nearest conflict/peak
            nearest = None
            for b in ch_bursts:
                bs = float(b.get("start_time", 0.0))
                role = (b.get("role") or "").lower()
                if role in {"conflict", "peak"} and float(r.get("start") or 0.0) - resolution_guard_window <= bs < float(r.get("start") or 0.0):
                    if nearest is None or bs > nearest:
                        nearest = bs
            if nearest is not None:
                r["start"] = min(r["start"], nearest)
                r["duration"] = float(r["end"]) - float(r["start"]) 
                guarded.append(r)
            # else: drop
    ranges = sorted(guarded, key=lambda r: float(r.get("start") or 0.0))

    # 3) Short-segment pruning
    pruned: List[Dict] = []
    for i, r in enumerate(ranges):
        s = float(r.get("start") or 0.0)
        e = float(r.get("end") or s)
        dur = e - s
        prev_end = float(ranges[i-1]["end"]) if i > 0 else None
        next_start = float(ranges[i+1]["start"]) if i + 1 < len(ranges) else None
        gap_prev = (s - prev_end) if prev_end is not None else None
        gap_next = (next_start - e) if next_start is not None else None
        major = (r.get("_major_role") or "").lower()
        avg_cz = float(r.get("_avg_chat_z") or 0.0)
        keep = True
        if dur < min_segment_s:
            close_neighbors = ((gap_prev is not None and gap_prev <= neighbor_gap_keep_small) or (gap_next is not None and gap_next <= neighbor_gap_keep_small))
            if (not close_neighbors) and avg_cz <= 0.10 and major in {"filler", "intro"}:
                # allow small resolution tail directly after a peak/conflict
                if major == "resolution" and gap_prev is not None and gap_prev <= 30.0 and i > 0 and (ranges[i-1].get("_has_peak") or ranges[i-1].get("_has_conflict")):
                    keep = True
                else:
                    keep = False
        if keep:
            pruned.append(r)
    ranges = pruned

    # 4) Topic/thread continuity merge
    merged: List[Dict] = []
    for r in ranges:
        if not merged:
            merged.append(r)
            continue
        prev = merged[-1]
        gap = float(r.get("start") or 0.0) - float(prev.get("end") or 0.0)
        same_topic = (r.get("topic_key") and prev.get("topic_key") and r.get("topic_key") == prev.get("topic_key"))
        same_thread = (r.get("_topic_thread") and prev.get("_topic_thread") and r.get("_topic_thread") == prev.get("_topic_thread"))
        if gap <= merge_gap_s and (same_topic or same_thread):
            prev["end"] = float(r.get("end") or prev["end"]) 
            prev["duration"] = float(prev["end"]) - float(prev.get("start") or 0.0)
            # propagate stronger energy and extend summary
            energies = [prev.get("energy"), r.get("energy")]
            order = {"low": 0, "medium": 1, "high": 2}
            prev["energy"] = max(energies, key=lambda x: order.get(str(x or "medium"), 1))
            if prev.get("summary") and r.get("summary"):
                prev["summary"] = prev["summary"] + " → " + r["summary"]
        else:
            merged.append(r)

    return merged


def main():
    """Enhanced director's cut selection with semantic grouping."""
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced director's cut selection with semantic grouping")
    parser.add_argument("vod_id", help="VOD ID")
    parser.add_argument("--write-db", action="store_true", help="Persist keep flags/blocks into metadata.db (default: off)")
    parser.add_argument("--use-retriever", action="store_true", help="Enable semantic retriever bridging (default: off)")
    args = parser.parse_args()

    bursts = load_bursts(args.vod_id)
    retriever = load_retriever(args.vod_id)
    if not args.use_retriever:
        try:
            retriever.have_index = False
            if not hasattr(retriever, "sim"):
                retriever.sim = lambda *_a, **_k: 0.0
        except Exception:
            class _Dummy:
                have_index = False
                def sim(self, *_a, **_k):
                    return 0.0
            retriever = _Dummy()
    chapters_map = load_chapters(args.vod_id)

    # Load atomic spans once per VOD
    atomic_spans = load_atomic_segments(args.vod_id)

    by_chap = group_by_chapter(bursts)

    manifests: List[Dict] = []

    for chap_id, ch_bursts in by_chap.items():
        # Compute per-mode adjusted chat z within this merged chapter
        _compute_mode_adjusted_chat_z(ch_bursts)
        jc = is_jc_chapter(chap_id, ch_bursts, chapters_map)
        is_game = not jc
        
        print(f"Processing chapter {chap_id} ({'JC' if jc else 'Gameplay'})")
        
        # Enhanced keep set with semantic similarity
        if is_game:
            keep = build_keep_set_gameplay_enhanced(ch_bursts, retriever)
        else:
            keep = build_keep_set_jc_enhanced(ch_bursts, retriever)
        
        # Use original pipeline with semantic enhancements
        from rag.director_cut_selector import context_wrap, pad_around_importants, salience_guard_gameplay, drop_kept_loners_gameplay, label_peak_blocks
        
        # context wrap (no merge, just adjacent neighbors)
        context_wrap(ch_bursts, keep)
        # padding around important bursts (anchors/peak/conflict/resolution)
        stats = chapter_stats(ch_bursts)
        pad_around_importants(ch_bursts, keep, stats, retriever, is_game)
        # Enhanced gap filling that uses semantic similarity for continuous content
        enhanced_smooth_fill_small_gaps(ch_bursts, keep, retriever, is_game, stats, max_gap_s=18.0)
        # salience guard ratchet (gameplay only)
        if is_game:
            keep = salience_guard_gameplay(ch_bursts, keep)
            # drop isolated micro-keeps
            keep = drop_kept_loners_gameplay(ch_bursts, keep, stats)
        # peak blocks
        blocks = label_peak_blocks(ch_bursts, keep)

        # ---------------------------------------------------------------------------------
        # Enforce atomic sponsor spans: if any burst within span is kept, keep all bursts in span
        # ---------------------------------------------------------------------------------
        if atomic_spans:
            for span_start, span_end in atomic_spans:
                # detect overlap with any kept burst
                overlap = False
                for i, b in enumerate(ch_bursts):
                    if not keep[i]:
                        continue
                    if b["end_time"] < span_start or b["start_time"] > span_end:
                        continue
                    overlap = True
                    break
                if overlap:
                    for i, b in enumerate(ch_bursts):
                        if b["start_time"] >= span_start and b["end_time"] <= span_end:
                            keep[i] = True
        
        # Update database (optional)
        if args.write_db:
            update_db_keep_blocks(args.vod_id, chap_id, ch_bursts, keep, blocks)
        
        # Use enhanced manifest flattening with semantic similarity
        man = enhanced_flatten_manifest(ch_bursts, keep, blocks, is_game, retriever)
        # Narrative-aware postprocess to improve coherence (BEFORE resolution tagging)
        man = postprocess_manifest_for_chapter(
            ch_bursts,
            man,
            is_game,
            atomic_spans,
            min_segment_s=45.0,
            neighbor_gap_keep_small=25.0,
            merge_gap_s=40.0,
            buildup_max_extend=75.0,
            buildup_frac=0.35,
            resolution_guard_window=90.0,
        )
        # Resolution post-pass from reaction hits (runs AFTER postprocessing so ranges are final)
        try:
            res_events = _score_resolution_candidates(ch_bursts)
            _apply_resolution_events_to_ranges(man, res_events, ch_bursts)
        except Exception:
            res_events = []
        for m in man:
            m["vod_id"] = args.vod_id
            # Normalize chapter_id to a non-null string for stable sorting downstream
            m["chapter_id"] = str(m.get("chapter_id") or "unknown")
        manifests.extend(man)
        
        # Logging - use actual VOD time span, not sum of segments
        if ch_bursts:
            total_sec = ch_bursts[-1]["end_time"] - ch_bursts[0]["start_time"]  # Actual VOD duration
            kept_sec = sum((b["end_time"] - b["start_time"]) for b, k in zip(ch_bursts, keep) if k)
            cut_ratio = 0.0 if total_sec <= 0 else (1.0 - kept_sec / total_sec)
            print(f"  Kept {kept_sec/60:.1f}m / {total_sec/60:.1f}m ({cut_ratio*100:.1f}% cut), ranges={len(man)}")
        else:
            print(f"  No bursts found, ranges={len(man)}")

    # Write enhanced manifest
    out_dir = Path(f"data/vector_stores/{args.vod_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Robust sort: coerce chapter_id to string and start to float to avoid None comparisons
    sorted_ranges = sorted(
        manifests,
        key=lambda x: (str(x.get("chapter_id") or "unknown"), float(x.get("start") or 0.0))
    )
    
    total_seconds = 0.0
    for r in sorted_ranges:
        dur = r.get("duration")
        if dur is None:
            dur = float(r.get("end", 0.0)) - float(r.get("start", 0.0))
            r["duration"] = dur
        total_seconds += float(dur)

    def _format_hms(sec: float) -> str:
        s = int(round(sec))
        h = s // 3600
        m = (s % 3600) // 60
        s2 = s % 60
        return f"{h:02d}:{m:02d}:{s2:02d}"

    manifest_obj = {
        "vod_id": args.vod_id,
        "rag_used": bool(getattr(retriever, "have_index", False)),
        "rag_threshold": 0.40,  # Higher threshold for semantic grouping
        "semantic_grouping": True,
        "total_ranges": len(sorted_ranges),
        "total_duration_seconds": round(total_seconds, 3),
        "total_duration_minutes": round(total_seconds / 60.0, 2),
        "total_duration_hms": _format_hms(total_seconds),
        "ranges": sorted_ranges,
    }

    (out_dir / "enhanced_director_cut_manifest.json").write_text(
        json.dumps(manifest_obj, indent=2), encoding="utf-8"
    )

    # Emit resolution events for auditing if available
    try:
        all_events: List[Dict] = []
        # Recompute or reuse per-chapter events for a simple global log
        # Note: we do not have the per-chapter bursts here; skip if unavailable
        # Users can inspect per-range _resolution_event_ts fields instead.
        (out_dir / "resolution_events.json").write_text(json.dumps(all_events, indent=2), encoding="utf-8")
    except Exception:
        pass

    print(f"✅ Enhanced director's cut manifest written: {out_dir / 'enhanced_director_cut_manifest.json'}")


if __name__ == "__main__":
    main()


