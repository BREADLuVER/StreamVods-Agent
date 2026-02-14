#!/usr/bin/env python3
"""
Heuristic arc detectors for story_archs.

Detects short, self-contained narrative arcs from merged Director's Cut ranges.
Targets: 10–30 minutes (tunable). Prefers sequences that exhibit
build_up/conflict → peak → resolution, continuous topic threads, or
gameplay-like rounds with a climax.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RangeLite:
    start: float
    end: float
    duration: float
    topic_key: str
    raw: Dict[str, Any]


def _safe_bool(x: Any) -> bool:
    try:
        return bool(x)
    except Exception:
        return False


def _get_float(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        v = d.get(key)
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str) and v.strip():
            return float(v)
    except Exception:
        return default
    return default


def _similarity(
    a: RangeLite,
    b: RangeLite,
    retriever,
    topic_bonus: float,
    time_tau: float,
) -> float:
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


def _id_for_range(r: RangeLite) -> Optional[str]:
    abid = str(r.raw.get("anchor_burst_id") or "").strip() or None
    if abid:
        return abid
    bids = r.raw.get("burst_ids") if isinstance(r.raw.get("burst_ids"), list) else []
    if bids:
        b0 = bids[0]
        return b0 if isinstance(b0, str) else None
    return None


def _as_lite(r: Dict[str, Any]) -> RangeLite:
    s = _get_float(r, "start", 0.0)
    e = _get_float(r, "end", s)
    d = _get_float(r, "duration", max(0.0, e - s))
    tk = str(r.get("topic_key") or "")
    return RangeLite(start=s, end=e, duration=d, topic_key=tk, raw=r)


def _contains_role_window(ranges: List[RangeLite], i: int, lo: int, hi: int, roles: List[str]) -> bool:
    lo_i = max(0, i + lo)
    hi_i = min(len(ranges) - 1, i + hi)
    for k in range(lo_i, hi_i + 1):
        role = str(ranges[k].raw.get("_major_role") or "").lower()
        if role in roles:
            return True
    return False


def _is_peak_like(r: RangeLite) -> bool:
    if str(r.raw.get("_major_role") or "").lower() == "peak":
        return True
    if _safe_bool(r.raw.get("_has_peak")):
        return True
    cz = float(r.raw.get("_avg_chat_z") or 0.0)
    if cz >= 0.6 and str(r.raw.get("energy") or "").lower() == "high":
        return True
    return False


def _has_resolution_like(r: RangeLite) -> bool:
    role = str(r.raw.get("_major_role") or "").lower()
    if role == "resolution":
        return True
    if _safe_bool(r.raw.get("_has_resolution")):
        return True
    if _safe_bool(r.raw.get("_resolution_evidence")):
        return True
    return False


def _arc_has_resolution_lite(acc: List[RangeLite]) -> bool:
    for it in acc:
        if _has_resolution_like(it):
            return True
    return False


def _topic_thread_key(r: RangeLite) -> str:
    return str(r.raw.get("_topic_thread") or "")


def _chapter_key(r: RangeLite) -> str:
    return str(r.raw.get("chapter_id") or "")


def detect_clutch_arcs(
    ranges_raw: List[Dict[str, Any]],
    retriever,
    target_min: float,
    target_max: float,
    sim_threshold: float,
    time_tau: float,
    topic_bonus: float,
    resolution_grace_seconds: float = 240.0,
    resolution_points: Optional[List[Dict[str, Any]]] = None,
) -> List[List[Dict[str, Any]]]:
    ranges = [_as_lite(r) for r in ranges_raw]
    arcs: List[List[RangeLite]] = []
    used = [False] * len(ranges)
    
    # Filter resolution points to strong ones only (for round-based snapping)
    strong_resolutions = []
    if resolution_points:
        strong_resolutions = sorted(
            [r for r in resolution_points if r['score'] >= 2.5],
            key=lambda x: x['ts']
        )
    for i, r in enumerate(ranges):
        if used[i]:
            continue
        if not _is_peak_like(r):
            continue
        # Extend left for build/conflict up to ~2 minutes or until target_min achieved
        acc: List[RangeLite] = [r]
        dur = r.duration
        # left extension
        j = i - 1
        while j >= 0 and dur < target_min:
            if used[j]:
                j -= 1
                continue
            prev = ranges[j]
            role = str(prev.raw.get("_major_role") or "").lower()
            if role in ("build_up", "conflict"):
                acc.insert(0, prev)
                dur += prev.duration
            else:
                sim = _similarity(prev, acc[0], retriever, topic_bonus, time_tau)
                if sim >= max(0.35, sim_threshold - 0.1):
                    acc.insert(0, prev)
                    dur += prev.duration
                else:
                    break
            j -= 1
        # right extension for resolution / near-peak continuity
        k = i + 1
        while k < len(ranges) and (dur < target_max or (dur < (target_max + max(0.0, float(resolution_grace_seconds))) and not _arc_has_resolution_lite(acc))):
            if used[k]:
                k += 1
                continue
            nxt = ranges[k]
            role = str(nxt.raw.get("_major_role") or "").lower()
            ok_role = (role == "resolution") or _contains_role_window(ranges, k, -1, 1, ["peak", "conflict", "resolution"]) 
            sim = _similarity(acc[-1], nxt, retriever, topic_bonus, time_tau)
            same_chapter = (_chapter_key(nxt) == _chapter_key(acc[0]))
            is_resolution_like = _has_resolution_like(nxt)
            if dur >= target_max:
                # After hitting target_max, require stronger conditions: resolution-like
                # or strong same-chapter continuity.
                if is_resolution_like or (same_chapter and sim >= sim_threshold):
                    acc.append(nxt)
                    dur += nxt.duration
                    k += 1
                    continue
                
                # ROUND-BASED MODE: Snap to strong resolution within grace window
                if strong_resolutions:
                    # Find strong resolution within grace window ahead of current position
                    current_end = acc[-1].end
                    target_resolution = None
                    for res in strong_resolutions:
                        res_ts = float(res['ts'])
                        if res_ts <= current_end:
                            continue  # Already passed
                        if res_ts > current_end + resolution_grace_seconds:
                            break  # Too far ahead
                        target_resolution = res
                        break  # Take first qualifying resolution
                    
                    if target_resolution:
                        # Find the range that contains or is closest to this resolution
                        res_ts = float(target_resolution['ts'])
                        bridge_end_index = -1
                        for lookahead_j in range(k, len(ranges)):
                            cand = ranges[lookahead_j]
                            if _chapter_key(cand) != _chapter_key(acc[0]):
                                break  # Don't cross chapters
                            # Check if resolution is within this range
                            if cand.start <= res_ts <= cand.end:
                                bridge_end_index = lookahead_j
                                break
                            # If we've gone past the resolution, take previous range
                            if cand.start > res_ts:
                                bridge_end_index = max(k, lookahead_j - 1)
                                break
                        
                        # Extend to include ranges up to and including the resolution
                        if bridge_end_index >= k:
                            while k <= bridge_end_index:
                                acc.append(ranges[k])
                                dur += ranges[k].duration
                                k += 1
                            # Successfully snapped to resolution - stop extending
                            break
                
                # FALLBACK: Lookahead for resolution-like range (original logic)
                lookahead_seconds = 120.0
                lookahead_j = k
                bridge_ok = False
                bridge_end_index = k
                last_end = acc[-1].end
                while lookahead_j < len(ranges):
                    cand = ranges[lookahead_j]
                    if _chapter_key(cand) != _chapter_key(acc[0]):
                        break
                    if (cand.end - acc[-1].start) > (dur + resolution_grace_seconds + 1.0):
                        break
                    if (cand.start - acc[-1].end) > lookahead_seconds:
                        break
                    if _has_resolution_like(cand):
                        bridge_ok = True
                        bridge_end_index = lookahead_j
                        break
                    last_end = cand.end
                    lookahead_j += 1
                if bridge_ok:
                    while k <= bridge_end_index:
                        acc.append(ranges[k])
                        dur += ranges[k].duration
                        k += 1
                    continue
                # Don't spill into a new chapter without resolution when already past target_max
                break
            # Before target_max, keep growing on role or similarity
            if ok_role or sim >= sim_threshold:
                acc.append(nxt)
                dur += nxt.duration
                k += 1
                continue
            break
        if dur >= max(0.65 * target_min, target_min - 60):
            for idx, rr in enumerate(ranges):
                # mark used within span to reduce overlap
                if acc[0].start <= rr.start and rr.end <= acc[-1].end:
                    used[idx] = True
            arcs.append(acc)
    # Convert back to raw dicts
    return [[r.raw for r in arc] for arc in arcs]


def detect_thread_arcs(
    ranges_raw: List[Dict[str, Any]],
    target_min: float,
    target_max: float,
) -> List[List[Dict[str, Any]]]:
    ranges = [_as_lite(r) for r in ranges_raw]
    arcs: List[List[Dict[str, Any]]] = []
    i = 0
    while i < len(ranges):
        thr = _topic_thread_key(ranges[i])
        if not thr:
            i += 1
            continue
        block = [ranges[i]]
        dur = ranges[i].duration
        j = i + 1
        while j < len(ranges) and _topic_thread_key(ranges[j]) == thr:
            block.append(ranges[j])
            dur += ranges[j].duration
            j += 1
        if dur >= target_min * 0.8 and dur <= target_max * 1.3:
            arcs.append([r.raw for r in block])
        i = j
    return arcs


def detect_chapter_round_arcs(
    ranges_raw: List[Dict[str, Any]],
    target_min: float,
    target_max: float,
) -> List[List[Dict[str, Any]]]:
    ranges = [_as_lite(r) for r in ranges_raw]
    arcs: List[List[Dict[str, Any]]] = []
    # Group contiguous by chapter_id
    i = 0
    while i < len(ranges):
        ch = _chapter_key(ranges[i])
        block = [ranges[i]]
        j = i + 1
        while j < len(ranges) and _chapter_key(ranges[j]) == ch:
            block.append(ranges[j])
            j += 1
        dur = sum(b.duration for b in block)
        roles = [str(b.raw.get("_major_role") or "").lower() for b in block]
        if dur >= target_min * 0.7 and dur <= target_max * 1.5 and ("peak" in roles) and ("resolution" in roles or "conflict" in roles):
            arcs.append([b.raw for b in block])
        i = j
    return arcs


def _contiguity_ratio(arc: List[Dict[str, Any]]) -> float:
    if not arc:
        return 0.0
    s = float(arc[0].get("start", 0.0))
    e = float(arc[-1].get("end", 0.0))
    span = max(0.0, e - s)
    covered = 0.0
    for r in arc:
        rs = float(r.get("start", 0.0))
        re = float(r.get("end", rs))
        covered += max(0.0, re - rs)
    if span <= 0.0:
        return 0.0
    return max(0.0, min(1.0, covered / span))


def _structure_score(arc: List[Dict[str, Any]]) -> float:
    roles = [str(r.get("_major_role") or "").lower() for r in arc]
    has_peak = ("peak" in roles) or any(bool(r.get("_has_peak")) for r in arc)
    has_res = ("resolution" in roles)
    has_conf = ("conflict" in roles)
    if has_peak and (has_res or has_conf):
        return 1.0
    # Same thread majority
    threads = [str(r.get("_topic_thread") or "") for r in arc if r.get("_topic_thread") is not None]
    if threads:
        most = max(threads, key=lambda x: threads.count(x))
        frac = threads.count(most) / float(len(threads))
        if frac >= 0.7:
            return 0.7
    # Same chapter majority
    chs = [str(r.get("chapter_id") or "") for r in arc]
    if chs:
        most2 = max(chs, key=lambda x: chs.count(x))
        frac2 = chs.count(most2) / float(len(chs))
        if frac2 >= 0.7:
            return 0.6
    return 0.0


def _signal_score(arc: List[Dict[str, Any]]) -> float:
    cz_vals = [max(0.0, float(r.get("_avg_chat_z") or 0.0)) for r in arc]
    if not cz_vals:
        return 0.0
    mx = max(cz_vals)
    # Normalize: cz of 1.0 treated as strong signal
    base = max(0.0, min(1.0, mx / 1.0))
    # Energy bump if any high
    has_high = any(str(r.get("energy") or "").lower() == "high" for r in arc)
    return max(0.0, min(1.0, base + (0.1 if has_high else 0.0)))


def _coherence_score(arc: List[Dict[str, Any]], retriever, topic_bonus: float, time_tau: float) -> float:
    if not arc:
        return 0.0
    if len(arc) == 1:
        return 0.8
    # Build RangeLite list
    rl = [_as_lite(r) for r in arc]
    # Adjacency sims (temporal continuity component)
    adj_sims = []
    for i in range(len(rl) - 1):
        adj_sims.append(_similarity(rl[i], rl[i + 1], retriever, topic_bonus, time_tau))
    adj_avg = (sum(adj_sims) / float(len(adj_sims))) if adj_sims else 0.0
    # Medoid anchor: pick element maximizing average sim to others
    if retriever and getattr(retriever, "have_index", False):
        n = len(rl)
        if n <= 1:
            return max(0.0, min(1.0, adj_avg))
        sim_matrix: List[List[float]] = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                s = _similarity(rl[i], rl[j], retriever, topic_bonus, time_tau)
                sim_matrix[i][j] = s
                sim_matrix[j][i] = s
        avg_sims = [sum(row) / float(max(1, n - 1)) for row in sim_matrix]
        medoid_idx = max(range(n), key=lambda k: avg_sims[k])
        medoid_row = sim_matrix[medoid_idx]
        # Top-K average (robust to noise)
        k = max(1, min(5, n - 1))
        topk = sorted([medoid_row[j] for j in range(n) if j != medoid_idx], reverse=True)[:k]
        medoid_topk = (sum(topk) / float(len(topk))) if topk else 0.0
        # Blend: heavier weight on medoid, keep adjacency influence
        return max(0.0, min(1.0, 0.65 * medoid_topk + 0.35 * adj_avg))
    return max(0.0, min(1.0, adj_avg))


def _switch_penalty(arc: List[Dict[str, Any]]) -> float:
    if len(arc) <= 1:
        return 0.0
    # Prefer continuity of _topic_thread; fall back to chapter_id; ignore empty values.
    def key_of(r: Dict[str, Any]) -> str:
        thr = r.get("_topic_thread")
        if thr is not None and str(thr) != "":
            return str(thr)
        ch = r.get("chapter_id")
        return str(ch) if ch is not None else ""
    total = 0
    switches = 0
    prev = key_of(arc[0])
    for i in range(1, len(arc)):
        cur = key_of(arc[i])
        if prev == "" and cur == "":
            continue
        total += 1
        if cur != prev:
            switches += 1
        prev = cur
    if total <= 0:
        return 0.0
    ratio = switches / float(total)
    # Cap the penalty; arcs with mixed threads shouldn't be auto-failed.
    capped = min(0.4, ratio)  # max 0.4 penalty
    return max(0.0, capped)


def compute_arc_score(
    arc: List[Dict[str, Any]],
    retriever,
    topic_bonus: float,
    time_tau: float,
) -> float:
    coh = _coherence_score(arc, retriever, topic_bonus, time_tau)
    cont = _contiguity_ratio(arc)
    stru = _structure_score(arc)
    sig = _signal_score(arc)
    swp = _switch_penalty(arc)
    # Emphasize semantic cohesion; penalize switches more
    score = 0.6 * coh + 0.15 * cont + 0.15 * stru + 0.10 * sig - 0.25 * swp
    return max(0.0, min(1.0, score))


def nms_by_score(arcs: List[List[Dict[str, Any]]], scores: List[float]) -> List[List[Dict[str, Any]]]:
    if not arcs:
        return []
    items = []
    for arc, sc in zip(arcs, scores):
        s = float(arc[0].get("start", 0.0))
        e = float(arc[-1].get("end", 0.0))
        items.append((s, e, sc, arc))
    # Sort by start then by score desc
    items.sort(key=lambda t: (t[0], -t[2]))
    out: List[List[Dict[str, Any]]] = []
    kept_intervals: List[Tuple[float, float]] = []
    for s, e, sc, arc in items:
        overlapped = False
        for ks, ke in kept_intervals:
            if not (e <= ks or s >= ke):
                # overlap
                overlapped = True
                break
        if overlapped:
            # Compare with the last kept if overlapping only with last; else skip
            # Simple policy: skip lower-scored overlapping arcs
            continue
        out.append(arc)
        kept_intervals.append((s, e))
    return out


def _arc_span_seconds(arc: List[Dict[str, Any]]) -> float:
    if not arc:
        return 0.0
    s = float(arc[0].get("start", 0.0))
    e = float(arc[-1].get("end", 0.0))
    return max(0.0, e - s)


def _arc_total_seconds(arc: List[Dict[str, Any]]) -> float:
    tot = 0.0
    for r in arc:
        s = float(r.get("start", 0.0))
        e = float(r.get("end", s))
        tot += max(0.0, e - s)
    return tot


def _arc_max_internal_gap(arc: List[Dict[str, Any]]) -> float:
    if not arc or len(arc) <= 1:
        return 0.0
    mx = 0.0
    for i in range(len(arc) - 1):
        gap = max(0.0, float(arc[i + 1].get("start", 0.0)) - float(arc[i].get("end", 0.0)))
        if gap > mx:
            mx = gap
    return mx


def detect_arcs(
    ranges_raw: List[Dict[str, Any]],
    retriever,
    target_min: float,
    target_max: float,
    sim_threshold: float,
    time_tau: float,
    topic_bonus: float,
    min_score: float = 0.7,
    debug: bool = False,
    min_contiguity: float = 0.6,
    max_gap_seconds: float = 180.0,
    max_gap_fraction: float = 0.4,
    resolution_grace_seconds: float = 240.0,
    resolution_points: Optional[List[Dict[str, Any]]] = None,
) -> List[List[Dict[str, Any]]]:
    if not ranges_raw:
        return []
    
    # Calculate resolution density to detect round-based games
    resolution_mode = False
    if resolution_points:
        total_hours = (ranges_raw[-1]['end'] - ranges_raw[0]['start']) / 3600.0
        strong_resolutions = [r for r in resolution_points if r['score'] >= 2.0]
        density = len(strong_resolutions) / max(0.1, total_hours)
        resolution_mode = (density >= 3.0)  # High density = round-based game
        if debug:
            print(f"[arc-debug] Resolution density: {density:.2f}/hour ({len(strong_resolutions)} strong in {total_hours:.1f}h) → mode={'ROUND-BASED' if resolution_mode else 'CONTINUOUS'}")
    
    # Apply multiple detectors
    candidates = []
    candidates += detect_clutch_arcs(
        ranges_raw,
        retriever,
        target_min,
        target_max,
        sim_threshold,
        time_tau,
        topic_bonus,
        resolution_grace_seconds=resolution_grace_seconds,
        resolution_points=resolution_points if resolution_mode else None,
    )
    candidates += detect_thread_arcs(ranges_raw, target_min, target_max)
    candidates += detect_chapter_round_arcs(ranges_raw, target_min, target_max)
    # Score and gate
    scores = [compute_arc_score(arc, retriever, topic_bonus, time_tau) for arc in candidates]
    # Special accept rule: arcs with clear peak structure and good contiguity can pass with lower score
    def _special_accept(arc: List[Dict[str, Any]], sc: float) -> bool:
        if sc >= float(min_score):
            return True
        stru = _structure_score(arc)
        cont = _contiguity_ratio(arc)
        # Apply stronger contiguity constraint in special rule
        return (stru >= 1.0 and cont >= max(0.85, float(min_contiguity)) and sc >= max(0.55, float(min_score) - 0.1))

    # Apply contiguity and gap constraints
    def _passes_gaps(arc: List[Dict[str, Any]]) -> bool:
        span = _arc_span_seconds(arc)
        if span <= 0.0:
            return False
        tot = _arc_total_seconds(arc)
        cont = max(0.0, min(1.0, tot / span))
        if cont < float(min_contiguity):
            return False
        max_gap = _arc_max_internal_gap(arc)
        if max_gap > float(max_gap_seconds):
            return False
        gap_frac = max(0.0, min(1.0, (span - tot) / span))
        if gap_frac > float(max_gap_fraction):
            return False
        return True

    gated_flags = [_special_accept(arc, sc) and _passes_gaps(arc) for arc, sc in zip(candidates, scores)]
    gated = [arc for arc, ok in zip(candidates, gated_flags) if ok]
    gated_scores = [sc for sc, ok in zip(scores, gated_flags) if ok]
    if debug:
        try:
            import json as _json
            dbg = []
            for arc, sc in zip(candidates, scores):
                dbg.append({
                    "start": float(arc[0].get("start", 0.0)) if arc else 0.0,
                    "end": float(arc[-1].get("end", 0.0)) if arc else 0.0,
                    "score": round(sc, 3),
                    "coherence": round(_coherence_score(arc, retriever, topic_bonus, time_tau), 3),
                    "contiguity": round(_contiguity_ratio(arc), 3),
                    "structure": round(_structure_score(arc), 3),
                    "signal": round(_signal_score(arc), 3),
                    "switch_penalty": round(_switch_penalty(arc), 3),
                    "span_seconds": round(_arc_span_seconds(arc), 3),
                    "max_internal_gap": round(_arc_max_internal_gap(arc), 3),
                    "gap_fraction": round(max(0.0, min(1.0, (_arc_span_seconds(arc) - _arc_total_seconds(arc)) / max(1e-6, _arc_span_seconds(arc)))), 3),
                    "accepted": bool(_special_accept(arc, sc)),
                    "len": len(arc),
                })
            # Print one line JSON for quick inspection
            print("[arc-debug]", _json.dumps(dbg[:50]))
        except Exception:
            pass
    # NMS by score to prevent overlaps
    final_arcs = nms_by_score(gated, gated_scores)
    # Sort final arcs by start time
    final_arcs.sort(key=lambda arc: float(arc[0].get("start", 0.0)))
    return final_arcs


