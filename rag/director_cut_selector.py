#!/usr/bin/env python3
"""
Director's Cut Selector

Applies deterministic keep/cut policy per chapter using roles, anchors,
per-chapter quantiles, and tiny retrieval windows (no new LLM calls here).
Outputs a flattened manifest of [start,end] ranges per peak block and
updates DB with keep_flag, peak_block_id, anchor_burst_id.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from rag.retrieval import (
    load_bursts,
    group_by_chapter,
    chapter_stats,
    compute_salience,
    detect_anchors,
    quantile,
    load_retriever,
    Retriever,
)


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def build_keep_set_gameplay(ch_bursts: List[Dict]) -> List[bool]:
    keep = [False]*len(ch_bursts)
    anchors = detect_anchors(ch_bursts)
    for i, b in enumerate(ch_bursts):
        if anchors[i]:
            keep[i] = True
        if b.get("role") in {"peak","conflict","resolution"}:
            keep[i] = True
    return keep


def build_keep_set_jc(ch_bursts: List[Dict]) -> List[bool]:
    """Keep almost everything in JC chapters.

    We drop only segments explicitly labelled as *afk* or *tech_issue* —
    conservative so the conversation stays coherent.  Any other role is kept.
    """
    drop_roles = {"afk", "tech_issue"}

    def is_lively(b: Dict) -> bool:
        """Return True if chat or reactions suggest active engagement."""
        if (b.get("chat_rate_z", 0.0) or 0.0) > 0.3:
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
    return keep


def context_wrap(ch_bursts: List[Dict], keep: List[bool]) -> None:
    # immediate neighbor wrap
    n = len(ch_bursts)
    for i in range(n):
        if not keep[i]:
            continue
        # left
        if i-1 >= 0:
            prev = ch_bursts[i-1]
            if prev.get("role") in {"build_up","talking","resolution"} and ch_bursts[i].get("same_topic_prev", False):
                keep[i-1] = True
        # right
        if i+1 < n:
            nxt = ch_bursts[i+1]
            if nxt.get("role") in {"resolution","talking"} and (nxt.get("same_topic_prev", False) or ch_bursts[i].get("role") in {"peak","conflict"}):
                keep[i+1] = True


def pad_around_importants(
    ch_bursts: List[Dict],
    keep: List[bool],
    stats: Dict[str, float],
    retriever: Retriever,
    is_game: bool,
) -> None:
    """Wrap anchors/peak/conflict/resolution with time padding around them.
    Gameplay: use tighter, asymmetric padding with topic continuity bias.
    JC: keep generous symmetric padding (dur_q50 each side)."""
    n = len(ch_bursts)
    q25 = stats.get("dur_q25", 0.0)
    q50 = stats.get("dur_q50", 0.0)
    if is_game:
        pad_left = q25
        pad_right = min(max(q25, 0.4 * q50), 12.0)
    else:
        pad_left = q50
        pad_right = q50

    for i, b in enumerate(ch_bursts):
        important = (b.get("role") in {"peak","conflict","resolution"}) or (sum((b.get("reaction_hits") or {}).values()) > 0)
        if not important:
            continue

        anchor_start = b["start_time"]
        anchor_end = b["end_time"]
        anchor_id = b["id"]

        # left padding (favor semantic continuity when gameplay)
        j = i - 1
        while j >= 0 and (anchor_start - ch_bursts[j]["start_time"]) <= pad_left:
            if not is_game:
                keep[j] = True
            else:
                if retriever.have_index and retriever.sim(anchor_id, ch_bursts[j]["id"]) >= 0.30:
                    keep[j] = True
            j -= 1

        # right padding
        k = i + 1
        while k < n and (ch_bursts[k]["end_time"] - anchor_end) <= pad_right:
            if not is_game:
                keep[k] = True
            else:
                if retriever.have_index and retriever.sim(anchor_id, ch_bursts[k]["id"]) >= 0.30:
                    keep[k] = True
            k += 1


def smooth_fill_small_gaps(
    ch_bursts: List[Dict],
    keep: List[bool],
    retriever: Retriever,
    is_game: bool,
    stats: Dict[str, float],
    max_gap_s: float = 15.0,
) -> None:
    """If two kept bursts are close, keep in-between bursts.
    Gameplay: stricter gap (<= min(8s, dur_q25)), require topic continuity, and skip AFK/filler unless lively.
    JC: keep permissive behavior."""
    if is_game:
        max_gap_s = min(8.0, stats.get("dur_q25", 8.0) or 8.0)
    n = len(ch_bursts)
    i = 0
    while i < n:
        if not keep[i]:
            i += 1
            continue
        # find next kept
        j = i + 1
        while j < n and not keep[j]:
            j += 1
        if j < n:
            gap = max(0.0, ch_bursts[j]["start_time"] - ch_bursts[i]["end_time"]) 
            if gap <= max_gap_s:
                if not is_game:
                    for k in range(i+1, j):
                        keep[k] = True
                else:
                    key_i = ch_bursts[i].get("topic_key")
                    key_j = ch_bursts[j].get("topic_key")
                    same_topic = key_i and key_j and key_i == key_j
                    if same_topic and retriever.have_index:
                        anchor_id = ch_bursts[i]["id"]
                        for k in range(i + 1, j):
                            b = ch_bursts[k]
                            role = b.get("role")
                            lively = (b.get("chat_rate_z") or 0) > 0.5 or sum((b.get("reaction_hits") or {}).values()) > 0
                            if (
                                retriever.sim(anchor_id, b["id"]) >= 0.30
                                and role != "afk"
                                and (role != "filler" or lively)
                            ):
                                keep[k] = True
        i = j


def salience_guard_gameplay(ch_bursts: List[Dict], keep: List[bool]) -> List[bool]:
    # drop build_up/talking below median; ratchet to p60/p70 if compression < 10%
    sal = compute_salience(ch_bursts)
    durations = [b["end_time"]-b["start_time"] for b in ch_bursts]
    total = sum(durations)

    def apply_guard(p: float) -> List[bool]:
        cut = quantile(sal, p)
        out = keep[:]
        for i, b in enumerate(ch_bursts):
            if out[i]:
                continue
            role = b.get("role")
            if role in {"build_up","talking"} and sal[i] < cut:
                out[i] = False  # explicit
        return out

    # compute current kept time
    def kept_time(mask: List[bool]) -> float:
        return sum(d for d, m in zip(durations, mask) if m)

    # first pass at median
    out = keep[:]
    cut_med = quantile(sal, 0.50)
    for i, b in enumerate(ch_bursts):
        if out[i]:
            continue
        role = b.get("role")
        if role in {"build_up","talking"} and sal[i] < cut_med:
            out[i] = False
    compressed = 1.0 - (kept_time(out) / max(total, 1e-6))
    if compressed < 0.10:
        out = apply_guard(0.60)
        compressed = 1.0 - (kept_time(out) / max(total, 1e-6))
        if compressed < 0.10:
            out = apply_guard(0.70)
    return out


def label_peak_blocks(ch_bursts: List[Dict], keep: List[bool]) -> List[Optional[str]]:
    # gap threshold = clamp(p90, 30s, 120s)
    gaps = []
    for i in range(len(ch_bursts) - 1):
        gaps.append(max(0.0, ch_bursts[i+1]["start_time"] - ch_bursts[i]["end_time"]))
    gap_p90 = quantile(gaps, 0.90) if gaps else 0.0
    thr = clamp(gap_p90, 30.0, 120.0)

    block_ids: List[Optional[str]] = [None]*len(ch_bursts)
    block_counter = 0

    i = 0
    while i < len(ch_bursts):
        if not keep[i]:
            i += 1
            continue
        important = (ch_bursts[i].get("role") in {"peak","conflict","resolution"}) or (sum((ch_bursts[i].get("reaction_hits") or {}).values()) > 0)
        if not important:
            i += 1
            continue
        block_counter += 1
        bid = f"PB-{block_counter:03d}"
        block_ids[i] = bid
        j = i + 1
        while j < len(ch_bursts) and keep[j]:
            gap = max(0.0, ch_bursts[j]["start_time"] - ch_bursts[j-1]["end_time"]) 
            if gap > thr:
                break
            rolej = ch_bursts[j].get("role")
            important_j = rolej in {"peak","conflict","resolution"} or (sum((ch_bursts[j].get("reaction_hits") or {}).values()) > 0)
            if important_j or (rolej == "build_up" and ch_bursts[j].get("same_topic_prev", False)):
                block_ids[j] = bid
                j += 1
            else:
                break
        i = j
    return block_ids


def drop_kept_loners_gameplay(ch_bursts: List[Dict], keep: List[bool], stats: Dict[str, float]) -> List[bool]:
    """Un-keep standalone short slices with no anchor and low salience.
    Conditions: kept[i] True, neighbors not kept, duration ≤ 12s, no reactions,
    role not in {peak, conflict, resolution}, and salience < median."""
    sal = compute_salience(ch_bursts)
    cut_med = quantile(sal, 0.50)
    out = keep[:]
    n = len(ch_bursts)
    for i, b in enumerate(ch_bursts):
        if not out[i]:
            continue
        left_kept = out[i-1] if i-1 >= 0 else False
        right_kept = out[i+1] if i+1 < n else False
        if left_kept or right_kept:
            continue
        dur = b["end_time"] - b["start_time"]
        if dur > 12.0:
            continue
        if b.get("role") in {"peak","conflict","resolution"}:
            continue
        if sum((b.get("reaction_hits") or {}).values()) > 0:
            continue
        if sal[i] >= cut_med:
            continue
        out[i] = False
    return out


def flatten_manifest(ch_bursts: List[Dict], keep: List[bool], block_ids: List[Optional[str]], is_game: bool) -> List[Dict]:
    # Merge adjacent kept bursts per (chapter_id, block_id) with gap ≤ 15.0s
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
                "anchor_burst_id": b["id"] if sum((b.get("reaction_hits") or {}).values())>0 else None,
                "topic_key_votes": [b.get("topic_key")],
                "energies": [b.get("energy")],
                "summaries": [b.get("summary")],
            }
            continue
        # same block and small gap
        if is_game:
            same_block = (current.get("peak_block_id") is not None) and (bid is not None) and (current.get("peak_block_id") == bid)
            gap = max(0.0, start - current["end"]) 
            gap_ok = gap <= 8.0
        else:
            same_block = (current.get("peak_block_id") == bid)
            gap = max(0.0, start - current["end"]) 
            gap_ok = gap <= 15.0
        if same_block and gap_ok:
            current["end"] = end
            current["burst_ids"].append(b["id"])
            if (not current.get("anchor_burst_id")) and sum((b.get("reaction_hits") or {}).values())>0:
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
                "anchor_burst_id": b["id"] if sum((b.get("reaction_hits") or {}).values())>0 else None,
                "topic_key_votes": [b.get("topic_key")],
                "energies": [b.get("energy")],
                "summaries": [b.get("summary")],
            }
    push_current()

    # finalize fields
    for m in out:
        # majority topic_key
        votes = [v for v in m["topic_key_votes"] if v]
        maj = None
        if votes:
            counts: Dict[str,int] = {}
            for v in votes:
                counts[v] = counts.get(v,0)+1
            maj = max(counts, key=counts.get)
        m["topic_key"] = maj or ""
        # energy mode or max
        order = {"low":0,"medium":1,"high":2}
        rev = {v:k for k,v in order.items()}
        ranks = [order.get(e,1) for e in m["energies"]]
        m["energy"] = rev[max(ranks)] if ranks else "medium"
        # stitched summary
        summaries = [s for s in m["summaries"] if s]
        m["summary"] = " → ".join(summaries[:8])
        del m["topic_key_votes"]
        del m["energies"]
        del m["summaries"]
    return out


def update_db_keep_blocks(vod_id: str, chapter_id: str, ch_bursts: List[Dict], keep: List[bool], block_ids: List[Optional[str]]):
    db_path = Path(f"data/vector_stores/{vod_id}/metadata.db")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    # auto-migrate columns
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
        cur.execute("UPDATE documents SET keep_flag=?, peak_block_id=? WHERE id=?", (1 if k else 0, pb, b["id"]))
    conn.commit()
    conn.close()


def is_jc_chapter(ch_bursts: List[Dict]) -> bool:
    keys = [(b.get("topic_key") or "").lower() for b in ch_bursts]
    topics = [(b.get("topic") or "").lower() for b in ch_bursts]
    blob = " ".join(keys + topics)
    return ("just chatting" in blob) or ("just_chatting" in blob) or ("jc" in keys)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Select director's cut ranges per chapter")
    parser.add_argument("vod_id", help="VOD ID")
    args = parser.parse_args()

    bursts = load_bursts(args.vod_id)
    retriever = load_retriever(args.vod_id)
    by_chap = group_by_chapter(bursts)

    manifests: List[Dict] = []

    for chap_id, ch_bursts in by_chap.items():
        jc = is_jc_chapter(ch_bursts)
        is_game = not jc
        # keep set
        keep = build_keep_set_gameplay(ch_bursts) if is_game else build_keep_set_jc(ch_bursts)
        # context wrap (no merge, just adjacent neighbors)
        context_wrap(ch_bursts, keep)
        # padding around important bursts (anchors/peak/conflict/resolution)
        stats = chapter_stats(ch_bursts)
        pad_around_importants(ch_bursts, keep, stats, retriever, is_game)
        # smoothing: fill small gaps up to 15s
        smooth_fill_small_gaps(ch_bursts, keep, retriever, is_game, stats, max_gap_s=15.0)
        # salience guard ratchet (gameplay only)
        if is_game:
            keep = salience_guard_gameplay(ch_bursts, keep)
            # drop isolated micro-keeps
            keep = drop_kept_loners_gameplay(ch_bursts, keep, stats)
        # peak blocks
        blocks = label_peak_blocks(ch_bursts, keep)
        # update DB
        update_db_keep_blocks(args.vod_id, chap_id, ch_bursts, keep, blocks)
        # flatten manifest ranges
        man = flatten_manifest(ch_bursts, keep, blocks, is_game)
        for m in man:
            m["vod_id"] = args.vod_id
        manifests.extend(man)
        # logging per chapter
        total_sec = sum((b["end_time"]-b["start_time"]) for b in ch_bursts)
        kept_sec = sum((b["end_time"]-b["start_time"]) for b, k in zip(ch_bursts, keep) if k)
        cut_ratio = 0.0 if total_sec <= 0 else (1.0 - kept_sec/total_sec)
        print(f"Chapter {chap_id}: kept {kept_sec/60:.1f}m / {total_sec/60:.1f}m ({cut_ratio*100:.1f}% cut), ranges={len(man)}")

    # write manifest
    out_dir = Path(f"data/vector_stores/{args.vod_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sorted_ranges = sorted(manifests, key=lambda x: (x["chapter_id"], x["start"]))
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
        "rag_threshold": 0.30,
        "total_ranges": len(sorted_ranges),
        "total_duration_seconds": round(total_seconds, 3),
        "total_duration_minutes": round(total_seconds / 60.0, 2),
        "total_duration_hms": _format_hms(total_seconds),
        "ranges": sorted_ranges,
    }

    (out_dir / "director_cut_manifest.json").write_text(json.dumps(manifest_obj, indent=2), encoding="utf-8")

    print(f"✅ Director's cut manifest written: {out_dir / 'director_cut_manifest.json'}")


if __name__ == "__main__":
    main()
