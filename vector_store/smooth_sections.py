#!/usr/bin/env python3
"""
Smoothing: Merge small slices into coherent sections per chapter.

- Anchor-first: compute per-chapter salience and pick anchors via quantiles + local max + NMS
- Grow anchors symmetrically by absorbing short/low-salience neighbors that are close and similar
- Attach orphans to best neighboring section
- Update DB: set section_id/section_title on documents
- Export sections to JSON per VOD (and per chapter)
"""

import json
import math
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from vector_store.vector_index import VectorIndex  # for path discovery only

# ---------- Utilities ----------

def quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    idx = max(0, min(len(v) - 1, int(q * (len(v) - 1))))
    return float(v[idx])


def rank_to_unit(values: List[float]) -> List[float]:
    if not values:
        return []
    sorted_vals = sorted(set(values))
    ranks = {val: i for i, val in enumerate(sorted_vals)}
    denom = max(1, len(sorted_vals) - 1)
    return [ranks[v] / denom for v in values]


def jaccard_tokens(a: str, b: str) -> float:
    def norm(s: str) -> List[str]:
        import re
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        tokens = [t for t in s.split() if t and t not in {"the","a","an","and","or","to","of","for","in"}]
        # naive stemming: strip common suffixes
        stem = []
        for t in tokens:
            for suf in ("ing","ed","es","s"):
                if len(t) > 4 and t.endswith(suf):
                    t = t[: -len(suf)]
                    break
            stem.append(t)
        return stem
    A, B = set(norm(a or "")), set(norm(b or ""))
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0


def topic_sim(a: Dict, b: Dict) -> float:
    tk_a = (a.get("topic_key") or a.get("topic") or "").lower()
    tk_b = (b.get("topic_key") or b.get("topic") or "").lower()
    if not tk_a and not tk_b:
        return 0.0
    if tk_a == tk_b and tk_a:
        return 1.0
    return jaccard_tokens(tk_a, tk_b)


def local_max_salience(segments: List[Dict], idx: int) -> bool:
    s = segments[idx]["_salience"]
    left = segments[idx - 1]["_salience"] if idx - 1 >= 0 else -1.0
    right = segments[idx + 1]["_salience"] if idx + 1 < len(segments) else -1.0
    return s >= left and s >= right and (s > left or s > right)


def compute_adjacent_gaps(segments: List[Dict]) -> List[float]:
    gaps = []
    for i in range(len(segments) - 1):
        gaps.append(max(0.0, segments[i+1]["start_time"] - segments[i]["end_time"]))
    return gaps


def suppress_dense_anchors(segments: List[Dict], anchor_idxs: List[int], min_separation: float) -> List[int]:
    if not anchor_idxs:
        return []
    kept: List[int] = []
    last_keep_end = -1e9
    for idx in anchor_idxs:
        start = segments[idx]["start_time"]
        if not kept:
            kept.append(idx)
            last_keep_end = segments[idx]["end_time"]
            continue
        # if too close to last kept anchor by time span, keep the higher-salience
        if start - last_keep_end < min_separation:
            if segments[idx]["_salience"] > segments[kept[-1]]["_salience"]:
                kept[-1] = idx
                last_keep_end = segments[idx]["end_time"]
        else:
            kept.append(idx)
            last_keep_end = segments[idx]["end_time"]
    return kept


def merge_chunk(chunk: List[Dict]) -> Dict:
    start = chunk[0]["start_time"]
    end = chunk[-1]["end_time"]
    duration = end - start
    # majority topic_key/topic
    def majority(items: List[Optional[str]]) -> str:
        counts: Dict[str,int] = {}
        for it in items:
            if not it:
                continue
            it = str(it)
            counts[it] = counts.get(it, 0) + 1
        if not counts:
            return ""
        return max(counts, key=counts.get)

    topic_key = majority([c.get("topic_key") for c in chunk])
    topic = chunk[0].get("topic") or majority([c.get("topic") for c in chunk])
    # energy mode or max rank
    order = {"low":0,"medium":1,"high":2}
    rev = {v:k for k,v in order.items()}
    energies = [order.get((c.get("energy") or "medium"),1) for c in chunk]
    energy = rev[max(energies)]
    # summary compact join
    def trim_sentence(t: str) -> str:
        t = (t or "").strip()
        return t if len(t) <= 140 else t[:137] + "…"
    summary = " → ".join(trim_sentence(c.get("summary","")) for c in chunk if c.get("summary"))
    # confidence max, reactions sum
    confidence = max((c.get("confidence",0.0) for c in chunk), default=0.0)
    reactions_sum = sum(sum((c.get("reaction_hits") or {}).values()) for c in chunk)
    chat_rate_z_max = max((c.get("chat_rate_z",0.0) for c in chunk), default=0.0)

    merged = {
        "start_time": start,
        "end_time": end,
        "duration": duration,
        "summary": summary,
        "topic": topic,
        "topic_key": topic_key,
        "energy": energy,
        "confidence": confidence,
        "reactions_sum": reactions_sum,
        "chat_rate_z_max": chat_rate_z_max,
        "burst_ids": [c["id"] for c in chunk],
    }
    return merged


# ---------- Core smoothing ----------

def smooth_segments_one_chapter(segments: List[Dict]) -> List[Dict]:
    if not segments:
        return []

    # 0) Stats & helpers
    durations = [s["end_time"] - s["start_time"] for s in segments]
    q25, q50, q75 = quantile(durations, 0.25), quantile(durations, 0.50), quantile(durations, 0.75)
    is_short = lambda d: d <= q25

    gaps = compute_adjacent_gaps(segments)
    gap_p90 = quantile(gaps, 0.90) if gaps else 0.0
    is_gap_ok = lambda gap: gap <= gap_p90

    norm_burst = rank_to_unit([s.get("burst_score",0.0) for s in segments])
    norm_chat  = rank_to_unit([max(0.0, s.get("chat_rate_z",0.0)) for s in segments])
    norm_react = rank_to_unit([sum((s.get("reaction_hits") or {}).values()) for s in segments])
    norm_conf  = rank_to_unit([s.get("confidence",0.0) for s in segments])

    for i, s in enumerate(segments):
        s["_salience"] = 0.45*norm_burst[i] + 0.25*norm_chat[i] + 0.20*norm_react[i] + 0.10*norm_conf[i]

    # Adaptive anchor percentile in [0.70, 0.90]
    def pick_anchor_cut():
        total_minutes = (segments[-1]["end_time"] - segments[0]["start_time"]) / 60.0
        target_min = max(1, int(total_minutes // 5))        # ~1 per 5m
        target_max = max(2, int(total_minutes // 5) * 3)    # up to ~3 per 5m
        for p in [0.70, 0.75, 0.80, 0.85, 0.90]:
            cut = quantile([s["_salience"] for s in segments], p)
            cand = [i for i in range(len(segments)) if segments[i]["_salience"] >= cut or local_max_salience(segments, i)]
            if target_min <= len(cand) <= max(target_max, target_min):
                return cut
        # fallback to 0.80
        return quantile([s["_salience"] for s in segments], 0.80)

    anchor_cut = pick_anchor_cut()

    # 1) Mark anchors
    anchor_idxs = [i for i in range(len(segments)) if segments[i]["_salience"] >= anchor_cut or local_max_salience(segments, i)]
    # NMS with min separation = median duration
    anchor_idxs = suppress_dense_anchors(segments, sorted(anchor_idxs, key=lambda i: segments[i]["start_time"]), q50)

    for s in segments:
        s["_anchor"] = False
    for i in anchor_idxs:
        segments[i]["_anchor"] = True

    # 2) Grow each anchor into a segment (symmetric greedy)
    used = [False]*len(segments)
    groups: List[Tuple[int,int]] = []

    def seg_sim(a: Dict, b: Dict) -> float:
        # topic_key fuzzy or same_topic_prev as a boost
        sim = topic_sim(a, b)
        if b.get("same_topic_prev") and a.get("id") == b.get("id"):  # trivial
            pass
        return sim

    def estimate_gap_threshold() -> float:
        return gap_p90

    for i, s in enumerate(segments):
        if used[i] or not s["_anchor"]:
            continue
        left = right = i
        used[i] = True
        # Expand until no candidate meets criteria
        while True:
            best_dir = None
            best_score = -1.0

            # check left
            j = left - 1
            if j >= 0 and not used[j] and not segments[j]["_anchor"]:
                gap_ok = is_gap_ok(max(0.0, segments[left]["start_time"] - segments[j]["end_time"]))
                if gap_ok:
                    dur = segments[j]["end_time"] - segments[j]["start_time"]
                    low_sal = segments[j]["_salience"] < anchor_cut
                    sim = seg_sim(segments[j], segments[left])
                    good_sim = sim >= 0.5 or segments[left].get("same_topic_prev", False)
                    strong_boundary = (not good_sim and sim < 0.3) or (segments[j]["_salience"] >= anchor_cut)
                    if is_short(dur) and low_sal and not strong_boundary:
                        score = 0.6*sim + 0.3*(1.0/(1.0 + max(1e-6, segments[left]["start_time"] - segments[j]["end_time"]))) + 0.1*segments[left]["_salience"]
                        best_dir = ("left", j, score) if score > best_score else best_dir
                        best_score = max(best_score, score)

            # check right
            k = right + 1
            if k < len(segments) and not used[k] and not segments[k]["_anchor"]:
                gap_ok = is_gap_ok(max(0.0, segments[k]["start_time"] - segments[right]["end_time"]))
                if gap_ok:
                    dur = segments[k]["end_time"] - segments[k]["start_time"]
                    low_sal = segments[k]["_salience"] < anchor_cut
                    sim = seg_sim(segments[right], segments[k])
                    good_sim = sim >= 0.5 or segments[k].get("same_topic_prev", False)
                    strong_boundary = (not good_sim and sim < 0.3) or (segments[k]["_salience"] >= anchor_cut)
                    if is_short(dur) and low_sal and not strong_boundary:
                        score = 0.6*sim + 0.3*(1.0/(1.0 + max(1e-6, segments[k]["start_time"] - segments[right]["end_time"]))) + 0.1*segments[right]["_salience"]
                        if score > best_score or (abs(score - best_score) < 1e-6 and best_dir and best_dir[0] == "right"):
                            best_dir = ("right", k, score)
                            best_score = score

            if best_dir is None:
                break

            direction, idx_cand, _ = best_dir
            used[idx_cand] = True
            if direction == "left":
                left = idx_cand
            else:
                right = idx_cand

            # soft max segment length cap (p95) to avoid over-merge
            curr_duration = segments[right]["end_time"] - segments[left]["start_time"]
            if curr_duration > quantile(durations, 0.95) or curr_duration > 8*60:
                break

        groups.append((left, right))

    # 3) Assign remaining orphans
    def nearest_used_left(i: int) -> Optional[int]:
        j = i - 1
        while j >= 0:
            if any(l <= j <= r for (l, r) in groups):
                return j
            j -= 1
        return None

    def nearest_used_right(i: int) -> Optional[int]:
        j = i + 1
        while j < len(segments):
            if any(l <= j <= r for (l, r) in groups):
                return j
            j += 1
        return None

    def group_of_index(idx: int) -> Optional[int]:
        for gi, (l, r) in enumerate(groups):
            if l <= idx <= r:
                return gi
        return None

    def attach_score(anchor_idx: int, free_idx: int) -> float:
        # anchor index denotes a member of a group; use its neighbors for salience
        sim = seg_sim(segments[anchor_idx], segments[free_idx])
        gap = 0.0
        if anchor_idx < free_idx:
            gap = max(0.0, segments[free_idx]["start_time"] - segments[anchor_idx]["end_time"]) 
        else:
            gap = max(0.0, segments[anchor_idx]["start_time"] - segments[free_idx]["end_time"]) 
        return 0.6*sim + 0.3*(1.0/(1.0 + gap)) + 0.1*segments[anchor_idx]["_salience"]

    for i, s in enumerate(segments):
        if any(l <= i <= r for (l, r) in groups):
            continue
        left_idx = nearest_used_left(i)
        right_idx = nearest_used_right(i)
        best = None
        best_sc = -1.0
        if left_idx is not None:
            gi = group_of_index(left_idx)
            if gi is not None:
                sc = attach_score(left_idx, i)
                best = (gi, sc, left_idx)
                best_sc = sc
        if right_idx is not None:
            gi = group_of_index(right_idx)
            if gi is not None:
                sc = attach_score(right_idx, i)
                if sc > best_sc:
                    best = (gi, sc, right_idx)
                    best_sc = sc
        if best is not None:
            gi, _, _ = best
            l, r = groups[gi]
            if i < l:
                groups[gi] = (i, r)
            elif i > r:
                groups[gi] = (l, i)
            else:
                # inside range, ignore
                pass
        else:
            groups.append((i, i))

    # 3.5) Absorb remaining short singletons into stronger neighbor (final cleanup)
    def group_rep_index(gr: Tuple[int,int]) -> int:
        l, r = gr
        best_idx = l
        best_sal = segments[l]["_salience"]
        for t in range(l, r+1):
            if segments[t]["_salience"] > best_sal:
                best_sal = segments[t]["_salience"]
                best_idx = t
        return best_idx

    groups = sorted(groups, key=lambda g: segments[g[0]]["start_time"])
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(groups):
            l, r = groups[i]
            if l == r:
                s = segments[l]
                dur = s["end_time"] - s["start_time"]
                if is_short(dur) and s["_salience"] < anchor_cut:
                    # evaluate left/right attach
                    best_side = None
                    best_sc = -1.0
                    # left
                    if i - 1 >= 0:
                        left_rep = group_rep_index(groups[i-1])
                        sc = attach_score(left_rep, l)
                        best_side = ("left", i-1, sc)
                        best_sc = sc
                    # right
                    if i + 1 < len(groups):
                        right_rep = group_rep_index(groups[i+1])
                        sc = attach_score(right_rep, l)
                        if sc > best_sc:
                            best_side = ("right", i+1, sc)
                            best_sc = sc
                    if best_side is not None and best_sc > 0:
                        side, idx_neighbor, _ = best_side
                        if side == "left":
                            nl, nr = groups[idx_neighbor]
                            groups[idx_neighbor] = (min(nl, l), max(nr, r))
                            groups.pop(i)
                        else:
                            nl, nr = groups[idx_neighbor]
                            groups[idx_neighbor] = (min(l, nl), max(r, nr))
                            groups.pop(i)
                        changed = True
                        continue  # do not increment i; list shrunk
            i += 1

    # 4) Materialize merged segments
    merged: List[Dict] = []
    for (l, r) in sorted(groups, key=lambda g: segments[g[0]]["start_time"]):
        chunk = [segments[x] for x in range(l, r+1)]
        merged.append(merge_chunk(chunk))

    # 5) Recompute continuity
    for idx in range(1, len(merged)):
        merged[idx]["same_topic_prev"] = topic_sim(merged[idx-1], merged[idx]) >= 0.5

    return merged


# ---------- DB I/O and CLI ----------

def load_bursts(vod_id: str) -> List[Dict]:
    db_path = Path(f"data/vector_stores/{vod_id}/metadata.db")
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, chapter_id, start_time, end_time,
               summary, topic, topic_key, energy, confidence,
               burst_score, chat_rate_z, reaction_hits,
               same_topic_prev
        FROM documents
        ORDER BY start_time
        """
    )
    rows = cur.fetchall()
    conn.close()
    bursts: List[Dict] = []
    for row in rows:
        bursts.append({
            "id": row[0],
            "chapter_id": row[1],
            "start_time": row[2],
            "end_time": row[3],
            "summary": row[4] or "",
            "topic": row[5] or "",
            "topic_key": row[6] or "",
            "energy": row[7] or "medium",
            "confidence": float(row[8] or 0.0),
            "burst_score": float(row[9] or 0.0),
            "chat_rate_z": float(row[10] or 0.0),
            "reaction_hits": json.loads(row[11]) if isinstance(row[11], str) and row[11] else (row[11] or {}),
            "same_topic_prev": bool(row[12]) if row[12] is not None else False,
        })
    return bursts


def update_sections(vod_id: str, chapter_id: str, sections: List[Dict]):
    db_path = Path(f"data/vector_stores/{vod_id}/metadata.db")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    # assign section_id to bursts and set section_title to topic_key
    for idx, sec in enumerate(sections):
        section_id = f"{vod_id}:{chapter_id}:sec:{idx+1:03d}"
        title = sec.get("topic_key") or sec.get("topic") or "section"
        for bid in sec.get("burst_ids", []):
            cur.execute(
                "UPDATE documents SET section_id=?, section_title=?, section_role=? WHERE id=?",
                (section_id, title, "story", bid)
            )
    conn.commit()
    conn.close()


def export_sections(vod_id: str, sections_by_chapter: Dict[str, List[Dict]]):
    base = Path(f"data/vector_stores/{vod_id}")
    base.mkdir(parents=True, exist_ok=True)
    # combined
    combined = []
    for chap, secs in sections_by_chapter.items():
        for i, s in enumerate(secs):
            s_out = dict(s)
            s_out["chapter_id"] = chap
            s_out["section_id"] = f"{vod_id}:{chap}:sec:{i+1:03d}"
            combined.append(s_out)
    (base / "sections.json").write_text(json.dumps({
        "vod_id": vod_id,
        "total_sections": sum(len(v) for v in sections_by_chapter.values()),
        "sections": combined,
    }, indent=2), encoding="utf-8")
    # per chapter
    for chap, secs in sections_by_chapter.items():
        (base / f"sections_{chap}.json").write_text(json.dumps({
            "vod_id": vod_id,
            "chapter_id": chap,
            "total_sections": len(secs),
            "sections": secs,
        }, indent=2), encoding="utf-8")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Smooth bursts into sections per chapter")
    parser.add_argument("vod_id", help="VOD ID")
    args = parser.parse_args()

    vod_id = args.vod_id
    bursts = load_bursts(vod_id)

    # Group per chapter
    by_chap: Dict[str, List[Dict]] = {}
    for b in bursts:
        by_chap.setdefault(b.get("chapter_id") or "unknown", []).append(b)
    # ensure sorted
    for chap in by_chap:
        by_chap[chap] = sorted(by_chap[chap], key=lambda x: x["start_time"])

    sections_by_chapter: Dict[str, List[Dict]] = {}
    for chap, segs in by_chap.items():
        sections = smooth_segments_one_chapter(segs)
        sections_by_chapter[chap] = sections
        update_sections(vod_id, chap, sections)

    export_sections(vod_id, sections_by_chapter)
    print(f"✅ Smoothing complete. Chapters: {len(sections_by_chapter)}")


if __name__ == "__main__":
    main()
