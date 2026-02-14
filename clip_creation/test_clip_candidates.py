#!/usr/bin/env python3
"""
Quick non-LLM clip candidate generator for isolation testing.

This CLI reuses the reaction/grouping/padding logic to produce candidate
windows without calling the LLM. It prints timestamps and a brief preview,
or writes a lightweight JSON manifest for inspection.

Usage:
  python -m clip_creation.test_clip_candidates <vod_id> \
      [--top-k 8] [--write] [--no-semantics] [--min-score 0.0]

Output fields (printed or JSON):
  - start, end, duration, start_hms, end_hms
  - anchor_time, anchor_time_hms
  - score (chat-based), preview (transcript snippet)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))  # project root

from rag.retrieval import load_retriever  # type: ignore
from vector_store.generate_clips_manifest import (
    WindowDoc,
    load_docs,
    pick_top_reaction_groups,
    build_reaction_arcs,
    extend_group_by_thread,
    extend_group_by_semantics,
    apply_dynamic_padding,
    bounds_of_indices,
    _snap_to_transcript_boundaries,
    _left_pad_to_sentence_start,
)


# ----------------------------- Helpers -----------------------------

def _format_hms(sec: float) -> str:
    try:
        s = int(round(float(sec)))
    except Exception:
        s = 0
    h = s // 3600
    m = (s % 3600) // 60
    s2 = s % 60
    return f"{h:02d}:{m:02d}:{s2:02d}"


def _reaction_total(doc: WindowDoc) -> int:
    if not doc.reaction_hits:
        return 0
    try:
        return int(sum(int(v) for v in doc.reaction_hits.values()))
    except Exception:
        return len(doc.reaction_hits)


GOODBYE_TERMS = (
    "goodbye",
    "good bye",
    "bye",
    "gn",
    "good night",
    "see you",
    "see ya",
    "signing off",
    "ending stream",
    "end of stream",
)


def _looks_like_goodbye(docs: List[WindowDoc]) -> bool:
    if not docs:
        return False
    # Heuristic: if >= 3 chat lines match goodbye-like terms, or any role suggests ending
    goodbye_hits = 0
    for d in docs:
        role = (d.role or "").strip().lower()
        if role in {"goodbye", "ending", "outro"}:
            return True
        text = (d.chat_text or "") + "\n" + (d.text or "")
        low = text.lower()
        for kw in GOODBYE_TERMS:
            if kw in low:
                goodbye_hits += 1
                if goodbye_hits >= 3:
                    return True
    return False


def _clip_quality_score(docs: List[WindowDoc], start: float, end: float) -> Tuple[float, float, float]:
    inner = [d for d in docs if d.start < end and d.end > start]
    if not inner:
        return (0.0, 0.0, 0.0)
    mean_chat_z = sum(max(0.0, d.chat_rate_z) for d in inner) / max(1, len(inner))
    max_chat_z = max(max(0.0, d.chat_rate_z) for d in inner)
    total_reacts = sum(_reaction_total(d) for d in inner)
    # Chat-centric score used elsewhere in the codebase
    score = 0.75 * mean_chat_z + 0.25 * max_chat_z
    return (score, mean_chat_z, float(total_reacts))


def _low_energy_reject(docs: List[WindowDoc], start: float, end: float) -> bool:
    inner = [d for d in docs if d.start < end and d.end > start]
    if not inner:
        return True
    high_energy_frac = sum(1 for d in inner if (d.energy or "").lower() == "high") / float(len(inner))
    score, mean_chat_z, total_reacts = _clip_quality_score(docs, start, end)
    # Reject very low energy/signals groups
    if high_energy_frac < 0.2 and mean_chat_z < 0.35 and total_reacts < 5:
        return True
    return False


def _anchor_center_for_group(docs: List[WindowDoc], group: List[int]) -> float:
    if not group:
        return 0.0
    g_docs = [docs[i] for i in group]
    anchor_doc = max(g_docs, key=_reaction_total)
    return 0.5 * (float(anchor_doc.start) + float(anchor_doc.end))


def _build_preview_text(docs: List[WindowDoc], start: float, end: float, limit: int = 140) -> str:
    inner = [d for d in docs if d.start < end and d.end > start]
    joined = " ".join((d.text or "").replace("\n", " ") for d in inner)
    return (joined[:limit]).strip()


# ----------------------------- Core -----------------------------

def generate_non_llm_candidates(
    vod_id: str,
    top_k: int,
    use_semantics: bool,
    min_score: float,
) -> List[Dict]:
    docs = load_docs(vod_id)
    retriever = load_retriever(vod_id)

    # Dynamic seeding similar to generate_clips
    total_docs = max(1, len(docs))
    chat_docs = sum(1 for d in docs if (d.mode or "").lower() == "chat")
    game_docs = sum(1 for d in docs if (d.mode or "").lower() == "game")
    share_chat = chat_docs / total_docs
    share_game = game_docs / total_docs

    # Expected lengths (seconds)
    expected_clip_chat = 55.0
    expected_clip_game = 75.0
    expected_clip_len = (
        (share_chat * expected_clip_chat) + (share_game * expected_clip_game)
        if (share_chat + share_game) > 0 else 60.0
    )

    # Seed budget
    dynamic_target = max(3, int(round(8 if top_k <= 0 else top_k * 2)))
    seeds_total = min(max(4, int(round(dynamic_target * 1.25))), 24)
    k_chat = max(1, int(round(seeds_total * (share_chat if (share_chat + share_game) > 0 else 0.5))))
    k_game = max(1, max(1, seeds_total - k_chat))

    # Thresholds per mode from reaction counts
    r_chat = [max(0, len(d.reaction_hits or {})) for d in docs if (d.mode or "").lower() == "chat"]
    r_game = [max(0, len(d.reaction_hits or {})) for d in docs if (d.mode or "").lower() == "game"]
    def _quantile(values: List[float], q: float) -> float:
        arr = sorted([v for v in values if isinstance(v, (int, float))])
        if not arr:
            return 0.0
        if q <= 0: return arr[0]
        if q >= 1: return arr[-1]
        idx = int(q * (len(arr) - 1))
        return arr[idx]
    thr_chat = max(2, int(round(_quantile(r_chat, 0.85)))) if r_chat else 2
    thr_game = max(3, int(round(_quantile(r_game, 0.85)))) if r_game else 3

    spacing_chat = max(45.0, min(90.0, round(0.7 * expected_clip_chat)))
    spacing_game = max(45.0, min(90.0, round(0.7 * expected_clip_game)))
    dedup_spacing = max(45.0, min(90.0, int(round(0.7 * expected_clip_len))))

    seed_chat = pick_top_reaction_groups(docs, k=k_chat, min_reactions=thr_chat, min_spacing=spacing_chat, mode_filter="chat")
    seed_game = pick_top_reaction_groups(docs, k=k_game, min_reactions=thr_game, min_spacing=spacing_game, mode_filter="game")

    groups = build_reaction_arcs(docs, seed_chat + seed_game)

    # Extend groups using thread continuity and optional semantics
    extended_groups: List[List[int]] = []
    for g in groups:
        g2 = extend_group_by_thread(docs, g)
        if use_semantics and getattr(retriever, "have_index", False):
            g3 = extend_group_by_semantics(docs, g2, retriever, time_window=60.0, sim_thr=0.40)
        else:
            g3 = g2
        extended_groups.append(g3)

    # Build windows and score
    candidates: List[Dict] = []
    for g in extended_groups:
        if not g:
            continue
        g_sorted = sorted(g)
        start_guess, end_guess = bounds_of_indices(docs, g_sorted)
        pad_lo = max(0.0, start_guess - 120.0)
        pad_hi = end_guess + 120.0
        start_guess, end_guess = pad_lo, pad_hi
        chapter_id = docs[g_sorted[0]].chapter_id
        start, end = apply_dynamic_padding(docs, chapter_id, start_guess, end_guess)

        # Ensure anchor is inside and not too late in the window
        anchor_center = _anchor_center_for_group(docs, g_sorted)
        win_len = min(180.0, max(30.0, end - start))
        if not (start <= anchor_center <= end):
            desired_start = anchor_center - (2.0 / 3.0) * win_len
            start = max(pad_lo, min(desired_start, pad_hi - win_len))
            end = start + win_len

        # Tighten window dynamically (target 35–120s) using signals
        ctx_docs = [d for d in docs if d.start < end and d.end > start]
        inner = ctx_docs
        if inner:
            # Mode majority inside window
            chat_count = sum(1 for d in inner if (d.mode or "").lower() == "chat")
            game_count = sum(1 for d in inner if (d.mode or "").lower() == "game")
            mode_major = "chat" if chat_count >= game_count else "game"
            high_energy_frac = sum(1 for d in inner if (d.energy or "").lower() == "high") / float(len(inner))
            mean_chat_z = sum(max(0.0, d.chat_rate_z) for d in inner) / max(1, len(inner))
            max_chat_z = max(max(0.0, d.chat_rate_z) for d in inner)
            total_reacts = sum(_reaction_total(d) for d in inner)

            # Base targets by mode
            desired_len = 55.0 if mode_major == "chat" else 75.0
            # Boost for strong signals (shorter is usually punchier)
            if mean_chat_z >= 1.0 and max_chat_z >= 2.0:
                desired_len -= 15.0
            if high_energy_frac >= 0.5:
                desired_len -= 10.0
            if total_reacts >= 20:
                desired_len -= 10.0
            # Clamp within 35..120s
            if desired_len < 35.0:
                desired_len = 35.0
            if desired_len > 120.0:
                desired_len = 120.0

            # Pre/post allocation by mode
            pre_ratio = 0.35 if mode_major == "chat" else 0.25
            pre_len = pre_ratio * desired_len
            # Center around anchor
            vstart = anchor_center - pre_len
            vstart = max(start, min(vstart, end - desired_len))
            vend = vstart + desired_len

            # Snap to transcript boundaries and allow slight left pad
            vstart, vend = _snap_to_transcript_boundaries(vstart, vend, ctx_docs, start, end, anchor_center)
            vstart, vend = _left_pad_to_sentence_start(vstart, vend, ctx_docs, start, max_left_pad=20.0)

            # Final containment and duration
            if vend - vstart >= 30.0:
                start, end = vstart, min(end, vstart + max(30.0, min(179.0, vend - vstart)))

        # Quality gates
        if _looks_like_goodbye([d for d in docs if d.start < end and d.end > start]):
            continue
        if _low_energy_reject(docs, start, end):
            continue

        score, mean_chat_z, total_reacts = _clip_quality_score(docs, start, end)
        if score < float(min_score):
            continue

        # Long windup guard: anchor should be within first 75% of the clip
        dur = max(0.0, end - start)
        if dur > 0.0:
            pos_frac = (anchor_center - start) / dur
            if pos_frac > 0.75:
                # Skip overly late-payoff windows in this non-LLM path
                continue

        preview = _build_preview_text(docs, start, end, limit=160)
        candidates.append({
            "vod_id": vod_id,
            "start": round(float(start), 3),
            "end": round(float(end), 3),
            "duration": round(float(end - start), 3),
            "start_hms": _format_hms(start),
            "end_hms": _format_hms(end),
            "anchor_time": round(float(max(start, min(anchor_center, end))), 3),
            "anchor_time_hms": _format_hms(max(start, min(anchor_center, end))),
            "score": round(float(score), 4),
            "mean_chat_z": round(float(mean_chat_z), 4),
            "total_reactions": int(total_reacts),
            "preview": preview,
        })

    # Simple dedup by center spacing and IoU
    def _iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        a0, a1 = a
        b0, b1 = b
        inter = max(0.0, min(a1, b1) - max(a0, b0))
        if inter <= 0: return 0.0
        union = (a1 - a0) + (b1 - b0) - inter
        return inter / union if union > 0 else 0.0

    candidates.sort(key=lambda c: c.get("score", 0.0), reverse=True)
    selected: List[Dict] = []
    for c in candidates:
        keep = True
        for s in selected:
            if _iou((c["start"], c["end"]), (s["start"], s["end"])) >= 0.5:
                keep = False
                break
            center_gap = abs(((c["start"] + c["end"]) * 0.5) - ((s["start"] + s["end"]) * 0.5))
            if center_gap < dedup_spacing:
                keep = False
                break
        if keep:
            selected.append(c)
        if top_k and len(selected) >= int(top_k):
            break

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate non-LLM clip candidates for testing")
    parser.add_argument("vod_id", help="VOD ID")
    parser.add_argument("--top-k", type=int, default=8, help="Max number of clips to output")
    parser.add_argument("--write", action="store_true", help="Write JSON manifest instead of only printing")
    parser.add_argument("--no-semantics", action="store_true", help="Disable semantic extension (vector sim)")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum chat-based score to include")
    args = parser.parse_args()

    clips = generate_non_llm_candidates(
        args.vod_id,
        top_k=int(args.top_k),
        use_semantics=(not bool(args.no_semantics)),
        min_score=float(args.min_score),
    )

    if not clips:
        print("No candidates produced.")
        return

    # Print concise lines for quick inspection
    for c in clips:
        print(
            f"{c['start_hms']} -> {c['end_hms']}  (dur={int(c['duration'])}s, score={c['score']}) | "
            f"anchor={c['anchor_time_hms']} | {c['preview']}"
        )

    if args.write:
        out_dir = Path(f"data/vector_stores/{args.vod_id}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "clips_manifest_test.json"
        obj = {
            "vod_id": args.vod_id,
            "total_selected": len(clips),
            "clips": clips,
            "note": "Non-LLM candidates for testing",
        }
        out_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        print(f"Wrote test manifest: {out_path}")


if __name__ == "__main__":
    main()


