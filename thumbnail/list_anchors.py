#!/usr/bin/env python3
"""
List anchor indices for a VOD using existing vector-store utilities.

Outputs:
- JSON sidecar with anchor indices (peak-score and reaction-based) and groups
- Optional ASCII timeline for quick visual inspection

Usage:
  python -m thumbnail.list_anchors <vod_id> [--mode both|peak|reactions] [--ascii]
                                   [--json-out <path>]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Ensure project root on sys.path for local imports
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


try:
    # Reuse core functions from the clips manifest pipeline
    from vector_store.generate_clips_manifest import (  # type: ignore
        load_docs,
        compute_peak_scores,
        select_anchor_indices,
        group_anchors,
        _reaction_total,
        _format_hms,
    )
except Exception as e:
    raise RuntimeError(f"Failed to import vector_store utilities: {e}")


def _load_sponsor_spans(vod_id: str) -> List[Tuple[float, float]]:
    try:
        from rag.enhanced_director_cut_selector import load_atomic_segments  # type: ignore
        return load_atomic_segments(vod_id)
    except Exception:
        return []


def _ascii_timeline(
    vod_start: float,
    vod_end: float,
    peak_times: List[float],
    react_times: List[float],
    width: int = 120,
) -> str:
    width = max(60, min(200, int(width)))
    span = max(1e-6, float(vod_end) - float(vod_start))

    def _pos(t: float) -> int:
        x = int(round(((t - vod_start) / span) * (width - 1)))
        return max(0, min(width - 1, x))

    line = ["."] * width
    for t in peak_times:
        line[_pos(float(t))] = "P" if line[_pos(float(t))] == "." else "X"
    for t in react_times:
        ch = line[_pos(float(t))]
        line[_pos(float(t))] = "R" if ch == "." else ("X" if ch != "R" else "R")

    # Build scale (HH:MM at ends)
    left = _format_hms(vod_start)
    right = _format_hms(vod_end)
    bar = "".join(line)
    pad = max(0, width - len(left) - len(right) - 2)
    scale = f"{left} " + ("-" * pad) + f" {right}"
    return f"{bar}\n{scale}\nLegend: P=peak-score anchors, R=reaction anchors, X=overlap\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="List anchors for a VOD (peak/reactions)")
    parser.add_argument("vod_id", help="VOD ID")
    parser.add_argument("--mode", choices=["both", "peak", "reactions"], default="both")
    parser.add_argument("--ascii", action="store_true", help="Print ASCII timeline")
    parser.add_argument("--json-out", default=None, help="Path to write JSON index")
    args = parser.parse_args()

    vod_id = str(args.vod_id)
    docs = load_docs(vod_id)
    if not docs:
        print("No documents loaded")
        return

    vod_start = float(docs[0].start)
    vod_end = float(docs[-1].end)

    sponsor_spans = _load_sponsor_spans(vod_id)
    peak_scores = compute_peak_scores(docs, sponsor_spans)

    out: Dict[str, object] = {
        "vod_id": vod_id,
        "vod_start": round(vod_start, 3),
        "vod_end": round(vod_end, 3),
        "vod_start_hms": _format_hms(vod_start),
        "vod_end_hms": _format_hms(vod_end),
        "modes": {},
    }

    def _anchors_payload(indices: List[int], label: str) -> Dict[str, object]:
        groups = group_anchors(docs, indices)
        anchors = []
        for i in indices:
            d = docs[i]
            anchors.append({
                "index": int(i),
                "start": round(float(d.start), 3),
                "end": round(float(d.end), 3),
                "start_hms": _format_hms(d.start),
                "end_hms": _format_hms(d.end),
                "reaction_total": int(_reaction_total(d)),
                "peak_score": float(peak_scores[i]) if 0 <= i < len(peak_scores) else 0.0,
            })
        grouped = []
        for g in groups:
            s = docs[g[0]].start
            e = docs[g[-1]].end
            grouped.append({
                "count": len(g),
                "start": round(float(s), 3),
                "end": round(float(e), 3),
                "start_hms": _format_hms(s),
                "end_hms": _format_hms(e),
                "indices": [int(x) for x in g],
            })
        return {"label": label, "count": len(indices), "anchors": anchors, "groups": grouped}

    peak_indices: List[int] = []
    react_indices: List[int] = []

    if args.mode in ("both", "peak"):
        peak_indices = select_anchor_indices(docs, peak_scores, use_reaction_hits=False)
        out["modes"]["peak"] = _anchors_payload(peak_indices, "peak")

    if args.mode in ("both", "reactions"):
        react_indices = select_anchor_indices(docs, peak_scores, use_reaction_hits=True)
        out["modes"]["reactions"] = _anchors_payload(react_indices, "reactions")

    # Emit JSON (default path under the VOD dir)
    json_out = args.json_out or str(Path(f"data/vector_stores/{vod_id}/anchors_index.json"))
    out_path = Path(json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote anchors JSON: {out_path}")

    # Optional ASCII timeline
    if args.ascii:
        peak_times = [docs[i].start for i in peak_indices]
        react_times = [docs[i].start for i in react_indices]
        graph = _ascii_timeline(vod_start, vod_end, peak_times, react_times, width=120)
        print(graph)


if __name__ == "__main__":
    main()


