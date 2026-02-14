#!/usr/bin/env python3
"""Query interface that turns a natural-language prompt into video ranges.

Usage
-----
python -m rag.query_video <vod_id> "funny sponsor moments" --top 12 --max-gap 2.0

• Performs semantic search over combined vector store (bursts + narrative moments).
• Expands narrative moments to underlying burst windows for accurate cutting.
• Merges adjacent ranges (≤ max-gap seconds) to output a concise manifest JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from rag.extended_retriever import load_extended_retriever
from rag.retrieval import load_bursts


def merge_ranges(ranges: List[Tuple[float, float]], max_gap: float = 2.0) -> List[Tuple[float, float]]:
    if not ranges:
        return []
    ranges = sorted(ranges, key=lambda x: x[0])
    merged = [list(ranges[0])]
    for s, e in ranges[1:]:
        last_s, last_e = merged[-1]
        if s - last_e <= max_gap:
            merged[-1][1] = max(last_e, e)
        else:
            merged.append([s, e])
    return [(a, b) for a, b in merged]


def expand_doc_to_ranges(doc: Dict, bursts_by_id: Dict[str, Tuple[float, float]]) -> List[Tuple[float, float]]:
    doc_id: str = doc.get("id", "")
    # Direct burst doc
    if ":win:" in doc_id:  # burst/windows id pattern
        return [(float(doc.get("start_time", doc.get("start", 0.0))), float(doc.get("end_time", doc.get("end", 0.0))))]
    # Narrative moment – we stored burst_ids in reaction_hits OR use time window
    rh = doc.get("reaction_hits") or {}
    burst_ids = rh.get("burst_ids") if isinstance(rh, dict) else None
    out: List[Tuple[float, float]] = []
    if burst_ids:
        for bid in burst_ids:
            if bid in bursts_by_id:
                out.append(bursts_by_id[bid])
    else:
        out.append((float(doc.get("start", 0.0)), float(doc.get("end", 0.0))))
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Query VOD vector store and output manifest ranges")
    parser.add_argument("vod_id")
    parser.add_argument("prompt", help="Natural language query")
    parser.add_argument("--top", type=int, default=20, help="Top K documents to retrieve")
    parser.add_argument("--max-gap", type=float, default=2.0, help="Gap threshold for merging ranges (seconds)")
    parser.add_argument("--out", default=None, help="Optional path to write manifest JSON")
    args = parser.parse_args()

    retriever = load_extended_retriever(args.vod_id)
    results = retriever.search(args.prompt, k=args.top)

    bursts = load_bursts(args.vod_id)
    bursts_by_id = {b["id"]: (b["start_time"], b["end_time"]) for b in bursts}

    raw_ranges: List[Tuple[float, float]] = []
    for doc, score in results:
        raw_ranges.extend(expand_doc_to_ranges(doc, bursts_by_id))

    merged = merge_ranges(raw_ranges, max_gap=args.max_gap)

    manifest = {
        "vod_id": args.vod_id,
        "prompt": args.prompt,
        "total_ranges": len(merged),
        "ranges": [{"start": s, "end": e, "duration": e - s} for s, e in merged],
    }

    if args.out:
        path = Path(args.out)
        path.write_text(json.dumps(manifest, indent=2))
        print(f"✅ Manifest written to {path}")
    else:
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
