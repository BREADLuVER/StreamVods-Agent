#!/usr/bin/env python3
"""Content curation – stitch documents into coherent video ranges.

This utility is shared by Director's-Cut, Short-Story and Clip builders.

Algorithm
---------
1. Expand each input document (burst or narrative) to concrete time ranges
   using the burst lookup table.
2. Sort ranges by start time.
3. Merge adjacent ranges if
   • Gap ≤ `max_gap` seconds OR
   • Same `topic_key` and gap ≤ `topic_gap` (larger)
4. Apply optional left/right padding.
5. De-duplicate overlaps.
6. Trim to `max_duration` budget (keep earliest ranges until budget filled).

CLI example
-----------
python -m rag.curate_manifest <vod_id> candidates.json --out curated.json \
       --max-gap 2 --topic-gap 10 --pad 0.5 1.0 --max-duration 1200

Where `candidates.json` is a list of document IDs or objects returned from
`extended_retriever.search`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Any

from rag.retrieval import load_bursts

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _load_docs(input_path: Path) -> List[Dict]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data  # assume list[dict]
    if isinstance(data, dict) and "documents" in data:
        return data["documents"]
    raise ValueError("Unsupported input JSON format, expecting list of documents")


def expand_docs_to_ranges(docs: Sequence[Dict], bursts_by_id: Dict[str, Tuple[float, float]]) -> List[Tuple[float, float, str]]:
    """Return list of (start,end,topic_key) tuples."""
    out: List[Tuple[float, float, str]] = []
    for d in docs:
        doc_id = d.get("id", "")
        topic_key = (d.get("topic_key") or d.get("category") or "").lower()
        if ":win:" in doc_id:  # burst doc
            s = float(d.get("start_time", d.get("start", 0.0)))
            e = float(d.get("end_time", d.get("end", 0.0)))
            out.append((s, e, topic_key))
            continue
        # narrative moment
        rh = d.get("reaction_hits") or {}
        bids = rh.get("burst_ids") if isinstance(rh, dict) else None
        if bids:
            for bid in bids:
                if bid in bursts_by_id:
                    s, e = bursts_by_id[bid]
                    out.append((s, e, topic_key))
        else:
            s = float(d.get("start", 0.0))
            e = float(d.get("end", s + 10.0))
            out.append((s, e, topic_key))
    return out


def merge_ranges(ranges: List[Tuple[float, float, str]], max_gap: float, topic_gap: float) -> List[Tuple[float, float]]:
    if not ranges:
        return []
    ranges = sorted(ranges, key=lambda x: x[0])
    merged: List[Tuple[float, float, str]] = [ranges[0]]
    for s, e, tk in ranges[1:]:
        ms, me, mk = merged[-1]
        gap = s - me
        if gap <= max_gap or (mk and tk and mk == tk and gap <= topic_gap):
            merged[-1] = (ms, max(me, e), mk or tk)
        else:
            merged.append((s, e, tk))
    return [(s, e) for s, e, _ in merged]


def apply_padding(ranges: List[Tuple[float, float]], pad_left: float, pad_right: float, vod_length: float) -> List[Tuple[float, float]]:
    out = []
    for s, e in ranges:
        ns = max(0.0, s - pad_left)
        ne = min(vod_length, e + pad_right)
        out.append((ns, ne))
    return out


def trim_to_budget(ranges: List[Tuple[float, float]], max_duration: float | None) -> List[Tuple[float, float]]:
    if max_duration is None or max_duration <= 0:
        return ranges
    acc = 0.0
    trimmed: List[Tuple[float, float]] = []
    for s, e in ranges:
        dur = e - s
        if acc + dur <= max_duration:
            trimmed.append((s, e))
            acc += dur
        else:
            break
    return trimmed


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def curate(vod_id: str, docs: Sequence[Dict], *, max_gap: float = 2.0, topic_gap: float = 8.0,
           pad_left: float = 0.3, pad_right: float = 0.7, max_duration: float | None = None) -> Dict:
    bursts = load_bursts(vod_id)
    bursts_by_id = {b["id"]: (b["start_time"], b["end_time"]) for b in bursts}
    vod_length = bursts[-1]["end_time"] if bursts else 0.0

    raw_ranges = expand_docs_to_ranges(docs, bursts_by_id)
    merged = merge_ranges(raw_ranges, max_gap, topic_gap)
    padded = apply_padding(merged, pad_left, pad_right, vod_length)
    final = trim_to_budget(padded, max_duration)

    manifest = {
        "vod_id": vod_id,
        "total_ranges": len(final),
        "total_duration": sum(e - s for s, e in final),
        "ranges": [{"start": s, "end": e, "duration": e - s} for s, e in final],
    }
    return manifest


# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Curate list of candidate docs into coherent ranges")
    parser.add_argument("vod_id")
    parser.add_argument("input", help="Path to JSON list of documents (from search)")
    parser.add_argument("--max-gap", type=float, default=2.0)
    parser.add_argument("--topic-gap", type=float, default=8.0)
    parser.add_argument("--pad", nargs=2, type=float, default=[0.3, 0.7], metavar=("LEFT","RIGHT"))
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    docs = _load_docs(Path(args.input))
    manifest = curate(
        args.vod_id,
        docs,
        max_gap=args.max_gap,
        topic_gap=args.topic_gap,
        pad_left=args.pad[0],
        pad_right=args.pad[1],
        max_duration=args.max_duration,
    )

    out_path = Path(args.out)
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"✅ Curated manifest written to {out_path}")


if __name__ == "__main__":
    main()
