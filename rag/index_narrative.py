#!/usr/bin/env python3
"""
Index narrative moments for semantic retrieval.

Reads `<vod_id>_narrative_analysis.json`, converts each atomic entry (content transition,
 gameplay event, sponsor segment, etc.) into a `VectorDocument`, embeds the text with the
 same sentence-transformers model used for bursts, and appends vectors + metadata to the
 existing vector store under `data/vector_stores/{vod_id}`.

After running this once, the extended `Retriever` will be able to surface both burst-level
and narrative-level results for RAG queries.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple
import re

# --- Local imports (lazy to avoid heavy deps when importing elsewhere) ---
from vector_store.document_builder import VectorDocument, format_embedding_text, extract_keywords
from vector_store.vector_index import VectorIndex

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts_to_seconds(ts: str | int | float) -> float:
    """Convert timestamp strings like 04:05, 1:23:45, or 18-33 into seconds."""
    if isinstance(ts, (int, float)):
        return float(ts)
    if not ts:
        return 0.0
    # Replace any non-digit separator with ':' then split
    cleaned = re.sub(r"[^0-9]", ":", str(ts).strip())
    parts = [p for p in cleaned.split(":") if p]
    try:
        parts = [int(p) for p in parts][-3:]
    except ValueError:
        return 0.0
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return float(h * 3600 + m * 60 + s)


def _load_burst_time_map(vod_id: str) -> List[Tuple[str, float, float]]:
    """Return list of (id,start,end) for bursts in this VOD (sorted)."""
    db_path = Path(f"data/vector_stores/{vod_id}/metadata.db")
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT id,start_time,end_time FROM documents ORDER BY start_time")
    rows = cur.fetchall()
    conn.close()
    return [(row[0], float(row[1]), float(row[2])) for row in rows]


def _burst_ids_in_range(bursts: List[Tuple[str, float, float]], start: float, end: float) -> List[str]:
    out = []
    for bid, bs, be in bursts:
        if be < start:
            continue
        if bs > end:
            break
        out.append(bid)
    return out


# ---------------------------------------------------------------------------
# Main conversion logic
# ---------------------------------------------------------------------------

def build_moment_documents(vod_id: str, default_len: float = 30.0) -> List[VectorDocument]:
    """Parse narrative_analysis and return VectorDocuments for each moment."""
    analysis_path = Path(f"data/ai_data/{vod_id}/{vod_id}_narrative_analysis.json")
    if not analysis_path.exists():
        raise FileNotFoundError(f"Narrative analysis not found: {analysis_path}")

    data = json.loads(analysis_path.read_text(encoding="utf-8"))
    docs: List[VectorDocument] = []

    bursts_cache = _load_burst_time_map(vod_id)

    # Iterate chunks
    for ch in data.get("chunks", []):
        chunk_idx = ch.get("chunk_index") or 0
        chunk_start_s = ch.get("start_time", 0)

        def add_moment(item: Dict, mtype: str, key_fields: List[str]):
            # Determine window
            if "start" in item and "end" in item:
                start_s = _ts_to_seconds(item["start"]) if isinstance(item["start"], str) else float(item["start"])
                end_s = _ts_to_seconds(item["end"]) if isinstance(item["end"], str) else float(item["end"])
            elif "timestamp" in item:
                start_s = _ts_to_seconds(item["timestamp"])
                end_s = start_s + default_len
            else:
                return  # skip malformed

            # Offset by chunk start if timestamps are local (<= chunk duration)
            if start_s < 7200 and chunk_start_s > 0:  # heuristic: local ts unlikely > 2h
                start_s += chunk_start_s
                end_s += chunk_start_s

            # Build text description from key_fields
            parts = [str(item.get(k, "")) for k in key_fields]
            text = ". ".join(p for p in parts if p)

            # Categories / tags
            keywords = extract_keywords(text, max_keywords=8)
            tags = [mtype] + keywords

            doc_id = f"{vod_id}:mom:{mtype}:{int(start_s)}"
            doc = VectorDocument(
                id=doc_id,
                vod_id=vod_id,
                start=start_s,
                end=end_s,
                len_s=end_s - start_s,
                chapter_id=str(ch.get("chunk_index")),
                category=mtype,
                mode="unknown",
                text=text,
                keywords=tags,
            )

            # Attach burst ids for later mapping (optional metadata)
            if bursts_cache:
                doc.reaction_hits = {"burst_ids": _burst_ids_in_range(bursts_cache, start_s, end_s)}

            docs.append(doc)

        # Map of narrative keys to (item_list, moment_type, text_fields)
        mapping = {
            "content_transitions": ("transition", ["description", "from", "to"]),
            # gameplay_events handled custom below for round_end detection
            "emotional_changes": ("emotion", ["emotion", "trigger"]),
            "high_points": ("high_point", ["description"]),
            "sponsor_segments": ("sponsor", ["content"]),
            "gameplay_segments": ("gameplay_segment", ["game_state", "events"]),
            "chat_interactions": ("chat", ["activity", "trigger"]),
            "technical_issues": ("tech_issue", ["issue", "resolution"]),
            "community_moments": ("community", ["event", "description"]),
            "reaction_segments": ("reaction", ["content", "description"]),
            "personal_moments": ("personal", ["topic", "description"]),
        }

        # Regular keys (except gameplay_events)
        for list_key, (mtype, fields) in mapping.items():
            if list_key == "gameplay_events":
                continue
            for item in ch.get(list_key, []):
                add_moment(item, mtype, fields)

        # Custom handling for gameplay_events to detect round ends
        for item in ch.get("gameplay_events", []):
            desc = (item.get("description", "") + " " + item.get("event", "")).lower()
            if "round" in desc and ("end" in desc or "over" in desc or "win" in desc or "loss" in desc):
                gtype = "round_end"
            else:
                gtype = "gameplay_event"
            add_moment(item, gtype, ["description", "event"])

    return docs


# ---------------------------------------------------------------------------
# Entry-point CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Index narrative moments for semantic retrieval")
    parser.add_argument("vod_id", help="VOD identifier")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Sentence-transformers model name")
    args = parser.parse_args()

    vod_id = args.vod_id
    docs = build_moment_documents(vod_id)
    print(f"Parsed {len(docs)} narrative moments → VectorDocuments")

    # Filter duplicates vs existing index
    index_path = str(Path(f"data/vector_stores/{vod_id}"))  # directory root
    vindex = VectorIndex(index_path, embedding_model_name=args.embedding_model)

    existing_ids = set(vindex.get_ids()) if hasattr(vindex, "get_ids") else set()
    new_docs = [d for d in docs if d.id not in existing_ids]
    if not new_docs:
        print("No new documents to add – index already up-to-date.")
        return

    embedding_texts = [format_embedding_text(d) for d in new_docs]
    vindex.add_documents(new_docs, embedding_texts)
    print(f"✅ Added {len(new_docs)} narrative moments to vector store: {index_path}")


if __name__ == "__main__":
    main()
