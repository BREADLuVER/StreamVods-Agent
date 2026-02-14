"""
Data loading functions for clip generation.
"""

import json
import sqlite3
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .types import WindowDoc


def ensure_float(x, default: float = 0.0) -> float:
    """Ensure value is float, return default if not."""
    try:
        return float(x)
    except Exception:
        return float(default)


def parse_reaction_hits(raw: Optional[str | Dict]) -> Dict[str, int]:
    """Parse reaction hits from database or JSON string."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        # Already dict-like
        cleaned = {str(k): int(v) for k, v in raw.items() if ensure_float(v, None) is not None}
    else:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                cleaned = {str(k): int(v) for k, v in obj.items() if isinstance(v, (int, float))}
            else:
                cleaned = {}
        except Exception:
            cleaned = {}

    # Remove farewell terms that were mistakenly logged as reactions
    FAREWELL_REGEX = re.compile(r"^(good\s*bye+|bye+)$", re.IGNORECASE)
    cleaned = {k: v for k, v in cleaned.items() if not FAREWELL_REGEX.match(k)}
    return cleaned


def load_docs(vod_id: str) -> List[WindowDoc]:
    """Load window documents from vector store database."""
    db_path = Path(f"data/vector_stores/{vod_id}/metadata.db")
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Discover optional columns
    cur.execute("PRAGMA table_info(documents)")
    cols = {row[1] for row in cur.fetchall()}
    has_same_topic = "same_topic_prev" in cols
    has_thread = "topic_thread" in cols
    has_peak_block = "peak_block_id" in cols

    sel = (
        "SELECT id,start_time,end_time,chapter_id,mode,excluded,chat_rate,chat_rate_z,"
        "burst_score,reaction_hits,energy,role,"
        f"{('same_topic_prev' if has_same_topic else '0')} as same_topic_prev,"
        f"{('topic_thread' if has_thread else 'NULL')} as topic_thread,"
        "text,chat_text,"
        f"{('peak_block_id' if has_peak_block else 'NULL')} as peak_block_id "
        "FROM documents ORDER BY start_time"
    )
    cur.execute(sel)
    rows = cur.fetchall()
    conn.close()

    out: List[WindowDoc] = []
    for row in rows:
        rid = str(row[0])
        start = ensure_float(row[1])
        end = ensure_float(row[2])
        chapter_id = str(row[3]) if row[3] is not None else "chapter_001"
        mode = (row[4] or "unknown").lower()
        excluded = bool(int(row[5] or 0))
        chat_rate = ensure_float(row[6])
        chat_rate_z = ensure_float(row[7])
        burst_score = ensure_float(row[8])
        reaction_hits = parse_reaction_hits(row[9])
        energy = (row[10] or "").lower()
        role = (row[11] or "").lower()
        same_topic_prev = bool(int(row[12] or 0))
        topic_thread = int(row[13]) if row[13] is not None else None
        text = row[14] or ""
        chat_text = row[15] or ""
        peak_block_id = str(row[16]) if row[16] is not None else None
        
        out.append(WindowDoc(
            id=rid, start=start, end=end, chapter_id=chapter_id, mode=mode, excluded=excluded,
            chat_rate=chat_rate, chat_rate_z=chat_rate_z, burst_score=burst_score,
            reaction_hits=reaction_hits, energy=energy, role=role,
            same_topic_prev=same_topic_prev, topic_thread=topic_thread,
            text=text, chat_text=chat_text, peak_block_id=peak_block_id
        ))
    return out


def load_retriever(vod_id: str):
    """Load retriever for semantic similarity."""
    from rag.retrieval import load_retriever
    return load_retriever(vod_id)


def load_sponsor_spans(vod_id: str) -> List[Tuple[float, float]]:
    """Load sponsor spans for exclusion."""
    try:
        from rag.enhanced_director_cut_selector import load_atomic_segments
        return load_atomic_segments(vod_id)
    except Exception:
        return []
