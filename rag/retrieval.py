#!/usr/bin/env python3
"""
RAG Retrieval helpers for director's cut selection.

This module reads burst metadata from the per-VOD SQLite DB and exposes
convenience functions to fetch anchors, local context, and boundary
candidates using per-chapter, data-driven quantiles (no magic constants).
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ----------------------------- Data I/O -----------------------------

def load_bursts(vod_id: str) -> List[Dict]:
    """Load all bursts for a VOD from metadata.db sorted by time."""
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
               role, role_confidence, same_topic_prev
        FROM documents
        ORDER BY start_time
        """
    )
    rows = cur.fetchall()
    conn.close()
    bursts: List[Dict] = []
    for row in rows:
        # Skip narrative-moment documents whose ids contain ':mom:'
        if ':mom:' in row[0]:
            continue
        bursts.append({
            "id": row[0],
            "chapter_id": row[1],
            "start_time": float(row[2]),
            "end_time": float(row[3]),
            "summary": row[4] or "",
            "topic": row[5] or "",
            "topic_key": (row[6] or "").lower(),
            "energy": row[7] or "medium",
            "confidence": float(row[8] or 0.0),
            "burst_score": float(row[9] or 0.0),
            "chat_rate_z": float(row[10] or 0.0),
            "reaction_hits": json.loads(row[11]) if isinstance(row[11], str) and row[11] else (row[11] or {}),
            "role": (row[12] or "").lower(),
            "role_confidence": float(row[13] or 0.0),
            "same_topic_prev": bool(row[14]) if row[14] is not None else False,
        })
    return bursts


def group_by_chapter(bursts: List[Dict]) -> Dict[str, List[Dict]]:
    by_chap: Dict[str, List[Dict]] = {}
    for b in bursts:
        by_chap.setdefault(b.get("chapter_id") or "unknown", []).append(b)
    for chap in by_chap:
        by_chap[chap] = sorted(by_chap[chap], key=lambda x: x["start_time"])
    return by_chap


# --------------------------- Quantile utils -------------------------

def quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    if q <= 0:
        return float(v[0])
    if q >= 1:
        return float(v[-1])
    idx = int(q * (len(v) - 1))
    return float(v[idx])


def chapter_stats(chapter_bursts: List[Dict]) -> Dict[str, float]:
    durations = [b["end_time"] - b["start_time"] for b in chapter_bursts]
    gaps = []
    for i in range(len(chapter_bursts) - 1):
        gaps.append(max(0.0, chapter_bursts[i+1]["start_time"] - chapter_bursts[i]["end_time"]))
    stats = {
        "dur_q25": quantile(durations, 0.25),
        "dur_q50": quantile(durations, 0.50),
        "dur_q75": quantile(durations, 0.75),
        "dur_q95": quantile(durations, 0.95),
        "gap_q75": quantile(gaps, 0.75) if gaps else 0.0,
        "gap_q90": quantile(gaps, 0.90) if gaps else 0.0,
        "gap_q95": quantile(gaps, 0.95) if gaps else 0.0,
    }
    return stats


# ----------------------- Anchor / salience --------------------------

def compute_salience(chapter_bursts: List[Dict]) -> List[float]:
    def rank_to_unit(vals: List[float]) -> List[float]:
        if not vals:
            return []
        uniq = sorted(set(vals))
        rank = {v: i for i, v in enumerate(uniq)}
        denom = max(1, len(uniq) - 1)
        return [rank[v] / denom for v in vals]

    norm_burst = rank_to_unit([b.get("burst_score", 0.0) for b in chapter_bursts])
    norm_chat  = rank_to_unit([max(0.0, b.get("chat_rate_z", 0.0)) for b in chapter_bursts])
    norm_react = rank_to_unit([sum((b.get("reaction_hits") or {}).values()) for b in chapter_bursts])
    norm_conf  = rank_to_unit([b.get("confidence", 0.0) for b in chapter_bursts])

    sal = []
    for i in range(len(chapter_bursts)):
        sal.append(0.45*norm_burst[i] + 0.25*norm_chat[i] + 0.20*norm_react[i] + 0.10*norm_conf[i])
    return sal


def detect_anchors(chapter_bursts: List[Dict]) -> List[bool]:
    sal = compute_salience(chapter_bursts)
    # Compute per-chapter average reaction count excluding zeros
    reaction_counts = [sum((b.get("reaction_hits") or {}).values()) for b in chapter_bursts]
    nonzero = [c for c in reaction_counts if c > 0]
    avg_react = (sum(nonzero) / len(nonzero)) if nonzero else 0.0

    anchors: List[bool] = []
    for c in reaction_counts:
        anchors.append(c >= avg_react and c > 0)

    # Fallback: if no anchors passed, promote local maxima above p85 salience
    if not any(anchors):
        cut = quantile(sal, 0.85)
        for i in range(len(chapter_bursts)):
            left = sal[i-1] if i-1 >= 0 else -1.0
            right = sal[i+1] if i+1 < len(chapter_bursts) else -1.0
            if sal[i] >= cut and sal[i] >= left and sal[i] >= right:
                anchors[i] = True
    return anchors


# ------------------------- Context windows --------------------------

def auto_window(stats: Dict[str, float]) -> Tuple[float, float]:
    """Return (min_span, max_span) around an anchor using dur_q50..dur_q75."""
    return (stats.get("dur_q50", 0.0), stats.get("dur_q75", 0.0))


def get_lead_in(chapter_bursts: List[Dict], idx_anchor: int, k: Optional[int], stats: Dict[str, float]) -> List[Dict]:
    left = []
    if k is None:
        min_span, max_span = auto_window(stats)
        acc = 0.0
        j = idx_anchor - 1
        while j >= 0 and acc < max_span:
            left.append(chapter_bursts[j])
            acc += chapter_bursts[j]["end_time"] - chapter_bursts[j]["start_time"]
            if acc >= min_span:
                break
            j -= 1
    else:
        j = idx_anchor - 1
        while j >= 0 and len(left) < k:
            left.append(chapter_bursts[j])
            j -= 1
    return list(reversed(left))


def get_cool_down(chapter_bursts: List[Dict], idx_anchor: int, k: Optional[int], stats: Dict[str, float]) -> List[Dict]:
    right = []
    if k is None:
        min_span, max_span = auto_window(stats)
        acc = 0.0
        j = idx_anchor + 1
        while j < len(chapter_bursts) and acc < max_span:
            right.append(chapter_bursts[j])
            acc += chapter_bursts[j]["end_time"] - chapter_bursts[j]["start_time"]
            if acc >= min_span:
                break
            j += 1
    else:
        j = idx_anchor + 1
        while j < len(chapter_bursts) and len(right) < k:
            right.append(chapter_bursts[j])
            j += 1
    return right


def get_between(chapter_bursts: List[Dict], idx_a_end: int, idx_b_start: int) -> List[Dict]:
    if idx_b_start <= idx_a_end + 1:
        return []
    return chapter_bursts[idx_a_end+1: idx_b_start]


def get_boundary(chapter_bursts: List[Dict], side: str = 'both', n: int = 3) -> Tuple[List[Dict], List[Dict]]:
    left = chapter_bursts[-n:]
    right: List[Dict] = []
    # Caller can pass next chapter for right side; this function returns only left by default.
    return (left, right)

# ---------------------- Semantic retrieval (vector cosine) ----------------------

import pickle  # noqa: E402  (local import on purpose)
import numpy as np  # noqa: E402

class Retriever:
    """Lightweight cosine-similarity retriever over pre-computed embeddings.

    The embedding matrix is loaded from ``vectors.pkl`` stored alongside
    ``metadata.db``.  Row ordering is aligned to the ``documents`` table order
    (``ORDER BY start_time``) so we can look up a burst's vector via an id →
    row-index map built from SQLite.
    """

    def __init__(self, have_index: bool, ids: List[str], id_to_idx: Dict[str, int], vecs: Optional[np.ndarray]):
        self.have_index = have_index
        self.ids = ids
        self.id_to_idx = id_to_idx
        self.vecs = vecs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _vec(self, win_id: str) -> Optional[np.ndarray]:
        if not self.have_index or self.vecs is None:
            return None
        idx = self.id_to_idx.get(win_id)
        if idx is None:
            return None
        return self.vecs[idx]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sim(self, a_id: str, b_id: str) -> float:
        """Return cosine similarity ∈ [-1,1].  -1 means vector missing/zero."""
        va = self._vec(a_id)
        vb = self._vec(b_id)
        if va is None or vb is None:
            return -1.0
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0.0:
            return -1.0
        return float(np.dot(va, vb) / denom)


def load_retriever(vod_id: str) -> "Retriever":
    """Load vectors + mapping for a specific VOD.  Falls back to dummy retriever."""
    root = Path(f"data/vector_stores/{vod_id}")
    meta_path = root / "metadata.db"
    vec_path = root / "vectors.pkl"
    ids_path = root / "vector_ids.pkl"

    if (not meta_path.exists()) or (not vec_path.exists()):
        return Retriever(False, [], {}, None)

    # Prefer persisted vector_ids order for alignment; fallback to DB order.
    ids: List[str]
    if ids_path.exists():
        try:
            with open(ids_path, "rb") as f:
                ids = [str(x) for x in pickle.load(f)]
        except Exception:
            ids = []
    else:
        ids = []

    if not ids:
        conn = sqlite3.connect(str(meta_path))
        cur = conn.cursor()
        rows = cur.execute("SELECT id FROM documents ORDER BY start_time").fetchall()
        conn.close()
        ids = [r[0] for r in rows]
    id_to_idx = {bid: i for i, bid in enumerate(ids)}

    with open(vec_path, "rb") as f:
        vecs = pickle.load(f)
    vecs = np.asarray(vecs, dtype=np.float32)

    # Safety: shape consistency
    if vecs.shape[0] != len(ids):
        print(f"[retrieval] Warning: vectors count {vecs.shape[0]} != ids {len(ids)} – disabling retriever.")
        return Retriever(False, ids, id_to_idx, None)

    return Retriever(True, ids, id_to_idx, vecs)