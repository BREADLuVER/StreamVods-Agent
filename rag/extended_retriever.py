#!/usr/bin/env python3
"""Extended Retriever that works over the combined vector index (bursts + narrative moments).

Provides simple `.sim(id_a, id_b)` for cosine similarity between two stored vectors and
`.search(query,k,filters)` for semantic search with optional metadata filter dict.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from vector_store.vector_index import VectorIndex, HAS_FAISS


class ExtendedRetriever:
    def __init__(self, root_dir: Path, embedding_model: str = "all-MiniLM-L6-v2"):
        self.root = root_dir
        self.vindex = VectorIndex(str(root_dir), embedding_model_name=embedding_model)
        # Load stored vectors fallback (numpy)
        self.vecs: Optional[np.ndarray] = None
        if self.vindex.vectors_path.exists():
            try:
                import pickle
                with open(self.vindex.vectors_path, "rb") as f:
                    self.vecs = pickle.load(f)
                self.vecs = np.asarray(self.vecs, dtype=np.float32)
            except Exception:
                self.vecs = None
        # Build id -> row mapping from metadata DB rowid
        self.id_to_idx: Dict[str, int] = {}
        try:
            conn = sqlite3.connect(str(self.vindex.metadata_db_path))
            cur = conn.cursor()
            rows = cur.execute("SELECT id, rowid FROM documents").fetchall()
            conn.close()
            for rid, rowid in rows:
                self.id_to_idx[rid] = int(rowid)-1  # rowid starts at 1
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _vec(self, doc_id: str) -> Optional[np.ndarray]:
        if self.vecs is None:
            return None
        idx = self.id_to_idx.get(doc_id)
        if idx is None or idx >= self.vecs.shape[0]:
            return None
        return self.vecs[idx]

    def sim(self, a_id: str, b_id: str) -> float:
        va = self._vec(a_id)
        vb = self._vec(b_id)
        if va is None or vb is None:
            return -1.0
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0.0:
            return -1.0
        return float(np.dot(va, vb) / denom)

    # ------------------------------------------------------------------
    def search(self, query: str, k: int = 10, filters: Optional[Dict] = None) -> List[Tuple[Dict, float]]:
        """Semantic search returning list[(document,row_score)]."""
        return self.vindex.search(query, k=k, filters=filters or {})


def load_extended_retriever(vod_id: str, embedding_model: str = "all-MiniLM-L6-v2") -> ExtendedRetriever:
    root = Path(f"data/vector_stores/{vod_id}")
    if not root.exists():
        raise FileNotFoundError(f"Vector store not found for VOD {vod_id}")
    return ExtendedRetriever(root, embedding_model=embedding_model)
