#!/usr/bin/env python3
"""
Adjust chat_rate_z for intro segments
-------------------------------------

Problem:
- Early stream/segment greetings (hi/hello spam, emotes, raids) inflate chat_rate_z
- These are often labeled role="intro" by burst_summarize

Solution:
- For each chapter, locate the initial contiguous block of windows with role == "intro"
- Replace their chat_rate_z with the average chat_rate_z of the next K non-intro windows in that chapter
- Preserve the original value in chat_rate_z_raw; mark rows as adjusted via chat_z_adjusted flag

Run:
    python vector_store/adjust_chat_z.py <vod_id> [--lookahead 5]

This script mutates data/vector_stores/<vod_id>/metadata.db in-place.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple


def ensure_columns(cur: sqlite3.Cursor) -> None:
    cur.execute("PRAGMA table_info(documents)")
    cols = {row[1] for row in cur.fetchall()}
    if "chat_rate_z_raw" not in cols:
        cur.execute("ALTER TABLE documents ADD COLUMN chat_rate_z_raw REAL")
    if "chat_z_adjusted" not in cols:
        cur.execute("ALTER TABLE documents ADD COLUMN chat_z_adjusted INTEGER DEFAULT 0")


def fetch_rows(cur: sqlite3.Cursor, vod_id: str) -> List[Tuple]:
    cur.execute(
        """
        SELECT id, chapter_id, start_time, role, chat_rate_z
        FROM documents
        WHERE vod_id = ?
        ORDER BY chapter_id, start_time
        """,
        (vod_id,),
    )
    return cur.fetchall()


def adjust_chapter_intro(rows: List[Tuple], lookahead: int) -> List[Tuple[str, float]]:
    """
    For a single chapter's rows (ordered by start_time),
    find prefix with role == "intro" and compute replacement z.

    Returns list of (id, new_chat_rate_z) to update.
    """
    if not rows:
        return []

    # Identify contiguous intro prefix
    prefix_end = -1
    for i, (_, _, _, role, _) in enumerate(rows):
        if (role or "").strip().lower() == "intro":
            prefix_end = i
        else:
            break

    if prefix_end < 0:
        return []

    # Collect next K non-intro z values
    z_pool: List[float] = []
    for _, _, _, role, z in rows[prefix_end + 1 : ]:
        if (role or "").strip().lower() != "intro":
            try:
                z_pool.append(float(z or 0.0))
            except Exception:
                z_pool.append(0.0)
            if len(z_pool) >= lookahead:
                break

    # Fallback: use any non-intro z in chapter
    if not z_pool:
        for _, _, _, role, z in rows:
            if (role or "").strip().lower() != "intro":
                try:
                    z_pool.append(float(z or 0.0))
                except Exception:
                    z_pool.append(0.0)

    # Final fallback
    replacement = sum(z_pool) / len(z_pool) if z_pool else 0.0

    updates: List[Tuple[str, float]] = []
    for i in range(prefix_end + 1):
        _id = rows[i][0]
        updates.append((_id, replacement))
    return updates


def main() -> None:
    parser = argparse.ArgumentParser(description="Adjust chat_rate_z for intro segments")
    parser.add_argument("vod_id", help="VOD ID to adjust")
    parser.add_argument("--lookahead", type=int, default=5, help="How many subsequent non-intro windows to average")
    args = parser.parse_args()

    db_path = Path(f"data/vector_stores/{args.vod_id}/metadata.db")
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    ensure_columns(cur)
    conn.commit()

    rows = fetch_rows(cur, args.vod_id)
    if not rows:
        print("No rows found; nothing to adjust.")
        conn.close()
        return

    # Group by chapter_id preserving order
    by_chapter: Dict[str, List[Tuple]] = {}
    for r in rows:
        chap = r[1] or "chapter_000"
        by_chapter.setdefault(chap, []).append(r)

    total_updates = 0
    for chap_id, chap_rows in by_chapter.items():
        updates = adjust_chapter_intro(chap_rows, args.lookahead)
        if not updates:
            continue

        # Backup original z, then set new z and mark flag
        for _id, new_z in updates:
            cur.execute(
                """
                UPDATE documents
                SET chat_rate_z_raw = COALESCE(chat_rate_z_raw, chat_rate_z),
                    chat_rate_z = ?,
                    chat_z_adjusted = 1
                WHERE id = ?
                """,
                (float(new_z), _id),
            )
        conn.commit()
        total_updates += len(updates)
        print(f"Chapter {chap_id}: adjusted {len(updates)} intro windows")

    print(f"✅ Done. Total intro windows adjusted: {total_updates}")
    conn.close()


if __name__ == "__main__":
    main()


