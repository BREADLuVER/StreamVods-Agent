#!/usr/bin/env python3
"""
VOD Quality Gate
----------------

Computes quick VOD-level quality indicators from metadata.db and writes
`data/vector_stores/{vod_id}/vod_quality.json` with pass/fail decisions.

Indicators:
  Q1: P95(chat_rate_z) ≥ 1.0
  Q2: minutes with chat_rate_z > 1.0 per hour ≥ 3
  Q3: number of spike clusters (chat_rate_z > 1.2 sustained 30s+) per hour ≥ 2

Decision:
  clip_gate_pass = (Q1) or (Q2 and Q3)
  director_cut_gate_pass = clip_gate_pass  # same rule for now

Run:
  python vector_store/vod_quality_gate.py 2522344537
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import List, Tuple


def _quantile(values: List[float], q: float) -> float:
    arr = sorted(v for v in values if isinstance(v, (int, float)))
    if not arr:
        return 0.0
    if q <= 0:
        return arr[0]
    if q >= 1:
        return arr[-1]
    idx = int(q * (len(arr) - 1))
    return arr[idx]


def _load_windows(db_path: Path) -> List[Tuple[float, float, float]]:
    """Return list of (start,end,chat_rate_z)."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT start_time, end_time, chat_rate_z FROM documents ORDER BY start_time")
    rows = [(float(r[0]), float(r[1]), float(r[2] or 0.0)) for r in cur.fetchall()]
    conn.close()
    return rows


def compute_indicators(windows: List[Tuple[float, float, float]]):
    if not windows:
        return {
            "p95_chat_z": 0.0,
            "minutes_hot_per_hour": 0.0,
            "clusters_per_hour": 0.0,
            "hours": 0.0,
        }

    # Basic stats
    total_duration = max(0.0, windows[-1][1] - windows[0][0])
    hours = max(1e-6, total_duration / 3600.0)

    chat_z_vals = [max(0.0, z) for _, _, z in windows]
    p95_chat_z = _quantile(chat_z_vals, 0.95)

    # Minutes with chat_z > 1.0 per hour
    hot_seconds = 0.0
    for s, e, z in windows:
        if z > 1.0:
            hot_seconds += max(0.0, e - s)
    minutes_hot_per_hour = (hot_seconds / 60.0) / hours

    # Spike clusters per hour (chat_z > 1.2 sustained >= 30s)
    clusters = 0
    streak = 0.0
    THRESH = 1.2
    MIN_STREAK = 30.0
    for s, e, z in windows:
        dur = max(0.0, e - s)
        if z > THRESH:
            streak += dur
        else:
            if streak >= MIN_STREAK:
                clusters += 1
            streak = 0.0
    if streak >= MIN_STREAK:
        clusters += 1
    clusters_per_hour = clusters / hours

    return {
        "p95_chat_z": p95_chat_z,
        "minutes_hot_per_hour": minutes_hot_per_hour,
        "clusters_per_hour": clusters_per_hour,
        "hours": hours,
    }


def decide_gate(ind: dict) -> dict:
    q1 = ind.get("p95_chat_z", 0.0) >= 1.0
    q2 = ind.get("minutes_hot_per_hour", 0.0) >= 3.0
    q3 = ind.get("clusters_per_hour", 0.0) >= 2.0
    clip_gate_pass = bool(q1 or (q2 and q3))
    return {
        "Q1_p95_ge_1": q1,
        "Q2_hot_min_per_hr_ge_3": q2,
        "Q3_clusters_per_hr_ge_2": q3,
        "clip_gate_pass": clip_gate_pass,
        "director_cut_gate_pass": clip_gate_pass,
    }


def main():
    p = argparse.ArgumentParser(description="Compute VOD quality indicators and write vod_quality.json")
    p.add_argument("vod_id", help="VOD ID")
    args = p.parse_args()

    root = Path(f"data/vector_stores/{args.vod_id}")
    db = root / "metadata.db"
    if not db.exists():
        raise SystemExit(f"DB not found: {db}")

    windows = _load_windows(db)
    ind = compute_indicators(windows)
    gate = decide_gate(ind)

    out = {
        "vod_id": args.vod_id,
        "indicators": ind,
        "gate": gate,
    }
    out_path = root / "vod_quality.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"vod_quality.json written: {out_path}")
    if not gate.get("clip_gate_pass", True):
        print("Low-energy VOD detected; downstream clip generation should be skipped.")


if __name__ == "__main__":
    main()


