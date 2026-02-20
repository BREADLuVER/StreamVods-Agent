#!/usr/bin/env python3
"""
Concurrent segment downloader for Director's Cut ranges using TwitchDownloaderCLI.

Defaults to 1080p per user guidance (1080p60 not required).
"""

from __future__ import annotations

import concurrent.futures
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.config import config


def _resolve_twitch_cli() -> str:
    override = os.getenv("TWITCH_DOWNLOADER_PATH") or config.twitch_downloader_path
    if override and Path(override).exists():
        return str(override)
    exe_path = Path("executables/TwitchDownloaderCLI.exe")
    if os.name == "nt" and exe_path.exists():
        return str(exe_path)
    return "TwitchDownloaderCLI"


def _hhmmss(seconds: float) -> str:
    sec = int(round(seconds))
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


@dataclass
class SegmentPlan:
    index: int
    start: float
    end: float
    out_path: Path


def plan_segments_from_ranges(ranges: List[dict], vod_id: str, max_segment_seconds: int = 300) -> List[SegmentPlan]:
    chunks_dir = config.get_chunk_dir(vod_id) / "director_cut_segments"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    plans: List[SegmentPlan] = []
    seg_index = 0
    for r in ranges:
        cur = float(r["start"])
        end = float(r["end"])
        while cur < end:
            seg_end = min(end, cur + max_segment_seconds)
            # If the remainder at tail would be very small (<10s) merge it into current
            if end - seg_end < 10 and (seg_end - cur) < max_segment_seconds:
                seg_end = end
            seg_index += 1
            out = chunks_dir / f"seg_{seg_index:04d}_{int(cur)}-{int(seg_end)}.mp4"
            plans.append(SegmentPlan(seg_index, cur, seg_end, out))
            cur = seg_end
    return plans


def _download_one(vod_id: str, plan: SegmentPlan, quality_prefs: List[str]) -> bool:
    twitch_cli = _resolve_twitch_cli()
    env = os.environ.copy()
    tmp = Path(env.get("TMP", "./data/temp")) / "twitch_dl"
    tmp.mkdir(parents=True, exist_ok=True)
    env["TEMP"] = str(tmp)
    env["TMP"] = str(tmp)
    env["TMPDIR"] = str(tmp)

    start_ts = _hhmmss(plan.start)
    end_ts = _hhmmss(plan.end)

    # prefer 1080p, then fall back progressively
    ladder = []
    for q in [*quality_prefs, "1080p", "720p", "480p", "360p"]:
        if q not in ladder:
            ladder.append(q)

    # Pre-delete file to avoid CLI prompts
    try:
        if plan.out_path.exists():
            plan.out_path.unlink()
    except Exception:
        pass

    # Ensure ffmpeg is in PATH for TwitchDownloaderCLI
    if os.name == 'nt':
        exec_dir = str(Path(__file__).parent.parent / "executables")
        env["PATH"] = f"{exec_dir}{os.pathsep}{env.get('PATH', '')}"

    for q in ladder:
        base_cmd = [
            twitch_cli, "videodownload",
            "--id", str(vod_id),
            "-b", start_ts,
            "-e", end_ts,
            "-o", str(plan.out_path),
            "-q", q,
        ]
        for attempt in range(1, 3):
            cmd = list(base_cmd)
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
            except FileNotFoundError:
                return False
            if res.returncode == 0 and plan.out_path.exists() and plan.out_path.stat().st_size > 0:
                return True
            # Log failure details for debugging
            last_err = (res.stderr or res.stdout or "").strip()[:200]
            print(f"⚠️ Download attempt {attempt} failed: {last_err}")
    return False


def download_segments(plans: List[SegmentPlan], vod_id: str, max_workers: Optional[int] = None, quality: str = "1080p") -> List[Path]:
    if not plans:
        return []
    if max_workers is None:
        # Respect DOWNLOAD_MAX_CONCURRENCY to avoid saturating local network
        dl_conc = os.getenv("DOWNLOAD_MAX_CONCURRENCY")
        if dl_conc and dl_conc.isdigit():
            max_workers = max(1, int(dl_conc))
        else:
            max_workers = max(1, int(os.getenv("MAX_WORKERS", str(config.max_workers))))
    # Normalize quality preference (no 60fps requirement per user)
    # Allow override via env (comma-separated list), default to requested quality then ladder
    env_pref = (os.getenv("DOWNLOAD_QUALITY_PREF") or "").strip()
    if env_pref:
        quality_prefs = [q.strip() for q in env_pref.split(',') if q.strip()]
    else:
        quality_prefs = [quality] if quality else ["1080p"]

    out_paths: List[Path] = [p.out_path for p in plans]
    # Skip downloads for existing non-empty files (must be > 1KB to be valid)
    todo: List[SegmentPlan] = []
    for p in plans:
        if p.out_path.exists():
            if p.out_path.stat().st_size > 1024:
                continue
            # Delete 0-byte or tiny corrupt files so we retry them
            try:
                p.out_path.unlink()
            except Exception:
                pass
        todo.append(p)

    if not todo:
        return out_paths

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_download_one, vod_id, plan, quality_prefs): plan for plan in todo
        }
        for fut in concurrent.futures.as_completed(futures):
            plan = futures[fut]
            try:
                ok = fut.result()
                if not ok:
                    # Don't leave marker files; let next run retry
                    try:
                        if plan.out_path.exists() and plan.out_path.stat().st_size == 0:
                            plan.out_path.unlink()
                    except Exception:
                        pass
            except Exception:
                pass

    # Return list in original order
    return [p.out_path for p in plans if p.out_path.exists() and p.out_path.stat().st_size > 1024]


