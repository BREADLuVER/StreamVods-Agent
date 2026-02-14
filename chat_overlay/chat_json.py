#!/usr/bin/env python3
"""
Utilities for downloading full chat JSON once and trimming to time windows.
Works with TwitchDownloaderCLI JSON format (comments with content_offset_seconds).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Optional


def _resolve_twitch_cli() -> str:
    override = os.getenv("TWITCH_DOWNLOADER_PATH")
    if override and Path(override).exists():
        return override
    exe = Path("executables/TwitchDownloaderCLI.exe")
    if os.name == "nt" and exe.exists():
        return str(exe)
    return "TwitchDownloaderCLI"


def _is_valid_json(p: Path) -> bool:
    try:
        if not p.exists() or p.stat().st_size < 10:
            return False
        _ = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        return True
    except Exception:
        return False


def ensure_full_chat_json(vod_id: str, out_dir: Path) -> Optional[Path]:
    """Locate or download full chat JSON for a VOD with emotes embedded.

    Preference order:
    1) Existing out_dir/<vod_id>_chat.json (what you already generate today)
    2) Existing out_dir/<vod_id>_chat_full.json (our fallback naming)
    3) Download to <vod_id>_chat_full.json
    """
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        legacy = out_dir / f"{vod_id}_chat.json"
        if _is_valid_json(legacy):
            return legacy
        out = out_dir / f"{vod_id}_chat_full.json"
        force = os.getenv("FORCE_CHAT_REDOWNLOAD", "0").lower() in ("1", "true", "yes")
        if _is_valid_json(out) and not force:
            return out
        cli = _resolve_twitch_cli()
        cmd = [
            cli, "chatdownload",
            "--id", vod_id,
            "--embed-images",
            "--bttv=true",
            "--ffz=true",
            "--stv=true",
            "-o", str(out),
        ]
        for _ in range(2):
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if res.returncode == 0 and _is_valid_json(out):
                return out
            try:
                if out.exists():
                    out.unlink()
            except Exception:
                pass
        return out if _is_valid_json(out) else None
    except Exception:
        return None


def write_chat_subset(full_chat_json: Path, out_json: Path, start_sec: float, end_sec: float, head_sec: float = 0.0) -> bool:
    """Create a subset chat JSON containing comments in [start_sec - head, end_sec]."""
    try:
        start = max(0.0, float(start_sec) - max(0.0, float(head_sec)))
        end = max(start, float(end_sec))
        data = json.loads(full_chat_json.read_text(encoding="utf-8", errors="ignore"))
        comments = data.get("comments") or data.get("messages") or []
        out_comments = []
        for c in comments:
            t = None
            # TwitchDownloader uses content_offset_seconds
            if isinstance(c, dict):
                t = c.get("content_offset_seconds")
                if t is None:
                    # Some schemas nest it
                    t = (c.get("message") or {}).get("content_offset_seconds")
            try:
                if t is None:
                    continue
                ts = float(t)
                if start <= ts <= end:
                    out_comments.append(c)
            except Exception:
                continue
        out_obj = dict(data)
        out_obj["comments"] = out_comments
        # Optionally update video.time if present
        try:
            if "video" in out_obj and isinstance(out_obj["video"], dict):
                out_obj["video"]["start"] = start
                out_obj["video"]["end"] = end
        except Exception:
            pass
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(out_obj, ensure_ascii=False), encoding="utf-8")
        return True
    except Exception:
        return False


