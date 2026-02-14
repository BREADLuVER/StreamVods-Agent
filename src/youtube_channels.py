#!/usr/bin/env python3
"""
YouTube channel registry and resolver.

Loads config/youtube_channels.json and resolves which channels to upload to
for a given VOD based on streamer. Defaults to a "default" channel.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_REGISTRY_PATH = Path("config/youtube_channels.json")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_registry() -> Dict[str, dict]:
    """Load the channels registry; return minimal default if missing."""
    if _REGISTRY_PATH.exists():
        data = _read_json(_REGISTRY_PATH)
        if isinstance(data, dict) and data.get("channels"):
            ch = data["channels"]
            return ch if isinstance(ch, dict) else {}
    # Minimal default pointing to root credentials file
    return {
        "default": {
            "credentials_file": "youtube_credentials.json",
            "channel_id": "",
            "include_default": True,
            "uploads": {"clips": True, "arcs": True},
        }
    }


def _streamer_from_context(vod_id: str) -> str:
    """Best-effort streamer detection from local AI context files."""
    try:
        from src.config import config  # lazy import
    except Exception:
        config = None  # type: ignore
    ai_dir = None
    if config is not None:
        try:
            ai_dir = config.get_ai_data_dir(vod_id)
        except Exception:
            ai_dir = None
    if ai_dir is None:
        ai_dir = Path("data/ai_data") / str(vod_id)
    # Prefer stream_context
    sc = ai_dir / f"{vod_id}_stream_context.json"
    if sc.exists():
        try:
            data = _read_json(sc)
            s = str(data.get("streamer") or "").strip()
            if s:
                return s
        except Exception:
            pass
    # Fallback to vod_info
    vi = ai_dir / f"{vod_id}_vod_info.json"
    if vi.exists():
        try:
            d = _read_json(vi)
            for k in [
                "streamer",
                "Streamer",
                "UserName",
                "user_name",
                "channel",
                "Channel",
                "display_name",
                "login",
                "user_login",
            ]:
                s = str(d.get(k) or "").strip()
                if s:
                    return s
        except Exception:
            pass
    return ""


def get_streamer_for_vod(vod_id: str) -> str:
    s = _streamer_from_context(vod_id)
    return s.strip()


def resolve_channels_for_vod(vod_id: str, *, content_type: str = "clips") -> List[str]:
    """Return list of channel keys to upload to, given the VOD and content type.

    content_type: "clips" or "arcs" controls optional per-channel filters.
    """
    registry = load_registry()
    keys: List[str] = []
    streamer = get_streamer_for_vod(vod_id).lower()

    include_default = True
    # Match channels by rules
    for key, entry in registry.items():
        if key == "default":
            continue
        match = entry.get("match") or {}
        streamers = [str(x).lower() for x in (match.get("streamers") or [])]
        if streamer and streamer.lower() in streamers:
            uploads = entry.get("uploads") or {}
            if uploads and not uploads.get(content_type, True):
                continue
            keys.append(key)
            if entry.get("include_default") is False:
                include_default = False

    # Add default unless suppressed or uploads disabled
    if include_default and "default" in registry:
        default_entry = registry.get("default") or {}
        uploads = default_entry.get("uploads") or {}
        # Only include default if uploads are explicitly enabled for this content type
        if uploads and uploads.get(content_type, False):
            keys.insert(0, "default")  # default first

    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def get_channel_credentials_file(channel_key: str) -> str:
    entry = load_registry().get(channel_key) or {}
    return str(entry.get("credentials_file") or "youtube_credentials.json")


def get_channel_id(channel_key: str) -> str:
    entry = load_registry().get(channel_key) or {}
    return str(entry.get("channel_id") or "")


