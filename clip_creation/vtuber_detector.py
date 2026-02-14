#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.ai_client import call_llm_vision
from .frame_utils import sample_frames
from .prompts import VTUBER_DETECT_PROMPT


def detect_vtuber(input_path: Path) -> Optional[bool]:
    """Return True if VTuber detected, False if not, None on failure."""
    frame_w, frame_h, frames = sample_frames(input_path, num_samples=1)
    if not frames:
        return None
    image_bytes = frames[0]
    try:
        resp = call_llm_vision(
            VTUBER_DETECT_PROMPT,
            [("frame", image_bytes, "image/jpeg")],
            max_tokens=10,
            temperature=0.1,
            request_tag="vtuber_detect",
        )
        text = (resp or "").strip().upper()
        if "YES" in text:
            return True
        if "NO" in text:
            return False
        return None
    except Exception:
        return None


