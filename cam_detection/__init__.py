#!/usr/bin/env python3
"""
Webcam/chat detection utilities for gating VOD processing.

Exports:
- detect_webcam_in_vod: returns True if webcam likely present in VOD
- detect_chat_in_vod: returns True if on-stream chat likely present in VOD
"""

def detect_webcam_in_vod(vod_id: str) -> bool:
    """Lazy import to avoid circular import issues."""
    from .detector import detect_webcam_in_vod as _detect_webcam_in_vod
    return _detect_webcam_in_vod(vod_id)


def detect_chat_in_vod(vod_id: str) -> bool:
    """Lazy import to avoid circular import issues."""
    from .detector import detect_chat_in_vod as _detect_chat_in_vod
    return _detect_chat_in_vod(vod_id)

