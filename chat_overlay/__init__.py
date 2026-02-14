"""Utilities for rendering and overlaying Twitch chat on videos.

Exports:
- render_chat_segment: Render chat color/mask videos for a VOD time range
- ensure_chat_json: Prefetch chat JSON for a time range
- overlay_chat_on_video: Overlay rendered chat onto a segment video
"""

from .renderer import render_chat_segment, ensure_chat_json  # noqa: F401
from .compose import overlay_chat_on_video  # noqa: F401


