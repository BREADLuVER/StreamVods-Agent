"""
Director's Cut pipeline utilities.

Modules:
- manifest: read and normalize enhanced director's cut manifest
- downloader: concurrent Twitch segment downloader (1080p default)
- ffmpeg_graph: FFmpeg xfade/audiograph builders with NVENC encode
- render: high-level render orchestration with micro-batching
- title: helpers for generating a Director's Cut title
"""

__all__ = [
    "manifest",
    "downloader",
    "ffmpeg_graph",
    "render",
    "title",
]


