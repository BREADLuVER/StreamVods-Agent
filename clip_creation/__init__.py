"""Clip creation intelligent layout package.

Provides a small OOP surface for analysing a clip segment and
producing an exact layout decision (with crops) that the encoder
can consume to build an FFmpeg filter graph.

Initial implementation is intentionally lightweight; detection can be
swapped without touching the caller.
"""

from .models import CropBox, LayoutDecision
from .analyzer import ClipAnalyzer
from .ffmpeg_layouts import build_filter_graph_from_decision

__all__ = [
	"CropBox",
	"LayoutDecision",
	"ClipAnalyzer",
	"build_filter_graph_from_decision",
]


