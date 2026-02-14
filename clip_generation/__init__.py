"""
Clip Generation Module

OOP-based clip generation pipeline that separates non-LLM selection from LLM title generation.
"""

from .pipeline import ClipPipeline
from .types import ClipCandidate, FinalClip, SeedGroup, ClipManifestMeta

__all__ = [
    "ClipPipeline",
    "ClipCandidate", 
    "FinalClip",
    "SeedGroup",
    "ClipManifestMeta",
]
