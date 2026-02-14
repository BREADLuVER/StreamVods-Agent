"""
Data classes for clip generation pipeline.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class WindowDoc:
    """Window document from vector store."""
    id: str
    start: float
    end: float
    chapter_id: str
    mode: str
    excluded: bool
    chat_rate: float
    chat_rate_z: float
    burst_score: float
    reaction_hits: Dict[str, int]
    energy: str
    role: str
    same_topic_prev: bool
    topic_thread: Optional[int]
    text: str
    chat_text: str
    peak_block_id: Optional[str]


@dataclass
class ClipCandidate:
    """Intermediate clip candidate before finalization."""
    vod_id: str
    start: float
    end: float
    duration: float
    start_hms: str
    end_hms: str
    anchor_time: float
    anchor_time_hms: str
    score: float
    mean_chat_z: float
    total_reactions: int
    preview: str
    anchor_burst_id: Optional[str] = None


@dataclass
class FinalClip:
    """Final clip with title and metadata."""
    vod_id: str
    start: float
    end: float
    duration: float
    start_hms: str
    end_hms: str
    anchor_time: float
    anchor_time_hms: str
    title: str
    score: float
    rationale: str
    anchor_burst_id: Optional[str] = None
    hook: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class SeedGroup:
    """Seed group for reaction-based selection."""
    indices: List[int]
    total_reactions: int
    mode: str


@dataclass
class ClipManifestMeta:
    """Metadata for clip manifest."""
    vod_id: str
    total_candidates: int
    total_selected: int
    min_score: float
    front_pad_s: float
    back_pad_s: float


def format_hms(sec: float) -> str:
    """Format seconds as HH:MM:SS."""
    try:
        s = int(round(float(sec)))
    except Exception:
        s = 0
    h = s // 3600
    m = (s % 3600) // 60
    s2 = s % 60
    return f"{h:02d}:{m:02d}:{s2:02d}"
