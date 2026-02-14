"""
Scoring and quality gate functions for clip generation.
"""

from typing import List, Tuple

from .types import WindowDoc
from .config import ClipConfig


# Goodbye terms for detection
GOODBYE_TERMS = (
    "goodbye",
    "good bye", 
    "bye",
    "gn",
    "good night",
    "see you",
    "see ya",
    "signing off",
    "ending stream",
    "end of stream",
)


def reaction_total(doc: WindowDoc) -> int:
    """Total number of reaction hits in a window."""
    if not doc.reaction_hits:
        return 0
    try:
        return int(sum(int(v) for v in doc.reaction_hits.values()))
    except Exception:
        return len(doc.reaction_hits)


def compute_quality_score(docs: List[WindowDoc], start: float, end: float) -> Tuple[float, float, float]:
    """Compute quality score for a clip window."""
    inner = [d for d in docs if d.start < end and d.end > start]
    if not inner:
        return (0.0, 0.0, 0.0)
    
    mean_chat_z = sum(max(0.0, d.chat_rate_z) for d in inner) / max(1, len(inner))
    max_chat_z = max(max(0.0, d.chat_rate_z) for d in inner)
    total_reacts = sum(reaction_total(d) for d in inner)
    
    # Chat-centric score used elsewhere in the codebase
    score = 0.75 * mean_chat_z + 0.25 * max_chat_z
    return (score, mean_chat_z, float(total_reacts))


def looks_like_goodbye(docs: List[WindowDoc]) -> bool:
    """Check if docs contain goodbye/outro content."""
    if not docs:
        return False
    
    # Heuristic: if >= 3 chat lines match goodbye-like terms, or any role suggests ending
    goodbye_hits = 0
    for d in docs:
        role = (d.role or "").strip().lower()
        if role in {"goodbye", "ending", "outro"}:
            return True
        text = (d.chat_text or "") + "\n" + (d.text or "")
        low = text.lower()
        for kw in GOODBYE_TERMS:
            if kw in low:
                goodbye_hits += 1
                if goodbye_hits >= 3:
                    return True
    return False


def low_energy_reject(docs: List[WindowDoc], start: float, end: float, config: ClipConfig) -> bool:
    """Check if window should be rejected for low energy/signals."""
    inner = [d for d in docs if d.start < end and d.end > start]
    if not inner:
        return True
    
    high_energy_frac = sum(1 for d in inner if (d.energy or "").lower() == "high") / float(len(inner))
    score, mean_chat_z, total_reacts = compute_quality_score(docs, start, end)
    
    # Reject very low energy/signals groups
    if (high_energy_frac < config.low_energy_energy_frac and 
        mean_chat_z < config.low_energy_chat_z and 
        total_reacts < config.low_energy_reactions):
        return True
    return False


def long_windup_guard(anchor_center: float, start: float, end: float, config: ClipConfig) -> bool:
    """Check if anchor is too late in the clip (long windup)."""
    dur = max(0.0, end - start)
    if dur <= 0.0:
        return True
    
    pos_frac = (anchor_center - start) / dur
    return pos_frac > config.long_windup_threshold


def anchor_center_for_group(docs: List[WindowDoc], group: List[int]) -> float:
    """Compute anchor center for a group of documents."""
    if not group:
        return 0.0
    g_docs = [docs[i] for i in group]
    anchor_doc = max(g_docs, key=reaction_total)
    return 0.5 * (float(anchor_doc.start) + float(anchor_doc.end))


def build_preview_text(docs: List[WindowDoc], start: float, end: float, limit: int = 140) -> str:
    """Build preview text for a clip window."""
    inner = [d for d in docs if d.start < end and d.end > start]
    joined = " ".join((d.text or "").replace("\n", " ") for d in inner)
    return (joined[:limit]).strip()
