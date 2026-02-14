"""
Reaction-based seeding for clip generation.
"""

from typing import List, Optional

from .types import WindowDoc, SeedGroup
from .config import ClipConfig


def reaction_total(doc: WindowDoc) -> int:
    """Total number of reaction hits in a window."""
    if not doc.reaction_hits:
        return 0
    try:
        return int(sum(int(v) for v in doc.reaction_hits.values()))
    except Exception:
        return len(doc.reaction_hits)


def mode_bucket(mode: str) -> str:
    """Coarse bucket: gameplay vs chat/other."""
    m = (mode or "").lower()
    if m == "game":
        return "game"
    return "chat"


def quantile(values: List[float], q: float) -> float:
    """Compute quantile of values."""
    arr = sorted([v for v in values if isinstance(v, (int, float))])
    if not arr:
        return 0.0
    if q <= 0:
        return arr[0]
    if q >= 1:
        return arr[-1]
    idx = int(q * (len(arr) - 1))
    return arr[idx]


def pick_top_reaction_groups(
    docs: List[WindowDoc],
    k: int = 10,
    min_reactions: int = 5,
    min_spacing: float = 45.0,
    mode_filter: Optional[str] = None,
) -> List[List[int]]:
    """Return groups each containing a single index, representing the K most
    reacted-to windows sorted by reaction count, non-overlapping by
    *min_spacing* seconds between *starts*.
    """
    if k <= 0:
        return []

    ranked = []
    for i, d in enumerate(docs):
        if mode_filter and mode_bucket(d.mode) != mode_filter:
            continue
        ranked.append((i, reaction_total(d)))
    ranked.sort(key=lambda t: t[1], reverse=True)
    groups: List[List[int]] = []
    taken_starts: List[float] = []
    for idx, cnt in ranked:
        if cnt < min_reactions:
            break
        start_time = docs[idx].start
        if any(abs(start_time - s) < min_spacing for s in taken_starts):
            continue
        groups.append([idx])  # single-window group
        taken_starts.append(start_time)
        if k and len(groups) >= k:
            break
    return groups


def create_seed_groups(
    docs: List[WindowDoc],
    config: ClipConfig,
    vod_duration_s: float,
) -> List[SeedGroup]:
    """Create seed groups based on VOD characteristics and config."""
    
    # Calculate mode distribution
    total_docs = max(1, len(docs))
    chat_docs = sum(1 for d in docs if mode_bucket(d.mode) == "chat")
    game_docs = sum(1 for d in docs if mode_bucket(d.mode) == "game")
    share_chat = chat_docs / total_docs
    share_game = game_docs / total_docs

    # Expected clip length
    expected_clip_len = (
        (share_chat * config.expected_clip_chat) + (share_game * config.expected_clip_game)
        if (share_chat + share_game) > 0.0 else 60.0
    )

    # Dynamic target clips based on VOD length
    clip_budget_frac = 0.12
    dynamic_target_clips = max(3, int(round((vod_duration_s * clip_budget_frac) / max(30.0, expected_clip_len))))
    
    # Seed budget
    seeds_total = max(config.min_seeds, int(round(dynamic_target_clips * config.seed_multiplier)))
    seeds_total = min(seeds_total, config.max_seeds)
    
    # Mode allocation
    if (share_chat + share_game) <= 0.0:
        k_chat = max(1, seeds_total // 2)
        k_game = max(1, seeds_total - k_chat)
    else:
        k_chat = max(1, int(round(seeds_total * share_chat)))
        k_game = max(1, max(1, seeds_total - k_chat))

    # Per-mode reaction thresholds
    r_chat = [max(0, reaction_total(d)) for d in docs if mode_bucket(d.mode) == "chat"]
    r_game = [max(0, reaction_total(d)) for d in docs if mode_bucket(d.mode) == "game"]
    thr_chat = max(config.min_chat_reactions, int(round(quantile(r_chat, config.chat_reaction_quantile)))) if r_chat else config.min_chat_reactions
    thr_game = max(config.min_game_reactions, int(round(quantile(r_game, config.game_reaction_quantile)))) if r_game else config.min_game_reactions

    # Dynamic spacing
    spacing_chat = max(config.min_spacing_chat, min(90.0, round(0.7 * config.expected_clip_chat)))
    spacing_game = max(config.min_spacing_game, min(90.0, round(0.7 * config.expected_clip_game)))

    # Get seed groups
    seed_chat = pick_top_reaction_groups(
        docs, k=k_chat, min_reactions=thr_chat, min_spacing=spacing_chat, mode_filter="chat"
    )
    seed_game = pick_top_reaction_groups(
        docs, k=k_game, min_reactions=thr_game, min_spacing=spacing_game, mode_filter="game"
    )

    # Convert to SeedGroup objects
    seed_groups = []
    for group in seed_chat:
        total_reacts = sum(reaction_total(docs[i]) for i in group)
        seed_groups.append(SeedGroup(indices=group, total_reactions=total_reacts, mode="chat"))
    
    for group in seed_game:
        total_reacts = sum(reaction_total(docs[i]) for i in group)
        seed_groups.append(SeedGroup(indices=group, total_reactions=total_reacts, mode="game"))

    return seed_groups
