"""
Configuration defaults for clip generation.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ClipConfig:
    """Configuration for clip generation pipeline."""
    
    # Target lengths (seconds)
    expected_clip_chat: float = 55.0
    expected_clip_game: float = 75.0
    min_clip_duration: float = 30.0
    max_clip_duration: float = 180.0
    
    # Dynamic length adjustments
    high_signal_shorten: float = 15.0  # seconds to shorten for high signals
    energy_shorten: float = 10.0       # seconds to shorten for high energy
    reaction_shorten: float = 10.0    # seconds to shorten for high reactions
    
    # Pre-roll ratios by mode
    chat_pre_ratio: float = 0.35      # 35% before anchor for chat
    game_pre_ratio: float = 0.25      # 25% before anchor for game
    
    # Quality gates
    min_score_threshold: float = 4.0  # Increased from 2.0 for higher quality clips
    low_energy_energy_frac: float = 0.2
    low_energy_chat_z: float = 0.35
    low_energy_reactions: int = 5
    long_windup_threshold: float = 0.75  # anchor must be within first 75%
    
    # Spacing and deduplication
    min_spacing_chat: float = 45.0
    min_spacing_game: float = 45.0
    dedup_iou_threshold: float = 0.5
    dedup_center_spacing: float = 50.0
    
    # Grouping
    max_gap_seconds: float = 40.0
    max_arc_duration: float = 180.0
    semantic_time_window: float = 60.0
    semantic_similarity_threshold: float = 0.40
    
    # Final padding
    default_front_pad: float = 10.0  # Increased from 1.0 for better context
    default_back_pad: float = 1.0
    max_left_pad: float = 30.0
    
    # Seeding
    seed_multiplier: float = 1.25
    max_seeds: int = 24
    min_seeds: int = 4
    
    # Reaction thresholds (quantile-based)
    chat_reaction_quantile: float = 0.85
    game_reaction_quantile: float = 0.85
    min_chat_reactions: int = 2
    min_game_reactions: int = 3


# Default configuration
DEFAULT_CONFIG = ClipConfig()
