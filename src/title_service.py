#!/usr/bin/env python3
"""
Centralized Title and Timestamp Generation Service.

Uses Gemini 3 Flash Preview for all title/timestamp generation with consistent,
simplified prompts that avoid AI-ish/blog-ish outputs.

Usage:
    from src.title_service import TitleService
    
    service = TitleService(vod_id)
    clip_title = service.generate_clip_title(transcript, streamer_name)
    arc_title = service.generate_arc_title(arc_summary, streamer_name)
    timestamps = service.generate_timestamps(ranges)
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the Gemini 3 Flash caller
from src.ai_client import call_gemini_3_flash

# Import transcript loader utility
try:
    from utils.transcript_loader import load_transcript_for_range, load_transcript_for_hms_range
    HAS_TRANSCRIPT_LOADER = True
except ImportError:
    HAS_TRANSCRIPT_LOADER = False
    load_transcript_for_range = None
    load_transcript_for_hms_range = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Title length limits (consistent across all types)
CLIP_TITLE_MAX_CHARS = 50  # Short clips/shorts
ARC_TITLE_MAX_CHARS = 100   # Arc/video titles
VIDEO_TITLE_MAX_CHARS = 80  # Full video titles

# Sample human-written titles for style reference (best examples from existing prompts)
SAMPLE_TITLES = [
    "<Streamer Name> on His Friendship",
    "The Most Chaotic PEAK Squad",
    "When Everything Goes Wrong",
    "<Streamer Name> Finally Knows How to Talk",
    "The Deadliest Impostor Duo",
    "becoming the greatest pokemon trainer",
    "the best moments of team toast",
    "the offlinetv PEAK experience",
    "What Mafia looks like when <Streamer Name> LOCKS IN",
    "we tried to stay fully incognito",
    "spending a day with the person i hate the most",
    "<Streamer Name> tries Outer Wilds for the first time",
    "The face off",
    "talking about the future of otv",
]


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class StreamContext:
    """Context about the stream for title generation."""
    streamer: str = ""
    vod_title: str = ""
    categories: List[str] = None
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = []
    
    def to_prompt_text(self) -> str:
        lines = []
        if self.streamer:
            lines.append(f"Streamer: {self.streamer}")
        if self.vod_title:
            lines.append(f"Stream title: {self.vod_title}")
        if self.categories:
            lines.append(f"Games/Categories: {', '.join(self.categories[:3])}")
        return "\n".join(lines) if lines else "Unknown streamer"


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def _clean_title(title: str, max_chars: int, streamer_name: str = "") -> str:
    """Clean and normalize a generated title."""
    if not title:
        return ""
    
    title = title.strip().strip('"\'')
    
    # Force remove the "Streamer Name: " or "Streamer Name - " prefix if AI hallucinates it
    if streamer_name:
        title = re.sub(f"^{re.escape(streamer_name)}[:\\s\\-]+", "", title, flags=re.IGNORECASE)
    
    # Remove common AI-isms
    title = re.sub(r'(?i)unfolding|chaos ensues|the journey|a surprising turn', '', title)
    
    # Remove AI commentary patterns
    title = re.sub(r'\s*\([^)]*(?:generated|based|analysis|provided|following)[^)]*\)\s*$', '', title, flags=re.IGNORECASE)
    
    # Remove trailing punctuation except ellipsis
    title = re.sub(r'[.,:;!?]+$', '', title)
    
    # Take only first line
    title = title.split('\n')[0].strip()
    
    # Enforce length limit
    if len(title) > max_chars:
        title = title[:max_chars].rsplit(' ', 1)[0].strip()
        
    return title


def _is_invalid_title(text: str) -> bool:
    """Check if title is clearly meta/invalid."""
    if not text or len(text) < 5:
        return True
    
    lowered = text.lower().strip()
    
    # Meta labels
    if lowered in {"title", "the title", "clip title", "video title"}:
        return True
    
    # AI instruction leakage
    bad_patterns = [
        "the user", "as an ai", "i am an ai", "as a language model",
        "here's a", "here is a", "based on the", "generated based", "Streamer"
    ]
    
    for pattern in bad_patterns:
        if pattern in lowered:
            return True
    
    return False


def _format_hms(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    s = int(max(0, round(float(seconds))))
    h = s // 3600
    m = (s % 3600) // 60
    s2 = s % 60
    return f"{h:02d}:{m:02d}:{s2:02d}"


def _load_stream_context(vod_id: str) -> StreamContext:
    """Load stream context from saved files."""
    try:
        from src.config import config
        sc_path = config.get_ai_data_dir(vod_id) / f"{vod_id}_stream_context.json"
        if sc_path.exists():
            data = json.loads(sc_path.read_text(encoding="utf-8"))
            cats = data.get("chapter_categories", [])
            return StreamContext(
                streamer=str(data.get("streamer") or "").strip(),
                vod_title=str(data.get("vod_title") or "").strip(),
                categories=[str(c).strip() for c in cats if str(c).strip()][:5] if isinstance(cats, list) else [],
            )
    except Exception:
        pass
    return StreamContext()


# -----------------------------------------------------------------------------
# Core Prompts (Simplified and Optimized for Gemini 3)
# -----------------------------------------------------------------------------

CLIP_TITLE_PROMPT = """Generate a short, high-click-through YouTube title for a clip.

CONTEXT:
{context}

TRANSCRIPT:
{transcript}

STYLE GUIDELINES:
- Use lowercase for a casual/modern feel (unless emphasizing a word).
- AVOID the "[Streamer] [Verbs] [Subject]" format. (e.g., No "Masayoshi plays Club Penguin")
- DYNAMIC PERSPECTIVE: Choose the most viral framing. Can be descriptive, use the streamer's name, or a direct quote.
- No "AI-speak": No "Unfolding", "Chaos ensues", "Journey", "Mastery", "Ultimate".
- Max {max_chars} characters.

PICK ONE ARCHETYPE (Mix it up):
1. THE HOOK: "the exact moment it all went wrong"
2. THE POV: "{streamer} actually can't believe this happened"
3. THE QUOTE: "my brain just completely broke"
4. THE DESCRIPTIVE: "the worst luck in the history of the game"
5. THE COMMENTARY: "club penguin is actually a horror game"

Return ONLY the title. No quotes."""


ARC_TITLE_PROMPT = """Generate a compelling YouTube video title for this segment.

CONTEXT:
{context}

CONTENT SUMMARY:
{summary}

RULES:
- Focus on the SPECIFIC story, event, or game match.
- IF GAMEPLAY: Mention the specific game, opponent, or key champion/strategy if relevant (e.g. "Game 3 vs Sentinels", "The 700 Stack Nasus").
- IF REACTION: Mention WHAT is being watched/reacted to (e.g. "Reacting to X", "Watching Y").
- Write it like a story arc, but include identifying details.
- Use "power words" that feel human, not like an AI blog.
- Max {max_chars} characters.

BAD EXAMPLES (Too Vague/Generic):
- The most intense match ever
- {streamer} reacts to a video
- The Civil War continues
- Chaos in the tournament
- Masayoshi's chaotic Club Penguin adventure

GOOD EXAMPLES (Specific & Clickable):
- {streamer} vs Sentinels: The 700 Stack Nasus Incident
- The Moment DSG Lost Game 5 vs Sentinels
- {streamer} Reacts to "The Fall of 100 Thieves"
- How {streamer} accidentally leaked the roster
- We built a cult in Lethal Company (Modded)
- {streamer} tried to become a club penguin god

Return ONLY the title."""


TIMESTAMP_PROMPT = """Create short, punchy chapter labels.

SEGMENTS:
{segments}

FORMAT:
- 2-5 words max.
- Write like a Twitch viewer would describe the moment.
- No "Streamer" or "He/She".
- No "Discussing" or "Talking About".

EXAMPLES:
- The Forbidden Lore
- Puzzle Fail
- Card Jitsu God Mode
- The Betrayal

Return JSON: {{"labels": ["Label 1", ...]}}"""


THUMBNAIL_TEXT_PROMPT = """Generate ONE short, viral thumbnail text (hook).

CONTEXT:
{context}

CONTENT SUMMARY:
{summary}

STRICT OUTPUT RULES:
1. Return ONLY the final text string. NO "thinking", NO explanations, NO multiple options.
2. shorter is better, max 6 words.
3. UPPERCASE ONLY.
4. NO PUNCTUATION (remove all periods, commas, exclamations).
5. NO EMOJIS.

Final Output:"""


DIRECTORS_CUT_PROMPT = """Generate a Director's Cut YouTube title.

STREAMER: {streamer}
GAMES: {games}
TOPICS: {topics}

RULES:
- 50-80 characters
- Personal and conversational tone
- Sound like a professional editor
- Focus on the most interesting event or dynamic
- Use specific details from the stream
- No generic terms like "highlights", "best moments"

STYLE PATTERNS:
- "[activity] adventures with [collaborator]"
- "{streamer} tried to force them to beat PEAK"
- "{streamer} didn't end stream until this happened"
- "the most chaotic [event] ever"
- "trying [game] for the first time"

Return ONLY the title, nothing else."""


# -----------------------------------------------------------------------------
# Title Service Class
# -----------------------------------------------------------------------------

class TitleService:
    """Centralized service for generating titles and timestamps using Gemini 3 Flash."""
    
    def __init__(self, vod_id: str):
        self.vod_id = vod_id
        self.context = _load_stream_context(vod_id)
    
    def generate_clip_title_for_range(
        self,
        start_seconds: float,
        end_seconds: float,
        streamer_name: Optional[str] = None,
        max_chars: int = CLIP_TITLE_MAX_CHARS,
    ) -> str:
        """Generate a title for a clip by loading transcript from ai_data files.
        
        This is the preferred method - it automatically loads the correct transcript
        from _filtered_ai_data.json for the given time range.
        
        Args:
            start_seconds: Clip start time in seconds
            end_seconds: Clip end time in seconds
            streamer_name: Override for streamer name
            max_chars: Maximum title length
            
        Returns:
            Generated title string
        """
        if not HAS_TRANSCRIPT_LOADER or load_transcript_for_range is None:
            print("⚠️ Transcript loader not available")
            return "Highlight Clip"
        
        # Load transcript from ai_data file
        transcript = load_transcript_for_range(self.vod_id, start_seconds, end_seconds)
        
        if not transcript or len(transcript.strip()) < 10:
            print(f"⚠️ No transcript found for range {start_seconds}s-{end_seconds}s")
            return "Highlight Clip"
        
        return self.generate_clip_title(transcript, streamer_name, max_chars)
    
    def generate_clip_title(
        self,
        transcript: str,
        streamer_name: Optional[str] = None,
        max_chars: int = CLIP_TITLE_MAX_CHARS,
    ) -> str:
        """Generate a title for a clip/short.
        
        Args:
            transcript: The transcript text for the clip window
            streamer_name: Override for streamer name (uses context if None)
            max_chars: Maximum title length
            
        Returns:
            Generated title string
        """
        streamer = streamer_name or self.context.streamer or "Streamer"
        
        if not transcript or len(transcript.strip()) < 10:
            print("⚠️ Empty or insufficient transcript provided")
            return "Highlight Clip"
        
        # Truncate transcript if too long
        transcript_text = transcript[:3000] if len(transcript) > 3000 else transcript
        
        prompt = CLIP_TITLE_PROMPT.format(
            context=self.context.to_prompt_text(),
            transcript=transcript_text,
            max_chars=max_chars,
            streamer=streamer,
        )
        
        try:
            result = call_gemini_3_flash(
                prompt,
                max_tokens=4096,
                temperature=0.9,
                request_tag=f"clip_title_{self.vod_id}",
            )
            
            title = _clean_title(result, max_chars, streamer)
            
            if _is_invalid_title(title):
                return "Highlight Clip"
            
            return title
            
        except Exception as e:
            print(f"⚠️ Clip title generation failed: {e}")
            return "Highlight Clip"
    
    def generate_arc_title_for_range(
        self,
        start_seconds: float,
        end_seconds: float,
        streamer_name: Optional[str] = None,
        max_chars: int = ARC_TITLE_MAX_CHARS,
    ) -> str:
        """Generate a title for an arc by loading transcript from ai_data files.
        
        Args:
            start_seconds: Arc start time in seconds
            end_seconds: Arc end time in seconds
            streamer_name: Override for streamer name
            max_chars: Maximum title length
            
        Returns:
            Generated title string
        """
        if not HAS_TRANSCRIPT_LOADER or load_transcript_for_range is None:
            print("⚠️ Transcript loader not available")
            return "Highlights"
        
        # Load transcript from ai_data file
        transcript = load_transcript_for_range(self.vod_id, start_seconds, end_seconds)
        
        if not transcript or len(transcript.strip()) < 10:
            print(f"⚠️ No transcript found for arc range {start_seconds}s-{end_seconds}s")
            return "Highlights"
        
        # Use transcript as the summary for arc title generation
        return self.generate_arc_title(transcript, streamer_name, max_chars)
    
    def generate_arc_title(
        self,
        summary: str,
        streamer_name: Optional[str] = None,
        max_chars: int = ARC_TITLE_MAX_CHARS,
    ) -> str:
        """Generate a title for an arc/video segment.
        
        Args:
            summary: Summary of the arc content (or transcript text)
            streamer_name: Override for streamer name
            max_chars: Maximum title length
            
        Returns:
            Generated title string
        """
        streamer = streamer_name or self.context.streamer or "Streamer"
        
        if not summary or len(summary.strip()) < 10:
            return "Highlights"
        
        prompt = ARC_TITLE_PROMPT.format(
            context=self.context.to_prompt_text(),
            summary=summary[:2000],
            max_chars=max_chars,
            streamer=streamer,
        )
        
        try:
            result = call_gemini_3_flash(
                prompt,
                max_tokens=4096,
                temperature=0.6,
                request_tag=f"arc_title_{self.vod_id}",
            )
            
            title = _clean_title(result, max_chars, streamer)
            
            if _is_invalid_title(title):
                # Fallback: use first part of summary
                fallback = summary[:max_chars].rsplit(' ', 1)[0] if summary else "Highlights"
                return fallback
            
            return title
            
        except Exception as e:
            print(f"⚠️ Arc title generation failed: {e}")
            return summary[:max_chars] if summary else "Highlights"
    
    def generate_directors_cut_title(
        self,
        topics: List[str] = None,
        games: List[str] = None,
        max_chars: int = VIDEO_TITLE_MAX_CHARS,
    ) -> str:
        """Generate a Director's Cut video title.
        
        Args:
            topics: List of topics covered in the video
            games: List of games played
            max_chars: Maximum title length
            
        Returns:
            Generated title string
        """
        streamer = self.context.streamer or "Streamer"
        games_list = games or self.context.categories or ["Gaming"]
        topics_list = topics or []
        
        prompt = DIRECTORS_CUT_PROMPT.format(
            streamer=streamer,
            games=", ".join(games_list[:3]),
            topics=", ".join(topics_list[:5]) if topics_list else "varied highlights",
        )
        
        try:
            result = call_gemini_3_flash(
                prompt,
                max_tokens=4096,
                temperature=0.6,
                request_tag=f"dc_title_{self.vod_id}",
            )
            
            title = _clean_title(result, max_chars, streamer)
            
            if _is_invalid_title(title):
                game = games_list[0] if games_list else "Gaming"
                return f"{game} Director's Cut"
            
            return title
            
        except Exception as e:
            print(f"⚠️ Director's Cut title generation failed: {e}")
            game = games_list[0] if games_list else "Gaming"
            return f"{game} Director's Cut"

    def generate_thumbnail_text(
        self,
        summary: str,
        streamer_name: Optional[str] = None,
    ) -> str:
        """Generate short text for a video thumbnail.
        
        Args:
            summary: Summary of the content (or transcript)
            streamer_name: Override for streamer name
            
        Returns:
            Short uppercase string (max 5 words)
        """
        streamer = streamer_name or self.context.streamer or "Streamer"
        
        if not summary or len(summary.strip()) < 10:
            return "HIGHLIGHTS"
        
        prompt = THUMBNAIL_TEXT_PROMPT.format(
            context=self.context.to_prompt_text(),
            summary=summary[:2000],
        )
        
        try:
            result = call_gemini_3_flash(
                prompt,
                max_tokens=1024,
                temperature=0.7,
                request_tag=f"thumb_text_{self.vod_id}",
            )
            
            # Cleanup
            text = result.strip().strip('"\'').upper()
            # Remove emojis (range of common emojis)
            text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
            # Remove trailing punctuation
            text = re.sub(r'[.,:;!?]+$', '', text)
            # Remove newlines
            text = text.replace('\n', ' ')
            
            if _is_invalid_title(text) or len(text) < 2:
                return "HIGHLIGHTS"
                
            return text.strip()
            
        except Exception as e:
            print(f"⚠️ Thumbnail text generation failed: {e}")
            return "HIGHLIGHTS"
    
    def generate_timestamps(
        self,
        ranges: List[Dict],
        arc_start_abs: int = 0,
    ) -> List[str]:
        """Generate timestamp labels for video chapters.
        
        Args:
            ranges: List of range dicts with 'start', 'end', 'summary' keys
                   OR with 'start_abs', 'end_abs' for absolute times
            arc_start_abs: Absolute start time of the arc (for relative timestamps)
            
        Returns:
            List of "HH:MM:SS Label" strings
        """
        if not ranges:
            return []
        
        # Build segment descriptions for the prompt
        # Load actual transcript for each range from ai_data files
        segments = []
        for i, r in enumerate(ranges[:20]):  # Limit to 20 segments
            # Get absolute start/end times
            start_abs = r.get("start_abs") or r.get("start", 0)
            end_abs = r.get("end_abs") or r.get("end", start_abs + 60)
            
            # Try to load actual transcript from ai_data
            transcript = ""
            if HAS_TRANSCRIPT_LOADER and load_transcript_for_range is not None:
                transcript = load_transcript_for_range(self.vod_id, start_abs, end_abs)
            
            # Fall back to summary if no transcript
            if not transcript:
                transcript = str(r.get("summary", "")).strip()
            
            topic = str(r.get("topic_key", "")).strip()
            energy = str(r.get("energy", "")).strip()
            
            if transcript:
                # Truncate long transcripts
                if len(transcript) > 500:
                    transcript = transcript[:500] + "..."
                segments.append(f"{i+1}. {transcript} (topic: {topic}, energy: {energy})")
        
        if not segments:
            return []
        
        prompt = TIMESTAMP_PROMPT.format(
            segments="\n".join(segments)
        )
        
        try:
            result = call_gemini_3_flash(
                prompt,
                max_tokens=8192,
                temperature=0.3,
                request_tag=f"timestamps_{self.vod_id}",
            )
            
            # Parse JSON response
            labels = self._parse_timestamp_labels(result)
            
            if not labels:
                # Fallback: use condensed summaries
                labels = [self._condense_summary(r.get("summary", ""), 60) or "Highlights" for r in ranges[:20]]
            
            # Build timestamp lines
            lines = []
            for i, r in enumerate(ranges[:len(labels)]):
                # Support both 'start' and 'start_abs' keys
                start_time = r.get("start_abs") or r.get("start", 0)
                start_rel = max(0, int(start_time) - arc_start_abs)
                stamp = _format_hms(start_rel)
                label = labels[i] if i < len(labels) else "Highlights"
                lines.append(f"{stamp} {label}")
            
            return lines
            
        except Exception as e:
            print(f"⚠️ Timestamp generation failed: {e}")
            # Fallback
            lines = []
            for r in ranges[:15]:
                start_rel = max(0, int(r.get("start", 0)) - arc_start_abs)
                stamp = _format_hms(start_rel)
                summary = self._condense_summary(r.get("summary", ""), 50) or "Highlights"
                lines.append(f"{stamp} {summary}")
            return lines
    
    def _parse_timestamp_labels(self, response: str) -> List[str]:
        """Parse JSON labels from Gemini response."""
        try:
            # Try to extract JSON
            text = response.strip()
            if text.startswith("```"):
                text = text.split("```", 2)[1] if "```" in text else text
                if text.startswith("json"):
                    text = text[4:].strip()
            
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                obj = json.loads(text[start:end+1])
                labels = obj.get("labels", [])
                if isinstance(labels, list):
                    return [str(label).strip() for label in labels if str(label).strip()]
        except Exception:
            pass
        return []
    
    def _condense_summary(self, text: str, max_chars: int) -> str:
        """Condense text to max_chars while preserving word boundaries."""
        if not text:
            return ""
        text = text.replace("→", ", ").replace("\n", " ")
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= max_chars:
            return text
        cut = text[:max_chars]
        if " " in cut:
            cut = cut[:cut.rfind(" ")]
        return cut.strip()


# -----------------------------------------------------------------------------
# Convenience Functions
# -----------------------------------------------------------------------------

def generate_clip_title(vod_id: str, transcript: str, streamer_name: Optional[str] = None) -> str:
    """Convenience function to generate a clip title."""
    service = TitleService(vod_id)
    return service.generate_clip_title(transcript, streamer_name)


def generate_arc_title(vod_id: str, summary: str, streamer_name: Optional[str] = None) -> str:
    """Convenience function to generate an arc title."""
    service = TitleService(vod_id)
    return service.generate_arc_title(summary, streamer_name)


def generate_directors_cut_title(vod_id: str, topics: List[str] = None, games: List[str] = None) -> str:
    """Convenience function to generate a Director's Cut title."""
    service = TitleService(vod_id)
    return service.generate_directors_cut_title(topics, games)


def generate_thumbnail_text(vod_id: str, summary: str, streamer_name: Optional[str] = None) -> str:
    """Convenience function to generate thumbnail text."""
    service = TitleService(vod_id)
    return service.generate_thumbnail_text(summary, streamer_name)


def generate_timestamps(vod_id: str, ranges: List[Dict], arc_start_abs: int = 0) -> List[str]:
    """Convenience function to generate timestamps."""
    service = TitleService(vod_id)
    return service.generate_timestamps(ranges, arc_start_abs)

