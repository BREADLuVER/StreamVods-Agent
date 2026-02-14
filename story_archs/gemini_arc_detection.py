#!/usr/bin/env python3
"""
Gemini-based arc detection for VOD transcripts.

Processes transcript chunk-by-chunk (30 min) to detect narrative arcs:
- INTRO: Beginning of a new activity, game round, or topic
- BUILD-UP: Tension or engagement increasing
- CLIMAX: Peak moment - highest energy, biggest reaction
- RESOLUTION: Immediate aftermath

This module replaces rag.narrative_analyzer for video creation,
providing more accurate arc boundaries using Gemini's language understanding.

Usage:
    python -m story_archs.gemini_arc_detection <vod_id> [--model MODEL] [--save]
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment
try:
    from dotenv import load_dotenv

    load_dotenv("config/streamsniped.env")
except ImportError:
    pass


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class Arc:
    arc_id: int
    arc_type: str
    intro_start: float
    intro_end: float
    climax_start: float
    climax_end: float
    resolution_start: Optional[float] = None
    resolution_end: Optional[float] = None
    confidence: float = 0.0
    summary: str = ""
    controversy_score: float = 0.0
    narrative_score: float = 0.0
    chunk_index: int = 0
    chapter: str = ""


@dataclass
class FillerSegment:
    start: float
    end: float
    reason: str


@dataclass
class ChunkResult:
    chunk_index: int
    arcs: List[Arc] = field(default_factory=list)
    filler_segments: List[FillerSegment] = field(default_factory=list)
    incomplete_arc: Optional[Dict] = None
    chunk_summary: str = ""


@dataclass
class AgentState:
    previous_chunk_summary: str = ""
    incomplete_arc: Optional[Dict] = None
    arc_count: int = 0
    current_game: Optional[str] = None


# -----------------------------------------------------------------------------
# Prompt Template
# -----------------------------------------------------------------------------

CHUNK_ARC_DETECTION_PROMPT = """You are analyzing a 30-minute chunk of a Twitch stream transcript (plus overlap) to detect narrative arcs.

CONTEXT FROM PREVIOUS CHUNK:
{previous_context}

CURRENT GAME/CATEGORY: {current_game}

---

TRANSCRIPT (with timestamps in seconds from stream start):
NOTE: Includes OVERLAP from the next chunk.
{transcript}

---

CHAT ACTIVITY PEAKS (moments with high chat engagement):
{chat_peaks}

---

YOUR TASK: Identify ARC BOUNDARIES in this chunk.

We are looking for specific types of content. Use these definitions:

1. **GAMEPLAY / EVENT** (Standard Arc)
   - Structure: INTRO → BUILD-UP → CLIMAX → RESOLUTION
   - Definition: A match, a boss fight, a round, or a specific in-game event.
   - Climax: High energy, yelling, big plays, victory/defeat.

2. **PARASOCIAL_YAP** (Topic/Story)
   - Structure: TOPIC_START → MAIN_POINT/RANT → CONCLUSION
   - Definition: Streamer talking to chat about a specific topic (news, drama, life story, opinion).
   - Climax: The "hottest take", the most controversial statement, or the punchline of the story. *Note: Energy might be lower than gameplay.*

3. **REACTION** (Watching Content)
   - Structure: CONTENT_START → REACTION_PEAK → CONTENT_END
   - Definition: Streamer watching a video, trailer, or reading a tweet/article.
   - Climax: The moment of biggest reaction (laughing, shock, pausing to comment).

4. **FILLER** (Ignore)
   - AFK, technical issues, silent grinding, waiting in queue.

OUTPUT FORMAT (JSON only, no markdown):
{{
  "arcs": [
    {{
      "arc_type": "gameplay|parasocial_yap|reaction",
      "intro_start_seconds": <number>,
      "intro_end_seconds": <number>,
      "climax_start_seconds": <number>,
      "climax_end_seconds": <number>,
      "resolution_start_seconds": <number or null if incomplete>,
      "resolution_end_seconds": <number or null if incomplete>,
      "confidence": <0.0-1.0>,
      "controversy_score": <1-10>,
      "narrative_score": <1-10>,
      "summary": "<one sentence describing this arc>"
    }}
  ],
  "filler_segments": [
    {{"start_seconds": <number>, "end_seconds": <number>, "reason": "AFK|tech_issue|repetitive_attempts|waiting"}}
  ],
  "incomplete_arc": {{
    "exists": <true/false>,
    "phase": "intro|build_up|climax",
    "started_at_seconds": <number>,
    "description": "<what's happening that continues next chunk>"
  }},
  "chunk_summary": "<2-3 sentences summarizing this chunk>"
}}

RULES:
1. **OPEN TRANSACTION PRIORITY**: If the "CONTEXT FROM PREVIOUS CHUNK" says we are in an `incomplete_arc`, your PRIMARY JOB is to find the climax/resolution of that specific arc.
2. **SCORING**:
    - `controversy_score` (1-10): Rate how controversial/divisive this topic is (1=wholesome/safe, 10=cancellable/heated debate/drama).
    - `narrative_score` (1-10): Rate how interesting the story/topic is for a YouTube video title (1=boring/grinding, 10=must-click/unique event).
3. Arcs must be at least 3 minutes long.
4. **OVERLAP HANDLING**: The transcript includes extra time from the NEXT chunk. If an arc climaxes or resolves in this overlap, INCLUDE IT FULLY in this chunk's output. **Claim it now** rather than leaving it for the next chunk.
5. **CONTINUITY**: If you are finishing an "incomplete_arc" from the previous chunk, you MUST use the original `started_at_seconds` from the context as the `intro_start_seconds`.
6. If an arc truly has no resolution even after the overlap, mark it in `incomplete_arc`.
7. Be conservative - better to have fewer confident arcs than many uncertain ones.

Output JSON only, no explanation text."""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _format_hms(sec: float) -> str:
    s = int(round(max(0.0, float(sec))))
    h = s // 3600
    m = (s % 3600) // 60
    s2 = s % 60
    return f"{h:02d}:{m:02d}:{s2:02d}"


def _format_duration_display(sec: float) -> str:
    s = int(round(max(0.0, float(sec))))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        m = s // 60
        s2 = s % 60
        return f"{m}m {s2}s" if s2 > 0 else f"{m}m"
    h = s // 3600
    m = (s % 3600) // 60
    return f"{h}h {m}m" if m > 0 else f"{h}h"


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------


def load_segments_for_chunk(
    vod_id: str, chunk_index: int, chunk_duration_seconds: int = 1800, overlap_seconds: int = 900
) -> Tuple[List[Dict], float, float]:
    """Load segments for a specific 30-min chunk (plus optional overlap)."""
    ai_data_path = Path(f"data/ai_data/{vod_id}/{vod_id}_filtered_ai_data.json")
    if not ai_data_path.exists():
        ai_data_path = Path(f"data/ai_data/{vod_id}/{vod_id}_ai_data.json")

    if not ai_data_path.exists():
        raise FileNotFoundError(f"AI data not found: {ai_data_path}")

    with open(ai_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_segments = data.get("segments", [])
    if not all_segments:
        raise ValueError("No segments found in AI data")

    vod_start = all_segments[0].get("start_time", 0)
    chunk_start = vod_start + (chunk_index * chunk_duration_seconds)
    chunk_end = chunk_start + chunk_duration_seconds
    fetch_end = chunk_end + overlap_seconds

    chunk_segments = [
        seg
        for seg in all_segments
        if seg.get("start_time", 0) >= chunk_start
        and seg.get("start_time", 0) < fetch_end
    ]

    return chunk_segments, chunk_start, chunk_end


def get_total_chunks(vod_id: str, chunk_duration_seconds: int = 1800) -> int:
    """Get total number of 30-min chunks in the VOD."""
    ai_data_path = Path(f"data/ai_data/{vod_id}/{vod_id}_filtered_ai_data.json")
    if not ai_data_path.exists():
        ai_data_path = Path(f"data/ai_data/{vod_id}/{vod_id}_ai_data.json")

    with open(ai_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        return 0

    vod_start = segments[0].get("start_time", 0)
    vod_end = segments[-1].get("end_time", segments[-1].get("start_time", 0))
    total_duration = vod_end - vod_start

    return max(1, int(total_duration / chunk_duration_seconds) + 1)


def load_chapters(vod_id: str) -> List[Dict]:
    """Load chapter data for context."""
    chapters_path = Path(f"data/ai_data/{vod_id}/{vod_id}_chapters.json")
    if not chapters_path.exists():
        return []

    with open(chapters_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("chapters", [])


def get_game_for_timestamp(chapters: List[Dict], timestamp_seconds: float) -> str:
    """Get the game/category being played at a specific timestamp."""
    for chapter in chapters:
        start = chapter.get("start_time", 0)
        end = chapter.get("end_time", 0)
        if start <= timestamp_seconds < end:
            return chapter.get("original_category", chapter.get("category", "Unknown"))
    return "Unknown"


def load_burst_data_for_chunk(
    vod_id: str, chunk_start: float, chunk_end: float
) -> List[Dict]:
    """Load burst data from metadata.db for climax enhancement."""
    db_path = Path(f"data/vector_stores/{vod_id}/metadata.db")
    if not db_path.exists():
        return []

    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, start_time, end_time, chat_rate_z, burst_score, reaction_hits, energy
            FROM documents
            WHERE start_time >= ? AND start_time < ?
            ORDER BY start_time
        """,
            (chunk_start, chunk_end),
        )
        rows = cur.fetchall()
        conn.close()

        bursts = []
        for row in rows:
            reaction_hits = {}
            if row[5]:
                try:
                    reaction_hits = (
                        json.loads(row[5]) if isinstance(row[5], str) else row[5]
                    )
                except Exception:
                    pass

            total_reactions = (
                sum(int(v) for v in reaction_hits.values())
                if isinstance(reaction_hits, dict)
                else 0
            )

            bursts.append(
                {
                    "id": row[0],
                    "start_time": row[1],
                    "end_time": row[2],
                    "chat_rate_z": row[3] or 0.0,
                    "burst_score": row[4] or 0.0,
                    "total_reactions": total_reactions,
                    "energy": row[6] or "low",
                }
            )
        return bursts
    except Exception:
        return []


# -----------------------------------------------------------------------------
# Transcript Formatting
# -----------------------------------------------------------------------------


def format_transcript_for_prompt(segments: List[Dict], max_chars: int = 200000) -> str:
    """Format segments into a readable transcript for the prompt."""
    lines = []
    total_chars = 0

    for seg in segments:
        start = seg.get("start_time", 0)
        transcript = seg.get("transcript", "").strip()
        if not transcript:
            continue

        minutes = int(start // 60)
        seconds = int(start % 60)
        line = f"[{minutes:02d}:{seconds:02d}] {transcript}"

        if total_chars + len(line) > max_chars:
            lines.append("... [transcript truncated for length]")
            break

        lines.append(line)
        total_chars += len(line) + 1

    return "\n".join(lines)


def extract_chat_peaks(segments: List[Dict], top_k: int = 15) -> List[Dict]:
    """Extract top chat activity moments."""
    peaks = []

    for seg in segments:
        chat_activity = seg.get("chat_activity", 0)
        chat_messages = seg.get("chat_messages", [])
        if chat_activity > 0 or len(chat_messages) > 3:
            start = seg.get("start_time", 0)
            sample_msgs = [msg.get("content", "")[:50] for msg in chat_messages[:5]]

            peaks.append(
                {
                    "timestamp_seconds": start,
                    "activity_score": chat_activity or len(chat_messages),
                    "sample_messages": sample_msgs,
                }
            )

    peaks.sort(key=lambda x: x["activity_score"], reverse=True)
    return peaks[:top_k]


# -----------------------------------------------------------------------------
# Gemini API
# -----------------------------------------------------------------------------


def call_gemini(prompt: str, model: str = "gemini-3-flash-preview") -> str:
    """Call Gemini API with the given prompt."""
    try:
        from google import genai
    except ImportError:
        raise ImportError("google-genai not installed. Run: pip install google-genai")

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set"
        )

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    return response.text


def parse_gemini_response(response_text: str) -> Dict:
    """Parse JSON from Gemini response, handling markdown code blocks."""
    text = response_text.strip()

    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]

    return json.loads(text)


# -----------------------------------------------------------------------------
# Arc Detection Agent
# -----------------------------------------------------------------------------


class ArcDetectionAgent:
    """Agent that processes VOD chunk by chunk, maintaining state."""

    def __init__(self, vod_id: str, model: str = "gemini-3-flash-preview"):
        self.vod_id = vod_id
        self.model = model
        self.state = AgentState()
        self.chapters = load_chapters(vod_id)
        self.all_results: List[ChunkResult] = []

    def process_chunk(self, chunk_index: int, dry_run: bool = False) -> ChunkResult:
        """Process a single 30-min chunk of the VOD."""
        print(f"\n{'=' * 60}")
        print(f"Processing Chunk {chunk_index + 1}")
        print(f"{'=' * 60}")

        # Add 15 minutes (900s) overlap to catch arcs that end just after the chunk mark
        segments, chunk_start, chunk_end = load_segments_for_chunk(
            self.vod_id, chunk_index, overlap_seconds=900
        )

        if not segments:
            print(f"  No segments found for chunk {chunk_index + 1}")
            return ChunkResult(chunk_index=chunk_index)

        print(f"  Time range: {_format_hms(chunk_start)} - {_format_hms(chunk_end)}")
        print(f"  Segments: {len(segments)}")

        current_game = get_game_for_timestamp(self.chapters, chunk_start)
        print(f"  Game/Category: {current_game}")

        transcript = format_transcript_for_prompt(segments)
        chat_peaks = extract_chat_peaks(segments)

        print(f"  Transcript length: {len(transcript)} chars")
        print(f"  Chat peaks: {len(chat_peaks)}")

        if self.state.previous_chunk_summary:
            previous_context = (
                f"Previous chunk summary: {self.state.previous_chunk_summary}"
            )
            if self.state.incomplete_arc:
                previous_context += f"\n\n**INCOMPLETE ARC from previous chunk** (PRIORITY): \n{json.dumps(self.state.incomplete_arc, indent=2)}"
        else:
            previous_context = "This is the first chunk of the stream."

        prompt = CHUNK_ARC_DETECTION_PROMPT.format(
            previous_context=previous_context,
            current_game=current_game,
            transcript=transcript,
            chat_peaks=json.dumps(chat_peaks, indent=2),
        )

        print(f"  Prompt length: {len(prompt)} chars (~{len(prompt) // 4} tokens)")

        if dry_run:
            print("\n  [DRY RUN] Would send prompt to Gemini")
            return ChunkResult(chunk_index=chunk_index)

        print("\n  Calling Gemini API...")
        try:
            response_text = call_gemini(prompt, model=self.model)
            print(f"  Response length: {len(response_text)} chars")
        except Exception as e:
            print(f"  ERROR calling Gemini: {e}")
            return ChunkResult(chunk_index=chunk_index)

        try:
            result_data = parse_gemini_response(response_text)
        except json.JSONDecodeError as e:
            print(f"  ERROR parsing response: {e}")
            print(f"  Raw response: {response_text[:500]}...")
            return ChunkResult(chunk_index=chunk_index)

        result = ChunkResult(chunk_index=chunk_index)

        for i, arc_data in enumerate(result_data.get("arcs", [])):
            arc = Arc(
                arc_id=self.state.arc_count + i,
                arc_type=arc_data.get("arc_type", "unknown"),
                intro_start=arc_data.get("intro_start_seconds", 0),
                intro_end=arc_data.get("intro_end_seconds", 0),
                climax_start=arc_data.get("climax_start_seconds", 0),
                climax_end=arc_data.get("climax_end_seconds", 0),
                resolution_start=arc_data.get("resolution_start_seconds"),
                resolution_end=arc_data.get("resolution_end_seconds"),
                confidence=arc_data.get("confidence", 0),
                controversy_score=arc_data.get("controversy_score", 0),
                narrative_score=arc_data.get("narrative_score", 0),
                summary=arc_data.get("summary", ""),
                chunk_index=chunk_index,
                chapter=current_game,
            )
            result.arcs.append(arc)

        for filler_data in result_data.get("filler_segments", []):
            filler = FillerSegment(
                start=filler_data.get("start_seconds", 0),
                end=filler_data.get("end_seconds", 0),
                reason=filler_data.get("reason", "unknown"),
            )
            result.filler_segments.append(filler)

        incomplete = result_data.get("incomplete_arc", {})
        if incomplete.get("exists"):
            result.incomplete_arc = incomplete

        result.chunk_summary = result_data.get("chunk_summary", "")

        self.state.previous_chunk_summary = result.chunk_summary
        self.state.incomplete_arc = result.incomplete_arc
        self.state.arc_count += len(result.arcs)
        self.state.current_game = current_game

        self.all_results.append(result)
        self._print_chunk_summary(result)

        return result

    def process_all_chunks(self, dry_run: bool = False) -> List[ChunkResult]:
        """Process all chunks of the VOD."""
        total_chunks = get_total_chunks(self.vod_id)
        print(f"\nProcessing {total_chunks} chunks (30-min) for VOD {self.vod_id}")

        for chunk_index in range(total_chunks):
            self.process_chunk(chunk_index, dry_run=dry_run)
            if dry_run:
                break

        return self.all_results

    def _print_chunk_summary(self, result: ChunkResult):
        """Print a summary of the chunk's results."""
        print(f"\n  RESULTS:")
        print("  --------")
        print(f"  Arcs detected: {len(result.arcs)}")

        for arc in result.arcs:
            print(
                f"\n    Arc {arc.arc_id}: {arc.arc_type} (confidence: {arc.confidence:.2f})"
            )
            print(f"      Score: controversy={arc.controversy_score}, narrative={arc.narrative_score}")
            print(
                f"      Intro: {_format_hms(arc.intro_start)} - {_format_hms(arc.intro_end)}"
            )
            print(
                f"      Climax: {_format_hms(arc.climax_start)} - {_format_hms(arc.climax_end)}"
            )
            if arc.resolution_start:
                print(
                    f"      Resolution: {_format_hms(arc.resolution_start)} - {_format_hms(arc.resolution_end or arc.resolution_start)}"
                )
            else:
                print("      Resolution: (incomplete)")
            print(f"      Summary: {arc.summary}")

        print(f"\n  Filler segments: {len(result.filler_segments)}")
        for filler in result.filler_segments:
            print(
                f"    {_format_hms(filler.start)} - {_format_hms(filler.end)}: {filler.reason}"
            )

        if result.incomplete_arc:
            print("\n  Incomplete arc (continues next chunk):")
            print(f"    Phase: {result.incomplete_arc.get('phase')}")
            print(f"    Description: {result.incomplete_arc.get('description')}")

        print(f"\n  Chunk summary: {result.chunk_summary}")


# -----------------------------------------------------------------------------
# Climax Enhancement with Burst Data
# -----------------------------------------------------------------------------


def enhance_arcs_with_burst_data(vod_id: str, arcs: List[Arc]) -> List[Arc]:
    """Enhance Gemini-detected climaxes with burst data metrics.

    The climax BOUNDARIES come from Gemini (it knows the narrative).
    The climax SCORE comes from burst data (chat_rate_z, reactions).
    """
    enhanced = []

    for arc in arcs:
        # Load burst data for this arc's time range
        bursts = load_burst_data_for_chunk(
            vod_id,
            arc.intro_start,
            arc.resolution_end or arc.climax_end + 300,  # 5 min buffer if no resolution
        )

        if not bursts:
            enhanced.append(arc)
            continue

        # Find best burst within the climax window
        climax_bursts = [
            b
            for b in bursts
            if b["start_time"] >= arc.climax_start - 30
            and b["end_time"] <= arc.climax_end + 30
        ]

        if climax_bursts:
            # Calculate climax score from burst data
            best_burst = max(
                climax_bursts,
                key=lambda b: (
                    b["chat_rate_z"] * 0.4
                    + b["burst_score"] * 0.3
                    + b["total_reactions"] * 0.3
                ),
            )

            # Store burst metrics on the arc (we'll use these in the manifest)
            arc.climax_score = (
                best_burst["chat_rate_z"] * 0.4
                + best_burst["burst_score"] * 0.3
                + best_burst["total_reactions"] * 0.1
            )
            arc.peak_chat_rate_z = best_burst["chat_rate_z"]
            arc.total_reactions = best_burst["total_reactions"]
            arc.climax_source = "burst_data"
        else:
            arc.climax_score = 0.0
            arc.peak_chat_rate_z = 0.0
            arc.total_reactions = 0
            arc.climax_source = "gemini_only"

        enhanced.append(arc)

    return enhanced


def deduplicate_arcs(arcs: List[Arc], time_threshold: float = 60.0) -> List[Arc]:
    """Deduplicate arcs that are likely the same event caught in adjacent chunks.

    Strategy:
    1. Sort by intro_start.
    2. Iterate and check for overlap with previous arc.
    3. If start times are close OR substantial time overlap exists:
       - Merge into a single arc.
       - Prefer the version with higher confidence or better resolution.
    """
    if not arcs:
        return []

    # Sort strictly by start time
    sorted_arcs = sorted(arcs, key=lambda x: x.intro_start)
    merged = []

    current = sorted_arcs[0]

    for next_arc in sorted_arcs[1:]:
        # Check for proximity or overlap
        time_diff = abs(next_arc.intro_start - current.intro_start)
        overlap = max(0, min(current.resolution_end or current.climax_end, next_arc.resolution_end or next_arc.climax_end) - max(current.intro_start, next_arc.intro_start))
        
        # Merge conditions:
        # 1. Start within 2 minutes of each other
        # 2. OR Strong overlap (> 50% of the shorter arc)
        is_duplicate = False
        if time_diff < 120:
            is_duplicate = True
        elif overlap > 0:
            duration_current = (current.resolution_end or current.climax_end) - current.intro_start
            duration_next = (next_arc.resolution_end or next_arc.climax_end) - next_arc.intro_start
            min_duration = min(duration_current, duration_next)
            if min_duration > 0 and (overlap / min_duration) > 0.5:
                is_duplicate = True

        if is_duplicate:
            # MERGE LOGIC
            # Keep the one with better resolution status first, then higher confidence
            current_resolved = current.resolution_start is not None
            next_resolved = next_arc.resolution_start is not None

            if next_resolved and not current_resolved:
                # Next one is better (completed the story)
                current = next_arc
            elif current_resolved and not next_resolved:
                # Current is better, keep it
                pass
            else:
                # Both resolved or both incomplete: prefer higher confidence
                if next_arc.confidence > current.confidence:
                    current = next_arc
                # Else keep current
            
            print(f"  [MERGE] Merged duplicate arc starting at {_format_hms(next_arc.intro_start)}")
        else:
            merged.append(current)
            current = next_arc

    merged.append(current)
    return merged


# -----------------------------------------------------------------------------
# Save Results
# -----------------------------------------------------------------------------


def save_gemini_arcs(vod_id: str, results: List[ChunkResult]) -> Path:
    """Save raw Gemini arc detection results."""
    output_dir = Path(f"data/ai_data/{vod_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{vod_id}_gemini_arcs.json"

    all_arcs = []
    all_filler = []

    for result in results:
        for arc in result.arcs:
            all_arcs.append(
                {
                    "arc_id": arc.arc_id,
                    "arc_type": arc.arc_type,
                    "intro_start": arc.intro_start,
                    "intro_end": arc.intro_end,
                    "climax_start": arc.climax_start,
                    "climax_end": arc.climax_end,
                    "resolution_start": arc.resolution_start,
                    "resolution_end": arc.resolution_end,
                    "confidence": arc.confidence,
                    "controversy_score": arc.controversy_score,
                    "narrative_score": arc.narrative_score,
                    "summary": arc.summary,
                    "chunk_index": arc.chunk_index,
                    "chapter": arc.chapter,
                }
            )

        for filler in result.filler_segments:
            all_filler.append(
                {
                    "start": filler.start,
                    "end": filler.end,
                    "reason": filler.reason,
                }
            )

    data = {
        "vod_id": vod_id,
        "total_arcs": len(all_arcs),
        "total_filler_segments": len(all_filler),
        "arcs": all_arcs,
        "filler_segments": all_filler,
        "chunks": [
            {
                "chunk_index": r.chunk_index,
                "arc_count": len(r.arcs),
                "filler_count": len(r.filler_segments),
                "chunk_summary": r.chunk_summary,
                "incomplete_arc": r.incomplete_arc,
            }
            for r in results
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] Gemini arcs saved to: {output_path}")
    return output_path


def generate_arc_manifest(vod_id: str, arcs: List[Arc], chapters: List[Dict]) -> Path:
    """Generate the gemini_arc_manifest.json in the format expected by gemini_to_arc_manifests.py."""
    output_dir = Path(f"data/vector_stores/{vod_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gemini_arc_manifest.json"

    # Enhance arcs with burst data
    enhanced_arcs = enhance_arcs_with_burst_data(vod_id, arcs)

    # Build arc entries
    arc_entries = []
    arc_types_count: Dict[str, int] = {}
    total_duration = 0.0

    for arc in enhanced_arcs:
        # Calculate arc boundaries
        start = arc.intro_start
        end = (
            arc.resolution_end or arc.climax_end + 60
        )  # Default 60s after climax if no resolution
        duration = end - start
        total_duration += duration

        arc_types_count[arc.arc_type] = arc_types_count.get(arc.arc_type, 0) + 1

        entry = {
            "arc_id": arc.arc_id,
            "arc_type": arc.arc_type,
            "chapter": arc.chapter,
            "start": start,
            "start_hms": _format_hms(start),
            "end": end,
            "end_hms": _format_hms(end),
            "duration": duration,
            "duration_display": _format_duration_display(duration),
            "intro": {
                "start": arc.intro_start,
                "start_hms": _format_hms(arc.intro_start),
                "end": arc.intro_end,
                "end_hms": _format_hms(arc.intro_end),
                "duration": arc.intro_end - arc.intro_start,
            },
            "climax": {
                "start": arc.climax_start,
                "start_hms": _format_hms(arc.climax_start),
                "end": arc.climax_end,
                "end_hms": _format_hms(arc.climax_end),
                "duration": arc.climax_end - arc.climax_start,
                "source": getattr(arc, "climax_source", "gemini"),
                "score": getattr(arc, "climax_score", 0.0),
                "peak_chat_rate_z": getattr(arc, "peak_chat_rate_z", 0.0),
                "total_reactions": getattr(arc, "total_reactions", 0),
            },
            "resolution": {
                "start": arc.resolution_start,
                "start_hms": _format_hms(arc.resolution_start)
                if arc.resolution_start
                else None,
                "end": arc.resolution_end,
                "end_hms": _format_hms(arc.resolution_end)
                if arc.resolution_end
                else None,
                "duration": (arc.resolution_end - arc.resolution_start)
                if arc.resolution_start and arc.resolution_end
                else 0,
            },
            "confidence": arc.confidence,
            "controversy_score": arc.controversy_score,
            "narrative_score": arc.narrative_score,
            "summary": arc.summary,
        }
        arc_entries.append(entry)

    # Build chapter entries
    chapter_entries = []
    for ch in chapters:
        chapter_entries.append(
            {
                "category": ch.get("original_category", ch.get("category", "Unknown")),
                "start": ch.get("start_time", 0),
                "start_hms": _format_hms(ch.get("start_time", 0)),
                "end": ch.get("end_time", 0),
                "end_hms": _format_hms(ch.get("end_time", 0)),
            }
        )

    manifest = {
        "vod_id": vod_id,
        "step": "generate_arc_manifest",
        "stats": {
            "total_arcs": len(arc_entries),
            "total_content_duration": total_duration,
            "total_content_duration_display": _format_duration_display(total_duration),
            "average_arc_duration": total_duration / len(arc_entries)
            if arc_entries
            else 0,
            "average_arc_duration_display": _format_duration_display(
                total_duration / len(arc_entries)
            )
            if arc_entries
            else "0s",
            "total_filler_segments": 0,  # We'll add this from results if needed
            "total_filler_duration": 0.0,
            "total_filler_duration_display": "0s",
            "arc_types": arc_types_count,
        },
        "filters": {
            "min_score": 0.0,
            "min_confidence": 0.0,
            "include_filler": False,
        },
        "chapters": chapter_entries,
        "arcs": arc_entries,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Arc manifest saved to: {output_path}")
    return output_path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def run_arc_detection(
    vod_id: str,
    model: str = "gemini-3-flash-preview",
    save: bool = True,
    dry_run: bool = False,
) -> Tuple[List[ChunkResult], Optional[Path]]:
    """Run the full arc detection pipeline."""
    print("[INFO] Gemini Arc Detection (30-min Chunks)")
    print(f"   VOD: {vod_id}")
    print(f"   Model: {model}")

    chapters = load_chapters(vod_id)
    if chapters:
        print(f"   Chapters: {len(chapters)}")
        for ch in chapters:
            cat = ch.get("original_category", ch.get("category"))
            print(
                f"     - {cat}: {_format_hms(ch.get('start_time', 0))} - {_format_hms(ch.get('end_time', 0))}"
            )

    agent = ArcDetectionAgent(vod_id, model=model)
    results = agent.process_all_chunks(dry_run=dry_run)

    manifest_path = None
    if save and not dry_run:
        # Collect all arcs
        all_arcs = []
        for r in results:
            all_arcs.extend(r.arcs)

        # Deduplicate arcs before saving
        print("\n[INFO] Deduplicating arcs...")
        unique_arcs = deduplicate_arcs(all_arcs)
        print(f"  Reduced {len(all_arcs)} arcs to {len(unique_arcs)} unique arcs.")

        # Save raw results (still keeps original chunk structure for debugging)
        save_gemini_arcs(vod_id, results)

        # Generate manifest with UNIQUE arcs
        manifest_path = generate_arc_manifest(vod_id, unique_arcs, chapters)

    # Final summary
    if not dry_run:
        print(f"\n{'=' * 60}")
        print("FINAL SUMMARY")
        print(f"{'=' * 60}")

        # Recalculate based on unique arcs
        total_arcs = len(unique_arcs) if 'unique_arcs' in locals() else sum(len(r.arcs) for r in results)
        total_filler = sum(len(r.filler_segments) for r in results)

        print(f"Total arcs detected: {total_arcs}")
        print(f"Total filler segments: {total_filler}")

        arc_types: Dict[str, int] = {}
        # Use unique arcs if available, otherwise raw results
        arcs_to_count = unique_arcs if 'unique_arcs' in locals() else [a for r in results for a in r.arcs]
        
        for arc in arcs_to_count:
            arc_types[arc.arc_type] = arc_types.get(arc.arc_type, 0) + 1

        print("\nArc types:")
        for arc_type, count in arc_types.items():
            print(f"  {arc_type}: {count}")

    return results, manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Gemini-based arc detection for VOD transcripts"
    )
    parser.add_argument("vod_id", help="VOD ID to process")
    parser.add_argument(
        "--chunk", type=int, help="Process only a specific chunk (0-indexed, 30m each)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be sent without calling API",
    )
    parser.add_argument(
        "--model",
        default="gemini-3-flash-preview",
        help="Gemini model to use (default: gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save results to JSON files"
    )

    args = parser.parse_args()

    if args.chunk is not None:
        # Process single chunk
        agent = ArcDetectionAgent(args.vod_id, model=args.model)
        agent.process_chunk(args.chunk, dry_run=args.dry_run)
    else:
        run_arc_detection(
            vod_id=args.vod_id,
            model=args.model,
            save=args.save,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
