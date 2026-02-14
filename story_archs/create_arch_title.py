#!/usr/bin/env python3
"""
Create arc titles and write them into arc manifests.

Simplified approach:
  - Load arc transcript from filtered_ai_data.json
  - Send full transcript to LLM
  - Generate single title (no candidates, no reranking)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


def _format_hms(sec: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    s = int(max(0, round(float(sec))))
    h = s // 3600
    m = (s % 3600) // 60
    s2 = s % 60
    return f"{h:02d}:{m:02d}:{s2:02d}"


def _read_stream_context(vod_id: str) -> Dict[str, str]:
    """Load stream context (streamer, title, categories)."""
    try:
        from src.config import config
        sc_path = config.get_ai_data_dir(vod_id) / f"{vod_id}_stream_context.json"
        if sc_path.exists():
            data = json.loads(sc_path.read_text(encoding="utf-8"))
            cats = data.get("chapter_categories")
            categories = ", ".join([str(c).strip() for c in cats if str(c).strip()]) if isinstance(cats, list) else ""
            return {
                "streamer": str(data.get("streamer") or data.get("streamer_name") or ""),
                "vod_title": str(data.get("vod_title") or data.get("title") or ""),
                "categories": categories,
            }
    except Exception:
        pass
    return {"streamer": "", "vod_title": "", "categories": ""}


def _load_arcs(vod_id: str, only_arc: Optional[int]) -> List[Path]:
    """Find arc manifest files."""
    root = Path(f"data/vector_stores/{vod_id}/arcs")
    if only_arc is not None:
        p = root / f"arc_{int(only_arc):03d}_manifest.json"
        return [p] if p.exists() else []
    return sorted([p for p in root.glob("arc_*_manifest.json") if p.is_file()])


def _load_arc_transcript(vod_id: str, start_time: float, end_time: float) -> List[Dict]:
    """Load transcript segments from filtered_ai_data.json for the arc's time range."""
    try:
        from src.config import config
        ai_data_path = config.get_ai_data_dir(vod_id) / f"{vod_id}_filtered_ai_data.json"
        
        if not ai_data_path.exists():
            print(f"  ⚠️  filtered_ai_data.json not found: {ai_data_path}")
            return []
        
        with open(ai_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract segments from the data structure
        segments = data.get('segments', []) if isinstance(data, dict) else data
        
        # Filter segments within arc time range
        arc_segments = []
        for seg in segments:
            seg_start = float(seg.get("start_time", 0))
            seg_end = float(seg.get("end_time", 0))
            
            # Include segment if it overlaps with arc
            if seg_start < end_time and seg_end > start_time:
                arc_segments.append({
                    "start_time": seg_start,
                    "end_time": seg_end,
                    "transcript": str(seg.get("transcript", "")).strip(),
                })
        
        return arc_segments
        
    except Exception as e:
        print(f"  ⚠️  Failed to load transcript: {e}")
        import traceback
        traceback.print_exc()
        return []


def _build_arc_context(
    vod_id: str,
    arc_manifest: Dict,
    transcript_segments: List[Dict],
    streamer_context: Dict,
) -> Dict:
    """Build context dict for LLM prompt."""
    start_time = float(arc_manifest.get("start_abs", 0))
    end_time = float(arc_manifest.get("end_abs", 0))
    duration = end_time - start_time
    
    # Format streamer context
    context_lines = []
    if streamer_context.get("streamer"):
        context_lines.append(f"Streamer: {streamer_context['streamer']}")
    if streamer_context.get("vod_title"):
        context_lines.append(f"Original VOD Title: {streamer_context['vod_title']}")
    if streamer_context.get("categories"):
        context_lines.append(f"Categories: {streamer_context['categories']}")
    
    streamer_info = "\n".join(context_lines) if context_lines else f"VOD ID: {vod_id}"
    
    return {
        "vod_id": vod_id,
        "streamer_context": streamer_info,
        "streamer_name": streamer_context.get("streamer", vod_id),
        "transcript_segments": transcript_segments,
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration,
        "start_hms": _format_hms(start_time),
        "end_hms": _format_hms(end_time),
        "duration_hms": _format_hms(duration),
    }


def _build_title_prompt(context: Dict, previous_titles: List[str]) -> str:
    """Build LLM prompt for title generation.
    
    PLACEHOLDER - User will paste custom prompt here.
    
    Available context variables:
    - context['streamer_context'] - Streamer name, VOD title, categories (formatted string)
    - context['streamer_name'] - Just the streamer name
    - context['duration_hms'] - Arc duration in HH:MM:SS format
    - context['start_hms'] / context['end_hms'] - Arc start/end times
    - context['transcript_segments'] - List of transcript segments with timestamps
    - previous_titles - List of titles already generated for previous arcs in this VOD
    """
    # Format transcript for prompt
    transcript_text = "\n".join([
        f"[{seg['start_time']:.0f}s] {seg['transcript']}"
        for seg in context['transcript_segments']
    ])
    
    # Limit transcript length to avoid token overflow
    max_chars = 50000
    if len(transcript_text) > max_chars:
        transcript_text = transcript_text[:max_chars] + "\n... (truncated)"
    
    streamer_name = context['streamer_name']
    
    # Format previous titles section
    previous_titles_text = ""
    if previous_titles:
        previous_titles_text = f"""
        PREVIOUS TITLES IN THIS VOD (avoid repeating these styles):
        {chr(10).join(f"- {title}" for title in previous_titles[-5:])}  
        
        ⚠️ IMPORTANT: Switch up your style! Don't repeat the same pattern/structure as above.
        """
    
    # PLACEHOLDER PROMPT - Replace with your custom prompt
    prompt = f"""
        You are a brattish viral clip editor. Generate a attention grabbing title that sounds like it came from a real human editor, not AI.
        Ground truth stream context: {context['streamer_context']}.
        Window: {int(context['start_time'])}s to {int(context['end_time'])}s (duration ~{int(context['duration'])}s)

        Transcript (with timestamps and context):
        {transcript_text}

        TASK:
        Find the specific thing that happened in this clip. Generate a attention grabbing title that doesn't sound AI or a blog post.
        
        CONTEXT ANALYSIS:
        - Look at the transcript to understand what actually happened
        - Understand the full situation from the broader context, not just the clip itself
        
        TITLE RULES:
        - Maximum 40 characters (including spaces)
        - Describe the specific action or event that happened
        - Use simple, direct language
        - Sound like a real person wrote it
        - Transcript names are unreliable, relationships between speakers are vague based on the transcript
        - If possible, the only person you may name is the streamer: {streamer_name} and not other characters
        - Try not to infer relationships between characters, forexample, you see streamer June, but is it a dog or a person? Is the dog named Hjune, Jun, Joun?
        - These are the previous titles in this VOD: {previous_titles_text} VARY your title style to keep things fresh and interesting

        CRITICAL OUTPUT FORMAT:
        - Return ONLY the title text itself
        - Maximum 40 characters (including spaces)
    """  # noqa: F821
    
    return prompt


def _generate_arc_title(
    vod_id: str,
    arc_manifest: Dict,
    streamer_context: Dict,
    previous_titles: List[str],
) -> Optional[str]:
    """Generate title for an arc using transcript.
    
    Args:
        vod_id: VOD identifier
        arc_manifest: Arc manifest data
        streamer_context: Streamer context (name, title, categories)
        previous_titles: List of titles already generated for previous arcs in this VOD
    """
    try:
        from src.ai_client import call_llm_ollama
        
        # Get arc time range
        start_time = float(arc_manifest.get("start_abs", 0))
        end_time = float(arc_manifest.get("end_abs", 0))
        duration_minutes = (end_time - start_time) / 60
        
        print(f"    Loading transcript ({start_time:.0f}s - {end_time:.0f}s, ~{duration_minutes:.1f} min)...")
        
        # Load transcript
        transcript_segments = _load_arc_transcript(vod_id, start_time, end_time)
        
        if not transcript_segments:
            print("    ⚠️  No transcript segments found")
            return None
        
        print(f"    Loaded {len(transcript_segments)} transcript segments")
        
        # Build context
        context = _build_arc_context(vod_id, arc_manifest, transcript_segments, streamer_context)
        
        # Build prompt with previous titles for variety
        if previous_titles:
            print(f"    Using {len(previous_titles)} previous titles for style variety")
        prompt = _build_title_prompt(context, previous_titles)
        
        # Generate title with Ollama
        print("    Generating title with Ollama...")
        resp = call_llm_ollama(
            prompt,
            max_tokens=60,  # Reduced for title-only output
            temperature=0.7,
            request_tag="arc_title_gen"
        )
        
        if not resp:
            return None
        
        # Debug: show raw response
        print(f"    Raw response: {resp[:100]}...")
        
        # Strict format enforcement
        title = _enforce_title_format(resp)
        
        if not title:
            print("    ⚠️  Format enforcement rejected response")
            return None
        
        return title
        
    except Exception as e:
        print(f"  ⚠️  Title generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _enforce_title_format(raw_response: str) -> Optional[str]:
    """Enforce strict title format rules and clean up LLM output."""
    import re
    
    # Start with basic cleanup
    title = raw_response.strip()
    
    # Remove quotes of any kind
    title = title.strip('"').strip("'").strip('"').strip('"').strip('`')
    
    # Take only the first line (ignore any explanations after)
    title = title.split('\n')[0].strip()
    
    # Remove common LLM artifacts and parentheticals
    artifacts = [
        r'\s*\([^)]*(?:generated|based|analysis|followed|note|i\'ve)[^)]*\)\s*$',
        r'\s*\[[^\]]*(?:generated|based|analysis|followed|note|i\'ve)[^\]]*\]\s*$',
        r'^\s*(?:title|generated title|arc title|clip title):\s*',
        r'^\s*[-•*]\s*',  # Remove bullet points
    ]
    
    for pattern in artifacts:
        title = re.sub(pattern, '', title, flags=re.IGNORECASE).strip()
    
    # Remove trailing punctuation (except question marks which might be intentional)
    title = re.sub(r'[.,;:!]+$', '', title).strip()
    
    # Enforce maximum length (40 characters as per prompt)
    if len(title) > 45:  # Allow small buffer
        # Try to cut at a word boundary
        title = title[:42]
        last_space = title.rfind(' ')
        if last_space > 30:  # Don't cut too early
            title = title[:last_space]
        title = title.rstrip('.,;:!') + '...'
    
    # Final validation
    if not title or len(title) < 3:
        return None
    
    # Check for obvious AI artifacts that shouldn't be in titles
    bad_patterns = [
        r'as an ai', r'i cannot', r'i can\'t', r'sorry', 
        r'unable to', r'i\'m ', r'here is', r'here\'s'
    ]
    for pattern in bad_patterns:
        if re.search(pattern, title, re.IGNORECASE):
            return None
    
    return title


def generate_titles(vod_id: str, only_arc: Optional[int] = None, force: bool = False) -> int:
    """Generate titles for arcs using full transcript.
    
    Args:
        vod_id: VOD identifier
        only_arc: If set, only process this arc number
        force: If True, regenerate titles even if they already exist
    
    Returns:
        Number of titles generated/updated
    """
    paths = _load_arcs(vod_id, only_arc)
    if not paths:
        print(f"No arc manifests found for VOD {vod_id}")
        return 0
    
    # Get stream context once
    streamer_context = _read_stream_context(vod_id)
    
    # Track previously generated titles in this VOD for style variety
    previous_titles: List[str] = []
    
    wrote = 0
    for mp in paths:
        try:
            man = json.loads(mp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  ⚠️  Failed to read {mp}: {e}")
            continue
        
        arc_idx = int(man.get("arc_index") or 0)
        
        # If arc already has a title, add it to previous_titles for context
        existing_title = str(man.get("title") or "").strip()
        
        # Skip if title already exists and not forcing
        if not force and existing_title:
            # Still track it for variety in subsequent arcs
            previous_titles.append(existing_title)
            continue
        
        print(f"\nArc {arc_idx:03d}:")
        
        # Generate title with awareness of previous titles
        title = _generate_arc_title(vod_id, man, streamer_context, previous_titles)
        
        if not title:
            print(f"  ⚠️  Failed to generate title for arc {arc_idx:03d}")
            continue
        
        # Write back to manifest
        man["title"] = title
        try:
            mp.write_text(json.dumps(man, indent=2), encoding="utf-8")
            wrote += 1
            print(f"  ✓ Generated: {title}")
            
            # Add to previous titles for next arcs to avoid repetition
            previous_titles.append(title)
        except Exception as e:
            print(f"  ⚠️  Failed to write manifest: {e}")
            continue
    
    return wrote


def main() -> None:
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate arc titles from transcript and write into manifests"
    )
    parser.add_argument("vod_id", help="VOD identifier")
    parser.add_argument("--arc", type=int, default=None, help="Process only this arc number")
    parser.add_argument("--force", action="store_true", help="Regenerate existing titles")
    args = parser.parse_args()
    
    print(f"Generating titles for VOD {args.vod_id}...")
    n = generate_titles(args.vod_id, only_arc=args.arc, force=bool(args.force))
    print(f"\n📝 Total titles generated/updated: {n}")


if __name__ == "__main__":
    main()
