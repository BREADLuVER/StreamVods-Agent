#!/usr/bin/env python3
"""
Enhanced YouTube metadata generation for Arc videos with Gemini 3 Flash.

This script generates YouTube metadata for arc videos and includes AI-generated timestamps
in the description using Gemini 3 Flash via the centralized TitleService.

Usage:
  python processing-scripts/generate_arch_youtube_metadata_enhanced.py <vod_id> [--arc 1]
"""

from __future__ import annotations

import json
import os
import re
import sys
import requests
from pathlib import Path
from typing import Dict, List, Optional
import importlib.util

# Ensure project imports resolve BEFORE any 'src.' imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# AI clients - use centralized service
from src.ai_client import call_llm, _call_ollama_json

# Import centralized title service
try:
    from src.title_service import TitleService
    USE_TITLE_SERVICE = True
except ImportError:
    USE_TITLE_SERVICE = False
    TitleService = None


def _format_hms(sec: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    s = int(max(0, round(float(sec))))
    h = s // 3600
    m = (s % 3600) // 60
    s2 = s % 60
    return f"{h:02d}:{m:02d}:{s2:02d}"


def _format_timestamp(sec: float) -> str:
    """Convert seconds to M:SS or H:MM:SS format for timestamps."""
    s = int(max(0, round(float(sec))))
    h = s // 3600
    m = (s % 3600) // 60
    s2 = s % 60
    
    if h > 0:
        return f"{h}:{m:02d}:{s2:02d}"
    else:
        return f"{m}:{s2:02d}"


def _load_streamer_info(vod_id: str) -> Dict[str, str]:
    """Load streamer information from stream context."""
    try:
        from src.config import config
        sc_path = config.get_ai_data_dir(vod_id) / f"{vod_id}_stream_context.json"
        if sc_path.exists():
            data = json.loads(sc_path.read_text(encoding="utf-8"))
            return {
                "streamer": str(data.get("streamer") or data.get("streamer_name") or ""),
                "vod_title": str(data.get("vod_title") or data.get("title") or ""),
            }
    except Exception:
        pass
    return {"streamer": "", "vod_title": ""}


def _load_arc_manifests(vod_id: str, arc_index: Optional[int]) -> List[Path]:
    """Load arc manifest files."""
    arcs_dir = Path(f"data/vector_stores/{vod_id}/arcs")
    if arc_index is not None:
        p = arcs_dir / f"arc_{int(arc_index):03d}_manifest.json"
        return [p] if p.exists() else []
    items = sorted([p for p in arcs_dir.glob("arc_*_manifest.json") if p.is_file()])
    return items


def _load_enhanced_director_cut_manifest(vod_id: str) -> Optional[Dict]:
    """Load the enhanced director cut manifest for timestamp generation."""
    try:
        manifest_path = Path(f"data/vector_stores/{vod_id}/enhanced_director_cut_manifest.json")
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"⚠️  Failed to load enhanced director cut manifest: {e}")
    return None


def _call_ollama(prompt: str, model: str = "llama3.1:8b") -> Optional[str]:
    """Call Ollama API to generate content."""
    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1000
            }
        }
        
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        print(f"Ollama API call failed: {e}")
        return None


def _collect_and_smooth_arc_ranges(enhanced_manifest: Dict, arc_start_abs: int, arc_end_abs: int) -> List[Dict]:
    """Collect ranges that overlap the arc, convert to arc-relative, then smooth overlaps.
    
    Filters to significant ranges only and merges adjacent ranges to avoid timestamp spam.
    Target: 8-15 meaningful chapter markers for typical arc.

    Returns list of dicts with keys: start (int seconds, relative), end (int seconds, relative), summary (str), topic_key, energy.
    """
    raw: List[Dict] = []
    for range_data in enhanced_manifest.get("ranges", []):
        try:
            range_start = int(float(range_data.get("start", 0)))
            range_end = int(float(range_data.get("end", 0)))
        except Exception:
            continue
        if range_start < arc_end_abs and range_end > arc_start_abs:
            # Clamp to arc window and convert to relative seconds
            adj_start = max(0, range_start - arc_start_abs)
            adj_end = max(0, min(arc_end_abs, range_end) - arc_start_abs)
            duration = adj_end - adj_start
            
            # Filter out very short ranges (less than 30 seconds)
            if duration < 30:
                continue
                
            raw.append({
                "start": adj_start,
                "end": adj_end,
                "duration": duration,
                "summary": str(range_data.get("summary") or "").strip(),
                "topic_key": str(range_data.get("topic_key") or "").strip(),
                "energy": str(range_data.get("energy") or "").strip(),
            })

    if not raw:
        return []

    # Sort by start
    raw.sort(key=lambda r: r["start"])

    # Merge adjacent ranges that are very close (within 60 seconds)
    # and have similar topics to reduce timestamp clutter
    # BUT: Be less aggressive for short arcs to ensure minimum timestamp count
    target_max = 15
    
    merged: List[Dict] = []
    current = raw[0]
    
    for r in raw[1:]:
        gap = r["start"] - current["end"]
        same_topic = r["topic_key"] == current["topic_key"]
        
        # Adaptive merging: only merge if we have way too many ranges
        # For short arcs, be very conservative with merging
        ranges_so_far = len(merged) + 1
        remaining_ranges = len(raw) - raw.index(r)
        estimated_total = ranges_so_far + remaining_ranges
        
        # Only merge if we'd end up with too many timestamps
        should_merge = False
        if estimated_total > target_max:
            # Merge if gap is very small and same topic
            should_merge = (gap < 45 and same_topic) or (gap < 20)
        
        if should_merge:
            # Extend current range
            current["end"] = r["end"]
            current["duration"] = current["end"] - current["start"]
            # Combine summaries
            if r["summary"] and r["summary"] not in current["summary"]:
                current["summary"] = f"{current['summary']}; {r['summary']}"
        else:
            merged.append(current)
            current = r
    
    # Don't forget the last one
    merged.append(current)

    # If still too many ranges, keep only the most significant ones
    # Prefer longer durations and higher energy
    if len(merged) > 20:
        # Score by duration and energy
        for r in merged:
            energy_score = 1.5 if r.get("energy") == "high" else (1.0 if r.get("energy") == "medium" else 0.5)
            r["score"] = r["duration"] * energy_score
        
        # Sort by score descending, keep top 15
        merged.sort(key=lambda r: r.get("score", 0), reverse=True)
        merged = merged[:15]
        
        # Re-sort by start time
        merged.sort(key=lambda r: r["start"])

    # Final smoothing: enforce non-decreasing starts
    smoothed: List[Dict] = []
    last_end = 0
    for r in merged:
        start = max(r["start"], last_end)
        end = max(start + 30, r["end"])  # ensure minimum duration 30s
        smoothed.append({
            "start": start,
            "end": end,
            "summary": r["summary"],
            "topic_key": r["topic_key"],
            "energy": r["energy"],
        })
        last_end = end
    
    return smoothed


def _generate_labels_with_ollama(arc_ranges: List[Dict]) -> Optional[List[str]]:
    """
    Ask Ollama for labels with strict JSON schema (exact count).
    """
    if not arc_ranges:
        return []

    # Build a small, deterministic prompt with clear examples
    items = [
        f"- {r['summary']} (topic: {r['topic_key']}, energy: {r['energy']})"
        for r in arc_ranges[:50]
    ]
    
    # Much more explicit prompt with examples - STRICT about weak verbs
    prompt = (
        "You create YouTube video chapter titles for stream highlights.\n\n"
        "CRITICAL RULES - You MUST follow ALL of these:\n"
        "1. Each label is 4-8 words maximum\n"
        "2. Use title case (capitalize first letter of each major word)\n"
        "3. NO periods, commas, or any punctuation at the end\n"
        "4. Be specific and descriptive, not vague\n"
        "5. Use STRONG action verbs: Explaining, Building, Fighting, Defeating, Planning, Strategizing\n"
        "6. BANNED WEAK VERBS - NEVER USE: Discussing, Talking, Sharing, Enjoying, Mentioning, Describing, Reflecting\n"
        "7. NO pronouns (he, she, they, user, streamer, player)\n"
        "8. NO generic phrases like 'Something Interesting'\n\n"
        "GOOD EXAMPLES:\n"
        "- Explaining Career Choice to Family\n"
        "- Building Minecraft Castle in Snow\n"
        "- Reacting to Banff Travel Recommendations\n"
        "- Fighting Cerberus Boss Battle\n"
        "- Planning Korean Food Restaurant Visit\n"
        "- Strategizing Healer Positioning for Raid\n\n"
        "BAD EXAMPLES (NEVER DO THIS):\n"
        "- Discussing career choices (uses 'discussing')\n"
        "- Talking about skincare routine (uses 'talking about')\n"
        "- Sharing personal anxiety (uses 'sharing')\n"
        "- Enjoying New York food (uses 'enjoying')\n"
        "- Players notice something interesting (has 'players', too vague)\n"
        "- Reflecting on gaming skills (uses 'reflecting')\n\n"
        "BETTER VERSIONS:\n"
        "- Explaining Career Choice to Parents\n"
        "- Showing Skincare Routine Steps\n"
        "- Revealing Streaming Anxiety Struggles\n"
        "- Rating Best New York Restaurants\n"
        "- Discovering Hidden Game Mechanic\n"
        "- Analyzing Personal Gaming Performance\n\n"
        "For each bullet below, write ONE short chapter title using STRONG verbs only.\n"
        "Return ONLY valid JSON matching the schema, no extra text.\n\n"
        "Segments to label:\n" + "\n".join(items)
    )

    count = len(items)
    schema = {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": { "type": "string", "minLength": 10, "maxLength": 100 },
                "minItems": count,
                "maxItems": count
            }
        },
        "required": ["labels"],
        "additionalProperties": False
    }

    obj = _call_ollama_json(
        prompt=prompt,
        schema=schema,
        model=os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b-instruct"),
        num_ctx=int(os.getenv("LOCAL_NUM_CTX", "16384")),
        num_predict=512,  # Increased from 256 to allow more output
        temperature=0.3,  # Lower temperature for more consistency
        top_p=0.9,
        top_k=40,  # Reduced for more focused outputs
        repeat_penalty=1.2  # Increased to avoid repetition
    )
    if not obj or not isinstance(obj.get("labels"), list):
        return None

    # Clean up labels - remove any trailing punctuation and fix weak verbs
    cleaned = []
    
    # Mapping of weak verbs to stronger alternatives
    weak_verb_fixes = {
        r'^Discussing\s+': 'Explaining ',
        r'^Talking About\s+': 'Explaining ',
        r'^Talks About\s+': 'Explaining ',
        r'^Sharing\s+': 'Revealing ',
        r'^Enjoying\s+': 'Experiencing ',
        r'^Mentioning\s+': 'Highlighting ',
        r'^Describing\s+': 'Explaining ',
        r'^Reflecting On\s+': 'Analyzing ',
        r'^Reflecting\s+': 'Analyzing ',
        r'\bPlayer\s+': '',
        r'\bPlayers\s+': '',
        r'\bStreamer\s+': '',
        r'\bUser\s+': '',
        r'\bSomething Interesting\b': 'Hidden Details',
        r'\bSomething\s+': '',
    }
    
    for s in obj["labels"]:
        t = str(s or "").strip().strip('"').replace("\n", " ")
        # Remove trailing punctuation
        t = re.sub(r'[.,;:!?]+$', '', t).strip()
        
        # Fix weak verbs with post-processing
        for pattern, replacement in weak_verb_fixes.items():
            t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)
        
        # Clean up double spaces and re-capitalize
        t = re.sub(r'\s+', ' ', t).strip()
        
        # Re-apply title case after fixes
        if t:
            words = t.split()
            # Capitalize first letter of each word except small words (unless first word)
            small_words = {'a', 'an', 'and', 'at', 'but', 'by', 'for', 'in', 'of', 'on', 'or', 'the', 'to', 'with'}
            title_words = []
            for i, word in enumerate(words):
                if i == 0 or word.lower() not in small_words:
                    title_words.append(word.capitalize())
                else:
                    title_words.append(word.lower())
            t = ' '.join(title_words)
            cleaned.append(t)
    
    return cleaned or None

def _generate_timestamps_with_ollama(vod_id: str, arc_manifest: Dict) -> Optional[str]:
    """Generate timestamp lines using Gemini 3 Flash via centralized TitleService.

    The returned string contains lines of the form: HH:MM:SS label
    """
    arc_start_abs = int(arc_manifest.get("start_abs", 0))
    arc_end_abs = int(arc_manifest.get("end_abs", 0))
    if arc_end_abs <= arc_start_abs:
        return None

    # Try enhanced manifest first
    enhanced_manifest = _load_enhanced_director_cut_manifest(vod_id)
    arc_ranges = None
    if enhanced_manifest:
        # Collect and smooth from enhanced manifest
        arc_ranges = _collect_and_smooth_arc_ranges(enhanced_manifest, arc_start_abs, arc_end_abs)
    
    # Fallback: Use arc manifest segments (intro/climax/resolution) if enhanced manifest not available or empty
    if not arc_ranges:
        print("No enhanced director cut manifest found, using arc manifest segments")
        arc_ranges = []
        
        # Build ranges from intro, climax, resolution segments
        for segment_name in ["intro", "climax", "resolution"]:
            segment = arc_manifest.get(segment_name)
            if segment and isinstance(segment, dict):
                seg_start = int(segment.get("start", 0))
                seg_end = int(segment.get("end", 0))
                if seg_start >= arc_start_abs and seg_end <= arc_end_abs and seg_end > seg_start:
                    # Convert to relative time
                    rel_start = seg_start - arc_start_abs
                    rel_end = seg_end - arc_start_abs
                    duration = rel_end - rel_start
                    if duration >= 30:  # Only include segments >= 30 seconds
                        arc_ranges.append({
                            "start": rel_start,
                            "start_abs": seg_start,
                            "end": rel_end,
                            "end_abs": seg_end,
                            "duration": duration,
                            "summary": arc_manifest.get("summary", ""),
                            "topic_key": segment_name,
                            "energy": "",
                        })
        
        # If no segments found, use the main arc range
        if not arc_ranges:
            duration = arc_end_abs - arc_start_abs
            if duration >= 30:
                arc_ranges.append({
                    "start": 0,
                    "start_abs": arc_start_abs,
                    "end": duration,
                    "end_abs": arc_end_abs,
                    "duration": duration,
                    "summary": arc_manifest.get("summary", ""),
                    "topic_key": arc_manifest.get("chapter", ""),
                    "energy": "",
                })
        
        if not arc_ranges:
            print("No valid ranges found for arc, skipping timestamp generation")
            return None

    # Try centralized TitleService first (Gemini 3 Flash)
    if USE_TITLE_SERVICE and TitleService is not None:
        try:
            print("🎯 Generating timestamps with Gemini 3 Flash...")
            service = TitleService(vod_id)
            timestamp_lines = service.generate_timestamps(arc_ranges, arc_start_abs)
            if timestamp_lines:
                print(f"✨ Generated {len(timestamp_lines)} timestamps")
                return "\n".join(timestamp_lines)
        except Exception as e:
            print(f"⚠️ TitleService timestamps failed, using legacy: {e}")

    # Legacy fallback: use Ollama
    labels = _generate_labels_with_ollama(arc_ranges)
    if not labels:
        # Fallback: condense summaries
        labels = [ _condense(r.get("summary", ""), 60) or "Highlights" for r in arc_ranges ]

    # Pair our computed starts with labels; keep counts aligned
    count = min(len(arc_ranges), len(labels))
    lines: List[str] = []
    for i in range(count):
        start_rel = int(arc_ranges[i]["start"])
        stamp = _format_hms(start_rel)
        label = str(labels[i]).strip()
        lines.append(f"{stamp} {label}")
    return "\n".join(lines) if lines else None


def _first_nonempty_summary(ranges: List[Dict]) -> str:
    """Get the first non-empty summary from ranges."""
    for r in ranges:
        s = str(r.get("summary") or "").strip()
        if s:
            return s
    return ""


def _condense(text: str, max_chars: int = 60) -> str:
    """Condense text to max_chars while preserving word boundaries."""
    if not text:
        return ""
    text = text.replace("→", ", ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.strip()


def _get_game_tags(vod_id: str) -> List[str]:
    """Get game tags for the VOD."""
    try:
        from src.config import config as _cfg_sc
        sc_path = _cfg_sc.get_ai_data_dir(vod_id) / f"{vod_id}_stream_context.json"
    except Exception:
        sc_path = None

    try:
        if sc_path and sc_path.exists():
            script_dir = Path(__file__).parent
            mod_path = script_dir / "generate_youtube_metadata.py"
            if mod_path.exists():
                spec = importlib.util.spec_from_file_location("generate_youtube_metadata", str(mod_path))
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    func = getattr(module, "get_game_tags_from_chapters", None)
                    if callable(func):
                        tags = func(vod_id)
                        if isinstance(tags, list):
                            return tags
    except Exception:
        pass
    
    # Lightweight fallback
    try:
        from src.config import config
        chapters_file = config.get_ai_data_dir(vod_id) / f"{vod_id}_chapters.json"
        tags: List[str] = []
        if chapters_file.exists():
            data = json.loads(chapters_file.read_text(encoding="utf-8"))
            chapters = data.get("chapters") if isinstance(data, dict) else data
            if isinstance(chapters, list):
                for ch in chapters:
                    if isinstance(ch, dict):
                        cat = str(ch.get("category") or "").strip()
                        if cat and cat not in tags:
                            tags.append(cat)
        tags.extend(_get_vod_info_game_tags(vod_id))
        seen = set()
        uniq: List[str] = []
        for t in tags:
            if t and t not in seen:
                seen.add(t)
                uniq.append(t)
        return uniq
    except Exception:
        pass
    return []


def _load_vod_info(vod_id: str) -> Dict[str, str]:
    """Load VOD info."""
    try:
        from src.config import config
        path = config.get_ai_data_dir(vod_id) / f"{vod_id}_vod_info.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _extract_streamer_from_vod_info(vod_info: Dict[str, str]) -> str:
    """Extract streamer name from VOD info."""
    candidates = [
        vod_info.get("Channel"),
        vod_info.get("User"),
        vod_info.get("Streamer"),
        vod_info.get("Broadcaster"),
        vod_info.get("user_name"),
        vod_info.get("display_name"),
        vod_info.get("user_login"),
        vod_info.get("login"),
    ]
    for v in candidates:
        s = str(v or "").strip()
        if s:
            return s
    return ""


def _get_vod_info_game_tags(vod_id: str) -> List[str]:
    """Get game tags from VOD info."""
    vod_info = _load_vod_info(vod_id)
    if not vod_info:
        return []
    for key in ("Game", "Game Name", "game_name", "Category"):
        val = str(vod_info.get(key) or "").strip()
        if val:
            return [val]
    return []


def _read_stream_context(vod_id: str) -> Dict[str, str]:
    """Read stream context for richer title prompts."""
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


def _load_ai_segments(vod_id: str) -> List[Dict]:
    """Load AI transcript segments."""
    try:
        from src.config import config
        base = config.get_ai_data_dir(vod_id)
        for name in [f"{vod_id}_filtered_ai_data.json", f"{vod_id}_ai_data.json"]:
            p = base / name
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                return data.get("segments", data) if isinstance(data, dict) else data
    except Exception:
        pass
    return []


def _build_arc_transcript_text(vod_id: str, man: Dict) -> str:
    """Build transcript text for the arc."""
    start_abs = int(man.get("start_abs") or 0)
    end_abs = int(man.get("end_abs") or 0)
    if end_abs <= start_abs:
        return ""
    segments = _load_ai_segments(vod_id)
    parts: List[str] = []
    for seg in segments:
        try:
            st = int(seg.get("start_time", 0))
            et = int(seg.get("end_time", 0))
            if st > end_abs or et < start_abs:
                continue
            text = str(seg.get("transcript") or "").strip()
            if text:
                parts.append(text)
        except Exception:
            continue
    joined = " ".join(parts)
    return joined[:4000] if len(joined) > 4000 else joined


def _build_arc_title_prompt(vod_id: str, arc_idx: int, man: Dict) -> str:
    """Build prompt for generating arc title."""
    sc = _read_stream_context(vod_id)
    streamer = sc.get("streamer", "").strip()
    vod_title = sc.get("vod_title", "").strip()
    categories = sc.get("categories", "").strip()
    if not streamer:
        streamer = _extract_streamer_from_vod_info(_load_vod_info(vod_id)).strip()

    start_hms = str(man.get("start_hms") or _format_hms(man.get("start_abs", 0)))
    end_hms = str(man.get("end_hms") or _format_hms(man.get("end_abs", 0)))
    ranges = man.get("ranges") or []

    lines: List[str] = []
    for r in ranges:
        try:
            s = str(r.get("summary") or "").strip()
            if not s:
                continue
            rs = _format_hms(r.get("start_abs") or r.get("start") or 0)
            re_ = _format_hms(r.get("end_abs") or r.get("end") or 0)
            lines.append(f"- {rs} → {re_}: {s}")
        except Exception:
            continue
    arc_summaries = "\n".join(lines) if lines else ""

    sc_lines: List[str] = []
    if streamer:
        sc_lines.append(f"Streamer: {streamer}")
    if vod_title:
        sc_lines.append(f"Original VOD Title: {vod_title}")
    if categories:
        sc_lines.append(f"Categories: {categories}")
    sc_block = ("\n".join(sc_lines) + "\n") if sc_lines else ""

    arc_transcript = _build_arc_transcript_text(vod_id, man)

    title_rules = (
        "\n\nTITLE RULES — enforce all:\n"
        "- Be as specific as possible, what is the content of the clip? Who is involved? What is the main event? What is the outcome? What title would make it viral?\n"
        "- Maximum 40 characters (including spaces)\n"
        "- No emojis or special characters\n"
        "- No quotes or punctuation marks\n"
        "- Do not use the words 'streamer'\n"
        "- Use simple, clear language; focus on the main event; be a little clickbaitish\n"
        "- Be as specific as possible\n"
        "- Name the streamer at the start of the title if possible\n"
        "Bad examples: INSANE GOAL!!!; EPIC MOMENT!!!; Streamer reacts; Clutch play to save a teammate\n"
    )

    parts: List[str] = [
        "You are an expert clip editor that has an eye for viral clips. You are strict and creative, knowing what audiences want and how to manipulate them with titles and hooks.\n",
        f"VOD: {vod_id}\n",
        f"{sc_block}",
        f"Arc window: {start_hms} → {end_hms}\n\n",
        "Arc summaries (ordered):\n",
        f"{arc_summaries}\n",
    ]
    if arc_transcript:
        parts.append("\nArc transcript (no timestamps):\n" + arc_transcript + "\n")
    parts.extend([
        f"{title_rules}\n\n",
        "Return STRICT JSON only with this schema (no prose, no code fences):\n",
        "{\n  \"title\": \"string\"\n}",
    ])
    prompt = "".join(parts)
    return prompt


def _extract_title_from_response(text: str) -> Optional[str]:
    """Extract title from LLM response."""
    try:
        s = text.strip()
        if s.startswith("```"):
            s = s.split("```", 2)[1] if "```" in s else s
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(s[start : end + 1])
            title = obj.get("title")
            if isinstance(title, str) and title.strip():
                return title.strip()
    except Exception:
        pass
    return None


def _normalize_timestamp_lines(timestamps: str) -> List[str]:
    """Normalize AI-produced timestamp lines to strict HH:MM:SS label format.

    Accepts inputs like ":21 label", "0:21 label", "5:37 label", "1:02:03 label".
    Returns lines like "00:00:21 label", "00:05:37 label", "01:02:03 label".
    """
    norm: List[str] = []
    for raw in (timestamps or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        # Extract first whitespace-separated token as time
        parts = line.split(maxsplit=1)
        time_token = parts[0]
        label = parts[1] if len(parts) > 1 else ""

        # Fix leading colon formats like ":21"
        if time_token.startswith(":"):
            time_token = "0" + time_token

        # Parse time token
        hh = 0
        mm = 0
        ss = 0
        token_parts = time_token.split(":")
        try:
            if len(token_parts) == 3:
                hh = int(token_parts[0])
                mm = int(token_parts[1])
                ss = int(token_parts[2])
            elif len(token_parts) == 2:
                # Interpret as M:SS
                mm = int(token_parts[0])
                ss = int(token_parts[1])
            else:
                # Unsupported format; skip
                continue
        except Exception:
            continue

        total = hh * 3600 + mm * 60 + ss
        stamp = _format_hms(total)
        # Remove any trailing range/end-time fragments from label (e.g., "→ 13:21:" or "-> 5:10:")
        label = re.sub(r"^\s*(→|->|-)\s*\d{1,2}(?::\d{2}){1,2}\s*:?\s*", "", label or "").strip()
        label_clean = re.sub(r"\s+", " ", label)
        if label_clean:
            norm.append(f"{stamp} {label_clean}")
        else:
            norm.append(stamp)
    return norm

def _llm_generate_arc_title(vod_id: str, arc_idx: int, man: Dict) -> Optional[str]:
    """Generate arc title using Gemini 3 Flash via centralized TitleService."""
    
    # Get arc time range for loading transcript from ai_data
    start_abs = int(man.get("start_abs", 0))
    end_abs = int(man.get("end_abs", 0))
    ranges = man.get("ranges") or []
    
    # Try centralized TitleService first (Gemini 3 Flash)
    # Use the new range-based method that loads transcript from ai_data files
    if USE_TITLE_SERVICE and TitleService is not None:
        try:
            print(f"🎯 Generating arc title with Gemini 3 Flash for arc {arc_idx}...")
            service = TitleService(vod_id)
            
            # Use range-based method if we have valid time range
            if end_abs > start_abs:
                title = service.generate_arc_title_for_range(start_abs, end_abs)
            else:
                # Fallback to summary if no valid time range
                summaries = []
                for r in ranges[:5]:
                    s = str(r.get("summary") or "").strip()
                    if s:
                        summaries.append(s)
                summary_text = " | ".join(summaries) if summaries else ""
                title = service.generate_arc_title(summary_text)
            
            if title and len(title) >= 5:
                print(f"✨ Generated arc title: '{title}'")
                return title
        except Exception as e:
            print(f"⚠️ TitleService failed, using legacy: {e}")
    
    # Legacy fallback: use old prompt-based approach
    prompt = _build_arc_title_prompt(vod_id, arc_idx, man)
    try:
        resp = call_llm(prompt, max_tokens=80, temperature=0.4, request_tag=f"arc_title_{vod_id}_{arc_idx:03d}")
        title = _extract_title_from_response(resp)
        if title:
            return title
    except Exception:
        pass
    
    # Final fallback
    first_summary = _first_nonempty_summary(ranges)
    if first_summary:
        return _condense(first_summary, 40)
    start_hms = str(man.get("start_hms") or _format_hms(man.get("start_abs", 0)))
    end_hms = str(man.get("end_hms") or _format_hms(man.get("end_abs", 0)))
    return f"Highlights {start_hms}–{end_hms}"


def _llm_generate_thumbnail_text(vod_id: str, arc_idx: int, man: Dict) -> str:
    """Generate thumbnail text using Gemini 3 Flash via centralized TitleService."""
    ranges = man.get("ranges") or []
    
    if USE_TITLE_SERVICE and TitleService is not None:
        try:
            service = TitleService(vod_id)
            # Use summaries combined
            summaries = []
            for r in ranges[:5]:
                s = str(r.get("summary") or "").strip()
                if s:
                    summaries.append(s)
            
            summary_text = " | ".join(summaries) if summaries else (str(man.get("summary") or ""))
            
            # If we have arc transcript helper, try to use it (optional)
            # But summary is usually enough for a hook
            
            return service.generate_thumbnail_text(summary_text)
        except Exception as e:
            print(f"⚠️ Thumbnail text generation failed: {e}")
            
    return "HIGHLIGHTS"


def _build_metadata_for_arc(vod_id: str, arc_idx: int, man: Dict, streamer: str, vod_title: str, generated_title: Optional[str] = None, timestamps: Optional[str] = None, thumbnail_text: Optional[str] = None) -> Dict:
    """Build metadata for arc with enhanced timestamps."""
    start_hms = str(man.get("start_hms") or _format_hms(man.get("start_abs", 0)))
    end_hms = str(man.get("end_hms") or _format_hms(man.get("end_abs", 0)))
    ranges = man.get("ranges") or []
    first_summary = _first_nonempty_summary(ranges)
    brief = _condense(first_summary, 60) if first_summary else f"Highlights {start_hms}–{end_hms}"
    
    base_title = f"{brief}"
    title = (generated_title.strip() if isinstance(generated_title, str) and generated_title.strip() else base_title)

    # Description with timestamps (no arc window/title/streamer lines)
    twitch_vod_url = f"https://www.twitch.tv/videos/{vod_id}"
    lines = [f"Original VOD: {twitch_vod_url}"]

    # Add normalized timestamps if available
    if timestamps:
        normalized = _normalize_timestamp_lines(timestamps)
        if normalized:
            lines.append("")
            lines.extend(normalized)

    # Tags
    game_tags = _get_game_tags(vod_id)
    tags: List[str] = []
    _streamer = streamer.strip() or _extract_streamer_from_vod_info(_load_vod_info(vod_id))
    if _streamer:
        tags.append(_streamer)
    tags.extend(game_tags)
    tags.extend(["TwitchHighlights", "Gaming"])

    # Build hashtags
    def _to_hashtag(value: str) -> str:
        s = re.sub(r"\s+", "", value or "").strip()
        s = re.sub(r"[^A-Za-z0-9]", "", s)
        return f"#{s}" if s else ""

    hashtag_candidates: List[str] = []
    if _streamer:
        hashtag_candidates.append(_to_hashtag(_streamer))
    for g in game_tags:
        ht = _to_hashtag(g)
        if ht:
            hashtag_candidates.append(ht)
    hashtag_candidates.extend([_to_hashtag("TwitchHighlights"), _to_hashtag("Gaming")])
    
    seen_ht = set()
    hashtags: List[str] = []
    for h in hashtag_candidates:
        if h and h not in seen_ht:
            seen_ht.add(h)
            hashtags.append(h)
    if hashtags:
        lines.append("")
        lines.append(" ".join(hashtags[:6]))
    
    description = "\n".join(lines)

    metadata = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": "20",
            "defaultLanguage": "en",
            "defaultAudioLanguage": "en",
        },
        "status": {
            "privacyStatus": "public",
            "madeForKids": False,
            "selfDeclaredMadeForKids": False,
        },
        "streamsniped_metadata": {
            "vod_id": vod_id,
            "arc_index": arc_idx,
            "start_hms": start_hms,
            "end_hms": end_hms,
            "thumbnail_text": thumbnail_text or "HIGHLIGHTS",
        },
    }
    return metadata


def main() -> None:
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python processing-scripts/generate_arch_youtube_metadata_enhanced.py <vod_id> [--arc <index>]")
        sys.exit(1)

    vod_id = sys.argv[1]
    arc_index: Optional[int] = None
    if "--arc" in sys.argv:
        try:
            i = sys.argv.index("--arc")
            if i + 1 < len(sys.argv):
                arc_index = int(sys.argv[i + 1])
        except Exception:
            print("X Invalid --arc flag usage")
            sys.exit(1)

    manifests = _load_arc_manifests(vod_id, arc_index)
    if not manifests:
        print("X No arc manifests found")
        sys.exit(1)

    from src.config import config
    out_dir = config.get_ai_data_dir(vod_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = _load_streamer_info(vod_id)
    streamer = info.get("streamer", "").strip()
    vod_title = info.get("vod_title", "").strip()

    for mp in manifests:
        # Per-arc guard so a single failure doesn't halt the whole run
        try:
            try:
                man = json.loads(mp.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"Failed reading {mp}: {e}")
                continue

            arc_idx = int(man.get("arc_index") or 0)

            # Generate title
            title_in_manifest = str(man.get("title") or "").strip()
            if title_in_manifest:
                llm_title = title_in_manifest
            else:
                llm_title = _llm_generate_arc_title(vod_id, arc_idx, man)
                try:
                    man["title"] = llm_title
                    mp.write_text(json.dumps(man, indent=2), encoding="utf-8")
                except Exception:
                    pass

            # Generate timestamps using Ollama
            print(f"Generating timestamps for arc {arc_idx}")
            timestamps = _generate_timestamps_with_ollama(vod_id, man)

            # Generate thumbnail text
            print(f"Generating thumbnail text for arc {arc_idx}")
            thumbnail_text = _llm_generate_thumbnail_text(vod_id, arc_idx, man)
            print(f"✨ Thumbnail text: {thumbnail_text}")

            if timestamps:
                # Print normalized timestamps to keep output clean
                normalized = _normalize_timestamp_lines(timestamps)
                print(f"Generated timestamps for arc {arc_idx}:")
                if normalized:
                    print("\n".join(normalized))
                else:
                    print(timestamps)
            else:
                print(f"No timestamps generated for arc {arc_idx}")

            # Build metadata
            metadata = _build_metadata_for_arc(vod_id, arc_idx, man, streamer, vod_title, llm_title, timestamps, thumbnail_text)
            out_path = out_dir / f"{vod_id}_arc_{arc_idx:03d}_youtube_metadata.json"
            out_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(f"Wrote enhanced arc metadata: {out_path}")
        except Exception as e:
            # Log and continue with next arc
            print(f"X Arc {arc_idx}: {e}")
            try:
                import traceback
                traceback.print_exc()
            except Exception:
                pass
            continue


if __name__ == "__main__":
    main()
