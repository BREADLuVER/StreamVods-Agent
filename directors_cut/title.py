#!/usr/bin/env python3
"""
Generate a Director's Cut title using Gemini 3 Flash via centralized TitleService.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.config import config
from storage import StorageManager
from src.ai_client import call_llm

# Import centralized title service
try:
    from src.title_service import TitleService
    USE_TITLE_SERVICE = True
except ImportError:
    USE_TITLE_SERVICE = False
    TitleService = None


def _read_json_if_exists(path: Path) -> Dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            return {}
    return {}


def generate_title(vod_id: str) -> str:
    """Generate a Director's Cut title using Gemini 3 Flash."""
    ai_dir = config.get_ai_data_dir(vod_id)
    narrative = _read_json_if_exists(Path(f"data/vector_stores/{vod_id}/enhanced_director_cut_manifest.json"))
    chapters = _read_json_if_exists(ai_dir / f"{vod_id}_chapters.json")

    # Extract topics from narrative ranges
    topics: List[str] = []
    for r in narrative.get('ranges', [])[:5]:
        tk = str(r.get('topic_key') or '')
        if tk and tk not in topics:
            topics.append(tk)
    
    # Extract games from chapters
    games: List[str] = []
    try:
        for ch in chapters.get('chapters', [])[:3]:
            game = str(ch.get('category') or '').strip()
            if game and game not in games:
                games.append(game)
    except Exception:
        pass

    # Try centralized TitleService first (Gemini 3 Flash)
    if USE_TITLE_SERVICE and TitleService is not None:
        try:
            print("🎯 Generating Director's Cut title with Gemini 3 Flash...")
            service = TitleService(vod_id)
            title = service.generate_directors_cut_title(topics, games)
            if title and len(title) >= 10:
                print(f"✨ Generated title: '{title}'")
                return title
        except Exception as e:
            print(f"⚠️ TitleService failed, using legacy: {e}")

    # Legacy fallback
    return _generate_title_legacy(vod_id, narrative, chapters, topics, games)


def _generate_title_legacy(vod_id: str, narrative: Dict, chapters: Dict, topics: List[str], games: List[str]) -> str:
    """Legacy title generation using call_llm."""
    topics_str = ", ".join(topics) if topics else "varied highlights"
    game = games[0] if games else "Gaming"

    # Load stream context
    sc_extra = ""
    try:
        sc_path = config.get_ai_data_dir(vod_id) / f"{vod_id}_stream_context.json"
        if sc_path.exists():
            _sc = json.loads(sc_path.read_text(encoding='utf-8'))
            _streamer = str(_sc.get('streamer') or '').strip()
            _vod_title = str(_sc.get('vod_title') or '').strip()
            _cats = _sc.get('chapter_categories')
            _cats_str = ", ".join([str(c).strip() for c in _cats if str(c).strip()]) if isinstance(_cats, list) else ''
            lines = []
            if _streamer:
                lines.append(f"Streamer: {_streamer}")
            if _vod_title:
                lines.append(f"Original VOD Title: {_vod_title}")
            if _cats_str:
                lines.append(f"Categories: {_cats_str}")
            if lines:
                sc_extra = "\n".join(lines)
    except Exception:
        pass

    prompt = f"""Generate a YouTube title for a Director's Cut gaming highlights video.

CONTEXT:
{sc_extra if sc_extra else f"Streamer: {vod_id}"}
Game: {game}
Topics: {topics_str}

RULES:
- 50-80 characters
- Personal, conversational tone
- No generic words like "highlights", "best moments", "epic"
- No trailing phrases after colons
- Sound like a professional editor.

EXAMPLES:
- "the offlinetv PEAK experience"
- "becoming the greatest pokemon trainer"
- "the squad tried to stay fully incognito"
- "What Mafia looks like when Toast LOCKS IN"

Return ONLY the title, nothing else."""

    try:
        title = call_llm(prompt, max_tokens=100, temperature=0.4, request_tag="director_cut_title")
        title = (title or '').strip().replace('"', '').replace("'", '')
        if len(title) > 80:
            title = title[:77] + '...'
        if not title:
            title = f"{game} Director's Cut"
        return title
    except Exception:
        return f"{game} Director's Cut"


def save_title(vod_id: str, title: str) -> Path:
    ai_dir = config.get_ai_data_dir(vod_id)
    out = ai_dir / f"{vod_id}_director_cut_title.json"
    data = {
        "vod_id": vod_id,
        "title": title,
    }
    StorageManager().save_json_with_cloud_backup(str(out), data, s3_key=f"ai_data/{vod_id}/{vod_id}_director_cut_title.json")
    return out

