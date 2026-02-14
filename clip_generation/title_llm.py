"""
LLM title generation for clips.

Now uses centralized TitleService with Gemini 3 Flash for consistent,
high-quality title generation.
"""

import json
import re
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict

from .types import FinalClip, WindowDoc

# Import centralized title service
try:
    from src.title_service import TitleService, generate_clip_title as _generate_clip_title
except ImportError:
    TitleService = None
    _generate_clip_title = None


@dataclass
class TitleCandidate:
    """A single title candidate with metadata."""
    title: str
    style_approach: str
    score: float = 0.0
    reasoning: str = ""


def _is_invalid_title_text(text: str) -> bool:
    """Return True if the title is clearly meta/instructional or user-referential.

    Filters obvious failure modes like "title", "**title**", or phrases like
    "the user asked me" / "ok user" that indicate instruction leakage.
    """
    if not text:
        return True
    lowered = text.strip().lower()
    # Remove common markup/punctuation wrappers
    cleaned = re.sub(r"[\*`'\"\(\)\[\]\{\}\.:;!?]", "", lowered).strip()

    # Exact meta labels
    if cleaned in {"title", "the title"}:
        return True

    # Instruction leakage phrases
    bad_substrings = [
        "the user asked me",
        "the user asked",
        "ok the user",
        "ok user",
        "the user",
        "as an ai",
        "i am an ai",
        "as a language model",
    ]
    for s in bad_substrings:
        if s in lowered:
            return True

    return False


def _to_title_case_preserve_acronyms(title: str) -> str:
    """Capitalize each word while preserving all-caps acronyms (length>=3).

    Also capitalizes hyphenated parts independently.
    """
    if not title:
        return title

    def cap_word(word: str) -> str:
        if not word:
            return word
        if len(word) >= 3 and word.isupper():
            return word
        return word[:1].upper() + word[1:].lower()

    def cap_token(token: str) -> str:
        if token.isspace():
            return token
        if "-" in token:
            parts = token.split("-")
            return "-".join(cap_word(p) for p in parts)
        return cap_word(token)

    # Keep original spacing tokens
    tokens = re.split(r"(\s+)", title.strip())
    capped = "".join(cap_token(t) for t in tokens)
    # Normalize any accidental multiple spaces
    capped = re.sub(r"\s+", " ", capped).strip()
    return capped


def select_top_chat_lines(chat_text: str, k: int = 30) -> List[str]:
    """Select top chat lines from chat text."""
    if not chat_text:
        return []
    lines = [ln.strip() for ln in chat_text.splitlines() if ln.strip()]
    return lines[:k]


def build_clip_context(
    docs: List[WindowDoc], 
    start: float, 
    end: float, 
    vod_id: str,
    context_padding_s: float = 60.0
) -> Dict:
    """Build full context (transcript + chat) for a clip with padding."""
    
    # Get docs within the clip window plus padding for context
    context_start = max(0, start - context_padding_s)
    context_end = end + context_padding_s
    clip_docs = [d for d in docs if d.start < context_end and d.end > context_start]
    
    # Build transcript segments with context markers
    transcript_segments = []
    for d in clip_docs:
        segment_type = "before"
        if d.start >= start and d.end <= end:
            segment_type = "clip"
        elif d.start < start and d.end > start:
            segment_type = "transition_in"
        elif d.start < end and d.end > end:
            segment_type = "transition_out"
        elif d.start >= end:
            segment_type = "after"
            
        transcript_segments.append({
            "start_time": round(float(d.start), 3),
            "end_time": round(float(d.end), 3),
            "text": (d.text or "").replace("\n", " "),
            "context": segment_type
        })
    
    # Build chat lines
    chat_lines = []
    for d in clip_docs:
        chat_lines.extend(select_top_chat_lines(d.chat_text, k=8))
    
    # Get streamer context if available (same as generate_clips_manifest.py)
    streamer_context = ""
    try:
        from pathlib import Path
        sc_path = Path(f"data/ai_data/{vod_id}/{vod_id}_stream_context.json")
        if sc_path.exists():
            with open(sc_path, 'r', encoding='utf-8') as f:
                sc = json.load(f)
            streamer = str(sc.get('streamer') or '').strip()
            title = str(sc.get('vod_title') or '').strip()
            cats = sc.get('chapter_categories')
            cats_str = ", ".join([str(c).strip() for c in cats if str(c).strip()]) if isinstance(cats, list) else ""
            
            context_lines = []
            if streamer:
                context_lines.append(f"Streamer: {streamer}")
            if title:
                # Align with directors_cut/title.py wording
                context_lines.append(f"Original VOD Title: {title}")
            if cats_str:
                context_lines.append(f"Categories: {cats_str}")
            if context_lines:
                streamer_context = "\n".join(context_lines) + "\n"
    except Exception:
        pass
    
    # Fallback: ensure the model gets at least some streamer context
    if not streamer_context:
        streamer_context = f"Streamer: {vod_id}\n"
    
    return {
        "vod_id": vod_id,
        "streamer_context": streamer_context,
        "transcript": transcript_segments,
        "chat": chat_lines[:15],  # Limit to 15 chat lines
        "start_time": start,
        "end_time": end,
        "duration": end - start,
    }


def generate_title_candidates(
    context: Dict,
    streamer_name: str,
    n_candidates: int = 5
) -> List[TitleCandidate]:
    """Generate multiple diverse title candidates using different style approaches."""
    
    # Style approaches for diversity
    style_approaches = [
        {
            "name": "streamer_action",
            "description": f"{streamer_name} + specific action (e.g. '{streamer_name} accidentally deletes save')",
        },
        {
            "name": "situation_focus",
            "description": "Focus on the situation/object, not who (e.g. 'The worst trade deal in history')",
        },
        {
            "name": "consequence_first",
            "description": "Start with the result (e.g. 'How one click ended the run')",
        },
        {
            "name": "quote_fragment",
            "description": "Short quote from streamer (e.g. 'I didn't mean to do that')",
        },
        {
            "name": "when_pattern",
            "description": "'When X happens' pattern (e.g. 'When the impostor likes chicken teriyaki')",
        },
    ]
    
    candidates = []
    
    for i, style in enumerate(style_approaches[:n_candidates]):
        prompt = f"""
            You are a viral clip editor. Generate a title that sounds like it came from a real human editor, not AI.
            Ground truth stream context: {context['streamer_context']}.
            Window: {int(context['start_time'])}s to {int(context['end_time'])}s (duration ~{int(context['duration'])}s)

            Transcript (with timestamps and context):
            {json.dumps(context['transcript'], ensure_ascii=False, indent=2)}

            TASK:
            Find the specific thing that happened in this clip. Generate a title that doesn't sound AI or a blog post.
            
            CONTEXT ANALYSIS:
            - Use the "before" context to understand what led up to this moment
            - Use the "clip" context to see what actually happened in the clip
            - Use the "after" context to understand the consequences or reactions
            - Look at the transcript to understand what actually happened
            - Understand the full situation from the broader context, not just the clip itself
            
            == CRITICAL RULES ==
            1) NEVER make titles about “who did what to whom” or infer relationships between characters.
            - Don’t write “X does [action] to Y.”
            2) The ONLY person you may name is the streamer: {streamer_name}.
            - Do NOT name teammates, opponents, friends, guests, chatters, viewers.
            - Do NOT copy names from the transcript (they are unreliable).
            3) Focus on WHAT HAPPENED (the event/phenomenon), not WHO did it to WHOM.
            4) Absolutely NO invented or misspelled names.
            - If you aren’t 100% sure a name is the streamer (from Stream Context), don’t use it.
            5) Why? Because transcript names are unreliable. 
            - John becomes Josh, Hjune becomes June, you copy names from the transcript and title can be completely wrong.
            6) Never infer relationships between characters.
            - Don’t write “X does [action] to Y.”
            - Don’t write “X and Y [action] together.”
            - Don’t write “X and Y [action] against Z.”
            - Don’t write “X and Y [action] with Z.”
            - Don’t write “X and Y [action] for Z.”
            
            TITLE RULES:
            - Maximum 40 characters (including spaces)
            - No emojis, quotes, or punctuation
            - Describe the specific action or event that happened
            - Use simple, direct language
            - Sound like a fan editor wrote it (use {streamer_name} instead of "I")

            CRITICAL OUTPUT FORMAT:
            - Return ONLY the title text itself
            - NO parenthetical notes like "(Generated based on...)" or "(I've followed...)"
            - NO explanations or commentary after the title
            - NO quotes around the title
            - Just the raw title text, period
            - Maximum 40 characters (including spaces)
            - Do not name any characters, only the streamer, do not infer relationships
        """  # noqa: F821
        
        try:
            from src.ai_client import call_llm_ollama, call_llm
            # Try Ollama first, then fallback to general LLM
            try:
                title = call_llm_ollama(
                    prompt,
                    max_tokens=4096,
                    temperature=0.7,  # Higher temp for diversity
                    request_tag=f"title_gen_{style['name']}",
                ).strip()
            except Exception as ollama_error:
                print(f" Ollama failed for {style['name']}: {ollama_error}")
                # Fallback to general LLM
                title = call_llm(
                    prompt,
                    max_tokens=4096,
                    temperature=0.7,
                    request_tag=f"title_gen_{style['name']}_fallback",
                ).strip()
            
            # Clean up
            if title.startswith('"') and title.endswith('"'):
                title = title[1:-1]
            if title.startswith("'") and title.endswith("'"):
                title = title[1:-1]
            
            # Strip out commentary that LLM adds despite instructions
            # Remove parenthetical notes at end: "(Generated based on...)", "(I've followed...)"
            title = re.sub(r'\s*\([^)]*(?:generated|followed|based|analysis|provided)[^)]*\)\s*$', '', title, flags=re.IGNORECASE)
            # Remove newlines and extra text after the title
            title = title.split('\n')[0].strip()
            
            # Enforce 50 char limit
            if len(title) > 50:
                title = title[:50].rsplit(' ', 1)[0]
            
            # Filter invalid/meta titles early
            if title and not _is_invalid_title_text(title):
                candidates.append(TitleCandidate(
                    title=title,
                    style_approach=style['name']
                ))
            else:
                if title:
                    print(f"⚠️  Dropping invalid generated title: {title}")
        except Exception as e:
            print(f"⚠️  Candidate generation failed for {style['name']}: {e}")
    
    return candidates


def rerank_candidates(
    candidates: List[TitleCandidate],
    context: Dict,
    streamer_name: str
) -> List[TitleCandidate]:
    """Use LLM to score each candidate based on how human and natural they sound."""
    
    if not candidates:
        return []
    
    # Safety net: drop any invalid/meta candidates before scoring
    pre_filter_len = len(candidates)
    candidates = [c for c in candidates if not _is_invalid_title_text(c.title)]
    if not candidates:
        print("⚠️  All candidates were invalid after filtering")
        return []
    if len(candidates) != pre_filter_len:
        print(f"⚠️  Dropped {pre_filter_len - len(candidates)} invalid candidate(s) before rerank")
    
    # Build candidate list for prompt
    candidates_text = "\n".join([
        f"{i+1}. {c.title} (approach: {c.style_approach})"
        for i, c in enumerate(candidates)
    ])
    
    prompt = f"""You are judging clip titles to find the most natural, 
    human-sounding one. CANDIDATES: {candidates_text}
    CONTEXT: - Streamer: {streamer_name} 
    - Duration: {context['duration']:.0f}s 
    SCORING CRITERIA: 
    1. Natural tone (1-10): Does it sound like a real person wrote it, not AI? 
    2. Specificity (1-10): Is it about a concrete event, not generic fluff? 
    4. Avoids formulaic (1-10): Is it unique? Like a poetic phrase, quote, not paraphrasing the transcript?
    5. Only one subject (1-10): Does it only have one subject and that subject is {streamer_name} or a noun/object? No "X and Y" No "X's Y"
    Rate each candidate 1-40 (sum of 4 criteria). 
    FORMAT YOUR RESPONSE EXACTLY LIKE THIS: 
    1. score=... reasoning=...
    Return ONLY the ratings in this exact format, one per line. """
    
    try:
        from src.ai_client import call_llm_ollama, call_llm
        # Try Ollama first, then fallback to general LLM
        try:
            response = call_llm_ollama(
                prompt,
                max_tokens=300,
                temperature=0.2,  # Lower temp for consistent scoring
                request_tag="title_rerank",
            )
        except Exception as ollama_error:
            print(f" Ollama reranking failed: {ollama_error}")
            # Fallback to general LLM
            response = call_llm(
                prompt,
                max_tokens=300,
                temperature=0.2,
                request_tag="title_rerank_fallback",
            )
        
        # Parse scores from response
        for line in response.strip().split('\n'):
            if not line.strip():
                continue
            
            try:
                # Parse: "1. score=32 reasoning=..."
                parts = line.split('score=', 1)
                if len(parts) < 2:
                    continue
                
                idx_str = parts[0].strip().rstrip('.')
                idx = int(idx_str) - 1
                
                score_and_reason = parts[1].split('reasoning=', 1)
                score = float(score_and_reason[0].strip())
                reasoning = score_and_reason[1].strip() if len(score_and_reason) > 1 else ""
                
                if 0 <= idx < len(candidates):
                    candidates[idx].score = score
                    candidates[idx].reasoning = reasoning
            except (ValueError, IndexError) as e:
                print(f"⚠️  Failed to parse ranking line: {line} ({e})")
                continue
        
    except Exception as e:
        print(f"⚠️  Reranking failed: {e}")
        # Fallback: assign default scores
        for i, c in enumerate(candidates):
            c.score = 20.0 - i  # Prefer earlier candidates as fallback
            c.reasoning = "Reranking failed, using generation order"
    
    # Sort by score descending
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def log_title_candidates(
    vod_id: str,
    clip_start: float,
    candidates: List[TitleCandidate],
    final_title: str
) -> None:
    """Log all title candidates and scores for analysis."""
    log_dir = Path(f"data/vector_stores/{vod_id}/title_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"clip_{int(clip_start)}.json"
    
    log_data = {
        "vod_id": vod_id,
        "clip_start": clip_start,
        "final_title": final_title,
        "candidates": [asdict(c) for c in candidates],
        "top_score": candidates[0].score if candidates else 0.0,
    }
    
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️  Failed to log candidates: {e}")


def generate_title_for_clip(
    clip: FinalClip,
    docs: List[WindowDoc],
    vod_id: str,
    use_reranking: bool = True
) -> str:
    """Generate title for a single clip using Gemini 3 Flash (centralized service).
    
    Args:
        clip: The clip to generate a title for
        docs: Window documents for context (legacy, now loads from ai_data files)
        vod_id: VOD identifier
        use_reranking: Legacy parameter, ignored (kept for API compatibility)
    """
    
    # Extract streamer name from stream context
    streamer_name = str(vod_id)
    try:
        import json
        from pathlib import Path
        sc_path = Path(f"data/ai_data/{vod_id}/{vod_id}_stream_context.json")
        if sc_path.exists():
            sc = json.loads(sc_path.read_text(encoding='utf-8'))
            name = str(sc.get('streamer') or '').strip()
            if name:
                streamer_name = name
    except Exception:
        pass
    
    # Use centralized TitleService with Gemini 3 Flash
    # Load transcript directly from ai_data files using clip time range
    if TitleService is not None:
        try:
            print(f"🎯 Generating title with Gemini 3 Flash for {clip.start_hms} → {clip.end_hms}...")
            service = TitleService(vod_id)
            
            # Use the new range-based method that loads transcript from ai_data files
            title = service.generate_clip_title_for_range(clip.start, clip.end, streamer_name)
            
            if title and not _is_invalid_title_text(title):
                final_title = _to_title_case_preserve_acronyms(title)
                print(f"✨ Generated: '{final_title}'")
                
                # Log for analysis
                log_title_candidates(vod_id, clip.start, [TitleCandidate(title=final_title, style_approach="gemini_3_flash", score=10.0)], final_title)
                
                return final_title
                
        except Exception as e:
            print(f"⚠️ Gemini 3 Flash failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Fallback if TitleService not available or failed
    print("⚠️ Using fallback title")
    return "Highlight Clip"


def generate_titles_for_clips(
    clips: List[FinalClip],
    docs: List[WindowDoc],
    vod_id: str,
    concurrent: bool = False,
    max_workers: int = 4
) -> List[FinalClip]:
    """Generate titles for multiple clips."""
    
    if not concurrent:
        # Sequential processing
        for clip in clips:
            clip.title = generate_title_for_clip(clip, docs, vod_id)
        return clips
    
    # Concurrent processing
    from concurrent.futures import ThreadPoolExecutor
    
    def process_clip(clip: FinalClip) -> FinalClip:
        clip.title = generate_title_for_clip(clip, docs, vod_id)
        return clip
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(process_clip, clips))