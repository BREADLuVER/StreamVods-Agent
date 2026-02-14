#!/usr/bin/env python3
"""
PASS 1: Burst summary labeling

Reads bursts from vector store DB, constructs compact prompts with
chapter intro + previous summary + current burst transcript/chat + metrics,
calls an LLM, and writes summary/topic/energy back to the DB.
"""

import sys
import os
import json
import time
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ai_client import call_llm
from concurrent.futures import ThreadPoolExecutor
from vector_store.document_builder import extract_keywords

# Load environment variables
from dotenv import load_dotenv
load_dotenv("config/streamsniped.env")


PROMPT_JC = (
    "You are an expert VOD segment grouper for a Just Chatting stream.\n"
    "For the CURRENT segment, produce:\n"
    "  1) a concise <=25 word summary,\n"
    "  2) a short subtopic label (2–5 words),\n"
    "  3) an energy bucket (low|medium|high),\n"
    "  4) a stable parent topic_key (1–3 general words),\n"
    "  5) whether the CURRENT continues the same narrative as PREVIOUS.\n"
    "\n"
    "Grouping rules:\n"
    "- When creating a new parent, prefer a general parent that could cover both PREV and CURRENT.\n"
    "- Treat close details/subtopics as part of the SAME parent unless the subject/goal truly changes.\n"
    "- Reuse a parent from RECENT_PARENT_KEYS whenever it still fits; avoid inventing new parents if an existing one applies.\n"
    "- Be deterministic and minimal. Output only the JSON.\n"
    "\n"
    "INPUTS\n"
    "CHAPTER_TOP: \"{chapter_intro}\"\n"
    "\n"
    "PREV_CONTEXT: {{\"prev_topic\":\"{prev_topic}\",\"prev_summary\":\"{prev_summary}\"}}\n"
    "\n"
    "RECENT_PARENT_KEYS: {recent_parent_keys}\n"
    "\n"
    "CURRENT_TRANSCRIPT:\n\"\n{burst_transcript}\n\"\n"
    "\n"
    "METRICS:\n{{ \"chat_rate_z\": {chat_rate_z:.2f}, \"burst_score\": {burst_score:.2f}, \"reactions\": {reactions} }}\n"
    "\n"
    "OUTPUT — return ONLY valid JSON with these keys and no extra text:\n"
    "{{\n"
    "  \"summary\": \"<<=25 words>\",\n"
    "  \"topic\": \"<2-5 word subtopic>\",\n"
    "  \"energy\": \"low|medium|high\",\n"
    "  \"topic_key\": \"<stable parent (1-3 general words)>\",\n"
    "  \"same_topic_prev\": true|false,\n"
    "  \"role\": \"intro|conclusion|reacting|talking|tech_issue|afk|filler\",\n"
    "  \"confidence\": <0.0-1.0>\n"
    "}}\n"
)

PROMPT_GAMEPLAY = (
    "You are an expert VOD segment grouper for a Gameplay stream.\n"
    "For the CURRENT segment, produce:\n"
    "  1) a concise <=25 word summary,\n"
    "  2) a short subtopic label (2–5 words),\n"
    "  3) an energy bucket (low|medium|high),\n"
    "  4) a stable goal topic_key (1–3 general words),\n"
    "  5) a narrative role: intro|build_up|conflict|peak|resolution|filler|afk.\n"
    "\n"
    "- Your job is to analyze a gameplay segment and label it with a role that describes the narrative flow of the segment.\n"
    "- Roles: 'peak' = climax/high-stakes; 'resolution' = aftermath/defeat/reset; 'conflict' = active struggle; 'build_up' = setup before conflict; 'intro' = chapter start/setup; 'filler' = low-stakes in-between; 'afk' = away from keyboard, transcript does not make sense.\n"
    "- Use CHAPTER_TOP and PREV_CONTEXT as priors; be deterministic; output JSON only.\n"
    "\n"
    "INPUTS\n"
    "CHAPTER_TOP: \"{chapter_intro}\"\n"
    "PREV_CONTEXT: {{\"prev_topic\":\"{prev_topic}\",\"prev_summary\":\"{prev_summary}\"}}\n"
    "RECENT_ROLES: {recent_roles}\n"
    "CURRENT_TRANSCRIPT:\n\"\n{burst_transcript}\n\"\n"
    "METRICS:\n{{\"chat_rate_z\": {chat_rate_z:.2f}, \"burst_score\": {burst_score:.2f}, \"reactions\": {reactions}}}\n"
    "\n"
    "OUTPUT — return ONLY valid JSON with these keys and no extra text:\n"
    "{{\n"
    "  \"summary\": \"<<=25 words>\",\n"
    "  \"topic\": \"<2-5 word subtopic>\",\n"
    "  \"energy\": \"low|medium|high\",\n"
    "  \"topic_key\": \"<goal (1-3 words)>\",\n"
    "  \"role\": \"intro|build_up|conflict|peak|resolution|filler|afk\",\n"
    "  \"confidence\": <0.0-1.0>\n"
    "}}\n"
)

def _db_path(vod_id: str) -> Path:
    return Path(f"data/vector_stores/{vod_id}/metadata.db")


def _load_segments(vod_id: str) -> List[Dict]:
    ai_data_dir = Path(f"data/ai_data/{vod_id}")
    
    # Try filtered data first
    filtered_path = ai_data_dir / f"{vod_id}_filtered_ai_data.json"
    raw_path = ai_data_dir / f"{vod_id}_ai_data.json"
    
    ai_data_path = None
    if filtered_path.exists():
        ai_data_path = filtered_path
    elif raw_path.exists():
        ai_data_path = raw_path
    else:
        return []
    
    data = json.loads(ai_data_path.read_text(encoding="utf-8"))
    return data.get("segments", [])


def _load_chapters(vod_id: str) -> Dict[str, Dict]:
    # Prefer unmerged chapters for labeling; fallback to merged
    base = Path(f"data/ai_data/{vod_id}")
    unmerged = base / f"{vod_id}_chapters_unmerged.json"
    merged = base / f"{vod_id}_chapters.json"
    load_path = unmerged if unmerged.exists() else merged
    if not load_path.exists():
        return {}
    raw = json.loads(load_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("chapters"), list):
        arr = raw["chapters"]
    elif isinstance(raw, list):
        arr = raw
    else:
        arr = []
    return {c.get("id"): c for c in arr if isinstance(c, dict) and c.get("id")}


def get_chapter_transcript(chapter: Dict, segments: List[Dict]) -> str:
    if not chapter:
        return ""
    start_time = chapter.get('start_time', 0)
    end_time = chapter.get('end_time', 0)
    parts: List[str] = []
    for s in segments:
        if start_time <= s.get('start_time', 0) and s.get('end_time', 0) <= end_time and s.get('transcript'):
            parts.append(s['transcript'])
    text = " ".join(parts)
    return text[:1200]


def _choose_top_chat(chat_text: str, k: int = 10) -> str:
    if not chat_text:
        return ""
    lines = [ln.strip() for ln in chat_text.split("\n") if ln.strip()]
    return "\n".join(lines[:k])


def _extract_json(text: str) -> Optional[Dict]:
    if not text:
        return None
    text = text.strip()
    # Fast path: direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass
    # Heuristic: find first {...}
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        chunk = text[start:end + 1]
        try:
            return json.loads(chunk)
        except Exception:
            return None
    return None


def _extract_json_array(text: str) -> Optional[List[Dict]]:
    if not text:
        return None
    t = text.strip()
    # Direct array parse
    try:
        val = json.loads(t)
        if isinstance(val, list):
            return val  # type: ignore
    except Exception:
        pass
    # Heuristic: slice outermost [...]
    start = t.find('[')
    end = t.rfind(']')
    if start != -1 and end != -1 and end > start:
        chunk = t[start:end + 1]
        try:
            val = json.loads(chunk)
            if isinstance(val, list):
                return val  # type: ignore
        except Exception:
            return None
    return None


def _energy_from_metrics(cz: float, bs: float) -> str:
    if (cz or 0) > 1.0 or (bs or 0) > 1.6:
        return "high"
    if (cz or 0) < -0.5 and (bs or 0) < 0.9:
        return "low"
    return "medium"


def _is_valid_payload_keys(data: Optional[Dict], required: List[str]) -> bool:
    if not isinstance(data, dict):
        return False
    for k in required:
        if k not in data:
            return False
    # non-empty summary/topic/topic_key
    if not (str(data.get("summary", "")).strip() and str(data.get("topic", "")).strip() and str(data.get("topic_key", "")).strip()):
        return False
    return True


def _fallback_labels(prev_topic_key: str, chapter_intro: str, text: str, cz: float, bs: float) -> Dict:
    topic_key = (prev_topic_key or (chapter_intro.split(" ")[0] if chapter_intro else "chat")).lower()
    summary = (text or "").strip()
    if len(summary) > 200:
        summary = summary[:200]
    topic = topic_key if len(topic_key.split()) <= 4 else "conversation"
    energy = _energy_from_metrics(cz, bs)
    return {
        "summary": summary or f"Continuation on {topic_key}",
        "topic": topic,
        "energy": energy,
        "topic_key": topic_key,
        "same_topic_prev": bool(prev_topic_key),
        "link_type": "continue" if prev_topic_key else "switch",
        "link_evidence": (text or "")[:60],
        "confidence": 0.0,
    }


def update_burst_labels(vod_id: str):
    db = _db_path(vod_id)
    if not db.exists():
        print(f"❌ DB not found: {db}")
        sys.exit(1)

    segments = _load_segments(vod_id)
    chapters = _load_chapters(vod_id)

    conn = sqlite3.connect(str(db))
    cur = conn.cursor()

    # Ensure required columns exist (migration when running summarizer standalone)
    def _ensure_columns():
        cur.execute("PRAGMA table_info(documents)")
        cols = {row[1] for row in cur.fetchall()}
        to_add = []
        if "same_topic_prev" not in cols:
            to_add.append(("same_topic_prev", "INTEGER DEFAULT 0"))
        if "topic_thread" not in cols:
            to_add.append(("topic_thread", "INTEGER DEFAULT 0"))
        if "topic_key" not in cols:
            to_add.append(("topic_key", "TEXT"))
        if "link_type" not in cols:
            to_add.append(("link_type", "TEXT"))
        if "link_evidence" not in cols:
            to_add.append(("link_evidence", "TEXT"))
        if "confidence" not in cols:
            to_add.append(("confidence", "REAL DEFAULT 0"))
        for name, decl in to_add:
            cur.execute(f"ALTER TABLE documents ADD COLUMN {name} {decl}")
        if to_add:
            conn.commit()

    _ensure_columns()

    cur.execute(
        """
        SELECT id, start_time, end_time, chapter_id, text, chat_text, chat_rate_z, burst_score, reaction_hits, mode,
               summary, topic_key, confidence, role, same_topic_prev
        FROM documents
        ORDER BY start_time
        """
    )
    rows = cur.fetchall()

    updated = 0
    prev_summary = ""
    current_thread = 0
    prev_topic_key = ""
    recent_parents: List[str] = []
    recent_roles: List[str] = []
    last_chapter_id: Optional[str] = None
    for row in rows:
        _id, _start, _end, chap_id, text, chat_text, cz, bs, reacts, mode, existing_summary, existing_topic_key, existing_confidence, existing_role, existing_same_topic_prev = row

        # Reset narrative thread state when chapter switches
        if chap_id != last_chapter_id:
            prev_summary = ""
            current_thread = 0
            prev_topic_key = ""
            recent_parents = []
            recent_roles = []
            last_chapter_id = chap_id

        chapter_intro = get_chapter_transcript(chapters.get(chap_id, {}), segments)
        prev_context = {
            "prev_topic": "",
            "prev_summary": prev_summary or "",
        }
        # Use mode from document (which uses original chapter type if available)
        is_jc_irl = mode == 'jc'

        if is_jc_irl:
            prompt = PROMPT_JC.format(
                chapter_intro=chapter_intro,
                prev_topic=prev_context["prev_topic"],
                prev_summary=prev_context["prev_summary"],
                recent_parent_keys=json.dumps(list(dict.fromkeys(recent_parents))[-8:]),
                burst_transcript=text or "",
                chat_rate_z=cz or 0.0,
                burst_score=bs or 0.0,
                reactions=reacts or "{}",
            )
        else:
            prompt = PROMPT_GAMEPLAY.format(
                chapter_intro=chapter_intro,
                prev_topic=prev_context["prev_topic"],
                prev_summary=prev_context["prev_summary"],
                recent_roles=json.dumps(list(dict.fromkeys(recent_roles))[-8:]),
                burst_transcript=text or "",
                chat_rate_z=cz or 0.0,
                burst_score=bs or 0.0,
                reactions=reacts or "{}",
            )
        # Retry loop with validation (JC vs Gameplay have different required keys)
        data: Optional[Dict] = None
        MAX_RETRIES = 3
        required_keys = ["summary","topic","topic_key","energy","same_topic_prev"] if is_jc_irl else ["summary","topic","topic_key","energy","role"]
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw = call_llm(prompt, max_tokens=200, temperature=0.2, request_tag=f"burst_{_id}_try{attempt}")
            except Exception:
                raw = ""
            data = _extract_json(raw)
            if _is_valid_payload_keys(data, required_keys):
                break
            # tighten prompt slightly on retries by appending explicit JSON instruction
            prompt = prompt + "\nRespond with JSON only."
            time.sleep(0.2)

        if not _is_valid_payload_keys(data, required_keys):
            data = _fallback_labels(prev_topic_key, chapter_intro, text or "", cz or 0.0, bs or 0.0)

        summary = (data.get("summary") or "")[:200]
        topic = (data.get("topic") or "")[:64]
        energy = data.get("energy") or ""
        topic_key = (data.get("topic_key") or topic or "").lower().strip()
        same_topic_prev_hint = data.get("same_topic_prev")
        link_type = (data.get("link_type") or "").lower().strip()
        link_evidence = (data.get("link_evidence") or "")[:120]
        confidence = float(data.get("confidence") or 0.0)
        role_raw = (data.get("role") or "").strip().lower()
        valid_game_roles = {"intro","build_up","conflict","peak","resolution","filler","afk"}
        valid_jc_roles = {"intro","conclusion","reacting","talking","tech_issue","afk","filler"}
        if is_jc_irl:
            role = role_raw if role_raw in valid_jc_roles else ""
        else:
            role = role_raw if role_raw in valid_game_roles else ""

        # Fallback heuristic if model didn't set same_topic_prev
        if same_topic_prev_hint is None:
            # simple lexical heuristic on keys
            same_topic_prev_hint = bool(prev_topic_key and topic_key and (prev_topic_key in topic_key or topic_key in prev_topic_key))

        if updated == 0:
            current_thread = 1
        else:
            current_thread = current_thread if same_topic_prev_hint else (current_thread + 1)

        # Store role for both; JC roles are lightweight utilities
        # If gameplay and role missing from model but high salience, infer as safety net
        if (not is_jc_irl) and not role:
            # infer via salience proxy
            inferred = "peak" if (cz or 0.0) >= np.quantile([0.0, 1.0], 0.90) else ("conflict" if (cz or 0.0) >= 0.75 else "filler")
            role = inferred
        role_to_store = role
        role_conf_to_store = (confidence if role_to_store else 0.0)

        cur.execute(
            "UPDATE documents SET summary=?, topic=?, energy=?, role=?, role_confidence=?, same_topic_prev=?, topic_thread=?, topic_key=?, link_type=?, link_evidence=?, confidence=? WHERE id=?",
            (summary, topic, energy, role_to_store, role_conf_to_store, 1 if same_topic_prev_hint else 0, current_thread, topic_key, link_type, link_evidence, confidence, _id),
        )
        conn.commit()
        prev_summary = summary or prev_summary
        prev_topic_key = topic_key or prev_topic_key
        if is_jc_irl:
            if topic_key:
                recent_parents.append(topic_key)
        else:
            if role:
                recent_roles.append(role)
        updated += 1
        time.sleep(0.1)

    conn.close()
    print(f"✅ Burst summaries updated. ({updated} rows)")


def _update_burst_labels_batched(
    vod_id: str,
    batch_size: int = 12,
    max_workers: int = 8,
    only_missing: bool = True,
    limit: int = 0,
    offset: int = 0,
) -> None:
    db = _db_path(vod_id)
    if not db.exists():
        print(f"❌ DB not found: {db}")
        sys.exit(1)

    segments = _load_segments(vod_id)
    chapters = _load_chapters(vod_id)

    conn = sqlite3.connect(str(db))
    cur = conn.cursor()

    # Load all rows, keep full context
    cur.execute(
        """
        SELECT id, start_time, end_time, chapter_id, text, chat_text, chat_rate_z, burst_score, reaction_hits, mode,
               summary, topic_key, confidence, role, same_topic_prev
        FROM documents
        ORDER BY start_time
        """
    )
    rows = cur.fetchall()
    if not rows:
        print("No rows to process.")
        return

    # Convert to dicts
    items: List[Dict] = []
    for row in rows:
        (_id, _start, _end, chap_id, text, chat_text, cz, bs, reacts, mode,
         existing_summary, existing_topic_key, existing_conf, existing_role, existing_same_topic_prev) = row
        try:
            reactions = json.loads(reacts) if isinstance(reacts, str) and reacts else (reacts or {})
        except Exception:
            reactions = {}
        items.append({
            "id": _id,
            "start": float(_start or 0.0),
            "end": float(_end or 0.0),
            "chapter_id": chap_id,
            "text": text or "",
            "chat_text": chat_text or "",
            "chat_rate_z": float(cz or 0.0),
            "burst_score": float(bs or 0.0),
            "reactions": reactions,
            "mode": (mode or '').lower(),
            "summary": existing_summary or "",
            "topic_key": (existing_topic_key or '').lower(),
            "confidence": float(existing_conf or 0.0),
            "role": (existing_role or '').lower(),
            "same_topic_prev": 1 if existing_same_topic_prev else 0,
        })

    # Group by chapter
    from collections import defaultdict
    chap_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, it in enumerate(items):
        chap_to_indices[str(it.get("chapter_id") or "")].append(idx)

    # Selection: only_missing filter
    def needs_update(it: Dict) -> bool:
        if not only_missing:
            return True
        return (not it.get("summary")) or (not it.get("topic_key")) or (float(it.get("confidence") or 0.0) < 0.2)

    selected_pairs: List[tuple] = []  # list of (chapter_id, idx)
    for cid, idxs in chap_to_indices.items():
        for i in idxs:
            if needs_update(items[i]):
                selected_pairs.append((cid, i))

    # Apply offset/limit globally
    if offset > 0:
        selected_pairs = selected_pairs[offset:]
    if limit and limit > 0:
        selected_pairs = selected_pairs[:limit]

    if not selected_pairs:
        print("✅ Nothing to update (only-missing satisfied)")
        conn.close()
        return

    # Build deterministic prev-context per chapter
    def build_prev_context_arrays(idxs: List[int]) -> Dict[int, Dict[str, object]]:
        ctx: Dict[int, Dict[str, object]] = {}
        recent_keys: List[str] = []
        for k, idx in enumerate(idxs):
            # previous item in chapter
            prev_it = items[idxs[k-1]] if k-1 >= 0 else None
            # prev_summary: prefer DB summary else transcript tail
            prev_summary = (prev_it.get("summary") if prev_it else "") or ( (prev_it.get("text") or "")[-220:] if prev_it else "")
            prev_topic = (prev_it.get("topic_key") or "") if prev_it else ""
            if not prev_topic and prev_it and prev_it.get("text"):
                kws = extract_keywords(prev_it["text"]) or []
                prev_topic = (kws[0].lower() if kws else "")
            # recent parents: previous 8 non-empty topic_keys
            if prev_it and prev_it.get("topic_key"):
                key = str(prev_it["topic_key"]).lower()
                if key and key not in recent_keys:
                    recent_keys.append(key)
                    if len(recent_keys) > 8:
                        recent_keys = recent_keys[-8:]
            ctx[idx] = {
                "prev_summary": prev_summary or "",
                "prev_topic": prev_topic or "",
                "recent_parent_keys": list(recent_keys),
            }
        return ctx

    # Prepare batch jobs per (chapter, mode)
    jobs = []  # list of (future, chapter_id, mode, batch_indices)
    executor = ThreadPoolExecutor(max_workers=max_workers)

    def build_prompt_for_batch(mode: str, chapter_top: str, batch_indices: List[int], ctx_map: Dict[int, Dict[str, object]]) -> str:
        # Build ITEMS payload
        items_payload: List[Dict[str, object]] = []
        for idx in batch_indices:
            it = items[idx]
            c = ctx_map.get(idx, {"prev_summary": "", "prev_topic": "", "recent_parent_keys": []})
            chat_top = _choose_top_chat(it.get("chat_text", ""), k=6)
            payload = {
                "id": it["id"],
                "prev_context": {"prev_topic": c["prev_topic"], "prev_summary": c["prev_summary"]},
                "recent_parent_keys": c["recent_parent_keys"],
                "transcript": (it.get("text") or "")[:1200],
                "chat_top": chat_top,
                "metrics": {
                    "chat_rate_z": float(it.get("chat_rate_z") or 0.0),
                    "burst_score": float(it.get("burst_score") or 0.0),
                    "reactions": it.get("reactions") or {},
                },
            }
            items_payload.append(payload)

        items_json = json.dumps(items_payload, ensure_ascii=False)

        if mode == 'jc':
            prompt = (
                "You are an expert VOD segment grouper for Just Chatting.\n"
                "Return a JSON ARRAY with one object per input (same order).\n"
                "Each object MUST include: id, summary (<=25w), topic (2-5w), energy (low|medium|high), topic_key (1-3w), same_topic_prev (bool), confidence (0-1).\n\n"
                f"CHAPTER_TOP: \"{chapter_top}\"\n\n"
                f"ITEMS: {items_json}\n\n"
                "OUTPUT: JSON array only."
            )
        else:
            prompt = (
                "You are an expert VOD segment grouper for Gameplay.\n"
                "Return a JSON ARRAY with one object per input (same order).\n"
                "Each object MUST include: id, summary (<=25w), topic (2-5w), energy (low|medium|high), topic_key (1-3w), role (intro|build_up|conflict|peak|resolution|filler|afk), confidence (0-1).\n\n"
                f"CHAPTER_TOP: \"{chapter_top}\"\n\n"
                f"ITEMS: {items_json}\n\n"
                "OUTPUT: JSON array only."
            )
        return prompt

    def submit_batch(chapter_id: str, mode: str, batch_indices: List[int]):
        ch_meta = chapters.get(chapter_id, {})
        chapter_top = get_chapter_transcript(ch_meta, segments)
        # Build per-chapter prev context map
        # Use the full chapter index ordering to compute prevs
        ch_idxs = chap_to_indices.get(str(chapter_id) if chapter_id is not None else "", [])
        ctx_map = build_prev_context_arrays(ch_idxs)
        prompt = build_prompt_for_batch(mode, chapter_top, batch_indices, ctx_map)
        # Conservative token cap per batch
        max_toks = min(300 * len(batch_indices), 4000)
        future = executor.submit(call_llm, prompt, max_toks, 0.0, 60, f"batch_{chapter_id}_{mode}_{batch_indices[0]}")
        jobs.append((future, chapter_id, mode, list(batch_indices)))

    # Enqueue batches
    for cid, idxs in chap_to_indices.items():
        # Filter selected for this chapter
        ch_selected = [i for (c, i) in selected_pairs if c == cid]
        if not ch_selected:
            continue
        # Split by mode for stable requirements
        jc_idxs = [i for i in ch_selected if items[i].get("mode") == 'jc']
        gm_idxs = [i for i in ch_selected if items[i].get("mode") != 'jc']
        # Chunk and submit
        for bucket, mode in ((jc_idxs, 'jc'), (gm_idxs, 'game')):
            if not bucket:
                continue
            for s in range(0, len(bucket), batch_size):
                submit_batch(cid, mode, bucket[s:s+batch_size])

    # Collect results
    id_to_payload: Dict[str, Dict] = {}
    for future, chapter_id, mode, batch_indices in jobs:
        try:
            raw = future.result()
        except Exception:
            raw = ""
        # Append explicit JSON only instruction retry if parse fails
        parsed = _extract_json_array(raw)
        if not parsed:
            # one retry with stricter instruction
            ch_meta = chapters.get(chapter_id, {})
            chapter_top = get_chapter_transcript(ch_meta, segments)
            # Rebuild prompt with extra line
            # minimal rebuild to avoid duplicating logic
            # Using same ctx_map
            ch_idxs = chap_to_indices.get(str(chapter_id) if chapter_id is not None else "", [])
            ctx_map = {idx: {
                "prev_summary": (items[ch_idxs[k-1]].get("summary") if k-1 >= 0 else "") or (((items[ch_idxs[k-1]].get("text") or "")[-220:]) if k-1 >= 0 else ""),
                "prev_topic": (items[ch_idxs[k-1]].get("topic_key") or "") if k-1 >= 0 else "",
                "recent_parent_keys": []
            } for k, idx in enumerate(ch_idxs)}
            prompt = (build_prompt_for_batch(mode, chapter_top, batch_indices, ctx_map) + "\nRespond with JSON array only.")
            try:
                raw2 = call_llm(prompt, min(300 * len(batch_indices), 4000), 0.0, 60, f"batch_retry_{chapter_id}_{mode}_{batch_indices[0]}")
            except Exception:
                raw2 = ""
            parsed = _extract_json_array(raw2)

        # Map back to ids in order
        if isinstance(parsed, list):
            for idx, obj in zip(batch_indices, parsed):
                if isinstance(obj, dict) and obj.get("id"):
                    id_to_payload[str(obj["id"])]= obj

    # Final pass: update DB with deterministic topic_thread per chapter
    updated = 0
    for cid, idxs in chap_to_indices.items():
        current_thread = 0
        prev_topic_key = ""
        for k, idx in enumerate(idxs):
            it = items[idx]
            is_jc = it.get("mode") == 'jc'
            payload = id_to_payload.get(it["id"])
            # If not selected for update, keep existing labels but recompute thread relative to existing same_topic_prev if available
            effective_summary = it.get("summary")
            effective_topic_key = it.get("topic_key")
            effective_energy = None
            effective_role = it.get("role")
            same_topic_prev_hint = it.get("same_topic_prev")
            link_type = ""
            link_evidence = ""
            confidence = float(it.get("confidence") or 0.0)

            if (cid, idx) in selected_pairs and isinstance(payload, dict):
                # Validate
                req = ["summary","topic","topic_key","energy","same_topic_prev"] if is_jc else ["summary","topic","topic_key","energy","role"]
                if all(r in payload for r in req) and str(payload.get("summary","")) and str(payload.get("topic_key","")):
                    effective_summary = str(payload.get("summary",""))[:200]
                    effective_topic_key = str(payload.get("topic_key",""))[:64].lower().strip()
                    effective_energy = str(payload.get("energy",""))
                    effective_role = str(payload.get("role","")) if not is_jc else ""
                    same_topic_prev_hint = bool(payload.get("same_topic_prev")) if is_jc else same_topic_prev_hint
                    link_type = str(payload.get("link_type",""))
                    link_evidence = str(payload.get("link_evidence",""))[:120]
                    confidence = float(payload.get("confidence") or 0.0)
                else:
                    # fallback from metrics
                    effective_summary = (it.get("text") or "")[:200]
                    if not effective_topic_key:
                        kws = extract_keywords(it.get("text") or "") or []
                        effective_topic_key = (kws[0].lower() if kws else (effective_topic_key or ""))
                    if same_topic_prev_hint is None:
                        same_topic_prev_hint = bool(prev_topic_key and effective_topic_key and (prev_topic_key in effective_topic_key or effective_topic_key in prev_topic_key))
                    if not effective_energy:
                        effective_energy = _energy_from_metrics(float(it.get("chat_rate_z") or 0.0), float(it.get("burst_score") or 0.0))
            else:
                # not updated; ensure energy present for consistency
                if not effective_energy:
                    effective_energy = _energy_from_metrics(float(it.get("chat_rate_z") or 0.0), float(it.get("burst_score") or 0.0))

            # Compute thread
            if k == 0:
                current_thread = 1
            else:
                same_flag = bool(same_topic_prev_hint)
                current_thread = current_thread if same_flag else (current_thread + 1)

            # Write updates: labels only for selected; topic_thread for all
            if (cid, idx) in selected_pairs:
                cur.execute(
                    "UPDATE documents SET summary=?, topic=?, energy=?, role=?, role_confidence=?, same_topic_prev=?, topic_thread=?, topic_key=?, link_type=?, link_evidence=?, confidence=? WHERE id=?",
                    (
                        (effective_summary or ""),
                        str(payload.get("topic") if (payload and payload.get("topic")) else ("")),
                        (effective_energy or ""),
                        (effective_role or ""),
                        float(confidence if (effective_role or "") else 0.0),
                        1 if bool(same_topic_prev_hint) else 0,
                        int(current_thread),
                        (effective_topic_key or ""),
                        link_type,
                        link_evidence,
                        float(confidence or 0.0),
                        it["id"],
                    ),
                )
                updated += 1
            else:
                # Update only topic_thread to keep continuity consistent
                cur.execute(
                    "UPDATE documents SET topic_thread=? WHERE id=?",
                    (int(current_thread), it["id"]),
                )
        conn.commit()

    conn.close()
    print(f"✅ Burst summaries updated (batched). ({updated} rows)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PASS 1: Burst summary labeling")
    parser.add_argument("vod_id", help="VOD ID")
    # Concurrency/batching flags (opt-in; default preserves existing sequential behavior)
    parser.add_argument("--concurrent", action="store_true", help="Enable batched concurrent summarization")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("BURST_SUMMARY_BATCH_SIZE", "12")),
        help="Items per LLM request when concurrent mode is enabled (env BURST_SUMMARY_BATCH_SIZE)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.getenv("BURST_SUMMARY_MAX_WORKERS", "6")),
        help="Max concurrent LLM requests (env BURST_SUMMARY_MAX_WORKERS)",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        default=(os.getenv("BURST_SUMMARY_ONLY_MISSING", "true").lower() in ("1", "true", "yes")),
        help="Process only rows with missing/low-confidence labels (env BURST_SUMMARY_ONLY_MISSING)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of bursts to process")
    parser.add_argument("--offset", type=int, default=0, help="Optional offset for processing window")

    args = parser.parse_args()

    # Enable concurrency by default unless explicitly disabled via env
    env_concurrent = os.getenv("BURST_SUMMARY_CONCURRENT", "true").lower() in ("1", "true", "yes")
    if (args.concurrent or env_concurrent) and args.batch_size > 1:
        _update_burst_labels_batched(
            vod_id=args.vod_id,
            batch_size=max(2, args.batch_size),
            max_workers=max(1, args.max_workers),
            only_missing=bool(args.only_missing),
            limit=args.limit,
            offset=args.offset,
        )
    else:
        update_burst_labels(args.vod_id)


if __name__ == "__main__":
    main()
