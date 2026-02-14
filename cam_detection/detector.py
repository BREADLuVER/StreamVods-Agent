#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

from .downloader import download_small_chunk_1080p
from .vod_info import TwitchVodInfoProvider


def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _get_vod_duration_seconds(vod_id: str) -> int:
    """Best-effort fetch of VOD duration using Twitch Helix API.

    Requires TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET in env to obtain an app token.
    Returns 0 on failure.
    """
    import requests  # lazy import

    client_id = os.getenv("TWITCH_CLIENT_ID", "").strip()
    client_secret = os.getenv("TWITCH_CLIENT_SECRET", "").strip()
    print(f"Twitch API credentials: client_id={'***' if client_id else 'None'}, client_secret={'***' if client_secret else 'None'}")
    if not client_id or not client_secret:
        print("Missing Twitch API credentials")
        return 0

    try:
        print(f"Requesting Twitch API token for VOD {vod_id}...")
        token_resp = requests.post(
            "https://id.twitch.tv/oauth2/token",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "client_credentials",
            },
            timeout=10,
        )
        print(f"Token response status: {token_resp.status_code}")
        token_resp.raise_for_status()
        access_token = token_resp.json().get("access_token", "")
        if not access_token:
            print("No access token received")
            return 0
        print("Got access token, fetching VOD info...")

        video_resp = requests.get(
            "https://api.twitch.tv/helix/videos",
            params={"id": vod_id},
            headers={
                "Client-Id": client_id,
                "Authorization": f"Bearer {access_token}",
            },
            timeout=10,
        )
        print(f"VOD API response status: {video_resp.status_code}")
        video_resp.raise_for_status()
        data = video_resp.json()
        print(f"VOD API response data: {data}")
        items = (data or {}).get("data") or []
        if not items:
            print("No VOD data found in API response")
            return 0
        # Duration format like '3h4m5s' or '45m12s'
        dur_str = (items[0] or {}).get("duration") or ""
        total = 0
        num = ""
        for ch in dur_str:
            if ch.isdigit():
                num += ch
            else:
                if ch == 'h':
                    total += int(num or 0) * 3600
                elif ch == 'm':
                    total += int(num or 0) * 60
                elif ch == 's':
                    total += int(num or 0)
                num = ""
        return int(total)
    except Exception:
        return 0


def _evenly_spaced_samples(duration_s: int, count: int, margin_s: int = 60) -> List[int]:
    if duration_s <= 0:
        return [60, 300, 600][: count or 3]
    if count <= 1:
        return [min(max(1, margin_s), max(1, duration_s - margin_s))]
    
    # Avoid early 10 minutes (600s) and late 5 minutes (300s) to skip intro/outro screens
    early_margin = 600  # 10 minutes
    late_margin = 300   # 5 minutes
    
    start = max(margin_s, early_margin)
    end = max(start + 1, duration_s - late_margin)
    
    if end <= start:
        # If VOD is too short, just use the middle
        return [max(1, duration_s // 2)]
    
    step = max(1, (end - start) // (count - 1))
    return [start + i * step for i in range(count)]


def _choose_segments(duration_s: int, samples: int, window_s: int) -> List[Tuple[int, int]]:
    positions = _evenly_spaced_samples(duration_s, samples)
    segs: List[Tuple[int, int]] = []
    for p in positions:
        s = max(0, p - (window_s // 2))
        e = min(max(1, duration_s), s + window_s)
        if e <= s:
            e = s + max(1, window_s)
        segs.append((s, e))
    return segs


def _cleanup_temp_files(temp_root: Path) -> None:
    """Clean up temporary files after analysis."""
    try:
        if temp_root.exists():
            import shutil
            shutil.rmtree(temp_root)
            print(f"Cleaned up temp files: {temp_root}")
    except Exception as e:
        print(f"Warning: Failed to clean up temp files {temp_root}: {e}")


def detect_webcam_in_vod(vod_id: str) -> bool:
    """Gate decision: keep VOD when it's IRL/JC or gameplay with a webcam.

    Chapter-first semantics:
    - If any chapter category indicates IRL/Just Chatting → KEEP (return True)
    - Else (game-only): decide by webcam overlay majority
      - Sample N small 1080p segments (default: N=4, window=6s)
      - YOLO majority vote: if detected in >=K segments → KEEP
      - Otherwise → SKIP (return False)
    """
    provider = TwitchVodInfoProvider()
    chapters = provider.get_vod_chapters(vod_id)
    info = provider.get_vod_info(vod_id)
    try:
        print(f"[Gate] Loaded chapters: {[c.get('category') for c in chapters]}")
        print(f"[Gate] VOD duration (info): {info.get('duration')}s")
    except Exception:
        pass

    def _is_irl_like(text: str) -> bool:
        s = (text or '').lower()
        return any(k in s for k in [
            'just_chatting', 'justchatting', 'irl', 'travel', 'outdoors', 'in_real_life', 'talk_shows', 'talkshows'
        ])

    # Gate 1: any IRL/JC-like chapter → keep
    for ch in chapters:
        if _is_irl_like(str(ch.get('category', ''))):
            print("[Gate] KEEP: IRL/Just Chatting chapter found")
            return True

    # Do not use title/game heuristics; rely on chapters + YOLO only
    samples = _get_env_int("WEBCAM_DET_SAMPLES", 4)
    window_s = _get_env_int("WEBCAM_DET_WINDOW_S", 6)
    majority_k = _get_env_int("WEBCAM_DET_MAJORITY_K", max(2, (samples // 2) + 1))

    # Compute segments
    duration_s = int(info.get('duration') or 0) or _get_vod_duration_seconds(vod_id)
    print(f"[Gate] VOD {vod_id} duration resolved: {duration_s}s")
    segments = _choose_segments(duration_s, samples, window_s)
    print(f"Selected segments: {segments}")

    temp_root = Path("data") / "temp" / "webcam_probe" / str(vod_id)
    temp_root.mkdir(parents=True, exist_ok=True)

    positive = 0
    tried = 0

    # Lazy import analyzer to keep startup light
    from clip_creation.yolo_face_locator import analyze_with_yolo

    for s, e in segments:
        out = temp_root / f"{vod_id}_{int(s)}_{int(e)}.mp4"
        if not out.exists():
            print(f"Downloading segment {s}-{e}s for VOD {vod_id}...")
            # Try download with retry
            ok = False
            for attempt in range(2):  # Try twice
                if attempt > 0:
                    print(f"Retry attempt {attempt + 1} for segment {s}-{e}s")
                ok = download_small_chunk_1080p(vod_id, s, e - s, out, quality=os.getenv("WEBCAM_DET_QUALITY", "1080p"))
                if ok:
                    break
                else:
                    print(f"Download attempt {attempt + 1} failed for segment {s}-{e}s")
            if not ok:
                print(f"Failed to download segment {s}-{e}s after retries")
                continue
            print(f"Downloaded segment to {out}")
        
        # Check if file is valid before processing
        if not out.exists() or out.stat().st_size == 0:
            print(f"Segment file {out} is missing or empty")
            continue
            
        tried += 1
        print(f"Analyzing segment {s}-{e}s with YOLO...")
        # YOLO analyzer samples multiple frames internally
        decision = analyze_with_yolo(out, enable_logs=False)
        if decision and getattr(decision, "layout", "") != "gameplay":
            positive += 1
            print(f"Webcam detected in segment {s}-{e}s")
        else:
            print(f"No webcam in segment {s}-{e}s")
        if positive >= majority_k:
            # Clean up temp files before early return
            _cleanup_temp_files(temp_root)
            print("[Gate] KEEP: YOLO majority satisfied")
            return True

    # If insufficient tries due to errors, keep to avoid false skips
    if tried < max(1, samples // 2):
        print(f"[Gate] KEEP: insufficient segments analyzed ({tried} < {max(1, samples // 2)})")
        _cleanup_temp_files(temp_root)
        return True
    
    result = positive >= majority_k
    
    # Clean up temp files after analysis
    _cleanup_temp_files(temp_root)
    
    if result:
        print("[Gate] KEEP: YOLO majority result true")
    else:
        print("[Gate] SKIP: YOLO majority result false and no IRL chapters")
    return result


def detect_chat_in_vod(vod_id: str, *, samples: Optional[int] = None, window_s: Optional[int] = None) -> bool:
    """Detect on-stream chat using a single LLM-vision vote for determinism.

    Procedure:
      1) Download one short segment (middle of VOD by default).
      2) Extract a few frames (start ~0.5s, middle, end-0.5s).
      3) Ask an LLM-vision model to answer strictly in JSON: {"has_chat": bool}.
      4) Parse robustly; on parsing failure, default to False.
    """
    # Local import to avoid heavy deps at module load
    try:
        from src.ai_client import call_llm_vision  # prefer vision for frame analysis
    except Exception:
        call_llm_vision = None  # type: ignore

    provider = TwitchVodInfoProvider()
    info = provider.get_vod_info(vod_id)
    # Deterministic single-vote: choose one window (middle)
    _samples = max(1, int(samples) if samples is not None else 1)
    _window = int(window_s if window_s is not None else _get_env_int("CHAT_DET_WINDOW_S", 8))

    duration_s = int(info.get('duration') or 0) or _get_vod_duration_seconds(vod_id)
    # Pick a single central segment deterministically
    mid = max(1, duration_s // 2)
    s = max(0, mid - (_window // 2))
    e = min(max(1, duration_s), s + _window)

    temp_root = Path("data") / "temp" / "chat_probe" / str(vod_id)
    temp_root.mkdir(parents=True, exist_ok=True)
    seg_path = temp_root / f"{vod_id}_{int(s)}_{int(e)}.mp4"
    try:
        if not seg_path.exists():
            ok = download_small_chunk_1080p(vod_id, s, e - s, seg_path, quality=os.getenv("WEBCAM_DET_QUALITY", "1080p"))
            if not ok:
                return False

        # Extract a few frames from the segment
        images: List[Tuple[str, bytes, str]] = []
        try:
            import cv2  # type: ignore
            cap = cv2.VideoCapture(str(seg_path))
            if cap.isOpened():
                fps = max(1.0, cap.get(cv2.CAP_PROP_FPS) or 30.0)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or int(2 * fps))
                idxs = [
                    int(max(0, round(0.5 * fps))),
                    total // 2,
                    int(max(0, min(total - 1, total - int(round(0.5 * fps)))))
                ]
                for i, idx in enumerate(idxs):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        continue
                    try:
                        ok2, buf = cv2.imencode('.png', frame)
                        if ok2:
                            images.append((f"frame_{i+1}.png", bytes(buf), "image/png"))
                    except Exception:
                        continue
            try:
                cap.release()
            except Exception:
                pass
        except Exception:
            images = []

        if not call_llm_vision or not images:
            # Cannot perform LLM-vision; be conservative
            return False

        prompt = (
            "You are analyzing frames from a Twitch VOD. Determine if an on-stream chat overlay is visible.\n"
            "On-stream chat usually appears as a tall, narrow column of chat messages aligned to the far right or left,\n"
            "with distinct message bubbles or lines scrolling vertically. Ignore watermark overlays, HUD, or subtitles.\n"
            "Respond ONLY with strict minified JSON: {\"has_chat\": true|false}. No extra text."
        )

        try:
            text = call_llm_vision(prompt, images, max_tokens=30, temperature=0.0, request_tag="chat_detection")
        except Exception:
            text = ""
        if not text:
            return False

        # Robust parsing: try JSON, then heuristic boolean extraction
        has_chat: Optional[bool] = None
        try:
            import json as _json
            obj = _json.loads(text)
            val = obj.get("has_chat")
            if isinstance(val, bool):
                has_chat = val
            elif isinstance(val, str):
                has_chat = val.strip().lower() in ("true", "yes", "1")
        except Exception:
            # Heuristic extraction
            t = text.strip().lower()
            if 'has_chat' in t:
                if 'true' in t and 'false' not in t:
                    has_chat = True
                elif 'false' in t and 'true' not in t:
                    has_chat = False

        if has_chat is None:
            # Final conservative fallback
            return False
        return bool(has_chat)
    finally:
        _cleanup_temp_files(temp_root)


def cli_main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Webcam presence detector for Twitch VOD")
    parser.add_argument("vod_id", help="Twitch VOD ID")
    parser.add_argument("--json", action="store_true", help="Print JSON result")
    args = parser.parse_args()

    has_cam = detect_webcam_in_vod(str(args.vod_id))
    # Also probe on-stream chat using YOLO chat weights
    try:
        has_chat = detect_chat_in_vod(str(args.vod_id))
    except Exception:
        has_chat = False
    if args.json:
        print(json.dumps({
            "vod_id": str(args.vod_id),
            "has_webcam": bool(has_cam),
            "has_chat": bool(has_chat),
        }))
    else:
        print(f"{args.vod_id}: {'WEBCAM' if has_cam else 'NO-CAM'}, {'CHAT' if has_chat else 'NO-CHAT'}")


if __name__ == "__main__":
    cli_main()


