#!/usr/bin/env python3
"""
Extract webcam crops around reaction anchors for a VOD.

Steps (minimal test harness):
- Load reaction-based anchors via existing vector_store utilities
- For each anchor: probe/download a tiny local MP4 segment around (t-2s..t+1s) using
  existing directors_cut.downloader if available; else read from full VOD path if present
- Run YOLO webcam locator to obtain a stable cam box
- Grab a single frame just before the anchor (t-0.5s), crop the cam box, save JPG

Outputs:
  data/thumbnails/<vod_id>/cams/<label>_<idx>_<time_ms>.jpg

Usage:
  python -m thumbnail.extract_cam_crops <vod_id> [--limit 20] [--source <path>]
                                         [--stride 1.0] [--pre 0.5]
                                         [--mode reactions|peak]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore


_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


from vector_store.generate_clips_manifest import (  # type: ignore
    load_docs,
    compute_peak_scores,
    select_anchor_indices,
    _format_hms,
)


def _load_sponsor_spans(vod_id: str):
    try:
        from rag.enhanced_director_cut_selector import load_atomic_segments  # type: ignore
        return load_atomic_segments(vod_id)
    except Exception:
        return []


def _load_vod_video_path(vod_id: str) -> Optional[Path]:
    # Try a few common locations from existing pipeline
    candidates = [
        Path(f"data/chunks/{vod_id}/vod.mp4"),
        Path(f"data/raw/{vod_id}.mp4"),
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def _extract_frame_at(video_path: Path, time_s: float) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if fps <= 0:
            fps = 30.0
        frame_index = max(0, int(round(time_s * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            return None
        return frame
    finally:
        cap.release()


def _run_yolo_cam_box(temp_video: Path):
    from clip_creation.yolo_face_locator import analyze_with_yolo  # type: ignore
    dec = analyze_with_yolo(temp_video, enable_logs=False)
    box = dec.crops.get("cam") if isinstance(dec.crops, dict) else None
    return box


def _write_jpg(img: np.ndarray, out_path: Path, quality: int = 92) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ok = cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return bool(ok)
    except Exception:
        return False


def _crop(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    H, W = img.shape[:2]
    x0 = max(0, min(W, x))
    y0 = max(0, min(H, y))
    x1 = max(0, min(W, x + w))
    y1 = max(0, min(H, y + h))
    if x1 <= x0 or y1 <= y0:
        return img
    return img[y0:y1, x0:x1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract webcam crops at reaction anchors")
    parser.add_argument("vod_id", help="VOD ID")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--mode", choices=["reactions", "peak"], default="reactions")
    parser.add_argument("--pre", type=float, default=0.5, help="Seconds before anchor to snapshot")
    parser.add_argument("--source", default=None, help="Explicit source video path (mp4)")
    args = parser.parse_args()

    vod_id = str(args.vod_id)
    docs = load_docs(vod_id)
    if not docs:
        print("No documents loaded")
        return

    sponsor_spans = _load_sponsor_spans(vod_id)
    peak_scores = compute_peak_scores(docs, sponsor_spans)

    use_reactions = (args.mode == "reactions")
    indices = select_anchor_indices(docs, peak_scores, use_reaction_hits=use_reactions)
    if not indices:
        print("No anchors found")
        return

    # Resolve source video
    video_path = Path(args.source) if args.source else _load_vod_video_path(vod_id)
    if not video_path or not video_path.exists():
        print("X No source video found; provide --source path")
        return

    out_dir = Path(f"data/thumbnails/{vod_id}/cams")
    taken = 0
    for i in indices:
        if taken >= max(1, int(args.limit)):
            break
        d = docs[i]
        anchor_t = float(d.start)  # using window start as anchor proxy
        t_snap = max(0.0, anchor_t - float(args.pre))

        # Run YOLO on a very small temp segment by writing a subclip file if needed.
        # For simplicity in this minimal harness, we analyze the whole video, since
        # yolo_face_locator handles frame sampling internally.
        temp_video = video_path
        box = _run_yolo_cam_box(temp_video)
        if not box:
            print(f"- [{i}] no webcam box; skipping")
            continue

        frame = _extract_frame_at(video_path, t_snap)
        if frame is None:
            print(f"- [{i}] no frame at {t_snap:.2f}s; skipping")
            continue

        crop = _crop(frame, int(box.x), int(box.y), int(box.width), int(box.height))
        rel_ms = int(round(t_snap * 1000.0))
        label = "react" if use_reactions else "peak"
        out_path = out_dir / f"{label}_{i:05d}_{rel_ms}.jpg"
        ok = _write_jpg(crop, out_path, quality=92)
        if ok:
            print(f"✓ wrote {out_path}")
            taken += 1
        else:
            print(f"X failed to write {out_path}")


if __name__ == "__main__":
    main()


