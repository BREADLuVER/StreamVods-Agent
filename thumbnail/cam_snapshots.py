#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _read_frame_at_seconds(video_path: Path, time_s: float):
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if fps <= 0:
            fps = 30.0
        idx = max(0, int(round(time_s * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            return None
        return frame
    finally:
        cap.release()


def _crop(img, x: int, y: int, w: int, h: int):
    if img is None:
        return None
    H, W = img.shape[:2]
    x0 = max(0, min(W, x))
    y0 = max(0, min(H, y))
    x1 = max(0, min(W, x + w))
    y1 = max(0, min(H, h + y))
    if x1 <= x0 or y1 <= y0:
        return img
    return img[y0:y1, x0:x1]


def _write_jpg(img, out_path: Path, quality: int = 92) -> bool:
    try:
        import cv2  # type: ignore
    except Exception:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return bool(cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]))
    except Exception:
        return False


def snapshot_cam_from_clip(
    vod_id: str,
    clip_path: Path,
    *,
    clip_index: int,
    start_time: float,
    end_time: float,
    anchor_time: Optional[float] = None,
) -> Optional[Path]:
    """Create a webcam crop snapshot near the anchor from a downloaded clip.

    If webcam not detected, optionally save a fallback full-frame screenshot when
    THUMBNAIL_FALLBACK_SCREENSHOT=1.
    """
    try:
        from clip_creation.yolo_face_locator import analyze_with_yolo  # type: ignore
    except Exception:
        return None

    if not clip_path or not clip_path.exists() or clip_path.stat().st_size == 0:
        return None

    # Analyze downloaded segment (YOLO samples frames internally)
    decision = analyze_with_yolo(clip_path, enable_logs=False)
    has_cam = bool(getattr(decision, 'crops', {}).get('cam'))

    # Choose snapshot time: a little before anchor if available, else mid-clip
    pre_s = 0.5
    if anchor_time is not None:
        snap_abs = max(float(start_time), float(anchor_time) - pre_s)
    else:
        snap_abs = float(start_time) + max(0.0, (float(end_time) - float(start_time)) * 0.5)
    snap_rel = max(0.0, min(float(end_time) - float(start_time) - 0.05, snap_abs - float(start_time)))

    frame = _read_frame_at_seconds(clip_path, snap_rel)
    if frame is None:
        return None

    out_root = Path(f"data/thumbnails/{vod_id}/cams")
    out_name = f"clip_{int(clip_index):03d}_{int(round(snap_abs * 1000.0))}.jpg"
    out_path = out_root / out_name

    if has_cam:
        cam = decision.crops['cam']
        crop = _crop(frame, int(cam.x), int(cam.y), int(cam.width), int(cam.height))
        ok = _write_jpg(crop, out_path, quality=92)
        return out_path if ok else None

    # Fallback: optional full-frame screenshot
    if os.getenv('THUMBNAIL_FALLBACK_SCREENSHOT', '').lower() in ('1', 'true', 'yes'):
        if _write_jpg(frame, out_path, quality=88):
            return out_path
    return None


def snapshot_from_cam_box(
    vod_id: str,
    clip_path: Path,
    *,
    start_time: float,
    end_time: float,
    cam_box,
    anchor_time: Optional[float] = None,
    rel_offset_s: Optional[float] = None,
    name_hint: Optional[str] = None,
) -> Optional[Path]:
    """Snapshot using an existing cam box (no YOLO call)."""
    if not clip_path or not clip_path.exists() or clip_path.stat().st_size == 0:
        return None
    # Choose absolute snapshot time
    if anchor_time is not None and rel_offset_s is not None:
        snap_abs = float(anchor_time) + float(rel_offset_s)
        # clamp within [start_time, end_time)
        if snap_abs < float(start_time):
            snap_abs = float(start_time)
        if snap_abs > float(end_time) - 0.05:
            snap_abs = float(end_time) - 0.05
    elif anchor_time is not None:
        pre_s = 0.5
        snap_abs = max(float(start_time), float(anchor_time) - pre_s)
    else:
        snap_abs = float(start_time) + max(0.0, (float(end_time) - float(start_time)) * 0.5)
    snap_rel = max(0.0, min(float(end_time) - float(start_time) - 0.05, snap_abs - float(start_time)))
    frame = _read_frame_at_seconds(clip_path, snap_rel)
    if frame is None:
        return None
    crop = _crop(frame, int(cam_box.x), int(cam_box.y), int(cam_box.width), int(cam_box.height))
    out_root = Path(f"data/thumbnails/{vod_id}/cams")
    # Include clip window in name for robust grouping
    base = (name_hint or "clip") + f"_{int(float(start_time))}-{int(float(end_time))}"
    suffix = f"_{int(round((rel_offset_s or 0.0)*1000.0))}ms" if rel_offset_s is not None else ""
    out_name = f"cam_{base}{suffix}_{int(round(snap_abs * 1000.0))}.jpg"
    out_path = out_root / out_name
    ok = _write_jpg(crop, out_path, quality=92)
    return out_path if ok else None


def snapshot_full_frame(
    vod_id: str,
    clip_path: Path,
    *,
    start_time: float,
    end_time: float,
    anchor_time: Optional[float] = None,
    name_hint: Optional[str] = None,
) -> Optional[Path]:
    """Write a full-frame snapshot (no cam) near anchor or midpoint."""
    if not clip_path or not clip_path.exists() or clip_path.stat().st_size == 0:
        return None
    pre_s = 0.5
    if anchor_time is not None:
        snap_abs = max(float(start_time), float(anchor_time) - pre_s)
    else:
        snap_abs = float(start_time) + max(0.0, (float(end_time) - float(start_time)) * 0.5)
    snap_rel = max(0.0, min(float(end_time) - float(start_time) - 0.05, snap_abs - float(start_time)))
    frame = _read_frame_at_seconds(clip_path, snap_rel)
    if frame is None:
        return None
    out_root = Path(f"data/thumbnails/{vod_id}/cams")
    base = name_hint or f"{int(start_time)}-{int(end_time)}"
    out_name = f"full_{base}_{int(round(snap_abs * 1000.0))}.jpg"
    out_path = out_root / out_name
    ok = _write_jpg(frame, out_path, quality=88)
    return out_path if ok else None


