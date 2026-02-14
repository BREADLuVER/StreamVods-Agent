#!/usr/bin/env python3
"""
Extract webcam cam crops for arc thumbnails when no clips are present.

For each arc under data/vector_stores/<vod_id>/arcs:
  - Pick an anchor (midpoint of arc span)
  - Download a small segment around anchor (TwitchDownloaderCLI)
  - Run YOLO webcam locator once to get cam box (unless just chatting/irl)
  - Emit multiple snapshots at anchor + offsets (e.g., -1,-0.5,0,0.5,1,2)

Outputs:
  - Normal arcs: data/thumbnails/<vod_id>/cams/
  - Just chatting/IRL arcs: data/thumbnails/<vod_id>/jc_cams/

Usage:
  python -m thumbnail.extract_arc_cam_crops <vod_id> [--arc 1] [--pre 1.0] [--post 2.0]
                                             [--quality 1080p]
                                             [--offsets "-1,-0.5,0,0.5,1,2"]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from thumbnail.arc_utils import is_just_chatting_arc


def _load_arc_manifests(vod_id: str, only_arc: Optional[int]) -> List[Path]:
    root = Path(f"data/vector_stores/{vod_id}/arcs")
    if not root.exists():
        return []
    if only_arc is not None:
        p = root / f"arc_{int(only_arc):03d}_manifest.json"
        return [p] if p.exists() else []
    return sorted([p for p in root.glob("arc_*_manifest.json") if p.is_file()])


def _choose_anchor(arc_man: Dict[str, object]) -> float:
    # Prefer midpoint of arc span
    try:
        s = float(arc_man.get("start_abs", 0.0))
        e = float(arc_man.get("end_abs", 0.0))
        if e > s:
            return 0.5 * (s + e)
    except Exception:
        pass
    # Fallback: first range midpoint
    try:
        ranges = arc_man.get("ranges") or []
        if isinstance(ranges, list) and ranges:
            r0 = ranges[0] or {}
            rs = float(r0.get("start", 0.0))
            re = float(r0.get("end", 0.0))
            if re > rs:
                return 0.5 * (rs + re)
    except Exception:
        pass
    return 0.0


def _download_chunk(vod_id: str, start_s: float, dur_s: float, out_path: Path, quality: str) -> bool:
    try:
        from cam_detection.downloader import download_small_chunk_1080p  # type: ignore
    except Exception:
        return False
    return bool(download_small_chunk_1080p(vod_id, start_s, dur_s, out_path, quality=quality))


def _detect_cam_box(video_path: Path):
    try:
        from clip_creation.yolo_face_locator import analyze_with_yolo  # type: ignore
    except Exception:
        return None
    dec = analyze_with_yolo(video_path, enable_logs=False)
    return getattr(dec, "crops", {}).get("cam") if dec else None


def _snapshot_offsets_env() -> List[float]:
    raw = os.getenv("SNAP_OFFSETS", "").strip()
    if raw:
        try:
            return [float(x) for x in raw.split(",") if x.strip()]
        except Exception:
            return [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    # More samples for better selection
    return [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


def _extract_full_frame(video_path: Path, time_offset: float, out_path: Path) -> bool:
    """Extract a full frame at given offset without cropping. For just chatting/IRL."""
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False
        
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if fps <= 0:
            fps = 30.0
        
        frame_idx = max(0, int(round(time_offset * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        
        if not ok or frame is None:
            return False
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        return bool(ok) and out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract arc webcam crops for thumbnails")
    parser.add_argument("vod_id", help="VOD ID")
    parser.add_argument("--arc", type=int, default=None, help="Only process specific arc index")
    parser.add_argument("--pre", type=float, default=1.0, help="Seconds before anchor to include in temp chunk")
    parser.add_argument("--post", type=float, default=2.0, help="Seconds after anchor to include in temp chunk")
    parser.add_argument("--quality", default="1080p")
    args = parser.parse_args()

    vod_id = str(args.vod_id)
    manifests = _load_arc_manifests(vod_id, args.arc)
    if not manifests:
        print("No arc manifests found. Generate arcs first.")
        return

    out_root = Path(f"data/thumbnails/{vod_id}/cams")
    out_root.mkdir(parents=True, exist_ok=True)

    temp_root = Path("data/temp/arc_cam_snaps") / vod_id
    temp_root.mkdir(parents=True, exist_ok=True)

    offsets = _snapshot_offsets_env()

    for mp in manifests:
        try:
            man = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            continue
        arc_idx = int(man.get("arc_index") or 0)
        s_abs = float(man.get("start_abs", 0.0))
        e_abs = float(man.get("end_abs", 0.0))
        anchor = _choose_anchor(man)
        start = max(0.0, anchor - float(args.pre))
        dur = float(args.pre) + float(args.post)
        temp_path = temp_root / f"arc_{arc_idx:03d}_{int(start)}_{int(start+dur)}.mp4"

        if not temp_path.exists() or temp_path.stat().st_size == 0:
            ok = _download_chunk(vod_id, start, dur, temp_path, args.quality)
            if not ok:
                print(f"X download failed for arc {arc_idx}")
                continue

        # Check if this is a just chatting/IRL arc
        is_jc = is_just_chatting_arc(vod_id, man)
        
        if is_jc:
            # Just chatting/IRL: extract full frames without YOLO
            jc_out_dir = Path(f"data/thumbnails/{vod_id}/jc_cams")
            jc_out_dir.mkdir(parents=True, exist_ok=True)
            
            name_hint = f"arc_{arc_idx:03d}_{int(s_abs)}-{int(e_abs)}"
            count = 0
            for rel in offsets:
                snap_time = start + float(args.pre) + rel  # relative to chunk start
                out_path = jc_out_dir / f"{name_hint}_off{rel:.1f}.jpg"
                if _extract_full_frame(temp_path, snap_time - start, out_path):
                    count += 1
            print(f"✓ arc {arc_idx:03d} (just chatting): {count} full frames extracted")
        else:
            # Normal arc: YOLO + crop
            cam_box = _detect_cam_box(temp_path)
            if not cam_box:
                print(f"- arc {arc_idx:03d}: no webcam detected; skipping")
                continue

            # Emit snapshots at anchor + offsets
            try:
                from thumbnail.cam_snapshots import snapshot_from_cam_box  # type: ignore
                name_hint = f"arc_{arc_idx:03d}_{int(s_abs)}-{int(e_abs)}"
                for rel in offsets:
                    _ = snapshot_from_cam_box(
                        vod_id=vod_id,
                        clip_path=temp_path,
                        start_time=start,
                        end_time=start + dur,
                        cam_box=cam_box,
                        anchor_time=anchor,
                        rel_offset_s=float(rel),
                        name_hint=name_hint,
                    )
                print(f"✓ arc {arc_idx:03d}: cam crops extracted")
            except Exception:
                print(f"X arc {arc_idx:03d}: snapshot error")


if __name__ == "__main__":
    main()


