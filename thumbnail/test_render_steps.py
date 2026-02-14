#!/usr/bin/env python3
"""
Step-by-step thumbnail render test for arcs.

Steps:
 1) background -> output
 2) background + 1 cam -> output
 3) background + 1 cam + title -> output

Usage:
  python -m thumbnail.test_render_steps <vod_id> --arc 1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _load_arc(vod_id: str, arc_idx: int) -> Optional[Dict]:
    p = Path(f"data/vector_stores/{vod_id}/arcs/arc_{int(arc_idx):03d}_manifest.json")
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        obj["_path"] = str(p)
        return obj
    except Exception:
        return None


def _load_cams_index(vod_id: str) -> List[Dict]:
    idx_path = Path(f"data/thumbnails/{vod_id}/cams_index.json")
    if not idx_path.exists():
        return []
    try:
        data = json.loads(idx_path.read_text(encoding="utf-8"))
        return data.get("items") or []
    except Exception:
        return []


def _select_cam(cams: List[Dict], vod_id: str, arc_idx: int, s_abs: int, e_abs: int) -> Optional[Path]:
    def fn(c: Dict) -> str:
        return c.get("filename") or ""
    pool = [c for c in cams if f"arc_{arc_idx:03d}_" in fn(c)]
    if not pool:
        pool = [c for c in cams if f"_{s_abs}-{e_abs}_" in fn(c)]
    if not pool:
        pool = list(cams)
    if not pool:
        return None
    pool.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
    p = Path(pool[0].get("path"))
    return p if p.exists() else None


def _ensure_arc_bg(vod_id: str, arc_idx: int, s_abs: int, e_abs: int) -> Optional[Path]:
    temp_dir = Path("data/temp/arc_cam_snaps") / vod_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    # Try existing arc temp
    for p in sorted(temp_dir.glob(f"arc_{arc_idx:03d}_*.mp4")):
        out = temp_dir / f"bg_arc_{arc_idx:03d}.jpg"
        if out.exists() and out.stat().st_size > 0:
            return out
        ffmpeg = os.getenv("FFMPEG_PATH") or os.getenv("FFMPEG_BIN") or "ffmpeg"
        subprocess.run([ffmpeg, "-y", "-ss", "1.0", "-i", str(p), "-frames:v", "1", str(out)], capture_output=True)
        if out.exists() and out.stat().st_size > 0:
            return out
    # Download 3s at midpoint
    try:
        from cam_detection.downloader import download_small_chunk_1080p  # type: ignore
        mid = max(0, (int(s_abs) + int(e_abs)) // 2)
        chunk = temp_dir / f"arc_{arc_idx:03d}_{mid-1}_{mid+2}.mp4"
        ok = download_small_chunk_1080p(vod_id, max(0, mid - 1.0), 3.0, chunk, quality=os.getenv("THUMB_DL_QUALITY", "1080p"))
        if ok and chunk.exists():
            out = temp_dir / f"bg_arc_{arc_idx:03d}.jpg"
            ffmpeg = os.getenv("FFMPEG_PATH") or os.getenv("FFMPEG_BIN") or "ffmpeg"
            subprocess.run([ffmpeg, "-y", "-ss", "1.5", "-i", str(chunk), "-frames:v", "1", str(out)], capture_output=True)
            if out.exists() and out.stat().st_size > 0:
                return out
    except Exception:
        pass
    return None


def _raster_title(title: str, out_png: Path) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception:
        return False
    W, H = 640, 280
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font_path = os.getenv("THUMB_FONT_PATH", "C:/Windows/Fonts/impact.ttf")
    try:
        font = ImageFont.truetype(font_path, size=90)
    except Exception:
        font = ImageFont.load_default()
    text = (title or "").strip()[:48]
    if not text:
        text = "Title"
    # Two-line naive wrap
    words = text.split()
    line1, line2 = "", ""
    for w in words:
        t = (line1 + " " + w).strip()
        try:
            box = draw.textbbox((0, 0), t, font=font)
            tw = box[2] - box[0]
        except Exception:
            tw = len(t) * 20
        if tw <= W:
            line1 = t
        else:
            line2 = (line2 + " " + w).strip()
    y = 0
    for line in [line1, line2]:
        if not line:
            continue
        draw.text((4, y + 4), line, font=font, fill=(0, 0, 0, 200))
        for dx, dy in [(-3, 0), (3, 0), (0, -3), (0, 3)]:
            draw.text((dx, y + dy), line, font=font, fill=(0, 0, 0, 255))
        draw.text((0, y), line, font=font, fill=(255, 255, 255, 255))
        y += int(getattr(font, "size", 90) * 1.15)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    return True


def _run(cmd: List[str], dbg_path: Optional[Path] = None) -> bool:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if dbg_path:
            try:
                dbg_path.write_text("\n".join(["CMD=" + " ".join(cmd), "STDOUT=", res.stdout or "", "STDERR=", res.stderr or ""]), encoding="utf-8")
            except Exception:
                pass
        return res.returncode == 0
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Step-by-step arc thumbnail test")
    parser.add_argument("vod_id", help="VOD ID")
    parser.add_argument("--arc", type=int, required=True)
    args = parser.parse_args()

    vod_id = str(args.vod_id)
    arc = _load_arc(vod_id, int(args.arc))
    if not arc:
        print("Arc manifest not found")
        return
    s_abs = int(float(arc.get("start_abs", 0)))
    e_abs = int(float(arc.get("end_abs", 0)))
    cams = _load_cams_index(vod_id)
    cam_path = _select_cam(cams, vod_id, int(args.arc), s_abs, e_abs)
    if not cam_path:
        print("No cam image found; run extract_arc_cam_crops and rate_cams first")
        return
    bg = _ensure_arc_bg(vod_id, int(args.arc), s_abs, e_abs)
    if not bg:
        print("No background frame available")
        return

    out_root = Path(f"data/thumbnails/{vod_id}/arch_debug")
    out_root.mkdir(parents=True, exist_ok=True)

    # Step 1: background -> output (dim + scale)
    step1 = out_root / f"arc_{int(args.arc):03d}_step1_bg.jpg"
    ffmpeg = os.getenv("FFMPEG_PATH") or os.getenv("FFMPEG_BIN") or "ffmpeg"
    f1 = "scale=1280:720:force_original_aspect_ratio=increase,format=rgb24,colorchannelmixer=rr=0.85:gg=0.85:bb=0.85,format=yuv420p"
    cmd1 = [ffmpeg, "-y", "-i", str(bg), "-vf", f1, "-update", "1", "-frames:v", "1", str(step1)]
    _run(cmd1, out_root / f"arc_{int(args.arc):03d}_step1_dbg.txt")
    print(f"Step1 -> {step1}")

    # Step 2: background + 1 cam -> output
    step2 = out_root / f"arc_{int(args.arc):03d}_step2_bg_cam.jpg"
    filter2 = (
        "[0:v]scale=1280:720:force_original_aspect_ratio=increase,format=rgb24,colorchannelmixer=rr=0.85:gg=0.85:bb=0.85[bg];"
        "[1:v]scale=-2:380,format=rgb24[cam];"
        "[bg][cam]overlay=820:220,format=yuv420p[v]"
    )
    cmd2 = [ffmpeg, "-y", "-i", str(step1), "-i", str(cam_path), "-filter_complex", filter2, "-map", "[v]", "-update", "1", "-frames:v", "1", str(step2)]
    _run(cmd2, out_root / f"arc_{int(args.arc):03d}_step2_dbg.txt")
    print(f"Step2 -> {step2}")

    # Step 3: background + 1 cam + title -> output
    step3 = out_root / f"arc_{int(args.arc):03d}_step3_full.jpg"
    title_png = out_root / f"arc_{int(args.arc):03d}_title.png"
    _ = _raster_title(str(arc.get("title") or vod_id), title_png)
    filter3 = (
        "[0:v]format=rgb24[bg];"
        "[1:v]format=rgb24[cam];"
        f"[bg][cam]overlay=820:220[tmp];[tmp][2:v]overlay=40:40,format=yuv420p[v]"
    )
    cmd3 = [ffmpeg, "-y", "-i", str(step2), "-i", str(cam_path), "-i", str(title_png), "-filter_complex", filter3, "-map", "[v]", "-update", "1", "-frames:v", "1", str(step3)]
    _run(cmd3, out_root / f"arc_{int(args.arc):03d}_step3_dbg.txt")
    print(f"Step3 -> {step3}")


if __name__ == "__main__":
    main()


