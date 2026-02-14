#!/usr/bin/env python3
"""
Render randomized arch thumbnails using rated cam crops and arc manifests.

Inputs:
  - data/vector_stores/<vod_id>/arcs/arc_*_manifest.json
  - data/thumbnails/<vod_id>/cams_index.json (from rate_cams)

Outputs:
  - data/thumbnails/<vod_id>/arch/arc_{idx:03d}_v{n}.jpg
  - sidecars with decisions (arc_{idx:03d}_v{n}.json)

Run:
  python -m thumbnail.render_arch_thumbnails 2568891328 --variants 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from thumbnail.arc_utils import is_just_chatting_arc


# ========== Data loading (unchanged) ==========

def _load_arcs(vod_id: str, only_arc: Optional[int]) -> List[Dict]:
    root = Path(f"data/vector_stores/{vod_id}/arcs")
    if not root.exists():
        return []
    items: List[Dict] = []
    paths = [root / f"arc_{int(only_arc):03d}_manifest.json"] if only_arc else sorted(root.glob("arc_*_manifest.json"))
    for p in paths:
        if not p.exists():
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            obj["_path"] = str(p)
            items.append(obj)
        except Exception:
            continue
    return items


def _load_cams_index(vod_id: str, index_name: str = "cams_index.json") -> List[Dict]:
    """Load cam index (either normal cams or just chatting cams)."""
    idx_path = Path(f"data/thumbnails/{vod_id}/{index_name}")
    if not idx_path.exists():
        return []
    try:
        data = json.loads(idx_path.read_text(encoding="utf-8"))
        return data.get("items") or []
    except Exception:
        return []


def _parse_source_arc_idx(filename: str) -> Optional[int]:
    """Parse source arc index from cam filename (e.g., 'cam_arc_006_...' -> 6)."""
    import re
    m = re.match(r"cam_arc_(\d+)_", filename)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    # For just chatting full frames: arc_001_240-3004_off-1.0.jpg
    m = re.match(r"arc_(\d+)_", filename)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _select_cam_global(cams: List[Dict], used_names: set[str], used_arc_sources: set[int], current_arc_idx: int) -> Optional[Dict]:
    """
    Pick highest-scoring cam with smart allocation:
    1. Hasn't been used yet (by filename)
    2. Prefer cams from adjacent arcs (current ± 1)
    3. Avoid cams from distant arcs that are already used
    """
    if not cams:
        return None
    ordered = sorted(cams, key=lambda r: float(r.get("score", 0.0)), reverse=True)
    
    # First pass: look for adjacent arcs (current ± 1)
    for c in ordered:
        name = str(c.get("filename") or "")
        if not name or name in used_names:
            continue
        
        source_arc = _parse_source_arc_idx(name)
        if source_arc is not None:
            # Prefer adjacent arcs (current ± 1)
            if abs(source_arc - current_arc_idx) <= 1:
                return c
    
    # Second pass: any unused cam
    for c in ordered:
        name = str(c.get("filename") or "")
        if not name or name in used_names:
            continue
        
        source_arc = _parse_source_arc_idx(name)
        if source_arc is not None and source_arc in used_arc_sources:
            continue
        
        return c
    
    # Third pass: allow reuse if we're out of options
    for c in ordered:
        name = str(c.get("filename") or "")
        if not name or name in used_names:
            continue
        return c
    
    return None


def _get_chapter_safe_name(filename: str) -> Optional[str]:
    """Extract chapter safe name from filename if possible."""
    # cam_I_Actually_Cant_Believe_We_Hit_The_Jackpot_20657-20713_0ms_20685000.jpg
    # pattern: cam_<NAME>_<FRAMES>_<MS>_<TS>.jpg
    parts = filename.split('_')
    if len(parts) > 4 and parts[0] == "cam":
        # Heuristic: Find the part that looks like frame range (e.g. 20657-20713)
        for i, p in enumerate(parts):
            if '-' in p and p.replace('-', '').isdigit():
                # Found the frame range, everything before (except cam_) is the name
                return "_".join(parts[1:i])
    return None


def _load_metadata_title(vod_id: str, arc_idx: int) -> Optional[str]:
    """Try to load thumbnail hook or title from youtube_metadata.json"""
    try:
        path = Path(f"data/ai_data/{vod_id}/{vod_id}_arc_{arc_idx:03d}_youtube_metadata.json")
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            # Prefer specifically generated thumbnail text (hook)
            hook = data.get("streamsniped_metadata", {}).get("thumbnail_text")
            if hook and isinstance(hook, str) and len(hook.strip()) > 1:
                return hook.strip()
            
            # Fallback to YouTube title if no hook
            return data.get("snippet", {}).get("title")
    except Exception:
        pass
    return None




# ========== Title raster with smart highlighting ==========

def _normalize_for_matching(text: str) -> str:
    """Normalize text for fuzzy matching: lowercase, remove special chars, collapse spaces."""
    import re
    t = text.lower()
    t = re.sub(r'[_\s\-+&]+', ' ', t)  # Replace separators with space
    t = re.sub(r'[^\w\s]', '', t)  # Remove special chars
    t = ' '.join(t.split())  # Collapse multiple spaces
    return t


def _get_stream_context(vod_id: str) -> dict:
    """Load stream context for smart highlighting."""
    try:
        from src.config import config
        sc_path = config.get_ai_data_dir(vod_id) / f"{vod_id}_stream_context.json"
        if sc_path.exists():
            import json
            data = json.loads(sc_path.read_text(encoding="utf-8"))
            cats = data.get("chapter_categories", [])
            categories = [str(c).strip() for c in cats if str(c).strip()] if isinstance(cats, list) else []
            
            # Build normalized mapping for fuzzy matching
            # Map: normalized_form -> original_display_name
            category_map = {}
            for cat in categories:
                normalized = _normalize_for_matching(cat)
                if normalized:
                    # Split into words for partial matching
                    words = normalized.split()
                    # Store full normalized form
                    category_map[normalized] = cat
                    # Also store each word separately for partial matching
                    for word in words:
                        if len(word) >= 3:  # Only meaningful words
                            if word not in category_map:
                                category_map[word] = cat
            
            return {
                "streamer": str(data.get("streamer") or data.get("streamer_name") or "").strip(),
                "categories": categories,
                "category_map": category_map,
            }
    except Exception:
        pass
    return {"streamer": "", "categories": [], "category_map": {}}


def _word_to_color(word: str) -> tuple[int, int, int]:
    """Generate deterministic but random-feeling color from word hash."""
    BRIGHT_COLORS = [
        (57, 255, 20),      # neon green
        (255, 20, 147),     # hot pink
        (255, 255, 0),      # electric yellow
        (255, 165, 0),      # neon orange
        (0, 255, 255),      # cyan
        (255, 0, 255),      # magenta
        (191, 255, 0),      # lime
        (255, 105, 180),    # hot pink 2
    ]
    # Use hash for deterministic randomness
    import hashlib
    h = int(hashlib.md5(word.lower().encode()).hexdigest(), 16)
    return BRIGHT_COLORS[h % len(BRIGHT_COLORS)]


def _raster_title(
    title: str, 
    out_png: Path, 
    max_w: int, 
    max_h: int, 
    is_just_chatting: bool = False, 
    vod_id: str = "",
    chapter_colors: Optional[Dict[str, tuple[int, int, int]]] = None,
) -> bool:
    """Render title with smart highlighting and styling."""
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception:
        return False

    STROKE = 10  # Thicker border for impact
    PAD = 10

    # Get stream context for smart highlighting
    context = _get_stream_context(vod_id) if vod_id else {"streamer": "", "categories": [], "category_map": {}}
    streamer = context.get("streamer", "").lower()
    categories = [c.lower() for c in context.get("categories", [])]
    category_map = context.get("category_map", {})  # normalized -> original
    
    # Use provided chapter colors or fall back to hash-based colors
    chapter_colors = chapter_colors or {}

    # Font setup
    font_path = os.getenv("THUMB_FONT_PATH", "").strip()
    if not font_path:
        for p in (
            "C:/Windows/Fonts/impact.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ):
            if Path(p).exists():
                font_path = p
                break
    
    base_size = int(os.getenv("THUMB_FONT_SIZE_JC", "80")) if is_just_chatting else int(os.getenv("THUMB_FONT_SIZE", "130"))
    
    # Scale up for short, punchy hooks
    if len(title) < 20:
        base_size = int(base_size * 1.4)
    elif len(title) < 30:
        base_size = int(base_size * 1.2)
    
    try:
        font_normal = ImageFont.truetype(font_path, size=base_size) if font_path else ImageFont.load_default()
        # For hooks, we usually want uniform sizing, but we can keep emphasis logic if needed
        # Actually for "I MESSED UP", we probably want uniform big impact font.
        font_streamer = font_normal
        font_game = font_normal
    except Exception:
        font_normal = font_streamer = font_game = ImageFont.load_default()

    # Parse title into styled segments
    segments = []
    
    # Split title into words (no forced line breaks)
    words = title.strip().split()
    
    # Randomize color strategy for the hook
    # Use deterministic seed based on title hash so it doesn't flicker
    import hashlib
    seed = int(hashlib.md5(title.encode()).hexdigest(), 16)
    r = random.Random(seed)
    
    # Bias strongly towards Yellow/Cyan for hooks (high visibility)
    # 40% Yellow, 40% Cyan, 10% White, 10% Smart
    color_strategy = r.choices(
        ["yellow", "cyan", "white", "smart"],
        weights=[40, 40, 10, 10],
        k=1
    )[0]
    
    print(f"  🎨 Color Strategy: {color_strategy}")
    
    for word in words:
        word_lower = word.lower().strip("'\".,!?")
        word_normalized = _normalize_for_matching(word)
        
        color = (255, 255, 255)
        if color_strategy == "smart":
            if streamer and streamer in word_lower:
                color = _word_to_color(streamer)
            elif word_normalized in category_map:
                matched_cat = category_map[word_normalized]
                color = chapter_colors.get(matched_cat, _word_to_color(matched_cat))
            elif any(cat in word_lower for cat in categories if cat):
                matched_cat = next((cat for cat in categories if cat in word_lower), "")
                color = chapter_colors.get(matched_cat, _word_to_color(matched_cat))
        elif color_strategy == "yellow":
            color = (255, 255, 0)
        elif color_strategy == "cyan":
            color = (0, 255, 255)
            
        segments.append({
            "text": word,
            "font": font_normal,
            "color": color,
            "is_newline_before": False,
        })
    
    if not segments:
        segments = [{"text": title, "font": font_normal, "color": (255, 255, 255), "is_newline_before": False}]
    
    # Render segments with wrapping
    tmp = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    meas = ImageDraw.Draw(tmp)
    
    def measure_text(text: str, font, stroke: int) -> tuple[int, int]:
        x0, y0, x1, y1 = meas.textbbox((0, 0), text, font=font, stroke_width=stroke)
        return (x1 - x0, y1 - y0)
    
    # Build lines with wrapping
    lines = []
    current_line = []
    current_width = 0
    
    # Allow slightly wider text for short hooks
    effective_max_w = max_w if len(title) > 20 else max_w + 100
    
    for seg in segments:
        if seg["is_newline_before"] and current_line:
            lines.append(current_line)
            current_line = []
            current_width = 0
        
        text_with_space = seg["text"] + " "
        seg_w, seg_h = measure_text(text_with_space, seg["font"], STROKE)
        
        if current_line and current_width + seg_w > effective_max_w:
            lines.append(current_line)
            current_line = [seg]
            current_width = seg_w
        else:
            current_line.append(seg)
            current_width += seg_w
    
    if current_line:
        lines.append(current_line)
    
    # Calculate canvas size
    max_line_h = int(base_size * 1.2)
    spacing = int(base_size * 0.1)
    
    # Measure total width of each line to center them
    line_widths = []
    for line in lines:
        w = 0
        for seg in line:
            sw, _ = measure_text(seg["text"] + " ", seg["font"], STROKE)
            w += sw
        line_widths.append(w)
    
    final_w = max(line_widths) + 2 * (PAD + STROKE)
    final_h = len(lines) * max_line_h + (len(lines) - 1) * spacing + 2 * (PAD + STROKE)
    
    img = Image.new("RGBA", (final_w, final_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    y = PAD + STROKE + int(base_size * 1.0) # Baseline adjustment
    for i, line in enumerate(lines):
        # Center align: Calculate starting X for this line
        line_w = line_widths[i]
        x = (final_w - line_w) // 2
        
        for seg in line:
            draw.text((x, y), seg["text"], font=seg["font"],
                     stroke_width=STROKE, stroke_fill=(0, 0, 0, 255),
                     fill=seg["color"], anchor='ls')
            seg_w, _ = measure_text(seg["text"] + " ", seg["font"], STROKE)
            x += seg_w
        y += max_line_h + spacing
    
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    return True


# ========== Dynamic title positioning ==========

def _calculate_title_position(cam_center_x: int, cam_center_y: int, title_w: int, title_h: int) -> Tuple[int, int]:
    """
    Calculate optimal title position based on foreground cam placement.
    Strategy:
    - Identify 'Cam Side' (Left/Right)
    - Place Text on Opposite Side (Center of that side +/- jitter)
    - Vertical Position: Randomized (Top, Center, Bottom)
    - Allow overlap if necessary
    
    Canvas: 1280x720
    """
    CANVAS_W = 1280
    CANVAS_H = 720
    
    # Determine cam side
    is_cam_left = cam_center_x < (CANVAS_W / 2)
    
    # Define Safe Zones for Text
    # If Cam Left -> Text Right (Range 640-1280)
    # If Cam Right -> Text Left (Range 0-640)
    
    if is_cam_left:
        target_center_x = 960  # Center of right half
        safe_min_x = 600
        safe_max_x = 1200
    else:
        target_center_x = 320  # Center of left half
        safe_min_x = 80
        safe_max_x = 680
        
    # Randomize X
    # Center text around target_center_x
    # x = target - w/2
    base_x = target_center_x - (title_w // 2)
    
    # Add jitter but keep in safe bounds
    jitter = random.randint(-50, 50)
    x = base_x + jitter
    
    # Clamp to safe zones (soft clamp)
    x = max(safe_min_x, min(x, safe_max_x - title_w))
    
    # Vertical Positioning
    # 3 modes: Top-Heavy, Center, Bottom-Heavy
    v_mode = random.choice(["top", "center", "center", "bottom"]) # bias to center
    
    if v_mode == "top":
        y = random.randint(40, 150)
    elif v_mode == "bottom":
        y = CANVAS_H - title_h - random.randint(40, 150)
    else:
        # Center vertically
        y = (CANVAS_H - title_h) // 2 + random.randint(-40, 40)
        
    # Ensure within bounds
    x = max(20, min(CANVAS_W - title_w - 20, x))
    y = max(20, min(CANVAS_H - title_h - 20, y))
    
    return (int(x), int(y))


def _get_title_dimensions(title_png: Path) -> Tuple[int, int]:
    """Get width and height of rendered title PNG."""
    try:
        from PIL import Image  # type: ignore
        img = Image.open(title_png)
        return img.size
    except Exception:
        return (800, 120)  # fallback estimate


# ========== Simple layouts ==========

@dataclass
class Layout:
    cam_h: int
    cam_xy: Tuple[int, int]
    title_xy: Tuple[int, int]
    flip_cam: bool = False

# 1280x720 canvas. No styling, just placements.
SIMPLE_LAYOUTS: List[Layout] = [
    Layout(cam_h=420, cam_xy=(900, 200), title_xy=(40, 40),  flip_cam=False),  # cam right, title left
    Layout(cam_h=420, cam_xy=(40,  200), title_xy=(680, 40), flip_cam=True ),  # cam left,  title right
    Layout(cam_h=360, cam_xy=(900, 320), title_xy=(60,  40), flip_cam=False),  # cam bottom-right
    Layout(cam_h=360, cam_xy=(40,  320), title_xy=(640, 40), flip_cam=True ),  # cam bottom-left
    Layout(cam_h=480, cam_xy=(880, 140), title_xy=(40,  40), flip_cam=False),  # tall cam right
    Layout(cam_h=480, cam_xy=(40,  140), title_xy=(720, 40), flip_cam=True ),  # tall cam left
    Layout(cam_h=400, cam_xy=(760, 260), title_xy=(60,  60), flip_cam=False),  # mid-right cam
    Layout(cam_h=400, cam_xy=(120, 260), title_xy=(640, 60), flip_cam=True ),  # mid-left cam
]

def _pick_layout(seed: int, variant: int) -> Layout:
    """
    Deterministic per (seed, variant) with tiny jitter to avoid identical pixels.
    """
    r = random.Random(seed * 131 + variant * 17)
    base = SIMPLE_LAYOUTS[variant % len(SIMPLE_LAYOUTS)]
    jx = r.randint(-12, 12)
    jy = r.randint(-10, 10)
    tjx = r.randint(-8, 8)
    tjy = r.randint(-6, 6)
    return Layout(
        cam_h=base.cam_h,
        cam_xy=(max(0, min(1230, base.cam_xy[0] + jx)),
                max(0, min(670,  base.cam_xy[1] + jy))),
        title_xy=(max(20, min(1240, base.title_xy[0] + tjx)),
                  max(20, min(700,  base.title_xy[1] + tjy))),
        flip_cam=base.flip_cam,
    )


# ========== FFmpeg composition (opaque, simple) ==========

def _compose_ffmpeg(
    bg_path: Path,
    cam_path: Path,
    title_png: Path,
    out_path: Path,
    *,
    cam_box_xy: Tuple[int, int],
    cam_box_wh: Tuple[int, int],
    title_xy: Tuple[int, int],
    flip_cam: bool = False,
) -> bool:
    """
    - Dim/crop BG to 1280x720 (opaque).
    - Scale cam larger (≥⅓ screen) and center over small bg cam.
    - Overlay cam, then title.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ffmpeg = os.getenv("FFMPEG_PATH") or os.getenv("FFMPEG_BIN") or "ffmpeg"

        flip = ",hflip" if flip_cam else ""
        x, y = int(cam_box_xy[0]), int(cam_box_xy[1])
        cw, ch = int(cam_box_wh[0]), int(cam_box_wh[1])

        # --- scale up the cam to be big ---
        min_h = int(os.getenv("THUMB_CAM_MIN_H", "360"))     # >1/3 screen (increased from 300)
        scale_up = float(os.getenv("THUMB_CAM_SCALE", "2.2"))  # 2.2× the small box (increased from 1.8)
        target_h = max(min_h, int(ch * scale_up))

        # --- filtergraph ---
        graph = (
            # 1. background to 1280x720, dim it, ensure RGBA
            "[0:v]scale=1280:720:force_original_aspect_ratio=increase,"
            "crop=1280:720,format=rgba,"
            "colorchannelmixer=rr=0.82:gg=0.82:bb=0.82:aa=1.0[bg];"
            # 2. enlarge cam to target_h, maintain aspect ratio, ensure RGBA
            f"[1:v]scale=-2:{target_h}{flip},format=rgba[camfit];"
            # 3. blank out old bg webcam area (opaque black)
            f"[bg]drawbox=x={x}:y={y}:w={cw}:h={ch}:color=black:t=fill[bgm];"
            # 4. center big cam over small bg cam
            f"[bgm][camfit]overlay="
            f"'min(max({x}+{cw}/2-w/2,0),W-w)':"
            f"'min(max({y}+{ch}/2-h/2,0),H-h)'[a];"
            # 5. overlay title
            f"[a][2:v]overlay="
            f"'min(max({int(title_xy[0])},0),W-w)':"
            f"'min(max({int(title_xy[1])},0),H-h)',"
            "format=yuv420p[vout]"
        )

        cmd = [
            ffmpeg, "-y",
            "-i", str(bg_path),
            "-i", str(cam_path),
            "-i", str(title_png),
            "-filter_complex", graph,
            "-map", "[vout]",
            "-an",
            "-frames:v", "1",
            "-q:v", "3",
            str(out_path),
        ]

        import subprocess
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print("FFmpeg error:\n", res.stderr)
        return res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0
    except Exception as e:
        print("Compose exception:", e)
        return False


def _compose_ffmpeg_just_chatting(
    bg_path: Path,
    title_png: Path,
    out_path: Path,
    *,
    title_xy: Tuple[int, int],
) -> bool:
    """
    Simplified composition for Just Chatting/IRL arcs:
    - BG to 1280x720 (NO dimming)
    - Title overlay only
    - No cam overlay
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ffmpeg = os.getenv("FFMPEG_PATH") or os.getenv("FFMPEG_BIN") or "ffmpeg"

        # Simplified filtergraph: just scale/crop BG and overlay title
        graph = (
            "[0:v]scale=1280:720:force_original_aspect_ratio=increase,"
            "crop=1280:720,format=rgb24[bg];"
            f"[bg][1:v]overlay="
            f"'min(max({int(title_xy[0])},0),W-w)':"
            f"'min(max({int(title_xy[1])},0),H-h)',"
            "format=yuv420p[vout]"
        )

        cmd = [
            ffmpeg, "-y",
            "-i", str(bg_path),
            "-i", str(title_png),
            "-filter_complex", graph,
            "-map", "[vout]",
            "-an",
            "-frames:v", "1",
            "-q:v", "3",
            str(out_path),
        ]

        import subprocess
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print("FFmpeg error:\n", res.stderr)
        return res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0
    except Exception as e:
        print("Compose exception:", e)
        return False


# ========== BG webcam box detection & caching ==========

def _detect_bg_cam_box(bg_path: Path) -> Optional[Tuple[int, int, int, int]]:
    """Detect webcam box on a BG pre-resized to the same 1280x720 canvas we composite on."""
    try:
        import cv2  # type: ignore
        from clip_creation.structural_detector import StructuralLayoutDetector  # type: ignore
    except Exception:
        return None

    img = cv2.imread(str(bg_path))
    if img is None:
        return None

    # === Reproduce ffmpeg's: scale=1280:720:force_original_aspect_ratio=increase, crop=1280:720 ===
    try:
        th, tw = 720, 1280
        h, w = img.shape[:2]
        # scale so that both dims >= target (cover)
        scale = max(tw / w, th / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # center crop to 1280x720
        x0 = max(0, (nw - tw) // 2)
        y0 = max(0, (nh - th) // 2)
        bg1280 = resized[y0:y0+th, x0:x0+tw].copy()  # (720,1280,3)

        det = StructuralLayoutDetector(enable_logs=False)
        mod = det._ensure_yolo()
        if not mod:
            return None

        res = mod(bg1280, verbose=False, conf=float(os.getenv("YOLO_CONF_BG", "0.45")))[0]
        names = getattr(mod, "names", {}) if hasattr(mod, "names") else {}

        best = None
        best_conf = 0.0
        for xyxy, cls, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
            label = names.get(int(cls), str(int(cls))) if isinstance(names, dict) else str(int(cls))
            if str(label).lower() != "webcam":
                continue
            c = float(conf)
            if c >= best_conf:
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                best = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
                best_conf = c
        return best
    except Exception:
        return None


def _load_cam_box_cache(out_dir: Path) -> Dict[str, List[int]]:
    cache_path = out_dir / "bg_cam_boxes.json"
    try:
        if cache_path.exists():
            obj = json.loads(cache_path.read_text(encoding="utf-8"))
            return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    return {}


def _save_cam_box_cache(out_dir: Path, cache: Dict[str, List[int]]) -> None:
    cache_path = out_dir / "bg_cam_boxes.json"
    try:
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        pass


# ========== BG frame (your grab, accurate -ss) ==========

def _grab_bg_frame(vod_id: str, arc: Dict, arc_idx: int) -> Optional[Path]:
    s = int(float(arc.get("start_abs", 0)))
    e = int(float(arc.get("end_abs", 0)))
    mid_abs = max(0, (s + e) // 2)
    temp_dir = Path("data/temp/arc_cam_snaps") / vod_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Try existing temp files for this arc index
    try:
        for p in sorted(temp_dir.glob(f"arc_{arc_idx:03d}_*.mp4")):
            out = temp_dir / f"bg_arc_{arc_idx:03d}_{mid_abs}.jpg"
            if out.exists() and out.stat().st_size > 0:
                return out
            ffmpeg = os.getenv("FFMPEG_PATH") or os.getenv("FFMPEG_BIN") or "ffmpeg"
            # Accurate seek: -ss after -i
            cmd = [ffmpeg, "-y", "-i", str(p), "-ss", "1.0", "-frames:v", "1", str(out)]
            import subprocess
            subprocess.run(cmd, capture_output=True)
            if out.exists() and out.stat().st_size > 0:
                return out
    except Exception:
        pass

    # Download a tiny chunk around midpoint
    try:
        from cam_detection.downloader import download_small_chunk_1080p  # type: ignore
        chunk = temp_dir / f"arc_{arc_idx:03d}_{mid_abs-1}_{mid_abs+2}.mp4"
        ok = download_small_chunk_1080p(
            vod_id, max(0, mid_abs - 1.0), 3.0, chunk,
            quality=os.getenv("THUMB_DL_QUALITY", "1080p")
        )
        if ok and chunk.exists():
            out = temp_dir / f"bg_arc_{arc_idx:03d}_{mid_abs}.jpg"
            ffmpeg = os.getenv("FFMPEG_PATH") or os.getenv("FFMPEG_BIN") or "ffmpeg"
            cmd = [ffmpeg, "-y", "-i", str(chunk), "-ss", "1.5", "-frames:v", "1", str(out)]
            import subprocess
            subprocess.run(cmd, capture_output=True)
            if out.exists() and out.stat().st_size > 0:
                return out
    except Exception:
        pass
    return None


# ========== Orchestration ==========

def main() -> None:
    parser = argparse.ArgumentParser(description="Render arch thumbnails (simple randomized layouts)")
    parser.add_argument("vod_id", help="VOD ID")
    parser.add_argument("--arc", type=int, default=None)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    vod_id = str(args.vod_id)
    random.seed(int(args.seed))
    arcs = _load_arcs(vod_id, args.arc)
    
    # Load both cam indexes
    normal_cams = _load_cams_index(vod_id, "cams_index.json")
    jc_cams = _load_cams_index(vod_id, "jc_cams_index.json")
    
    if not arcs:
        print("No arcs found. Run arc generation first.")
        return
    
    if not normal_cams and not jc_cams:
        print("No cam indexes found. Run extract_arc_cam_crops and rate_cams first.")
        return

    out_dir = Path(f"data/thumbnails/{vod_id}/arch")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize persistent chapter colors
    from cam_detection.vod_info import TwitchVodInfoProvider
    vod_provider = TwitchVodInfoProvider()
    
    # Extract all chapter names from arcs
    chapter_names = set()
    for arc in arcs:
        ranges = arc.get("ranges", [])
        if ranges:
            ch_name = ranges[0].get("_chapter_name") or ranges[0].get("_chapter_base")
            if ch_name:
                chapter_names.add(str(ch_name).strip())
    
    # Ensure colors exist for all chapters
    chapter_colors = vod_provider.ensure_chapter_colors(vod_id, list(chapter_names))
    print(f"📊 Loaded chapter colors for {len(chapter_colors)} chapters")

    # Track used cams globally across all arcs
    used_names: set[str] = set()
    used_arc_sources: set[int] = set()  # Track which source arcs we've used cams from
    
    for arc in arcs:
        arc_idx = int(arc.get("arc_index") or 0)
        
        # 1. Try metadata title/hook FIRST (Video title logic was flawed for thumbnails)
        # We want the hook, not the full title.
        title = ""
        meta_title = _load_metadata_title(vod_id, arc_idx)
        if meta_title:
            title = meta_title
            
        # 2. Fallback to manifest title if no hook found
        if not title:
            title = str(arc.get("title") or "")
        
        # 3. Fallback to VOD ID
        if not title:
            title = vod_id

        # Remove hashtag and everything after it
        if "#" in title:
            title = title.split("#")[0].strip()
        
        bg = _grab_bg_frame(vod_id, arc, arc_idx)
        
        if not bg:
            print(f"X arc {arc_idx:03d}: bg frame missing")
            continue
        
        # Check if this is a just chatting arc
        is_jc = is_just_chatting_arc(vod_id, arc)
        
        if is_jc:
            # Just chatting: no cam needed, simplified composition
            # Position title at bottom, full width
            title_xy = (0, 600)  # Bottom of 720p frame
            
            for v in range(1, max(1, int(args.variants)) + 1):
                title_png = out_dir / f"arc_{arc_idx:03d}_v{v}_title.png"
                # Use full width for JC titles (1280px - padding), smaller font
                if not _raster_title(title, title_png, max_w=1200, max_h=0, is_just_chatting=True, vod_id=vod_id, chapter_colors=chapter_colors):
                    print(f"X arc {arc_idx:03d}: title gen failed")
                    continue
                
                out = out_dir / f"arc_{arc_idx:03d}_v{v}.jpg"
                ok = _compose_ffmpeg_just_chatting(
                    bg_path=bg,
                    title_png=title_png,
                    out_path=out,
                    title_xy=title_xy,
                )
                
                sidecar = {
                    "vod_id": vod_id,
                    "arc_index": arc_idx,
                    "variant": v,
                    "bg": str(bg),
                    "cam": None,
                    "title": title,
                    "layout": {
                        "type": "just_chatting",
                        "title_xy": title_xy,
                    },
                    "out": str(out),
                }
                (out_dir / f"arc_{arc_idx:03d}_v{v}.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
                print(("✓" if ok else "X"), f"arc {arc_idx:03d} v{v} (JC) -> {out}")
        else:
            # Normal arc: select from normal cams, use full composition
            cam = _select_cam_global(normal_cams, used_names, used_arc_sources, arc_idx)
            if not cam:
                # Fallback to jc_cams if normal cams exhausted
                cam = _select_cam_global(jc_cams, used_names, used_arc_sources, arc_idx)
            
            if not cam:
                print(f"X arc {arc_idx:03d}: no available cams")
                continue
            
            cam_path = Path(cam.get("path"))
            used_name = str(cam.get("filename") or "")
            if used_name:
                used_names.add(used_name)
                # Track the source arc to avoid using similar cams
                source_arc = _parse_source_arc_idx(used_name)
                if source_arc is not None:
                    used_arc_sources.add(source_arc)

            # Detect webcam box on bg (or fallback)
            cam_box_cache = _load_cam_box_cache(out_dir)
            bg_key = str(bg)
            if bg_key in cam_box_cache:
                x, y, w, h = cam_box_cache[bg_key]
            else:
                det = _detect_bg_cam_box(bg)
                if det is not None:
                    x, y, w, h = det
                    cam_box_cache[bg_key] = [x, y, w, h]
                    _save_cam_box_cache(out_dir, cam_box_cache)
                else:
                    x, y, w, h = (900, 220, 320, 320)

            cam_h = max(240, min(520, int(max(h, 240))))
            flip_cam = False

            for v in range(1, max(1, int(args.variants)) + 1):
                cam_box_xy = (int(x), int(y))
                cam_box_wh = (max(60, int(w)), max(60, int(h)))

                # Render title first to get dimensions
                title_png = out_dir / f"arc_{arc_idx:03d}_v{v}_title.png"
                if not _raster_title(title, title_png, max_w=880, max_h=0, is_just_chatting=False, vod_id=vod_id, chapter_colors=chapter_colors):
                    print(f"X arc {arc_idx:03d}: title gen failed")
                    continue
                
                # Calculate foreground cam center position (centered over bg cam box)
                cam_center_x = x + (w // 2)
                cam_center_y = y + (h // 2)
                
                # Get title dimensions and calculate optimal position
                title_w, title_h = _get_title_dimensions(title_png)
                title_xy = _calculate_title_position(cam_center_x, cam_center_y, title_w, title_h)

                out = out_dir / f"arc_{arc_idx:03d}_v{v}.jpg"
                ok = _compose_ffmpeg(
                    bg_path=bg,
                    cam_path=cam_path,
                    title_png=title_png,
                    out_path=out,
                    cam_box_xy=cam_box_xy,
                    cam_box_wh=cam_box_wh,
                    title_xy=title_xy,
                    flip_cam=flip_cam,
                )

                sidecar = {
                    "vod_id": vod_id,
                    "arc_index": arc_idx,
                    "variant": v,
                    "bg": str(bg),
                    "cam": cam,
                    "title": title,
                    "layout": {
                        "type": "normal",
                        "cam_h": cam_h,
                        "cam_center": (cam_center_x, cam_center_y),
                        "title_xy": title_xy,
                        "title_size": (title_w, title_h),
                        "flip_cam": flip_cam,
                    },
                    "out": str(out),
                }
                (out_dir / f"arc_{arc_idx:03d}_v{v}.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
                print(("✓" if ok else "X"), f"arc {arc_idx:03d} v{v} -> {out}")


if __name__ == "__main__":
    main()
