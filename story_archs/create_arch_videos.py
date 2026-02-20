#!/usr/bin/env python3
"""
Create arc videos (no transitions) from arc manifests.

For each accepted arc under data/vector_stores/<vod_id>/arcs, this CLI:
  - Downloads the exact video ranges via the existing downloader (TwitchDownloaderCLI under the hood)
  - Optionally overlays chat using the same renderer/compose pipeline as director's cut
  - Concatenates segments losslessly (copy) into a single MP4 per arc

Usage:
  python -m story_archs.create_arch_videos <vod_id> [--arc 1] [--quality 1080p] [--max-workers 4]
                                            [--no-chat] [--chat-w 150] [--chat-h 200] [--chat-margin 24]
                                            [--chat-head-start 10] [--output-dir <dir>] [--use-existing]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


def _format_hms(sec: float) -> str:
    s = int(max(0, round(float(sec))))
    h = s // 3600
    m = (s % 3600) // 60
    s2 = s % 60
    return f"{h:02d}:{m:02d}:{s2:02d}"


def _load_arc_manifests(vod_id: str, arc_index: Optional[int]) -> List[Path]:
    arcs_dir = Path(f"data/vector_stores/{vod_id}/arcs")
    if not arcs_dir.exists():
        return []
    if arc_index is not None:
        p = arcs_dir / f"arc_{int(arc_index):03d}_manifest.json"
        return [p] if p.exists() else []
    # Load all arc manifests
    items = sorted([p for p in arcs_dir.glob("arc_*_manifest.json") if p.is_file()])
    return items


def _hms_to_seconds(hms: str) -> Optional[int]:
    try:
        parts = hms.strip().split(':')
        if len(parts) != 3:
            return None
        h, m, s = [int(p) for p in parts]
        return max(0, h * 3600 + m * 60 + s)
    except Exception:
        return None


def _load_arc_times(vod_id: str, arc_index: Optional[int]) -> List[Tuple[int, int, int]]:
    """Read arcs_vod_times.txt if present.

    Returns a list of tuples: (arc_index, start_sec, end_sec)
    If a specific arc_index is requested, only returns that one when present.
    """
    path = Path(f"data/vector_stores/{vod_id}/arcs/arcs_vod_times.txt")
    if not path.exists():
        return []
    out: List[Tuple[int, int, int]] = []
    import re
    for line in (path.read_text(encoding='utf-8', errors='ignore').splitlines()):
        m = re.search(r"Arc\s+(\d+):\s+(\d{2}:\d{2}:\d{2})\s*->\s*(\d{2}:\d{2}:\d{2})", line)
        if not m:
            continue
        idx = int(m.group(1))
        if arc_index is not None and idx != int(arc_index):
            continue
        s = _hms_to_seconds(m.group(2))
        e = _hms_to_seconds(m.group(3))
        if s is None or e is None or e <= s:
            continue
        out.append((idx, s, e))
    out.sort(key=lambda t: t[0])
    return out


def _parse_times(path: Path) -> Optional[Tuple[int, int]]:
    try:
        name = path.stem  # e.g., seg_0001_123-456 or seg_0001_123-456_chat
        m = __import__('re').search(r"_(\d+)-(\d+)(?:_chat)?$", name)
        if not m:
            return None
        return (int(m.group(1)), int(m.group(2)))
    except Exception:
        return None


def _find_chat_color_mask(chat_base: Path) -> Tuple[Optional[Path], Optional[Path]]:
    color = chat_base.with_name(chat_base.stem + "_raw.mp4")
    mask_a = color.with_name(color.stem + "_mask" + color.suffix)
    mask_b = color.with_name(color.stem + ".mask" + color.suffix)
    mask = mask_a if mask_a.exists() else (mask_b if mask_b.exists() else None)
    return (color if color.exists() else None, mask)


def _concat_copy(paths: List[Path], output: Path, timeout: Optional[int] = None) -> bool:
    try:
        from directors_cut.render import _concat_copy as _cc  # type: ignore
        return bool(_cc(paths, output, timeout=timeout))
    except Exception:
        return False


def _concat_best(paths: List[Path], output: Path, timeout: Optional[int] = None) -> bool:
    """Try stream copy first, then re-encode with high-quality settings if needed.

    This avoids quality losses when simple concat fails due to mismatched
    container parameters, while still being fast when copy works.
    """
    try:
        from directors_cut.render import _concat_copy as _cc, _concat_encode as _ce  # type: ignore
    except Exception:
        return _concat_copy(paths, output, timeout=timeout)
    
    # Try stream copy first (fastest, lossless, best for sync if single file)
    if _cc(paths, output, timeout=timeout):
        return True
    
    # NVENC fallback first, then CPU
    if _ce(paths, output, use_nvenc=True, timeout=timeout):
        return True
    return _ce(paths, output, use_nvenc=False, timeout=timeout)


def _download_segments_for_ranges(vod_id: str, ranges: List[Dict[str, Any]], quality: str, max_workers: int) -> List[Path]:
    try:
        from directors_cut.downloader import plan_segments_from_ranges, download_segments  # type: ignore
    except Exception as e:
        print(f"X downloader unavailable: {e}")
        return []
    # Allow concurrent multi-part downloads for speed; configurable via env
    # DEFAULT CHANGED: 300 -> 14400 (4 hours) to prevent audio desync from stitching small chunks
    # We prefer single continuous downloads for arcs to maintain sync.
    seg_cap = 0
    try:
        seg_cap = int(os.getenv('ARC_DL_SEG_SEC', '14400'))
    except Exception:
        seg_cap = 14400
    # Cap at 12 hours just in case
    seg_cap = max(30, min(43200, seg_cap))
    segments = plan_segments_from_ranges(ranges, vod_id, max_segment_seconds=int(seg_cap))
    paths = download_segments(segments, vod_id, quality=quality, max_workers=max_workers)
    return [p for p in paths if p and p.exists() and p.stat().st_size > 0]


def _detect_onstream_chat(segment_path: Path, conf_thresh: float = 0.25) -> bool:
    """Detect on-stream chat using YOLO weights under weights/chat_detector.(pt|onnx).

    Strategy:
      - Load YOLO once per process (cached on function attribute)
      - Read a few frames (start ~0.5s, middle, end-0.5s)
      - If any frame has a 'chat' detection, return True
    """
    # Resolve weights
    weights: Optional[Path] = None
    for cand in [
        Path("weights/chat_detector.pt"),
        Path("weights/chat_detector.onnx"),
    ]:
        if cand.exists():
            weights = cand
            break
    if weights is None:
        return False

    # Lazy-load YOLO model
    try:
        if not hasattr(_detect_onstream_chat, "_model") or getattr(_detect_onstream_chat, "_model") is None:
            from ultralytics import YOLO  # type: ignore
            setattr(_detect_onstream_chat, "_model", YOLO(str(weights)))
        model = getattr(_detect_onstream_chat, "_model")
    except Exception:
        return False

    # OpenCV sampling
    try:
        import cv2  # type: ignore
    except Exception:
        return False

    cap = cv2.VideoCapture(str(segment_path))
    if not cap.isOpened():
        return False
    try:
        fps = max(1.0, cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or int(2 * fps))
        # sample ~0.5s, middle, end-0.5s
        idxs = [
            int(max(0, round(0.5 * fps))),
            total // 2,
            int(max(0, min(total - 1, total - int(round(0.5 * fps)))))
        ]
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            try:
                res = model(frame, verbose=False, conf=float(conf_thresh))[0]
                names = getattr(model, "names", {}) if hasattr(model, "names") else {}
                for xyxy, cls, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
                    label = names.get(int(cls), str(int(cls))) if isinstance(names, dict) else str(int(cls))
                    if str(label).lower() == "chat" and float(conf) >= float(conf_thresh):
                        return True
            except Exception:
                # Continue to next frame if inference fails
                continue
        return False
    finally:
        cap.release()


def _overlay_chat_for_segments(vod_id: str, segments: List[Path], canvas_w: int, canvas_h: int, chat_w: int, chat_h: int, chat_margin: int, chat_head: int) -> List[Path]:
    if not segments:
        return []
    # Chat overlay prerequisites
    try:
        from chat_overlay.chat_json import ensure_full_chat_json, write_chat_subset  # type: ignore
        from chat_overlay.renderer import render_chat_segment  # type: ignore
        from chat_overlay.compose import overlay_chat_on_video  # type: ignore
    except Exception as e:
        print(f"⚠️  Chat overlay modules unavailable: {e}")
        return segments

    # Probe the first segment resolution if canvas not provided
    def _probe_resolution(p: Path) -> Tuple[int, int]:
        try:
            ffprobe = 'ffprobe'
            if os.name == 'nt' and (Path('executables/ffmpeg.exe')).exists():
                ffprobe = str(Path('executables/ffmpeg.exe')).replace('ffmpeg.exe', 'ffprobe.exe')
            result = __import__('subprocess').run([
                ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height',
                '-of', 'csv=p=0:s=x', str(segments[0])
            ], capture_output=True, text=True, timeout=10)
            out = (result.stdout or '').strip()
            if 'x' in out:
                w_str, h_str = out.split('x')
                w = max(2, int(w_str))
                h = max(2, int(h_str))
                return (w, h)
        except Exception:
            pass
        return (1920, 1080)

    first_w, first_h = _probe_resolution(segments[0])
    canvas_w = canvas_w or first_w
    canvas_h = canvas_h or first_h
    x = max(0, canvas_w - chat_w - chat_margin)
    y = max(0, (canvas_h - chat_h) // 2)

    # Prepare chat json
    full_chat_dir = Path(f"data/chats/{vod_id}")
    chat_dir = Path(f"data/chunks/{vod_id}/arcs_chat")
    chat_dir.mkdir(parents=True, exist_ok=True)
    full_json = ensure_full_chat_json(vod_id, full_chat_dir)
    if not (full_json and full_json.exists()):
        print("⚠️  Full chat JSON unavailable; skipping chat overlay")
        return segments

    # Group contiguous segments by filename times
    parsed: List[Tuple[Path, int, int]] = []
    for p in segments:
        t = _parse_times(p)
        if not t:
            parsed.append((p, 0, 0))
        else:
            parsed.append((p, t[0], t[1]))
    blocks: List[Tuple[int, int]] = []
    cur_s: Optional[int] = None
    cur_e: Optional[int] = None
    for _, s, e in parsed:
        if cur_s is None:
            cur_s, cur_e = s, e
            continue
        if s == cur_e:
            cur_e = e
        else:
            blocks.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    if cur_s is not None and cur_e is not None:
        blocks.append((cur_s, cur_e))

    # Pre-render chat for each block window and overlay per segment
    block_assets: Dict[Tuple[int, int], Tuple[Optional[Path], Optional[Path]]] = {}
    for bstart, bend in blocks:
        subset_json = chat_dir / f"chat_{bstart}_{bend}.json"
        if not (subset_json.exists() and subset_json.stat().st_size > 10):
            ok_sub = write_chat_subset(full_json, subset_json, bstart, bend, head_sec=float(chat_head))
            if not ok_sub:
                print(f"⚠️  Failed to prepare chat subset for block {bstart}-{bend}")
                continue
        chat_out = chat_dir / f"chat_{bstart}_{bend}_{chat_w}x{chat_h}.webm"
        cached_color, cached_mask = _find_chat_color_mask(chat_out)
        if not (chat_out.exists() and chat_out.stat().st_size > 10 and cached_color):
            try:
                ok = render_chat_segment(
                    vod_id,
                    float(bstart),
                    float(bend),
                    chat_out,
                    chat_w=chat_w,
                    chat_h=chat_h,
                    head_start_sec=chat_head,
                    message_hex="#BFBFBF",
                    bg_hex="#00000000",
                    alt_bg_hex="#00000000",
                    chat_json_override=subset_json,
                )
                if not ok:
                    print(f"⚠️  Block chat render failed for {bstart}-{bend}")
                    continue
            except Exception:
                print(f"⚠️  Block chat render exception for {bstart}-{bend}")
                continue
        color, mask = _find_chat_color_mask(chat_out)
        block_assets[(bstart, bend)] = (color if color else Path(), mask)

    overlaid: List[Path] = []
    for p, s, e in parsed:
        # find covering block
        blk = None
        for b in blocks:
            if b[0] <= s and e <= b[1]:
                blk = b
                break
        if blk is None or blk not in block_assets or not block_assets[blk][0]:
            overlaid.append(p)
            continue
        color, mask = block_assets[blk]
        offset = max(0, (s - blk[0]) + chat_head)
        out_seg = p.with_name(p.stem + "_chat" + p.suffix)
        if out_seg.exists() and out_seg.stat().st_size > 0:
            overlaid.append(out_seg)
            continue
        try:
            ok2 = overlay_chat_on_video(
                segment_path=p,
                output_path=out_seg,
                chat_color_path=color,
                chat_mask_path=mask if (mask and mask.exists()) else None,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                chat_w=chat_w,
                chat_h=chat_h,
                pos=(x, y),
                trim_preroll_sec=int(offset),
                use_nvenc=True,
            )
            overlaid.append(out_seg if ok2 else p)
        except Exception:
            overlaid.append(p)
    return overlaid


def main() -> None:
    parser = argparse.ArgumentParser(description="Create arc videos (no transitions)")
    parser.add_argument("vod_id", help="VOD ID")
    parser.add_argument("--arc", type=int, default=None, help="Only process a specific arc index (e.g., 1)")
    parser.add_argument("--quality", default="1080p60")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--no-chat", dest="chat", action="store_false", help="Disable chat overlay")
    parser.add_argument("--chat-w", type=int, default=250)
    parser.add_argument("--chat-h", type=int, default=300)
    parser.add_argument("--chat-margin", type=int, default=16)
    parser.add_argument("--chat-head-start", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--use-existing", action="store_true", help="Reuse already-downloaded segments if present")
    parser.set_defaults(chat=True)
    args = parser.parse_args()

    vod_id = args.vod_id
    # Honor orchestrator-level chat disable flag
    try:
        if os.getenv('ARC_NO_CHAT', '').lower() in ('1', 'true', 'yes'):
            args.chat = False
            print("🛈 ARC_NO_CHAT=1 → disabling chat overlay")
    except Exception:
        pass
    arc_paths = _load_arc_manifests(vod_id, args.arc)
    arc_times = _load_arc_times(vod_id, args.arc)
    if not arc_paths and not arc_times:
        print("No arc manifests or arcs_vod_times.txt entries found")
        raise SystemExit(1)

    out_root = args.output_dir or Path(f"data/chunks/{vod_id}/arcs")
    out_root.mkdir(parents=True, exist_ok=True)

    # Environment flags similar to director's cut chat overlay
    os.environ.setdefault('CHAT_FRAMERATE', '10')
    os.environ.setdefault('CHAT_UPDATE_RATE', '0.5')
    os.environ.setdefault('CHAT_GENERATE_MASK', '1')

    # Cache on-stream chat detection per VOD for this process
    vod_chat_present: Optional[bool] = None

    # Prefer arcs_vod_times when present; otherwise fall back to per-arc manifests
    if arc_times:
        iterable = [(None, s, e, i) for (i, s, e) in arc_times]
    else:
        iterable = arc_paths

    for item in iterable:
        try:
            import json
            if arc_times:
                _, s, e, idx = item
                man = {"arc_index": int(idx), "ranges": [{"start": float(s), "end": float(e)}], "start_abs": s, "end_abs": e}
                mp = Path(f"data/vector_stores/{vod_id}/arcs/arc_{int(idx):03d}_manifest.json")
            else:
                mp = item
                man = json.loads(mp.read_text(encoding='utf-8'))
        except Exception:
            print(f"⚠️  Could not read manifest: {mp}")
            continue
        arc_idx = int(man.get('arc_index') or 0)
        ranges = man.get('ranges') or []
        if not isinstance(ranges, list) or not ranges:
            print(f"⚠️  Empty ranges for arc {arc_idx}")
            continue

        print(f"🎯 Processing arc {arc_idx:03d} ({_format_hms(man.get('start_abs', 0))} → {_format_hms(man.get('end_abs', 0))})")
        seg_paths = _download_segments_for_ranges(vod_id, ranges, quality=args.quality, max_workers=max(1, args.max_workers))
        if not seg_paths:
            print(f"⚠️  No segments for arc {arc_idx}")
            continue
        if args.chat:
            # Probe for canvas based on first segment
            canvas_w_env = int(os.getenv('DC_CANVAS_W', '0') or '0')
            canvas_h_env = int(os.getenv('DC_CANVAS_H', '0') or '0')
            # YOLO chat detection on original segment: if chat exists, skip overlay
            if vod_chat_present is None:
                try:
                    vod_chat_present = _detect_onstream_chat(seg_paths[0])
                except Exception:
                    vod_chat_present = False
            if vod_chat_present:
                print("🛈 On-stream chat detected via YOLO → skipping chat overlay for this arc")
                overlaid = seg_paths
            else:
                overlaid = _overlay_chat_for_segments(
                    vod_id,
                    seg_paths,
                    canvas_w=canvas_w_env,
                    canvas_h=canvas_h_env,
                    chat_w=int(args.chat_w),
                    chat_h=int(args.chat_h),
                    chat_margin=int(args.chat_margin),
                    chat_head=int(args.chat_head_start),
                )
        else:
            overlaid = seg_paths

        out_path = out_root / f"{vod_id}_arc_{arc_idx:03d}.mp4"
        ok = _concat_best(overlaid, out_path)
        if not ok:
            print(f"❌ Concat failed for arc {arc_idx}")
            continue
        print(f"✅ Created arc video: {out_path}")


if __name__ == "__main__":
    main()


