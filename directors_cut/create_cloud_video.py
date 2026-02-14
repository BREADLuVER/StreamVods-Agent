#!/usr/bin/env python3
"""
Create Director's Cut video from enhanced manifest with efficient downloads and transitions.

Usage:
  python -m directors_cut.create_cloud_video <vod_id> [--quality 1080p] [--max-workers 4] [--no-transitions] [--audio-crossfade]
                                             [--batch-size 24] [--transition-duration 1.0] [--output <path>]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from pathlib import Path as _Path

# Allow running as a script (python directors_cut/create_cloud_video.py) or module (-m directors_cut.create_cloud_video)
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from directors_cut.manifest import load_manifest
from directors_cut.downloader import plan_segments_from_ranges, download_segments
from directors_cut.render import render_with_transitions
from chat_overlay.renderer import render_chat_segment
from chat_overlay.compose import overlay_chat_on_video
from chat_overlay.chat_json import ensure_full_chat_json, write_chat_subset


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Director's Cut video from manifest")
    parser.add_argument("vod_id", help="VOD ID")
    parser.add_argument("--quality", default="1080p")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--no-transitions", action="store_true")
    # Chat overlay controls (enabled by default)
    parser.add_argument("--no-chat", dest="chat", action="store_false", help="Disable chat overlay on segments")
    parser.add_argument("--chat-w", type=int, default=0, help="Chat overlay width (px). 0 = auto")
    parser.add_argument("--chat-h", type=int, default=0, help="Chat overlay height (px). 0 = auto")
    parser.add_argument("--chat-margin", type=int, default=24, help="Right margin for chat placement (px)")
    parser.add_argument("--chat-head-start", type=int, default=10, help="Seconds to preroll chat before segment start")
    # Defaults on for convenience; provide negative flags to disable when necessary
    parser.add_argument("--no-audio-crossfade", dest="audio_crossfade", action="store_false", help="Disable audio acrossfade transitions")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--transition-duration", type=float, default=1.0)
    parser.add_argument("--output", type=Path, help="Output MP4 path")
    parser.add_argument("--keep-temp", action="store_true", help="Keep local artifacts after render (skip cleanup)")
    parser.add_argument("--no-s3-upload", action="store_true", help="Skip S3 upload (for local processing)")
    parser.add_argument("--debug-transitions", action="store_true", help="Print xfade/acrossfade filter graph and durations for validation")
    parser.add_argument("--audio-transition-duration", type=float, default=None, help="Audio crossfade duration in seconds (defaults to video transition duration)")
    parser.add_argument("--debug-pts", action="store_true", help="Attach showinfo/ashowinfo to print PTS around transitions")
    parser.add_argument("--no-use-existing", dest="use_existing", action="store_false", help="Do not reuse existing downloaded segments")
    parser.add_argument("--use-existing", dest="use_existing", action="store_true", help="Reuse existing downloaded segments if found")
    parser.add_argument("--no-normalize", action="store_true", help="Skip per-clip normalization step (faster on reruns)")
    parser.add_argument("--timeout-seconds", type=int, default=0, help="Max seconds for FFmpeg steps (0 = no timeout)")
    # Encoding tuning
    parser.add_argument("--v-bitrate", type=str, default=None, help="Video target bitrate, e.g. '3500k' or '3.5M'")
    parser.add_argument("--v-maxrate", type=str, default=None, help="Video maxrate, e.g. '5M'")
    parser.add_argument("--v-bufsize", type=str, default=None, help="Video bufsize, e.g. '10M'")
    parser.add_argument("--v-cq", type=str, default=None, help="NVENC constant quality, e.g. '22'")
    parser.set_defaults(audio_crossfade=True, use_existing=True, chat=True, keep_temp=False)
    args = parser.parse_args()

    # When testing, user can pass --keep-temp to skip cleanup routines
    keep_temp: bool = bool(getattr(args, "keep_temp", False))
    debug_transitions: bool = bool(getattr(args, "debug_transitions", False))
    debug_pts: bool = bool(getattr(args, "debug_pts", False))
    audio_td = args.audio_transition_duration
    overall_timeout = int(getattr(args, 'timeout_seconds', 0) or 0)

    vod_id = args.vod_id
    quality = args.quality or "1080p"
    max_workers = max(1, args.max_workers)
    batch_size = None if args.batch_size <= 0 else args.batch_size
    transition_duration = max(0.1, args.transition_duration)

    man = load_manifest(vod_id)
    ranges = man.get("ranges") or []
    if not ranges:
        print("No ranges in manifest; nothing to do")
        raise SystemExit(1)

    # Prefer existing pre-downloaded segments if requested or detected
    seg_dir = Path(f"data/chunks/{vod_id}/director_cut_segments")
    existing_base = []
    existing_chat = []
    if seg_dir.exists():
        all_mp4 = [p for p in seg_dir.glob("seg_*.mp4") if p.is_file() and p.stat().st_size > 0]
        existing_base = sorted([p for p in all_mp4 if not p.stem.endswith("_chat")])
        existing_chat = sorted([p for p in all_mp4 if p.stem.endswith("_chat")])
    skip_overlay_due_to_existing = False
    if args.use_existing and (existing_base or existing_chat):
        if existing_base:
            print(f"🔎 Using {len(existing_base)} existing base segments from {seg_dir}")
            paths = existing_base
        else:
            print(f"🔎 Using {len(existing_chat)} existing chat-augmented segments from {seg_dir}")
            paths = existing_chat
            skip_overlay_due_to_existing = True
    else:
        segments = plan_segments_from_ranges(ranges, vod_id, max_segment_seconds=300)
        paths = download_segments(segments, vod_id, max_workers=max_workers, quality=quality)
        if not paths:
            print("No segments available (downloaded or existing)")
            raise SystemExit(1)

    # Determine canvas size from first segment or env overrides
    def _probe_resolution(p: Path) -> tuple[int, int]:
        try:
            ffprobe = 'ffprobe'
            if os.name == 'nt' and (Path('executables/ffmpeg.exe')).exists():
                ffprobe = str(Path('executables/ffmpeg.exe'))
                ffprobe = ffprobe.replace('ffmpeg.exe', 'ffprobe.exe')
            result = __import__('subprocess').run([
                ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height',
                '-of', 'csv=p=0:s=x', str(p)
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

    # Configure chat overlay
    env_enable = (os.getenv('DC_CHAT_ENABLE', '1').lower() in ('1', 'true', 'yes'))
    chat_enabled = bool(args.chat and env_enable and not skip_overlay_due_to_existing)
    canvas_w_env = int(os.getenv('DC_CANVAS_W', '0') or '0')
    canvas_h_env = int(os.getenv('DC_CANVAS_H', '0') or '0')
    first_w, first_h = _probe_resolution(paths[0]) if paths else (1920, 1080)
    canvas_w = canvas_w_env or first_w
    canvas_h = canvas_h_env or first_h
    # Chat size: fixed for DC per user request
    fixed_w = 250
    fixed_h = 300
    chat_w = int(os.getenv('DC_CHAT_W', str(args.chat_w or fixed_w)))
    chat_h = int(os.getenv('DC_CHAT_H', str(args.chat_h or fixed_h)))
    chat_margin = int(os.getenv('DC_CHAT_MARGIN', str(args.chat_margin)))
    chat_head = int(os.getenv('DC_CHAT_HEAD_START', str(args.chat_head_start)))

    def _parse_times(path: Path) -> tuple[int, int] | None:
        try:
            name = path.stem  # e.g., seg_0001_123-456 or seg_0001_123-456_chat
            m = __import__('re').search(r"_(\d+)-(\d+)(?:_chat)?$", name)
            if not m:
                return None
            return (int(m.group(1)), int(m.group(2)))
        except Exception:
            return None

    def _find_chat_color_mask(chat_base: Path) -> tuple[Path | None, Path | None]:
        color = chat_base.with_name(chat_base.stem + "_raw.mp4")
        mask_a = color.with_name(color.stem + "_mask" + color.suffix)
        mask_b = color.with_name(color.stem + ".mask" + color.suffix)
        mask = mask_a if mask_a.exists() else (mask_b if mask_b.exists() else None)
        return (color if color.exists() else None, mask)

    overlaid_paths = []
    if chat_enabled:
        print(f"🗨️  Chat overlay enabled: canvas={canvas_w}x{canvas_h}, chat={chat_w}x{chat_h}, margin={chat_margin}, head={chat_head}s")
        # Performance defaults for DC runs (override via env if needed)
        os.environ.setdefault('CHAT_FRAMERATE', '24')
        os.environ.setdefault('CHAT_UPDATE_RATE', '0.5')
        os.environ.setdefault('CHAT_GENERATE_MASK', '1')
        chat_dir = Path(f"data/chunks/{vod_id}/director_cut_chat")
        chat_dir.mkdir(parents=True, exist_ok=True)
        # Download full chat JSON once
        full_chat_dir = Path(f"data/chats/{vod_id}")
        full_json = ensure_full_chat_json(vod_id, full_chat_dir)
        if not (full_json and full_json.exists()):
            print("⚠️  Full chat JSON unavailable; proceeding without chat")
            overlaid_paths = paths
        else:
            # Parse segments with times
            parsed: list[tuple[Path,int,int]] = []
            for p in paths:
                # If already a chat-augmented segment, keep as-is
                if p.stem.endswith('_chat'):
                    overlaid_paths.append(p)
                    continue
                t = _parse_times(p)
                if not t:
                    print(f"⚠️  Could not parse times from {p.name}; skipping chat overlay for this segment")
                    overlaid_paths.append(p)
                    continue
                parsed.append((p, t[0], t[1]))

            # Build contiguous blocks
            blocks: list[tuple[int,int]] = []
            cur_s: int | None = None
            cur_e: int | None = None
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

            # Pre-render chat once per block (uses subset JSON from the full JSON you already have)
            block_assets: dict[tuple[int,int], tuple[Path, Path | None]] = {}
            for bstart, bend in blocks:
                subset_json = chat_dir / f"chat_{bstart}_{bend}.json"
                if not (subset_json.exists() and subset_json.stat().st_size > 10):
                    ok_sub = write_chat_subset(full_json, subset_json, bstart, bend, head_sec=float(chat_head))
                    if not ok_sub:
                        print(f"⚠️  Failed to prepare chat subset for block {bstart}-{bend}")
                        continue
                chat_out = chat_dir / f"chat_{bstart}_{bend}_{chat_w}x{chat_h}.webm"
                # Reuse cached chat assets if present
                cached_color, cached_mask = _find_chat_color_mask(chat_out)
                if chat_out.exists() and chat_out.stat().st_size > 10 and cached_color:
                    block_assets[(bstart, bend)] = (cached_color, cached_mask)
                    continue
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

            # Overlay each segment by trimming from its block asset
            for p, s, e in parsed:
                # Find block covering this segment
                blk = None
                for b in blocks:
                    if b[0] <= s and e <= b[1]:
                        blk = b
                        break
                if blk is None or blk not in block_assets or not block_assets[blk][0]:
                    overlaid_paths.append(p)
                    continue
                color, mask = block_assets[blk]
                # Compute offset inside block (including head start)
                offset = max(0, (s - blk[0]) + chat_head)
                # Position: far right middle
                x = max(0, canvas_w - chat_w - chat_margin)
                y = max(0, (canvas_h - chat_h) // 2)
                out_seg = p.with_name(p.stem + "_chat" + p.suffix)
                # Reuse existing chat-augmented segment when available
                if out_seg.exists() and out_seg.stat().st_size > 0:
                    overlaid_paths.append(out_seg)
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
                    if ok2:
                        overlaid_paths.append(out_seg)
                    else:
                        overlaid_paths.append(p)
                except Exception:
                    overlaid_paths.append(p)
    else:
        overlaid_paths = paths

    out_dir = Path(f"data/chunks/{vod_id}/director_cut")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output or out_dir / f"{vod_id}_directors_cut.mp4"

    # ------------------------------------------------------------
    # Optional per-clip normalization (fixes negative PTS, aligns audio/video)
    # ------------------------------------------------------------

    def _normalize_clip(inp: Path) -> Path:
        """Re-encode the clip to clean timestamps and equal A/V length."""
        out = inp.with_name(inp.stem + "_norm" + inp.suffix)
        if out.exists():
            return out
        ffmpeg = 'executables/ffmpeg.exe' if os.name == 'nt' and Path('executables/ffmpeg.exe').exists() else 'ffmpeg'
        cmd = [
            ffmpeg, '-y', '-fflags', '+genpts', '-i', str(inp),
            '-vf', 'setpts=PTS-STARTPTS,fps=60,format=yuv420p',
            '-af', 'aresample=async=1:first_pts=0,apad,atrim=0:exact=1,aformat=sample_fmts=s16:channel_layouts=stereo',
            '-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '18',
            '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', str(out)
        ]
        from directors_cut.ffmpeg_graph import run_ffmpeg  # type: ignore
        ok, _, _ = run_ffmpeg(cmd, timeout=(overall_timeout if overall_timeout > 0 else None))
        return out if ok and out.exists() else inp

    # Allow disabling normalization via CLI or env (DC_NORMALIZE=0)
    env_norm = os.getenv('DC_NORMALIZE', '1').lower() in ('1','true','yes')
    normalize = (not bool(getattr(args, 'no_normalize', False))) and env_norm
    if normalize:
        paths = [_normalize_clip(p) for p in paths]

    # ------------------------------------------------------------
    # Collapse slices within the same merged group into a single group file
    # so transitions only occur between groups (eliminates internal xfade)
    # ------------------------------------------------------------
    def _parse_times_from_name(p: Path) -> tuple[float, float] | None:
        try:
            name = p.stem
            m = __import__('re').search(r"_(\d+)-(\d+)(?:_chat)?$", name)
            if not m:
                return None
            return (float(m.group(1)), float(m.group(2)))
        except Exception:
            return None

    def _group_index_for_segment(seg_start: float, seg_end: float) -> int:
        # Assign segment to the merged range that fully contains it (with tiny tolerance)
        tol = 0.75
        for gi, r in enumerate(ranges):
            rs = float(r.get('start') or 0.0)
            re = float(r.get('end') or 0.0)
            if (seg_start + tol) >= rs and (seg_end - tol) <= re:
                return gi
        # Fallback: closest by start
        best = 0
        best_d = 1e18
        for gi, r in enumerate(ranges):
            d = abs((seg_start + seg_end) * 0.5 - (float(r.get('start') or 0.0) + float(r.get('end') or 0.0)) * 0.5)
            if d < best_d:
                best = gi
                best_d = d
        return best

    # Map each segment to its group index
    group_to_segments: dict[int, list[tuple[float, float, Path]]] = {}
    for p in overlaid_paths:
        te = _parse_times_from_name(p)
        if not te:
            # If filename doesn't carry timing, treat as its own group to avoid internal transitions
            gid = len(group_to_segments)
            group_to_segments.setdefault(gid, []).append((0.0, 0.0, p))
            continue
        s, e = te
        gid = _group_index_for_segment(s, e)
        group_to_segments.setdefault(gid, []).append((s, e, p))

    # Sort groups by their manifest order; sort segments within each group by start time
    ordered_group_ids = sorted(group_to_segments.keys())
    per_group_outputs: list[Path] = []
    out_dir = Path(f"data/chunks/{vod_id}/director_cut")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import to avoid circulars
    from directors_cut.render import _concat_copy  # type: ignore

    for gid in ordered_group_ids:
        entries = sorted(group_to_segments[gid], key=lambda t: t[0])
        seg_paths = [p for _, _, p in entries]
        if len(seg_paths) == 1:
            per_group_outputs.append(seg_paths[0])
            continue
        group_out = out_dir / f"{vod_id}_group_{gid:03d}.mp4"
        if not group_out.exists():
            ok_concat = _concat_copy(seg_paths, group_out)
            if not ok_concat:
                # Fallback: keep original segments if concat failed (rare)
                per_group_outputs.extend(seg_paths)
                continue
        per_group_outputs.append(group_out)

    # Replace inputs with group-level outputs so transitions apply only between groups
    overlaid_paths = per_group_outputs

    # Prepare manifest-based durations for each group file to drive exact xfade offsets
    group_durations: list[float] = []
    for gid in ordered_group_ids:
        if gid < len(ranges):
            rs = float(ranges[gid].get('start') or 0.0)
            re = float(ranges[gid].get('end') or 0.0)
            group_durations.append(max(0.1, round(re - rs, 3)))
        else:
            # Fallback: probe
            try:
                ffprobe = 'ffprobe'
                if os.name == 'nt' and (Path('executables/ffmpeg.exe')).exists():
                    ffprobe = str(Path('executables/ffmpeg.exe')).replace('ffmpeg.exe', 'ffprobe.exe')
                res = __import__('subprocess').run([
                    ffprobe, '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(per_group_outputs[len(group_durations)])
                ], capture_output=True, text=True, timeout=10)
                d = float((res.stdout or '').strip() or '0')
                group_durations.append(max(0.1, round(d, 3)))
            except Exception:
                group_durations.append(0.1)

    if args.no_transitions or len(overlaid_paths) <= 1:
        # Simple concat copy path
        from directors_cut.render import _concat_copy  # type: ignore
        overall_timeout = int(getattr(args, 'timeout_seconds', 0) or 0)
        ok = _concat_copy(overlaid_paths, out_path, timeout=(overall_timeout if overall_timeout > 0 else None))
        if not ok:
            print("Concat copy failed; trying transitions path with single-graph fallback")
        else:
            print(f"✅ Created: {out_path}")
            raise SystemExit(0)

    seed = int(vod_id) if vod_id.isdigit() else abs(hash(vod_id)) % (2**31)
    use_nvenc = True  # laptop has RTX 3060; NVENC available
    print(f"🎬 Using xfade transitions with {len(overlaid_paths)} inputs (duration={transition_duration:.2f}s, audio_crossfade={args.audio_crossfade})")
    ok = render_with_transitions(
        inputs=overlaid_paths,
        output=out_path,
        seed=seed,
        transition_duration=transition_duration,
        batch_size=batch_size,
        audio_crossfade=args.audio_crossfade,
        use_nvenc=use_nvenc,
        durations_override=group_durations,
        debug=debug_transitions,
        audio_transition_duration=audio_td,
        debug_pts=debug_pts,
        timeout=(int(getattr(args, 'timeout_seconds', 0)) or None),
        v_bitrate=args.v_bitrate,
        v_maxrate=args.v_maxrate,
        v_bufsize=args.v_bufsize,
        v_cq=args.v_cq,
    )
    if not ok:
        print("Render failed")
        raise SystemExit(1)
    print(f"✅ Created: {out_path}")

    # Optional S3 upload if running in container/cloud, then clean local artifacts
    try:
        # Check for local processing mode flags first
        disable_s3_uploads = os.getenv('DISABLE_S3_UPLOADS', '').lower() in ('1', 'true', 'yes')
        local_test_mode = os.getenv('LOCAL_TEST_MODE', '').lower() in ('1', 'true', 'yes')
        container_mode = os.getenv('CONTAINER_MODE', 'false').lower() in ['true','1','yes']
        upload_videos = os.getenv('UPLOAD_VIDEOS', 'false').lower() in ['true','1','yes']
        
        # Only upload if not in local mode and not explicitly disabled
        should_upload = (container_mode or upload_videos) and not disable_s3_uploads and not local_test_mode
        
        if should_upload and not args.no_s3_upload and out_path.exists():
            from storage import StorageManager
            s3_bucket = os.getenv('S3_BUCKET', 'streamsniped-dev-videos')
            s3_uri = f"s3://{s3_bucket}/videos/{vod_id}/director_cut/{out_path.name}"
            storage = StorageManager()
            # Idempotent: skip if already on S3
            if storage.exists(s3_uri):
                print(f"⏭️  Director's Cut already on S3, skipping upload: {s3_uri}")
            else:
                storage.upload_file(str(out_path), s3_uri)
                print(f"✅ Uploaded to S3: {s3_uri}")
            # Remove local final to save disk (only if not in local mode)
            if not keep_temp and not local_test_mode:
                try:
                    out_path.unlink()
                    print("🧹 Deleted local final video to save disk space")
                except Exception as _e:
                    print(f"⚠️ Could not delete local final: {_e}")
        elif args.no_s3_upload:
            print("⏭️ S3 upload skipped (--no-s3-upload flag)")
        elif disable_s3_uploads or local_test_mode:
            print("⏭️ S3 upload skipped (local processing mode)")
    except Exception as e:
        print(f"⚠️ S3 upload skipped: {e}")

    # Cleanup downloaded segments by default to save space
    try:
        if args.use_existing and not keep_temp:
            seg_dir = Path(f"data/chunks/{vod_id}/director_cut_segments")
            if seg_dir.exists():
                for p in seg_dir.glob("*.mp4"):
                    try:
                        p.unlink()
                    except Exception:
                        pass
                try:
                    seg_dir.rmdir()
                except Exception:
                    pass
                print("🧹 Cleaned up temporary segments")
    except Exception:
        pass

    # Optional cleanup of per-group intermediate outputs when not keeping temp
    try:
        if not keep_temp:
            for p in overlaid_paths:
                # Do not delete the final output
                if p.exists() and p.name.startswith(f"{vod_id}_group_"):
                    try:
                        p.unlink()
                    except Exception:
                        pass
    except Exception:
        pass

    # Purge chat overlay assets for this VOD to save disk space
    try:
        chat_dir_cleanup = Path(f"data/chunks/{vod_id}/director_cut_chat")
        if chat_dir_cleanup.exists() and not keep_temp:
            # Remove known asset types (.webm, *_raw.mp4, *_mask.mp4, .json)
            for glob_pat in ("*.webm", "*_raw.mp4", "*_mask.mp4", "*.json"):
                for f in chat_dir_cleanup.glob(glob_pat):
                    try:
                        f.unlink()
                    except Exception:
                        pass
            # Attempt to remove the directory if empty
            try:
                chat_dir_cleanup.rmdir()
            except Exception:
                pass
            print("🧹 Purged chat overlay assets for VOD")
    except Exception:
        pass


if __name__ == "__main__":
    main()


