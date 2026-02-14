#!/usr/bin/env python3
"""
Composition helpers to overlay pre-rendered chat onto a video segment.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple


def _resolve_ffmpeg_bin() -> str:
    if os.name == "nt":
        cand = Path("executables/ffmpeg.exe")
        if cand.exists():
            return str(cand)
    return "ffmpeg"


def _probe_video_meta(path: Path) -> Tuple[int, int, float]:
    """Return (width, height, fps) using ffprobe. Fallbacks are sensible defaults.

    We prefer avg_frame_rate, then r_frame_rate. Values are returned as floats.
    """
    try:
        ffprobe = _resolve_ffmpeg_bin().replace("ffmpeg", "ffprobe")
        # Query width,height, and frame rates
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,r_frame_rate",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        out = (result.stdout or "").strip().split("\n")[0]
        # Expected format: width,height,avg,rf
        parts = out.split(",") if out else []
        w = int(parts[0]) if len(parts) >= 1 and parts[0].isdigit() else 1920
        h = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 1080

        def _to_fps(s: str) -> float:
            if not s:
                return 30.0
            if "/" in s:
                num, den = s.split("/", 1)
                try:
                    n = float(num)
                    d = float(den)
                    return n / d if d else 30.0
                except Exception:
                    return 30.0
            try:
                return float(s)
            except Exception:
                return 30.0

        avg = _to_fps(parts[2] if len(parts) >= 3 else "")
        rf = _to_fps(parts[3] if len(parts) >= 4 else "")
        fps = avg if avg > 0.1 else (rf if rf > 0.1 else 30.0)
        return (w, h, fps)
    except Exception:
        return (1920, 1080, 30.0)


def _probe_has_audio(path: Path) -> bool:
    try:
        ffprobe = _resolve_ffmpeg_bin().replace("ffmpeg", "ffprobe")
        r = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return bool((r.stdout or "").strip())
    except Exception:
        return False


def overlay_chat_on_video(
    segment_path: Path,
    output_path: Path,
    chat_color_path: Path,
    chat_mask_path: Optional[Path],
    canvas_w: int,
    canvas_h: int,
    chat_w: int,
    chat_h: int,
    pos: Tuple[int, int],
    trim_preroll_sec: int = 20,
    use_nvenc: bool = True,
    target_fps: Optional[float] = None,
) -> bool:
    """Overlay chat (color+mask or single alpha) onto a segment video at pos.

    The chat renderer usually generates a preroll so we trim that by trim_preroll_sec.
    """
    try:
        ffmpeg = _resolve_ffmpeg_bin()
        x, y = pos

        # Probe base video to preserve its native FPS when encoding
        base_w_probe, base_h_probe, base_fps = _probe_video_meta(segment_path)
        # Allow explicit overrides from caller/env
        if target_fps is None:
            env_fps = os.getenv("OVERLAY_TARGET_FPS", "").strip()
            try:
                target_fps = float(env_fps) if env_fps else base_fps
            except Exception:
                target_fps = base_fps
        # Choose sane defaults for encoder bitrates based on FPS
        is_60 = float(target_fps or 30.0) >= 50.0
        default_bitrate = "18M" if is_60 else "12M"
        default_maxrate = "36M" if is_60 else "24M"
        default_bufsize = default_maxrate
        nv_preset = os.getenv("OVERLAY_NV_PRESET", "p4")
        nv_rc = os.getenv("OVERLAY_NV_RC", "vbr_hq")
        nv_cq = os.getenv("OVERLAY_NV_CQ", "18")
        v_bitrate = os.getenv("OVERLAY_VBITRATE", default_bitrate)
        v_maxrate = os.getenv("OVERLAY_VMAXRATE", default_maxrate)
        v_bufsize = os.getenv("OVERLAY_VBUFSIZE", default_bufsize)

        # Build filter graph
        # [0:v] segment video scaled/padded to canvas (if needed)
        # [1:v] chat color; [2:v] optional chat mask → alphamerge
        pre = []
        inputs = ["-i", str(segment_path), "-i", str(chat_color_path)]
        if chat_mask_path and chat_mask_path.exists():
            inputs += ["-i", str(chat_mask_path)]
            pre.append(f"[1:v]trim=start={trim_preroll_sec},setpts=PTS-STARTPTS,setsar=1[c0]")
            pre.append(f"[2:v]trim=start={trim_preroll_sec},setpts=PTS-STARTPTS,setsar=1[c1]")
            # Ensure overlay sees a proper alpha plane
            pre.append("[c0][c1]alphamerge[am]")
            pre.append("[am]format=rgba[chat]")
            chat_label = "[chat]"
        else:
            pre.append(f"[1:v]trim=start={trim_preroll_sec},setpts=PTS-STARTPTS,setsar=1,format=rgba[chat]")
            chat_label = "[chat]"

        # Ensure base video fits canvas size
        pre.append(
            f"[0:v]scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=decrease:flags=lanczos+accurate_rnd+full_chroma_inp,"
            f"pad={canvas_w}:{canvas_h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,setpts=PTS-STARTPTS[base]"
        )

        # Position chat
        pre.append(f"[base]{chat_label}overlay=x={x}:y={y}:eof_action=pass[vout]")

        # Optional audio normalization: rebase PTS to 0 and async resample to avoid drift
        has_audio = _probe_has_audio(segment_path)
        if has_audio:
            pre.append("[0:a]asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0[aout]")

        filter_complex = ";".join(pre)

        cmd = [ffmpeg, "-y", *inputs, "-filter_complex", filter_complex, "-map", "[vout]"]
        if has_audio:
            cmd += ["-map", "[aout]"]
        else:
            cmd += ["-map", "0:a?"]
        # Preserve or set target FPS at container level (avoid accidental 30 fps lock)
        if target_fps and target_fps > 0:
            tf = float(target_fps)
            # Map common NTSC rates to exact rationals so timestamps stay perfect
            if abs(tf - 29.97) < 0.05:
                fps_arg = "30000/1001"
            elif abs(tf - 59.94) < 0.1:
                fps_arg = "60000/1001"
            elif abs(tf - 23.976) < 0.05:
                fps_arg = "24000/1001"
            else:
                fps_arg = str(int(round(tf)))
            cmd += ["-r", fps_arg]
        if use_nvenc:
            cmd += [
                "-c:v", "h264_nvenc",
                "-preset", nv_preset,
                "-rc", nv_rc,
                "-cq", nv_cq,
                "-b:v", v_bitrate,
                "-maxrate", v_maxrate,
                "-bufsize", v_bufsize,
                "-pix_fmt", "yuv420p",
                "-profile:v", "high",
                "-level", "4.2",
                "-g", str(int(max(1, int(round((target_fps or base_fps) * 2))))),
                "-bf", "3",
                "-colorspace", "bt709",
                "-color_primaries", "bt709",
                "-color_trc", "bt709",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
            ]
        else:
            cmd += [
                "-c:v", "libx264",
                "-preset", os.getenv("OVERLAY_X264_PRESET", "slow"),
                "-crf", os.getenv("OVERLAY_X264_CRF", "18"),
                "-pix_fmt", "yuv420p",
                "-profile:v", "high",
                "-level", "4.2",
                "-g", str(int(max(1, int(round((target_fps or base_fps) * 2))))),
                "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
            ]
        cmd += [str(output_path)]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        ok = result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0
        if not ok:
            err = (result.stderr or "") + (result.stdout or "")
            tail = err[-2000:] if len(err) > 2000 else err
            print(f"X overlay ffmpeg failed: {tail}")
        return ok
    except Exception as e:
        print(f"X overlay_chat_on_video error: {e}")
        return False


