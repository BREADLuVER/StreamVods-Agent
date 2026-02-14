#!/usr/bin/env python3
"""
Render orchestrator for Director's Cut.

Supports:
- Direct single-graph xfade render
- Micro-batch merging to avoid giant filter graphs
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import subprocess

from .ffmpeg_graph import build_xfade_command, run_ffmpeg, run_ffmpeg_streaming


def _concat_copy(inputs: List[Path], output: Path, timeout: int | None = None) -> bool:
    try:
        ffmpeg = 'executables/ffmpeg.exe' if os.name == 'nt' and Path('executables/ffmpeg.exe').exists() else 'ffmpeg'
        concat_list = output.parent / 'concat_list.txt'
        with concat_list.open('w', encoding='utf-8') as f:
            for p in inputs:
                f.write(f"file '{p.absolute()}'\n")
        cmd = [ffmpeg, '-f', 'concat', '-safe', '0', '-fflags', '+genpts', '-i', str(concat_list), '-c', 'copy', '-movflags', '+faststart', '-y', str(output)]
        ok, _, _ = run_ffmpeg(cmd, timeout=timeout if (timeout is not None and timeout > 0) else None)
        try:
            concat_list.unlink()
        except Exception:
            pass
        return ok and output.exists()
    except Exception:
        return False


def _concat_encode(inputs: List[Path], output: Path, use_nvenc: bool = True, timeout: int | None = None) -> bool:
    try:
        ffmpeg = 'executables/ffmpeg.exe' if os.name == 'nt' and Path('executables/ffmpeg.exe').exists() else 'ffmpeg'
        ffprobe = str(Path('executables/ffmpeg.exe')).replace('ffmpeg.exe', 'ffprobe.exe') if os.name == 'nt' and Path('executables/ffmpeg.exe').exists() else 'ffprobe'

        def _probe_fps(p: Path) -> float:
            try:
                import subprocess
                r = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=avg_frame_rate,r_frame_rate', '-of', 'csv=p=0', str(p)], capture_output=True, text=True, timeout=10)
                out = (r.stdout or '').strip().split('\n')[0]
                parts = out.split(',') if out else []
                def _to(s: str) -> float:
                    if not s:
                        return 30.0
                    if '/' in s:
                        a, b = s.split('/', 1)
                        try:
                            return float(a) / float(b)
                        except Exception:
                            return 30.0
                    try:
                        return float(s)
                    except Exception:
                        return 30.0
                avg = _to(parts[0] if len(parts) >= 1 else '')
                rf = _to(parts[1] if len(parts) >= 2 else '')
                return avg if avg > 0.1 else (rf if rf > 0.1 else 30.0)
            except Exception:
                return 30.0

        native_fps = _probe_fps(inputs[0]) if inputs else 30.0
        rounded_fps = int(round(native_fps))

        concat_list = output.parent / 'concat_list.txt'
        with concat_list.open('w', encoding='utf-8') as f:
            for p in inputs:
                f.write(f"file '{p.absolute()}'\n")
        cmd = [ffmpeg, '-f', 'concat', '-safe', '0', '-fflags', '+genpts', '-i', str(concat_list)]
        # Preserve native FPS and enforce CFR to avoid VFR glue drift
        # Use exact NTSC fractions when applicable
        if abs(rounded_fps - 30) <= 1 and 29.5 < native_fps < 30.5:
            fps_arg = '30000/1001'
        elif abs(rounded_fps - 60) <= 1 and 59.0 < native_fps < 60.5:
            fps_arg = '60000/1001'
        elif abs(rounded_fps - 24) <= 1 and 23.0 < native_fps < 24.5:
            fps_arg = '24000/1001'
        else:
            fps_arg = str(rounded_fps)
        cmd += ['-r', fps_arg]
        # Normalize audio timing to prevent boundary desyncs
        cmd += ['-af', 'aresample=async=1:first_pts=0']
        if use_nvenc:
            # High quality NVENC settings with room for 1080p60
            v_bitrate = os.getenv('CONCAT_VBITRATE', '18M' if rounded_fps >= 50 else '12M')
            v_maxrate = os.getenv('CONCAT_VMAXRATE', '36M' if rounded_fps >= 50 else '24M')
            v_bufsize = os.getenv('CONCAT_VBUFSIZE', v_maxrate)
            cmd += ['-c:v', 'h264_nvenc', '-preset', os.getenv('CONCAT_NV_PRESET', 'p4'), '-rc', os.getenv('CONCAT_NV_RC', 'vbr_hq'), '-cq', os.getenv('CONCAT_NV_CQ', '18'), '-b:v', v_bitrate, '-maxrate', v_maxrate, '-bufsize', v_bufsize, '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-level', '4.2', '-g', str(int(max(1, rounded_fps * 2))), '-bf', '3', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-c:a', 'aac', '-b:a', os.getenv('CONCAT_ABITRATE', '192k'), '-movflags', '+faststart']
        else:
            cmd += ['-c:v', 'libx264', '-preset', os.getenv('CONCAT_X264_PRESET', 'slow'), '-crf', os.getenv('CONCAT_X264_CRF', '18'), '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-level', '4.2', '-g', str(int(max(1, rounded_fps * 2))), '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-c:a', 'aac', '-b:a', os.getenv('CONCAT_ABITRATE', '192k'), '-movflags', '+faststart']
        cmd += ['-y', str(output)]
        ok, _, _ = run_ffmpeg(cmd, timeout=timeout if (timeout is not None and timeout > 0) else None)
        try:
            concat_list.unlink()
        except Exception:
            pass
        return ok and output.exists()
    except Exception:
        return False


def _has_audio(path: Path) -> bool:
    """Return True if the file contains at least one audio stream."""
    try:
        # Resolve ffprobe path correctly on Windows
        if os.name == 'nt' and Path('executables/ffmpeg.exe').exists():
            ffprobe = str(Path('executables/ffmpeg.exe')).replace('ffmpeg.exe', 'ffprobe.exe')
        else:
            ffprobe = 'ffprobe'
        # check for any audio stream; csv output is empty when none found
        res = subprocess.run(
            [ffprobe, '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=index',
             '-of', 'csv=p=0', str(path)],
            capture_output=True, text=True, timeout=10,
        )
        return bool((res.stdout or '').strip())
    except Exception:
        return False


def render_with_transitions(
    inputs: List[Path],
    output: Path,
    seed: int,
    transition_duration: float = 1.0,
    batch_size: Optional[int] = 24,
    audio_crossfade: bool = False,
    use_nvenc: bool = True,
    durations_override: Optional[List[float]] = None,
    debug: bool = False,
    audio_transition_duration: Optional[float] = None,
    debug_pts: bool = False,
    timeout: Optional[int] = None,
    v_bitrate: Optional[str] = None,
    v_maxrate: Optional[str] = None,
    v_bufsize: Optional[str] = None,
    v_cq: Optional[str] = None,
) -> bool:
    if not inputs:
        return False
    # ------------------------------------------------------------
    # Disable audio cross-fade automatically when any clip lacks audio
    # ------------------------------------------------------------
    if audio_crossfade and any(not _has_audio(p) for p in inputs):
        print("⚠️  At least one input has no audio track – disabling audio cross-fades to avoid xfade failure")
        audio_crossfade = False
    output.parent.mkdir(parents=True, exist_ok=True)
    if len(inputs) == 1:
        return _concat_copy(inputs, output, timeout=timeout)

    # If manageable number of inputs, try single graph first
    if not batch_size or len(inputs) <= batch_size:
        cmd = build_xfade_command(inputs, output, seed=seed, transition_duration=transition_duration, audio_crossfade=audio_crossfade, use_nvenc=use_nvenc, durations_override=durations_override, debug=debug, audio_transition_duration=audio_transition_duration, debug_pts=debug_pts, v_bitrate=v_bitrate, v_maxrate=v_maxrate, v_bufsize=v_bufsize, v_cq=v_cq)
        # Stream FFmpeg stats to stdout so progress appears in logs
        ok, out, err = run_ffmpeg_streaming(cmd, timeout=timeout)
        if not ok:
            print("❌ xfade render failed – FFmpeg stderr (first 40 lines):")
            print("\n".join(err.splitlines()[:40]))
            print("❌ xfade render failed – FFmpeg stderr (last 40 lines):")
            print("\n".join(err.splitlines()[-40:]))
        if ok and output.exists():
            return True
        # As fallback, try CPU encode
        cmd_cpu = build_xfade_command(inputs, output, seed=seed, transition_duration=transition_duration, audio_crossfade=audio_crossfade, use_nvenc=False, durations_override=durations_override, debug=debug, audio_transition_duration=audio_transition_duration, debug_pts=debug_pts, v_bitrate=v_bitrate, v_maxrate=v_maxrate, v_bufsize=v_bufsize, v_cq=v_cq)
        ok2, _, _ = run_ffmpeg_streaming(cmd_cpu, timeout=timeout)
        if ok2 and output.exists():
            return True
        # If single-graph failed, drop into micro-batch rendering instead of giving up
        # Choose a conservative batch size to reduce filtergraph memory
        batch_size = max(6, min(10, len(inputs)))

    # Micro-batch rendering
    intermediates: List[Path] = []
    for i in range(0, len(inputs), int(batch_size)):
        part = inputs[i:i + int(batch_size)]
        inter = output.with_name(f"{output.stem}_part_{i//int(batch_size)+1:02d}{output.suffix}")
        part_durs = None
        if durations_override:
            part_durs = durations_override[i:i + int(batch_size)]
        cmd = build_xfade_command(part, inter, seed=seed + i, transition_duration=transition_duration, audio_crossfade=audio_crossfade, use_nvenc=use_nvenc, durations_override=part_durs, debug=debug, audio_transition_duration=audio_transition_duration, debug_pts=debug_pts, v_bitrate=v_bitrate, v_maxrate=v_maxrate, v_bufsize=v_bufsize, v_cq=v_cq)
        ok, _, _ = run_ffmpeg_streaming(cmd, timeout=timeout)
        if not (ok and inter.exists()):
            # Try CPU fallback for this batch
            cmd_cpu = build_xfade_command(part, inter, seed=seed + i, transition_duration=transition_duration, audio_crossfade=audio_crossfade, use_nvenc=False, durations_override=part_durs, debug=debug, audio_transition_duration=audio_transition_duration, debug_pts=debug_pts, v_bitrate=v_bitrate, v_maxrate=v_maxrate, v_bufsize=v_bufsize, v_cq=v_cq)
            ok2, _, _ = run_ffmpeg_streaming(cmd_cpu, timeout=timeout)
            if not (ok2 and inter.exists()):
                # Try non-transition concat for this batch
                if not _concat_copy(part, inter, timeout=timeout):
                    if not _concat_encode(part, inter, use_nvenc=True, timeout=timeout):
                        if not _concat_encode(part, inter, use_nvenc=False, timeout=timeout):
                            return False
        intermediates.append(inter)

    # Final concat (copy) of intermediates
    final_ok = False
    if _concat_copy(intermediates, output, timeout=timeout):
        final_ok = True
    elif _concat_encode(intermediates, output, use_nvenc=True, timeout=timeout):
        final_ok = True
    else:
        final_ok = _concat_encode(intermediates, output, use_nvenc=False, timeout=timeout)

    # Cleanup intermediates to save disk
    if final_ok:
        for p in intermediates:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
    return final_ok


