#!/usr/bin/env python3
"""
FFmpeg graph builders for Director's Cut rendering with transitions.

Notes:
- Uses CPU-side xfade filter; encodes with NVENC when available
- Defaults to 1080p output, 30fps, yuv420p, stereo 48k
"""

from __future__ import annotations

import os
import random
import re
import subprocess
from pathlib import Path
from typing import List, Tuple


def _resolve_ffmpeg() -> str:
    try:
        exe = Path('executables/ffmpeg.exe')
        if os.name == 'nt' and exe.exists():
            return str(exe)
    except Exception:
        pass
    return 'ffmpeg'


def _probe_stream_duration_seconds(path: Path, selector: str) -> float:
    """Return stream duration seconds for selector like 'a:0' or 'v:0', 0.0 if unknown."""
    try:
        ffprobe = _resolve_ffmpeg().replace('ffmpeg', 'ffprobe')
        res = subprocess.run(
            [ffprobe, '-v', 'error', '-select_streams', selector, '-show_entries', 'stream=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(path)],
            capture_output=True, text=True, timeout=10,
        )
        out = (res.stdout or '').strip()
        return float(out) if out else 0.0
    except Exception:
        return 0.0


def _probe_best_duration_seconds(path: Path) -> float:
    """Prefer audio stream duration, then video stream, then container format duration."""
    try:
        # Prefer audio stream
        a = _probe_stream_duration_seconds(path, 'a:0')
        if a and a > 0:
            return a
        # Then video stream
        v = _probe_stream_duration_seconds(path, 'v:0')
        if v and v > 0:
            return v
        # Finally, container format duration
        ffprobe = _resolve_ffmpeg().replace('ffmpeg', 'ffprobe')
        res = subprocess.run(
            [ffprobe, '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(path)],
            capture_output=True, text=True, timeout=10,
        )
        out = (res.stdout or '').strip()
        fmt = float(out) if out else 0.0
        return fmt if fmt > 0 else 0.0
    except Exception:
        return 0.0


def _probe_resolution(path: Path) -> Tuple[int, int]:
    """Return (width,height) using ffprobe; fallback 1920x1080."""
    try:
        ffprobe = _resolve_ffmpeg().replace('ffmpeg', 'ffprobe')
        res = subprocess.run(
            [ffprobe, '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=width,height', '-of', 'csv=p=0:s=x', str(path)],
            capture_output=True, text=True, timeout=10,
        )
        out = (res.stdout or '').strip()
        if 'x' in out:
            w, h = map(int, out.split('x'))
            return max(2, w), max(2, h)
    except Exception:
        pass
    return (1920, 1080)


def build_xfade_command(
    inputs: List[Path],
    output: Path,
    seed: int,
    transition_duration: float = 1.0,
    audio_crossfade: bool = False,
    use_nvenc: bool = True,
    durations_override: List[float] | None = None,
    debug: bool = False,
    audio_transition_duration: float | None = None,
    debug_pts: bool = False,
    v_bitrate: str | None = None,
    v_maxrate: str | None = None,
    v_bufsize: str | None = None,
    v_cq: str | None = None,
) -> List[str]:
    ffmpeg = _resolve_ffmpeg()
    if len(inputs) == 1:
        # Fast path: copy stream
        return [ffmpeg, '-y', '-i', str(inputs[0]), '-c', 'copy', str(output)]

    # Probe durations separately for video and audio; prefer stream durations
    def _probe_v(path: Path) -> float:
        v = _probe_stream_duration_seconds(path, 'v:0')
        return v if v and v > 0 else _probe_best_duration_seconds(path)

    def _probe_a(path: Path) -> float:
        a = _probe_stream_duration_seconds(path, 'a:0')
        return a if a and a > 0 else _probe_best_duration_seconds(path)

    if durations_override and len(durations_override) == len(inputs):
        v_durations = [max(0.1, float(d)) for d in durations_override]
    else:
        v_durations = [max(0.1, _probe_v(p)) for p in inputs]
    a_durations = [max(0.1, _probe_a(p)) for p in inputs]
    # Choose audio transition duration (defaults to video transition duration)
    audio_td = float(audio_transition_duration) if (audio_transition_duration is not None) else float(transition_duration)

    # Choose transitions deterministically
    rng = random.Random(seed)
    choices = ['fade']
    selected = [rng.choice(choices) for _ in range(len(inputs) - 1)]

    # Calculate proper offsets that account for time gaps between segments
    def parse_segment_times(path: Path) -> Tuple[float, float] | None:
        """Parse start and end times from segment filename like seg_0001_123-456.mp4"""
        try:
            name = path.stem
            # Match pattern like seg_0001_123-456 or seg_0001_123-456_chat
            m = re.search(r"_(\d+)-(\d+)(?:_chat)?$", name)
            if not m:
                return None
            start = float(m.group(1))
            end = float(m.group(2))
            return (start, end)
        except Exception:
            return None

    def calculate_proper_offsets(inputs: List[Path], transition_duration: float) -> List[float]:
        """Calculate xfade offsets for merged groups - transitions only between groups."""
        if len(inputs) < 2:
            return []
        
        offsets = []
        
        for i in range(1, len(inputs)):
            prev_times = parse_segment_times(inputs[i-1])
            curr_times = parse_segment_times(inputs[i])
            
            if prev_times and curr_times:
                prev_start, prev_end = prev_times
                curr_start, curr_end = curr_times
                
                # Calculate the time gap between merged groups
                time_gap = curr_start - prev_end
                
                # Treat as distinct groups only when time gap exceeds manifest merge threshold (15s)
                if time_gap > 15.0:
                    # Start transition near the end of the first merged group
                    offset = max(0, (prev_end - prev_start) - transition_duration)
                else:
                    # Flag to indicate we should fallback to cumulative timing
                    # (i.e., let original logic place the transition at the end of the previous
                    #  segment rather than the very start, which caused visible glitches).
                    offset = -1.0
            else:
                # Fallback: unknown timing — signal to use cumulative fallback instead of 0s
                offset = -1.0
            
            offsets.append(offset)
        
        return offsets

    # Normalize streams and build xfade chain
    v_parts: List[str] = []
    a_parts: List[str] = []

    # target resolution (first clip) so all inputs match for xfade
    tgt_w, tgt_h = _probe_resolution(inputs[0])

    for i in range(len(inputs)):
        if i == 0:
            # First clip also gets trimmed to start earlier for consistent timing
            v_parts.append(
                f"[{i}:v]trim=start={transition_duration},setpts=PTS-STARTPTS,fps=30,"
                f"scale={tgt_w}:{tgt_h}:force_original_aspect_ratio=decrease,pad={tgt_w}:{tgt_h}:(ow-iw)/2:(oh-ih)/2,"
                f"format=yuv420p,setsar=1[v{i}]"
            )
            a_parts.append(
                f"[{i}:a]"
                f"atrim=start={transition_duration},asetpts=PTS-STARTPTS,"
                f"aformat=sample_fmts=s16:channel_layouts=stereo,"
                f"aresample=48000:async=1:first_pts=0[a{i}]"
            )
        else:
            # Trim the first 'transition_duration' seconds so that this clip begins earlier and aligns post wipe
            v_parts.append(
                f"[{i}:v]trim=start={transition_duration},setpts=PTS-STARTPTS,fps=30,"
                f"scale={tgt_w}:{tgt_h}:force_original_aspect_ratio=decrease,pad={tgt_w}:{tgt_h}:(ow-iw)/2:(oh-ih)/2,"
                f"format=yuv420p,setsar=1[v{i}]"
            )
            a_parts.append(
                f"[{i}:a]"
                f"atrim=start={transition_duration},asetpts=PTS-STARTPTS,"
                f"aformat=sample_fmts=s16:channel_layouts=stereo,"
                f"aresample=48000:async=1:first_pts=0[a{i}]"
            )

    prev_v = '[v0]'
    prev_a = '[a0]'
    v_out = ''
    a_out = ''
    # Composite timeline length so far (after applying previous xfades)
    composite = v_durations[0]
    offsets_used: List[float] = []
    v_filters: List[str] = []
    a_filters: List[str] = []

    v_filters.extend(v_parts)
    a_filters.extend(a_parts)

    # Calculate proper offsets for gaps
    proper_offsets = calculate_proper_offsets(inputs, transition_duration)

    for i in range(1, len(inputs)):
        trans = selected[i - 1]
        # Use calculated offset if available, otherwise fallback to original logic
        if i-1 < len(proper_offsets) and proper_offsets[i-1] >= 0:
            # Use the pre-computed offset when it is a real value (>=0). Negative values
            # act as a sentinel meaning "use fallback cumulative timing".
            offset = proper_offsets[i-1]
        else:
            # When clips are trimmed by transition_duration, adjust offset to account for trimming
            # offset = (composite - transition_duration) - transition_duration
            offset = max(round(composite, 3) - (2 * transition_duration), 0.0)
        
        v_out = f"[vm{i:02d}]"
        cur_v = f"[v{i}]"
        offset = round(offset, 3)
        v_filters.append(f"{prev_v}{cur_v}xfade=transition={trans}:duration={transition_duration}:offset={offset:.3f}{v_out}")
        prev_v = v_out
        if audio_crossfade:
            # Simple acrossfade: uses last 'transition_duration' of prev and first 'transition_duration' of cur
            a_out = f"[am{i:02d}]"
            cur_a = f"[a{i}]"
            a_filters.append(f"{prev_a}{cur_a}acrossfade=d={audio_td}{a_out}")
            prev_a = a_out
        # Extend composite timeline: add current duration minus overlap amount
        offsets_used.append(offset)
        composite += v_durations[i] - transition_duration

    if not audio_crossfade:
        # Build hard-cut audio strictly aligned to video offsets
        # keep_len[0] = offsets_used[0]
        # keep_len[i] = offsets_used[i] - offsets_used[i-1] for middle clips
        # keep_len[last] = full a_durations[last]
        num = len(inputs)
        trimmed_labels: List[str] = []
        keep_lens: List[float] = []
        for i in range(num):
            if i < num - 1:
                kl = a_durations[i]
            else:
                kl = a_durations[i]
            keep_lens.append(round(kl, 3))
        for i in range(num):
            lbl = f"[a{i}]"
            if i < num - 1:
                trimmed = f"[at{i}]"
                a_filters.append(f"{lbl}atrim=0:{keep_lens[i]:.3f},asetpts=PTS-STARTPTS{trimmed}")
                trimmed_labels.append(trimmed)
            else:
                trimmed_labels.append(lbl)
        a_concat_in = ''.join(trimmed_labels)
        a_filters.append(f"{a_concat_in}concat=n={len(trimmed_labels)}:v=0:a=1[aout]")
        prev_a = '[aout]'

    # Optionally print PTS info by attaching showinfo/ashowinfo at final labels
    if debug_pts:
        vdbg = '[vdbg]'
        adbg = '[adbg]'
        v_filters.append(f"{prev_v}showinfo{vdbg}")
        a_filters.append(f"{prev_a}ashowinfo{adbg}")
        prev_v = vdbg
        prev_a = adbg

    filter_complex = ';'.join(v_filters + a_filters)

    if debug:
        try:
            print("\n=== Transition Debug ===")
            print(f"inputs: {[str(p) for p in inputs]}")
            print(f"v_durations_used: {v_durations}")
            print(f"a_durations_used: {a_durations}")
            print(f"transition_duration: {transition_duration}")
            print(f"audio_crossfade: {audio_crossfade}")
            print(f"selected_video_transitions: {selected}")
            print(f"proper_offsets: {proper_offsets}")
            print(f"offsets_used: {offsets_used}")
            if not audio_crossfade:
                print(f"audio_keep_lens: {keep_lens}")
            print("filter_complex:")
            print(filter_complex)
            print("=== End Debug ===\n")
        except Exception:
            pass

    cmd: List[str] = [ffmpeg, '-y', '-fflags', '+genpts']
    for p in inputs:
        cmd += ['-i', str(p)]
    # Control logging verbosity and stats frequency
    loglevel = os.getenv('FFMPEG_LOGLEVEL', 'error')
    stats_period = os.getenv('FFMPEG_STATS_PERIOD', '120')
    cmd += [
        '-filter_complex', filter_complex,
        '-map', prev_v,
        '-map', prev_a,
        '-r', '30',
        '-v', loglevel,
        '-stats',
        '-stats_period', str(stats_period),
    ]
    if use_nvenc:
        # Allow overriding NVENC rate control via CLI/env
        vb = v_bitrate or os.getenv('DC_VBITRATE') or '5M'
        mr = v_maxrate or os.getenv('DC_MAXRATE') or '10M'
        bs = v_bufsize or os.getenv('DC_BUFSIZE') or '10M'
        cq = v_cq or os.getenv('DC_CQ') or '18'
        cmd += [
            '-c:v', 'h264_nvenc',
            '-preset', 'p5',
            '-rc', 'vbr',
            '-cq', str(cq),
            '-b:v', str(vb),
            '-maxrate', str(mr),
            '-bufsize', str(bs),
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
        ]
    else:
        cmd += [
            '-c:v', 'libx264',
            '-preset', 'veryfast',
            '-crf', '20',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
        ]
    cmd += [str(output)]
    return cmd


def run_ffmpeg(cmd: List[str], timeout: int | None = None) -> Tuple[bool, str, str]:
    try:
        if timeout is None or timeout <= 0:
            res = subprocess.run(cmd, capture_output=True, text=True)
        else:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (res.returncode == 0, res.stdout or '', res.stderr or '')
    except subprocess.TimeoutExpired:
        return (False, '', 'timeout')
    except Exception as e:
        return (False, '', str(e))


def run_ffmpeg_streaming(cmd: List[str], timeout: int | None = None) -> Tuple[bool, str, str]:
    """Run FFmpeg and stream stderr lines for real-time progress visibility.

    Returns (ok, '', '') to keep signature compatibility with callers that ignore output.
    """
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # Stream stderr (FFmpeg prints stats to stderr)
        try:
            if proc.stderr is not None:
                for line in proc.stderr:
                    try:
                        print(line.rstrip())
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            if timeout is None or timeout <= 0:
                code = proc.wait()
            else:
                code = proc.wait(timeout)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
            return (False, '', 'timeout')
        return (code == 0, '', '')
    except Exception as e:
        return (False, '', str(e))


