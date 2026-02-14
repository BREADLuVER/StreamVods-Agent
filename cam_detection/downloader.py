#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional


def _resolve_twitch_cli_executable() -> str:
    override = os.getenv("TWITCH_DOWNLOADER_PATH", "").strip()
    if override and Path(override).exists():
        return override
    candidates = [
        override,
        str(Path("executables") / "TwitchDownloaderCLI.exe"),
        "TwitchDownloaderCLI.exe",
        "./TwitchDownloaderCLI.exe",
        "TwitchDownloaderCLI",
        "./TwitchDownloaderCLI",
        "twitch-downloader",
        "./twitch-downloader",
    ]
    for c in candidates:
        if not c:
            continue
        try:
            res = subprocess.run([c, "--version"], capture_output=True, text=True, timeout=10)
            out = (res.stdout or "") + (res.stderr or "")
            if "TwitchDownloaderCLI" in out:
                return c
        except Exception:
            continue
    raise FileNotFoundError("TwitchDownloaderCLI not found. Set TWITCH_DOWNLOADER_PATH or place executable in ./executables")


def download_small_chunk_1080p(
    vod_id: str,
    start_s: float,
    duration_s: float,
    output_path: Path,
    *,
    quality: str = "1080p",
    threads: Optional[int] = None,
) -> bool:
    """Download a small 1080p chunk suitable for frame sampling.

    Uses TwitchDownloaderCLI videodownload with begin/end timestamps.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start = int(round(max(0.0, float(start_s))))
    end = max(start + 1, int(round(start + max(1.0, float(duration_s)))))

    start_ts = f"{start // 3600:02d}:{(start % 3600) // 60:02d}:{start % 60:02d}"
    end_ts = f"{end // 3600:02d}:{(end % 3600) // 60:02d}:{end % 60:02d}"

    twitch_cli = _resolve_twitch_cli_executable()
    cmd = [
        twitch_cli, "videodownload",
        "--id", str(vod_id),
        "-b", start_ts,
        "-e", end_ts,
        "-o", str(output_path),
        "-q", quality,
        "--trim-mode", "Safe",
    ]
    t = threads if threads is not None else os.getenv("CLIP_DL_THREADS") or os.getenv("TD_MAX_PARALLEL") or "4"
    if t:
        cmd += ["-t", str(t)]

    # Prefer local ffmpeg if provided
    ffmpeg_path = os.getenv("FFMPEG_PATH") or os.getenv("FFMPEG_BIN")
    if ffmpeg_path:
        cmd += ["--ffmpeg-path", str(ffmpeg_path)]

    # Ensure overwrite behavior by pre-deleting any existing file
    if output_path.exists():
        try:
            output_path.unlink()
        except Exception:
            pass

    try:
        print(f"Running command: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        print(f"Command exit code: {res.returncode}")
        if res.stdout:
            print(f"STDOUT: {res.stdout}")
        if res.stderr:
            print(f"STDERR: {res.stderr}")
    except Exception as e:
        print(f"Command failed with exception: {e}")
        return False

    if res.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
        print(f"Download successful: {output_path} ({output_path.stat().st_size} bytes)")
        return True
    else:
        print(f"Download failed: returncode={res.returncode}, exists={output_path.exists()}, size={output_path.stat().st_size if output_path.exists() else 'N/A'}")
        return False


