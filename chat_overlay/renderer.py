#!/usr/bin/env python3
"""
Chat rendering utilities built around TwitchDownloaderCLI, extracted for reuse.

This module mirrors the proven logic from processing-scripts/create_individual_clips.py
and exposes helpers to render chat color/mask at exact sizes for composition.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional
import json


def _resolve_twitch_cli_executable() -> str:
    """Resolve TwitchDownloaderCLI executable path"""
    override = os.getenv("TWITCH_DOWNLOADER_PATH")
    if override and Path(override).exists():
        return override

    possible_paths = [
        os.getenv("TWITCH_DOWNLOADER_PATH", ""),
        str(Path(__file__).parent.parent / "executables" / "TwitchDownloaderCLI.exe"),
        "C:/Users/bread/Documents/StreamSniped/TwitchDownloaderCLI.exe",
        "TwitchDownloaderCLI.exe",
        "./TwitchDownloaderCLI.exe",
        "TwitchDownloaderCLI",
        "./TwitchDownloaderCLI",
        "twitch-downloader",
        "./twitch-downloader",
    ]

    for path in possible_paths:
        try:
            if not path:
                continue
            result = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10)
            if "TwitchDownloaderCLI" in result.stdout or "TwitchDownloaderCLI" in result.stderr:
                print(f" Found TwitchDownloaderCLI: {path}")
                return path
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    try:
        result = subprocess.run("TwitchDownloaderCLI --version", shell=True, capture_output=True, text=True, timeout=10)
        if "TwitchDownloaderCLI" in result.stdout or "TwitchDownloaderCLI" in result.stderr:
            print(" Found TwitchDownloaderCLI via shell")
            return "TwitchDownloaderCLI"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    raise FileNotFoundError("TwitchDownloaderCLI not found. Please install it first.")


def render_chat_segment(
    vod_id: str,
    start_time: float,
    end_time: float,
    output_path: Path,
    chat_w: int,
    chat_h: int,
    head_start_sec: int = 10,
    message_hex: str = "#BFBFBF",
    bg_hex: str = "#00000000",
    alt_bg_hex: str = "#00000000",
    chat_json_override: Optional[Path] = None,
) -> bool:
    """Render chat video with transparency for a specific VOD time range.

    Writes color and mask raw MP4s next to output_path for graph composition.
    """
    try:
        try:
            head_start_override = int(os.getenv("CHAT_HEAD_START_SEC", str(head_start_sec)))
        except Exception:
            head_start_override = head_start_sec
        head_start_sec = max(0, int(head_start_override))

        safe_start = max(0, int(start_time - max(0, head_start_sec)))
        start_timestamp = f"{int(safe_start // 3600):02d}:{int((safe_start % 3600) // 60):02d}:{int(safe_start % 60):02d}"
        end_timestamp = f"{int(end_time // 3600):02d}:{int((end_time % 3600) // 60):02d}:{int(end_time % 60):02d}"

        twitch_cli = _resolve_twitch_cli_executable()

        tmp_dir = output_path.parent
        tmp_dir.mkdir(parents=True, exist_ok=True)
        chat_json = chat_json_override if chat_json_override else (tmp_dir / f"chat_{int(start_time)}_{int(end_time)}.json")

        force_redl = os.getenv("FORCE_CHAT_REDOWNLOAD", "0").lower() in ("1", "true", "yes")

        def _is_valid_json(p: Path) -> bool:
            try:
                if not p.exists() or p.stat().st_size < 10:
                    return False
                data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                return isinstance(data, (dict, list))
            except Exception:
                return False

        def _download_chat_json() -> bool:
            dl_cmd = [
                twitch_cli,
                "chatdownload",
                "--id",
                vod_id,
                "-b",
                start_timestamp,
                "-e",
                end_timestamp,
                "--embed-images",
                "--bttv=true",
                "--ffz=true",
                "--stv=true",
                "-o",
                str(chat_json),
            ]
            # Retry up to 2 times on invalid/empty JSON
            for attempt in range(1, 3):
                try:
                    res = subprocess.run(dl_cmd, capture_output=True, text=True, timeout=1800)
                except Exception as e:
                    print(f"X chatdownload exception: {e}")
                    continue
                if res.returncode == 0 and _is_valid_json(chat_json):
                    return True
                # Best-effort: remove bad file before retry to avoid false positives
                try:
                    if chat_json.exists():
                        chat_json.unlink()
                except Exception:
                    pass
                tail = (res.stderr or res.stdout or "")
                tail = tail[-2000:] if len(tail) > 2000 else tail
                print(f"X chatdownload invalid JSON (attempt {attempt}): {tail}")
            return _is_valid_json(chat_json)

        if chat_json_override is None and (force_redl or not _is_valid_json(chat_json)):
            if not _download_chat_json():
                print("X chatdownload produced invalid chat JSON; skipping chat overlay")
                return False
        elif chat_json_override is not None:
            if not _is_valid_json(chat_json):
                print("X provided chat_json_override is invalid; skipping chat overlay")
                return False

        try:
            font_size = int(os.getenv("CHAT_FONT_PX", "14"))
        except Exception:
            font_size = 16

        try:
            if output_path.exists():
                output_path.unlink()
        except Exception:
            pass

        raw_out = output_path.with_name(output_path.stem + "_raw.mp4")
        force_rerender = os.getenv("FORCE_CHAT_RERENDER", "0").lower() in ("1", "true", "yes")
        existing_mask = None
        for cand in [raw_out.with_name(raw_out.stem + "_mask" + raw_out.suffix), raw_out.with_name(raw_out.stem + ".mask" + raw_out.suffix)]:
            if cand.exists():
                existing_mask = cand
                break
        if (not force_rerender) and raw_out.exists() and (existing_mask is not None):
            return True

        render_cmd = [
            twitch_cli,
            "chatrender",
            "-i",
            str(chat_json),
            "-h",
            str(chat_h),
            "-w",
            str(chat_w),
            "--framerate",
            str(int(os.getenv("CHAT_FRAMERATE", "30"))),
            "--update-rate",
            str(float(os.getenv("CHAT_UPDATE_RATE", "0.5"))),
            "--font-size",
            str(font_size),
            "--background-color",
            bg_hex,
            "--alt-background-color",
            alt_bg_hex,
            "--message-color",
            "#FFFFFFFF",
            "--sub-messages",
            "false",
            "--sharpening",
            "false",
            "--collision",
            "Overwrite",
            "-o",
            str(raw_out),
        ]
        # Optional outline for readability; disabled by default for speed
        if os.getenv("CHAT_OUTLINE", "0").lower() in ("1", "true", "yes"):
            render_cmd.insert(render_cmd.index("--sub-messages"), "--outline")
        if os.getenv("CHAT_GENERATE_MASK", "1").lower() in ("1", "true", "yes"):
            render_cmd.insert(render_cmd.index("--collision"), "--generate-mask")
        print(f"🎛️ chatrender size request: {chat_w}x{chat_h}, font={font_size}")
        res2 = subprocess.run(render_cmd, capture_output=True, text=True, timeout=1800)
        if res2.returncode != 0 or not raw_out.exists():
            tail = (res2.stderr or res2.stdout or "")
            tail = tail[-2000:] if len(tail) > 2000 else tail
            print(f"X chatrender failed: {tail}")
            return False

        mask_path = None
        for cand in [raw_out.with_name(raw_out.stem + "_mask" + raw_out.suffix), raw_out.with_name(raw_out.stem + ".mask" + raw_out.suffix)]:
            if cand.exists():
                mask_path = cand
                break
        if mask_path is None:
            print(" Mask not found; using raw chat without alpha")
            return True

        try:
            if os.getenv("CHAT_DEBUG_BORDER", "0").lower() in ("1", "true", "yes"):
                ffmpeg_bin = "ffmpeg"
                if os.name == "nt":
                    candidate = Path(__file__).parent.parent / "executables" / "ffmpeg.exe"
                    if candidate.exists():
                        ffmpeg_bin = str(candidate)
                dbg_out = output_path.with_suffix(".dbg" + output_path.suffix)
                dbg_cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-i",
                    str(output_path),
                    "-vf",
                    "format=rgba,drawbox=x=0:y=0:w=iw:h=ih:color=lime@0.4:t=4",
                    "-c:v",
                    "libvpx-vp9" if output_path.suffix.lower() == ".webm" else "libx264",
                    "-pix_fmt",
                    "yuva420p",
                    str(dbg_out),
                ]
                _ = subprocess.run(dbg_cmd, capture_output=True, text=True, timeout=30)
                try:
                    output_path.unlink()
                except Exception:
                    pass
                dbg_out.rename(output_path)
        except Exception:
            pass

        return True
    except subprocess.TimeoutExpired:
        print("X chat render timeout")
        return False
    except Exception as e:
        print(f"X chat render error: {e}")
        return False


def ensure_chat_json(
    vod_id: str,
    start_time: float,
    end_time: float,
    chat_dir: Path,
    head_start_sec: int = 10,
) -> Optional[Path]:
    """Ensure chat JSON exists for a time range (prefetch step)."""
    try:
        chat_dir.mkdir(parents=True, exist_ok=True)
        try:
            head_start_override = int(os.getenv("CHAT_HEAD_START_SEC", str(head_start_sec)))
        except Exception:
            head_start_override = head_start_sec
        head_start_sec = max(0, int(head_start_override))

        safe_start = max(0, int(start_time - max(0, head_start_sec)))
        start_timestamp = f"{int(safe_start // 3600):02d}:{int((safe_start % 3600) // 60):02d}:{int(safe_start % 60):02d}"
        end_timestamp = f"{int(end_time // 3600):02d}:{int((end_time % 3600) // 60):02d}:{int(end_time % 60):02d}"

        twitch_cli = _resolve_twitch_cli_executable()
        chat_json = chat_dir / f"chat_{int(start_time)}_{int(end_time)}.json"
        force_redl = os.getenv("FORCE_CHAT_REDOWNLOAD", "0").lower() in ("1", "true", "yes")

        def _is_valid_json(p: Path) -> bool:
            try:
                if not p.exists() or p.stat().st_size < 10:
                    return False
                data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                return isinstance(data, (dict, list))
            except Exception:
                return False

        if chat_json.exists() and not force_redl and _is_valid_json(chat_json):
            return chat_json

        dl_cmd = [
            twitch_cli,
            "chatdownload",
            "--id",
            vod_id,
            "-b",
            start_timestamp,
            "-e",
            end_timestamp,
            "--embed-images",
            "--bttv=true",
            "--ffz=true",
            "--stv=true",
            "-o",
            str(chat_json),
        ]
        for attempt in range(1, 3):
            res = subprocess.run(dl_cmd, capture_output=True, text=True, timeout=1800)
            if res.returncode == 0 and _is_valid_json(chat_json):
                return chat_json
            try:
                if chat_json.exists():
                    chat_json.unlink()
            except Exception:
                pass
        return chat_json if _is_valid_json(chat_json) else None
    except Exception:
        return None


