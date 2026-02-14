#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import requests


class TwitchVodInfoProvider:
    """Encapsulates fetching VOD metadata and chapters.

    Prefers TwitchDownloaderCLI for speed and local availability,
    with a Helix API fallback when CLI info is unavailable.
    """

    def __init__(self, twitch_cli_path: Optional[str] = None) -> None:
        self.twitch_cli_path = twitch_cli_path or os.getenv("TWITCH_DOWNLOADER_PATH", "TwitchDownloaderCLI")

    # -------------------- Public API --------------------

    def get_vod_info(self, vod_id: str) -> Dict:
        # Cache first
        cached = self._read_cached_info(vod_id)
        if cached:
            try:
                p = self._cache_dir(vod_id) / "info.json"
                print(f"[VodInfo] cache hit: {p}")
            except Exception:
                pass
            return cached
        info = self._get_vod_info_cli(vod_id) or self._get_vod_info_api(vod_id)
        if info:
            self._write_cached_info(vod_id, info)
            try:
                p = self._cache_dir(vod_id) / "info.json"
                print(f"[VodInfo] cache write: {p}")
            except Exception:
                pass
        return info

    def get_vod_chapters(self, vod_id: str) -> List[Dict]:
        cached = self._read_cached_chapters(vod_id)
        if cached:
            try:
                p = self._cache_dir(vod_id) / "chapters.json"
                print(f"[VodInfo] chapters cache hit: {p} ({len(cached)} items)")
            except Exception:
                pass
            return cached
        chapters = self._get_vod_chapters_cli(vod_id)
        if chapters:
            self._write_cached_chapters(vod_id, chapters)
            try:
                p = self._cache_dir(vod_id) / "chapters.json"
                print(f"[VodInfo] chapters cache write: {p} ({len(chapters)} items)")
            except Exception:
                pass
        return chapters

    # -------------------- Cache helpers --------------------

    def _cache_dir(self, vod_id: str) -> Path:
        d = Path("data") / "cache" / "vod_info" / str(vod_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _read_cached_info(self, vod_id: str) -> Dict:
        p = self._cache_dir(vod_id) / "info.json"
        if p.exists():
            try:
                import json
                return json.loads(p.read_text(encoding='utf-8'))
            except Exception:
                return {}
        return {}

    def _write_cached_info(self, vod_id: str, info: Dict) -> None:
        try:
            import json
            p = self._cache_dir(vod_id) / "info.json"
            p.write_text(json.dumps(info, ensure_ascii=False), encoding='utf-8')
        except Exception:
            pass

    def _read_cached_chapters(self, vod_id: str) -> List[Dict]:
        p = self._cache_dir(vod_id) / "chapters.json"
        if p.exists():
            try:
                import json
                data = json.loads(p.read_text(encoding='utf-8'))
                return data if isinstance(data, list) else []
            except Exception:
                return []
        return []

    def _write_cached_chapters(self, vod_id: str, chapters: List[Dict]) -> None:
        try:
            import json
            p = self._cache_dir(vod_id) / "chapters.json"
            p.write_text(json.dumps(chapters, ensure_ascii=False), encoding='utf-8')
        except Exception:
            pass

    # -------------------- Chapter color mapping --------------------

    def get_chapter_colors(self, vod_id: str) -> Dict[str, tuple[int, int, int]]:
        """Load persistent chapter-to-color mappings for this VOD.
        
        Returns a dict mapping chapter_name -> (R, G, B) tuple.
        """
        p = self._cache_dir(vod_id) / "chapter_colors.json"
        if p.exists():
            try:
                import json
                data = json.loads(p.read_text(encoding='utf-8'))
                # Convert lists back to tuples
                return {k: tuple(v) for k, v in data.items() if isinstance(v, list) and len(v) == 3}
            except Exception:
                return {}
        return {}

    def save_chapter_colors(self, vod_id: str, color_map: Dict[str, tuple[int, int, int]]) -> None:
        """Save chapter-to-color mappings for this VOD.
        
        Args:
            vod_id: The VOD ID
            color_map: Dict mapping chapter_name -> (R, G, B) tuple
        """
        try:
            import json
            p = self._cache_dir(vod_id) / "chapter_colors.json"
            # Convert tuples to lists for JSON serialization
            serializable = {k: list(v) for k, v in color_map.items()}
            p.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f"[VodInfo] Saved chapter colors: {p} ({len(color_map)} chapters)")
        except Exception as e:
            print(f"[VodInfo] Failed to save chapter colors: {e}")

    def ensure_chapter_colors(self, vod_id: str, chapter_names: List[str]) -> Dict[str, tuple[int, int, int]]:
        """Ensure all chapters have assigned colors, generating new ones if needed.
        
        Args:
            vod_id: The VOD ID
            chapter_names: List of chapter names to ensure colors for
            
        Returns:
            Dict mapping chapter_name -> (R, G, B) tuple
        """
        BRIGHT_COLORS = [
            (57, 255, 20),      # neon green
            (255, 20, 147),     # hot pink
            (255, 255, 0),      # electric yellow
            (255, 165, 0),      # neon orange
            (0, 255, 255),      # cyan
            (255, 0, 255),      # magenta
            (191, 255, 0),      # lime
            (255, 105, 180),    # hot pink 2
            (138, 43, 226),     # blue violet
            (0, 255, 127),      # spring green
            (255, 69, 0),       # red orange
            (30, 144, 255),     # dodger blue
        ]
        
        color_map = self.get_chapter_colors(vod_id)
        needs_save = False
        
        # Assign colors to new chapters
        for chapter_name in chapter_names:
            if chapter_name not in color_map:
                # Use deterministic hash for consistent colors across runs
                import hashlib
                h = int(hashlib.md5(chapter_name.lower().encode()).hexdigest(), 16)
                color_map[chapter_name] = BRIGHT_COLORS[h % len(BRIGHT_COLORS)]
                needs_save = True
        
        if needs_save:
            self.save_chapter_colors(vod_id, color_map)
        
        return color_map

    # -------------------- Internals: CLI --------------------

    def _make_tmp_env(self) -> Dict[str, str]:
        temp_dir = Path("/tmp/streamsniped_downloads") if os.name != 'nt' else Path("C:/temp/streamsniped_downloads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env['TEMP'] = str(temp_dir)
        env['TMP'] = str(temp_dir)
        env['TMPDIR'] = str(temp_dir)
        return env

    def _get_vod_info_cli(self, vod_id: str) -> Dict:
        try:
            env = self._make_tmp_env()
            result = subprocess.run(
                [self.twitch_cli_path, "info", "--id", vod_id, "--format", "table"],
                capture_output=True,
                encoding='utf-8',
                errors='ignore',
                timeout=30,
                env=env,
            )
            if result.returncode != 0 or not result.stdout:
                return {}
            lines = result.stdout.strip().split('\n')
            vod_info: Dict[str, object] = {}
            for line in lines:
                if '│' in line and not any(ch in line for ch in ['─','═','║','╔','╗','╚','╝','╠','╣','╤','╧','╪','╫','├','┤','┬','┴','┌','┐','└','┘']):
                    parts = [p.strip() for p in line.split('│')]
                    if len(parts) >= 3:
                        key = parts[1].strip()
                        value = parts[2].strip()
                        if key and value:
                            vod_info[key] = value
            # Duration parsing (HH:MM:SS or MM:SS)
            length = str(vod_info.get('Length', '')).strip()
            duration_seconds = 0
            if length and ':' in length:
                bits = [int(x) for x in length.split(':') if x.isdigit()]
                if len(bits) == 3:
                    duration_seconds = bits[0] * 3600 + bits[1] * 60 + bits[2]
                elif len(bits) == 2:
                    duration_seconds = bits[0] * 60 + bits[1]
            vod_info['duration'] = duration_seconds
            return vod_info
        except subprocess.TimeoutExpired:
            return {}
        except Exception:
            return {}

    def _get_vod_chapters_cli(self, vod_id: str) -> List[Dict]:
        try:
            env = self._make_tmp_env()
            result = subprocess.run(
                [self.twitch_cli_path, "info", "--id", vod_id, "--format", "table"],
                capture_output=True,
                encoding='utf-8',
                errors='ignore',
                timeout=30,
                env=env,
            )
            if result.returncode != 0 or not result.stdout:
                return []
            return self._parse_chapters_from_table_output(result.stdout)
        except subprocess.TimeoutExpired:
            return []
        except Exception:
            return []

    # -------------------- Internals: Helix API --------------------

    def _get_vod_info_api(self, vod_id: str) -> Dict:
        client_id = os.getenv("TWITCH_CLIENT_ID")
        client_secret = os.getenv("TWITCH_CLIENT_SECRET")
        if not client_id or not client_secret:
            return {}
        try:
            token_resp = requests.post(
                "https://id.twitch.tv/oauth2/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "grant_type": "client_credentials",
                },
                timeout=10,
            )
            access_token = (token_resp.json() or {}).get("access_token")
            if not access_token:
                return {}
            headers = {"Client-ID": client_id, "Authorization": f"Bearer {access_token}"}
            resp = requests.get(
                "https://api.twitch.tv/helix/videos",
                params={"id": vod_id},
                headers=headers,
                timeout=10,
            )
            if resp.status_code != 200:
                return {}
            data = (resp.json() or {}).get('data') or []
            if not data:
                return {}
            video = data[0]
            duration_seconds = self._parse_twitch_duration(video.get("duration", "0s"))
            return {
                "Title": video.get("title", "Unknown"),
                "duration": duration_seconds,
                "UserName": video.get("user_name"),
                "GameName": video.get("game_name"),
            }
        except Exception:
            return {}

    # -------------------- Parsing helpers --------------------

    @staticmethod
    def _parse_chapters_from_table_output(output: str) -> List[Dict]:
        lines = output.strip().split('\n')
        chapters: List[Dict] = []
        in_section = False
        for line in lines:
            if 'Video Chapters' in line:
                in_section = True
                continue
            if not in_section:
                continue
            # Skip box-drawing lines and headers
            if any(c in line for c in ['─','═','║','╔','╗','╚','╝','╠','╣','╤','╧','╪','╫','├','┤','┬','┴','┌','┐','└','┘']):
                continue
            if not line.strip():
                continue
            if any(h in line for h in ['Category', 'Type', 'Start', 'End', 'Length']):
                continue
            parsed = TwitchVodInfoProvider._parse_chapter_line(line)
            if parsed:
                # Clean category at the source for reliable matching
                original_category = parsed['category']
                cleaned = TwitchVodInfoProvider.clean_chapter_name_standard(original_category)
                parsed['category'] = cleaned
                parsed['original_category'] = original_category
                chapters.append(parsed)
        return chapters

    @staticmethod
    def _parse_chapter_line(line: str) -> Optional[Dict]:
        line = line.strip()
        box_chars = ['─','═','║','╔','╗','╚','╝','╠','╣','╤','╧','╪','╫','│','├','┤','┬','┴','┌','┐','└','┘']
        if not line or all(ch in box_chars for ch in line.strip()):
            return None
        if '│' in line:
            parts = [p.strip() for p in line.split('│') if p.strip()]
            if len(parts) >= 5:
                category = parts[0].strip()
                chapter_type = parts[1].strip()
                start_time = parts[2].strip()
                end_time = parts[3].strip()
                length = parts[4].strip() if len(parts) > 4 else ""
                try:
                    start_seconds = TwitchVodInfoProvider.parse_timestamp(start_time)
                    end_seconds = TwitchVodInfoProvider.parse_timestamp(end_time)
                    if end_seconds > start_seconds >= 0:
                        return {
                            'category': category,
                            'type': chapter_type,
                            'start_time': start_seconds,
                            'end_time': end_seconds,
                            'duration': end_seconds - start_seconds,
                            'start_timestamp': start_time,
                            'end_timestamp': end_time,
                            'length': length,
                        }
                except Exception:
                    return None
        # Fallback loose parsing
        parts = line.split()
        if len(parts) >= 4:
            try:
                category = parts[0]
                chapter_type = parts[1]
                start_time = parts[2]
                end_time = parts[3]
                length = parts[4] if len(parts) > 4 else ""
                start_seconds = TwitchVodInfoProvider.parse_timestamp(start_time)
                end_seconds = TwitchVodInfoProvider.parse_timestamp(end_time)
                if end_seconds > start_seconds >= 0:
                    return {
                        'category': category,
                        'type': chapter_type,
                        'start_time': start_seconds,
                        'end_time': end_seconds,
                        'duration': end_seconds - start_seconds,
                        'start_timestamp': start_time,
                        'end_timestamp': end_time,
                        'length': length,
                    }
            except Exception:
                return None
        return None

    @staticmethod
    def parse_timestamp(timestamp_str: str) -> int:
        parts = timestamp_str.split(':')
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes * 60 + seconds
        raise ValueError(f"Invalid timestamp: {timestamp_str}")

    @staticmethod
    def _parse_twitch_duration(dur: str) -> int:
        total = 0
        num = ""
        for ch in dur or "":
            if ch.isdigit():
                num += ch
            else:
                if ch == 'h':
                    total += int(num or 0) * 3600
                elif ch == 'm':
                    total += int(num or 0) * 60
                elif ch == 's':
                    total += int(num or 0)
                num = ""
        return int(total)

    @staticmethod
    def clean_chapter_name_standard(name: str) -> str:
        s = (name or "").lower()
        for frm, to in [
            (' ', '_'), ('+', '_'), ('&', '_'), ('-', '_'), ('/', '_'), ('\\', '_'),
            ('(', '_'), (')', '_'), ('[', '_'), (']', '_'), ('{', '_'), ('}', '_'),
            (':', '_'), (';', '_'), (',', '_'), ('.', '_'), ('!', '_'), ('?', '_'),
            ('@', '_'), ('#', '_'), ('$', '_'), ('%', '_'), ('^', '_'), ('*', '_'),
            ('=', '_'), ('|', '_'), ('"', '_'), ("'", '_'), ('`', '_'), ('~', '_'),
        ]:
            s = s.replace(frm, to)
        while '__' in s:
            s = s.replace('__', '_')
        return s.strip('_')


