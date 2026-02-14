#!/usr/bin/env python3
"""
Cloud-optimized AI data generation using TwitchDownloaderCLI
Downloads specific portions of VODs instead of full streams
Memory-efficient approach for cloud environments with OOM prevention
"""

import json
import subprocess
import sys
import re
import gc
import psutil
import os
import requests  # Added for Twitch API fallback
from threading import BoundedSemaphore
import time
import random
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.chat_utils import chat_utils
from src.downloader import downloader
from storage import StorageManager
from src.transcription.faster_whisper_client import transcribe_audio_file
from utils.chapter_merge import merge_short_chapters

# Ensure project env is loaded (including config/streamsniped.env with ASSEMBLYAI_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv("config/streamsniped.env")
except Exception:
    pass


@dataclass
class ChatMessage:
    """Efficient chat message storage"""
    message_id: str
    timestamp: int
    content: str
    username: str = "viewer"
    emotes: List[str] = None


@dataclass
class NarrativeSegment:
    """Coherent narrative segment with context"""
    segment_id: str
    start_time: int
    end_time: int
    duration: int
    transcript: str
    chat_messages: List[Dict]
    chat_activity: int
    # Original chapter information (preserved even after merging)
    original_chapter_type: str = 'unknown'
    original_chapter_category: str = 'unknown'
    original_chapter_id: str = 'unknown'


def setup_openrouter_api() -> str:
    """Setup OpenRouter API with API key from environment"""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    
    print(" OpenRouter API configured")
    return api_key


def call_openrouter_with_retry(prompt: str, max_retries: int = 3, base_delay: float = 2.0) -> str:
    """Call OpenRouter API with exponential backoff"""
    api_key = setup_openrouter_api()
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/your-repo/streamsniped",
        "X-Title": "StreamSniped Chapter Analysis"
    }
    
    data = {
        "model": "google/gemini-2.0-flash-exp:free",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3,  # Lower temperature for more consistent analysis
        "max_tokens": 200
    }
    
    QUIET_RETRY_LOGS = os.getenv('QUIET_RETRY_LOGS', 'true').lower() in ['true', '1', 'yes']
    for attempt in range(max_retries):
        try:
            if not QUIET_RETRY_LOGS:
                print(f"🤖 Attempting OpenRouter API call (attempt {attempt + 1})")
            elif attempt == 0:
                print("🤖 Calling OpenRouter (retries suppressed in logs)...")
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code != 200:
                if not QUIET_RETRY_LOGS:
                    print(f" OpenRouter API error: {response.status_code} - {response.text}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    if not QUIET_RETRY_LOGS:
                        print(f" Waiting {delay:.1f}s before retry...")
                    time.sleep(delay)
                continue
            
            response_data = response.json()
            result = response_data['choices'][0]['message']['content'].strip()
            
            print(f" OpenRouter OK")
            return result
            
        except Exception as e:
            error_msg = str(e).lower()
            
            if 'timeout' in error_msg or 'rate' in error_msg:
                if not QUIET_RETRY_LOGS:
                    print(f" Rate limit/timeout, retrying...")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    if not QUIET_RETRY_LOGS:
                        print(f" Waiting {delay:.1f}s before retry...")
                    time.sleep(delay)
                continue
            else:
                if not QUIET_RETRY_LOGS:
                    print(f" Error with OpenRouter API: {e}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    if not QUIET_RETRY_LOGS:
                        print(f" Waiting {delay:.1f}s before retry...")
                    time.sleep(delay)
                continue
    
    # If all retries failed, raise exception
    print(f" OpenRouter failed after {max_retries} retries")
    raise RuntimeError(f"OpenRouter API failed after {max_retries} retries")





def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage information"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
        'percent': process.memory_percent()
    }


def check_memory_threshold(threshold_mb: float = 3000) -> bool:
    """Check if memory usage is below threshold"""
    memory_usage = get_memory_usage()
    return memory_usage['rss_mb'] < threshold_mb


def force_garbage_collection():
    """Force garbage collection to free memory"""
    collected = gc.collect()
    print(f"Garbage collection freed {collected} objects")


def cleanup_temp_files(temp_dir: Path):
    """Clean up temporary files to free disk space"""
    try:
        if temp_dir.exists():
            for file_path in temp_dir.rglob('*'):
                if file_path.is_file() and file_path.suffix in ['.mp4', '.wav', '.m4a']:
                    file_path.unlink()
                    print(f"Cleaned up temp file: {file_path.name}")
    except Exception as e:
        print(f" Error cleaning temp files: {e}")


def get_environment_limits() -> Dict[str, int]:
    """Get processing limits from environment variables"""
    return {
        'max_chunks': int(os.environ.get('MAX_CHUNKS', '240')),
        'chunk_duration': int(os.environ.get('CHUNK_DURATION', '180')),  # 3 minutes (optimized from 5)
        'memory_threshold_mb': int(os.environ.get('MEMORY_THRESHOLD_MB', '3000')),
        'quality': os.environ.get('QUALITY', '720p')
    }


# -------------------------------------------------------------
# Twitch API helpers (fallback when TwitchDownloaderCLI fails)
# -------------------------------------------------------------

def _parse_twitch_duration(duration: str) -> int:
    """Convert Twitch duration string (e.g. '2h34m55s') to seconds"""
    total_seconds = 0
    for value, unit in re.findall(r"(\d+)([hms])", duration):
        value = int(value)
        if unit == "h":
            total_seconds += value * 3600
        elif unit == "m":
            total_seconds += value * 60
        else:
            total_seconds += value
    return total_seconds


def get_vod_info_from_api(vod_id: str) -> Dict:
    """Fallback: Get VOD info via Twitch Helix API using client credentials"""
    client_id = os.getenv("TWITCH_CLIENT_ID")
    client_secret = os.getenv("TWITCH_CLIENT_SECRET")

    if not client_id or not client_secret:
        return {}

    try:
        # Get an app access token (valid for ~60 days)
        token_resp = requests.post(
            "https://id.twitch.tv/oauth2/token",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "client_credentials",
            },
            timeout=10,
        )
        access_token = token_resp.json().get("access_token")
        if not access_token:
            return {}

        headers = {
            "Client-ID": client_id,
            "Authorization": f"Bearer {access_token}",
        }

        resp = requests.get(
            "https://api.twitch.tv/helix/videos",
            params={"id": vod_id},
            headers=headers,
            timeout=10,
        )

        if resp.status_code != 200:
            return {}

        data = resp.json().get("data", [])
        if not data:
            return {}

        video = data[0]
        duration_seconds = _parse_twitch_duration(video.get("duration", "0s"))

        return {
            "Title": video.get("title", "Unknown"),
            "duration": duration_seconds,
            "UserName": video.get("user_name"),
        }
    except Exception as exc:
        print(f"[WARNING] Twitch API fallback failed: {exc}")
        return {}


def clean_chapter_name_standard(chapter_name: str) -> str:
    """Clean chapter name to standard format (no spaces, no special chars)"""
    # Convert to lowercase and replace problematic characters
    clean_name = chapter_name.lower()
    clean_name = clean_name.replace(' ', '_')
    clean_name = clean_name.replace('+', '_')
    clean_name = clean_name.replace('&', '_')
    clean_name = clean_name.replace('-', '_')
    clean_name = clean_name.replace('/', '_')
    clean_name = clean_name.replace('\\', '_')
    clean_name = clean_name.replace('(', '_')
    clean_name = clean_name.replace(')', '_')
    clean_name = clean_name.replace('[', '_')
    clean_name = clean_name.replace(']', '_')
    clean_name = clean_name.replace('{', '_')
    clean_name = clean_name.replace('}', '_')
    clean_name = clean_name.replace(':', '_')
    clean_name = clean_name.replace(';', '_')
    clean_name = clean_name.replace(',', '_')
    clean_name = clean_name.replace('.', '_')
    clean_name = clean_name.replace('!', '_')
    clean_name = clean_name.replace('?', '_')
    clean_name = clean_name.replace('@', '_')
    clean_name = clean_name.replace('#', '_')
    clean_name = clean_name.replace('$', '_')
    clean_name = clean_name.replace('%', '_')
    clean_name = clean_name.replace('^', '_')
    clean_name = clean_name.replace('*', '_')
    clean_name = clean_name.replace('=', '_')
    clean_name = clean_name.replace('|', '_')
    clean_name = clean_name.replace('"', '_')
    clean_name = clean_name.replace("'", '_')
    clean_name = clean_name.replace("`", '_')
    clean_name = clean_name.replace("~", '_')
    
    # Remove consecutive underscores
    while '__' in clean_name:
        clean_name = clean_name.replace('__', '_')
    
    # Remove leading/trailing underscores
    clean_name = clean_name.strip('_')
    
    return clean_name


def get_vod_info(vod_id: str) -> Dict:
    """Get VOD information using TwitchDownloaderCLI"""
    print(f"[INFO] Getting VOD info for {vod_id}...")
    
    try:
        temp_dir = Path("/tmp/streamsniped_downloads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        env = os.environ.copy()
        env['TEMP'] = str(temp_dir)
        env['TMP'] = str(temp_dir)
        env['TMPDIR'] = str(temp_dir)
        
        result = subprocess.run(
            [os.getenv("TWITCH_DOWNLOADER_PATH", "TwitchDownloaderCLI"), "info", "--id", vod_id, "--format", "table"],
            capture_output=True,
            encoding='utf-8',
            errors='ignore',
            timeout=30,
            env=env
        )
        
        if result.returncode != 0 or not result.stdout:
            print(f"[WARNING] Failed to get VOD info via TwitchDownloaderCLI: {result.stderr}")
            # Fallback to Twitch API
            api_info = get_vod_info_from_api(vod_id)
            if api_info:
                print("[INFO] Retrieved VOD info via Twitch API fallback")
                return api_info
            return {}
        
        # Parse table format output
        lines = result.stdout.strip().split('\n')
        vod_info = {}
        
        for line in lines:
            # Look for table rows with pipe separators (│)
            if '│' in line and not any(char in line for char in ['─', '═', '║', '╔', '╗', '╚', '╝', '╠', '╣', '╤', '╧', '╪', '╫', '├', '┤', '┬', '┴', '┌', '┐', '└', '┘']):
                parts = line.split('│')
                if len(parts) >= 3:
                    key = parts[1].strip()
                    value = parts[2].strip()
                    if key and value:
                        vod_info[key] = value
        
        # Convert duration from HH:MM:SS to seconds if present
        if 'Length' in vod_info:
            duration_str = vod_info['Length']
            try:
                # Parse HH:MM:SS format
                if ':' in duration_str:
                    parts = duration_str.split(':')
                    if len(parts) == 3:  # HH:MM:SS
                        hours, minutes, seconds = map(int, parts)
                        duration_seconds = hours * 3600 + minutes * 60 + seconds
                        vod_info['duration'] = duration_seconds
                    elif len(parts) == 2:  # MM:SS
                        minutes, seconds = map(int, parts)
                        duration_seconds = minutes * 60 + seconds
                        vod_info['duration'] = duration_seconds
            except (ValueError, IndexError):
                pass
        
        print(f"[SUCCESS] Got VOD info: {vod_info.get('Title', 'Unknown')}")
        return vod_info
        
    except subprocess.TimeoutExpired:
        print(f"[ERROR] VOD info request timed out")
        return {}
    except Exception as e:
        print(f"[ERROR] Error getting VOD info: {e}")
        return {}


def get_vod_chapters(vod_id: str) -> List[Dict]:
    """Extract VOD chapters using TwitchDownloaderCLI"""
    print(f"[INFO] Extracting VOD chapters for {vod_id}...")
    
    try:
        temp_dir = Path("/tmp/streamsniped_downloads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        env = os.environ.copy()
        env['TEMP'] = str(temp_dir)
        env['TMP'] = str(temp_dir)
        env['TMPDIR'] = str(temp_dir)
        
        result = subprocess.run(
            [os.getenv("TWITCH_DOWNLOADER_PATH", "TwitchDownloaderCLI"), "info", "--id", vod_id, "--format", "table"],
            capture_output=True,
            encoding='utf-8',
            errors='ignore',
            timeout=30,
            env=env
        )
        
        if result.returncode != 0 or not result.stdout:
            print(f"[WARNING] Failed to get VOD chapters: {result.stderr}")
            return []
        
        lines = result.stdout.strip().split('\n')
        chapters = []
        
        in_chapters_section = False
        for line in lines:
            if 'Video Chapters' in line:
                in_chapters_section = True
                continue
            
            if in_chapters_section:
                if any(char in line for char in ['─', '═', '║', '╔', '╗', '╚', '╝', '╠', '╣', '╤', '╧', '╪', '╫', '├', '┤', '┬', '┴', '┌', '┐', '└', '┘']):
                    continue
                
                if not line.strip():
                    continue
                
                if any(header in line for header in ['Category', 'Type', 'Start', 'End', 'Length']):
                    continue
                
                chapter = parse_chapter_line(line)
                if chapter:
                    # Clean the chapter name at the source to eliminate future issues
                    original_category = chapter['category']
                    clean_category = clean_chapter_name_standard(original_category)
                    chapter['category'] = clean_category
                    chapter['original_category'] = original_category  # Keep original for reference
                    
                    chapters.append(chapter)
                    print(f"[SUCCESS] Chapter: {clean_category} (was: {original_category}) ({chapter['start_timestamp']} - {chapter['end_timestamp']})")
        
        if chapters:
            print(f"Found {len(chapters)} game chapters")
        else:
            print(f"No chapters found for VOD {vod_id}")
        
        return chapters
        
    except subprocess.TimeoutExpired:
        print(f"[ERROR] VOD info request timed out")
        return []
    except Exception as e:
        print(f"[ERROR] Error getting VOD chapters: {e}")
        return []


def parse_chapter_line(line: str) -> Optional[Dict]:
    """Parse chapter line with multiple strategies"""
    line = line.strip()
    
    box_chars = ['─', '═', '║', '╔', '╗', '╚', '╝', '╠', '╣', '╤', '╧', '╪', '╫', '│', '├', '┤', '┬', '┴', '┌', '┐', '└', '┘']
    if not line or all(char in box_chars for char in line.strip()):
        return None
    
    if '│' in line:
        parts = line.split('│')
        parts = [part.strip() for part in parts if part.strip()]
        
        if len(parts) >= 5:
            category = parts[0].strip()
            chapter_type = parts[1].strip()
            start_time = parts[2].strip()
            end_time = parts[3].strip()
            length = parts[4].strip()
            
            if any(not part.strip() or any(char in part for char in box_chars) for part in [category, chapter_type, start_time, end_time, length]):
                return None
            
            try:
                start_seconds = parse_timestamp(start_time)
                end_seconds = parse_timestamp(end_time)
                
                if start_seconds >= 0 and end_seconds > start_seconds:
                    return {
                        'category': category,
                        'type': chapter_type,
                        'start_time': start_seconds,
                        'end_time': end_seconds,
                        'duration': end_seconds - start_seconds,
                        'start_timestamp': start_time,
                        'end_timestamp': end_time,
                        'length': length
                    }
            except ValueError:
                pass
    
    parts = line.split()
    if len(parts) >= 4:
        try:
            category = parts[0]
            chapter_type = parts[1]
            start_time = parts[2]
            end_time = parts[3]
            length = parts[4] if len(parts) > 4 else ""
            
            start_seconds = parse_timestamp(start_time)
            end_seconds = parse_timestamp(end_time)
            
            if start_seconds >= 0 and end_seconds > start_seconds:
                return {
                    'category': category,
                    'type': chapter_type,
                    'start_time': start_seconds,
                    'end_time': end_seconds,
                    'duration': end_seconds - start_seconds,
                    'start_timestamp': start_time,
                    'end_timestamp': end_time,
                    'length': length
                }
        except (ValueError, IndexError):
            pass
    
    return None


def parse_timestamp(timestamp_str: str) -> int:
    """Parse HH:MM:SS or MM:SS timestamp to seconds"""
    try:
        parts = timestamp_str.split(':')
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes * 60 + seconds
        else:
            raise ValueError("Invalid timestamp format")
    except ValueError:
        raise ValueError(f"Invalid timestamp format: {timestamp_str}. Use HH:MM:SS or MM:SS")


def download_vod_chunk(vod_id: str, start_time: int, end_time: int, output_path: Path, quality: str = "720p") -> bool:
    """Download a specific chunk of VOD using TwitchDownloaderCLI (audio-only) with retries and fallbacks."""
    # Global download concurrency throttle
    global _TD_DL_SEM
    try:
        # Build timestamps
        start_timestamp = f"{start_time // 3600:02d}:{(start_time % 3600) // 60:02d}:{start_time % 60:02d}"
        end_timestamp = f"{end_time // 3600:02d}:{(end_time % 3600) // 60:02d}:{end_time % 60:02d}"

        # Work dir and env
        temp_dir = Path("/tmp/streamsniped_downloads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env['TEMP'] = str(temp_dir)
        env['TMP'] = str(temp_dir)
        env['TMPDIR'] = str(temp_dir)

        # Output temp path (CLI supports mp4/m4a)
        audio_output_path = output_path.with_suffix('.mp4')

        # CLI paths and options
        twitch_cli = os.getenv("TWITCH_DOWNLOADER_PATH", "TwitchDownloaderCLI")
        ffmpeg_path = os.getenv("FFMPEG_PATH")
        trim_mode = os.getenv("TD_TRIM_MODE", "Safe")
        threads = int(os.getenv("TD_THREADS", "4"))

        # Quality fallback list (try requested quality first)
        # Prefer audio-only first to minimize bandwidth; then step up through low→high video resolutions
        quality_order_env = os.getenv("DOWNLOAD_QUALITIES", "360p,480p,720p,720p60,1080p,1080p60")
        quality_order = [q.strip() for q in quality_order_env.split(',') if q.strip()]
        if quality and quality not in quality_order:
            quality_order.insert(0, quality)

        # Retry/backoff
        max_retries = int(os.getenv("TD_RETRY_LIMIT", "3"))
        backoff_base = float(os.getenv("TD_BACKOFF_BASE", "1.5"))

        print(f"Downloading audio chunk: {start_timestamp} - {end_timestamp}")

        with _TD_DL_SEM:
            for q in quality_order:
                for attempt in range(1, max_retries + 1):
                    # Clean any previous partial
                    try:
                        if audio_output_path.exists():
                            audio_output_path.unlink()
                    except Exception:
                        pass

                    cmd = [
                        twitch_cli,
                        "videodownload",
                        "--id", vod_id,
                        "-b", start_timestamp,
                        "-e", end_timestamp,
                        "-o", str(audio_output_path),
                        "-q", q,
                        "-t", str(threads),
                        "--trim-mode", trim_mode,
                    ]
                    if ffmpeg_path:
                        cmd.extend(["--ffmpeg-path", ffmpeg_path])

                    if attempt == 1:
                        print(f"📡 Command: {cmd[0]} {cmd[1]} --id ... -q {q} -t {threads} --trim-mode {trim_mode}")

                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
                    except subprocess.TimeoutExpired:
                        result = None

                    if result and result.returncode == 0 and audio_output_path.exists() and audio_output_path.stat().st_size > 0:
                        file_size = audio_output_path.stat().st_size / (1024 * 1024)
                        print(f"[SUCCESS] Audio chunk downloaded: {audio_output_path.name} ({file_size:.1f}MB)")
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            audio_output_path.rename(output_path)
                        except Exception:
                            # Fallback to copy then remove
                            import shutil
                            shutil.copyfile(str(audio_output_path), str(output_path))
                            try:
                                audio_output_path.unlink()
                            except Exception:
                                pass
                        return True

                    err = (result.stderr if result else 'timeout')
                    print(f"[WARN] Download attempt {attempt}/{max_retries} failed (q={q}): {err[:240] if err else ''}")
                    if attempt < max_retries:
                        import random
                        delay = backoff_base ** attempt + random.uniform(0.0, 0.8)
                        time.sleep(delay)
                # Next quality

        print("[ERROR] Audio chunk download failed after all retries/qualities")
        return False
    except Exception as e:
        print(f"[ERROR] Error downloading audio chunk: {e}")
        return False


def extract_audio_from_chunk(video_path: Path, audio_path: Path) -> bool:
    """Extract audio from video chunk using FFmpeg"""
    try:
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",  # No video
            "-ac", "1",  # Mono audio
            "-ar", "16000",  # 16kHz sample rate (optimal for Whisper)
            "-y",  # Overwrite output
            str(audio_path)
        ]
        
        print(f"Extracting audio from chunk...")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 minutes
        
        if result.returncode == 0 and audio_path.exists():
            file_size = audio_path.stat().st_size / (1024 * 1024)  # MB
            print(f"[SUCCESS] Audio extracted: {audio_path.name} ({file_size:.1f}MB)")
            return True
        else:
            print(f"[ERROR] Audio extraction failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Audio extraction timed out")
        return False
    except Exception as e:
        print(f"[ERROR] Error extracting audio: {e}")
        return False


def convert_audio_to_wav(input_path: Path, output_path: Path) -> bool:
    """Convert audio file to WAV format for Whisper compatibility"""
    try:
        cmd = [
            "ffmpeg",
            "-i", str(input_path),
            "-ac", "1",  # Mono audio
            "-ar", "16000",  # 16kHz sample rate (optimal for Whisper)
            "-y",  # Overwrite output
            str(output_path)
        ]
        
        print(f"Converting audio to WAV format...")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minutes
        
        if result.returncode == 0 and output_path.exists():
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"[SUCCESS] Audio converted to WAV: {output_path.name} ({file_size:.1f}MB)")
            return True
        else:
            print(f"[ERROR] Audio conversion failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Audio conversion timed out")
        return False
    except Exception as e:
        print(f"[ERROR] Error converting audio: {e}")
        return False


# Global Whisper model for reuse
_whisper_model = None
_TD_DL_SEM = BoundedSemaphore(max(1, int(os.environ.get('TD_MAX_PARALLEL', os.environ.get('AI_DATA_MAX_WORKERS', '4')))))

def get_whisper_model():
    """Get or create Whisper model singleton (faster-whisper preferred)."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    
    provider = getattr(config, 'whisper_provider', 'faster-whisper')
    try:
        import torch
        device = (
            "cuda" if torch.cuda.is_available() else (
                "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
            )
        ) if config.whisper_device == "auto" else config.whisper_device
    except Exception:
        device = "cpu"
    
    if provider == 'faster-whisper':
        try:
            from faster_whisper import WhisperModel
            compute_type = getattr(config, 'whisper_compute_type', 'int8')
            fw_device = device if device in ("cpu", "cuda") else "cpu"
            print(f"🔧 Loading faster-whisper '{config.whisper_model}' on {fw_device} ({compute_type})")
            _whisper_model = WhisperModel(config.whisper_model, device=fw_device, compute_type=compute_type)
            return _whisper_model
        except Exception as e:
            print(f" faster-whisper load failed ({e}), falling back to openai-whisper")
    
    # Fallback to openai-whisper
    import whisper
    print(f"🔧 Loading openai-whisper '{config.whisper_model}' on {device}")
    _whisper_model = whisper.load_model(config.whisper_model, device=device)
    return _whisper_model

# AssemblyAI helpers moved to src/transcription/assemblyai_client.py


def _extract_emotes_from_message(message: Dict) -> List[str]:
    """Extract a list of emote names from a TwitchDownloaderCLI message structure."""
    # Use the same logic as chat_utils.py for consistency
    try:
        # Handle different chat formats like chat_utils does
        if 'content_offset_seconds' in message:
            # TwitchDownloaderCLI format
            emotes = []
            if message.get('message', {}).get('fragments'):
                for fragment in message['message']['fragments']:
                    # Each fragment with an emoticon is an emote
                    if fragment.get('emoticon'):
                        # The fragment text is the emote name
                        emote_text = fragment.get('text', '')
                        if emote_text:
                            emotes.append(emote_text)
            
            # Fallback to legacy emoticons array
            if not emotes and message.get('message', {}).get('emoticons'):
                for emote in message['message']['emoticons']:
                    if isinstance(emote, dict) and 'emoticon_id' in emote:
                        emotes.append(f"emote_{emote['emoticon_id']}")
            
            return emotes
        else:
            # Simple format
            return message.get('emotes', [])
    except Exception:
        return []


def clean_chat(chat_data: List[Dict]) -> List[Dict]:
    """Clean chat by removing subscription messages and extracting emotes"""
    clean_messages = []
    filtered_count = 0
    
    if 'comments' in chat_data:
        messages = chat_data['comments']
    else:
        messages = chat_data
    
    # Use chat_utils for consistent emote parsing
    df = chat_utils.parse_chat_messages(messages)
    
    for _, row in df.iterrows():
        content = row['content']
        
        if is_sub_message(content):
            filtered_count += 1
            continue
        
        clean_message = {
            'timestamp': int(row['timestamp']),
            'content': content,
            'username': row['username'],
            'emotes': row['emotes'] if isinstance(row['emotes'], list) else []
        }
        
        clean_messages.append(clean_message)
    
    print(f"Cleaned chat: {len(clean_messages)} messages (filtered {filtered_count} sub messages)")
    return clean_messages


def is_sub_message(content: str) -> bool:
    """Check if message is a sub/prime message"""
    content_lower = content.lower()
    patterns = [
        "subscribed with prime",
        "subscribed at tier",
        "gifted a tier", 
        "gifted a sub",
        "is gifting",
        "gifted to",
        "they've subscribed for",
        "currently on a",
        "month streak",
        "gifted subs",
        "gifting subs",
        "raided the channel",
        "raid"
    ]
    return any(pattern in content_lower for pattern in patterns)


def is_spam_message(content: str) -> bool:
    """Simple spam detection - only filter system messages and links"""
    if not content or len(content.strip()) == 0:
        return True
    
    content_lower = content.lower().strip()
    
    # Only filter system messages and links
    gift_patterns = [
        r"gifted\s+(?:a\s+)?(?:tier\s+)?(?:sub(?:scription)?)",
        r"is\s+gifting",
        r"gifted\s+to",
        r"subscribed\s+(?:with\s+)?prime",
        r"prime\s+subscription",
        r"tier\s+\d+\s+subscription",
        r"gift\s+subscription",
        r"gifted\s+\d+\s+subs",
        r"gifting\s+\d+\s+subs",
    ]
    
    system_patterns = [
        r"raided\s+the\s+channel",
        r"hosted\s+the\s+channel",
        r"followed\s+the\s+channel",
        r"cheered\s+\d+",
        r"bits\s+donation",
        r"channel\s+point\s+redemption",
    ]
    
    # HTTP/HTTPS link patterns
    http_patterns = [
        r"https?://",  # Any http or https URL
        r"www\.",  # www. links
        r"\.com/",  # .com/ links
        r"\.org/",  # .org/ links
        r"\.net/",  # .net/ links
        r"\.io/",  # .io/ links
        r"youtu\.be/",  # YouTube short links
        r"open\.spotify\.com",  # Spotify links
        r"twitch\.tv/",  # Twitch links
        r"discord\.gg/",  # Discord invite links
        r"bit\.ly/",  # Bit.ly links
        r"tinyurl\.com/",  # TinyURL links
    ]
    
    all_patterns = gift_patterns + system_patterns + http_patterns
    
    for pattern in all_patterns:
        if re.search(pattern, content_lower):
            return True
    
    return False


def filter_chat_messages(chat_data: List[Dict]) -> List[ChatMessage]:
    """Filter and convert chat messages to efficient format"""
    filtered_messages = []
    message_id_counter = 0
    
    for msg in chat_data:
        if 'message' in msg and 'body' in msg['message']:
            content = msg['message']['body']
        elif 'content' in msg:
            content = msg['content']
        else:
            content = str(msg)
        
        if is_spam_message(content):
            continue
        
        timestamp = int(msg.get('content_offset_seconds', msg.get('timestamp', 0)))
        username = msg.get('username', msg.get('commenter', {}).get('display_name', 'viewer') if isinstance(msg.get('commenter'), dict) else 'viewer')
        # Extract emotes - handle both simple format and TwitchDownloaderCLI format
        emotes = []
        if 'emotes' in msg and isinstance(msg.get('emotes'), list):
            # Simple format with emotes list
            emotes = msg.get('emotes', [])
        else:
            # Try TwitchDownloaderCLI format
            emotes = _extract_emotes_from_message(msg)
        
        message = ChatMessage(
            message_id=f"msg_{message_id_counter:06d}",
            timestamp=timestamp,
            content=content,
            username=username,
            emotes=emotes or []
        )
        
        filtered_messages.append(message)
        message_id_counter += 1
    
    return filtered_messages


def get_chat_messages_for_chunk(chat_messages: List[ChatMessage], start_time: int, end_time: int) -> List[Dict]:
    """Get actual chat messages for a chunk"""
    chunk_messages = []
    
    for msg in chat_messages:
        if start_time <= msg.timestamp <= end_time:
            chunk_messages.append({
                'timestamp': msg.timestamp,
                'content': msg.content,
                'username': msg.username,
                'emotes': msg.emotes or []
            })
    
    return chunk_messages


def determine_chapter_type_for_time(timestamp: int, original_chapters: List[Dict]) -> str:
    """Determine the original chapter type for a given timestamp"""
    for chapter in original_chapters:
        if chapter['start_time'] <= timestamp < chapter['end_time']:
            return chapter.get('file_safe_name', 'unknown')
    return 'unknown'


def determine_chapter_category_for_time(timestamp: int, original_chapters: List[Dict]) -> str:
    """Determine the original chapter category for a given timestamp"""
    for chapter in original_chapters:
        if chapter['start_time'] <= timestamp < chapter['end_time']:
            return chapter.get('category', 'unknown')
    return 'unknown'


def determine_chapter_id_for_time(timestamp: int, original_chapters: List[Dict]) -> str:
    """Determine the original chapter ID for a given timestamp"""
    for chapter in original_chapters:
        if chapter['start_time'] <= timestamp < chapter['end_time']:
            return chapter.get('id', 'unknown')
    return 'unknown'


def process_vod_chunk(vod_id: str, start_time: int, end_time: int, chunk_name: str) -> Optional[Dict]:
    """Process a single VOD chunk: download audio, transcribe, and analyze"""
    print(f"\n Processing chunk: {chunk_name}")
    print(f"   Time range: {start_time}s - {end_time}s")
    print(f"   Duration: {end_time - start_time}s")
    
    # Create chunk-specific directories
    chunk_dir = config.get_chunk_dir(vod_id) / chunk_name
    chunk_dir.mkdir(parents=True, exist_ok=True)
    
    # Download audio chunk directly (no video needed for transcription)
    audio_path = chunk_dir / f"{vod_id}_{chunk_name}_audio.mp3"
    if not download_vod_chunk(vod_id, start_time, end_time, audio_path):
        print(f"[ERROR] Failed to download audio chunk: {chunk_name}")
        return None
    
    # Transcribe audio (AssemblyAI, via shared client)
    transcript = transcribe_audio_file(audio_path)
    if not transcript or not transcript.get('segments'):
        print(f"[ERROR] Failed to transcribe chunk: {chunk_name}")
        return None
    
    print(f"Audio-only processing - no video cleanup needed")
    
    return {
        'chunk_name': chunk_name,
        'start_time': start_time,
        'end_time': end_time,
        'duration': end_time - start_time,
        'transcript': transcript,
        'audio_path': audio_path,
        'video_path': None  # No video file since we download audio directly
    }


def save_chunk_data(chunk_data: Dict, vod_id: str) -> Path:
    """Save chunk data to file"""
    chunk_dir = config.get_chunk_dir(vod_id) / chunk_data['chunk_name']
    chunk_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = chunk_dir / f"{vod_id}_{chunk_data['chunk_name']}_data.json"
    
    # Convert to serializable format
    data = {
        'vod_id': vod_id,
        'chunk_name': chunk_data['chunk_name'],
        'start_time': chunk_data['start_time'],
        'end_time': chunk_data['end_time'],
        'duration': chunk_data['duration'],
        'transcript': chunk_data['transcript'],
        'audio_path': str(chunk_data['audio_path']),
        'metadata': {
            'segments_count': len(chunk_data['transcript'].get('segments', [])),
            'text_length': len(chunk_data['transcript'].get('text', '')),
            'created_at': str(Path().cwd())
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[SUCCESS] Chunk data saved: {output_path}")
    return output_path


def _save_chapters_to_path(chapters: List[Dict], vod_id: str, chapters_path: Path) -> Path:
    """Internal: save chapters list to the specified path with IDs and metadata."""
    chapters_with_ids: List[Dict] = []
    for i, chapter in enumerate(chapters):
        chapter_with_id = chapter.copy()
        if not chapter_with_id.get('id'):
            chapter_with_id['id'] = f"chapter_{i+1:03d}"
        if not chapter_with_id.get('file_safe_name'):
            chapter_with_id['file_safe_name'] = clean_chapter_name_standard(chapter_with_id.get('category', ''))
        chapters_with_ids.append(chapter_with_id)

    data = {
        'vod_id': vod_id,
        'chapters': chapters_with_ids,
        'metadata': {
            'total_chapters': len(chapters_with_ids),
            'total_duration': sum(ch.get('duration', 0) for ch in chapters_with_ids) if chapters_with_ids else 0,
            'chapter_system_version': '2.0'
        }
    }

    chapters_path.parent.mkdir(parents=True, exist_ok=True)
    with open(chapters_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f" Chapters saved: {chapters_path}")
    return chapters_path


def save_chapters_unmerged(chapters: List[Dict], vod_id: str) -> Path:
    """Save the raw/unmerged chapters for archival and vector-store metadata usage."""
    target = config.get_ai_data_dir(vod_id) / f"{vod_id}_chapters_unmerged.json"
    return _save_chapters_to_path(chapters, vod_id, target)


def save_chapters(chapters: List[Dict], vod_id: str) -> Path:
    """Save merged/final chapters to the canonical chapters.json path."""
    target = config.get_ai_data_dir(vod_id) / f"{vod_id}_chapters.json"
    return _save_chapters_to_path(chapters, vod_id, target)





def main():
    """Main function for cloud-optimized VOD processing with memory management"""
    # Set UTF-8 encoding for console output
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    if len(sys.argv) < 2:
        print("Usage: python generate_ai_data_cloud.py <vod_url> [--chunk-size <minutes>] [--start-time <HH:MM:SS>] [--end-time <HH:MM:SS>] [--force-regenerate] [--skip-cache] [--chapters-only] [--finalize-chapters-only]")
        print("  --chunk-size: Process VOD in chunks of N minutes (default: 3)")
        print("  --start-time: Start time in HH:MM:SS format")
        print("  --end-time: End time in HH:MM:SS format")
        print("  --force-regenerate: Force regeneration of AI data")
        print("  --skip-cache: Skip cache checks")
        print("  --chapters-only: Generate only chapters.json file")
        print("  --finalize-chapters-only: Rebuild chapters (unmerged + merged) and exit")
        sys.exit(1)
    
    # Check for help flag
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python generate_ai_data_cloud.py <vod_url> [--chunk-size <minutes>] [--start-time <HH:MM:SS>] [--end-time <HH:MM:SS>] [--force-regenerate] [--skip-cache] [--chapters-only] [--finalize-chapters-only]")
        print("  --chunk-size: Process VOD in chunks of N minutes (default: 3)")
        print("  --start-time: Start time in HH:MM:SS format")
        print("  --end-time: End time in HH:MM:SS format")
        print("  --force-regenerate: Force regeneration of AI data")
        print("  --skip-cache: Skip cache checks")
        print("  --chapters-only: Generate only chapters.json file")
        print("  --finalize-chapters-only: Rebuild chapters (unmerged + merged) and exit")
        sys.exit(0)
    
    vod_url = sys.argv[1]
    
    # Parse command line arguments
    chapters_only = "--chapters-only" in sys.argv
    finalize_only = "--finalize-chapters-only" in sys.argv
    
    # Get environment limits for memory optimization
    limits = get_environment_limits()
    
    # Parse optional arguments
    chunk_size_minutes = limits['chunk_duration'] // 60  # Use environment default
    start_time = None
    end_time = None
    force_regenerate = False
    skip_cache = False  # Default: USE cache/local files when available
    
    for i, arg in enumerate(sys.argv):
        if arg == "--chunk-size" and i + 1 < len(sys.argv):
            chunk_size_minutes = int(sys.argv[i + 1])
        elif arg == "--start-time" and i + 1 < len(sys.argv):
            start_time = parse_timestamp(sys.argv[i + 1])
        elif arg == "--end-time" and i + 1 < len(sys.argv):
            end_time = parse_timestamp(sys.argv[i + 1])
        elif arg == "--force-regenerate":
            force_regenerate = True
        elif arg == "--skip-cache":
            skip_cache = True
    
    print(f"Cloud-optimized processing for VOD: {vod_url}")
    print(f"Chunk size: {chunk_size_minutes} minutes")
    print(f"Max chunks: {limits['max_chunks']}")
    print(f"Memory threshold: {limits['memory_threshold_mb']}MB")
    print(f"Quality: {limits['quality']}")
    
    # Apply 4-minute cutoff to reduce intro screen time
    INTRO_CUTOFF_SECONDS = 240  # 4 minutes
    
    if start_time:
        # Apply cutoff to user-specified start time
        adjusted_start_time = start_time + INTRO_CUTOFF_SECONDS
        print(f"Original start time: {start_time // 3600:02d}:{(start_time % 3600) // 60:02d}:{start_time % 60:02d}")
        print(f"Adjusted start time (+4min): {adjusted_start_time // 3600:02d}:{(adjusted_start_time % 3600) // 60:02d}:{adjusted_start_time % 60:02d}")
        start_time = adjusted_start_time
    else:
        # Apply cutoff to beginning of VOD
        start_time = INTRO_CUTOFF_SECONDS
        print(f"Applied 4-minute intro cutoff: {start_time // 60:02d}:{start_time % 60:02d}")
    
    if end_time:
        print(f"End time: {end_time // 3600:02d}:{(end_time % 3600) // 60:02d}:{end_time % 60:02d}")
    
    # Initial memory check
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory['rss_mb']:.1f}MB")
    
    try:
        # Extract VOD ID
        if "twitch.tv/videos/" in vod_url:
            vod_id = vod_url.split("/videos/")[1].split("?")[0]
        else:
            vod_id = vod_url
        
        vod_id = vod_id.strip().strip('"').strip("'")
        print(f"VOD ID: {vod_id}")

        # Fast local short-circuit: if core outputs already exist, skip heavy work
        if not force_regenerate and not chapters_only and not finalize_only:
            ai_data_file_path = config.get_ai_data_dir(vod_id) / f"{vod_id}_ai_data.json"
            chat_clean_file_path = config.get_chat_dir(vod_id) / f"{vod_id}_chat_clean.json"
            if ai_data_file_path.exists() and chat_clean_file_path.exists():
                print("[SKIP] Local AI data and clean chat already exist.")
                print(f"[SKIP] AI Data: {ai_data_file_path}")
                print(f"[SKIP] Clean Chat: {chat_clean_file_path}")
                print(f"[SKIP] Ready for classification: python vod_review/scripts/classify_video_sections.py {vod_id} --focused")
                return
        
        # Check cache if not forcing regeneration
        if not force_regenerate and not skip_cache:
            try:
                # Add project root to path for imports if not already present
                project_root = str(Path(__file__).parent.parent)
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                
                from cache_utils import should_skip_with_cache, mark_step_completed
                
                # Check if AI data already exists
                ai_data_file = f"data/ai_data/{vod_id}/{vod_id}_ai_data.json"
                chat_file = f"data/chats/{vod_id}/{vod_id}_chat_clean.json"
                
                should_skip_result = should_skip_with_cache(
                    vod_id=vod_id,
                    step_name="generate_ai_data",
                    output_files=[ai_data_file, chat_file],
                    max_age_hours=24,
                    force_regenerate=force_regenerate
                )

                # Normalize return value: support both bool and (bool, reason)
                if isinstance(should_skip_result, tuple):
                    should_skip, reason = should_skip_result
                else:
                    should_skip = bool(should_skip_result)
                    reason = ""

                if should_skip:
                    print(f"Skipping AI data generation: {reason}")
                    print(f"Using cached AI data: {ai_data_file}")
                    print(f"Using cached chat: {chat_file}")
                    
                    # Mark step as completed in cache
                    mark_step_completed(
                        vod_id=vod_id,
                        step_name="generate_ai_data",
                        output_files=[ai_data_file, chat_file]
                    )
                    
                    print("Cloud-optimized processing complete (using cache)!")
                    print(f"AI Data: {ai_data_file}")
                    print(f"Clean Chat: {chat_file}")
                    print(f"Ready for classification: python vod_review/scripts/classify_video_sections.py {vod_id} --focused")
                    return
                    
            except ImportError:
                print(" Cache utilities not available, proceeding with full processing")
            except Exception as e:
                print(f" Cache check failed: {e}, proceeding with full processing")
        
        # Create video-specific directories
        config.ensure_video_directories(vod_id)
        
        # Get VOD info and chapters (cache-first via TwitchVodInfoProvider)
        print("[INFO] Getting VOD information...")
        try:
            # Add project root to path for imports if not already present
            project_root = str(Path(__file__).parent.parent)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from cam_detection.vod_info import TwitchVodInfoProvider  # type: ignore
            provider = TwitchVodInfoProvider()
            vod_info = provider.get_vod_info(vod_id)
            chapters = provider.get_vod_chapters(vod_id)
            print("[INFO] VOD info/chapters loaded via cache/provider")
        except Exception as e:
            print(f"[WARN] cam_detection provider unavailable ({e}); falling back to local helpers")
            vod_info = get_vod_info(vod_id)
            chapters = get_vod_chapters(vod_id)

        # Save VOD info to a file for later use (separate from provider cache)
        vod_info_path = config.get_ai_data_dir(vod_id) / f"{vod_id}_vod_info.json"
        try:
            with open(vod_info_path, 'w', encoding='utf-8') as f:
                json.dump(vod_info or {}, f, indent=2, ensure_ascii=False)
        except Exception as _e:
            print(f"[WARNING] Could not persist vod_info to AI data dir: {_e}")

        # Centralize light-weight stream context for downstream consumers
        try:
            ai_data_dir = config.get_ai_data_dir(vod_id)
            stream_context_path = ai_data_dir / f"{vod_id}_stream_context.json"
            # Derive canonical streamer, vod_title, duration
            streamer_name = (
                vod_info.get('Streamer')
                or vod_info.get('UserName')
                or vod_info.get('streamer')
                or vod_info.get('user_name')
                or ''
            ) if isinstance(vod_info, dict) else ''
            vod_title = (
                vod_info.get('Title')
                or vod_info.get('title')
                or vod_info.get('original_title')
                or ''
            ) if isinstance(vod_info, dict) else ''
            duration = (
                vod_info.get('duration')  # This is the converted seconds value
                or vod_info.get('Duration')
                or 0
            ) if isinstance(vod_info, dict) else 0

            # Collect chapter categories with both original and normalized forms
            categories: List[str] = []
            category_map: Dict[str, str] = {}  # normalized -> original
            try:
                for ch in chapters or []:
                    # Prefer original_category for display names
                    original = str(ch.get('original_category') or '').strip()
                    cat = str(ch.get('category') or '').strip()
                    file_safe = str(ch.get('file_safe_name') or '').strip()
                    
                    # Use the best available name
                    display_name = original or cat
                    if display_name:
                        categories.append(display_name)
                        # Map normalized forms back to original for matching
                        normalized = display_name.lower().replace('_', ' ').replace('+', ' ').strip()
                        if normalized:
                            category_map[normalized] = display_name
                        # Also map file_safe version if different
                        if file_safe and file_safe != normalized:
                            category_map[file_safe] = display_name
            except Exception:
                pass
            # Keep order but unique
            seen = set()
            unique_categories = []
            for c in categories:
                if c not in seen:
                    seen.add(c)
                    unique_categories.append(c)

            stream_context = {
                "vod_id": vod_id,
                "streamer": streamer_name,
                "vod_title": vod_title,
                "duration": duration,
                "chapter_categories": unique_categories,
                "category_map": category_map,  # for fuzzy matching
            }
            stream_context_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stream_context_path, 'w', encoding='utf-8') as f:
                json.dump(stream_context, f, indent=2, ensure_ascii=False)
            print(f"Stream context saved: {stream_context_path}")
        except Exception as e:
            print(f"[WARNING] Could not write stream context: {e}")

        # Save raw/unmerged chapters
        chapters_unmerged_path = save_chapters_unmerged(chapters, vod_id)
        print(f"Saved {len(chapters)} unmerged chapters for reference")

        # Merge and save merged chapters
        merged_chapters = merge_short_chapters(chapters, min_minutes=30)
        chapters_path = save_chapters(merged_chapters, vod_id)
        print(f"Saved merged chapters")
        
        # If chapters-only mode, upload to S3 and exit early
        if chapters_only or finalize_only:
            print(" Chapters-only mode: uploading chapter files to S3 and exiting...")
            
            # Upload chapters to S3 if in cloud environment
            container_mode = os.environ.get('CONTAINER_MODE', 'false').lower()
            if container_mode in ['true', '1', 'yes']:
                try:
                    storage = StorageManager()
                    s3_bucket = os.environ.get('S3_BUCKET', 'streamsniped-dev-videos')
                    # Upload merged
                    chapters_s3_key = f"ai_data/{vod_id}/{vod_id}_chapters.json"
                    chapters_s3_uri = f"s3://{s3_bucket}/{chapters_s3_key}"
                    storage.upload_file(str(chapters_path), chapters_s3_uri)
                    print(f" Chapters (merged) uploaded to S3: {chapters_s3_uri}")

                    # Upload unmerged
                    chapters_unmerged_s3_key = f"ai_data/{vod_id}/{vod_id}_chapters_unmerged.json"
                    chapters_unmerged_s3_uri = f"s3://{s3_bucket}/{chapters_unmerged_s3_key}"
                    storage.upload_file(str(chapters_unmerged_path), chapters_unmerged_s3_uri)
                    print(f" Chapters (unmerged) uploaded to S3: {chapters_unmerged_s3_uri}")
                except Exception as e:
                    print(f" Failed to upload chapters to S3: {e}")
            
            print("Chapters-only generation complete!")
            return
        
        # Determine processing chunks (split each chapter into 3-minute sub-chunks)
        if chapters:
            print(f"Processing {len(chapters)} raw chapters (splitting into {limits['chunk_duration']//60}-minute sub-chunks)")

            chunks = []
            chunk_size_seconds = limits['chunk_duration']

            for i, chapter in enumerate(chapters):
                chapter_category = chapter['category'].lower().replace(' ', '_')
                # Apply intro cutoff to chapter start
                chapter_start = max(chapter['start_time'], start_time)
                chapter_end = chapter['end_time']
                if chapter_end <= chapter_start:
                    continue

                # Slice chapter into fixed-size sub-chunks
                part_idx = 0
                current = chapter_start
                while current < chapter_end:
                    sub_end = min(current + chunk_size_seconds, chapter_end)
                    if sub_end > current:
                        chunk_name = f"chapter_{i:02d}_{chapter_category}_part_{part_idx:03d}"
                        chunks.append({
                            'name': chunk_name,
                            'start_time': current,
                            'end_time': sub_end,
                            'category': chapter['category']
                        })
                        part_idx += 1
                    current = sub_end

                print(f"Chapter {i:02d} → {part_idx} sub-chunks ({(chapter_end-chapter_start)//60}min)")
            print(f"Created {len(chunks)} sub-chunks from {len(chapters)} chapters")
            
        else:
            print(f"No chapters found, processing as single chunk")
            # If no chapters, create a single chunk or multiple chunks based on time range
            if start_time is not None and end_time is not None:
                chunks = [{
                    'name': 'full_range',
                    'start_time': start_time,
                    'end_time': end_time,
                    'category': 'Full Range'
                }]
            else:
                # Process in 5-minute chunks (optimized from 30-minute chunks)
                chunk_size_seconds = chunk_size_minutes * 60
                chunks = []
                current_time = start_time  # Start after intro cutoff
                chunk_num = 0
                
                # Determine total duration (fallback to Twitch API if needed)
                total_duration = vod_info.get('duration')
                if total_duration is None:
                    api_info = get_vod_info_from_api(vod_id)
                    total_duration = api_info.get('duration', 1800)  # Default 30 min if still unknown

                while current_time < total_duration:
                    chunk_end = min(current_time + chunk_size_seconds, total_duration)
                    chunks.append({
                        'name': f'chunk_{chunk_num:02d}',
                        'start_time': current_time,
                        'end_time': chunk_end,
                        'category': f'Chunk {chunk_num + 1}'
                    })
                    current_time = chunk_end
                    chunk_num += 1
        
        # Limit chunks to prevent memory issues
        if len(chunks) > limits['max_chunks']:
            print(f" Limiting chunks from {len(chunks)} to {limits['max_chunks']} to prevent memory issues")
            chunks = chunks[:limits['max_chunks']]
        
        print(f" Will process {len(chunks)} chunks")
        
        # Download and process chat
        print("[DOWNLOAD] Downloading chat...")
        chat_path = config.get_chat_dir(vod_id) / f"{vod_id}_chat.json"
        
        if chat_path.exists():
            print(f"[SUCCESS] Chat already exists: {chat_path.name}")
        else:
            chat_path = downloader.download_chat(vod_id, chat_path)
            print(f"[SUCCESS] Chat downloaded: {chat_path.name}")
        
        # Load and clean chat with memory management
        print("[CLEAN] Cleaning chat...")
        with open(chat_path, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
        
        clean_chat_data = clean_chat(chat_data)
        filtered_chat = filter_chat_messages(clean_chat_data)
        
        # Clear original chat data to free memory
        del chat_data
        force_garbage_collection()
        
        # Save clean chat
        clean_chat_path = config.get_chat_dir(vod_id) / f"{vod_id}_chat_clean.json"
        with open(clean_chat_path, 'w', encoding='utf-8') as f:
            json.dump(clean_chat_data, f, indent=2, ensure_ascii=False)
        
        print(f"[SUCCESS] Clean chat saved: {clean_chat_path}")
        
        # Process each chunk with optional concurrency
        processed_chunks = []
        temp_dir = Path("/tmp/streamsniped_downloads")

        concurrent = os.environ.get('AI_DATA_CONCURRENT', 'true').lower() in ['true', '1', 'yes']
        max_workers = max(1, int(os.environ.get('AI_DATA_MAX_WORKERS', os.environ.get('LLM_MAX_PARALLEL', '4'))))

        if concurrent and len(chunks) > 1:
            print(f"\nConcurrent chunk processing enabled (workers={max_workers})")

            def _worker_task(ch: Dict) -> Optional[Dict]:
                # Light memory guard inside workers
                if not check_memory_threshold(limits['memory_threshold_mb']):
                    force_garbage_collection()
                    cleanup_temp_files(temp_dir)
                return process_vod_chunk(
                    vod_id,
                    ch['start_time'],
                    ch['end_time'],
                    ch['name'],
                )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_chunk = {executor.submit(_worker_task, ch): ch for ch in chunks}
                for fut in as_completed(future_to_chunk):
                    ch = future_to_chunk[fut]
                    try:
                        chunk_data = fut.result()
                    except Exception:
                        chunk_data = None

                    if chunk_data:
                        # Get chat messages for this chunk
                        chunk_chat_messages = get_chat_messages_for_chunk(
                            filtered_chat,
                            ch['start_time'],
                            ch['end_time']
                        )

                        # Add chat data and save
                        chunk_data['chat_messages'] = chunk_chat_messages
                        chunk_data['chat_activity'] = len(chunk_chat_messages)
                        save_chunk_data(chunk_data, vod_id)
                        processed_chunks.append(chunk_data)

                        print(f" Chunk {ch['name']} processed successfully")
                        print(f"   Transcript segments: {len(chunk_data['transcript'].get('segments', []))}")
                        print(f"   Chat messages: {len(chunk_chat_messages)}")
                    else:
                        print(f"X Failed to process chunk {ch['name']}")
        else:
            for i, chunk in enumerate(chunks):
                print(f"\n Processing chunk {i+1}/{len(chunks)}: {chunk['name']} ({chunk['category']})")

                # Check memory before processing
                memory_before = get_memory_usage()
                print(f"Memory before chunk: {memory_before['rss_mb']:.1f}MB")

                if not check_memory_threshold(limits['memory_threshold_mb']):
                    print(f" Memory usage high ({memory_before['rss_mb']:.1f}MB), forcing cleanup...")
                    force_garbage_collection()
                    cleanup_temp_files(temp_dir)

                chunk_data = process_vod_chunk(
                    vod_id,
                    chunk['start_time'],
                    chunk['end_time'],
                    chunk['name']
                )

                if chunk_data:
                    # Get chat messages for this chunk
                    chunk_chat_messages = get_chat_messages_for_chunk(
                        filtered_chat,
                        chunk['start_time'],
                        chunk['end_time']
                    )

                    # Add chat data to chunk
                    chunk_data['chat_messages'] = chunk_chat_messages
                    chunk_data['chat_activity'] = len(chunk_chat_messages)

                    # Save chunk data
                    save_chunk_data(chunk_data, vod_id)
                    processed_chunks.append(chunk_data)

                    print(f" Chunk {chunk['name']} processed successfully")
                    print(f"   Transcript segments: {len(chunk_data['transcript'].get('segments', []))}")
                    print(f"   Chat messages: {len(chunk_chat_messages)}")

                    # Memory cleanup after each chunk
                    memory_after = get_memory_usage()
                    print(f"Memory after chunk: {memory_after['rss_mb']:.1f}MB")

                    # Force cleanup if memory usage is high
                    if memory_after['rss_mb'] > limits['memory_threshold_mb'] * 0.8:
                        print("Performing aggressive memory cleanup...")
                        force_garbage_collection()
                        cleanup_temp_files(temp_dir)
                else:
                    print(f"X Failed to process chunk {chunk['name']}")
        
        # Create combined AI data
        print(f"\nCreating combined AI data...")
        combined_segments = []
        
        for chunk in processed_chunks:
            transcript = chunk['transcript']
            segments = transcript.get('segments', [])
            
            for segment in segments:
                # Adjust timestamps to be relative to the full VOD
                adjusted_segment = segment.copy()
                adjusted_segment['start'] += chunk['start_time']
                adjusted_segment['end'] += chunk['start_time']
                
                combined_segments.append(adjusted_segment)
        
        # Create narrative segments from combined transcript
        narrative_segments = []
        segment_counter = 0
        
        for segment in combined_segments:
            start_time = int(segment['start'])
            end_time = int(segment['end'])
            text = segment['text'].strip()
            
            if not text:
                continue
            
            # Get chat messages for this segment
            segment_chat_messages = get_chat_messages_for_chunk(filtered_chat, start_time, end_time)
            
            # Determine original chapter information for this segment
            original_chapter_type = determine_chapter_type_for_time(start_time, chapters)
            original_chapter_category = determine_chapter_category_for_time(start_time, chapters)
            original_chapter_id = determine_chapter_id_for_time(start_time, chapters)
            
            narrative_segment = NarrativeSegment(
                segment_id=f"narrative_{segment_counter:03d}",
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                transcript=text,
                chat_messages=segment_chat_messages,
                chat_activity=len(segment_chat_messages),
                original_chapter_type=original_chapter_type,
                original_chapter_category=original_chapter_category,
                original_chapter_id=original_chapter_id
            )
            
            narrative_segments.append(narrative_segment)
            segment_counter += 1
        
        # Bridge gaps between adjacent narrative segments so chat in gaps isn't lost
        if narrative_segments:
            # Ensure order by start_time
            narrative_segments.sort(key=lambda s: s.start_time)
            for i in range(len(narrative_segments) - 1):
                current_seg = narrative_segments[i]
                next_seg = narrative_segments[i + 1]
                # Extend current segment to just before the next segment starts
                target_end = next_seg.start_time - 1
                if target_end > current_seg.end_time:
                    current_seg.end_time = target_end
                    current_seg.duration = current_seg.end_time - current_seg.start_time

            # Recompute chat messages for updated time spans
            for seg in narrative_segments:
                seg.chat_messages = get_chat_messages_for_chunk(
                    filtered_chat,
                    seg.start_time,
                    seg.end_time
                )
                seg.chat_activity = len(seg.chat_messages)
        
        # Save combined AI data
        ai_data_dir = config.get_ai_data_dir(vod_id)
        ai_data_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = ai_data_dir / f"{vod_id}_ai_data.json"
        
        segments_data = []
        for segment in narrative_segments:
            segments_data.append({
                "segment_id": segment.segment_id,
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "duration": segment.duration,
                "transcript": segment.transcript,
                "chat_messages": segment.chat_messages,
                "chat_activity": segment.chat_activity
            })
        
        data = {
            "vod_id": vod_id,
            "segments": segments_data,
            "metadata": {
                "total_segments": len(segments_data),
                "total_chunks": len(processed_chunks),
                "format_version": "3.0",
                "processing_method": "cloud_optimized_chunks",
                "improvements": [
                    "memory_efficient_chunk_processing",
                    "cloud_optimized_architecture",
                    "eliminated_full_vod_download",
                    "enhanced_spam_filtering",
                    "direct_audio_to_ai_data"
                ]
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Upload results to S3 if in cloud mode
        container_mode = os.environ.get('CONTAINER_MODE', 'false').lower()
        s3_bucket = os.environ.get('S3_BUCKET', 'streamsniped-dev-videos')
        
        # Check for explicit disable flag
        disable_s3 = os.environ.get('DISABLE_S3_UPLOADS', 'false').lower() in ['true', '1', 'yes']
        
        # Force S3 upload if we're in AWS environment (detect by checking for AWS_REGION)
        aws_region = os.environ.get('AWS_REGION')
        force_s3_upload = aws_region is not None and s3_bucket != 'streamsniped-dev-data'
        
        if disable_s3:
            print("S3 uploads explicitly disabled via DISABLE_S3_UPLOADS")
            force_s3_upload = False
            container_mode = 'false'
        
        if container_mode in ['true', '1', 'yes'] or force_s3_upload:
            print(f"\nUploading results to S3...")
            storage = StorageManager()
            
            # Get S3 bucket from environment with fallback
            s3_bucket = os.environ.get('S3_BUCKET', 'streamsniped-dev-videos')
            print(f"Using S3 bucket: {s3_bucket}")
            
            # Verify S3 access
            try:
                import boto3
                s3_client = boto3.client('s3')
                s3_client.head_bucket(Bucket=s3_bucket)
                print(f" S3 bucket '{s3_bucket}' is accessible")
            except Exception as e:
                print(f"X S3 bucket access failed: {e}")
                print(f" Skipping S3 upload due to access issues")
                s3_bucket = None
            
            # Only proceed with uploads if S3 bucket is accessible
            if s3_bucket:
                # Upload AI data
                ai_data_s3_key = f"ai_data/{vod_id}/{vod_id}_ai_data.json"
                ai_data_s3_uri = f"s3://{s3_bucket}/{ai_data_s3_key}"
                try:
                    storage.upload_file(str(output_path), ai_data_s3_uri)
                    print(f" AI Data uploaded: {ai_data_s3_uri}")
                except Exception as e:
                    print(f"X Failed to upload AI data: {e}")
                
                # Upload clean chat
                chat_s3_key = f"chats/{vod_id}/{vod_id}_chat_clean.json"
                chat_s3_uri = f"s3://{s3_bucket}/{chat_s3_key}"
                try:
                    storage.upload_file(str(clean_chat_path), chat_s3_uri)
                    print(f" Clean chat uploaded: {chat_s3_uri}")
                except Exception as e:
                    print(f"X Failed to upload clean chat: {e}")
                
                # Upload chapters if they exist
                if chapters:
                    # Merged
                    chapters_s3_key = f"ai_data/{vod_id}/{vod_id}_chapters.json"
                    chapters_s3_uri = f"s3://{s3_bucket}/{chapters_s3_key}"
                    try:
                        storage.upload_file(str(chapters_path), chapters_s3_uri)
                        print(f" Chapters (merged) uploaded: {chapters_s3_uri}")
                    except Exception as e:
                        print(f"X Failed to upload merged chapters: {e}")

                    # Unmerged
                    chapters_unmerged_s3_key = f"ai_data/{vod_id}/{vod_id}_chapters_unmerged.json"
                    chapters_unmerged_s3_uri = f"s3://{s3_bucket}/{chapters_unmerged_s3_key}"
                    try:
                        storage.upload_file(str(chapters_unmerged_path), chapters_unmerged_s3_uri)
                        print(f" Chapters (unmerged) uploaded: {chapters_unmerged_s3_uri}")
                    except Exception as e:
                        print(f"X Failed to upload unmerged chapters: {e}")
                
                # Upload chunk data
                chunk_dir = config.get_chunk_dir(vod_id)
                uploaded_chunks = 0
                for chunk in processed_chunks:
                    chunk_name = chunk['chunk_name']
                    chunk_data_path = chunk_dir / chunk_name / f"{vod_id}_{chunk_name}_data.json"
                    if chunk_data_path.exists():
                        chunk_s3_key = f"chunks/{vod_id}/{chunk_name}/{vod_id}_{chunk_name}_data.json"
                        chunk_s3_uri = f"s3://{s3_bucket}/{chunk_s3_key}"
                        try:
                            storage.upload_file(str(chunk_data_path), chunk_s3_uri)
                            print(f" Chunk data uploaded: {chunk_s3_uri}")
                            uploaded_chunks += 1
                        except Exception as e:
                            print(f"X Failed to upload chunk data: {e}")
                
                print(f"S3 Upload Summary:")
                print(f"   AI Data: ")
                print(f"   Clean Chat: ")
                print(f"   Chapters: {'' if chapters else 'X'}")
                print(f"   Chunk Files: {uploaded_chunks}/{len(processed_chunks)}")
            else:
                print(f" S3 uploads skipped due to bucket access issues")
        
        # Mark step as completed in cache
        if not skip_cache:
            try:
                from cache_utils import mark_step_completed
                # Prepare output files list - only core AI data and chat (chapters handled separately)
                output_files = [str(output_path), str(clean_chat_path)]
                
                mark_step_completed(
                    vod_id=vod_id,
                    step_name="generate_ai_data",
                    output_files=output_files
                )
            except ImportError:
                print(" Cache utilities not available, skipping cache marking")
            except Exception as e:
                print(f" Failed to mark step in cache: {e}")
        
        print(f"\nCloud-optimized processing complete!")
        print(f"AI Data: {output_path}")
        print(f"Clean Chat: {clean_chat_path}")
        print(f"Chunk Data: {config.get_chunk_dir(vod_id)}")
        print(f"Processed {len(processed_chunks)} chunks")
        print(f"Created {len(narrative_segments)} narrative segments")
        print(f"Ready for classification: python vod_review/scripts/classify_video_sections.py {vod_id} --focused")
        
    except Exception as e:
        print(f"[ERROR] Cloud-optimized processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 