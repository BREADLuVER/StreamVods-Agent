"""
VOD and chat download functionality using TwitchDownloaderCLI
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests
from loguru import logger

from .config import config


class DownloadError(Exception):
    """Raised when download fails"""
    pass


class TwitchDownloader:
    """Handles VOD and chat downloads using TwitchDownloaderCLI"""
    
    def __init__(self):
        self.twitch_downloader_path = self._find_twitch_downloader()
        self._validate_tools()
    
    def _find_twitch_downloader(self) -> str:
        """Find TwitchDownloaderCLI executable"""
        if config.twitch_downloader_path:
            path = Path(config.twitch_downloader_path)
            if path.exists():
                return str(path)
            raise DownloadError(f"TwitchDownloaderCLI not found at {path}")
        
        # Try common names
        possible_names = [
            "TwitchDownloaderCLI",
            "TwitchDownloaderCLI.exe",
            "twitch-downloader",
        ]
        
        for name in possible_names:
            try:
                result = subprocess.run(
                    [name, "--help"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return name
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        raise DownloadError(
            "TwitchDownloaderCLI not found. Please install it and add to PATH, "
            "or set TWITCH_DOWNLOADER_PATH in .env"
        )
    
    def _validate_tools(self) -> None:
        """Validate that required tools are available"""
        # Check FFmpeg
        try:
            ffmpeg_cmd = [config.ffmpeg_path or "ffmpeg", "-version"]
            subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise DownloadError(
                "FFmpeg not found. Please install FFmpeg and add to PATH, "
                "or set FFMPEG_PATH in .env"
            )
    
    def extract_video_id(self, url: str) -> str:
        """Extract video ID from Twitch VOD URL"""
        patterns = [
            r"twitch\.tv/videos/(\d+)",
            r"twitch\.tv/video/(\d+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise DownloadError(f"Could not extract video ID from URL: {url}")
    
    def get_video_info(self, video_id: str) -> dict:
        """Get video information from Twitch API"""
        url = f"https://www.twitch.tv/videos/{video_id}"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Extract video info from page
            # This is a simplified approach - in production you might want to use Twitch API
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', response.text)
            title = title_match.group(1) if title_match else f"VOD {video_id}"
            
            return {
                "id": video_id,
                "title": title,
                "url": url,
            }
        except requests.RequestException as e:
            raise DownloadError(f"Failed to get video info: {e}")
    
    def download_vod(self, video_id: str, output_path: Path) -> Path:
        """Download VOD using TwitchDownloaderCLI"""
        if output_path.exists() and not config.force_download:
            logger.info(f"VOD already exists at {output_path}")
            return output_path
        
        cmd = [
            self.twitch_downloader_path,
            "videodownload",
            "--id", video_id,
            "-o", str(output_path),
            "-q", config.video_quality,
        ]
        
        logger.info(f"Downloading VOD {video_id} to {output_path}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                raise DownloadError(
                    f"VOD download failed: {result.stderr}"
                )
            
            if not output_path.exists():
                raise DownloadError("VOD file not created after download")
            
            logger.success(f"VOD downloaded successfully: {output_path}")
            return output_path
            
        except subprocess.TimeoutExpired:
            raise DownloadError("VOD download timed out")
        except Exception as e:
            raise DownloadError(f"VOD download failed: {e}")
    
    def download_chat(self, video_id: str, output_path: Path) -> Path:
        """Download chat using TwitchDownloaderCLI and create clean version"""
        if output_path.exists() and not config.force_download:
            logger.info(f"Chat already exists at {output_path}")
            return output_path
        
        cmd = [
            self.twitch_downloader_path,
            "chatdownload",
            "--id", video_id,
            "--embed-images",
            "--bttv=true",
            "--ffz=true",
            "--stv=true",
            "-o", str(output_path),
        ]
        
        logger.info(f"Downloading chat for VOD {video_id} to {output_path}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                raise DownloadError(
                    f"Chat download failed: {result.stderr}"
                )
            
            if not output_path.exists():
                raise DownloadError("Chat file not created after download")
            
            logger.success(f"Chat downloaded successfully: {output_path}")
            
            # Create clean chat file (filtered for clip detection)
            self._create_clean_chat(output_path)
            
            return output_path
            
        except subprocess.TimeoutExpired:
            raise DownloadError("Chat download timed out")
        except Exception as e:
            raise DownloadError(f"Chat download failed: {e}")
    
    def _create_clean_chat(self, chat_path: Path) -> None:
        """Create a clean version of chat file without sub/prime messages"""
        clean_path = chat_path.parent / f"{chat_path.stem}_clean.json"
        
        if clean_path.exists() and not config.force_download:
            logger.info(f"Clean chat already exists at {clean_path}")
            return
        
        logger.info(f"Creating clean chat file: {clean_path}")
        
        try:
            # Load original chat
            with open(chat_path, 'r', encoding='utf-8') as f:
                chat_data = json.load(f)
            
            # Handle different chat data structures
            if isinstance(chat_data, dict) and 'comments' in chat_data:
                # New TwitchDownloaderCLI format
                messages = chat_data['comments']
            elif isinstance(chat_data, list):
                # Old format (direct array)
                messages = chat_data
            else:
                logger.error(f"Unknown chat data format: {type(chat_data)}")
                return
            
            # Filter out sub/prime messages
            clean_messages = []
            filtered_count = 0
            
            for message in messages:
                # Extract content from different possible structures
                if isinstance(message, dict):
                    if 'message' in message and 'body' in message['message']:
                        content = message['message']['body']
                    elif 'content' in message:
                        content = message['content']
                    else:
                        content = ''
                else:
                    content = str(message)
                
                # Skip sub/prime messages
                if self._is_sub_message(content):
                    filtered_count += 1
                    continue
                
                clean_messages.append(message)
            
            # Save clean chat
            with open(clean_path, 'w', encoding='utf-8') as f:
                json.dump(clean_messages, f, indent=2, ensure_ascii=False)
            
            logger.success(f"Clean chat created: {len(clean_messages)} messages (filtered {filtered_count} sub messages)")
            
        except Exception as e:
            logger.error(f"Failed to create clean chat: {e}")
            # Don't fail the whole download if clean chat creation fails
    
    def _is_sub_message(self, content: str) -> bool:
        """Check if a message is a sub/prime message that should be filtered"""
        content_lower = content.lower()
        
        # Filter patterns
        patterns = [
            "subscribed with prime",
            "gifted a tier",
            "gifted a sub",
            "is gifting",
            "gifted to",
        ]
        
        for pattern in patterns:
            if pattern in content_lower:
                return True
        
        return False
    
    def download(self, vod_url: str) -> Tuple[Path, Path, dict]:
        """Download VOD and chat, return paths and video info"""
        # Extract video ID
        video_id = self.extract_video_id(vod_url)
        logger.info(f"Extracted video ID: {video_id}")
        
        # Ensure video-specific directories exist
        config.ensure_video_directories(video_id)
        
        # Get video info
        video_info = self.get_video_info(video_id)
        logger.info(f"Video title: {video_info['title']}")
        
        # Generate output paths using video-specific directories
        safe_title = re.sub(r'[^\w\-_.]', '_', video_info['title'])
        vod_path = config.get_vod_dir(video_id) / f"{video_id}_{safe_title}.mp4"
        chat_path = config.get_chat_dir(video_id) / f"{video_id}_chat.json"
        
        # Download VOD and chat in parallel
        try:
            vod_path = self.download_vod(video_id, vod_path)
            chat_path = self.download_chat(video_id, chat_path)
            
            return vod_path, chat_path, video_info
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise


# Global downloader instance
downloader = TwitchDownloader() 