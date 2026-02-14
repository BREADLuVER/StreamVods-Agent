"""
Configuration management for StreamSniped
"""

import os
from pathlib import Path
from typing import Optional

import pydantic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config(pydantic.BaseModel):
    """Application configuration"""
    
    # Processing settings
    max_workers: int = pydantic.Field(default=4, ge=1, le=16)
    
    # Phase 1: Clip Generation settings
    default_clip_count: str = "unlimited"  # "unlimited" or number
    min_clip_duration: int = pydantic.Field(default=20, ge=10, le=600)
    max_clip_duration: int = pydantic.Field(default=300, ge=30, le=1800)
    lead_in_buffer: int = pydantic.Field(default=10, ge=0, le=60)
    lead_out_buffer: int = pydantic.Field(default=10, ge=0, le=60)
    
    # Scoring settings
    chat_threshold: float = pydantic.Field(default=5.0, ge=0.0, le=20.0)  # Legacy, kept for backward compatibility
    z_threshold: float = pydantic.Field(default=4.0, ge=0.0, le=20.0)  # Local z-score threshold
    msg_threshold: int = pydantic.Field(default=20, ge=1, le=1000)  # Minimum message count per second
    
    # Score filtering thresholds (centralized)
    ai_direct_clipper_score_threshold: float = pydantic.Field(default=7.5, ge=0.0, le=10.0)  # AI Direct Clipper minimum score
    classification_sections_score_threshold: float = pydantic.Field(default=7.5, ge=0.0, le=10.0)  # Classification sections minimum score
    highlight_score_threshold: float = pydantic.Field(default=7.5, ge=0.0, le=10.0)  # General highlight minimum score
    
    # Adaptive threshold settings
    use_adaptive_thresholds: bool = pydantic.Field(default=True)  # Use percentile-based adaptive thresholds
    sensitivity: str = pydantic.Field(default="medium")  # low, medium, high sensitivity
    min_msg_count: int = pydantic.Field(default=3, ge=1, le=10)  # Minimum messages to consider (filters silence)
    max_clips_per_hour: int = pydantic.Field(default=5, ge=1, le=999)  # Limit to 5 clips per hour
    
    # Phase 2: VOD Review settings
    chunk_duration: int = pydantic.Field(default=180, ge=60, le=600)  # Chunk duration in seconds (2-10 min)
    chunk_overlap: int = pydantic.Field(default=30, ge=0, le=120)  # Overlap between chunks
    max_narrative_duration: int = pydantic.Field(default=3600, ge=300, le=7200)  # Max final video duration (5min-2hr)
    min_chunk_score: float = pydantic.Field(default=5.0, ge=0.0, le=10.0)  # Minimum score for chunks to be included
    
    # File paths
    data_dir: Path = Path("./data")
    vod_dir: Path = Path("./data/vods")
    chat_dir: Path = Path("./data/chats")
    chunk_dir: Path = Path("./data/chunks")
    clip_dir: Path = Path("./data/clips")
    temp_dir: Path = Path("./data/temp")
    transcript_dir: Path = Path("./data/transcripts")
    narrative_dir: Path = Path("./data/narratives")
    
    # TwitchDownloaderCLI settings
    twitch_downloader_path: Optional[str] = None
    video_quality: str = "1080p"
    
    # FFmpeg settings
    ffmpeg_path: Optional[str] = None
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    
    # Whisper settings
    whisper_path: Optional[str] = None
    whisper_model: str = "medium.en"  # model size (e.g., base, small, medium, medium.en)
    whisper_device: str = "auto"  # auto, cpu, cuda, mps
    whisper_fp16: bool = True  # Use half precision for GPU
    whisper_optimize: bool = True  # Enable optimizations
    whisper_provider: str = "faster-whisper"  # faster-whisper | openai-whisper
    whisper_compute_type: str = "int8"  # faster-whisper: int8, int8_float16, float16, float32
    whisper_beam_size: int = 5  # Beam search width (accuracy)
    whisper_best_of: int = 5  # Number of candidates when sampling (used when temperature > 0)
    whisper_temperature: float = 0.0  # Deterministic by default
    whisper_word_timestamps: bool = True  # Word-level timestamps enabled
    whisper_initial_prompt: Optional[str] = None  # Domain prompt, e.g., Twitch slang
    whisper_language: str = "en"  # Language hint
    
    # AI settings
    openai_api_key: Optional[str] = None
    gpt_model: str = "gpt-3.5-turbo"  # gpt-3.5-turbo, gpt-4
    gemini_api_key: Optional[str] = None
    
    # GPU and Cloud settings
    container_mode: bool = False
    gpu_enabled: bool = False
    cloud_environment: bool = False
    
    # Logging
    log_level: str = "INFO"
    verbose: bool = False
    save_logs: bool = True
    
    # Development
    debug: bool = False
    keep_temp: bool = False
    force_download: bool = False
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables"""
        # Detect environment
        container_mode = os.getenv("CONTAINER_MODE", "false").lower() == "true"
        gpu_enabled = os.getenv("GPU_ENABLED", "false").lower() == "true"
        cloud_environment = os.getenv("AWS_REGION") is not None or container_mode
        
        return cls(
            # Processing settings
            max_workers=int(os.getenv("MAX_WORKERS", "4")),
            
            # Phase 1: Clip Generation settings
            default_clip_count=os.getenv("DEFAULT_CLIP_COUNT", "unlimited"),
            min_clip_duration=int(os.getenv("MIN_CLIP_DURATION", "20")),
            max_clip_duration=int(os.getenv("MAX_CLIP_DURATION", "300")),
            lead_in_buffer=int(os.getenv("LEAD_IN_BUFFER", "10")),
            lead_out_buffer=int(os.getenv("LEAD_OUT_BUFFER", "10")),
            
            # Scoring settings
            chat_threshold=float(os.getenv("CHAT_THRESHOLD", "5.0")),
            z_threshold=float(os.getenv("Z_THRESHOLD", "4.0")),
            msg_threshold=int(os.getenv("MSG_THRESHOLD", "20")),
            
            # Score filtering thresholds (centralized)
            ai_direct_clipper_score_threshold=float(os.getenv("AI_DIRECT_CLIPPER_SCORE_THRESHOLD", "7.5")),
            classification_sections_score_threshold=float(os.getenv("CLASSIFICATION_SECTIONS_SCORE_THRESHOLD", "7.5")),
            highlight_score_threshold=float(os.getenv("HIGHLIGHT_SCORE_THRESHOLD", "7.5")),
            
            # Adaptive threshold settings
            use_adaptive_thresholds=os.getenv("USE_ADAPTIVE_THRESHOLDS", "true").lower() == "true",
            sensitivity=os.getenv("SENSITIVITY", "medium"),
            min_msg_count=int(os.getenv("MIN_MSG_COUNT", "3")),
            max_clips_per_hour=int(os.getenv("MAX_CLIPS_PER_HOUR", "5")),
            
            # Phase 2: VOD Review settings
            chunk_duration=int(os.getenv("CHUNK_DURATION", "180")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "30")),
            max_narrative_duration=int(os.getenv("MAX_NARRATIVE_DURATION", "3600")),
            min_chunk_score=float(os.getenv("MIN_CHUNK_SCORE", "5.0")),
            
            # File paths
            data_dir=Path(os.getenv("DATA_DIR", "./data")),
            vod_dir=Path(os.getenv("VOD_DIR", "./data/vods")),
            chat_dir=Path(os.getenv("CHAT_DIR", "./data/chats")),
            chunk_dir=Path(os.getenv("CHUNK_DIR", "./data/chunks")),
            clip_dir=Path(os.getenv("CLIP_DIR", "./data/clips")),
            temp_dir=Path(os.getenv("TEMP_DIR", "./data/temp")),
            transcript_dir=Path(os.getenv("TRANSCRIPT_DIR", "./data/transcripts")),
            narrative_dir=Path(os.getenv("NARRATIVE_DIR", "./data/narratives")),
            
            # TwitchDownloaderCLI settings
            twitch_downloader_path=os.getenv("TWITCH_DOWNLOADER_PATH"),
            video_quality=os.getenv("VIDEO_QUALITY", "1080p"),
            
            # FFmpeg settings
            ffmpeg_path=os.getenv("FFMPEG_PATH"),
            video_codec=os.getenv("VIDEO_CODEC", "libx264"),
            audio_codec=os.getenv("AUDIO_CODEC", "aac"),
            
            # Whisper settings
            whisper_path=os.getenv("WHISPER_PATH"),
            whisper_model=os.getenv("WHISPER_MODEL", "medium.en"),
            whisper_device=os.getenv("WHISPER_DEVICE", "auto"),
            whisper_fp16=os.getenv("WHISPER_FP16", "true").lower() == "true",
            whisper_optimize=os.getenv("WHISPER_OPTIMIZE", "true").lower() == "true",
            whisper_provider=os.getenv("WHISPER_PROVIDER", "faster-whisper"),
            whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            whisper_beam_size=int(os.getenv("WHISPER_BEAM_SIZE", "5")),
            whisper_best_of=int(os.getenv("WHISPER_BEST_OF", "5")),
            whisper_temperature=float(os.getenv("WHISPER_TEMPERATURE", "0.0")),
            whisper_word_timestamps=os.getenv("WHISPER_WORD_TIMESTAMPS", "true").lower() == "true",
            whisper_initial_prompt=os.getenv("WHISPER_INITIAL_PROMPT") or None,
            whisper_language=os.getenv("WHISPER_LANGUAGE", "en"),
            
            # AI settings
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            gpt_model=os.getenv("GPT_MODEL", "gpt-3.5-turbo"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            
            # GPU and Cloud settings
            container_mode=container_mode,
            gpu_enabled=gpu_enabled,
            cloud_environment=cloud_environment,
            
            # Logging
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            verbose=os.getenv("VERBOSE", "false").lower() == "true",
            save_logs=os.getenv("SAVE_LOGS", "true").lower() == "true",
            
            # Development
            debug=os.getenv("DEBUG", "false").lower() == "true",
            keep_temp=os.getenv("KEEP_TEMP", "false").lower() == "true",
            force_download=os.getenv("FORCE_DOWNLOAD", "false").lower() == "true"
        )
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        directories = [
            self.data_dir,
            self.vod_dir,
            self.chat_dir,
            self.chunk_dir,
            self.clip_dir,
            self.temp_dir,
            self.transcript_dir,
            self.narrative_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    # Video-specific path helpers
    def get_vod_dir(self, vod_id: str) -> Path:
        """Get video-specific VOD directory"""
        return self.vod_dir / vod_id
    
    def get_chat_dir(self, vod_id: str) -> Path:
        """Get video-specific chat directory"""
        return self.chat_dir / vod_id
    
    def get_transcript_dir(self, vod_id: str) -> Path:
        """Get video-specific transcript directory"""
        return self.transcript_dir / vod_id
    
    def get_ai_data_dir(self, vod_id: str) -> Path:
        """Get video-specific AI data directory"""
        return self.data_dir / "ai_data" / vod_id
    
    def get_focused_dir(self, vod_id: str) -> Path:
        """Get video-specific focused analysis directory"""
        return self.data_dir / "ai_data" / vod_id / "focused"
    
    def get_clip_dir(self, vod_id: str) -> Path:
        """Get video-specific clip directory"""
        return self.clip_dir / vod_id
    
    def get_chunk_dir(self, vod_id: str) -> Path:
        """Get video-specific chunk directory"""
        return self.chunk_dir / vod_id
    
    def get_narrative_dir(self, vod_id: str) -> Path:
        """Get video-specific narrative directory"""
        return self.narrative_dir / vod_id
    
    def get_temp_dir(self, vod_id: str) -> Path:
        """Get video-specific temp directory"""
        return self.temp_dir / vod_id
    
    def get_chat_context_dir(self, vod_id: str) -> Path:
        """Get video-specific chat context directory"""
        return self.data_dir / "chat_contexts" / vod_id
    
    def ensure_video_directories(self, vod_id: str) -> None:
        """Create all video-specific directories"""
        directories = [
            self.get_vod_dir(vod_id),
            self.get_chat_dir(vod_id),
            self.get_transcript_dir(vod_id),
            self.get_ai_data_dir(vod_id),
            self.get_focused_dir(vod_id),
            self.get_clip_dir(vod_id),
            self.get_chunk_dir(vod_id),
            self.get_narrative_dir(vod_id),
            self.get_temp_dir(vod_id),
            self.get_chat_context_dir(vod_id),
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def clip_count(self) -> Optional[int]:
        """Get clip count as integer or None for unlimited"""
        if self.default_clip_count == "unlimited":
            return None
        return int(self.default_clip_count)


# Global config instance
config = Config.from_env() 