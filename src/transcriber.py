"""
Audio transcription module for StreamSniped
Uses Whisper to transcribe audio clips with GPU optimization
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

from .config import config


class Transcriber:
    """Handles audio transcription using Whisper with GPU optimization"""
    
    def __init__(self):
        self.model = config.whisper_model
        self.cache_dir = config.transcript_dir
        self.cache_dir.mkdir(exist_ok=True)
        
        # GPU optimization
        self.device = self._get_optimal_device()
        self.model_instance = None  # Cache the model instance
        self.provider = getattr(config, 'whisper_provider', 'openai-whisper')
        
    def _get_optimal_device(self) -> str:
        """Get the best available device for transcription"""
        if torch.cuda.is_available():
            logger.info("🚀 Using CUDA GPU for transcription")
            return "cuda"
        elif torch.backends.mps.is_available():
            logger.info("🍎 Using Apple Silicon GPU for transcription")
            return "mps"
        else:
            logger.info("💻 Using CPU for transcription")
            return "cpu"
    
    def _load_model(self):
        """Load and cache the Whisper model (openai-whisper or faster-whisper)."""
        if self.model_instance is not None:
            return self.model_instance
        
        if self.provider == 'faster-whisper':
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                logger.error("faster-whisper not installed. Install with: pip install faster-whisper")
                raise
            compute_type = getattr(config, 'whisper_compute_type', 'int8')
            # Map unsupported devices
            fw_device = self.device if self.device in ("cpu", "cuda") else "cpu"
            logger.info(f"📦 Loading faster-whisper model: {self.model} on {fw_device} ({compute_type})")
            try:
                self.model_instance = WhisperModel(self.model, device=fw_device, compute_type=compute_type)
            except Exception as e:
                logger.error(f"Failed to load faster-whisper model: {e}")
                raise
            return self.model_instance
        
        # Fallback to openai-whisper
        try:
            import whisper
            logger.info(f"📦 Loading openai-whisper model: {self.model} on {self.device}")
            self.model_instance = whisper.load_model(self.model, device=self.device)
            if self.device == "cuda" and hasattr(self.model_instance, 'half') and config.whisper_fp16:
                try:
                    self.model_instance = self.model_instance.half()
                    logger.info("⚡ Model optimized with half precision for GPU")
                except Exception:
                    pass
        except ImportError:
            logger.error("openai-whisper not installed. Install with: pip install openai-whisper")
            raise
        except Exception as e:
            logger.error(f"Failed to load openai-whisper model: {e}")
            raise
        return self.model_instance
    
    def transcribe_clips(self, clip_paths: List[Path]) -> Dict[Path, Dict]:
        """
        Transcribe multiple audio clips with optimized performance
        
        Args:
            clip_paths: List of paths to audio/video files
            
        Returns:
            Dictionary mapping clip paths to transcription results
        """
        results = {}
        
        # Load model once for all clips
        model = self._load_model()
        
        for i, clip_path in enumerate(clip_paths, 1):
            logger.info(f"Transcribing clip {i}/{len(clip_paths)}: {clip_path.name}")
            
            # Check cache first
            cached_result = self._load_cached_transcript(clip_path)
            if cached_result:
                logger.info(f"Using cached transcript for {clip_path.name}")
                results[clip_path] = cached_result
                continue
            
            # Transcribe and cache
            try:
                result = self._transcribe_single_clip(clip_path, model)
                self._save_cached_transcript(clip_path, result)
                results[clip_path] = result
                
            except Exception as e:
                logger.error(f"Failed to transcribe {clip_path}: {e}")
                results[clip_path] = {
                    'text': '',
                    'language': 'en',
                    'segments': [],
                    'error': str(e)
                }
        
        return results
    
    def _load_cached_transcript(self, clip_path: Path) -> Optional[Dict]:
        """Load cached transcript if available"""
        # Extract VOD ID from filename (e.g., "2488522748_Twitch.mp4" -> "2488522748")
        vod_id = clip_path.stem.split('_')[0] if '_' in clip_path.stem else clip_path.stem
        cache_file = self.cache_dir / f"{vod_id}_transcript.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached transcript {cache_file}: {e}")
        
        return None
    
    def _save_cached_transcript(self, clip_path: Path, result: Dict):
        """Save transcript to cache"""
        # Extract VOD ID from filename (e.g., "2488522748_Twitch.mp4" -> "2488522748")
        vod_id = clip_path.stem.split('_')[0] if '_' in clip_path.stem else clip_path.stem
        cache_file = self.cache_dir / f"{vod_id}_transcript.json"
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved transcript cache: {cache_file}")
        except Exception as e:
            logger.error(f"Failed to save transcript cache {cache_file}: {e}")
    
    def _transcribe_single_clip(self, clip_path: Path, model) -> Dict:
        """Transcribe a single audio clip using configured Whisper backend."""
        try:
            if self.provider == 'faster-whisper':
                # faster-whisper returns a generator of segments and an info object
                segments, info = model.transcribe(
                    str(clip_path),
                    language=config.whisper_language,
                    beam_size=config.whisper_beam_size,
                    temperature=config.whisper_temperature,
                    initial_prompt=config.whisper_initial_prompt,
                    word_timestamps=config.whisper_word_timestamps,
                )
                seg_list = []
                for seg in segments:
                    seg_dict = {
                        'start': float(getattr(seg, 'start', 0.0)),
                        'end': float(getattr(seg, 'end', 0.0)),
                        'text': getattr(seg, 'text', '').strip(),
                    }
                    # Optional words
                    words = getattr(seg, 'words', None)
                    if words and config.whisper_word_timestamps:
                        seg_dict['words'] = [
                            {'start': float(getattr(w, 'start', 0.0)), 'end': float(getattr(w, 'end', 0.0)), 'word': getattr(w, 'word', '')}
                            for w in words
                        ]
                    seg_list.append(seg_dict)
                language = getattr(info, 'language', config.whisper_language)
                text_combined = ' '.join(s['text'] for s in seg_list).strip()
                return {
                    'text': text_combined,
                    'language': language,
                    'segments': seg_list,
                }
            
            # openai-whisper path
            result = model.transcribe(
                str(clip_path),
                language=config.whisper_language,
                verbose=False,
                fp16=(self.device == "cuda" and config.whisper_fp16),
                temperature=config.whisper_temperature,
                beam_size=config.whisper_beam_size if config.whisper_temperature == 0.0 else None,
                best_of=config.whisper_best_of if config.whisper_temperature > 0.0 else None,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
                initial_prompt=config.whisper_initial_prompt,
                word_timestamps=config.whisper_word_timestamps,
                prepend_punctuations="\"'([{-",
                append_punctuations="\"'.!?):]}"
            )
            return {
                'text': result['text'].strip(),
                'language': result.get('language', config.whisper_language),
                'segments': result['segments']
            }
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return {
                'text': '',
                'language': 'en',
                'segments': [],
                'error': str(e)
            }