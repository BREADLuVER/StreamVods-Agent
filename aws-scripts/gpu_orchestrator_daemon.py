#!/usr/bin/env python3
"""
StreamSniped GPU Orchestrator Daemon
Unified daemon that manages both clip generation and video rendering
Processes jobs sequentially to avoid resource contention
"""

import os
import sys
import json
import time
import argparse
import logging
from logging.handlers import RotatingFileHandler
import threading  # noqa: F401 (reserved for future parallel features)
import subprocess
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime, timedelta  # noqa: F401 (timedelta reserved for future use)
import signal
import psutil

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    print("X boto3 not available - install with: pip install boto3")
    sys.exit(1)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Import after sys.path is adjusted so project modules resolve
from youtube_trigger import (
    build_youtube_trigger_payload,
    is_truthy,
    send_youtube_trigger,
    sync_metadata_to_s3,
)
from utils.job_tracker import JobTracker, JobStatus  # DynamoDB job lock/heartbeat
try:
    # Resolve YouTube channels for uploads
    from src.youtube_channels import resolve_channels_for_vod
except Exception:
    resolve_channels_for_vod = None  # type: ignore
try:
    # Optional webcam/chat gate
    from cam_detection import detect_webcam_in_vod as _detect_webcam_in_vod
    from cam_detection import detect_chat_in_vod as _detect_chat_in_vod
except Exception:
    _detect_webcam_in_vod = None  # type: ignore
    _detect_chat_in_vod = None  # type: ignore

# make TwitchDownloaderCLI see ffmpeg.exe in executables/
exec_dir = str(Path(__file__).parent.parent / "executables")
os.environ["PATH"] = f"{exec_dir}{os.pathsep}{os.environ.get('PATH', '')}"

# Ensure logs directory exists for file logging
try:
    Path('logs').mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# Configure logging
_stream_handler = logging.StreamHandler()
_file_handler = RotatingFileHandler('logs/gpu_orchestrator.log', maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[_stream_handler, _file_handler]
)
logger = logging.getLogger(__name__)

# Sanitize console output on Windows consoles that cannot render unicode/emoji
class _AsciiLogFilter(logging.Filter):
    def __init__(self, allow_unicode: bool):
        super().__init__()
        self.allow_unicode = allow_unicode

    def filter(self, record: logging.LogRecord) -> bool:
        if self.allow_unicode:
            return True
        try:
            message = record.getMessage()
        except Exception:
            return True
        sanitized = message.encode('ascii', 'ignore').decode('ascii')
        if sanitized != message:
            record.msg = sanitized
            record.args = ()
        return True

# Detect whether console supports unicode; allow override via env
_stdout_encoding = (getattr(sys.stdout, 'encoding', '') or '').lower()
_force_unicode_env = os.getenv('FORCE_UNICODE_LOGS', '').lower() in ('1', 'true', 'yes')
_force_plain_env = os.getenv('PLAIN_LOGS', '').lower() in ('1', 'true', 'yes')
_console_supports_unicode = _stdout_encoding in ('utf-8', 'utf8', 'cp65001')
_allow_unicode_console = (_console_supports_unicode or _force_unicode_env) and not _force_plain_env

# Attach filter to stream handler to avoid UnicodeEncodeError in terminals
for _handler in logging.getLogger().handlers:
    if isinstance(_handler, logging.StreamHandler):
        _handler.addFilter(_AsciiLogFilter(_allow_unicode_console))

class GPUResourceMonitor:
    """Monitor GPU and system resources"""
    
    def __init__(self, gpu_usage_threshold: Optional[int] = None, memory_threshold: Optional[int] = None, cpu_threshold: Optional[int] = None):
        # Allow thresholds to be overridden via CLI args or environment variables
        # Defaults remain conservative to avoid overloading the system
        env_gpu = os.getenv('GPU_USAGE_THRESHOLD')
        env_mem = os.getenv('MEMORY_USAGE_THRESHOLD')
        env_cpu = os.getenv('CPU_USAGE_THRESHOLD')
        self.gpu_usage_threshold = int(env_gpu) if env_gpu else (gpu_usage_threshold if gpu_usage_threshold is not None else 80)
        self.memory_threshold = int(env_mem) if env_mem else (memory_threshold if memory_threshold is not None else 85)
        self.cpu_threshold = int(env_cpu) if env_cpu else (cpu_threshold if cpu_threshold is not None else 90)
        
    def check_gpu_available(self) -> bool:
        """Check if GPU is available for processing"""
        result = None
        try:
            # Try to run nvidia-smi to check GPU status
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                logger.warning("nvidia-smi not available or failed")
                return True  # Assume available if we can't check
                
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        gpu_util = int(parts[0])
                        mem_used = int(parts[1])
                        mem_total = int(parts[2])
                        mem_util = (mem_used / mem_total) * 100 if mem_total > 0 else 0
                        
                        if gpu_util > self.gpu_usage_threshold or mem_util > self.memory_threshold:
                            logger.info(f"GPU busy: {gpu_util}% util, {mem_util:.1f}% memory")
                            return False
                            
            return True
            
        except Exception as e:
            logger.warning(f"GPU check failed: {e}")
            return True  # Assume available if check fails
        finally:
            # Explicitly clean up subprocess resources
            if result is not None:
                del result
    
    def check_system_resources(self) -> bool:
        """Check system memory and CPU usage"""
        try:
            memory = psutil.virtual_memory()
            # Use non-blocking CPU check to avoid memory accumulation
            cpu_percent = psutil.cpu_percent(interval=None)
            
            if memory.percent > self.memory_threshold:
                logger.info(f"System memory high: {memory.percent:.1f}%")
                return False
                
            if cpu_percent > self.cpu_threshold:
                logger.info(f"System CPU high: {cpu_percent:.1f}%")
                return False
                
            return True
            
        except Exception as e:
            logger.warning(f"System resource check failed: {e}")
            return True  # Assume available if check fails
    
    def wait_for_resources(self, timeout_minutes: int = 30) -> bool:
        """Wait for resources to become available"""
        logger.info("Waiting for GPU and system resources to become available...")
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        while time.time() - start_time < timeout_seconds:
            if self.check_gpu_available() and self.check_system_resources():
                logger.info("Resources available - proceeding with job")
                return True
                
            time.sleep(30)  # Check every 30 seconds
            
        logger.warning(f"Timeout waiting for resources after {timeout_minutes} minutes")
        return False

class JobProcessor:
    """Process individual jobs (clip or render)"""
    
    def __init__(self, job_type: str, manifest_uri: str, vod_id: str):
        self.job_type = job_type
        self.manifest_uri = manifest_uri
        self.vod_id = vod_id
        self.start_time = None
        self.end_time = None
        self.local_postprocess_enabled = os.getenv('LOCAL_POSTPROCESS', 'false').lower() in ['true', '1', 'yes']
        self.upload_queue_url = os.getenv('YOUTUBE_UPLOAD_QUEUE_URL', '').strip()
        self.upload_videos = is_truthy(os.getenv('UPLOAD_VIDEOS'), default=False)
        self.disable_s3_uploads_default = os.getenv('DISABLE_S3_UPLOADS_DEFAULT')
        self.s3_bucket = os.getenv('S3_BUCKET', 'streamsniped-dev-videos')
        
        # Unified logging setup per job
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.jobs_dir = Path('logs') / 'jobs' / self.vod_id
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.jobs_dir / f"full_{self.run_id}.log"
        
        self.child_pid: Optional[int] = None
        self.child_pid_file: Optional[Path] = None
        # Propagate conservative network/concurrency defaults to child processes unless already set
        self.env_overrides: Dict[str, str] = {}
        # Respect user overrides if provided; otherwise set safe defaults
        dl_conc = os.getenv('DOWNLOAD_MAX_CONCURRENCY', '').strip()
        if not dl_conc:
            # Default to 3 concurrent downloads to avoid saturating ~300 Mbps connections
            self.env_overrides['DOWNLOAD_MAX_CONCURRENCY'] = '3'
        else:
            self.env_overrides['DOWNLOAD_MAX_CONCURRENCY'] = dl_conc
        # Keep TwitchDownloader parallelism aligned
        if not os.getenv('TD_MAX_PARALLEL'):
            self.env_overrides['TD_MAX_PARALLEL'] = self.env_overrides['DOWNLOAD_MAX_CONCURRENCY']
        # Limit AI data worker concurrency by default
        if not os.getenv('AI_DATA_MAX_WORKERS'):
            self.env_overrides['AI_DATA_MAX_WORKERS'] = '1'
        # Prefer high but not 60fps by default; allow override via env
        if os.getenv('DOWNLOAD_QUALITY_PREF'):
            self.env_overrides['DOWNLOAD_QUALITY_PREF'] = os.getenv('DOWNLOAD_QUALITY_PREF', '1080p')

    def _run_subprocess(self, cmd: List[str], timeout_seconds: int, step_name: str, env: dict = None) -> bool:
        """Run a subprocess with streaming logs and unique run-id filenames."""
        # Log to main orchestrator log for immediate visibility
        logger.info(f"Running {step_name}: {' '.join(cmd)}")
        logger.info(f"📝 Job log: {self.log_file.name} (run {self.run_id})")
        
        base_name = f"{self.job_type}_{self.run_id}"
        
        try:
            with self.log_file.open('a', encoding='utf-8', errors='replace') as fout:
                # Write command info to both files
                fout.write(f"\n{'='*50}\n")
                fout.write(f"=== {step_name} ===\n")
                fout.write(f"Command: {' '.join(cmd)}\n")
                fout.write(f"Started: {datetime.now().isoformat()}\n")
                fout.write("=" * 50 + "\n\n")
                fout.flush()
                
                # Ensure child Python processes flush output immediately so logs stream
                patched_cmd = list(cmd)
                try:
                    from os.path import basename
                    first = patched_cmd[0]
                    if basename(first).lower().startswith('python') and (len(patched_cmd) < 2 or patched_cmd[1] != '-u'):
                        patched_cmd = [first, '-u', *patched_cmd[1:]]
                except Exception:
                    pass

                disable_s3 = self._should_disable_s3_uploads()
                upload_videos = self._should_upload_videos()

                # Merge environment overrides for child process
                child_env = {
                    **os.environ,
                    **self.env_overrides,
                    'PYTHONUNBUFFERED': '1',
                    'PYTHONIOENCODING': 'utf-8',
                    'JOB_RUN_ID': self.run_id,
                    'VOD_ID': self.vod_id,
                    'JOB_TYPE': self.job_type,
                    'RUN_ID': self.run_id,
                    'DISABLE_S3_UPLOADS': 'true' if disable_s3 else 'false',
                    'LOCAL_TEST_MODE': 'true' if disable_s3 else 'false',
                    'CONTAINER_MODE': 'false' if disable_s3 else 'true',
                    'UPLOAD_VIDEOS': 'true' if upload_videos else 'false',
                }
                
                # Add custom environment variables if provided
                if env:
                    child_env.update(env)

                proc = subprocess.Popen(
                    patched_cmd,
                    stdout=fout,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    bufsize=1,  # line-buffered
                    env=child_env,
                )
                # Expose PID for external termination and write PID file
                self.child_pid = proc.pid
                self.child_pid_file = self.jobs_dir / f"{base_name}.pid"
                try:
                    with self.child_pid_file.open('w', encoding='utf-8') as pf:
                        pf.write(str(self.child_pid))
                except Exception:
                    pass
                start_time = time.time()
                try:
                    code = proc.wait(timeout_seconds)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    logger.error(f"{step_name} timed out after {timeout_seconds}s")
                    fout.write(f"\n\n=== TIMEOUT after {timeout_seconds}s ===\n")
                    return False

                # Write completion info
                end_time = datetime.now()
                duration_seconds = time.time() - start_time
                duration = f"{duration_seconds:.2f}s"
                fout.write("\n\n=== COMPLETED ===\n")
                fout.write(f"Ended: {end_time.isoformat()}\n")
                fout.write(f"Duration: {duration}\n")
                fout.write(f"Exit code: {proc.returncode}\n")
                
                # Use exit code from wait()
                if code == 0:
                    logger.info(f"✅ {step_name} completed successfully")
                    return True
                logger.error(f"❌ {step_name} failed (exit {code})")
                return False
                
        except Exception as e:
            logger.error(f"💥 {step_name} error: {e}")
            # Write error to log files
            try:
                with self.log_file.open('a', encoding='utf-8', errors='replace') as fout:
                    fout.write(f"\n\n=== ERROR ===\n{e}\n")
            except Exception:
                pass
            return False
        finally:
            # Best-effort remove PID file when done
            try:
                if self.child_pid_file and self.child_pid_file.exists():
                    self.child_pid_file.unlink()
            except Exception:
                pass

    def kill_child_tree(self) -> None:
        """Terminate the active child process and its tree (Windows-friendly)."""
        pid = self.child_pid
        if not pid:
            return
        try:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'], capture_output=True)
            else:
                try:
                    import psutil as _ps
                    proc = _ps.Process(pid)
                    for child in proc.children(recursive=True):
                        try:
                            child.kill()
                        except Exception:
                            pass
                    try:
                        proc.kill()
                    except Exception:
                        pass
                except Exception:
                    # Fallback to plain kill
                    os.kill(pid, 9)
        except Exception:
            pass

    def _run_local_postprocess(self) -> bool:
        """Run 3c/3d/3e locally after clips are created"""
        logger.info("LOCAL_POSTPROCESS enabled - starting local post-process (3c/3d/3e)")
        # 3c: Merge per-chapter clips
        merge_ok = self._run_subprocess(
            [sys.executable, '-u', 'processing-scripts/merge_chapter_clips.py', self.vod_id],
            timeout_seconds=900,
            step_name='Step 3c (merge chapter clips)'
        )
        if not merge_ok:
            logger.error("Aborting local post-process due to merge failure")
            return False
        # 3d: Generate metadata
        meta_ok = self._run_subprocess(
            [sys.executable, '-u', 'processing-scripts/generate_clip_youtube_metadata.py', self.vod_id],
            timeout_seconds=900,
            step_name='Step 3d (generate clip YouTube metadata)'
        )
        if not meta_ok:
            logger.error("Aborting local post-process due to metadata failure")
            return False
        # 3e: Upload locally to YouTube (remote queue deprecated)
        upload_ok = self._run_subprocess(
            [sys.executable, '-u', 'processing-scripts/upload_clips_to_youtube.py', self.vod_id],
            timeout_seconds=3600,
            step_name='Step 3e (upload clips to YouTube)'
        )
        if not upload_ok:
            logger.error("Local post-process upload step failed")
            return False
        logger.info("Local post-process completed (3c/3d/3e)")
        return True
    
    def _build_local_processing_env(self) -> Dict[str, str]:
        """Construct environment flags for local-only processing (no S3)."""
        env = os.environ.copy()
        env['DISABLE_S3_UPLOADS'] = 'true'
        env['LOCAL_TEST_MODE'] = 'true'
        env['CONTAINER_MODE'] = 'false'
        env['UPLOAD_VIDEOS'] = 'false'
        env['GPU_PROCESSING_MODE'] = 'local_only'
        return env

    def _generate_clips_manifest(self, vod_id: str) -> bool:
        """Generate clips manifest for the VOD using new clip generation pipeline."""
        cmd = [sys.executable, '-u', 'clip_generation/cli_generate.py', vod_id, '--top-k', '8', '--min-score', '4.0', '--front-pad', '10.0', '--back-pad', '1.0']
        logger.info(f"Running clips manifest: {' '.join(cmd)}")
        return self._run_subprocess(cmd, timeout_seconds=1200, step_name='Generate clips manifest')

    def _create_individual_clips(self, vod_id: str, env: Dict[str, str]) -> bool:
        """Create individual clips from the manifest."""
        cmd = [sys.executable, '-u', 'processing-scripts/create_individual_clips.py', vod_id]
        logger.info(f"Running clip creation: {' '.join(cmd)}")
        return self._run_subprocess(cmd, timeout_seconds=7200, step_name='Create individual clips', env=env)

    def _generate_clip_metadata(self, vod_id: str) -> bool:
        """Generate YouTube metadata for clips."""
        return self._run_subprocess(
            [sys.executable, '-u', 'processing-scripts/generate_clip_youtube_metadata.py', vod_id],
            timeout_seconds=900,
            step_name='Generate clip YouTube metadata',
        )

    def _upload_clips_local(self, vod_id: str, force_public: bool) -> bool:
        """Upload clips to YouTube locally (no remote queue)."""
        upload_cmd = [sys.executable, '-u', 'processing-scripts/upload_clips_to_youtube.py', vod_id]
        # Resolve channels automatically
        try:
            if resolve_channels_for_vod:
                channels = resolve_channels_for_vod(vod_id, content_type='clips')
                if channels:
                    upload_cmd += ['--channels', ','.join(channels)]
        except Exception:
            pass
        if force_public:
            upload_cmd.append('--public')
        logger.info("Starting local YouTube uploads for clips (remote queue deprecated)")
        return self._run_subprocess(upload_cmd, timeout_seconds=7200, step_name='Upload clips to YouTube (local)')

    def _cleanup_after_upload(self, vod_id: str) -> bool:
        """Clean up local files post upload (best-effort)."""
        return self._run_subprocess(
            [sys.executable, '-u', 'processing-scripts/cleanup_after_upload.py', vod_id, '--sweep-ttl-days', '7'],
            timeout_seconds=300,
            step_name='Clean up local files after successful upload'
        )

    # -------------------- ARCH (Story Arcs) Helpers --------------------

    def _create_or_update_arc_manifests(self, vod_id: str) -> bool:
        """Generate arc manifests with gap filling for full coverage."""
        cmd = [
            sys.executable, '-u', '-m', 'story_archs.create_story_archs', vod_id,
            '--target-min', '900',
            '--target-max', '3200',
            '--min-score', '0.5',
            '--min-contiguity', '0.4', 
            '--max-gap-seconds', '3600',
            '--max-gap-fraction', '0.3',
            '--resolution-grace-seconds', '300',
            '--max-arcs', '0'  # Full coverage with gap filling
        ]
        logger.info(f"Ensure arc manifests: {' '.join(cmd)}")
        return self._run_subprocess(cmd, timeout_seconds=1200, step_name='Create/Update arc manifests')

    def _render_arch_videos(self, vod_id: str, env: Dict[str, str]) -> bool:
        """Render arc videos into data/chunks/<vod_id>/arcs."""
        cmd = [sys.executable, '-u', '-m', 'story_archs.create_arch_videos', vod_id]
        logger.info(f"Render arch videos: {' '.join(cmd)}")
        return self._run_subprocess(cmd, timeout_seconds=7200, step_name='Render arch videos', env=env)

    def _generate_arch_metadata(self, vod_id: str) -> bool:
        """Generate YouTube metadata for arcs with Ollama timestamp generation."""
        cmd = [sys.executable, '-u', 'processing-scripts/generate_arch_youtube_metadata_enhanced.py', vod_id]
        logger.info(f"Generate arch metadata: {' '.join(cmd)}")
        return self._run_subprocess(cmd, timeout_seconds=900, step_name='Generate arch YouTube metadata with timestamps')

    def _upload_arch_videos(self, vod_id: str, force_public: bool) -> bool:
        """Upload arc videos to YouTube using enhanced metadata with timestamps."""
        cmd = [sys.executable, '-u', 'processing-scripts/auto_youtube_upload_arch.py', vod_id]
        try:
            if resolve_channels_for_vod:
                channels = resolve_channels_for_vod(vod_id, content_type='arcs')
                if channels:
                    cmd += ['--channels', ','.join(channels)]
        except Exception:
            pass
        if force_public:
            cmd.append('--public')
        logger.info(f"Upload arch videos with enhanced metadata (includes timestamps): {' '.join(cmd)}")
        success = self._run_subprocess(cmd, timeout_seconds=7200, step_name='Upload arch videos to YouTube with timestamps')
        if success:
            logger.info("Arch videos uploaded successfully with AI-generated timestamps in descriptions")
        else:
            logger.warning("Arch video uploads failed or were scheduled for retry")
        return success

    # -------------------- GEMINI Arc Detection Helpers --------------------

    def _run_gemini_arc_detection(self, vod_id: str) -> bool:
        """Run Gemini-based arc detection on transcript data.
        
        This replaces rag.narrative_analyzer with a more accurate arc detection
        that identifies intro/climax/resolution boundaries directly.
        """
        cmd = [
            sys.executable, '-u', '-m', 'story_archs.gemini_arc_detection', vod_id,
            '--save',
            '--model', os.getenv('GEMINI_ARC_MODEL', 'gemini-3-flash-preview'),  # Gemini 3 Flash
        ]
        logger.info(f"Running Gemini arc detection: {' '.join(cmd)}")
        return self._run_subprocess(cmd, timeout_seconds=3600, step_name='Gemini arc detection')

    def _convert_gemini_arcs_to_manifests(self, vod_id: str) -> bool:
        """Convert Gemini arc output to individual arc manifests for video rendering.
        
        Uses dynamic rating system to select only the best arcs.
        """
        top_k = os.getenv('GEMINI_ARC_TOP_K', '10')  # Default: top 10 arcs
        min_rating = os.getenv('GEMINI_ARC_MIN_RATING', '89')  # Default: rating >= 89/100
        cmd = [
            sys.executable, '-u', '-m', 'story_archs.gemini_to_arc_manifests', vod_id,
            '--top-k', top_k,
            '--min-rating', min_rating,
            '--min-duration', '300',   # 5 min minimum
            '--max-duration', '2400',  # 40 min maximum
        ]
        logger.info(f"Converting Gemini arcs to manifests: {' '.join(cmd)}")
        return self._run_subprocess(cmd, timeout_seconds=300, step_name='Convert Gemini arcs to manifests')
    
    def _should_disable_s3_uploads(self) -> bool:
        # Force disable S3 uploads to save costs
        return True
        
        # Original logic below for reference
        # if self.disable_s3_uploads_default is not None:
        #     return is_truthy(self.disable_s3_uploads_default, default=False)
        # explicit_flag = os.getenv('DISABLE_S3_UPLOADS')
        # if explicit_flag is not None:
        #     return is_truthy(explicit_flag, default=False)
        # return False
    
    def _should_upload_videos(self) -> bool:
        if self.upload_videos:
            return True
        explicit_flag = os.getenv('UPLOAD_VIDEOS')
        if explicit_flag is not None:
            return is_truthy(explicit_flag, default=False)
        return True
    
    def _trigger_remote_upload(self, *, include_clips: bool) -> bool:
        if self._should_disable_s3_uploads():
            logger.info("S3 uploads disabled; skipping remote upload trigger")
            return False

        if not self.upload_queue_url:
            return False
        try:
            metadata_keys = sync_metadata_to_s3(
                self.vod_id,
                self.s3_bucket,
                include_clips=include_clips,
            )
            payload = build_youtube_trigger_payload(
                self.vod_id,
                self.s3_bucket,
                metadata_keys,
                include_clips=include_clips,
            )
            artifacts = payload.get('artifacts') if isinstance(payload, dict) else None
            if not artifacts:
                logger.warning("No upload artifacts found after sync; skipping trigger")
                return False
            response = send_youtube_trigger(
                self.upload_queue_url,
                payload,
                region=os.getenv('AWS_REGION'),
            )
            logger.info(
                "Queued YouTube upload: message %s", response.get('MessageId', '<unknown>')
            )
            return True
        except Exception as error:
            logger.error("Failed to enqueue YouTube upload: %s", error)
            return False
        
    def process_clip_job(self) -> bool:
        """Process a clip generation job"""
        logger.info(f"[JOB START] type=clip vod_id={self.vod_id}")
        self.start_time = datetime.now()
        
        try:
            # Optional webcam gate + chat-on-stream probe
            try:
                if is_truthy(os.getenv('WEBCAM_GATE'), default=False) and _detect_webcam_in_vod:
                    logger.info("Webcam gate enabled - probing VOD for webcam presence")
                    if not _detect_webcam_in_vod(self.vod_id):
                        logger.info("No webcam detected - skipping clip job")
                        self.end_time = datetime.now()
                        return True
                # Chat-on-stream probe (independent): if chat exists on-stream, disable chat overlay later
                if _detect_chat_in_vod:
                    try:
                        has_chat = bool(_detect_chat_in_vod(self.vod_id))
                    except Exception:
                        has_chat = False
                    if has_chat:
                        logger.info("On-stream chat detected; will disable chat overlay for arcs")
                        # Set a flag in process environment so downstream steps honor it
                        os.environ['ARC_NO_CHAT'] = '1'
            except Exception as _e:
                logger.warning(f"Webcam gate probe failed; continuing: {_e}")

            # Ensure vector store assets exist locally (download from S3 prefix)
            try:
                from storage import StorageManager
                sm = StorageManager()
                s3_bucket = os.getenv('S3_BUCKET', 'streamsniped-dev-videos')
                prefix = f"s3://{s3_bucket}/vector_stores/{self.vod_id}/"
                local_dir = Path(f"data/vector_stores/{self.vod_id}")
                local_dir.mkdir(parents=True, exist_ok=True)
                files = sm.list_files(prefix)
                for uri in files:
                    try:
                        name = Path(uri).name
                        sm.download_file(uri, str(local_dir / name))
                    except Exception as _e:
                        logger.warning(f"Could not download {uri}: {_e}")
            except Exception as _e:
                logger.warning(f"Vector store prefetch skipped: {_e}")

            # 1) Generate clips manifest using new pipeline
            cmd1 = [sys.executable, '-u', 'clip_generation/cli_generate.py', self.vod_id, '--top-k', '8', '--min-score', '4.0', '--front-pad', '10.0', '--back-pad', '1.0']
            logger.info(f"Running clips manifest: {' '.join(cmd1)}")
            ok1 = self._run_subprocess(cmd1, timeout_seconds=3600, step_name='Generate clips manifest')
            if not ok1:
                self.end_time = datetime.now()
                duration = self.end_time - self.start_time
                logger.error(f"[JOB END] type=clip vod_id={self.vod_id} status=fail(duration {duration}) at manifest stage")
                return False

            # 2) Create individual clips from manifest
            cmd2 = [sys.executable, '-u', 'processing-scripts/create_individual_clips.py', self.vod_id]
            logger.info(f"Running clip creation: {' '.join(cmd2)}")
            ok = self._run_subprocess(cmd2, timeout_seconds=7200, step_name='Create individual clips')
            self.end_time = datetime.now()
            duration = self.end_time - self.start_time
            if ok:
                logger.info(f"[JOB END] type=clip vod_id={self.vod_id} status=success duration={duration}")
                return True
            # Clean up partial clip artifacts on failure
            try:
                self._run_subprocess(
                    [sys.executable, '-u', 'processing-scripts/cleanup_local_files.py', self.vod_id, '--keep-ai-data', '--clips-only', '--sweep-ttl-days', '7'],
                    timeout_seconds=300,
                    step_name='Clean up partial clip files after failure'
                )
            except Exception:
                pass
            logger.error(f"[JOB END] type=clip vod_id={self.vod_id} status=fail duration={duration}")
            return False
                
        except subprocess.TimeoutExpired:
            self.end_time = datetime.now()
            duration = self.end_time - self.start_time
            logger.error(f"[JOB END] type=clip vod_id={self.vod_id} status=timeout duration={duration}")
            return False
        except Exception as e:
            self.end_time = datetime.now()
            duration = self.end_time - self.start_time
            logger.error(f"[JOB END] type=clip vod_id={self.vod_id} status=error duration={duration} error={e}")
            return False
    
    def process_render_job(self) -> bool:
        """Process a video rendering job"""
        logger.info(f"[JOB START] type=render vod_id={self.vod_id}")
        self.start_time = datetime.now()
        
        try:
            # Render Director's Cut from manifest: ensure manifest locally, then run module
            cmd = [sys.executable, '-u', '-c', (
                "import os,sys,subprocess; "
                "from pathlib import Path; "
                "from storage import StorageManager; "
                "manifest_uri=sys.argv[1]; vod_id=os.getenv('VOD_ID') or '' ; "
                "if not vod_id and manifest_uri.startswith('s3://'): vod_id=Path(manifest_uri).parts[-2]; "
                "local=Path(f'data/vector_stores/{vod_id}/enhanced_director_cut_manifest.json'); "
                "local.parent.mkdir(parents=True, exist_ok=True); "
                "StorageManager().download_file(manifest_uri, str(local)); "
                "bs=os.getenv('DC_BATCH_SIZE','4'); "
                "args=[sys.executable,'-m','directors_cut.create_cloud_video',vod_id,'--batch-size',str(bs),'--transition-duration','2.0']; "
                "subprocess.check_call(args)"
            ), self.manifest_uri]
            
            logger.info(f"Running render command: {' '.join(cmd)}")
            ok = self._run_subprocess(cmd, timeout_seconds=7200, step_name='Render video')
            self.end_time = datetime.now()
            duration = self.end_time - self.start_time
            if ok:
                logger.info(f"[JOB END] type=render vod_id={self.vod_id} status=success duration={duration}")
                return True
            logger.error(f"[JOB END] type=render vod_id={self.vod_id} status=fail duration={duration}")
            return False
                
        except subprocess.TimeoutExpired:
            self.end_time = datetime.now()
            duration = self.end_time - self.start_time
            logger.error(f"[JOB END] type=render vod_id={self.vod_id} status=timeout duration={duration}")
            return False
        except Exception as e:
            self.end_time = datetime.now()
            duration = self.end_time - self.start_time
            logger.error(f"[JOB END] type=render vod_id={self.vod_id} status=error duration={duration} error={e}")
            return False

    def process_full_job(self) -> bool:
        """Process a full local workflow job (AI data -> vector store -> render -> clips/metadata/upload)."""
        logger.info(f"[JOB START] type=full vod_id={self.vod_id}")
        self.start_time = datetime.now()

        try:
            # Optional webcam gate + chat-on-stream probe
            try:
                if is_truthy(os.getenv('WEBCAM_GATE'), default=False) and _detect_webcam_in_vod:
                    logger.info("Webcam gate enabled - probing VOD for webcam presence")
                    if not _detect_webcam_in_vod(self.vod_id):
                        logger.info("No webcam detected - skipping full job")
                        self.end_time = datetime.now()
                        return True
                # Chat-on-stream probe (independent): if chat exists on-stream, disable chat overlay later
                if _detect_chat_in_vod:
                    try:
                        has_chat = bool(_detect_chat_in_vod(self.vod_id))
                    except Exception:
                        has_chat = False
                    if has_chat:
                        logger.info("On-stream chat detected; will disable chat overlay for arcs")
                        os.environ['ARC_NO_CHAT'] = '1'
            except Exception as _e:
                logger.warning(f"Webcam gate probe failed; continuing: {_e}")

            vod_id = self.vod_id
            url = f"https://www.twitch.tv/videos/{vod_id}"

            # 1) AI Data (GPU/CPU depending on faster-whisper availability). Keep timeout generous.
            # Cache gate: if core AI JSONs already exist locally or in S3, skip regeneration
            ai_dir = Path('data/ai_data') / vod_id
            ai_dir.mkdir(parents=True, exist_ok=True)
            s3_bucket = os.getenv('S3_BUCKET', 'streamsniped-dev-videos')
            core_ai_files = [
                f"{vod_id}_ai_data.json",
                f"{vod_id}_chapters.json",
                f"{vod_id}_clip_titles.json",
            ]
            from storage import StorageManager
            storage = StorageManager()
            need_ai = False
            for name in core_ai_files:
                local_path = ai_dir / name
                s3_uri = f"s3://{s3_bucket}/ai_data/{vod_id}/{name}"
                if local_path.exists():
                    continue
                if storage.exists(s3_uri):
                    try:
                        storage.download_file(s3_uri, str(local_path))
                        continue
                    except Exception:
                        pass
                need_ai = True
            if need_ai:
                # AI data generation timeout: Allow up to 6 hours for very long VODs
                # This step is stable and just needs time to process all chunks
                # Average: ~3-5 minutes per hour of VOD content
                ok_ai = self._run_subprocess(
                    [sys.executable, '-u', 'processing-scripts/generate_ai_data_cloud.py', url],
                    timeout_seconds=21600,  # 6 hours - prevents killing jobs that are almost done
                    step_name='Generate AI data'
                )
                if not ok_ai:
                    self.end_time = datetime.now()
                    logger.error(f"[JOB END] type=full vod_id={vod_id} status=fail at ai-data")
                    return False
            
            # Clean up temp audio chunks after AI data generation
            self._run_subprocess([sys.executable, '-u', 'processing-scripts/cleanup_audio_chunks.py', vod_id, '--sweep-ttl-days', '7'],
                                 timeout_seconds=300, step_name='Clean up temp audio chunks')

            # 2) Vector store + RAG pipeline (clips foundation)
            # These steps build the metadata.db with chat_rate_z, burst_score, etc.
            # Used by BOTH clips (data-driven climax) and videos (arc context)
            pipeline_steps: List[List[str]] = [
                [sys.executable, '-u', 'processing-scripts/filter_transcript_boundaries.py', vod_id],
                [sys.executable, '-u', 'vector_store/full_vod_documenter.py', vod_id],
                [sys.executable, '-u', 'vector_store/vod_quality_gate.py', vod_id],
                [sys.executable, '-u', 'vector_store/burst_summarize.py', vod_id],
                # NOTE: rag.narrative_analyzer and rag.index_narrative are REPLACED by Gemini arc detection
                # They are no longer needed for the video pipeline
            ]
            for cmd in pipeline_steps:
                if not self._run_subprocess(cmd, timeout_seconds=1800, step_name=f"Pipeline: {' '.join(cmd[-2:])}"):
                    self.end_time = datetime.now()
                    logger.error(f"[JOB END] type=full vod_id={vod_id} status=fail at pipeline")
                    return False
            
            # 2b) Gemini Arc Detection (replaces rag.narrative_analyzer for video creation)
            # Runs hour-by-hour transcript analysis to detect intro/climax/resolution arcs
            use_gemini_arcs = os.getenv('USE_GEMINI_ARCS', 'true').lower() in ('1', 'true', 'yes')
            if use_gemini_arcs:
                if not self._run_gemini_arc_detection(vod_id):
                    logger.warning("Gemini arc detection failed; videos may not be generated")
                else:
                    # Convert Gemini arcs to manifests for video rendering
                    if not self._convert_gemini_arcs_to_manifests(vod_id):
                        logger.warning("Gemini arc manifest conversion failed; falling back to legacy arc creation")
                        # Fallback: use legacy enhanced_director_cut_selector + create_story_archs
                        if not self._run_subprocess([sys.executable, '-u', '-m', 'rag.enhanced_director_cut_selector', vod_id],
                                                    timeout_seconds=1800, step_name='Legacy: enhanced_director_cut_selector'):
                            logger.warning("Legacy arc fallback also failed")
            else:
                # Legacy path: use rag.narrative_analyzer + enhanced_director_cut_selector
                logger.info("Gemini arcs disabled; using legacy RAG pipeline for videos")
                legacy_steps = [
                    [sys.executable, '-u', '-m', 'rag.narrative_analyzer', vod_id],
                    [sys.executable, '-u', '-m', 'rag.index_narrative', vod_id],
                    [sys.executable, '-u', '-m', 'rag.enhanced_director_cut_selector', vod_id],
                ]
                for cmd in legacy_steps:
                    if not self._run_subprocess(cmd, timeout_seconds=1800, step_name=f"Legacy: {' '.join(cmd[-2:])}"):
                        logger.warning(f"Legacy step failed: {cmd[-1]}")

            # 3) Clips Processing (First - takes less time) with cache gate
            clips_created = False
            if not self._generate_clips_manifest(vod_id):
                logger.warning("Clip manifest failed; continuing without clips")
            else:
                # Force local-only clip production to keep files on disk and skip S3
                env = self._build_local_processing_env()

                # Robust cache gate: only skip when S3 manifest exists and mp4 count >= manifest count
                try:
                    prefix = f"s3://{s3_bucket}/clips/{vod_id}/"
                    s3_files = storage.list_files(prefix)
                except Exception:
                    s3_files = []
                manifest_key = f"s3://{s3_bucket}/clips/{vod_id}/.clips_manifest.json"
                manifest_ok = False
                manifest_count = 0
                try:
                    from tempfile import TemporaryDirectory
                    with TemporaryDirectory() as td:
                        local_manifest = Path(td) / ".clips_manifest.json"
                        if storage.exists(manifest_key):
                            storage.download_file(manifest_key, str(local_manifest))
                            if local_manifest.exists():
                                try:
                                    data = json.loads(local_manifest.read_text(encoding='utf-8'))
                                    manifest_count = int(data.get('count') or 0)
                                    manifest_ok = manifest_count >= 0
                                except Exception:
                                    manifest_ok = False
                except Exception:
                    manifest_ok = False

                mp4_count = sum(1 for f in s3_files if isinstance(f, str) and f.lower().endswith('.mp4'))
                cache_complete = bool(manifest_ok and mp4_count >= max(1, manifest_count))
                if cache_complete:
                    logger.info(f"Clips already complete on S3; skipping generation (mp4={mp4_count}, manifest_count={manifest_count})")
                    clips_created = True
                else:
                    if s3_files:
                        logger.info(f"Clip cache incomplete on S3 (mp4={mp4_count}, manifest_count={manifest_count}); regenerating clips")
                    else:
                        logger.info("No clips found on S3; generating clips")
                    clips_success = self._create_individual_clips(vod_id, env)
                    clips_created = clips_success
                    if not clips_success:
                        # Clean up partial clip artifacts on failure
                        self._run_subprocess(
                            [sys.executable, '-u', 'processing-scripts/cleanup_local_files.py', vod_id, '--keep-ai-data', '--clips-only', '--sweep-ttl-days', '7'],
                            timeout_seconds=300,
                            step_name='Clean up partial clip files after failure'
                        )

            # 4) Clips Metadata and Local YouTube Upload (remote queue deprecated)
            if clips_created:
                # Generate clip metadata
                self._generate_clip_metadata(vod_id)

                # Local YouTube uploads
                try:
                    force_public = os.getenv('UPLOAD_YOUTUBE_PUBLIC', 'false').lower() in ('1', 'true', 'yes')
                except Exception:
                    force_public = False
                clips_uploaded = self._upload_clips_local(vod_id, force_public)
                if clips_uploaded:
                    self._cleanup_after_upload(vod_id)
                else:
                    logger.warning(f"Local clip uploads failed or skipped for VOD {vod_id}")

                # -------------------- ARCH (VIDEO) WORKFLOW --------------------
                # Run after clip generation and uploads
                # Uses Gemini arc detection results from step 2b
                arch_env = self._build_local_processing_env()
                
                # Propagate ARC_NO_CHAT flag if set by chat probe
                try:
                    if os.getenv('ARC_NO_CHAT') in ('1', 'true', 'True', 'YES', 'yes'):
                        arch_env['ARC_NO_CHAT'] = '1'
                except Exception:
                    pass
                
                # Check if Gemini arc manifests exist (created in step 2b)
                arcs_index_path = Path(f"data/vector_stores/{vod_id}/arcs/arcs_index.json")
                if not arcs_index_path.exists():
                    logger.warning("No Gemini arc manifests found; skipping arch pipeline")
                else:
                    logger.info("Found Gemini arc manifests; proceeding with video generation")
                    
                    # 1. Generate metadata/titles FIRST so thumbnails can use them
                    metadata_ok = self._generate_arch_metadata(vod_id)
                    if not metadata_ok:
                        logger.warning("Arch metadata generation failed; thumbnails might lack titles")
                    else:
                        logger.info("Generated arch titles and metadata")
                    
                    # Optional: Generate arc cam crops + rate + render thumbnails
                    try:
                        arch_env['WEBCAM_GATE'] = os.getenv('WEBCAM_GATE', 'true')
                        arch_env['WEBCAM_DET_SAMPLES'] = os.getenv('WEBCAM_DET_SAMPLES', '4')
                        arch_env['WEBCAM_DET_WINDOW_S'] = os.getenv('WEBCAM_DET_WINDOW_S', '6')
                        arch_env['WEBCAM_DET_MAJORITY_K'] = os.getenv('WEBCAM_DET_MAJORITY_K', '3')
                        arch_env['WEBCAM_DET_QUALITY'] = os.getenv('WEBCAM_DET_QUALITY', '1080p')
                        arch_env['SNAP_OFFSETS'] = os.getenv('SNAP_OFFSETS', '-1,-0.5,0,0.5,1,2')
                    except Exception:
                        pass

                    # a) Extract arc cam crops (optional, for thumbnails)
                    self._run_subprocess(
                        [sys.executable, '-u', '-m', 'thumbnail.extract_arc_cam_crops', vod_id, '--quality', '1080p'],
                        timeout_seconds=1800,
                        step_name='Extract arc cam crops',
                        env=arch_env,
                    )
                    # b) Rate cams (pool from clips + arcs)
                    self._run_subprocess(
                        [sys.executable, '-u', '-m', 'thumbnail.rate_cams', vod_id, '--top-k', '24'],
                        timeout_seconds=900,
                        step_name='Rate cam crops',
                        env=arch_env,
                    )
                    # c) Render arch thumbnails (1 variant per arc)
                    self._run_subprocess(
                        [sys.executable, '-u', '-m', 'thumbnail.render_arch_thumbnails', vod_id, '--variants', '1'],
                        timeout_seconds=900,
                        step_name='Render arch thumbnails',
                        env=arch_env,
                    )

                    # Render arc videos
                    if not self._render_arch_videos(vod_id, arch_env):
                        logger.warning("Arch rendering failed; skipping arch upload")
                    elif metadata_ok:
                        # Upload arch videos to YouTube (respect same public toggle as clips)
                        self._upload_arch_videos(vod_id, force_public)
                    else:
                        logger.warning("Skipping arch upload due to missing metadata")

            # 5) Director's Cut title and render (DISABLED LOCALLY)
            # logger.info("Director's Cut steps are disabled locally; skipping DC title, render, and metadata")
            # if not self._run_subprocess([sys.executable, '-u', '-m', 'directors_cut.generate_director_cut_name', vod_id],
            #                             timeout_seconds=600, step_name='Generate Director name'):
            #     self.end_time = datetime.now()
            #     logger.error(f"[JOB END] type=full vod_id={vod_id} status=fail at dc-title")
            #     return False
            #
            # dc_env = os.environ.copy()
            # if os.getenv('GPU_PROCESSING_MODE', 'hybrid') == 'local_only':
            #     if not self.upload_queue_url:
            #         dc_env['DISABLE_S3_UPLOADS'] = 'true'
            #         dc_env['LOCAL_TEST_MODE'] = 'true'
            #         dc_env['CONTAINER_MODE'] = 'false'
            #         dc_env['UPLOAD_VIDEOS'] = 'false'
            #     else:
            #         dc_env['DISABLE_S3_UPLOADS'] = 'false'
            #         dc_env['LOCAL_TEST_MODE'] = 'false'
            #         dc_env['CONTAINER_MODE'] = 'true'
            #         dc_env['UPLOAD_VIDEOS'] = 'true'
            #
            # dc_filename = f"{vod_id}_directors_cut.mp4"
            # dc_local = Path(f"data/videos/{vod_id}/director_cut/{dc_filename}")
            # dc_s3 = f"s3://{s3_bucket}/videos/{vod_id}/director_cut/{dc_filename}"
            # dc_exists = dc_local.exists() or storage.exists(dc_s3)
            # if not dc_exists:
            #     if not self._run_subprocess([sys.executable, '-u', '-m', 'directors_cut.create_cloud_video', vod_id, '--batch-size', os.getenv('DC_BATCH_SIZE','4'), '--transition-duration', '2.0', '--keep-temp'],
            #                                 timeout_seconds=9200, step_name='Render Director Cut', env=dc_env):
            #         self.end_time = datetime.now()
            #         logger.error(f"[JOB END] type=full vod_id={vod_id} status=fail at dc-render")
            #         return False
            #
            # self._run_subprocess(
            #     [sys.executable, '-u', 'processing-scripts/generate_youtube_metadata.py', vod_id],
            #     timeout_seconds=600,
            #     step_name='Generate YouTube metadata',
            # )
            # if self.upload_queue_url:
            #     director_triggered = self._trigger_remote_upload(include_clips=False)
            #     if director_triggered:
            #         self._run_subprocess(
            #             [sys.executable, '-u', 'processing-scripts/cleanup_local_files.py', vod_id, '--keep-ai-data'],
            #             timeout_seconds=300,
            #             step_name='Clean up local files after S3 upload'
            #         )
            #     else:
            #         logger.warning("Remote Director's Cut upload trigger failed; video remains local")
            # else:
            #     logger.info("No YouTube upload queue configured for Director's Cut")

            # 7) Legacy fallback no longer needed; uploads handled in step 4

            self.end_time = datetime.now()
            duration = self.end_time - self.start_time
            logger.info(f"[JOB END] type=full vod_id={vod_id} status=success duration={duration}")
            return True
        except Exception as e:
            self.end_time = datetime.now()
            duration = self.end_time - self.start_time
            logger.error(f"[JOB END] type=full vod_id={self.vod_id} status=error duration={duration} error={e}")
            return False
    
    def process(self) -> bool:
        """Process the job based on type"""
        if self.job_type == "clip":
            return self.process_clip_job()
        elif self.job_type == "render":
            return self.process_render_job()
        elif self.job_type == "full":
            return self.process_full_job()
        else:
            logger.error(f"Unknown job type: {self.job_type}")
            return False

class GPUOrchestratorDaemon:
    """Main orchestrator daemon that manages both queues"""
    
    def __init__(self, clip_queue_url: str, render_queue_url: str, full_queue_url: Optional[str] = None, region: str = 'us-east-1', *, gpu_threshold: Optional[int] = None, mem_threshold: Optional[int] = None, cpu_threshold: Optional[int] = None, sleep_seconds: Optional[int] = None):
        self.clip_queue_url = clip_queue_url
        self.render_queue_url = render_queue_url
        self.full_queue_url = full_queue_url
        self.region = region
        self.running = False
        self.resource_monitor = GPUResourceMonitor(gpu_usage_threshold=gpu_threshold, memory_threshold=mem_threshold, cpu_threshold=cpu_threshold)
        # Control how often we poll and how long we wait when idle/busy
        default_sleep = int(os.getenv('ORCH_SLEEP_SECONDS', '30'))
        self.sleep_seconds = sleep_seconds if sleep_seconds is not None else default_sleep
        
        # Initialize AWS clients
        try:
            self.sqs_client = boto3.client('sqs', region_name=region)
            self.s3_client = boto3.client('s3', region_name=region)
        except NoCredentialsError:
            logger.error("X AWS credentials not configured")
            sys.exit(1)
        
        # Job tracking
        self.current_job = None
        self.job_history = []
        self.current_receipt_handle: Optional[str] = None
        self.current_queue_url: Optional[str] = None
        self._last_visibility_extend: float = 0.0
        self._visibility_extend_seconds: int = int(os.getenv('ORCH_VIS_EXT_SECONDS', '300'))  # 5 minutes
        self._tracker = JobTracker(table_name=os.getenv('DYNAMODB_TABLE', 'streamsniped_jobs'), region=os.getenv('DYNAMODB_REGION', region))
        # Testing mode: disable retries and never re-run failed/interrupted items
        self.no_retries: bool = is_truthy(os.getenv('ORCH_NO_RETRIES'), default=False)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def receive_message(self, queue_url: str, wait_time: Optional[int] = None) -> Optional[Dict]:
        """Receive a message from SQS queue"""
        try:
            # Use long polling but cap by our sleep interval for responsiveness
            effective_wait = 20 if wait_time is None else max(0, min(int(wait_time), 20))
            response = self.sqs_client.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=effective_wait,
                VisibilityTimeout=3600  # 1 hour visibility timeout
            )
            
            if 'Messages' in response:
                message = response['Messages'][0]
                logger.info(f"[SQS] message received id={message['MessageId']}")
                return message
            else:
                return None
                
        except ClientError as e:
            logger.error(f"Error receiving message from {queue_url}: {e}")
            return None
    
    def delete_message(self, queue_url: str, receipt_handle: str) -> bool:
        """Delete a message from SQS queue"""
        try:
            self.sqs_client.delete_message(
                QueueUrl=queue_url,
                ReceiptHandle=receipt_handle
            )
            logger.info("Message deleted from queue")
            return True
        except ClientError as e:
            logger.error(f"Error deleting message: {e}")
            return False
    
    def process_clip_queue(self) -> bool:
        """Process clip queue - returns True if job was processed"""
        logger.info("🔍 Checking clip queue...")
        
        message = self.receive_message(self.clip_queue_url)
        if not message:
            return False
        
        try:
            body = json.loads(message['Body'])
            vod_id = body.get('vod_id')
            manifest_uri = body.get('manifest')
            # Manual force re-run support for clip jobs (robustly parse booleans/strings)
            raw_force = body.get('force') if isinstance(body, dict) else None
            if isinstance(raw_force, bool):
                force_flag = raw_force
            elif isinstance(raw_force, (int, float)):
                force_flag = (raw_force != 0)
            elif isinstance(raw_force, str):
                force_flag = is_truthy(raw_force, default=False)
            else:
                force_flag = False
            try:
                source = str(body.get('source', '')).lower() if isinstance(body, dict) else ''
            except Exception:
                source = ''
            always_run_manual = is_truthy(os.getenv('ORCH_ALWAYS_RUN_MANUAL'), default=False)
            force_rerun = bool(force_flag or (source == 'manual-trigger' and always_run_manual))
            
            if not vod_id or not manifest_uri:
                logger.error("Invalid clip job message format")
                self.delete_message(self.clip_queue_url, message['ReceiptHandle'])
                return False
            
            logger.info(f"[JOB RECEIVED] type=clip vod_id={vod_id} manifest={manifest_uri}")
            # Drop duplicates or handle retries policy based on DynamoDB status
            try:
                existing = self._tracker.get_job(vod_id)
            except Exception:
                existing = None
            if existing and existing.get('status') in (JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value):
                if not force_rerun:
                    logger.info(f"Job {vod_id} already {existing.get('status')}; dropping duplicate message")
                    self.delete_message(self.clip_queue_url, message['ReceiptHandle'])
                    return False
                else:
                    logger.info(f"Force re-run enabled; bypassing duplicate drop for job {vod_id}")

            # No-retries policy: mark stale RUNNING as FAILED and drop
            if (not force_rerun) and self.no_retries and existing and existing.get('status') == JobStatus.RUNNING.value:
                try:
                    lease = int(existing.get('lease_expires_at') or 0)
                except Exception:
                    lease = 0
                now_ts = int(time.time())
                if lease <= now_ts:
                    logger.info(f"No-retries mode: dropping stale RUNNING job {vod_id} and marking FAILED")
                    try:
                        self._tracker.release_and_mark(vod_id, JobStatus.FAILED, metadata={'error_message': 'no-retries: lease expired'})
                    except Exception:
                        pass
                else:
                    logger.info(f"Job {vod_id} currently RUNNING elsewhere; dropping message (no-retries mode)")
                self.delete_message(self.clip_queue_url, message['ReceiptHandle'])
                return False

            # Acquire distributed lock (allowed to proceed)
            if not self._tracker.acquire_lock(vod_id, job_type='clip', lease_seconds=max(600, self._visibility_extend_seconds*3)):
                logger.info(f"Job {vod_id} is locked or running; deleting duplicate message")
                self.delete_message(self.clip_queue_url, message['ReceiptHandle'])
                return False

            # Create and process clip job
            job = JobProcessor("clip", manifest_uri, vod_id)
            self.current_job = job
            self.current_queue_url = self.clip_queue_url
            self.current_receipt_handle = message['ReceiptHandle']
            
            success = job.process()
            
            # Mark status and delete message regardless of success (to avoid infinite retries)
            try:
                final_status = JobStatus.COMPLETED if success else JobStatus.FAILED
                self._tracker.release_and_mark(vod_id, final_status)
            except Exception:
                pass
            self.delete_message(self.clip_queue_url, message['ReceiptHandle'])
            
            # Record job history (cap to last 100 items to avoid memory growth)
            self.job_history.append({
                'type': 'clip',
                'vod_id': vod_id,
                'success': success,
                'start_time': job.start_time.isoformat() if job.start_time else None,
                'end_time': job.end_time.isoformat() if job.end_time else None
            })
            if len(self.job_history) > 100:
                self.job_history = self.job_history[-100:]
            
            self.current_job = None
            self.current_queue_url = None
            self.current_receipt_handle = None
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in clip message: {e}")
            self.delete_message(self.clip_queue_url, message['ReceiptHandle'])
            return False
        except Exception as e:
            logger.error(f"Error processing clip job: {e}")
            if 'ReceiptHandle' in message:
                self.delete_message(self.clip_queue_url, message['ReceiptHandle'])
            return False
    
    def process_render_queue(self) -> bool:
        """Process render queue - returns True if job was processed"""
        logger.info("🔍 Checking render queue...")
        
        message = self.receive_message(self.render_queue_url)
        if not message:
            return False

    def process_full_queue(self) -> bool:
        """Process full workflow queue - returns True if job was processed"""
        if not self.full_queue_url:
            return False
        logger.info("🔍 Checking full queue...")
        message = self.receive_message(self.full_queue_url)
        if not message:
            return False
        try:
            body = json.loads(message['Body'])
            vod_id = body.get('vod_id') or body.get('id')
            # Manual force re-run support for full jobs (robustly parse booleans/strings)
            raw_force = body.get('force') if isinstance(body, dict) else None
            if isinstance(raw_force, bool):
                force_flag = raw_force
            elif isinstance(raw_force, (int, float)):
                force_flag = (raw_force != 0)
            elif isinstance(raw_force, str):
                force_flag = is_truthy(raw_force, default=False)
            else:
                force_flag = False
            try:
                source = str(body.get('source', '')).lower() if isinstance(body, dict) else ''
            except Exception:
                source = ''
            always_run_manual = is_truthy(os.getenv('ORCH_ALWAYS_RUN_MANUAL'), default=False)
            force_rerun = bool(force_flag or (source == 'manual-trigger' and always_run_manual))
            if not vod_id:
                logger.error("Invalid full job message format")
                self.delete_message(self.full_queue_url, message['ReceiptHandle'])
                return False
            logger.info(f"[JOB RECEIVED] type=full vod_id={vod_id}")
            # Drop duplicates or handle retries policy based on DynamoDB status
            try:
                existing = self._tracker.get_job(vod_id)
            except Exception:
                existing = None
            if existing and existing.get('status') in (JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value):
                if not force_rerun:
                    logger.info(f"Job {vod_id} already {existing.get('status')}; dropping duplicate message")
                    self.delete_message(self.full_queue_url, message['ReceiptHandle'])
                    return False
                else:
                    logger.info(f"Force re-run enabled; bypassing duplicate drop for job {vod_id}")

            if (not force_rerun) and self.no_retries and existing and existing.get('status') == JobStatus.RUNNING.value:
                try:
                    lease = int(existing.get('lease_expires_at') or 0)
                except Exception:
                    lease = 0
                now_ts = int(time.time())
                if lease <= now_ts:
                    logger.info(f"No-retries mode: dropping stale RUNNING job {vod_id} and marking FAILED")
                    try:
                        self._tracker.release_and_mark(vod_id, JobStatus.FAILED, metadata={'error_message': 'no-retries: lease expired'})
                    except Exception:
                        pass
                else:
                    logger.info(f"Job {vod_id} currently RUNNING elsewhere; dropping message (no-retries mode)")
                self.delete_message(self.full_queue_url, message['ReceiptHandle'])
                return False

            if not self._tracker.acquire_lock(vod_id, job_type='full', lease_seconds=max(1800, self._visibility_extend_seconds*6)):
                logger.info(f"Job {vod_id} is locked or running; deleting duplicate message")
                self.delete_message(self.full_queue_url, message['ReceiptHandle'])
                return False

            job = JobProcessor("full", None, vod_id)
            self.current_job = job
            self.current_queue_url = self.full_queue_url
            self.current_receipt_handle = message['ReceiptHandle']
            success = job.process()
            try:
                final_status = JobStatus.COMPLETED if success else JobStatus.FAILED
                self._tracker.release_and_mark(vod_id, final_status)
            except Exception:
                pass
            self.delete_message(self.full_queue_url, message['ReceiptHandle'])
            self.job_history.append({
                'type': 'full', 'vod_id': vod_id, 'success': success,
                'start_time': job.start_time.isoformat() if job.start_time else None,
                'end_time': job.end_time.isoformat() if job.end_time else None
            })
            if len(self.job_history) > 100:
                self.job_history = self.job_history[-100:]
            self.current_job = None
            self.current_queue_url = None
            self.current_receipt_handle = None
            return True
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in full message: {e}")
            self.delete_message(self.full_queue_url, message['ReceiptHandle'])
            return False
        except Exception as e:
            logger.error(f"Error processing full job: {e}")
            if 'ReceiptHandle' in message:
                self.delete_message(self.full_queue_url, message['ReceiptHandle'])
            return False
        
        try:
            body = json.loads(message['Body'])
            vod_id = body.get('vod_id')
            manifest_uri = body.get('manifest')
            
            if not vod_id or not manifest_uri:
                logger.error("Invalid render job message format")
                self.delete_message(self.render_queue_url, message['ReceiptHandle'])
                return False
            
            logger.info(f"[JOB RECEIVED] type=render vod_id={vod_id} manifest={manifest_uri}")
            # Drop duplicates or handle retries policy based on DynamoDB status
            try:
                existing = self._tracker.get_job(vod_id)
            except Exception:
                existing = None
            if existing and existing.get('status') in (JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value):
                logger.info(f"Job {vod_id} already {existing.get('status')}; dropping duplicate message")
                self.delete_message(self.render_queue_url, message['ReceiptHandle'])
                return False

            if self.no_retries and existing and existing.get('status') == JobStatus.RUNNING.value:
                try:
                    lease = int(existing.get('lease_expires_at') or 0)
                except Exception:
                    lease = 0
                now_ts = int(time.time())
                if lease <= now_ts:
                    logger.info(f"No-retries mode: dropping stale RUNNING job {vod_id} and marking FAILED")
                    try:
                        self._tracker.release_and_mark(vod_id, JobStatus.FAILED, metadata={'error_message': 'no-retries: lease expired'})
                    except Exception:
                        pass
                else:
                    logger.info(f"Job {vod_id} currently RUNNING elsewhere; dropping message (no-retries mode)")
                self.delete_message(self.render_queue_url, message['ReceiptHandle'])
                return False

            if not self._tracker.acquire_lock(vod_id, job_type='render', lease_seconds=max(1200, self._visibility_extend_seconds*4)):
                logger.info(f"Job {vod_id} is locked or running; deleting duplicate message")
                self.delete_message(self.render_queue_url, message['ReceiptHandle'])
                return False

            # Create and process render job
            job = JobProcessor("render", manifest_uri, vod_id)
            self.current_job = job
            self.current_queue_url = self.render_queue_url
            self.current_receipt_handle = message['ReceiptHandle']
            
            success = job.process()
            
            # Mark and delete message regardless of success (to avoid infinite retries)
            try:
                final_status = JobStatus.COMPLETED if success else JobStatus.FAILED
                self._tracker.release_and_mark(vod_id, final_status)
            except Exception:
                pass
            self.delete_message(self.render_queue_url, message['ReceiptHandle'])
            
            # Record job history (cap to last 100 items to avoid memory growth)
            self.job_history.append({
                'type': 'render',
                'vod_id': vod_id,
                'success': success,
                'start_time': job.start_time.isoformat() if job.start_time else None,
                'end_time': job.end_time.isoformat() if job.end_time else None
            })
            if len(self.job_history) > 100:
                self.job_history = self.job_history[-100:]
            
            self.current_job = None
            self.current_queue_url = None
            self.current_receipt_handle = None
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in render message: {e}")
            self.delete_message(self.render_queue_url, message['ReceiptHandle'])
            return False
        except Exception as e:
            logger.error(f"Error processing render job: {e}")
            if 'ReceiptHandle' in message:
                self.delete_message(self.render_queue_url, message['ReceiptHandle'])
            return False
    
    def _retry_pending_uploads(self) -> bool:
        """Check and retry any pending uploads that are ready."""
        try:
            from utils.upload_scheduler import UploadScheduler
            scheduler = UploadScheduler()
            
            # Clean up old entries first (older than 7 days)
            try:
                scheduler.clear_old_entries(days=7)
            except Exception:
                pass
            
            MAX_RETRIES = 3
            ready_uploads = scheduler.get_ready_uploads(max_retries=MAX_RETRIES)
            if not ready_uploads:
                return False
            
            logger.info(f"Found {len(ready_uploads)} pending uploads ready to retry")
            
            any_success = False
            for upload in ready_uploads:
                vod_id = upload.get('vod_id')
                arc_idx = upload.get('arc_index')
                channel = upload.get('channel')
                retry_count = upload.get('retry_count', 0)
                
                # Double-check retry count (safety guard)
                if retry_count >= MAX_RETRIES:
                    logger.warning(f"VOD {vod_id} arc {arc_idx} exceeded max retries ({retry_count}/{MAX_RETRIES}), marking as permanently failed")
                    scheduler.mark_upload_permanently_failed(
                        vod_id=vod_id,
                        arc_index=arc_idx,
                        channel=channel,
                        reason=f"Max retries exceeded ({retry_count}/{MAX_RETRIES})"
                    )
                    continue
                
                logger.info(f"Retrying upload for VOD {vod_id} arc {arc_idx} on {channel} (attempt #{retry_count + 1}/{MAX_RETRIES})")
                
                # Create a temporary JobProcessor for this retry
                job = JobProcessor("full", None, vod_id)
                
                # Build retry command
                cmd = [sys.executable, '-u', 'processing-scripts/auto_youtube_upload_arch.py', vod_id]
                if arc_idx is not None:
                    cmd += ['--arc', str(arc_idx)]
                if channel and channel != 'default':
                    cmd += ['--channels', channel]
                
                success = job._run_subprocess(
                    cmd, 
                    timeout_seconds=3600, 
                    step_name=f'Retry upload VOD {vod_id} arc {arc_idx}'
                )
                
                if success:
                    any_success = True
                    logger.info(f"✅ Successfully uploaded VOD {vod_id} arc {arc_idx} on retry")
                else:
                    # Check if this was the last retry attempt
                    new_retry_count = retry_count + 1
                    if new_retry_count >= MAX_RETRIES:
                        logger.warning(f"❌ Retry failed for VOD {vod_id} arc {arc_idx} - max retries reached, marking as permanently failed")
                        scheduler.mark_upload_permanently_failed(
                            vod_id=vod_id,
                            arc_index=arc_idx,
                            channel=channel,
                            reason=f"Failed after {new_retry_count} retry attempts"
                        )
                    else:
                        logger.warning(f"⚠️ Retry failed for VOD {vod_id} arc {arc_idx} - will retry again later ({new_retry_count}/{MAX_RETRIES})")
            
            return any_success
            
        except Exception as e:
            logger.error(f"Error processing pending uploads: {e}")
            return False
    
    def get_queue_status(self) -> Dict:
        """Get status of both queues"""
        status = {}
        
        for queue_name, queue_url in [("clip", self.clip_queue_url), ("render", self.render_queue_url)]:
            try:
                response = self.sqs_client.get_queue_attributes(
                    QueueUrl=queue_url,
                    AttributeNames=['ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible']
                )
                attributes = response['Attributes']
                status[queue_name] = {
                    'pending': int(attributes.get('ApproximateNumberOfMessages', 0)),
                    'processing': int(attributes.get('ApproximateNumberOfMessagesNotVisible', 0))
                }
            except ClientError as e:
                logger.error(f"Error getting {queue_name} queue status: {e}")
                status[queue_name] = {'pending': 0, 'processing': 0}
        # Full queue is optional
        if self.full_queue_url:
            try:
                response = self.sqs_client.get_queue_attributes(
                    QueueUrl=self.full_queue_url,
                    AttributeNames=['ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible']
                )
                attributes = response['Attributes']
                status['full'] = {
                    'pending': int(attributes.get('ApproximateNumberOfMessages', 0)),
                    'processing': int(attributes.get('ApproximateNumberOfMessagesNotVisible', 0))
                }
            except ClientError as e:
                logger.error(f"Error getting full queue status: {e}")
                status['full'] = {'pending': 0, 'processing': 0}
        
        return status
    
    def print_status(self):
        """Print current status"""
        status = self.get_queue_status()
        
        print("\n" + "="*60)
        print("GPU Orchestrator Daemon Status")
        print("="*60)
        print(f"Clip Queue: {status['clip']['pending']} pending, {status['clip']['processing']} processing")
        print(f"Render Queue: {status['render']['pending']} pending, {status['render']['processing']} processing")
        if 'full' in status:
            print(f"Full Queue: {status['full']['pending']} pending, {status['full']['processing']} processing")
        
        if self.current_job:
            job_type = self.current_job.job_type.upper()
            vod_id = self.current_job.vod_id
            duration = ""
            if self.current_job.start_time:
                elapsed = datetime.now() - self.current_job.start_time
                duration = f" (running for {elapsed})"
            print(f"Current Job: {job_type} for VOD {vod_id}{duration}")
        else:
            print("Current Job: None")
        
        # Show recent job history
        if self.job_history:
            print(f"\nRecent Jobs ({len(self.job_history)} total):")
            for job in self.job_history[-5:]:  # Last 5 jobs
                status_icon = "[OK]" if job['success'] else "[FAIL]"
                print(f"  {status_icon} {job['type'].upper()} - VOD {job['vod_id']}")
        
        print("="*60)
    
    def run(self):
        """Main daemon loop"""
        logger.info("Starting GPU Orchestrator Daemon")
        logger.info(f"Clip Queue: {self.clip_queue_url}")
        logger.info(f"Render Queue: {self.render_queue_url}")
        if self.full_queue_url:
            logger.info(f"Full Queue: {self.full_queue_url}")
        logger.info(f"Thresholds: GPU>{self.resource_monitor.gpu_usage_threshold}% | RAM>{self.resource_monitor.memory_threshold}% | CPU>{self.resource_monitor.cpu_threshold}%")
        logger.info(f"Sleep interval: {self.sleep_seconds}s")
        
        self.running = True
        iteration = 0
        max_iterations = 10000  # Reset counter to prevent unbounded growth
        last_pending_check = 0.0  # Track last time we checked for pending uploads
        
        while self.running:
            try:
                iteration += 1
                # Reset iteration counter to prevent memory issues with very long runs
                if iteration > max_iterations:
                    iteration = 1
                    logger.info(f"Reset iteration counter (was {max_iterations})")
                
                logger.info(f"Iteration {iteration}")
                
                # Print status every 10 iterations
                if iteration % 10 == 0:
                    self.print_status()
                
                # Check for pending uploads every hour (when not busy)
                now = time.time()
                if now - last_pending_check > 3600:  # 1 hour
                    try:
                        from utils.upload_scheduler import UploadScheduler
                        scheduler = UploadScheduler()
                        pending_count = scheduler.get_pending_count()
                        if pending_count > 0:
                            logger.info(f"📋 {pending_count} uploads pending retry")
                    except Exception:
                        pass
                    last_pending_check = now
                
                # Check if we're already processing a job
                if self.current_job:
                    # Heartbeat to keep lease and optionally extend SQS visibility timeout
                    try:
                        self._tracker.heartbeat(self.current_job.vod_id, lease_seconds=max(300, self._visibility_extend_seconds*2))
                    except Exception:
                        pass
                    # Extend message visibility while processing to prevent redelivery
                    try:
                        now = time.time()
                        if self.current_queue_url and self.current_receipt_handle and (now - self._last_visibility_extend) >= max(30, self.sleep_seconds):
                            self.sqs_client.change_message_visibility(
                                QueueUrl=self.current_queue_url,
                                ReceiptHandle=self.current_receipt_handle,
                                VisibilityTimeout=self._visibility_extend_seconds,
                            )
                            self._last_visibility_extend = now
                            logger.debug("Extended SQS message visibility")
                    except Exception as _e:
                        logger.warning(f"Could not extend SQS visibility: {_e}")
                    logger.info("Job in progress, waiting...")
                    time.sleep(self.sleep_seconds)
                    continue
                
                # Check resources before starting any job
                if not self.resource_monitor.check_gpu_available():
                    logger.info("GPU busy, waiting...")
                    time.sleep(self.sleep_seconds)
                    continue
                
                if not self.resource_monitor.check_system_resources():
                    logger.info("System resources busy, waiting...")
                    time.sleep(self.sleep_seconds)
                    continue
                
                # Priority: Process FULL first, then clips, then renders, then pending uploads
                job_processed = False
                # Full workflow
                if self.process_full_queue():
                    job_processed = True
                    logger.info("Full job processed")
                else:
                    # Try clip queue next
                    if self.process_clip_queue():
                        job_processed = True
                        logger.info("Clip job processed")
                    else:
                        # Try render queue if no clip jobs
                        if self.process_render_queue():
                            job_processed = True
                            logger.info("Render job processed")
                        else:
                            # Try pending uploads if no queue jobs
                            try:
                                if self._retry_pending_uploads():
                                    job_processed = True
                                    logger.info("Pending uploads processed")
                            except Exception as e:
                                logger.error(f"Error checking pending uploads: {e}")
                
                if not job_processed:
                    logger.info("No jobs available, sleeping...")
                    time.sleep(self.sleep_seconds)
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                # Increase sleep time for network errors to reduce resource usage
                if "Could not connect" in str(e) or "endpoint URL" in str(e):
                    logger.info("Network error detected, extending sleep time...")
                    time.sleep(self.sleep_seconds * 2)  # Double sleep for network issues
                else:
                    time.sleep(self.sleep_seconds)
        
        logger.info("GPU Orchestrator Daemon stopped")

def main():
    parser = argparse.ArgumentParser(description='StreamSniped GPU Orchestrator Daemon')
    parser.add_argument('--clip-queue-url', default=os.getenv('CLIP_QUEUE_URL'), 
                       help='SQS clip queue URL')
    parser.add_argument('--render-queue-url', default=os.getenv('RENDER_QUEUE_URL'), 
                       help='SQS render queue URL')
    parser.add_argument('--full-queue-url', default=os.getenv('FULL_QUEUE_URL'), 
                       help='SQS full-workflow queue URL')
    parser.add_argument('--region', default=os.getenv('AWS_REGION', 'us-east-1'),
                       help='AWS region')
    parser.add_argument('--status', action='store_true',
                       help='Show queue status and exit')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--gpu-threshold', type=int, default=int(os.getenv('GPU_USAGE_THRESHOLD', '80')),
                       help='Max allowed GPU utilization percent before deferring work (default 80)')
    parser.add_argument('--mem-threshold', type=int, default=int(os.getenv('MEMORY_USAGE_THRESHOLD', '85')),
                       help='Max allowed RAM utilization percent before deferring work (default 85)')
    parser.add_argument('--cpu-threshold', type=int, default=int(os.getenv('CPU_USAGE_THRESHOLD', '90')),
                       help='Max allowed CPU utilization percent before deferring work (default 90)')
    parser.add_argument('--sleep-seconds', type=int, default=int(os.getenv('ORCH_SLEEP_SECONDS', '30')),
                       help='Seconds to sleep between loops or when busy/idle (default 30)')
    
    args = parser.parse_args()
    
    if not args.clip_queue_url or not args.render_queue_url:
        print("X Missing queue URLs. Set CLIP_QUEUE_URL and RENDER_QUEUE_URL environment variables")
        sys.exit(1)
    
    # Configure logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    else:
        # Quiet down noisy libraries by default
        logging.getLogger('botocore').setLevel(logging.WARNING)
        logging.getLogger('boto3').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # Create orchestrator
    orchestrator = GPUOrchestratorDaemon(
        clip_queue_url=args.clip_queue_url,
        render_queue_url=args.render_queue_url,
        full_queue_url=args.full_queue_url,
        region=args.region,
        gpu_threshold=args.gpu_threshold,
        mem_threshold=args.mem_threshold,
        cpu_threshold=args.cpu_threshold,
        sleep_seconds=args.sleep_seconds,
    )
    
    if args.status:
        orchestrator.print_status()
        return
    
    # Run the daemon
    orchestrator.run()

if __name__ == '__main__':
    main()
