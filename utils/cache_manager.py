#!/usr/bin/env python3
"""
Cache Manager for StreamSniped Cloud Workflow
Handles caching logic to prevent redundant data generation
"""

import json
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

# Add project root to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from storage import StorageManager
from src.config import config

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching for StreamSniped workflow steps"""
    
    def __init__(self, vod_id: str):
        self.vod_id = vod_id
        self.storage = StorageManager()
        self.cache_dir = Path(f"data/cache/{vod_id}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache metadata file
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata from file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save cache metadata to file"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def _get_file_hash(self, file_path: Path) -> Optional[str]:
        """Calculate SHA256 hash of file"""
        try:
            if not file_path.exists():
                return None
            
            # Skip directories - they can't be hashed like files
            if file_path.is_dir():
                logger.warning(f"Skipping directory hash calculation: {file_path}")
                return None
            
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return None
    
    def _get_s3_file_hash(self, s3_uri: str) -> Optional[str]:
        """Get file hash from S3 metadata or calculate it"""
        if not s3_uri:
            return None
            
        try:
            # Try to get hash from S3 metadata first
            bucket, key = self.storage._parse_s3_uri(s3_uri)
            s3_client = self.storage.s3_client
            
            if s3_client:
                response = s3_client.head_object(Bucket=bucket, Key=key)
                metadata = response.get('Metadata', {})
                if 'file-hash' in metadata:
                    return metadata['file-hash']
                
                # If no hash in metadata, download and calculate
                temp_file = self.cache_dir / f"temp_{Path(key).name}"
                self.storage.download_file(s3_uri, str(temp_file))
                file_hash = self._get_file_hash(temp_file)
                temp_file.unlink()  # Clean up
                return file_hash
        except Exception as e:
            logger.warning(f"Failed to get S3 file hash for {s3_uri}: {e}")
        return None
    
    def _check_file_exists(self, file_path: Path, s3_uri: Optional[str] = None) -> bool:
        """Check if file exists locally or in S3"""
        # Check local file first
        if file_path.exists():
            return True
        
        # Check S3 if URI provided
        if s3_uri and self.storage.exists(s3_uri):
            return True
        
        # For local paths, also check equivalent S3 location
        if not s3_uri and not file_path.is_absolute():
            # Convert local path to S3 equivalent
            s3_key = self._normalize_path_for_comparison(str(file_path))
            s3_equivalent = f"s3://{os.getenv('S3_BUCKET', 'streamsniped-dev-videos')}/{s3_key}"
            if self.storage.exists(s3_equivalent):
                return True
        
        # For S3 paths, also check equivalent local location
        if s3_uri and s3_uri.startswith('s3://'):
            # Convert S3 path to local equivalent
            s3_key = self._normalize_path_for_comparison(s3_uri)
            local_equivalent = Path(f"data/{s3_key}")
            if local_equivalent.exists():
                return True
        
        return False
    
    def _normalize_path_for_comparison(self, file_path: str) -> str:
        """Convert S3 and local paths to comparable format"""
        if file_path.startswith('s3://'):
            # Extract just the key part for comparison
            bucket, key = self.storage._parse_s3_uri(file_path)
            return key
        else:
            # Convert local path to S3-style key format
            path_obj = Path(file_path)
            # Remove 'data/' prefix if present and convert to forward slashes
            path_str = str(path_obj).replace('\\', '/')
            if path_str.startswith('data/'):
                path_str = path_str[5:]  # Remove 'data/' prefix
            return path_str
    
    def _paths_are_equivalent(self, path1: str, path2: str) -> bool:
        """Check if two paths (S3 and local) refer to the same logical file"""
        norm1 = self._normalize_path_for_comparison(path1)
        norm2 = self._normalize_path_for_comparison(path2)
        return norm1 == norm2
    
    def _get_file_timestamp(self, file_path: Path, s3_uri: Optional[str] = None) -> Optional[float]:
        """Get file modification timestamp"""
        if file_path.exists():
            return file_path.stat().st_mtime
        
        if s3_uri:
            try:
                bucket, key = self.storage._parse_s3_uri(s3_uri)
                s3_client = self.storage.s3_client
                if s3_client:
                    response = s3_client.head_object(Bucket=bucket, Key=key)
                    return response['LastModified'].timestamp()
            except Exception as e:
                logger.warning(f"Failed to get S3 timestamp for {s3_uri}: {e}")
        
        return None
    
    def get_cache_key(self, step_name: str, **kwargs) -> str:
        """Generate consistent cache key for step"""
        # Create deterministic key based on step name and parameters
        key_parts = [step_name, self.vod_id]
        
        # Add sorted kwargs to ensure consistent ordering
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        
        return "_".join(key_parts)
    
    def should_skip_step(self, step_name: str, output_files: List[str], 
                        dependencies: List[str] = None,
                        force_regenerate: bool = False) -> Tuple[bool, str]:
        """
        Check if step should be skipped based on cache
        
        Args:
            step_name: Name of the workflow step
            output_files: List of expected output file paths
            dependencies: List of dependency file paths to check
            force_regenerate: Force regeneration regardless of cache
        
        Returns:
            (should_skip, reason)
        """
        if force_regenerate:
            return False, "Force regeneration requested"
        
        cache_key = self.get_cache_key(step_name)
        
        # Check if we have recent cache metadata
        if cache_key in self.metadata:
            cache_info = self.metadata[cache_key]
            cache_time = cache_info.get('timestamp', 0)
            
                    # Skip age check - cache is valid as long as files exist
        # S3 cleanup is managed externally by design
        
        # Check if all output files exist and are recent
        all_outputs_exist = True
        missing_outputs = []
        
        # Check all output files for existence (both requested and cached versions)
        files_to_check = output_files[:]  # Start with requested files
        
        # Add cached outputs if they exist and are equivalent
        if cache_key in self.metadata:
            cached_outputs = self.metadata[cache_key].get('output_files', [])
            
            # For each requested output, check if there's an equivalent cached version
            for output_file in output_files:
                for cached_output in cached_outputs:
                    if (self._paths_are_equivalent(output_file, cached_output) and 
                        cached_output not in files_to_check):
                        files_to_check.append(cached_output)
        
        # Check all files (both requested and cached equivalents)
        for file_path_str in files_to_check:
            file_path = Path(file_path_str)
            s3_uri = None
            
            # Determine S3 URI based on file path
            if file_path_str.startswith('s3://'):
                s3_uri = file_path_str
                # For S3 paths, create a dummy local path for age checking
                file_path = Path(f"data/cache/{self.vod_id}/{Path(file_path_str).name}")
            else:
                # For local paths, determine equivalent S3 URI
                s3_key = self._normalize_path_for_comparison(file_path_str)
                s3_uri = f"s3://{os.getenv('S3_BUCKET', 'streamsniped-dev-videos')}/{s3_key}"
            
            # Check file existence (will check both local and S3)
            if not self._check_file_exists(file_path, s3_uri):
                all_outputs_exist = False
                missing_outputs.append(file_path_str)
                continue
            
            # Check file age - prefer local file timestamp, fall back to S3
            timestamp = None
            if file_path_str.startswith('s3://'):
                # For S3 files, check if local equivalent exists first
                s3_key = self._normalize_path_for_comparison(file_path_str)
                local_equivalent = Path(f"data/{s3_key}")
                if local_equivalent.exists():
                    timestamp = self._get_file_timestamp(local_equivalent, None)
                else:
                    timestamp = self._get_file_timestamp(file_path, s3_uri)
            else:
                # For local files, use local timestamp
                if file_path.exists():
                    timestamp = self._get_file_timestamp(file_path, None)
                else:
                    # Fall back to S3 timestamp if local doesn't exist
                    timestamp = self._get_file_timestamp(file_path, s3_uri)
            
            # Skip age check - files are valid as long as they exist
            # S3 cleanup is managed externally by design
                
        # If no files were checked (no cache metadata), check requested files directly
        if not files_to_check:
            for output_file in output_files:
                file_path = Path(output_file)
                s3_uri = None
                
                # Determine S3 URI based on file path
                if output_file.startswith('s3://'):
                    s3_uri = output_file
                    file_path = Path(f"data/cache/{self.vod_id}/{Path(output_file).name}")
                else:
                    # For local paths, determine equivalent S3 URI
                    s3_key = self._normalize_path_for_comparison(output_file)
                    s3_uri = f"s3://{os.getenv('S3_BUCKET', 'streamsniped-dev-videos')}/{s3_key}"
                
                # Check file existence (will check both local and S3)
                if not self._check_file_exists(file_path, s3_uri):
                    all_outputs_exist = False
                    missing_outputs.append(output_file)
                    continue
                
                # Check file age - prefer local file timestamp, fall back to S3
                timestamp = None
                if output_file.startswith('s3://'):
                    # For S3 files, check if local equivalent exists first
                    s3_key = self._normalize_path_for_comparison(output_file)
                    local_equivalent = Path(f"data/{s3_key}")
                    if local_equivalent.exists():
                        timestamp = self._get_file_timestamp(local_equivalent, None)
                    else:
                        timestamp = self._get_file_timestamp(file_path, s3_uri)
                else:
                    # For local files, use local timestamp
                    if file_path.exists():
                        timestamp = self._get_file_timestamp(file_path, None)
                    else:
                        # Fall back to S3 timestamp if local doesn't exist
                        timestamp = self._get_file_timestamp(file_path, s3_uri)
                
                # Skip age check - files are valid as long as they exist
                # S3 cleanup is managed externally by design
        
        if not all_outputs_exist:
            return False, f"Missing outputs: {', '.join(missing_outputs)}"
        
        # Check dependencies if provided
        if dependencies:
            for dep_file in dependencies:
                file_path = Path(dep_file)
                s3_uri = None
                
                if dep_file.startswith('s3://'):
                    s3_uri = dep_file
                    file_path = Path(f"data/cache/{self.vod_id}/{Path(dep_file).name}")
                
                if not self._check_file_exists(file_path, s3_uri):
                    return False, f"Dependency missing: {dep_file}"
                
                # Check if dependency has changed since last cache
                if cache_key in self.metadata:
                    cache_info = self.metadata[cache_key]
                    dep_hashes = cache_info.get('dependency_hashes', {})
                    
                    current_hash = self._get_file_hash(file_path) or self._get_s3_file_hash(s3_uri)
                    cached_hash = dep_hashes.get(dep_file)
                    
                    if current_hash and cached_hash and current_hash != cached_hash:
                        return False, f"Dependency changed: {dep_file}"
        
        return True, "All outputs exist and dependencies unchanged"
    
    def mark_step_completed(self, step_name: str, output_files: List[str], 
                           dependencies: List[str] = None, **kwargs):
        """Mark step as completed and cache its metadata"""
        cache_key = self.get_cache_key(step_name, **kwargs)
        
        # Calculate dependency hashes
        dependency_hashes = {}
        if dependencies:
            for dep_file in dependencies:
                file_path = Path(dep_file)
                s3_uri = None
                
                if dep_file.startswith('s3://'):
                    s3_uri = dep_file
                    file_path = Path(f"data/cache/{self.vod_id}/{Path(dep_file).name}")
                
                file_hash = self._get_file_hash(file_path) or self._get_s3_file_hash(s3_uri)
                if file_hash:
                    dependency_hashes[dep_file] = file_hash
        
        # Update metadata
        self.metadata[cache_key] = {
            'timestamp': datetime.now().timestamp(),
            'output_files': output_files,
            'dependencies': dependencies or [],
            'dependency_hashes': dependency_hashes,
            'step_name': step_name,
            'vod_id': self.vod_id
        }
        
        self._save_metadata()
        logger.info(f"Cached step completion: {step_name}")
    
    def get_cached_outputs(self, step_name: str, **kwargs) -> List[str]:
        """Get list of cached output files for a step"""
        cache_key = self.get_cache_key(step_name, **kwargs)
        
        if cache_key in self.metadata:
            return self.metadata[cache_key].get('output_files', [])
        
        return []
    
    def clear_cache(self, step_name: str = None):
        """Clear cache for specific step or all steps"""
        if step_name:
            cache_key = self.get_cache_key(step_name)
            if cache_key in self.metadata:
                del self.metadata[cache_key]
                logger.info(f"Cleared cache for step: {step_name}")
        else:
            self.metadata.clear()
            logger.info("Cleared all cache metadata")
        
        self._save_metadata()
    
    def get_cache_status(self) -> Dict:
        """Get current cache status"""
        return {
            'vod_id': self.vod_id,
            'cached_steps': list(self.metadata.keys()),
            'total_cached_steps': len(self.metadata),
            'cache_dir': str(self.cache_dir)
        }


def create_cache_manager(vod_id: str) -> CacheManager:
    """Factory function to create cache manager"""
    return CacheManager(vod_id) 