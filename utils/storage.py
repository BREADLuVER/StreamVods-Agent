#!/usr/bin/env python3
"""
Storage abstraction layer for StreamSniped
Handles local files and S3 with unified interface
"""

import json
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, BinaryIO
from urllib.parse import urlparse
import logging

# Optional S3 imports
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Storage operation failed"""
    pass


class StorageManager:
    """Unified storage interface for local files and S3"""
    
    def __init__(self, temp_dir: str = "/tmp"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
        # S3 client (if available)
        self.s3_client = None
        self.storage_type = 'local'  # Default to local storage
        
        if S3_AVAILABLE:
            try:
                self.s3_client = boto3.client('s3')
                # Test S3 connectivity
                self.s3_client.list_buckets()
                logger.info("S3 client initialized successfully")
                self.storage_type = 's3'  # S3 is available and working
            except (NoCredentialsError, ClientError) as e:
                logger.warning(f"S3 not available: {e}")
                self.s3_client = None
    
    def _is_s3_uri(self, uri: str) -> bool:
        """Check if URI is S3 path"""
        return uri.startswith('s3://')
    
    def _parse_s3_uri(self, uri: str) -> tuple[str, str]:
        """Parse S3 URI into bucket and key"""
        if not self._is_s3_uri(uri):
            raise ValueError(f"Not an S3 URI: {uri}")
        
        parsed = urlparse(uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        return bucket, key
    
    def _get_content_type(self, uri: str, default: str = "application/octet-stream") -> str:
        """Determine content type based on file extension"""
        ext = Path(uri).suffix.lower()
        
        content_types = {
            '.json': 'application/json',
            '.txt': 'text/plain',
            '.mp4': 'video/mp4',
            '.wav': 'audio/wav',
            '.vtt': 'text/vtt',
            '.srt': 'application/x-subrip',
            '.ass': 'text/x-ass',
            '.log': 'text/plain',
        }
        
        return content_types.get(ext, default)
    
    def read_json(self, uri: str) -> Dict[str, Any]:
        """Read JSON from local file or S3"""
        try:
            if self._is_s3_uri(uri):
                return self._read_json_s3(uri)
            else:
                return self._read_json_local(uri)
        except Exception as e:
            logger.error(f"Failed to read JSON from {uri}: {e}")
            raise StorageError(f"Failed to read JSON from {uri}: {e}")
    
    def read_text(self, uri: str) -> str:
        """Read text from local file or S3"""
        try:
            if self._is_s3_uri(uri):
                return self._read_text_s3(uri)
            else:
                return self._read_text_local(uri)
        except Exception as e:
            logger.error(f"Failed to read text from {uri}: {e}")
            raise StorageError(f"Failed to read text from {uri}: {e}")
    
    def _read_json_local(self, uri: str) -> Dict[str, Any]:
        """Read JSON from local file"""
        with open(uri, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _read_text_local(self, uri: str) -> str:
        """Read text from local file"""
        with open(uri, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _read_json_s3(self, uri: str) -> Dict[str, Any]:
        """Read JSON from S3"""
        if not self.s3_client:
            raise StorageError("S3 client not available")
        
        bucket, key = self._parse_s3_uri(uri)
        
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            return json.loads(content)
        except ClientError as e:
            raise StorageError(f"S3 read failed for {uri}: {e}")
    
    def _read_text_s3(self, uri: str) -> str:
        """Read text from S3"""
        if not self.s3_client:
            raise StorageError("S3 client not available")
        
        bucket, key = self._parse_s3_uri(uri)
        
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            return response['Body'].read().decode('utf-8')
        except ClientError as e:
            raise StorageError(f"S3 read failed for {uri}: {e}")
    
    def write_json(self, uri: str, data: Dict[str, Any]) -> None:
        """Write JSON to local file or S3"""
        try:
            if self._is_s3_uri(uri):
                self._write_json_s3(uri, data)
            else:
                self._write_json_local(uri, data)
        except Exception as e:
            logger.error(f"Failed to write JSON to {uri}: {e}")
            raise StorageError(f"Failed to write JSON to {uri}: {e}")
    
    def save_json_with_cloud_backup(self, local_path: str, data: Dict[str, Any], 
                                   s3_key: str = None, force_s3: bool = None) -> bool:
        """
        Save JSON data locally and optionally to S3 based on environment
        
        Args:
            local_path: Local file path to save to
            data: JSON data to save
            s3_key: S3 key path (if None, derived from local_path)
            force_s3: Force S3 upload regardless of environment
        
        Returns:
            True if successful (at least local save worked)
        """
        # Always save locally first
        try:
            self._write_json_local(local_path, data)
            logger.info(f"Saved locally: {local_path}")
        except Exception as e:
            logger.error(f"Failed to save locally: {local_path}: {e}")
            return False
        
        # Determine if we should upload to S3
        container_mode = os.environ.get('CONTAINER_MODE', 'false').lower() in ['true', '1', 'yes']
        aws_region = os.environ.get('AWS_REGION')
        should_upload_s3 = force_s3 or container_mode or (aws_region is not None)
        
        if should_upload_s3 and self.s3_client:
            try:
                # Generate S3 key if not provided
                if s3_key is None:
                    # Convert local path to S3 key format
                    path_obj = Path(local_path)
                    if str(path_obj).startswith('data/'):
                        s3_key = str(path_obj)[5:]  # Remove 'data/' prefix
                    else:
                        s3_key = str(path_obj)
                
                # Get S3 bucket
                s3_bucket = os.environ.get('S3_BUCKET', 'streamsniped-dev-videos')
                s3_uri = f"s3://{s3_bucket}/{s3_key}"
                
                # Upload to S3
                self.upload_file(local_path, s3_uri)
                logger.info(f"Uploaded to S3: {s3_uri}")
                
            except Exception as e:
                logger.warning(f"S3 upload failed for {local_path}: {e}")
                # Don't fail the whole operation if S3 upload fails
        
        return True
    
    def write_text(self, uri: str, text: str) -> bool:
        """Write text to local file or S3"""
        try:
            if self._is_s3_uri(uri):
                self._write_text_s3(uri, text)
            else:
                self._write_text_local(uri, text)
            return True
        except Exception as e:
            logger.error(f"Failed to write text to {uri}: {e}")
            return False
    
    def _write_json_local(self, uri: str, data: Dict[str, Any]) -> None:
        """Write JSON to local file"""
        path = Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _write_text_local(self, uri: str, text: str) -> None:
        """Write text to local file"""
        path = Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
    
    def _write_json_s3(self, uri: str, data: Dict[str, Any]) -> None:
        """Write JSON to S3"""
        if not self.s3_client:
            raise StorageError("S3 client not available")
        
        bucket, key = self._parse_s3_uri(uri)
        content = json.dumps(data, indent=2, ensure_ascii=False)
        
        try:
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=content.encode('utf-8'),
                ContentType='application/json'
            )
        except ClientError as e:
            raise StorageError(f"S3 write failed for {uri}: {e}")
    
    def _write_text_s3(self, uri: str, text: str) -> None:
        """Write text to S3"""
        if not self.s3_client:
            raise StorageError("S3 client not available")
        
        bucket, key = self._parse_s3_uri(uri)
        
        try:
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=text.encode('utf-8'),
                ContentType='text/plain'
            )
        except ClientError as e:
            raise StorageError(f"S3 write failed for {uri}: {e}")
    
    def upload_file(self, local_path: str, uri: str) -> None:
        """Upload local file to S3 or copy to local destination"""
        try:
            if self._is_s3_uri(uri):
                self._upload_to_s3(local_path, uri)
            else:
                self._copy_local(local_path, uri)
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to {uri}: {e}")
            raise StorageError(f"Failed to upload {local_path} to {uri}: {e}")
    
    def _copy_local(self, src: str, dst: str) -> None:
        """Copy local file"""
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    
    def _upload_to_s3(self, local_path: str, uri: str) -> None:
        """Upload file to S3 with multipart support for large files"""
        if not self.s3_client:
            raise StorageError("S3 client not available")
        
        bucket, key = self._parse_s3_uri(uri)
        file_size = os.path.getsize(local_path)
        content_type = self._get_content_type(uri)
        
        try:
            # Use multipart upload for files > 100MB
            if file_size > 100 * 1024 * 1024:  # 100MB
                self._multipart_upload(local_path, bucket, key, content_type)
            else:
                self.s3_client.upload_file(
                    local_path,
                    bucket,
                    key,
                    ExtraArgs={'ContentType': content_type}
                )
        except ClientError as e:
            raise StorageError(f"S3 upload failed for {local_path} to {uri}: {e}")
    
    def _multipart_upload(self, local_path: str, bucket: str, key: str, content_type: str) -> None:
        """Upload large file using multipart upload"""
        try:
            self.s3_client.upload_file(
                local_path,
                bucket,
                key,
                ExtraArgs={
                    'ContentType': content_type,
                    'ServerSideEncryption': 'AES256'
                },
                Config=boto3.s3.transfer.TransferConfig(
                    multipart_threshold=1024 * 25,  # 25MB
                    max_concurrency=10,
                    multipart_chunksize=1024 * 25,  # 25MB
                    use_threads=True
                )
            )
        except Exception as e:
            raise StorageError(f"Multipart upload failed: {e}")
    
    def download_file(self, uri: str, local_path: str) -> None:
        """Download file from S3 or copy local file"""
        try:
            if self._is_s3_uri(uri):
                self._download_from_s3(uri, local_path)
            else:
                self._copy_local(uri, local_path)
        except Exception as e:
            logger.error(f"Failed to download {uri} to {local_path}: {e}")
            raise StorageError(f"Failed to download {uri} to {local_path}: {e}")
    
    def _download_from_s3(self, uri: str, local_path: str) -> None:
        """Download file from S3"""
        if not self.s3_client:
            raise StorageError("S3 client not available")
        
        bucket, key = self._parse_s3_uri(uri)
        local_dir = Path(local_path).parent
        local_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.s3_client.download_file(bucket, key, local_path)
        except ClientError as e:
            raise StorageError(f"S3 download failed for {uri}: {e}")
    
    def exists(self, uri: str) -> bool:
        """Check if file exists"""
        try:
            if self._is_s3_uri(uri):
                return self._exists_s3(uri)
            else:
                return Path(uri).exists()
        except Exception:
            return False
    
    def delete(self, uri: str) -> bool:
        """Delete file from local filesystem or S3"""
        try:
            if self._is_s3_uri(uri):
                return self._delete_s3(uri)
            else:
                return self._delete_local(uri)
        except Exception as e:
            logger.error(f"Failed to delete {uri}: {e}")
            return False
    
    def _exists_s3(self, uri: str) -> bool:
        """Check if file exists in S3"""
        if not self.s3_client:
            return False
        
        bucket, key = self._parse_s3_uri(uri)
        
        try:
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False
    
    def _delete_local(self, uri: str) -> bool:
        """Delete local file"""
        try:
            Path(uri).unlink()
            return True
        except Exception:
            return False
    
    def _delete_s3(self, uri: str) -> bool:
        """Delete file from S3"""
        if not self.s3_client:
            return False
        
        bucket, key = self._parse_s3_uri(uri)
        
        try:
            self.s3_client.delete_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False
    
    def get_temp_dir(self, vod_id: str) -> Path:
        """Get temporary directory for VOD processing"""
        temp_dir = self.temp_dir / vod_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir
    
    def cleanup_temp(self, vod_id: str) -> None:
        """Clean up temporary files for VOD"""
        temp_dir = self.temp_dir / vod_id
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temp directory for VOD {vod_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp for VOD {vod_id}: {e}")
    
    def list_files(self, prefix: str = "", limit: int = 100) -> list:
        """List files in storage with optional prefix"""
        try:
            if self._is_s3_uri(prefix) or (self.s3_client and prefix.startswith('s3://')):
                return self._list_files_s3(prefix, limit)
            else:
                return self._list_files_local(prefix, limit)
        except Exception as e:
            logger.error(f"Failed to list files with prefix {prefix}: {e}")
            return []
    
    def _list_files_local(self, prefix: str, limit: int) -> list:
        """List files in local directory"""
        try:
            path = Path(prefix) if prefix else Path(".")
            if not path.exists():
                return []
            
            files = []
            for item in path.iterdir():
                if item.is_file():
                    files.append(str(item))
                    if len(files) >= limit:
                        break
            return files
        except Exception:
            return []
    
    def _list_files_s3(self, prefix: str, limit: int) -> list:
        """List files in S3 bucket"""
        if not self.s3_client:
            return []
        
        try:
            if prefix.startswith('s3://'):
                bucket, key_prefix = self._parse_s3_uri(prefix)
            else:
                # Assume default bucket
                bucket = "streamsniped-temp"
                key_prefix = prefix
            
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=key_prefix,
                MaxKeys=limit
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    files.append(f"s3://{bucket}/{obj['Key']}")
            
            return files
        except Exception as e:
            logger.error(f"Failed to list S3 files: {e}")
            return []


# Global storage instance
storage = StorageManager()


# Convenience functions for backward compatibility
def read_json(uri: str) -> Dict[str, Any]:
    """Read JSON from storage"""
    return storage.read_json(uri)


def write_json(uri: str, data: Dict[str, Any]) -> None:
    """Write JSON to storage"""
    return storage.write_json(uri, data)


def upload_file(local_path: str, uri: str) -> None:
    """Upload file to storage"""
    return storage.upload_file(local_path, uri)


def download_file(uri: str, local_path: str) -> None:
    """Download file from storage"""
    return storage.download_file(uri, local_path) 