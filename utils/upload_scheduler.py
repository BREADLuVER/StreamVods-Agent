#!/usr/bin/env python3
"""
Upload Scheduler - Track failed uploads and retry them later
Handles YouTube quota limits and other temporary failures
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class UploadScheduler:
    """Manage pending uploads and retry scheduling"""
    
    def __init__(self, scheduler_file: str = "data/pending_uploads.json"):
        self.scheduler_file = Path(scheduler_file)
        self.scheduler_file.parent.mkdir(parents=True, exist_ok=True)
        
    def _load_pending(self) -> Dict:
        """Load pending uploads from file"""
        if not self.scheduler_file.exists():
            return {"pending": [], "completed": [], "failed": []}
        try:
            with open(self.scheduler_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load pending uploads: {e}")
            return {"pending": [], "completed": [], "failed": []}
    
    def _save_pending(self, data: Dict) -> None:
        """Save pending uploads to file"""
        try:
            with open(self.scheduler_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save pending uploads: {e}")
    
    def add_failed_upload(
        self, 
        vod_id: str, 
        arc_index: Optional[int], 
        reason: str,
        channel: str = "default",
        retry_after_hours: int = 24
    ) -> None:
        """Add a failed upload to the pending queue"""
        data = self._load_pending()
        
        # Calculate retry time
        retry_after = (datetime.now() + timedelta(hours=retry_after_hours)).isoformat()
        
        # Create upload entry
        entry = {
            "vod_id": vod_id,
            "arc_index": arc_index,
            "channel": channel,
            "reason": reason,
            "failed_at": datetime.now().isoformat(),
            "retry_after": retry_after,
            "retry_count": 0
        }
        
        # Check if already exists
        existing = None
        for i, item in enumerate(data["pending"]):
            if (item.get("vod_id") == vod_id and 
                item.get("arc_index") == arc_index and
                item.get("channel") == channel):
                existing = i
                break
        
        if existing is not None:
            # Update retry count
            data["pending"][existing]["retry_count"] = data["pending"][existing].get("retry_count", 0) + 1
            data["pending"][existing]["retry_after"] = retry_after
            data["pending"][existing]["failed_at"] = datetime.now().isoformat()
            data["pending"][existing]["reason"] = reason
            logger.info(f"Updated pending upload for VOD {vod_id} arc {arc_index} (retry #{data['pending'][existing]['retry_count']})")
        else:
            # Add new entry
            data["pending"].append(entry)
            logger.info(f"Added pending upload for VOD {vod_id} arc {arc_index}, retry after {retry_after}")
        
        self._save_pending(data)
    
    def get_ready_uploads(self, max_retries: int = 3) -> List[Dict]:
        """Get uploads that are ready to retry"""
        data = self._load_pending()
        now = datetime.now()
        
        ready = []
        for item in data["pending"]:
            # Skip if max retries exceeded
            if item.get("retry_count", 0) >= max_retries:
                continue
            
            # Check if retry time has passed
            try:
                retry_after = datetime.fromisoformat(item["retry_after"])
                if now >= retry_after:
                    ready.append(item)
            except Exception:
                # Invalid date, skip
                continue
        
        return ready
    
    def mark_upload_completed(self, vod_id: str, arc_index: Optional[int], channel: str) -> None:
        """Mark an upload as successfully completed"""
        data = self._load_pending()
        
        # Find and remove from pending
        remaining = []
        completed_item = None
        for item in data["pending"]:
            if (item.get("vod_id") == vod_id and 
                item.get("arc_index") == arc_index and
                item.get("channel") == channel):
                completed_item = item.copy()
                completed_item["completed_at"] = datetime.now().isoformat()
            else:
                remaining.append(item)
        
        data["pending"] = remaining
        if completed_item:
            data["completed"].append(completed_item)
            # Keep only last 100 completed
            data["completed"] = data["completed"][-100:]
            logger.info(f"Marked upload as completed for VOD {vod_id} arc {arc_index}")
        
        self._save_pending(data)
    
    def mark_upload_permanently_failed(self, vod_id: str, arc_index: Optional[int], channel: str, reason: str) -> None:
        """Mark an upload as permanently failed (max retries exceeded)"""
        data = self._load_pending()
        
        # Find and remove from pending
        remaining = []
        failed_item = None
        for item in data["pending"]:
            if (item.get("vod_id") == vod_id and 
                item.get("arc_index") == arc_index and
                item.get("channel") == channel):
                failed_item = item.copy()
                failed_item["permanently_failed_at"] = datetime.now().isoformat()
                failed_item["final_reason"] = reason
            else:
                remaining.append(item)
        
        data["pending"] = remaining
        if failed_item:
            data["failed"].append(failed_item)
            # Keep only last 100 failed
            data["failed"] = data["failed"][-100:]
            logger.warning(f"Marked upload as permanently failed for VOD {vod_id} arc {arc_index}: {reason}")
        
        self._save_pending(data)
    
    def get_pending_count(self) -> int:
        """Get count of pending uploads"""
        data = self._load_pending()
        return len(data["pending"])
    
    def get_pending_for_vod(self, vod_id: str) -> List[Dict]:
        """Get all pending uploads for a specific VOD"""
        data = self._load_pending()
        return [item for item in data["pending"] if item.get("vod_id") == vod_id]
    
    def clear_vod_pending(self, vod_id: str, mark_as_failed: bool = True) -> int:
        """Clear all pending uploads for a specific VOD"""
        data = self._load_pending()
        
        # Find all pending for this VOD
        vod_pending = []
        remaining_pending = []
        for item in data["pending"]:
            if item.get("vod_id") == vod_id:
                vod_pending.append(item)
            else:
                remaining_pending.append(item)
        
        if mark_as_failed:
            # Move to failed list
            for item in vod_pending:
                item["permanently_failed_at"] = datetime.now().isoformat()
                item["final_reason"] = "Manually cleared/cancelled"
                data["failed"].append(item)
        
        data["pending"] = remaining_pending
        self._save_pending(data)
        
        count = len(vod_pending)
        if count > 0:
            logger.info(f"Cleared {count} pending uploads for VOD {vod_id}")
        return count
    
    def clear_old_entries(self, days: int = 30) -> None:
        """Clear completed/failed entries older than specified days, and mark old pending as permanently failed"""
        data = self._load_pending()
        cutoff = datetime.now() - timedelta(days=days)
        
        # Move old pending entries to failed
        old_pending = []
        remaining_pending = []
        for item in data["pending"]:
            try:
                failed_at = datetime.fromisoformat(item.get("failed_at", "1970-01-01"))
                if failed_at < cutoff:
                    old_pending.append(item)
                else:
                    remaining_pending.append(item)
            except Exception:
                remaining_pending.append(item)  # Keep if we can't parse date
        
        # Mark old pending as permanently failed
        for item in old_pending:
            item["permanently_failed_at"] = datetime.now().isoformat()
            item["final_reason"] = f"Stale upload (failed_at > {days} days ago)"
            data["failed"].append(item)
            logger.info(f"Marked stale pending upload as failed: VOD {item.get('vod_id')} arc {item.get('arc_index')}")
        
        data["pending"] = remaining_pending
        
        # Filter completed
        data["completed"] = [
            item for item in data["completed"]
            if datetime.fromisoformat(item.get("completed_at", "1970-01-01")) > cutoff
        ]
        
        # Filter failed
        data["failed"] = [
            item for item in data["failed"]
            if datetime.fromisoformat(item.get("permanently_failed_at", "1970-01-01")) > cutoff
        ]
        
        self._save_pending(data)
        logger.info(f"Cleared entries older than {days} days ({len(old_pending)} pending moved to failed)")


def is_quota_exceeded_error(error_message: str) -> bool:
    """Check if error message indicates YouTube quota exceeded"""
    if not error_message:
        return False
    
    error_lower = str(error_message).lower()
    quota_indicators = [
        "uploadlimitexceeded",
        "quota exceeded",
        "exceeded the number of videos",
        "upload limit",
    ]
    
    return any(indicator in error_lower for indicator in quota_indicators)


def is_retriable_error(error_message: str) -> bool:
    """Check if error is retriable (quota, temporary network, etc)"""
    if not error_message:
        return False
    
    error_lower = str(error_message).lower()
    retriable_indicators = [
        "quota",
        "uploadlimitexceeded",
        "exceeded the number of videos",
        "upload limit",
        "timeout",
        "network",
        "connection",
        "503",
        "502",
        "504",
    ]
    
    return any(indicator in error_lower for indicator in retriable_indicators)

