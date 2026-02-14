#!/usr/bin/env python3
"""
Logging configuration for StreamSniped
Supports structured logging, environment-based config, and proper error handling
"""

import os
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)


def setup_logging(
    level: str = None,
    format_type: str = None,
    container_mode: bool = None,
    vod_id: str = None,
    log_file: str = None
) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format_type: Log format (json, text)
        container_mode: Whether running in container
        vod_id: VOD ID for structured logging
        log_file: Optional path to log file
    """
    # Get configuration from environment
    if level is None:
        level = os.getenv('LOG_LEVEL', 'INFO')
    
    if format_type is None:
        format_type = os.getenv('LOG_FORMAT', 'text')
    
    if container_mode is None:
        container_mode = os.getenv('CONTAINER_MODE', 'false').lower() == 'true'
    
    # Convert string level to logging constant
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    log_level = level_map.get(level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Set formatter based on format type
    if format_type.lower() == 'json':
        formatter = StructuredFormatter()
    else:
        # Text formatter with extra context
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        if vod_id:
            format_string = f'[VOD:{vod_id}] ' + format_string
        if container_mode:
            format_string = '[CONTAINER] ' + format_string
        formatter = logging.Formatter(format_string)
    
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if log_file is provided
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Create logger for StreamSniped
    logger = logging.getLogger('streamsniped')
    logger.setLevel(log_level)
    
    # Add extra context if provided
    if vod_id:
        logger = logging.getLogger(f'streamsniped.{vod_id}')
    
    return logger


def log_with_context(
    logger: logging.Logger,
    level: str,
    message: str,
    vod_id: str = None,
    step: str = None,
    extra: Dict[str, Any] = None
):
    """Log message with structured context"""
    extra_fields = {}
    
    if vod_id:
        extra_fields['vod_id'] = vod_id
    
    if step:
        extra_fields['step'] = step
    
    if extra:
        extra_fields.update(extra)
    
    # Add extra fields to log record
    record = logger.makeRecord(
        logger.name, getattr(logging, level.upper()), 
        '', 0, message, (), None
    )
    record.extra_fields = extra_fields
    
    logger.handle(record)


def log_job_start(logger: logging.Logger, vod_id: str, metadata: Dict[str, Any] = None):
    """Log job start with metadata"""
    extra = {'event': 'job_start'}
    if metadata:
        extra.update(metadata)
    
    log_with_context(logger, 'INFO', f"Starting job for VOD: {vod_id}", vod_id, extra=extra)


def log_job_step(logger: logging.Logger, vod_id: str, step: str, metadata: Dict[str, Any] = None):
    """Log job step transition"""
    extra = {'event': 'job_step', 'step': step}
    if metadata:
        extra.update(metadata)
    
    log_with_context(logger, 'INFO', f"Job step: {step}", vod_id, step, extra)


def log_job_complete(logger: logging.Logger, vod_id: str, metadata: Dict[str, Any] = None):
    """Log job completion"""
    extra = {'event': 'job_complete'}
    if metadata:
        extra.update(metadata)
    
    log_with_context(logger, 'INFO', f"Job completed for VOD: {vod_id}", vod_id, extra=extra)


def log_job_error(logger: logging.Logger, vod_id: str, error: str, metadata: Dict[str, Any] = None):
    """Log job error"""
    extra = {'event': 'job_error', 'error': error}
    if metadata:
        extra.update(metadata)
    
    log_with_context(logger, 'ERROR', f"Job failed for VOD: {vod_id}: {error}", vod_id, extra=extra)


def log_command_execution(logger: logging.Logger, vod_id: str, command: str, success: bool, 
                         duration: float = None, metadata: Dict[str, Any] = None):
    """Log command execution"""
    extra = {
        'event': 'command_execution',
        'command': command,
        'success': success
    }
    
    if duration is not None:
        extra['duration_seconds'] = duration
    
    if metadata:
        extra.update(metadata)
    
    status = "succeeded" if success else "failed"
    log_with_context(logger, 'INFO', f"Command {status}: {command}", vod_id, extra=extra)


def log_storage_operation(logger: logging.Logger, vod_id: str, operation: str, 
                         file_path: str, success: bool, metadata: Dict[str, Any] = None):
    """Log storage operation"""
    extra = {
        'event': 'storage_operation',
        'operation': operation,
        'file_path': file_path,
        'success': success
    }
    
    if metadata:
        extra.update(metadata)
    
    status = "succeeded" if success else "failed"
    log_with_context(logger, 'INFO', f"Storage {operation} {status}: {file_path}", vod_id, extra=extra)


# Convenience function for quick setup
def get_logger(vod_id: str = None) -> logging.Logger:
    """Get configured logger for StreamSniped"""
    return setup_logging(vod_id=vod_id) 