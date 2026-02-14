from __future__ import annotations
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

# Optional AWS CloudWatch support (requires boto3 but not mandatory)
try:
    import boto3  # type: ignore

    _AWS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _AWS_AVAILABLE = False

_LOGGERS: dict[str, logging.Logger] = {}


def _new_run_id() -> str:
    """Generate sortable timestamp-based run identifier."""
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    rand = uuid.uuid4().hex[:6]
    return f"{ts}-{rand}"


class _CloudWatchHandler(logging.Handler):
    """Simple CloudWatch Logs handler using boto3.

    Created lazily – safe no-op if AWS creds or boto3 are missing. Any client
    errors are swallowed so local logging continues uninterrupted.
    """

    def __init__(self, group: str, stream: str):
        super().__init__()
        self.group = group
        self.stream = stream

        self.client = boto3.client("logs")  # type: ignore[name-defined]
        self.sequence_token: Optional[str] = None

        # Ensure group/stream exist (idempotent)
        try:
            self.client.create_log_group(logGroupName=group)
        except self.client.exceptions.ResourceAlreadyExistsException:  # type: ignore[attr-defined]
            pass
        except Exception:
            # Tolerate transient AWS control-plane failures
            pass
        try:
            self.client.create_log_stream(logGroupName=group, logStreamName=stream)
        except self.client.exceptions.ResourceAlreadyExistsException:  # type: ignore[attr-defined]
            pass
        except Exception:
            # Tolerate transient AWS control-plane failures
            pass

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        try:
            msg = self.format(record)
            event = {
                "timestamp": int(record.created * 1000),
                "message": msg,
            }
            kwargs = {
                "logGroupName": self.group,
                "logStreamName": self.stream,
                "logEvents": [event],
            }
            if self.sequence_token:
                kwargs["sequenceToken"] = self.sequence_token
            resp = self.client.put_log_events(**kwargs)  # type: ignore[arg-type]
            self.sequence_token = resp.get("nextSequenceToken")
        except Exception:
            # Never let CW failures break local logging
            logging.getLogger(__name__).warning("CloudWatch logging failed", exc_info=False)
            self.sequence_token = None


def get_job_logger(job_type: str, vod_id: str, run_id: Optional[str] = None) -> tuple[str, logging.Logger]:
    """Return `(run_id, logger)` for the given job.

    • If `run_id` is provided, reuse it to unify logs across parent/child.
    • Otherwise, generate a new run-id so reruns never collide.
    • Handlers: file under `logs/jobs/<vod_id>/`, stdout, optional CloudWatch.
    """

    run_id = run_id or _new_run_id()
    key = f"{job_type}:{vod_id}:{run_id}"
    if key in _LOGGERS:
        return run_id, _LOGGERS[key]

    # ---------- create logger ----------
    logger = logging.getLogger(key)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = "% (asctime)s | %(levelname)-8s | %(run_id)s | %(message)s".replace("% (", "%(")  # formatter string tidy
    formatter = logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S")

    # file handler
    log_dir = Path("logs") / "jobs" / vod_id
    log_dir.mkdir(parents=True, exist_ok=True)
    file_path = log_dir / f"{job_type}_{run_id}.log"
    fh = logging.FileHandler(file_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # stdout handler (streaming)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # CloudWatch (optional)
    if _AWS_AVAILABLE and os.getenv("ENABLE_CLOUDWATCH", "true").lower() in {"1", "true", "yes"}:
        try:
            cw_handler = _CloudWatchHandler("StreamSnipedJobs", f"{job_type}-{vod_id}")
            cw_handler.setFormatter(formatter)
            logger.addHandler(cw_handler)
        except Exception:
            # Ensure formatter has required field even before adapter is created
            logger.warning("CloudWatch handler not attached", exc_info=False, extra={"run_id": run_id})

    # Inject run_id into every record automatically
    adapter = logging.LoggerAdapter(logger, {"run_id": run_id})
    _LOGGERS[key] = adapter  # type: ignore[assignment]
    return run_id, adapter


def close_job_logger(job_type: str, vod_id: str, run_id: str) -> None:
    """Flush, close, and detach handlers for a previously created job logger.

    Prevents handler/file-descriptor leaks in long-lived processes.
    Safe to call multiple times.
    """
    key = f"{job_type}:{vod_id}:{run_id}"
    logger = _LOGGERS.pop(key, None)
    if not logger:
        return
    base_logger = getattr(logger, "logger", logger)
    for handler in list(getattr(base_logger, "handlers", [])):
        try:
            handler.flush()
        except Exception:
            pass
        try:
            handler.close()
        except Exception:
            pass
        try:
            base_logger.removeHandler(handler)
        except Exception:
            pass
