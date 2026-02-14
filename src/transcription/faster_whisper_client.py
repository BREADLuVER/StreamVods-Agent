#!/usr/bin/env python3
"""
Faster-Whisper client helpers with GPU/CPU autodetect and concurrency control.

Refs: https://github.com/SYSTRAN/faster-whisper
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple
from threading import Lock, BoundedSemaphore

# Environment guards to prevent duplicate OpenMP runtime initialization crashes.
# This is particularly important on Windows when libraries like NumPy/MKL and
# CTranslate2 (used by faster-whisper) both load OpenMP (libiomp5md.dll).
# We set these before importing any heavy dependencies.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Allow duplicate OpenMP runtimes
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("WHISPER_OMP_THREADS", "1"))
os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("WHISPER_MKL_THREADS", "1"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ.get("WHISPER_OPENBLAS_THREADS", "1"))
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ.get("WHISPER_NUMEXPR_THREADS", "1"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from faster_whisper import WhisperModel


_MODEL = None
_MODEL_LOCK = Lock()
_WHISPER_SEM = BoundedSemaphore(max(1, int(os.getenv("WHISPER_MAX_PARALLEL", "1"))))


def _device_and_compute() -> Tuple[str, str]:
    device = os.getenv("WHISPER_DEVICE", "auto")
    compute = os.getenv("WHISPER_COMPUTE_TYPE", "auto")
    if device == "auto":
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        except Exception:
            device = "cpu"
    if compute == "auto":
        if device == "cuda":
            compute = os.getenv("WHISPER_CUDA_COMPUTE", "int8_float16")
        else:
            compute = "int8"
    return device, compute


def _load_model() -> WhisperModel:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        model_size = os.getenv("WHISPER_MODEL", "large-v3")
        device, compute_type = _device_and_compute()
        _MODEL = WhisperModel(model_size, device=device, compute_type=compute_type)
        return _MODEL


def transcribe_audio_file(audio_path: Path) -> Dict:
    """Transcribe a single audio file and return { text, language, segments }.

    segments: List[{ start, end, text }]
    """
    model = _load_model()
    with _WHISPER_SEM:
        seg_iter, info = model.transcribe(
            str(audio_path),
            beam_size=int(os.getenv("WHISPER_BEAM_SIZE", "5")),
            vad_filter=(os.getenv("WHISPER_VAD", "true").lower() in ("1","true","yes")),
        )
        segments_list = list(seg_iter)

    out_segments: List[Dict] = []
    full_text_parts: List[str] = []
    for s in segments_list:
        start = float(getattr(s, "start", 0.0) or 0.0)
        end = float(getattr(s, "end", 0.0) or 0.0)
        text = str(getattr(s, "text", "") or "").strip()
        if text:
            full_text_parts.append(text)
        out_segments.append({"start": start, "end": end, "text": text})

    return {
        "text": " ".join(full_text_parts).strip(),
        "language": getattr(info, "language", "en"),
        "segments": out_segments,
    }


