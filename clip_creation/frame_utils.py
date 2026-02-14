#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple


def sample_frames(input_path: Path, num_samples: int = 3) -> Tuple[int, int, List[bytes]]:
    """Sample frames from a video or return static image as repeated frames.

    Returns (width, height, [image_bytes...]).
    """
    try:
        import cv2
    except ImportError:
        print("X OpenCV not available for frame sampling")
        return 0, 0, []

    # Static image
    if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        frame = cv2.imread(str(input_path))
        if frame is None:
            return 0, 0, []
        h, w = frame.shape[:2]
        _, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()
        return w, h, [image_bytes] * num_samples

    # Video file
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"X Failed to open video: {input_path}")
        return 0, 0, []

    try:
        fps = max(1.0, cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or int(fps * 2))
        duration = max(1e-3, frame_count / fps)

        # Evenly spaced times, avoiding first/last 0.5s when possible
        if num_samples <= 1:
            sample_times = [min(0.5, duration * 0.5)]
        else:
            start = min(0.5, duration * 0.1)
            end = max(duration - 0.5, duration * 0.9)
            # Guard: if duration is very short, collapse to [0.1d, 0.5d, 0.9d]
            if end <= start:
                start = duration * 0.1
                end = duration * 0.9
            step = (end - start) / (num_samples - 1)
            sample_times = [start + i * step for i in range(num_samples)]

        frames = []
        w, h = 0, 0
        for t in sample_times:
            frame_idx = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                if w == 0:
                    h, w = frame.shape[:2]
                _, buffer = cv2.imencode('.jpg', frame)
                frames.append(buffer.tobytes())
            else:
                print(f" Failed to read frame at {t:.1f}s")

        return w, h, frames
    finally:
        cap.release()


