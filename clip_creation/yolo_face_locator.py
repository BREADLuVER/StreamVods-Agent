#!/usr/bin/env python3
"""
YOLO-based webcam locator for clip creation.

Decision policy (YOLO-only):
- Sample multiple frames across the clip
- Run YOLO webcam detector per frame
- Accept detections that meet confidence, aspect ratio, and area gates
- Require majority vote across frames and temporal consistency
- Produce a stable median cam crop when accepted, else fallback to gameplay
"""

import sys
from pathlib import Path
import os, json
import numpy as np
import cv2
from .models import CropBox, LayoutDecision
from .frame_utils import sample_frames
from .structural_detector import StructuralLayoutDetector
# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)


def _ensure_min_size_and_clamp(box: CropBox, frame_w: int, frame_h: int, min_w: int = 350, min_h: int = 300) -> CropBox:
    """Ensure the crop meets minimum size and stays within frame bounds."""
    width = max(min_w, box.width)
    height = max(min_h, box.height)

    x = max(0, min(box.x, frame_w - width))
    y = max(0, min(box.y, frame_h - height))

    width = min(width, frame_w - x)
    height = min(height, frame_h - y)

    return CropBox(x=x, y=y, width=int(width), height=int(height))


def create_static_cam_crop(face_center: np.ndarray, frame_w: int, frame_h: int) -> CropBox:
    """Create a static cam crop around the face center with minimum size enforcement."""
    # Start with a square that generally fits head+shoulders
    target_width = 540
    target_height = 540

    face_x, face_y = face_center

    crop_x = int(face_x - target_width * 0.5)
    # Bias the face slightly above vertical center inside the crop (more room for chin)
    crop_y = int(face_y - target_height * 0.45)

    # Clamp to frame and enforce minimums
    box = CropBox(crop_x, crop_y, target_width, target_height)
    return _ensure_min_size_and_clamp(box, frame_w, frame_h)


def analyze_with_yolo(input_path: Path, enable_logs: bool = True) -> LayoutDecision:
    """YOLO-only multi-frame majority vote for webcam detection."""
    
    def log(msg: str):
        if enable_logs:
            try:
                print(f"[YOLO] {msg}", flush=True)
            except Exception:
                # Best-effort logging
                try:
                    print("[YOLO] log failed to flush")
                except Exception:
                    pass
    
    # Configurable thresholds via env
    def _get_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except Exception:
            return default
    def _get_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except Exception:
            return default
    
    samples = _get_int("YOLO_SAMPLES", 5)
    conf_main = _get_float("YOLO_CONF_MAIN", 0.65)
    conf_fallback = _get_float("YOLO_CONF_FALLBACK", 0.45)
    ar_min = _get_float("YOLO_AR_MIN", 1.10)
    ar_max = _get_float("YOLO_AR_MAX", 2.10)
    area_min = _get_float("YOLO_AREA_MIN", 0.01)
    area_max = _get_float("YOLO_AREA_MAX", 0.35)
    majority_k = _get_int("YOLO_MAJORITY_K", 3)
    iou_median_min = _get_float("YOLO_TEMP_IOU_MEDIAN_MIN", 0.20)
    
    log(f"Analyzing {input_path.name} with YOLO webcam detection (N={samples})")
    
    # Step 1: Sample frames
    frame_w, frame_h, frames = sample_frames(input_path, num_samples=max(3, samples))
    if not frames:
        log("X No frames sampled → returning gameplay (reason=no_frames)")
        return LayoutDecision(layout="gameplay", crops={}, confidence=0.0, reason="no_frames")
    
    log(f"📐 Frame dimensions: {frame_w}x{frame_h}")
    
    # Helper: IoU
    def _iou(a: CropBox, b: CropBox) -> float:
        x1 = max(a.x, b.x); y1 = max(a.y, b.y)
        x2 = min(a.x + a.width, b.x + b.width); y2 = min(a.y + a.height, b.y + b.height)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter <= 0:
            return 0.0
        ua = a.width * a.height + b.width * b.height - inter
        return inter / float(max(1, ua))
    
    def _run_pass(threshold: float, require_k: int):
        try:
            detector = StructuralLayoutDetector(enable_logs=True)
            mod = detector._ensure_yolo()
            if not mod:
                log("X YOLO model unavailable → returning gameplay")
                return [], []
        except Exception as e:
            log(f"X Failed to load YOLO model: {e}")
            return [], []
        accepted: list[CropBox] = []
        confs: list[float] = []
        ar_vals: list[float] = []
        area_vals: list[float] = []
        for idx, fb in enumerate(frames):
            try:
                img = cv2.imdecode(np.frombuffer(fb, np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                img = None
            if img is None:
                continue
            try:
                res = mod(img, verbose=False, conf=threshold)[0]
                names = getattr(mod, "names", {})
                best = None
                best_conf = 0.0
                # Iterate detections, keep best 'webcam'
                for xyxy, cls, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
                    label = names.get(int(cls), str(int(cls))) if isinstance(names, dict) else str(int(cls))
                    if label != "webcam":
                        continue
                    c = float(conf)
                    if c >= best_conf:
                        x1, y1, x2, y2 = [int(v) for v in xyxy]
                        best = CropBox(x1, y1, x2 - x1, y2 - y1)
                        best_conf = c
                if best is None:
                    continue
                # Gating by AR and area
                ar = best.width / float(max(1, best.height))
                area_frac = (best.width * best.height) / float(max(1, frame_w * frame_h))
                if not (ar_min <= ar <= ar_max and area_min <= area_frac <= area_max):
                    continue
                accepted.append(best)
                confs.append(best_conf)
                ar_vals.append(ar)
                area_vals.append(area_frac)
            except Exception:
                continue
        # Temporal consistency check
        ious = []
        for i in range(len(accepted)):
            for j in range(i + 1, len(accepted)):
                ious.append(_iou(accepted[i], accepted[j]))
        median_iou = float(np.median(ious)) if ious else 0.0
        metrics = {
            "frames_sampled": len(frames),
            "detections_accepted": len(accepted),
            "majority_required": require_k,
            "conf_stats": [min(confs) if confs else 0.0, float(np.mean(confs)) if confs else 0.0, max(confs) if confs else 0.0],
            "ar_stats": [min(ar_vals) if ar_vals else 0.0, float(np.mean(ar_vals)) if ar_vals else 0.0, max(ar_vals) if ar_vals else 0.0],
            "area_stats": [min(area_vals) if area_vals else 0.0, float(np.mean(area_vals)) if area_vals else 0.0, max(area_vals) if area_vals else 0.0],
            "median_pairwise_iou": median_iou,
            "thresholds": {
                "conf": threshold, "ar": [ar_min, ar_max], "area": [area_min, area_max], "iou_median_min": iou_median_min
            }
        }
        log(f"METRICS: {json.dumps(metrics)}")
        if len(accepted) >= require_k and median_iou >= iou_median_min:
            return accepted, confs
        return [], []
    
    # Pass 1: main threshold
    accepted, confs = _run_pass(conf_main, majority_k)
    if not accepted:
        # Pass 2: fallback threshold, slightly lower majority requirement when using fallback
        accepted, confs = _run_pass(conf_fallback, max(2, majority_k - 1))
    
    if not accepted:
        log("Result: gameplay (no stable webcam majority)")
        return LayoutDecision(layout="gameplay", crops={}, confidence=0.7, reason="yolo_majority_none")
    
    # Stable median box
    xs = sorted([b.x for b in accepted]); ys = sorted([b.y for b in accepted])
    ws = sorted([b.width for b in accepted]); hs = sorted([b.height for b in accepted])
    m = len(accepted) // 2
    median_box = CropBox(xs[m], ys[m], ws[m], hs[m])
    cam_crop = _ensure_min_size_and_clamp(median_box, frame_w, frame_h, min_w=1, min_h=1)
    log(f"✅ Majority accepted. Cam crop: ({cam_crop.x},{cam_crop.y}) {cam_crop.width}x{cam_crop.height}")
    
    top_h = 576
    return LayoutDecision(
        layout="cam-top-40",
        crops={"cam": cam_crop},
        params={"top_h": top_h},
        confidence=0.9,
        reason="yolo_majority_webcam",
    )
