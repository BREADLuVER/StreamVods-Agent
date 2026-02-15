#!/usr/bin/env python3
"""
Rate webcam crops using MediaPipe Face Mesh (for eye openness) and FER (for emotion).
Prioritizes images with open eyes and high happiness/surprise.

Scoring:
  base_score = (Happy + Surprise) - 0.25*Neutral
  
  Penalties:
  - If eyes are closed (EAR < threshold): score -= 10.0 (Hard reject)
  - If blurry (Laplacian var < threshold): score -= 2.0
  
  Bonuses:
  - Size bonus: +0.2 * (area / 720^2)

Outputs:
  - JSON: data/thumbnails/<vod_id>/cams_index.json
  - CSV:  data/thumbnails/<vod_id>/cams_index.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# -----------------------------------------------------------------------------
# Imports & Lazy Loading
# -----------------------------------------------------------------------------

def _try_import_fer():
    try:
        from fer import FER  # type: ignore
        return FER
    except ImportError:
        return None

def _try_import_mediapipe():
    try:
        import mediapipe as mp
        return mp
    except ImportError:
        return None

def _load_image(path: Path):
    """Robust Unicode-safe image loader using imdecode."""
    try:
        import cv2
        import numpy as np
        with open(path, 'rb') as f:
            data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

# -----------------------------------------------------------------------------
# Analysis Logic
# -----------------------------------------------------------------------------

def _euclidean_dist(p1, p2) -> float:
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def _calculate_ear(landmarks, width, height) -> Tuple[float, float]:
    """
    Calculate Eye Aspect Ratio (EAR) for left and right eyes.
    Returns (left_ear, right_ear).
    """
    # MediaPipe Face Mesh indices
    # Left Eye: 33 (left), 160 (top1), 158 (top2), 133 (right), 153 (bot2), 144 (bot1)
    # Right Eye: 362 (left), 385 (top1), 387 (top2), 263 (right), 373 (bot2), 380 (bot1)
    
    left_indices = [33, 160, 158, 133, 153, 144]
    right_indices = [362, 385, 387, 263, 373, 380]
    
    def get_point(idx):
        lm = landmarks[idx]
        return (lm.x * width, lm.y * height)

    def ear(indices):
        p1 = get_point(indices[0])
        p2 = get_point(indices[1])
        p3 = get_point(indices[2])
        p4 = get_point(indices[3])
        p5 = get_point(indices[4])
        p6 = get_point(indices[5])
        
        # Vertical distances
        v1 = _euclidean_dist(p2, p6)
        v2 = _euclidean_dist(p3, p5)
        
        # Horizontal distance
        h = _euclidean_dist(p1, p4)
        
        if h == 0: return 0.0
        return (v1 + v2) / (2.0 * h)

    return ear(left_indices), ear(right_indices)

def _analyze_face_quality(mp_face_mesh, img) -> Dict[str, Any]:
    """
    Run MediaPipe Face Mesh to get EAR and other quality metrics.
    """
    import cv2
    
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = mp_face_mesh.process(rgb)
    
    if not results.multi_face_landmarks:
        return {"has_face": False, "left_ear": 0.0, "right_ear": 0.0}
        
    # Take the first face
    landmarks = results.multi_face_landmarks[0].landmark
    left_ear, right_ear = _calculate_ear(landmarks, w, h)
    
    return {
        "has_face": True,
        "left_ear": left_ear,
        "right_ear": right_ear,
        "avg_ear": (left_ear + right_ear) / 2.0
    }

def _calculate_blur(img) -> float:
    """
    Calculate Laplacian variance. Lower = blurrier.
    """
    import cv2
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def _analyze_emotion(fer_detector, img) -> Dict[str, float]:
    """
    Returns emotions distribution keys: happy, surprise, neutral, etc.
    """
    try:
        import cv2
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = fer_detector.detect_emotions(rgb)
        if detections:
            # Pick highest confidence / largest box
            best = max(detections, key=lambda d: (d.get('emotions') or {}).get('happy', 0.0) + (d.get('emotions') or {}).get('surprise', 0.0))
            emos = best.get('emotions') or {}
            return {k: float(emos.get(k, 0.0)) for k in ['happy','surprise','neutral','angry','sad','disgust','fear']}
    except Exception:
        pass
    return {k: 0.0 for k in ['happy','surprise','neutral','angry','sad','disgust','fear']}

def _size_bonus(img) -> float:
    try:
        h, w = img.shape[:2]
        area = float(w * h)
        ref = float(720 * 720)
        return max(0.0, min(1.0, area / ref))
    except Exception:
        return 0.0

def _calculate_score(
    em: Dict[str, float], 
    quality: Dict[str, Any], 
    blur_score: float, 
    size_b: float
) -> Tuple[float, List[str]]:
    """
    Calculate final score and return reasons for penalties.
    """
    reasons = []
    
    happy = float(em.get('happy', 0.0))
    surprise = float(em.get('surprise', 0.0))
    neutral = float(em.get('neutral', 0.0))
    
    # Base score from emotion
    base = 1.0 * max(happy, surprise) - 0.25 * neutral
    
    # Quality penalties
    penalty = 0.0
    
    # 1. Eyes Closed (EAR threshold ~0.2 is common, let's be safe with 0.18)
    EAR_THRESHOLD = 0.18
    if quality["has_face"]:
        if quality["avg_ear"] < EAR_THRESHOLD:
            penalty += 10.0
            reasons.append("eyes_closed")
    else:
        # No face detected by MediaPipe (but maybe FER saw one?)
        # If MediaPipe fails, it's usually a bad crop or occlusion
        penalty += 0.5 
        reasons.append("no_mp_face")

    # 2. Blur (Threshold varies, < 100 is usually blurry for clear frames)
    # For webcams, it might be lower. Let's say < 50 is bad.
    BLUR_THRESHOLD = 50.0
    if blur_score < BLUR_THRESHOLD:
        penalty += 2.0
        reasons.append("blurry")

    final_score = base + (0.2 * size_b) - penalty
    return final_score, reasons

def _parse_rel_ms(name: str) -> Optional[int]:
    import re
    m = re.search(r"_(-?\d+)ms_", name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

# -----------------------------------------------------------------------------
# Main Processing
# -----------------------------------------------------------------------------

def _rate_directory(vod_id: str, cams_dir: Path, fer_detector, mp_face_mesh) -> List[Dict]:
    rows: List[Dict] = []
    
    # Pre-check files
    files = sorted(list(cams_dir.glob("*.jpg")))
    print(f"Rating {len(files)} images in {cams_dir}...")
    
    for i, p in enumerate(files):
        img = _load_image(p)
        if img is None:
            continue
            
        # 1. Emotion (FER)
        emos = _analyze_emotion(fer_detector, img)
        
        # 2. Quality (MediaPipe)
        quality = _analyze_face_quality(mp_face_mesh, img)
        
        # 3. Blur
        blur = _calculate_blur(img)
        
        # 4. Size
        size_b = _size_bonus(img)
        
        # 5. Final Score
        score, penalties = _calculate_score(emos, quality, blur, size_b)
        
        h, w = img.shape[:2]
        rows.append({
            "vod_id": vod_id,
            "path": str(p),
            "filename": p.name,
            "score": round(float(score), 6),
            "happy": round(float(emos.get('happy', 0.0)), 6),
            "surprise": round(float(emos.get('surprise', 0.0)), 6),
            "neutral": round(float(emos.get('neutral', 0.0)), 6),
            "ear": round(float(quality.get("avg_ear", 0.0)), 4),
            "blur": round(float(blur), 2),
            "penalties": ",".join(penalties),
            "size_bonus": round(float(size_b), 6),
            "width": int(w),
            "height": int(h),
            "rel_ms": _parse_rel_ms(p.name),
        })
        
        if i % 10 == 0:
            print(f"Processed {i}/{len(files)}...", end="\r")
            
    print(f"Processed {len(files)}/{len(files)} done.")
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows

def main() -> None:
    parser = argparse.ArgumentParser(description="Rate webcam crops using FER + MediaPipe")
    parser.add_argument("vod_id", help="VOD ID")
    parser.add_argument("--dir", dest="cams_dir", default=None, help="Directory of cam JPGs")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--copy", action="store_true", help="Copy top-K to data/thumbnails/<vod_id>/best")
    args = parser.parse_args()

    vod_id = str(args.vod_id)
    
    # Init FER
    FER_cls = _try_import_fer()
    if FER_cls is None:
        print("FER library not found. Install with: pip install fer")
        return
    fer_detector = FER_cls(mtcnn=False)
    
    # Init MediaPipe
    mp = _try_import_mediapipe()
    if mp is None:
        print("MediaPipe not found. Install with: pip install mediapipe")
        return
    
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    out_dir = Path(f"data/thumbnails/{vod_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    def process_dir(target_dir: Path, name_suffix: str = ""):
        if not target_dir.exists():
            return False
            
        rows = _rate_directory(vod_id, target_dir, fer_detector, mp_face_mesh)
        
        prefix = "jc_" if "jc" in name_suffix else ""
        json_path = out_dir / f"{prefix}cams_index.json"
        csv_path = out_dir / f"{prefix}cams_index.csv"
        
        json_path.write_text(json.dumps({"vod_id": vod_id, "items": rows}, indent=2), encoding="utf-8")
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            cols = ["rank","score","happy","surprise","neutral","ear","blur","penalties","size_bonus","width","height","rel_ms","filename"]
            writer.writerow(cols)
            for idx, r in enumerate(rows, 1):
                writer.writerow([idx, r["score"], r["happy"], r["surprise"], r["neutral"], 
                                 r["ear"], r["blur"], r["penalties"], r["size_bonus"], 
                                 r["width"], r["height"], r["rel_ms"], r["filename"]])
        
        print(f"✓ {name_suffix} cams: {len(rows)} rated → {json_path}")
        
        if args.copy and rows:
            best_dir = out_dir / "best"
            best_dir.mkdir(exist_ok=True)
            import shutil
            for i in range(min(args.top_k, len(rows))):
                item = rows[i]
                src = Path(item["path"])
                dst = best_dir / f"rank_{i+1:02d}_{item['filename']}"
                shutil.copy2(src, dst)
            print(f"  Copied top {args.top_k} to {best_dir}")
        return True

    # Logic
    if args.cams_dir:
        process_dir(Path(args.cams_dir), "custom")
    else:
        processed = 0
        if process_dir(Path(f"data/thumbnails/{vod_id}/cams"), "normal"):
            processed += 1
        if process_dir(Path(f"data/thumbnails/{vod_id}/jc_cams"), "jc"):
            processed += 1
            
        if processed == 0:
            print("No cam directories found.")

    mp_face_mesh.close()

if __name__ == "__main__":
    main()
