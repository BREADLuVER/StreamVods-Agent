#!/usr/bin/env python3
"""
Rate webcam crops using a tiny FER emotion model to pick clickbait-friendly stills.

Scoring (simple):
  score = 1.0*max(happy, surprise) - 0.25*neutral + 0.2*size_bonus
Where size_bonus = min(1.0, (w*h)/(720*720)) to slightly reward larger crops.

Handles both normal cam crops and just chatting full frames separately.

Outputs:
  - JSON: data/thumbnails/<vod_id>/cams_index.json (normal crops)
  - JSON: data/thumbnails/<vod_id>/jc_cams_index.json (just chatting)
  - CSV:  data/thumbnails/<vod_id>/cams_index.csv
  - CSV:  data/thumbnails/<vod_id>/jc_cams_index.csv
  - Optional: copy top-K images to data/thumbnails/<vod_id>/best

Usage:
  python -m thumbnail.rate_cams <vod_id> [--dir <cams_dir>] [--top-k 12] [--copy]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _try_import_fer():
    try:
        from fer import FER  # type: ignore
        return FER
    except Exception:
        return None


def _load_image(path: Path):
    """Robust Unicode-safe image loader using imdecode."""
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        with open(path, 'rb') as f:
            data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def _analyze_emotion(fer_detector, img) -> Dict[str, float]:
    # Returns emotions distribution keys: happy, surprise, neutral, angry, sad, disgust, fear
    try:
        # fer expects RGB; OpenCV loads BGR
        import cv2  # type: ignore
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = fer_detector.detect_emotions(rgb)
        if detections:
            # pick the highest confidence detection (largest box heuristic)
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


def _score(em: Dict[str, float], size_b: float) -> float:
    happy = float(em.get('happy', 0.0))
    surprise = float(em.get('surprise', 0.0))
    neutral = float(em.get('neutral', 0.0))
    primary = max(happy, surprise)
    return 1.0 * primary - 0.25 * neutral + 0.2 * size_b


def _parse_rel_ms(name: str) -> Optional[int]:
    # Parse a substring like "_-500ms_" out of the filename if present
    import re
    m = re.search(r"_(-?\d+)ms_", name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _rate_directory(vod_id: str, cams_dir: Path, fer_detector, output_suffix: str = "") -> List[Dict]:
    """Rate all images in a directory and return sorted results."""
    rows: List[Dict] = []
    for p in sorted(cams_dir.glob("*.jpg")):
        img = _load_image(p)
        if img is None:
            continue
        emos = _analyze_emotion(fer_detector, img)
        size_b = _size_bonus(img)
        s = _score(emos, size_b)
        h, w = img.shape[:2]
        rows.append({
            "vod_id": vod_id,
            "path": str(p),
            "filename": p.name,
            "score": round(float(s), 6),
            "happy": round(float(emos.get('happy', 0.0)), 6),
            "surprise": round(float(emos.get('surprise', 0.0)), 6),
            "neutral": round(float(emos.get('neutral', 0.0)), 6),
            "size_bonus": round(float(size_b), 6),
            "width": int(w),
            "height": int(h),
            "rel_ms": _parse_rel_ms(p.name),
        })
    
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Rate webcam crops using tiny FER")
    parser.add_argument("vod_id", help="VOD ID")
    parser.add_argument("--dir", dest="cams_dir", default=None, help="Directory of cam JPGs (default: auto-detect both cams and jc_cams)")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--copy", action="store_true", help="Copy top-K to data/thumbnails/<vod_id>/best")
    args = parser.parse_args()

    vod_id = str(args.vod_id)
    
    FER_cls = _try_import_fer()
    if FER_cls is None:
        print("FER library not found. Install with: pip install fer")
        return
    fer_detector = FER_cls(mtcnn=False)

    out_dir = Path(f"data/thumbnails/{vod_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # If explicit dir provided, only rate that
    if args.cams_dir:
        cams_dir = Path(args.cams_dir)
        if not cams_dir.exists():
            print(f"No cams dir: {cams_dir}")
            return
        rows = _rate_directory(vod_id, cams_dir, fer_detector)
        json_path = out_dir / "cams_index.json"
        csv_path = out_dir / "cams_index.csv"
        json_path.write_text(json.dumps({"vod_id": vod_id, "items": rows}, indent=2), encoding="utf-8")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["rank","score","happy","surprise","neutral","size_bonus","width","height","rel_ms","filename"]) 
            for idx, r in enumerate(rows, 1):
                writer.writerow([idx, r["score"], r["happy"], r["surprise"], r["neutral"], r["size_bonus"], r["width"], r["height"], r["rel_ms"], r["filename"]])
        print(f"Wrote JSON: {json_path}")
        print(f"Wrote CSV:  {csv_path}")
    else:
        # Auto-detect: rate both cams and jc_cams if they exist
        processed = 0
        
        # Rate normal cams
        cams_dir = Path(f"data/thumbnails/{vod_id}/cams")
        if cams_dir.exists():
            rows = _rate_directory(vod_id, cams_dir, fer_detector)
            if rows:
                json_path = out_dir / "cams_index.json"
                csv_path = out_dir / "cams_index.csv"
                json_path.write_text(json.dumps({"vod_id": vod_id, "items": rows}, indent=2), encoding="utf-8")
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["rank","score","happy","surprise","neutral","size_bonus","width","height","rel_ms","filename"]) 
                    for idx, r in enumerate(rows, 1):
                        writer.writerow([idx, r["score"], r["happy"], r["surprise"], r["neutral"], r["size_bonus"], r["width"], r["height"], r["rel_ms"], r["filename"]])
                print(f"✓ Normal cams: {len(rows)} rated → {json_path}")
                processed += 1
        
        # Rate just chatting cams
        jc_cams_dir = Path(f"data/thumbnails/{vod_id}/jc_cams")
        if jc_cams_dir.exists():
            jc_rows = _rate_directory(vod_id, jc_cams_dir, fer_detector)
            if jc_rows:
                jc_json_path = out_dir / "jc_cams_index.json"
                jc_csv_path = out_dir / "jc_cams_index.csv"
                jc_json_path.write_text(json.dumps({"vod_id": vod_id, "items": jc_rows}, indent=2), encoding="utf-8")
                with open(jc_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["rank","score","happy","surprise","neutral","size_bonus","width","height","rel_ms","filename"]) 
                    for idx, r in enumerate(jc_rows, 1):
                        writer.writerow([idx, r["score"], r["happy"], r["surprise"], r["neutral"], r["size_bonus"], r["width"], r["height"], r["rel_ms"], r["filename"]])
                print(f"✓ Just chatting cams: {len(jc_rows)} rated → {jc_json_path}")
                processed += 1
        
        if processed == 0:
            print("No cam directories found. Run extract_arc_cam_crops first.")


if __name__ == "__main__":
    main()


