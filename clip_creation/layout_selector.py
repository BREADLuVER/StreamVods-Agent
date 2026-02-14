from __future__ import annotations

import os
import hashlib
from typing import Tuple


def _parse_weights(env_value: str, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    try:
        parts = [float(x) for x in (env_value or '').split(':')]
        if len(parts) != 3:
            return default
        s = sum(parts)
        if s <= 0:
            return default
        return (parts[0] / s, parts[1] / s, parts[2] / s)
    except Exception:
        return default


def _det_seed(vod_id: str | None, start_time: float | None, end_time: float | None) -> int:
    key = f"{vod_id or ''}|{int(start_time or 0)}|{int(end_time or 0)}"
    h = hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]
    return int(h, 16)


def choose_layout(
    *,
    cam_present: bool,
    cam_width: int | None,
    frame_width: int | None = None,
    vod_id: str | None,
    start_time: float | None,
    end_time: float | None,
    forced_choice: str | None = None,
) -> str:
    """Deterministic, weighted layout selection.

    Returns one of: 'A','B','C','JC'.
    - forced_choice overrides everything when provided.
    - If no cam_present → 'JC'.
    - Small cam (<450px): 50/50 between A and C.
    - Big cam (≥450px): weighted among B/A/C (env-configurable, default 0.8/0.1/0.1).
    """
    # Overrides
    if forced_choice:
        return forced_choice.upper().strip()

    if not cam_present:
        return 'JC'

    width = int(cam_width or 0)
    fw = int(frame_width or 0)
    # Normalized threshold for big cam (default 0.20 of frame width); fallback to absolute 450 if frame unknown
    thr_env = os.getenv('LAYOUT_BIGCAM_WIDTH_FRAC', '')
    try:
        thr_frac = float(thr_env) if thr_env else 0.20
    except Exception:
        thr_frac = 0.20
    is_big = False
    if fw > 0:
        is_big = (width / float(fw)) >= thr_frac
    else:
        is_big = width >= 450

    if not is_big:
        # Deterministic choice: usually 50/50 A or C, but A is disabled.
        # So we force C for small cams.
        return 'C'

    # Big cam weights (B, A, C) — default to 50% combined A/C (25% each)
    # A is disabled, so we redistribute its weight to B and C.
    # New default: B=0.6, C=0.4 (approx)
    w_env = os.getenv('LAYOUT_WEIGHTS_BIGCAM', '')
    wB, wA, wC = _parse_weights(w_env, (0.6, 0.0, 0.4))
    import random as _random
    rng = _random.Random(_det_seed(vod_id, start_time, end_time))
    return rng.choices(['B', 'A', 'C'], weights=[wB, 0.0, wC], k=1)[0]


