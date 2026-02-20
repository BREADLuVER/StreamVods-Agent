#!/usr/bin/env python3
"""
Create individual short-form clips from a VOD using a clips manifest.

Source:
- data/vector_stores/<vod_id>/clips_manifest.json (or explicit manifest path)

Each manifest clip becomes a vertical 9:16 video with its title.
"""

# --- universal log adapter -----------------------------------------------
# Now local package imports that rely on project_root being in sys.path
import boto3
# Removed cv2 import - no longer needed for YOLO-based classification
# Removed LLM vision API dependency - now using YOLO-based classification
import shutil
import os, logging, sys
from pathlib import Path
# Add project root to path BEFORE trying to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)
if os.getenv("JOB_RUN_ID"):
    try:
        from utils.log import get_job_logger
        job_type = os.getenv("JOB_TYPE") or "clip"
        _, logger = get_job_logger(job_type, vod_id=os.getenv("VOD_ID", "unknown"), run_id=os.getenv("RUN_ID"))
        
        # Replace print with logger.info for this module
        def _log_print(*args, **kwargs):
            file = kwargs.get('file', sys.stdout)
            if file == sys.stderr:
                logger.error(' '.join(str(arg) for arg in args))
            else:
                logger.info(' '.join(str(arg) for arg in args))
        
        # Override print in this module's global scope
        print = _log_print
        
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        pass
# -------------------------------------------------------------------------

import sys
print("🔧 DEBUG: Script starting - imports beginning...", file=sys.stderr)
print("🔧 DEBUG: Script starting - imports beginning...")  # Also to stdout

import json
import subprocess
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
# Shared chat utilities (rendering and JSON prefetch). These supersede legacy functions below.
from chat_overlay.renderer import render_chat_segment, ensure_chat_json  # type: ignore

# Ensure UTF-8 console output on Windows to avoid Unicode crashes
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
except Exception:
    pass

# Add project root to path for imports before any internal package imports
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
print(f"🔧 DEBUG: Added to path: {project_root}")


try:
    from src.config import config
    print("🔧 DEBUG: Successfully imported config")
except ImportError as e:
    print(f"💥 IMPORT ERROR: Failed to import config: {e}")
    print(f"🔧 DEBUG: Current working directory: {os.getcwd()}")
    print(f"🔧 DEBUG: Python path: {sys.path[:5]}")
    sys.exit(1)


def apply_clips_per_hour_limit(clips: List[Dict]) -> List[Dict]:
    """Apply maximum clips per hour limit to prevent excessive clip generation"""
    if not clips:
        return clips
        
    max_clips_per_hour = config.max_clips_per_hour
    if max_clips_per_hour <= 0:
        return clips
    
    print(f"📏 Applying clips per hour limit: {max_clips_per_hour} clips/hour")
    
    # Group clips by hour based on start_time
    hourly_groups = {}
    for clip in clips:
        start_time = clip.get('start_time', 0)
        hour = int(start_time // 3600)  # Convert seconds to hour
        if hour not in hourly_groups:
            hourly_groups[hour] = []
        hourly_groups[hour].append(clip)
    
    # Limit each hour to max_clips_per_hour, keeping highest scoring clips
    limited_clips = []
    for hour, hour_clips in hourly_groups.items():
        # Sort by score (highest first) and take top N per hour
        hour_clips.sort(key=lambda x: x.get('score', 0), reverse=True)
        selected_for_hour = hour_clips[:max_clips_per_hour]
        limited_clips.extend(selected_for_hour)
        
        if len(hour_clips) > max_clips_per_hour:
            print(f"  Hour {hour}: Limited from {len(hour_clips)} to {max_clips_per_hour} clips")
        else:
            print(f" Hour {hour}: {len(hour_clips)} clips (within limit)")
    
    # Sort final result by start_time for better organization
    limited_clips.sort(key=lambda x: x.get('start_time', 0))
    
    original_count = len(clips)
    limited_count = len(limited_clips)
    
    if original_count > limited_count:
        print(f" Clips per hour limit applied: {original_count} → {limited_count} clips")
    else:
        print(f" All clips within hourly limits: {limited_count} clips")
    
    return limited_clips


## Legacy data loaders removed: manifest is the single source of truth


def _resolve_clips_manifest_path(token: str) -> Optional[Path]:
    """Resolve a token to a clips manifest path.

    Accepts either an explicit path or a VOD ID and looks for
    data/vector_stores/<vod_id>/clips_manifest.json
    """
    try:
        p = Path(token)
        if p.exists():
            return p
    except Exception:
        pass
    base = Path("data") / "vector_stores" / str(token)
    cand = base / "clips_manifest.json"
    if cand.exists():
        return cand
    return None


def _load_clips_from_manifest(path: Path) -> Tuple[List[Dict], List[Dict]]:
    """Load clips and titles from a clips manifest.

    Returns (clip_titles, clips_data) where each list is aligned 1:1
    with the manifest ordering.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"X Failed to read clips manifest: {e}")
        return [], []

    clips_raw = data.get("clips") or []
    if not isinstance(clips_raw, list) or not clips_raw:
        return [], []

    clip_titles: List[Dict] = []
    clips_data: List[Dict] = []
    for c in clips_raw:
        try:
            start = float(c.get("start", 0.0))
            end = float(c.get("end", 0.0))
            dur = float(c.get("duration", max(0.0, end - start)))
            score = float(c.get("score", 0.0))
            title = str(c.get("title", "Clip")).strip() or "Clip"
            reasoning = str(c.get("rationale", ""))
            anchor_time = c.get("anchor_time")
            anchor_time = float(anchor_time) if anchor_time is not None else None
        except Exception:
            continue
        clip_titles.append({
            "title": title,
            "score": score,
            "reasoning": reasoning,
        })
        clips_data.append({
            "start_time": start,
            "end_time": end,
            "duration": dur,
            "score": score,
            "reasoning": reasoning,
            "type": "short_form_highlight",
            "anchor_time": anchor_time,
        })
    return clip_titles, clips_data


def classify_clip_type(input_path: Path) -> str:
    """Classify clip type using YOLO-based webcam detection."""
    try:
        print(f"🔍 Classifying clip using YOLO webcam detection: {input_path}")
        
        # Use the robust YOLO-based webcam detection from clip_creation
        from clip_creation.yolo_face_locator import analyze_with_yolo
        
        # Analyze the clip with YOLO
        decision = analyze_with_yolo(input_path, enable_logs=True)
        
        # Check if webcam is detected
        has_webcam = False
        if decision and hasattr(decision, 'crops') and 'cam' in decision.crops:
            cam_crop = decision.crops['cam']
            # Verify the webcam detection is valid (reasonable size)
            if cam_crop.width >= 10 and cam_crop.height >= 10:
                has_webcam = True
                print(f"🔎 YOLO detected webcam: w={cam_crop.width} h={cam_crop.height} at ({cam_crop.x},{cam_crop.y})")
            else:
                print(f"🔎 YOLO webcam too small: w={cam_crop.width} h={cam_crop.height}")
        else:
            print("🔎 YOLO found no webcam in clip")
        
        # Binary classification based on webcam detection
        if has_webcam:
            classification = "gameplay_cam"
            print(f"✅ Classification: {classification} (webcam detected)")
        else:
            classification = "other"
            print(f"✅ Classification: {classification} (no webcam)")
        
        return classification
            
    except Exception as e:
        print(f"❌ Clip classification failed: {e}")
        # Default to 'other' for safety when classification fails
        return "other"


def convert_to_shorts_format(input_path: Path, output_path: Path, vod_id: Optional[str] = None, start_time: Optional[float] = None, end_time: Optional[float] = None, anchor_time: Optional[float] = None) -> bool:
    """Convert video to vertical 9:16 format for YouTube Shorts using FFmpeg.

    Strategy:
    1. Classify clip type using YOLO-based webcam detection
    2. Skip expensive layout detection for clips without webcam
    3. Use layout detection only for gameplay_cam clips (webcam detected)
    4. Apply appropriate cropping/formatting based on type

    Prefers NVENC for fast encoding; falls back to libx264 if NVENC fails or is unavailable.
    Honor FORCE_CPU_ENCODING to skip NVENC attempts.
    """
    try:
        # Step 1: Classify clip type first (cost optimization)
        print(" Step 1: Classifying clip type...")
        clip_type = classify_clip_type(input_path)
        print(f"🔍 DEBUG: Detected clip type: '{clip_type}'")

        # Global override for layout, regardless of classification
        forced_layout = os.getenv('LAYOUT_FORCE', '').upper().strip()
        forced_choice = ''
        if forced_layout in {'B', 'C'}:
            clip_type = 'gameplay_cam'
            forced_choice = forced_layout
            print(f"⚙️ Layout override active: LAYOUT_FORCE={forced_layout}")
        elif forced_layout == 'JC':
            clip_type = 'just_chat'
            forced_choice = 'JC'
            print("⚙️ Layout override active: LAYOUT_FORCE=JC")

        # Chat is always present now; we will render an external chat.mp4 for gameplay+cam
        
        filter_graph = ""
        layout_system = os.getenv('CLIP_LAYOUT_SYSTEM', 'advanced').lower()
        try:
            print(f"🧭 Routing context: vod_id={vod_id} window=({start_time},{end_time}) layout_system={layout_system} forced_choice={forced_choice or '-'} env.LAYOUT_TEST={os.getenv('LAYOUT_TEST','').upper().strip()}")
        except Exception:
            pass
        
        # Step 2: Apply appropriate processing based on clip type
        if clip_type in ("other", "just_chat"):
            print("💬 JC clip detected - Layout JC with external chat")
            try:
                if vod_id is None or start_time is None or end_time is None:
                    raise Exception("vod_id/start_time/end_time required to render chat segment")

                # Compose with Layout JC (fixed chat size) and render chat at exact same size
                from clip_creation.ffmpeg_layouts import build_chat_top20_full_stream_bottom_external as layout_JC
                jc_w = 648
                jc_h = 300
                # Use standard font size for JC layout (20px)
                os.environ["CHAT_FONT_PX"] = "20"

                # Render chat segment for this time window (transient – always regenerate)
                chat_dir = output_path.parent / "chat_segments"
                chat_dir.mkdir(parents=True, exist_ok=True)
                chat_path = chat_dir / f"chat_{int(start_time)}_{int(end_time)}_{jc_w}x{jc_h}.webm"
                print(f"🗨️ JC chat render: {chat_path.name} size={jc_w}x{jc_h}")
                try:
                    if chat_path.exists():
                        chat_path.unlink()
                except Exception:
                    pass
                head_start_sec = 15
                if not render_chat_segment(
                    vod_id,
                    start_time,
                    end_time,
                    chat_path,
                    chat_w=jc_w,
                    chat_h=jc_h,
                    head_start_sec=head_start_sec,
                    message_hex="#BFBFBF",
                    bg_hex="#00000000",
                    alt_bg_hex="#00000000",
                ):
                    raise Exception("Failed to render chat segment")

                base_graph = layout_JC(top_h=384, chat_w=jc_w, chat_h=jc_h, bottom_margin=8)
                # Prefer color+mask if present → alphamerge inside main filtergraph
                chat_color = chat_dir / f"chat_{int(start_time)}_{int(end_time)}_{jc_w}x{jc_h}_raw.mp4"
                chat_mask = None
                for cand in [chat_color.with_name(chat_color.stem + "_mask" + chat_color.suffix), chat_color.with_name(chat_color.stem + ".mask" + chat_color.suffix)]:
                    if cand.exists():
                        chat_mask = cand
                        break
                if chat_color.exists() and chat_mask and chat_mask.exists():
                    # Trim off pre-roll so chat starts aligned with clip time, but pre-populated
                    prefix = f"[1:v]trim=start={head_start_sec},setpts=PTS-STARTPTS[c0];[2:v]trim=start={head_start_sec},setpts=PTS-STARTPTS[c1];[c0][c1]alphamerge[chat_a];"
                    filter_graph = prefix + base_graph.replace("[1:v]", "[chat_a]")
                    external_chat_inputs = [str(chat_color), str(chat_mask)]
                    print("🗨️ JC inputs: color+mask (alphamerge)")
                else:
                    filter_graph = base_graph
                    external_chat_inputs = [str(chat_path)]
                    print("🗨️ JC inputs: single webm (alpha pre-merged)")
            except Exception as e:
                print(f" Just Chat composition failed: {e}")
                # Fallback: JC layout without external chat (top 20% bar + stream bottom 80%)
                # Compose 1080x384 black bar + scaled stream 1080x1536
                filter_graph = (
                    "color=c=black:s=1080x384[bar];"
                    "[0:v]scale=1080:1536:force_original_aspect_ratio=decrease[stream];"
                    "[bar][stream]vstack=inputs=2[vout]"
                )
                print(" Using JC layout without chat (external chat unavailable)")
            # Thumbnail fallback snapshot when no webcam (optional)
            try:
                if clip_type in ("other", "just_chat") and os.getenv('THUMBNAIL_FALLBACK_SCREENSHOT', '').lower() in ('1','true','yes'):
                    try:
                        from thumbnail.cam_snapshots import snapshot_full_frame  # type: ignore
                        _ = snapshot_full_frame(
                            vod_id=str(vod_id) if vod_id is not None else "",
                            clip_path=input_path,
                            start_time=float(start_time) if start_time is not None else 0.0,
                            end_time=float(end_time) if end_time is not None else 0.0,
                            anchor_time=None,
                            name_hint=os.path.splitext(output_path.name)[0],
                        )
                    except Exception:
                        pass
            except Exception:
                pass
            
        elif clip_type in ["gameplay_cam", "gameplay"]:
            if clip_type == "gameplay":
                print("🎮 Gameplay clip detected - using Layout JC (no cam detected)")
                print(f"🔍 DEBUG: Taking gameplay path with Layout JC")
            else:
                print("🎮+📹 Gameplay+Cam clip detected - dynamic layouts with external chat")
                print(f"🔍 DEBUG: Taking gameplay_cam path with advanced layout system")
            # Handle gameplay vs gameplay_cam differently
            if clip_type == "gameplay":
                # No cam detected - use Layout JC (just chat + gameplay)
                try:
                    from clip_creation.ffmpeg_layouts import build_chat_top20_full_stream_bottom_external as layout_JC
                    
                    if vod_id is None or start_time is None or end_time is None:
                        raise Exception("vod_id/start_time/end_time required to render chat segment")

                    # Render chat segment for this time window
                    chat_dir = output_path.parent / "chat_segments"
                    chat_dir.mkdir(parents=True, exist_ok=True)
                    jc_w, jc_h = 648, 300
                    # Use standard font size for gameplay JC layout (20px)
                    os.environ["CHAT_FONT_PX"] = "20"
                    chat_path = chat_dir / f"chat_{int(start_time)}_{int(end_time)}_{jc_w}x{jc_h}.webm"
                    print(f"🗨️ Gameplay JC chat render: {chat_path.name} size={jc_w}x{jc_h}")
                    try:
                        if chat_path.exists():
                            chat_path.unlink()
                    except Exception:
                        pass
                    head_start_sec = 15
                    if not render_chat_segment(
                        vod_id,
                        start_time,
                        end_time,
                        chat_path,
                        chat_w=jc_w,
                        chat_h=jc_h,
                        head_start_sec=head_start_sec,
                        message_hex="#BFBFBF",
                        bg_hex="#00000000",
                        alt_bg_hex="#00000000",
                    ):
                        raise Exception("Failed to render chat segment")

                    # Use Layout JC for gameplay without cam
                    base_graph = layout_JC(top_h=384, chat_w=jc_w, chat_h=jc_h, bottom_margin=8)
                    # Prefer color+mask if present → alphamerge inside main filtergraph
                    chat_color = chat_dir / f"chat_{int(start_time)}_{int(end_time)}_{jc_w}x{jc_h}_raw.mp4"
                    chat_mask = None
                    for cand in [chat_color.with_name(chat_color.stem + "_mask" + chat_color.suffix), chat_color.with_name(chat_color.stem + ".mask" + chat_color.suffix)]:
                        if cand.exists():
                            chat_mask = cand
                            break
                    if chat_color.exists() and chat_mask and chat_mask.exists():
                        # Trim off pre-roll so chat starts aligned with clip time, but pre-populated
                        prefix = f"[1:v]trim=start={head_start_sec},setpts=PTS-STARTPTS[c0];[2:v]trim=start={head_start_sec},setpts=PTS-STARTPTS[c1];[c0][c1]alphamerge[chat_a];"
                        filter_graph = prefix + base_graph.replace("[1:v]", "[chat_a]")
                        external_chat_inputs = [str(chat_color), str(chat_mask)]
                        print("🗨️ Gameplay JC inputs: color+mask (alphamerge)")
                    else:
                        filter_graph = base_graph
                        external_chat_inputs = [str(chat_path)]
                        print("🗨️ Gameplay JC inputs: single webm (alpha pre-merged)")
                    print(f"✅ DEBUG: Layout JC applied for gameplay clip")
                    
                except Exception as e:
                    print(f" Layout JC composition failed: {e}")
                    # Final fallback: simple gameplay layout
                    filter_graph = (
                        "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease[stream];"
                        "[stream]pad=1080:1920:(ow-iw)/2:(oh-ih)/2[vout]"
                    )
                    print(" Using simple gameplay layout (no chat)")
            else:
                # gameplay_cam - detect webcam crop and compose with external chat video
                try:
                    from clip_creation.yolo_face_locator import analyze_with_yolo
                    from clip_creation.ffmpeg_layouts import build_cam_top30_chat_in_gameplay_external as layout_B, build_insta_padding_external as layout_C, build_chat_top20_full_stream_bottom_external as layout_JC
                    
                    if vod_id is None or start_time is None or end_time is None:
                        raise Exception("vod_id/start_time/end_time required to render chat segment")

                    # Webcam detection
                    print("🔎 YOLO: calling analyze_with_yolo(...) for cam crop")
                    decision = analyze_with_yolo(input_path, enable_logs=True)
                    try:
                        crop_keys = list(getattr(decision, 'crops', {}).keys())
                        print(f"🔎 YOLO: decision crops keys={crop_keys}")
                        if 'cam' in getattr(decision, 'crops', {}):
                            _c = decision.crops['cam']
                            print(f"🔎 YOLO: cam crop w={_c.width} h={_c.height} at ({_c.x},{_c.y})")
                    except Exception:
                        pass
                    if "cam" not in decision.crops:
                        # User policy: if YOLO fails to detect webcam, discard clip
                        print("🛑 Policy: YOLO found no webcam → discarding clip")
                        return False

                    cam_crop = decision.crops["cam"]
                    # Multi-snapshot around anchor (no scoring): env-tunable
                    try:
                        from thumbnail.cam_snapshots import snapshot_from_cam_box  # type: ignore
                        def _env_float(name: str, default: float) -> float:
                            try:
                                return float(os.getenv(name, str(default)))
                            except Exception:
                                return default
                        def _env_int(name: str, default: int) -> int:
                            try:
                                return int(os.getenv(name, str(default)))
                            except Exception:
                                return default
                        # Prefer explicit offsets list if provided
                        raw_offsets = os.getenv("SNAP_OFFSETS", "").strip()
                        offsets: list[float] = []
                        if raw_offsets:
                            try:
                                offsets = [float(x) for x in raw_offsets.split(",") if x.strip()]
                            except Exception:
                                offsets = []
                        if not offsets:
                            pre = _env_float("SNAP_PRE_S", 1.0)
                            post = _env_float("SNAP_POST_S", 2.0)
                            samples = max(1, _env_int("SNAP_SAMPLES", 6))
                            # Default specific set when samples==6
                            if samples == 6 and pre >= 1.0 and post >= 2.0:
                                offsets = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
                            else:
                                # Evenly spaced in [-pre, +post]
                                span = pre + post
                                step = span / max(1, samples - 1)
                                offsets = [(-pre + i * step) for i in range(samples)]
                        base_anchor = anchor_time if anchor_time is not None else (float(start_time) if start_time is not None else None)
                        for rel in offsets:
                            _ = snapshot_from_cam_box(
                                vod_id=str(vod_id) if vod_id is not None else "",
                                clip_path=input_path,
                                start_time=float(start_time) if start_time is not None else 0.0,
                                end_time=float(end_time) if end_time is not None else 0.0,
                                cam_box=cam_crop,
                                anchor_time=base_anchor,
                                rel_offset_s=float(rel),
                                name_hint=os.path.splitext(output_path.name)[0],
                            )
                    except Exception:
                        pass

                    # Choose layout A/B/C deterministically using selector module
                    env_choice = os.getenv('LAYOUT_TEST', '').upper().strip()
                    choice = forced_choice or (env_choice if env_choice in {'B','C','JC'} else '')
                    if not choice:
                        from clip_creation.layout_selector import choose_layout
                        choice = choose_layout(
                            cam_present=True,
                            cam_width=int(cam_crop.width),
                            frame_width=int(1080 if True else 0),
                            vod_id=str(vod_id) if vod_id is not None else None,
                            start_time=float(start_time) if start_time is not None else None,
                            end_time=float(end_time) if end_time is not None else None,
                        )
                    # Log ratio for visibility
                    try:
                        cam_ratio = float(cam_crop.width) / 1080.0
                    except Exception:
                        cam_ratio = 0.0
                    print(f"🔀 Layout chosen: {choice} (cam_w={cam_crop.width}, ratio={cam_ratio:.3f})")

                    # Determine chat render size based on layout choice (fixed)
                    jc_w = 648
                    jc_h = 300
                    chat_w = jc_w
                    chat_h = jc_h
                    if choice == 'B':
                        # Layout B: chat centered in top gameplay, same size as JC
                        # Use standard font size for Layout B (20px)
                        os.environ["CHAT_FONT_PX"] = "20"
                        base_graph = layout_B(cam_crop, top_h=672, chat_w=jc_w, chat_h=jc_h, margin=0)
                    elif choice == 'C':
                        # Layout C: increase top padding by 1.5x; cam sticks to bottom of top padding; chat JC size centered in gameplay
                        # Use standard font size for Layout C (20px)
                        os.environ["CHAT_FONT_PX"] = "20"
                        top_pad_val = int(round(250 * 1.5))
                        # Increase cam box size by ~100px in both dimensions
                        try:
                            cam_crop.width = int(cam_crop.width + 100)
                            cam_crop.height = int(cam_crop.height + 100)
                        except Exception:
                            pass
                        base_graph = layout_C(cam_crop, top_pad=top_pad_val, bottom_pad=90, chat_w=jc_w, chat_h=jc_h)
                    elif choice == 'JC':
                        # Just chat: chat top 20%, full stream bottom 80% with wider chat
                        # Use standard font size for Layout JC (20px)
                        os.environ["CHAT_FONT_PX"] = "20"
                        base_graph = layout_JC(top_h=384, chat_w=jc_w, chat_h=jc_h, bottom_margin=8)
                    else:
                        raise Exception(f"Unsupported layout choice: {choice}")

                    # Render chat segment to match the chosen layout (transient – always regenerate)
                    chat_dir = output_path.parent / "chat_segments"
                    chat_dir.mkdir(parents=True, exist_ok=True)
                    chat_path = chat_dir / f"chat_{int(start_time)}_{int(end_time)}_{chat_w}x{chat_h}.webm"
                    print(f"🗨️ Gameplay+Cam chat render: {chat_path.name} size={chat_w}x{chat_h} layout={choice}")
                    try:
                        if chat_path.exists():
                            chat_path.unlink()
                    except Exception:
                        pass
                    head_start_sec = 15
                    if not render_chat_segment(
                        vod_id,
                        start_time,
                        end_time,
                        chat_path,
                        chat_w=chat_w,
                        chat_h=chat_h,
                        head_start_sec=head_start_sec,
                        message_hex="#BFBFBF",
                        bg_hex="#00000000",
                        alt_bg_hex="#00000000",
                    ):
                        raise Exception("Failed to render chat segment")

                    # Prefer color+mask if present → alphamerge inside main filtergraph
                    chat_color = chat_dir / f"chat_{int(start_time)}_{int(end_time)}_{chat_w}x{chat_h}_raw.mp4"
                    chat_mask = None
                    for cand in [chat_color.with_name(chat_color.stem + "_mask" + chat_color.suffix), chat_color.with_name(chat_color.stem + ".mask" + chat_color.suffix)]:
                        if cand.exists():
                            chat_mask = cand
                            break
                    if chat_color.exists() and chat_mask and chat_mask.exists():
                        prefix = f"[1:v]trim=start={head_start_sec},setpts=PTS-STARTPTS[c0];[2:v]trim=start={head_start_sec},setpts=PTS-STARTPTS[c1];[c0][c1]alphamerge[chat_a];"
                        filter_graph = prefix + base_graph.replace("[1:v]", "[chat_a]")
                        external_chat_inputs = [str(chat_color), str(chat_mask)]
                        print("🗨️ Gameplay+Cam inputs: color+mask (alphamerge)")
                    else:
                        filter_graph = base_graph
                        external_chat_inputs = [str(chat_path)]
                        print("🗨️ Gameplay+Cam inputs: single webm (alpha pre-merged)")
                except Exception as _an_err:
                    print(f" Gameplay+Cam composition failed: {_an_err}")
                    raise
        else:
            print("❓ Unknown clip type - using Layout JC fallback")
            print(f"🔍 DEBUG: Taking fallback path for clip type: '{clip_type}'")
            # Fallback to JC layout for unknown types
            try:
                if vod_id is None or start_time is None or end_time is None:
                    raise Exception("vod_id/start_time/end_time required to render chat segment")

                # Render chat segment for this time window (transient – always regenerate)
                chat_dir = output_path.parent / "chat_segments"
                chat_dir.mkdir(parents=True, exist_ok=True)
                print("🗨️ Unknown→JC fallback: will render chat at standard 648x300")
                # Use standard font size for fallback JC layout (20px)
                os.environ["CHAT_FONT_PX"] = "20"
                chat_path = chat_dir / f"chat_{int(start_time)}_{int(end_time)}_{648}x{300}.webm"
                try:
                    if chat_path.exists():
                        chat_path.unlink()
                except Exception:
                    pass
                head_start_sec = 15
                if not render_chat_segment(
                    vod_id,
                    start_time,
                    end_time,
                    chat_path,
                    chat_w=648,
                    chat_h=300,
                    head_start_sec=head_start_sec,
                    message_hex="#BFBFBF",
                    bg_hex="#00000000",
                    alt_bg_hex="#00000000",
                ):
                    raise Exception("Failed to render chat segment")

                # Compose with Layout JC
                from clip_creation.ffmpeg_layouts import build_chat_top20_full_stream_bottom_external as layout_JC
                # Use standardized chat size for JC fallback
                jc_w = 648
                jc_h = 300
                base_graph = layout_JC(top_h=384, chat_w=jc_w, chat_h=jc_h, bottom_margin=8)
                # Prefer color+mask if present → alphamerge inside main filtergraph
                chat_color = chat_dir / f"chat_{int(start_time)}_{int(end_time)}_{jc_w}x{jc_h}_raw.mp4"
                chat_mask = None
                for cand in [chat_color.with_name(chat_color.stem + "_mask" + chat_color.suffix), chat_color.with_name(chat_color.stem + ".mask" + chat_color.suffix)]:
                    if cand.exists():
                        chat_mask = cand
                        break
                if chat_color.exists() and chat_mask and chat_mask.exists():
                    # Trim off pre-roll so chat starts aligned with clip time, but pre-populated
                    prefix = f"[1:v]trim=start={head_start_sec},setpts=PTS-STARTPTS[c0];[2:v]trim=start={head_start_sec},setpts=PTS-STARTPTS[c1];[c0][c1]alphamerge[chat_a];"
                    filter_graph = prefix + base_graph.replace("[1:v]", "[chat_a]")
                    external_chat_inputs = [str(chat_color), str(chat_mask)]
                else:
                    filter_graph = base_graph
                    external_chat_inputs = [str(chat_path)]
            except Exception as e:
                print(f" JC fallback composition failed: {e}")
                # Final fallback: JC layout without external chat (top bar + stream)
                filter_graph = (
                    "color=c=black:s=1080x384[bar];"
                    "[0:v]scale=1080:1536:force_original_aspect_ratio=decrease[stream];"
                    "[bar][stream]vstack=inputs=2[vout]"
                )
                print(" Using JC layout without chat (external chat unavailable)")

        # Debug: Show filter graph status
        if filter_graph:
            print(f"✅ DEBUG: Filter graph generated successfully (length: {len(filter_graph)})")
            try:
                print(f"🧩 External inputs: {len(external_chat_inputs) if 'external_chat_inputs' in locals() else 0}")
            except Exception:
                pass
        else:
            print(f"❌ DEBUG: No filter graph generated")

        # Step 3: Fallback to heuristics if no filter graph generated
        if not filter_graph:
            print(f"⚠️ WARNING: Advanced layout system failed to generate filter graph!")
            print(f"   Layout system: {layout_system}")
            print(f"   Clip type: {clip_type}")
            print(f"   Forced choice: {forced_choice}")
            print(f"   Falling back to legacy heuristic system")
            # Heuristic fallback retained for robustness (legacy)
            stream_category = os.getenv('STREAM_CATEGORY', '').lower()
            explicit_layout = os.getenv('CLIP_LAYOUT', '').lower()

            if explicit_layout in {"split", "streamer"}:
                layout = explicit_layout
            else:
                if any(term in stream_category for term in ["chat", "irl", "just"]):
                    layout = "streamer"
                else:
                    layout = "split"

            if layout == "split":
                filter_graph = (
                    "[0:v]crop=480:270:0:0,scale=1080:960:force_original_aspect_ratio=decrease[cam];"
                    "[cam]pad=1080:960:(ow-iw)/2:(oh-ih)/2[cam_p];"
                    "[0:v]scale=1080:960:force_original_aspect_ratio=decrease[game];"
                    "[game]pad=1080:960:(ow-iw)/2:(oh-ih)/2[game_p];"
                    "[cam_p][game_p]vstack=inputs=2[vout]"
                )
            elif layout == "streamer":
                filter_graph = (
                    "[0:v]crop=480:270:0:0,scale=1080:1920:force_original_aspect_ratio=decrease[tmp];"
                    "[tmp]pad=1080:1920:(ow-iw)/2:(oh-ih)/2[vout]"
                )

        force_cpu = os.getenv('FORCE_CPU_ENCODING', 'false').lower() in ['true', '1', 'yes']

        # Resolve ffmpeg path: prefer bundled on Windows
        ffmpeg_bin = "ffmpeg"
        try:
            if os.name == 'nt':
                candidate = Path(__file__).parent.parent / "executables" / "ffmpeg.exe"
                if candidate.exists():
                    ffmpeg_bin = str(candidate)
        except Exception:
            pass

        # Auto-detect NVENC availability once per run; disable if not present
        nvenc_available = False
        if not force_cpu:
            try:
                probe_cmd = [ffmpeg_bin, "-hide_banner", "-encoders"]
                probe = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=15)
                enc_list = (probe.stdout or "") + (probe.stderr or "")
                nvenc_available = ("h264_nvenc" in enc_list or "nvenc" in enc_list.lower())
            except Exception:
                nvenc_available = False
        if not nvenc_available:
            force_cpu = True

        # Attempt NVENC first unless forced CPU
        if not force_cpu:
            # Build NVENC command with conservative threading/queues to reduce memory
            ff_threads = os.getenv("FFMPEG_THREADS", "2")
            ff_fthreads = os.getenv("FFMPEG_FILTER_THREADS", "1")
            ff_cfthreads = os.getenv("FFMPEG_FILTER_COMPLEX_THREADS", "1")
            ff_tq = os.getenv("FFMPEG_THREAD_QUEUE_SIZE", "1024")
            # Add -ss 0.001 to force seek/reset timestamps at start
            nvenc_cmd = [ffmpeg_bin, "-y", "-threads", ff_threads, "-filter_threads", ff_fthreads, "-filter_complex_threads", ff_cfthreads, "-thread_queue_size", ff_tq, "-ss", "0.001", "-i", str(input_path)]
            if 'external_chat_inputs' in locals():
                for _eci in external_chat_inputs:
                    nvenc_cmd += ["-thread_queue_size", ff_tq, "-i", _eci]
            try:
                print(f"🎛️ NVENC: using {'NVENC' if not force_cpu else 'CPU'} enc, external_inputs={len(external_chat_inputs) if 'external_chat_inputs' in locals() else 0}")
            except Exception:
                pass
            nvenc_cmd += [
                "-filter_complex", filter_graph,
                "-shortest",
                "-map", "[vout]",
                "-map", "0:a?",
                "-max_muxing_queue_size", os.getenv("FFMPEG_MAX_MUXING_QUEUE_SIZE", "4096"),
                "-c:v", "h264_nvenc",
                "-preset", "p4",
                "-rc", "vbr",
                "-cq", "18",
                "-b:v", "5M",
                "-maxrate", "10M",
                "-bufsize", "10M",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "128k",
                "-af", "aresample=async=1:first_pts=0",
                "-fps_mode", "cfr",
                "-movflags", "+faststart",
                str(output_path)
            ]
            try:
                result_nvenc = subprocess.run(nvenc_cmd, capture_output=True, text=True, timeout=300)
            except FileNotFoundError:
                result_nvenc = subprocess.CompletedProcess(nvenc_cmd, returncode=127, stdout="", stderr="ffmpeg not found")
            if result_nvenc.returncode == 0:
                print(f" Converted to Shorts format (NVENC): {output_path}")
                return True
            print(f" NVENC convert failed, falling back to CPU. Return code: {result_nvenc.returncode}")
            if result_nvenc.stderr:
                err = result_nvenc.stderr
                tail = err[-2000:] if len(err) > 2000 else err
                print(f"🔍 NVENC STDERR (tail): {tail}")

        # CPU fallback
        # Resolve ffmpeg path: prefer bundled executables/ffmpeg.exe on Windows
        ff_threads = os.getenv("FFMPEG_THREADS", "2")
        ff_fthreads = os.getenv("FFMPEG_FILTER_THREADS", "1")
        ff_cfthreads = os.getenv("FFMPEG_FILTER_COMPLEX_THREADS", "1")
        ff_tq = os.getenv("FFMPEG_THREAD_QUEUE_SIZE", "1024")
        # Add -ss 0.001 to force seek/reset timestamps at start
        cpu_cmd = [ffmpeg_bin, "-y", "-threads", ff_threads, "-filter_threads", ff_fthreads, "-filter_complex_threads", ff_cfthreads, "-thread_queue_size", ff_tq, "-ss", "0.001", "-i", str(input_path)]
        if 'external_chat_inputs' in locals():
            for _eci in external_chat_inputs:
                cpu_cmd += ["-thread_queue_size", ff_tq, "-i", _eci]
        cpu_cmd += [
            "-filter_complex", filter_graph,
            "-shortest",
            "-map", "[vout]",
            "-map", "0:a?",
            "-max_muxing_queue_size", os.getenv("FFMPEG_MAX_MUXING_QUEUE_SIZE", "4096"),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-af", "aresample=async=1:first_pts=0",
            "-fps_mode", "cfr",
            "-movflags", "+faststart",
            str(output_path)
        ]
        try:
            # Robust timeout to avoid hanging runs during local tests
            result_cpu = subprocess.run(cpu_cmd, capture_output=True, text=True, timeout=600)
        except FileNotFoundError as e:
            print(f"X FFmpeg not found: {e}")
            return False
        if result_cpu.returncode == 0:
            print(f" Converted to Shorts format (CPU): {output_path}")
            return True
        err = (result_cpu.stderr or "") + (result_cpu.stdout or "")
        tail = err[-2000:] if len(err) > 2000 else err
        print(f"X FFmpeg conversion failed (CPU): {tail}")
        return False
            
    except Exception as e:
        print(f"X Error converting to Shorts format: {e}")
        return False

def _resolve_twitch_cli_executable() -> str:
    """Resolve TwitchDownloaderCLI executable path"""
    # Try environment variable first
    override = os.getenv("TWITCH_DOWNLOADER_PATH")
    if override and Path(override).exists():
        return override
    
    # Try common locations
    possible_paths = [
        os.getenv("TWITCH_DOWNLOADER_PATH", ""),
        str(Path(__file__).parent.parent / "executables" / "TwitchDownloaderCLI.exe"),
        "C:/Users/bread/Documents/StreamSniped/TwitchDownloaderCLI.exe",
        "TwitchDownloaderCLI.exe",
        "./TwitchDownloaderCLI.exe",
        "TwitchDownloaderCLI",
        "./TwitchDownloaderCLI",
        "twitch-downloader",
        "./twitch-downloader"
    ]
    
    for path in possible_paths:
        try:
            if not path:
                continue
            result = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10)
            if "TwitchDownloaderCLI" in result.stdout or "TwitchDownloaderCLI" in result.stderr:
                print(f" Found TwitchDownloaderCLI: {path}")
                return path
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    # Try with shell=True for Windows
    try:
        result = subprocess.run("TwitchDownloaderCLI --version", shell=True, capture_output=True, text=True, timeout=10)
        if "TwitchDownloaderCLI" in result.stdout or "TwitchDownloaderCLI" in result.stderr:
            print(" Found TwitchDownloaderCLI via shell")
            return "TwitchDownloaderCLI"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    raise FileNotFoundError("TwitchDownloaderCLI not found. Please install it first.")


def download_single_clip(vod_id: str, start_time: float, end_time: float, output_path: Path, quality: str = "1080p") -> bool:
    """Download a single video clip using TwitchDownloaderCLI"""
    try:
        # Convert seconds to HH:MM:SS format
        start_timestamp = f"{int(start_time // 3600):02d}:{int((start_time % 3600) // 60):02d}:{int(start_time % 60):02d}"
        end_timestamp = f"{int(end_time // 3600):02d}:{int((end_time % 3600) // 60):02d}:{int(end_time % 60):02d}"
        
        # Create temp directory
        temp_dir = Path("/tmp/streamsniped_downloads") if os.name != 'nt' else Path("C:/temp/streamsniped_downloads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        env = os.environ.copy()
        env['TEMP'] = str(temp_dir)
        env['TMP'] = str(temp_dir)
        env['TMPDIR'] = str(temp_dir)
        
        # Raw VOD segments are heavy and can be cached; however, if caller
        # wants a fresh regeneration of the final video, we still reuse the
        # raw segment to save bandwidth. If FORCE_RE_DOWNLOAD is set, bypass cache.
        try:
            force_redownload = os.getenv('FORCE_RE_DOWNLOAD', '').lower() in ['1', 'true', 'yes']
            cache_dir = Path(f"data/cache/raw_clips/{vod_id}")
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_name = f"{int(start_time)}-{int(end_time)}.mp4"
            cache_path = cache_dir / cache_name
            if cache_path.exists() and cache_path.stat().st_size > 0 and not force_redownload:
                shutil.copy2(cache_path, output_path)
                print(f" Used cached raw clip: {cache_path.name}")
                return True
        except Exception:
            # Non-fatal
            pass

        # Configure parallel download threads
        try:
            dl_threads = int(os.getenv("CLIP_DL_THREADS", "8"))
        except Exception:
            dl_threads = 8

        # Resolve TwitchDownloaderCLI path
        twitch_cli = _resolve_twitch_cli_executable()
        
        cmd = [
            twitch_cli, "videodownload",
            "--id", vod_id,
            "-b", start_timestamp,
            "-e", end_timestamp,
            "-o", str(output_path),
            "-q", quality,
            "-t", str(dl_threads),
        ]
        
        print(f"🔄 Downloading clip: {start_timestamp} - {end_timestamp}")
        
        # Retry with exponential backoff up to 3 attempts
        attempts = 0
        last_err = ""
        while attempts < 3:
            attempts += 1
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
            if result.returncode == 0 and output_path.exists():
                file_size = output_path.stat().st_size / (1024 * 1024)
                print(f" Clip downloaded: {output_path.name} ({file_size:.1f}MB)")
                # Populate cache best-effort
                try:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(output_path, cache_path)
                except Exception:
                    pass
                return True
            last_err = (result.stderr or result.stdout or "").strip()[:200]
            if attempts < 3:
                wait = 2 ** attempts
                print(f" Download attempt {attempts} failed. Retrying in {wait}s...")
                import time as _time
                _time.sleep(wait)
        print(f"X Failed to download clip: {last_err}")
        return False
            
    except subprocess.TimeoutExpired:
        print(f"X Download timeout for clip {start_time}-{end_time}")
        return False
    except Exception as e:
        print(f"X Error downloading clip: {e}")
        return False


def upload_clip_to_s3(file_path: Path, vod_id: str, clip_title: str, bucket_name: str = "streamsniped-dev-videos") -> bool:
    """Upload individual clip to S3 with organized structure"""
    # Optional: skip uploads in local test mode
    if os.getenv("DISABLE_S3_UPLOADS", "").lower() in ("1", "true", "yes") or os.getenv("LOCAL_TEST_MODE", "").lower() in ("1", "true", "yes"):
        print(" Skipping S3 upload (DISABLE_S3_UPLOADS/LOCAL_TEST_MODE set)")
        return True
    try:
        s3 = boto3.client('s3')
        safe_title = "".join(c for c in clip_title if c.isalnum() or c in (' ', '_', '-')).rstrip()
        safe_title = safe_title.replace(' ', '_')
        s3_key = f"clips/{vod_id}/{file_path.name}"
        # Idempotent: skip if already exists in S3
        try:
            s3.head_object(Bucket=bucket_name, Key=s3_key)
            print(f" ⏭️  Clip already exists on S3, skipping: s3://{bucket_name}/{s3_key}")
            return True
        except Exception:
            pass
        s3.upload_file(str(file_path), bucket_name, s3_key)
        print(f" Uploaded to S3: s3://{bucket_name}/{s3_key}")
        return True
    except Exception as e:
        print(f"X Error uploading to S3: {e}")
        return False


def create_individual_clips(
    vod_id: str,
    quality: str = "1080p",
    max_clips: Optional[int] = None,
    manifest_token: Optional[str] = None,
) -> int:
    """Create individual video clips with their generated titles.
    
    For Shorts:
    - Downloads in original quality
    - Converts to vertical 9:16 format (1080x1920)
    - Adds blurred background for letterboxing
    """
    
    print(f" Creating individual clips for VOD: {vod_id}")
    print(f"🎥 Quality: {quality}")
    
    # Load clip titles and clip data (simplified - no more TikTok clips dependency)
    print(f"🔧 DEBUG: Loading data for VOD {vod_id}", file=sys.stderr)
    print(f"🔧 DEBUG: Loading data for VOD {vod_id}")
    
    # Determine manifest path (required)
    manifest_path: Optional[Path] = None
    if manifest_token:
        manifest_path = _resolve_clips_manifest_path(manifest_token)
    if manifest_path is None:
        manifest_path = _resolve_clips_manifest_path(vod_id)

    if manifest_path is None:
        print("X Clips manifest not found. Generate it first with vector_store/generate_clips_manifest.py")
        return 0

    # If the manifest includes a vod_id, prefer it over the positional argument
    try:
        man_obj = json.loads(manifest_path.read_text(encoding="utf-8"))
        man_vod = str(man_obj.get("vod_id") or "").strip()
        if man_vod:
            vod_id = man_vod
    except Exception:
        pass

    # Load from manifest
    print(f"🗂️ Using clips manifest: {manifest_path}")
    clip_titles, clips_data = _load_clips_from_manifest(manifest_path)
    print(f"🔧 DEBUG: Loaded {len(clips_data)} clips from manifest")
    
    # Check for clips exceeding 179s limit in manifest
    MAX_CLIP_DURATION = 179.0
    long_clips = []
    for i, clip in enumerate(clips_data):
        duration = clip.get('duration', clip.get('end_time', 0) - clip.get('start_time', 0))
        if duration > MAX_CLIP_DURATION:
            long_clips.append((i, duration))
    
    if long_clips:
        print(f"⚠️ Found {len(long_clips)} clips in manifest exceeding {MAX_CLIP_DURATION}s limit:")
        for i, duration in long_clips:
            print(f"   Clip {i+1}: {duration:.1f}s")
        print(f"   These will be trimmed to {MAX_CLIP_DURATION}s during processing")
    
    if not clip_titles:
        print("X No clip titles found")
        return 0
    
    if not clips_data:
        print(f"X No high-scoring clips found (score >= {config.classification_sections_score_threshold})")
        print("💡 Try lowering the score threshold via environment variables if needed")
        return 0
    
    print(f" Found {len(clip_titles)} titles and {len(clips_data)} clips from manifest")
    
    # Limit clips if specified
    if max_clips:
        clips_data = clips_data[:max_clips]
        # We'll adjust titles after synthesizing to ensure counts match
        print(f"📊 Limited to {max_clips} clips for testing")

    # Ensure 1:1 mapping (trim to shorter length)
    pair_count = min(len(clip_titles), len(clips_data))
    clip_titles = clip_titles[:pair_count]
    clips_data = clips_data[:pair_count]
    
    # Create clips directory
    clips_dir = Path(f"data/clips/{vod_id}")
    clips_dir.mkdir(parents=True, exist_ok=True)
    
    created_clips = 0
    produced_files: list[str] = []
    
    # Pipelined download + convert with chat prefetch
    dl_pool = ThreadPoolExecutor(max_workers=1)
    chat_pool = ThreadPoolExecutor(max_workers=int(os.getenv("CHAT_PREFETCH_WORKERS", "2")))
    pending: Optional[Tuple[Future, Dict]] = None
    try:
        for i, (title_data, clip_data) in enumerate(zip(clip_titles, clips_data), 1):
            if max_clips is not None and i > max_clips:
                break
            print(f"\n Creating clip {i}/{len(clip_titles)}")
            
            clip_title = title_data.get('title', f'Clip_{i}')
            start_time = clip_data.get('start_time', 0)
            end_time = clip_data.get('end_time', 0)
            anchor_time = clip_data.get('anchor_time')
            duration = clip_data.get('duration', end_time - start_time)
            score = clip_data.get('score', title_data.get('score', 0))
            
            # Enforce 179s limit for YouTube Shorts compatibility
            MAX_CLIP_DURATION = 179.0
            if duration > MAX_CLIP_DURATION:
                print(f"⚠️ Clip duration {duration:.1f}s exceeds {MAX_CLIP_DURATION}s limit - trimming")
                end_time = start_time + MAX_CLIP_DURATION
                duration = MAX_CLIP_DURATION
                print(f"✂️ Trimmed to: {start_time:.1f}s - {end_time:.1f}s ({duration:.1f}s)")
            
            print(f"📝 Title: {clip_title}")
            print(f"⏱️ Duration: {duration:.1f}s (score: {score:.1f})")
            
            safe_filename = "".join(c for c in clip_title if c.isalnum() or c in (' ', '_', '-')).rstrip()
            safe_filename = safe_filename.replace(' ', '_')
            clip_filename = f"{safe_filename}.mp4"
            clip_path = clips_dir / clip_filename
            try:
                if clip_path.exists():
                    clip_path.unlink()
            except Exception:
                pass
            temp_path = clip_path.parent / f"temp_{clip_path.name}"
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass

            # Start prefetch of chat JSON in background
            chat_dir = clips_dir / "chat_segments"
            _ = chat_pool.submit(ensure_chat_json, vod_id, start_time, end_time, chat_dir)

            if pending is None:
                fut = dl_pool.submit(download_single_clip, vod_id, start_time, end_time, temp_path, quality)
                pending = (fut, {"clip_path": clip_path, "temp_path": temp_path, "clip_title": clip_title, "start_time": start_time, "end_time": end_time, "anchor_time": anchor_time})
                continue

            # Process previous while downloading current
            prev_fut, ctx = pending
            fut = dl_pool.submit(download_single_clip, vod_id, start_time, end_time, temp_path, quality)
            ok = prev_fut.result()
            if ok:
                if convert_to_shorts_format(ctx["temp_path"], ctx["clip_path"], vod_id=vod_id, start_time=ctx["start_time"], end_time=ctx["end_time"], anchor_time=ctx.get("anchor_time")):
                    if upload_clip_to_s3(ctx["clip_path"], vod_id, ctx["clip_title"]):
                        # Only clean up local files if not in local mode
                        # Treat disable-s3 or local test mode as local workflow too
                        local_mode = (
                            os.getenv('GPU_PROCESSING_MODE', 'hybrid') == 'local_only' or
                            os.getenv('DISABLE_S3_UPLOADS', '').lower() in ('1', 'true', 'yes') or
                            os.getenv('LOCAL_TEST_MODE', '').lower() in ('1', 'true', 'yes')
                        )
                        if not local_mode:
                            # Remove local artifacts to save disk space
                            try:
                                ctx["temp_path"].unlink()
                            except Exception:
                                pass
                            try:
                                ctx["clip_path"].unlink()
                            except Exception:
                                pass
                        else:
                            print(f"📁 Keeping local clip: {ctx['clip_path'].name} (local mode)")
                    created_clips += 1
                    print(f" Clip {i-1} created successfully")
                    produced_files.append(ctx["clip_path"].name)
                else:
                    print(f"X Clip {i-1} conversion failed; skipping upload")
            else:
                print(f"X Failed to create clip {i-1}")
            pending = (fut, {"clip_path": clip_path, "temp_path": temp_path, "clip_title": clip_title, "start_time": start_time, "end_time": end_time, "anchor_time": anchor_time})

        # Drain last pending
        if pending is not None:
            prev_fut, ctx = pending
            ok = prev_fut.result()
            if ok:
                if convert_to_shorts_format(ctx["temp_path"], ctx["clip_path"], vod_id=vod_id, start_time=ctx["start_time"], end_time=ctx["end_time"], anchor_time=ctx.get("anchor_time")):
                    if upload_clip_to_s3(ctx["clip_path"], vod_id, ctx["clip_title"]):
                        # Only clean up local files if not in local mode
                        # Treat disable-s3 or local test mode as local workflow too
                        local_mode = (
                            os.getenv('GPU_PROCESSING_MODE', 'hybrid') == 'local_only' or
                            os.getenv('DISABLE_S3_UPLOADS', '').lower() in ('1', 'true', 'yes') or
                            os.getenv('LOCAL_TEST_MODE', '').lower() in ('1', 'true', 'yes')
                        )
                        if not local_mode:
                            # Remove local artifacts to save disk space
                            try:
                                ctx["temp_path"].unlink()
                            except Exception:
                                pass
                            try:
                                if not local_mode:
                                    ctx["clip_path"].unlink()
                            except Exception:
                                pass
                        else:
                            print(f"📁 Keeping local clip: {ctx['clip_path'].name} (local mode)")
                    created_clips += 1
                    print(f" Clip {len(produced_files)+1} created successfully")
                    produced_files.append(ctx["clip_path"].name)
                else:
                    print(f"X Final clip conversion failed; skipping upload")
            else:
                print(f"X Final clip download failed")
    finally:
        try:
            dl_pool.shutdown(wait=False, cancel_futures=False)
        except Exception:
            pass
        try:
            chat_pool.shutdown(wait=False, cancel_futures=False)
        except Exception:
            pass
    
    # Write manifest for cache robustness (and upload to S3 so cache works across machines)
    try:
        manifest = {
            "vod_id": vod_id,
            "count": created_clips,
            "files": produced_files,
            "source": "vector_store_manifest",
        }
        manifest_path_local = clips_dir / ".clips_manifest.json"
        with open(manifest_path_local, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2, ensure_ascii=False)
        print(f"📝 Clips manifest written: {manifest_path_local}")
        # Upload manifest and remove local copy if configured
        try:
            # Check for local processing mode flags
            disable_s3_uploads = os.getenv("DISABLE_S3_UPLOADS", "").lower() in ("1", "true", "yes")
            local_test_mode = os.getenv("LOCAL_TEST_MODE", "").lower() in ("1", "true", "yes")
            upload_videos = os.getenv("UPLOAD_VIDEOS", "false").lower() in ("1","true","yes")
            
            if upload_videos and not disable_s3_uploads and not local_test_mode:
                s3 = boto3.client('s3')
                bucket = os.getenv('S3_BUCKET', 'streamsniped-dev-videos')
                s3_key = f"clips/{vod_id}/.clips_manifest.json"
                s3.upload_file(str(manifest_path_local), bucket, s3_key)
                print(f"✅ Uploaded clips manifest to s3://{bucket}/{s3_key}")
                try:
                    manifest_path_local.unlink()
                except Exception:
                    pass
            elif disable_s3_uploads or local_test_mode:
                print("⏭️ Manifest S3 upload skipped (local processing mode)")
        except Exception as _e:
            print(f"⚠️ Manifest S3 upload skipped: {_e}")
    except Exception as e:
        print(f" Failed to write clips manifest: {e}")

    # Save the used titles mapping alongside clips (manifest-based)
    try:
        used_titles = []
        for i, (title_data, clip_data) in enumerate(zip(clip_titles, clips_data), 1):
            used_titles.append({
                'clip_index': i,
                'title': title_data.get('title', f'Clip_{i}'),
                'score': clip_data.get('score', title_data.get('score', 0)),
                'start_time': clip_data.get('start_time', 0),
                'end_time': clip_data.get('end_time', 0),
                'duration': clip_data.get('duration', clip_data.get('end_time', 0) - clip_data.get('start_time', 0)),
                'reasoning': clip_data.get('reasoning', title_data.get('reasoning', '')),
                'source': 'from_manifest'
            })
        meta_path = clips_dir / "used_titles.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({"vod_id": vod_id, "clip_titles": used_titles}, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved titles mapping: {meta_path}")
    except Exception as e:
        print(f"  Failed to save titles mapping: {e}")

    print(f"\n🎉 Individual clip creation completed!")
    print(f"📊 Created {created_clips}/{len(clip_titles)} clips")
    print(f"📁 Local directory: {clips_dir}")
    print(f"📁 S3 location: s3://streamsniped-dev-videos/clips/{vod_id}/")
    print(f"✨ Manifest-only workflow")
    
    return created_clips


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python create_individual_clips.py <vod_id|manifest_path> [quality] [--max-clips <count>] [--manifest <path|vod_id>]")
        sys.exit(1)
    
    print(f"🔧 DEBUG: Script called with args: {sys.argv}", file=sys.stderr)
    print(f"🔧 DEBUG: Script called with args: {sys.argv}")
    print(f"🔧 DEBUG: Python path: {sys.path[:3]}...", file=sys.stderr)
    print(f"🔧 DEBUG: Python path: {sys.path[:3]}...")  # Show first 3 path entries
    
    vod_id = sys.argv[1]
    quality = "1080p"
    max_clips = None
    manifest_token: Optional[str] = None
    
    # Parse arguments
    for i, arg in enumerate(sys.argv[2:], 2):
        if arg in ['360p', '480p', '720p', '1080p']:
            quality = arg
        elif arg == '--max-clips' and i + 1 < len(sys.argv):
            try:
                max_clips = int(sys.argv[i + 1])
            except ValueError:
                print("X Invalid max-clips value")
                sys.exit(1)
        elif arg == '--manifest' and i + 1 < len(sys.argv):
            manifest_token = sys.argv[i + 1]

    _created = create_individual_clips(
        vod_id,
        quality,
        max_clips,
        manifest_token=manifest_token,
    )
    
    if _created > 0:
        print(f"\n💡 Your {_created} clips are now ready for:")
        print(f"   • TikTok videos")
        print(f"   • YouTube Shorts") 
        print(f"   • Instagram Reels")
        print(f"   • Twitter/X clips")
        sys.exit(0)
    else:
        print(f"\nX No clips were created successfully", file=sys.stderr)
        sys.exit(1)


print("🔧 DEBUG: Script reached end, about to call main...")

if __name__ == "__main__":
    print("🔧 DEBUG: __name__ == __main__ - calling main()")
    main()
else:
    print(f"🔧 DEBUG: Script imported as module, __name__ = {__name__}")