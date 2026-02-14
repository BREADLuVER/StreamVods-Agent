from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import math

from .models import CropBox, LayoutDecision


@dataclass
class _Rect:
	x: int
	y: int
	w: int
	h: int
	area_frac: float
	aspect: float
	corner_dist_norm: float


class StructuralLayoutDetector:
	"""Lightweight document-style structural detector.

	Heuristics only (no ML): detects candidate rectangles and classifies
	into one of: full-vod, game+cam, gameplay. Returns a LayoutDecision
	with an optional cam and chat box.
	"""

	def __init__(self, enable_logs: bool = True) -> None:
		self.enable_logs = enable_logs
		# Lazy YOLO detector holder
		self._yolo = None

	def _ensure_yolo(self):
		"""Load YOLO webcam detector if available."""
		if self._yolo is not None:
			return self._yolo
		try:
			from ultralytics import YOLO  # type: ignore
		except Exception as e:
			self._log(f"YOLO unavailable: {e}")
			self._yolo = False
			return self._yolo
		# Resolve weights: prefer explicit weights/webcam_detector.pt else best.pt from runs
		from pathlib import Path as _Path
		candidates = [
			_Path("weights/webcam_detector.pt"),
			_Path("weights/webcam_detector.onnx"),
		]
		# auto pick latest best.pt
		best_pts = sorted(_Path("runs/detect").glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
		best_onnx = sorted(_Path("runs/detect").glob("*/weights/best.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
		if best_pts:
			candidates.append(best_pts[0])
		if best_onnx:
			candidates.append(best_onnx[0])
		for c in candidates:
			if c.exists():
				try:
					self._yolo = YOLO(str(c))
					self._log(f"Loaded YOLO weights: {c}")
					return self._yolo
				except Exception as e:
					self._log(f"Failed to load YOLO weights {c}: {e}")
		self._yolo = False
		return self._yolo

	def _detect_webcam_yolo(self, frame) -> Optional[CropBox]:
		mod = self._ensure_yolo()
		if not mod:
			return None
		try:
			res = mod(frame, verbose=False, conf=0.25)[0]
			names = getattr(mod, "names", {})
			best = None
			best_conf = 0.0
			for xyxy, cls, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
				label = names.get(int(cls), str(int(cls))) if isinstance(names, dict) else str(int(cls))
				if label != "webcam":
					continue
				if float(conf) > best_conf:
					x1, y1, x2, y2 = [int(v) for v in xyxy]
					best = CropBox(x1, y1, x2 - x1, y2 - y1)
					best_conf = float(conf)
				self._log(f"YOLO det: label={label} conf={float(conf):.3f} box=({int(xyxy[0])},{int(xyxy[1])},{int(xyxy[2])},{int(xyxy[3])})")
			if best is None:
				self._log("YOLO det: no 'webcam' class above conf threshold")
		except Exception as e:
			self._log(f"YOLO inference failed: {e}")
			return None
		return best

	def _log(self, msg: str) -> None:
		if self.enable_logs:
			print(f"[Struct] {msg}")

	def _read_frame(self, input_path: Path) -> Tuple[int, int, Optional[object]]:
		try:
			import cv2  # type: ignore
		except Exception as e:
			self._log(f"OpenCV unavailable: {e}")
			return 0, 0, None

		# Handle both video files and static images
		if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
			# Static image
			frame = cv2.imread(str(input_path))
			if frame is None:
				self._log("Failed to read image")
				return 0, 0, None
			h, w = frame.shape[:2]
			return w, h, frame
		else:
			# Video file
			cap = cv2.VideoCapture(str(input_path))
			if not cap.isOpened():
				self._log("Failed to open video")
				return 0, 0, None
			# seek ~0.5s
			fps = max(1.0, cap.get(cv2.CAP_PROP_FPS) or 30.0)
			cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 0.5))
			ret, frame = cap.read()
			cap.release()
			if not ret or frame is None:
				self._log("Failed to read frame")
				return 0, 0, None
			h, w = frame.shape[:2]
			return w, h, frame

	def _read_frames_sampled(self, input_path: Path, num_samples: int = 3) -> Tuple[int, int, List[object]]:
		"""Read up to num_samples frames spread across the clip.

		For images, returns the same image repeated.
		"""
		w, h, frame = self._read_frame(input_path)
		if frame is None or w == 0 or h == 0:
			return 0, 0, []
		try:
			import cv2  # type: ignore
		except Exception:
			return w, h, [frame]
		if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
			return w, h, [frame for _ in range(num_samples)]
		# Video path: sample by frame indices
		cap = cv2.VideoCapture(str(input_path))
		if not cap.isOpened():
			return w, h, [frame]
		fps = max(1.0, cap.get(cv2.CAP_PROP_FPS) or 30.0)
		count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or fps * 2)
		indices = []
		if count > 3:
			indices = [
				max(0, int(0.5 * fps)),
				max(0, count // 2),
				max(0, min(count - 1, count - int(0.5 * fps)))
			]
		else:
			indices = [0, 1, 2]
		frames: List[object] = []
		for idx in indices[:num_samples]:
			cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
			ret, f = cap.read()
			if ret and f is not None:
				frames.append(f)
		cap.release()
		if not frames:
			frames = [frame]
		return w, h, frames

	def _read_frames_at_times(self, input_path: Path, times_sec: List[float]) -> Tuple[int, int, List[object]]:
		"""Read frames at specific timestamps (seconds). For images, returns the same image.
		"""
		w, h, first = self._read_frame(input_path)
		if first is None or w == 0 or h == 0:
			return 0, 0, []
		try:
			import cv2  # type: ignore
		except Exception:
			return w, h, [first for _ in times_sec]
		if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
			return w, h, [first for _ in times_sec]
		cap = cv2.VideoCapture(str(input_path))
		if not cap.isOpened():
			return w, h, [first]
		fps = max(1.0, cap.get(cv2.CAP_PROP_FPS) or 30.0)
		frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or fps * 2)
		duration = frame_count / fps
		frames: List[object] = []
		for t in times_sec:
			seek_t = max(0.0, min(duration - 1.0 / fps, t))
			idx = int(round(seek_t * fps))
			cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
			ret, f = cap.read()
			if ret and f is not None:
				frames.append(f)
		cap.release()
		return w, h, frames if frames else [first]

	def _build_cam_only_prompt(self, w: int, h: int, id_to_rect: Dict[str, _Rect]) -> str:
		lines = [
			"You are given a full frame and several blurred region thumbnails (A,B,C,...) from that frame.",
			"Pick which region ID contains the streamer webcam (a real person's face in a rectangular overlay).",
			"IMPORTANT: A streamer cam shows a REAL PERSON'S FACE, not game characters or UI elements.",
			"Look for: human face, microphone, headphones, real person talking.",
			"AVOID: game characters, UI elements, menus, HUD overlays.",
			"Return strict JSON: {\"cam_id\": \"<ID or none>\"}.",
			f"Frame: {w}x{h}",
			"Regions:",
		]
		for rid, r in id_to_rect.items():
			lines.append(f"- {rid}: x={r.x}, y={r.y}, w={r.w}, h={r.h}")
		lines.append("Only output JSON with cam_id.")
		return "\n".join(lines)

	def _cam_from_frame_simple(self, frame) -> Optional[CropBox]:
		"""Improved cam detection with better filtering and validation."""
		# Detect rectangles and filter for reasonable cam candidates
		rects = self._detect_rects(frame)
		self._log(f"Detected {len(rects)} rectangles for cam detection")
		
		# Filter rectangles for reasonable cam characteristics
		cam_candidates = []
		for i, r in enumerate(rects):
			self._log(f"  Rect {i}: x={r.x}, y={r.y}, w={r.w}, h={r.h} (area: {r.area_frac:.3f})")
			
			# Filter criteria for cam candidates
			if (r.area_frac > 0.01 and  # Not too small
				r.area_frac < 0.8 and   # Not too large (not the whole frame)
				r.w >= 32 and r.h >= 32 and  # Minimum size
				r.aspect > 0.5 and r.aspect < 2.0):  # Reasonable aspect ratio
				cam_candidates.append(r)
		
		if not cam_candidates:
			self._log("No suitable cam candidates found")
			return None
		
		# Use LLM to identify cam from filtered candidates
		images, id_to_rect = self._prepare_llm_inputs(frame, cam_candidates)
		self._log(f"Prepared {len(images)} images for LLM cam detection")
		
		try:
			from src.ai_client import call_llm_vision
			prompt = self._build_cam_only_prompt(frame.shape[1], frame.shape[0], id_to_rect)
			resp = call_llm_vision(prompt, images, max_tokens=60, temperature=0.1, request_tag="cam_only")
			self._log(f"LLM cam response: {resp[:200]}...")
		except Exception as e:
			self._log(f"LLM cam detection failed: {e}")
			resp = "{}"
		
		# Parse LLM response
		import json as _json, re as _re
		cand = resp.strip()
		if cand.startswith("```"):
			cand = _re.sub(r"^```[a-zA-Z]*\n?", "", cand).strip()
			cand = _re.sub(r"```\s*$", "", cand).strip()
		if not cand.startswith("{") and "{" in cand:
			cand = cand[cand.find("{"):cand.rfind("}")+1]
		
		cam_id = None
		try:
			obj = _json.loads(cand)
			cam_id = str(obj.get("cam_id", "none")).strip().upper()
			self._log(f"Parsed cam_id: {cam_id}")
		except Exception:
			self._log("Failed to parse LLM response")
			cam_id = None
		
		if cam_id and cam_id != "NONE" and cam_id in id_to_rect:
			cam_box = self._to_crop(id_to_rect[cam_id])
			self._log(f"LLM selected cam box: x={cam_box.x}, y={cam_box.y}, w={cam_box.width}, h={cam_box.height}")
			return cam_box
		
		# Fallback: face detection
		self._log("LLM did not select valid cam, trying face detection fallback")
		try:
			import cv2
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			casc = cv2.CascadeClassifier(getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml")
			faces = casc.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(32, 32)) if not casc.empty() else []
			self._log(f"Face detection found {len(faces)} faces")
			
			if len(faces) > 0:
				# Find largest face
				fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
				# Create a reasonable cam box around the face
				# For 30:70 layout, target aspect ratio is 1080:576 = 1.875
				target_ar = 1080 / 576.0
				
				# Calculate ideal cam dimensions based on face size
				face_center_x = fx + fw // 2
				face_center_y = fy + fh // 2
				
				# Make cam box proportional to face size but with target aspect ratio
				cam_height = int(fh * 3.0)  # 3x face height
				cam_width = int(cam_height * target_ar)
				
				# Center the cam box around the face
				cam_x = max(0, face_center_x - cam_width // 2)
				cam_y = max(0, face_center_y - cam_height // 2)
				
				# Clamp to frame boundaries
				cam_x = min(cam_x, frame.shape[1] - cam_width)
				cam_y = min(cam_y, frame.shape[0] - cam_height)
				
				if cam_width > 0 and cam_height > 0:
					face_box = CropBox(cam_x, cam_y, cam_width, cam_height)
					self._log(f"Face fallback cam box: x={face_box.x}, y={face_box.y}, w={face_box.width}, h={face_box.height}")
					return face_box
		except Exception as e:
			self._log(f"Face detection fallback failed: {e}")
		
		self._log("No cam detected in this frame")
		return None

	def analyze_cam_consensus_simple(self, input_path: Path) -> LayoutDecision:
		"""Fixed clip type detection and processing.
		
		Step 1: Determine clip type (gameplay+cam, pure gameplay, pure cam)
		Step 2: Process based on type with appropriate fallbacks
		"""
		# Read frames at ~0.5s, middle, and end-0.5s
		w, h, frames = self._read_frames_at_times(input_path, [0.5, 5_000_000.0, 9_000_000.0])
		self._log(f"Sampled {len(frames)} frames for clip type analysis")
		if not frames:
			return LayoutDecision(layout="full-vod", crops={}, confidence=0.0, reason="no_frames")
		
		# Step 1: Determine clip type using LLM
		clip_type = self._determine_clip_type(frames[0], w, h)
		self._log(f"Detected clip type: {clip_type}")
		
		# Step 2: Process based on type
		if clip_type == "gameplay+cam":
			return self._process_gameplay_cam_clip(frames, w, h)
		elif clip_type == "pure_gameplay":
			return LayoutDecision(layout="gameplay", crops={}, confidence=0.85, reason="pure_gameplay_detected")
		elif clip_type == "pure_cam":
			return LayoutDecision(layout="full-vod", crops={}, confidence=0.85, reason="pure_cam_detected")
		else:
			# Fallback: assume gameplay
			return LayoutDecision(layout="gameplay", crops={}, confidence=0.6, reason="unknown_type_fallback")

	def _determine_clip_type(self, frame, w: int, h: int) -> str:
		"""Use LLM to determine if clip is gameplay+cam, pure gameplay, or pure cam."""
		try:
			from src.ai_client import call_llm_vision
		except Exception as e:
			self._log(f"LLM unavailable for clip type detection: {e}")
			return "unknown"
		
		# Prepare frame for LLM
		frame_bytes = self._encode_png_bytes(frame)
		if not frame_bytes:
			return "unknown"
		
		prompt = (
			"Analyze this livestream/VOD screenshot and determine the clip type. "
			"Return STRICT JSON ONLY with key 'clip_type' and one of these values:\n"
			"- 'gameplay+cam': Both gameplay and streamer webcam are visible\n"
			"- 'pure_gameplay': Only gameplay is visible, no streamer cam\n"
			"- 'pure_cam': Only streamer talking to camera, no gameplay\n"
			"Rules: If you see both game content AND a person's face/webcam → 'gameplay+cam'. "
			"If you see only game content → 'pure_gameplay'. "
			"If you see only a person talking to camera → 'pure_cam'."
		)
		
		try:
			response = call_llm_vision(prompt, [("frame", frame_bytes, "image/png")], request_tag="clip_type_detect")
			self._log(f"LLM clip type response: {response[:200]}...")
			
			# Parse JSON response
			import json
			candidate = response.strip()
			if candidate.startswith("```"):
				candidate = re.sub(r"^```[a-zA-Z]*\n?", "", candidate).strip()
				candidate = re.sub(r"```\s*$", "", candidate).strip()
			if not candidate.startswith("{"):
				l = candidate.find("{")
				r = candidate.rfind("}")
				if l != -1 and r != -1 and r > l:
					candidate = candidate[l:r+1]
			
			obj = json.loads(candidate)
			clip_type = obj.get("clip_type", "").strip().lower()
			
			if clip_type in ["gameplay+cam", "pure_gameplay", "pure_cam"]:
				return clip_type
			else:
				self._log(f"Invalid clip type from LLM: {clip_type}")
				return "unknown"
				
		except Exception as e:
			self._log(f"LLM clip type detection failed: {e}")
			return "unknown"

	def _process_gameplay_cam_clip(self, frames, w: int, h: int) -> LayoutDecision:
		"""Process a clip that contains both gameplay and cam."""
		cam_boxes: List[CropBox] = []
		
		for i, frame in enumerate(frames):
			self._log(f"Processing frame {i+1}/{len(frames)} for cam detection")
			cam_box = self._cam_from_frame_simple(frame)
			if cam_box is not None:
				# Clamp to bounds
				x = max(0, min(cam_box.x, w - 1))
				y = max(0, min(cam_box.y, h - 1))
				bw = min(cam_box.width, w - x)
				bh = min(cam_box.height, h - y)
				if bw >= 32 and bh >= 32:  # Minimum reasonable cam size
					cam_boxes.append(CropBox(x, y, bw, bh))
					self._log(f"Added cam box {len(cam_boxes)}: x={x}, y={y}, w={bw}, h={bh}")
			else:
				self._log(f"Frame {i+1}: no cam detected")
		
		if not cam_boxes:
			self._log("No cam boxes found across all frames - falling back to gameplay")
			return LayoutDecision(layout="gameplay", crops={}, confidence=0.7, reason="cam_none_3frames_fallback")
		
		# Use median box for consensus
		cam_final = self._median_box(cam_boxes)
		self._log(f"Final consensus cam box: x={cam_final.x}, y={cam_final.y}, w={cam_final.width}, h={cam_final.height}")
		
		# For 30:70 split, use fixed top height of 576 (30% of 1920)
		top_h = 576  # Fixed 30% height
		self._log(f"Using fixed top_h: {top_h} (30% of 1920)")
		
		return LayoutDecision(
			layout="cam-top-40", 
			crops={"cam": cam_final}, 
			params={"top_h": top_h}, 
			confidence=0.85, 
			reason="cam_consensus_3frames"
		)

	@staticmethod
	def _to_crop(rect: _Rect) -> CropBox:
		return CropBox(rect.x, rect.y, rect.w, rect.h)

	def _encode_png_bytes(self, bgr_img) -> Optional[bytes]:
		try:
			import cv2  # type: ignore
		except Exception:
			return None
		ok, buf = cv2.imencode('.png', bgr_img)
		if not ok:
			return None
		return bytes(buf)

	def _prepare_llm_inputs(self, frame, rects: List[_Rect]) -> Tuple[List[Tuple[str, bytes, str]], Dict[str, _Rect]]:
		"""Create blurred thumbnails for each candidate region and include a resized full frame.

		Returns (images, id_to_rect) suitable for vision LLM call.
		"""
		import cv2  # type: ignore
		h, w = frame.shape[:2]
		# Resize full frame to a manageable width while preserving AR
		max_w = 1280
		scale = min(1.0, max_w / float(max(w, 1)))
		frame_small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
		images: List[Tuple[str, bytes, str]] = []
		frame_bytes = self._encode_png_bytes(frame_small)
		if frame_bytes:
			images.append(("frame", frame_bytes, "image/png"))

		id_to_rect: Dict[str, _Rect] = {}
		alpha = 23  # blur kernel base
		for idx, r in enumerate(rects):
			# Clamp ROI to frame bounds
			x0 = max(0, r.x)
			y0 = max(0, r.y)
			x1 = min(frame.shape[1], r.x + r.w)
			y1 = min(frame.shape[0], r.y + r.h)
			if x1 <= x0 or y1 <= y0:
				continue
			roi = frame[y0:y1, x0:x1]
			if roi.size == 0:
				continue
			# Strong blur to avoid sensitive content leakage while keeping semantics
			roi_h, roi_w = roi.shape[:2]
			k = max(5, (min(roi_w, roi_h) // 15) | 1)
			blurred = cv2.GaussianBlur(roi, (k, k), sigmaX=0)
			# Downscale thumbnails for upload efficiency
			thumb_w = 512
			thumb_scale = min(1.0, thumb_w / float(max(roi_w, 1)))
			if thumb_scale < 1.0:
				new_w = max(1, int(round(roi_w * thumb_scale)))
				new_h = max(1, int(round(roi_h * thumb_scale)))
				thumb = cv2.resize(blurred, (new_w, new_h))
			else:
				thumb = blurred
			thumb_bytes = self._encode_png_bytes(thumb)
			if not thumb_bytes:
				continue
			region_id = chr(ord('A') + idx)
			images.append((f"region_{region_id}", thumb_bytes, "image/png"))
			id_to_rect[region_id] = r
		return images, id_to_rect

	def _build_label_prompt(self, w: int, h: int, id_to_rect: Dict[str, _Rect]) -> str:
		lines = [
			"You are an expert video layout analyst. You will receive a full frame and multiple blurred region thumbnails.",
			"Each region is identified by an ID like A, B, C with its pixel box coordinates from the original frame.",
			"Task: For each region, assign ONE semantic label from this set:",
			"- gameplay",
			"- streamer_cam (aka webcam, facecam)",
			"- chat",
			"- sponsor_banner",
			"- hud_or_overlay",
			"- menu_or_ui",
			"- other",
			"Also infer high-level scene tags: split_screen_gameplay (two distinct gameplay panes), podcast_or_talking_heads, or none.",
			"Return STRICT JSON only with keys 'regions' and 'scene_tags'. Example:",
			'{"regions":[{"id":"A","label":"gameplay"},{"id":"B","label":"streamer_cam"}],"scene_tags":["none"]}',
			"Original frame size: %dx%d" % (w, h),
			"Regions:",
		]
		for rid, r in id_to_rect.items():
			lines.append(f"- {rid}: x={r.x}, y={r.y}, w={r.w}, h={r.h}")
		lines.append("Only output JSON. Do not include explanations.")
		return "\n".join(lines)

	def _pick_cam_by_labeling(
		self,
		frame,
		rects: List[_Rect],
		preferred_corner: Optional[str],
	) -> Tuple[Optional[CropBox], bool]:
		"""Ask LLM to label regions and return (cam CropBox, found_flag).

		If multiple regions are labeled as streamer_cam, select by proximity to
		the preferred corner when provided; otherwise choose the largest reasonable.
		"""
		images, id_to_rect = self._prepare_llm_inputs(frame, rects)
		if not images:
			return None, False
		try:
			from src.ai_client import call_llm_vision  # type: ignore
		except Exception:
			return None, False
		prompt = self._build_label_prompt(frame.shape[1], frame.shape[0], id_to_rect)
		try:
			text = call_llm_vision(prompt, images, max_tokens=600, temperature=0.1, request_tag="region_labeling")
			self._log(f"LLM region JSON (truncated): {text[:600]}")
		except Exception:
			return None, False
		# Parse JSON
		import json
		candidate = text.strip()
		if candidate.startswith("```"):
			candidate = re.sub(r"^```[a-zA-Z]*\n?", "", candidate).strip()
			candidate = re.sub(r"```\s*$", "", candidate).strip()
		if not candidate.startswith("{"):
			l = candidate.find("{"); r = candidate.rfind("}")
			if l != -1 and r != -1 and r > l:
				candidate = candidate[l:r+1]
		try:
			obj = json.loads(candidate)
			regions = obj.get("regions", []) or []
			cam_ids = [str(it.get("id","")) for it in regions if str(it.get("label","")) in ["streamer_cam","webcam","facecam","camera"]]
			if cam_ids:
				# Build candidate list
				cands: List[_Rect] = []
				for cid in cam_ids:
					r = id_to_rect.get(cid.strip().upper())
					if r:
						cands.append(r)
				if not cands:
					return None, False
				# Prefer by proximity to preferred corner, else by area within reasonable bounds
				w = frame.shape[1]; h = frame.shape[0]
				corner_map = {
					"top-left": (0, 0),
					"top-right": (w, 0),
					"bottom-left": (0, h),
					"bottom-right": (w, h),
					"center": (w // 2, h // 2),
					"left": (0, h // 2),
					"middle-left": (0, h // 2),
					"mid-left": (0, h // 2),
					"left-middle": (0, h // 2),
					"right": (w, h // 2),
					"middle-right": (w, h // 2),
				}
				if preferred_corner:
					pc = preferred_corner.strip().lower()
					cx, cy = corner_map.get(pc, (w // 2, h // 2))
					best = None; bestd = 1e18
					for r in cands:
						rx = r.x + r.w / 2.0; ry = r.y + r.h / 2.0
						d = (rx - cx) ** 2 + (ry - cy) ** 2
						if d < bestd:
							bestd = d; best = r
					if best is not None:
						return self._to_crop(best), True
				# Fallback: choose largest reasonable area
				cands_sorted = sorted(cands, key=lambda r: r.area_frac, reverse=True)
				return self._to_crop(cands_sorted[0]), True
		except Exception:
			return None, False
		return None, False

	def analyze_llm(self, input_path: Path) -> LayoutDecision:
		"""LLM-first vision analysis.

		Steps:
		1) Ask LLM to classify the full frame (scene type only).
		2) If gameplay+cam, ask LLM where the cam is (corner). Pick nearest detected rectangle.
		3) If split-screen, select two largest panes by geometry for a split layout.
		"""
		w, h, frame = self._read_frame(input_path)
		if frame is None or w == 0 or h == 0:
			return LayoutDecision(layout="unknown", crops={}, confidence=0.0, reason="no_frame")

		# Step 1: Full frame scene classification
		try:
			from src.ai_client import call_llm_vision  # type: ignore
		except Exception as e:
			self._log(f"LLM client unavailable: {e}")
			return LayoutDecision(layout="unknown", crops={}, confidence=0.0, reason="llm_unavailable")

		# Resize full frame to <= 1280px width for upload efficiency
		try:
			import cv2  # type: ignore
		except Exception:
			cv2 = None  # type: ignore
		frame_small = frame
		if cv2 is not None:
			max_w = 1280
			scale = min(1.0, max_w / float(max(w, 1)))
			if scale < 1.0:
				frame_small = cv2.resize(frame, (int(w * scale), int(h * scale)))
		frame_bytes = self._encode_png_bytes(frame_small)
		if not frame_bytes:
			return LayoutDecision(layout="unknown", crops={}, confidence=0.0, reason="image_encode_failed")

		scene_prompt = (
			"You will receive a single screenshot from a livestream/VOD. "
			"Return STRICT JSON ONLY with keys: has_gameplay (bool), has_streamer_cam (bool), has_chat (bool), is_split_screen (bool), scene (string). "
			"The 'scene' MUST be logically consistent with booleans and one of: ['gameplay','gameplay+cam','full-vod','split','unknown']. "
			"Rules: if is_split_screen is true => scene='split'. Else if has_gameplay and has_streamer_cam => 'gameplay+cam'. Else if has_gameplay => 'gameplay'. Else => 'full-vod'."
		)
		try:
			scene_text = call_llm_vision(scene_prompt, [("frame", frame_bytes, "image/png")], request_tag="scene_classify")
		except Exception as e:
			self._log(f"LLM vision call failed: {e}")
			return LayoutDecision(layout="unknown", crops={}, confidence=0.0, reason="llm_failed")
		self._log(f"LLM scene JSON (truncated): {scene_text[:600]}")

		# Parse scene JSON
		import json
		candidate = scene_text.strip()
		if candidate.startswith("```"):
			candidate = re.sub(r"^```[a-zA-Z]*\n?", "", candidate).strip()
			candidate = re.sub(r"```\s*$", "", candidate).strip()
		if not candidate.startswith("{"):
			l = candidate.find("{")
			r = candidate.rfind("}")
			if l != -1 and r != -1 and r > l:
				candidate = candidate[l:r+1]
		try:
			obj = json.loads(candidate)
			has_gameplay = str(obj.get("has_gameplay", "")).strip().lower() in ["true", "1", "yes"]
			has_streamer_cam = str(obj.get("has_streamer_cam", "")).strip().lower() in ["true", "1", "yes"]
			is_split_screen = str(obj.get("is_split_screen", "")).strip().lower() in ["true", "1", "yes"]
			# Derive final scene from booleans, ignoring potentially inconsistent 'scene'
			if is_split_screen:
				scene = "split"
			elif has_gameplay and has_streamer_cam:
				scene = "gameplay+cam"
			elif has_gameplay:
				scene = "gameplay"
			else:
				scene = "full-vod"
		except Exception:
			scene = "unknown"

		# Pre-compute rectangles for possible later steps
		rects = self._detect_rects(frame)

		# Step 2: Decide by scene
		if scene == "gameplay+cam":
			# Ask LLM for cam corner
			corner_prompt = (
				"You will receive the same screenshot. "
				"Answer the single key 'cam_corner' with one of: top-left, top-right, bottom-left, bottom-right, center. "
				"Return STRICT JSON only."
			)
			try:
				corner_text = call_llm_vision(corner_prompt, [("frame", frame_bytes, "image/png")], request_tag="cam_corner")
				self._log(f"LLM cam-corner JSON (truncated): {corner_text[:300]}")
			except Exception:
				corner_text = "{\"cam_corner\": \"bottom-right\"}"
			candidate = corner_text.strip()
			if candidate.startswith("```"):
				candidate = re.sub(r"^```[a-zA-Z]*\n?", "", candidate).strip()
				candidate = re.sub(r"```\s*$", "", candidate).strip()
			if not candidate.startswith("{"):
				l = candidate.find("{")
				r = candidate.rfind("}")
				if l != -1 and r != -1 and r > l:
					candidate = candidate[l:r+1]
			try:
				cam_corner = json.loads(candidate).get("cam_corner", "").strip().lower()
			except Exception:
				cam_corner = ""

			# Resolve cam box strictly via LLM region labeling (no geometry fallbacks)
			best_crop: Optional[CropBox] = None
			llm_cam_found = False
			if rects:
				best_crop, llm_cam_found = self._pick_cam_by_labeling(frame, rects, cam_corner)
			# If LLM did not confirm a cam rectangle, do not force cam layout
			if not llm_cam_found or best_crop is None:
				return LayoutDecision(
					layout=("gameplay" if has_gameplay else "full-vod"),
					crops={},
					confidence=0.7,
					reason="llm_scene_gameplay+cam_no_rect",
				)

			if best_crop is not None:
				# Dynamic top height: proportion to cam aspect (cap between 30-50%)
				cam_ar = best_crop.width / float(max(best_crop.height, 1))
				top_h = int(min(0.5, max(0.3, 0.38 + (0.1 if cam_ar < 1.2 else -0.05))) * 1920)
				return LayoutDecision(
					layout="cam-top-40",
					crops={"cam": best_crop},
					params={"top_h": top_h},
					confidence=0.85,
					reason=f"llm_scene_gameplay+cam_{cam_corner or 'corner'}",
				)
			# If we reached here, we have a cam crop from LLM labeling

		if scene == "split":
			# Heuristic: pick two largest panes by area
			large = sorted(rects, key=lambda r: r.area_frac, reverse=True)[:2] if rects else []
			if len(large) == 2:
				ordered = sorted(large, key=lambda r: r.y)
				return LayoutDecision(
					layout="split-gameplay",
					crops={
						"gameplay_top": self._to_crop(ordered[0]),
						"gameplay_bottom": self._to_crop(ordered[1]),
					},
					confidence=0.8,
					reason="llm_scene_split",
				)
			return LayoutDecision(layout="full-vod", crops={}, confidence=0.7, reason="llm_scene_split_no_rects")

		if scene == "gameplay":
			return LayoutDecision(layout="gameplay", crops={}, confidence=0.75, reason="llm_scene_gameplay")

		if scene == "full-vod":
			return LayoutDecision(layout="full-vod", crops={}, confidence=0.75, reason="llm_scene_full_vod")

		# Unknown scene → default gameplay
		return LayoutDecision(layout="gameplay", crops={}, confidence=0.6, reason="llm_scene_unknown")

	# ---- NEW: Multi-frame consensus analyzer with verifiers ----

	def _face_stats(self, frame, crop: CropBox) -> Tuple[int, float]:
		"""Return (num_faces, face_coverage_ratio) inside crop."""
		try:
			import cv2  # type: ignore
		except Exception:
			return 0, 0.0
		x0 = max(0, crop.x); y0 = max(0, crop.y)
		x1 = min(frame.shape[1], crop.x + crop.width)
		y1 = min(frame.shape[0], crop.y + crop.height)
		if x1 <= x0 or y1 <= y0:
			return 0, 0.0
		roi = frame[y0:y1, x0:x1]
		if roi.size == 0:
			return 0, 0.0
		gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		casc = cv2.CascadeClassifier(getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml")
		faces = casc.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(24, 24)) if not casc.empty() else []
		if len(faces) == 0:
			return 0, 0.0
		a = 0
		for (fx, fy, fw, fh) in faces:
			a += int(fw) * int(fh)
		coverage = min(1.0, a / float(max(1, roi.shape[0] * roi.shape[1])))
		return len(faces), coverage

	def _edge_density(self, frame, crop: CropBox) -> float:
		"""Edge pixel fraction in crop (0..1)."""
		try:
			import cv2  # type: ignore
		except Exception:
			return 0.0
		x0 = max(0, crop.x); y0 = max(0, crop.y)
		x1 = min(frame.shape[1], crop.x + crop.width)
		y1 = min(frame.shape[0], crop.y + crop.height)
		if x1 <= x0 or y1 <= y0:
			return 0.0
		roi = frame[y0:y1, x0:x1]
		if roi.size == 0:
			return 0.0
		gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(gray, 60, 140)
		return float((edges > 0).sum()) / float(max(1, edges.size))

	def _motion_score(self, prev, curr, crop: CropBox) -> float:
		"""Absolute-difference motion score in crop (0..1)."""
		try:
			import cv2  # type: ignore
		except Exception:
			return 0.0
		if prev is None or curr is None:
			return 0.0
		x0 = max(0, crop.x); y0 = max(0, crop.y)
		x1 = min(curr.shape[1], crop.x + crop.width)
		y1 = min(curr.shape[0], crop.y + crop.height)
		if x1 <= x0 or y1 <= y0:
			return 0.0
		roi_prev = prev[y0:y1, x0:x1]
		roi_curr = curr[y0:y1, x0:x1]
		if roi_prev.shape != roi_curr.shape or roi_prev.size == 0:
			return 0.0
		g1 = cv2.cvtColor(roi_prev, cv2.COLOR_BGR2GRAY)
		g2 = cv2.cvtColor(roi_curr, cv2.COLOR_BGR2GRAY)
		diff = cv2.absdiff(g1, g2)
		return float((diff > 20).sum()) / float(max(1, diff.size))

	def _median_box(self, boxes: List[CropBox]) -> CropBox:
		xs = sorted([b.x for b in boxes]); ys = sorted([b.y for b in boxes])
		ws = sorted([b.width for b in boxes]); hs = sorted([b.height for b in boxes])
		m = len(boxes) // 2
		return CropBox(xs[m], ys[m], ws[m], hs[m])

	def analyze_llm_consensus(self, input_path: Path) -> LayoutDecision:
		"""Multi-frame LLM labeling + deterministic verifiers + consensus."""
		w, h, frames = self._read_frames_sampled(input_path, num_samples=3)
		if not frames:
			return LayoutDecision(layout="unknown", crops={}, confidence=0.0, reason="no_frames")

		accepted_cam_boxes: List[CropBox] = []
		accepted_game_boxes: List[CropBox] = []
		prev = None
		for f in frames:
			# Per-frame region detection and labeling
			rects = self._detect_rects(f)
			if not rects:
				prev = f
				continue
			# Ask cam_corner lightly to bias selection; if it fails, None
			cam_corner = None
			try:
				from src.ai_client import call_llm_vision  # type: ignore
				fb = self._encode_png_bytes(f)
				if fb:
					corner_prompt = (
						"You will receive a screenshot. Answer with JSON {\"cam_corner\": one of [top-left, top-right, bottom-left, bottom-right, center, mid-left, mid-right] }."
					)
					ct = call_llm_vision(corner_prompt, [("frame", fb, "image/png")], max_tokens=30, temperature=0.1, request_tag="cam_corner_hint")
					if '{' in ct:
						import json as _json
						try:
							cam_corner = _json.loads(ct[ct.find('{'):ct.rfind('}')+1]).get("cam_corner", None)
						except Exception:
							cam_corner = None
			except Exception:
				cam_corner = None

			cam_box, cam_found = self._pick_cam_by_labeling(f, rects, cam_corner)
			if cam_found and cam_box is not None:
				n_faces, cov = self._face_stats(f, cam_box)
				self._log(f"Cam box: faces={n_faces}, coverage={cov:.3f}")
				# More lenient face detection: accept if faces OR if LLM labeled it as streamer_cam
				if (n_faces >= 1 and 0.05 <= cov <= 0.90) or cam_found:
					accepted_cam_boxes.append(cam_box)
					self._log(f" Cam box accepted (faces={n_faces}, coverage={cov:.3f})")
				else:
					self._log(f"X Cam box rejected: faces={n_faces}, coverage={cov:.3f}")

			# Gameplay acceptance from labeled gameplay regions
			images, id_to_rect = self._prepare_llm_inputs(f, rects)
			labels_text = ""
			try:
				from src.ai_client import call_llm_vision  # type: ignore
				labels_text = call_llm_vision(self._build_label_prompt(w, h, id_to_rect), images, max_tokens=600, temperature=0.1, request_tag="region_labeling_consensus")
			except Exception:
				labels_text = "{}"
			import json as _json
			cand = labels_text.strip()
			if cand.startswith("```") or cand.startswith("```json"):
				import re as _re
				cand = _re.sub(r"^```[a-zA-Z]*\n?", "", cand).strip()
				cand = _re.sub(r"```\s*$", "", cand).strip()
			if not cand.startswith("{") and "{" in cand:
				cand = cand[cand.find("{"):cand.rfind("}")+1]
			try:
				obj = _json.loads(cand)
				regions = obj.get("regions", []) or []
			except Exception:
				regions = []
			game_ids = [str(it.get("id","")) for it in regions if str(it.get("label","")) == "gameplay"]
			for gid in game_ids:
				r = id_to_rect.get(gid.strip().upper())
				if not r:
					continue
				b = self._to_crop(r)
				# Verifiers: motion OR edge density (more lenient)
				motion = self._motion_score(prev, f, b) if prev is not None else 0.0
				edge = self._edge_density(f, b)
				self._log(f"Game box {gid}: motion={motion:.4f}, edge={edge:.4f}")
				if motion >= 0.005 or edge >= 0.015:  # More lenient thresholds
					accepted_game_boxes.append(b)
					self._log(f" Game box {gid} accepted")
				else:
					self._log(f"X Game box {gid} rejected: motion={motion:.4f}, edge={edge:.4f}")
			prev = f

		# Consensus decision
		self._log(f"Consensus: {len(accepted_cam_boxes)} cam boxes, {len(accepted_game_boxes)} game boxes")
		cam_box_final: Optional[CropBox] = None
		if len(accepted_cam_boxes) >= 2:
			cam_box_final = self._median_box(accepted_cam_boxes)
			self._log(f"Using median cam box from {len(accepted_cam_boxes)} samples")
		elif len(accepted_cam_boxes) == 1:
			cam_box_final = accepted_cam_boxes[0]
			self._log(f"Using single cam box")

		# Distinct gameplay boxes by IoU separation
		def _iou_box(a: CropBox, b: CropBox) -> float:
			x1 = max(a.x, b.x); y1 = max(a.y, b.y)
			x2 = min(a.x + a.width, b.x + b.width); y2 = min(a.y + a.height, b.y + b.height)
			inter = max(0, x2 - x1) * max(0, y2 - y1)
			ua = a.width * a.height + b.width * b.height - inter
			return inter / ua if ua > 0 else 0.0

		unique_games: List[CropBox] = []
		for gb in accepted_game_boxes:
			if all(_iou_box(gb, u) < 0.3 for u in unique_games):
				unique_games.append(gb)
			if len(unique_games) >= 3:
				break

		# Final layout decision
		if len(unique_games) >= 2:
			ordered = sorted(unique_games[:2], key=lambda b: b.y)
			return LayoutDecision(
				layout="split-gameplay",
				crops={"gameplay_top": ordered[0], "gameplay_bottom": ordered[1]},
				confidence=0.8,
				reason="consensus_split",
			)

		if cam_box_final is not None and len(unique_games) >= 1:
			# Clamp cam box to frame boundaries instead of rejecting
			clamped_cam = CropBox(
				x=max(0, min(cam_box_final.x, w - 1)),
				y=max(0, min(cam_box_final.y, h - 1)),
				width=min(cam_box_final.width, w - max(0, min(cam_box_final.x, w - 1))),
				height=min(cam_box_final.height, h - max(0, min(cam_box_final.y, h - 1)))
			)
			
			# Dynamic height based on cam position and size
			cam_center_y = clamped_cam.y + clamped_cam.height / 2
			cam_ratio = cam_center_y / float(h)  # 0 = top, 1 = bottom
			
			# If cam is in top half, give it more space (30-40%)
			# If cam is in bottom half, give it less space (20-30%)
			if cam_ratio < 0.5:
				top_h = int(min(0.4, max(0.25, 0.3 + (0.1 * (0.5 - cam_ratio)))) * 1920)
			else:
				top_h = int(min(0.35, max(0.15, 0.25 - (0.1 * (cam_ratio - 0.5)))) * 1920)

			# Ensure even height to satisfy yuv420p requirements and stacking math
			top_h = max(2, top_h - (top_h % 2))
			
			self._log(f" Cam box: x={clamped_cam.x}, y={clamped_cam.y}, w={clamped_cam.width}, h={clamped_cam.height} (ratio: {cam_ratio:.2f}, top_h: {top_h})")
			return LayoutDecision(
				layout="cam-top-40",
				crops={"cam": clamped_cam},
				params={"top_h": top_h},
				confidence=0.85,
				reason="consensus_gameplay+cam",
			)

		if len(unique_games) >= 1:
			return LayoutDecision(layout="gameplay", crops={}, confidence=0.75, reason="consensus_gameplay")

		if cam_box_final is not None:
			return LayoutDecision(layout="full-vod", crops={}, confidence=0.7, reason="consensus_talking_head")

		return LayoutDecision(layout="gameplay", crops={}, confidence=0.6, reason="consensus_default")

	def _detect_rects(self, frame) -> List[_Rect]:
		import cv2  # type: ignore
		h, w = frame.shape[:2]
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.bilateralFilter(gray, 7, 50, 50)
		edges = cv2.Canny(gray, 50, 120)
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
		dil = cv2.dilate(edges, kernel, iterations=1)
		er = cv2.erode(dil, kernel, iterations=1)
		contours, _ = cv2.findContours(er, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		rects: List[_Rect] = []
		for c in contours:
			if cv2.contourArea(c) < 0.002 * w * h:
				continue
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.02 * peri, True)
			x, y, rw, rh = cv2.boundingRect(approx)
			area_frac = (rw * rh) / float(w * h)
			aspect = rw / float(max(rh, 1))
			# min distance to any corner (normalized 0..1)
			dx = min(x, w - (x + rw))
			dy = min(y, h - (y + rh))
			corner_dist_norm = math.hypot(dx, dy) / math.hypot(w, h)
			rects.append(_Rect(x, y, rw, rh, area_frac, aspect, corner_dist_norm))
		return rects

	def analyze(self, input_path: Path) -> LayoutDecision:
		w, h, frame = self._read_frame(input_path)
		if frame is None or w == 0 or h == 0:
			return LayoutDecision(layout="unknown", crops={}, confidence=0.0, reason="no_frame")

		# Portrait source → full-vod
		if w < h:
			return LayoutDecision(layout="full-vod", crops={}, confidence=0.8, reason="portrait_input")

		rects = self._detect_rects(frame)
		if not rects:
			# default to gameplay when nothing found
			return LayoutDecision(layout="gameplay", crops={}, confidence=0.4, reason="no_rects")

		# Debug: log detected rectangles
		self._log(f"Detected {len(rects)} rectangles:")
		for i, r in enumerate(rects[:5]):  # Show first 5
			self._log(f"  {i}: {r.x},{r.y} {r.w}x{r.h} (area: {r.area_frac:.3f}, aspect: {r.aspect:.2f})")

		# Identify chat: tall, narrow, hugging right/left
		chat: Optional[_Rect] = None
		for r in rects:
			if r.w / float(r.h) < 0.35 and r.h / float(h) > 0.75 and (r.x < 0.05 * w or r.x + r.w > 0.95 * w):
				chat = r if chat is None or r.area_frac > chat.area_frac else chat

		# Identify cam: small rectangle near corner with AR ~ [1.2, 2.0]
		cam: Optional[_Rect] = None
		corner_margin = 0.25
		for r in rects:
			ar_ok = 1.15 <= r.aspect <= 2.2
			area_ok = 0.02 <= r.area_frac <= 0.35
			near_corner = (
				(r.x < corner_margin * w and r.y < corner_margin * h)
				or (r.x + r.w > (1 - corner_margin) * w and r.y < corner_margin * h)
				or (r.x < corner_margin * w and r.y + r.h > (1 - corner_margin) * h)
				or (r.x + r.w > (1 - corner_margin) * w and r.y + r.h > (1 - corner_margin) * h)
			)
			if ar_ok and area_ok and near_corner:
				cam = r if cam is None or r.area_frac > cam.area_frac else cam

		# Large panes (for podcast-like full-vod) - more lenient
		large_rects = [r for r in rects if r.area_frac >= 0.20]
		# Require similarity to avoid false positives on big HUD boxes
		def _similar(a: _Rect, b: _Rect) -> bool:
			height_sim = abs(a.h - b.h) / float(max(a.h, b.h, 1)) <= 0.12
			aspect_sim = abs(a.aspect - b.aspect) <= 0.35
			return height_sim and aspect_sim
		if len(large_rects) >= 2 and chat is None:
			lr_sorted = sorted(large_rects, key=lambda r: r.area_frac, reverse=True)[:3]
			if any(_similar(lr_sorted[i], lr_sorted[j]) for i in range(len(lr_sorted)) for j in range(i+1, len(lr_sorted))):
				self._log(f"Found ≥2 similar large rectangles (≥20% area) - likely podcast")
				return LayoutDecision(layout="full-vod", crops={}, confidence=0.75, reason="multi_large_panes")

		# Additional podcast detection: if no chat, no cam, and at least one large rectangle
		if chat is None and cam is None and any(r.area_frac >= 0.35 for r in rects):
			self._log("No chat/cam detected but large content area - likely podcast")
			return LayoutDecision(layout="full-vod", crops={}, confidence=0.6, reason="large_content_no_overlays")

		if cam is not None:
			return LayoutDecision(
				layout="game+cam",
				crops={"cam": self._to_crop(cam)},
				confidence=0.65,
				reason="cam_near_corner",
			)

		# Gameplay by default (chat presence implies we'll crop centre 9:16)
		return LayoutDecision(layout="gameplay", crops={}, confidence=0.6, reason="default_gameplay")


