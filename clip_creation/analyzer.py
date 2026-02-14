from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, List

from .models import CropBox, LayoutDecision
from .structural_detector import StructuralLayoutDetector


class ClipAnalyzer:
	"""Pluggable analyzer with two tiers:

	1) Fixed-box override via env vars (CAM_BOX/GAME_BOX)
	2) Fast face-detection on the first frame to infer layout

	We deliberately use only the first frame to minimise compute; most
	streams maintain a stable layout for the entire chapter.
	"""

	def __init__(self, enable_logs: bool = True) -> None:
		self.enable_logs = enable_logs

	def _log(self, message: str) -> None:
		if self.enable_logs:
			print(f"[Analyzer] {message}")

	@staticmethod
	def _parse_box(value: str) -> Optional[CropBox]:
		try:
			parts = [int(p.strip()) for p in value.split(",")]
			if len(parts) != 4:
				return None
			return CropBox(parts[0], parts[1], parts[2], parts[3])
		except Exception:
			return None

	@staticmethod
	def _expand_box(box: CropBox, frame_w: int, frame_h: int, margin: float) -> CropBox:
		mw = int(box.width * margin)
		mh = int(box.height * margin)
		x = max(0, box.x - mw)
		y = max(0, box.y - mh)
		w = min(frame_w - x, box.width + 2 * mw)
		h = min(frame_h - y, box.height + 2 * mh)
		return CropBox(x, y, w, h)

	@staticmethod
	def _area(box: CropBox) -> int:
		return max(0, box.width) * max(0, box.height)

	@staticmethod
	def _iou(a: CropBox, b: CropBox) -> float:
		x1 = max(a.x, b.x)
		y1 = max(a.y, b.y)
		x2 = min(a.x + a.width, b.x + b.width)
		y2 = min(a.y + a.height, b.y + b.height)
		inter = max(0, x2 - x1) * max(0, y2 - y1)
		union = ClipAnalyzer._area(a) + ClipAnalyzer._area(b) - inter
		return inter / union if union > 0 else 0.0

	@staticmethod
	def _union(a: CropBox, b: CropBox) -> CropBox:
		x1 = min(a.x, b.x)
		y1 = min(a.y, b.y)
		x2 = max(a.x + a.width, b.x + b.width)
		y2 = max(a.y + a.height, b.y + b.height)
		return CropBox(x1, y1, x2 - x1, y2 - y1)

	def _merge_faces(self, faces: List[CropBox]) -> CropBox:
		"""Merge overlapping/nearby faces into one cam cluster (greedy)."""
		if not faces:
			return CropBox(0, 0, 0, 0)
		clusters: List[CropBox] = []
		for f in sorted(faces, key=self._area, reverse=True):
			placed = False
			for i, c in enumerate(clusters):
				if self._iou(c, f) > 0.1:
					clusters[i] = self._union(c, f)
					placed = True
					break
			if not placed:
				clusters.append(f)
		# Return the largest cluster by area
		clusters.sort(key=self._area, reverse=True)
		return clusters[0]

	def _detect_faces_first_frame(self, input_path: Path) -> Tuple[int, int, list[CropBox]]:
		try:
			import cv2  # type: ignore
		except Exception as e:
			self._log(f"OpenCV not available: {e}")
			return 0, 0, []

		cap = cv2.VideoCapture(str(input_path))
		if not cap.isOpened():
			self._log("Failed to open video for face detection")
			return 0, 0, []
		# Seek to ~0.5s to avoid blank/transition first frame
		try:
			fps = max(1.0, cap.get(cv2.CAP_PROP_FPS) or 30.0)
			mid_frame = int(fps * 0.5)
			cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
		except Exception:
			pass
		ret, frame = cap.read()
		cap.release()
		if not ret or frame is None:
			self._log("Failed to read first frame")
			return 0, 0, []

		h, w = frame.shape[:2]
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Try multiple frontal cascades for robustness
		cascade_files = [
			"haarcascade_frontalface_default.xml",
			"haarcascade_frontalface_alt2.xml",
			"haarcascade_frontalface_alt.xml",
		]
		faces: list[CropBox] = []
		for fname in cascade_files:
			cpath = getattr(cv2.data, "haarcascades", "") + fname
			casc = cv2.CascadeClassifier(cpath)
			if casc.empty():
				continue
			# Adaptive minimum size (8% of shortest side, at least 24px)
			min_side = max(24, int(min(w, h) * 0.08))
			dets = casc.detectMultiScale(
				gray,
				scaleFactor=1.05,
				minNeighbors=4,
				minSize=(min_side, min_side),
			)
			if len(dets):
				faces = [CropBox(int(x), int(y), int(sw), int(sh)) for (x, y, sw, sh) in dets]
				break

		if not faces:
			# Fallback: try eye cascade to approximate a face region
			try:
				import cv2  # type: ignore
				eye_casc = cv2.CascadeClassifier(getattr(cv2.data, "haarcascades", "") + "haarcascade_eye.xml")
				if not eye_casc.empty():
					eds = eye_casc.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(16, 16))
					if len(eds) >= 2:
						# Build a rough box enclosing both eyes
						x1 = min([int(x) for (x, y, sw, sh) in eds])
						y1 = min([int(y) for (x, y, sw, sh) in eds])
						x2 = max([int(x + sw) for (x, y, sw, sh) in eds])
						y2 = max([int(y + sh) for (x, y, sw, sh) in eds])
						faces = [CropBox(x1, max(0, y1 - (y2 - y1)), max(1, x2 - x1), max(1, 2 * (y2 - y1)))]
			except Exception:
				pass
			if not faces:
				self._log("No faces detected with Haar cascades")
				return w, h, []

		self._log(f"Detected {len(faces)} face(s) on first frame [{w}x{h}]")
		return w, h, faces

	def analyze(self, input_path: Path) -> LayoutDecision:
		"""Return a layout decision for the given input.

		Order:
		- Respect fixed-box env overrides
		- LLM-first structural analysis (full-frame classify + cam mapping)
		- Fallback to document-style structural detector
		- Fallback to face-based heuristic
		- Else, return unknown to let caller fallback
		"""
		cam_box_env = os.getenv("CAM_BOX")
		game_box_env = os.getenv("GAME_BOX")

		cam_box = self._parse_box(cam_box_env) if cam_box_env else None
		game_box = self._parse_box(game_box_env) if game_box_env else None

		if cam_box and game_box:
			self._log("Using fixed boxes (game+cam)")
			return LayoutDecision(
				layout="game+cam",
				crops={"cam": cam_box, "gameplay": game_box},
				confidence=0.95,
				reason="fixed_boxes",
			)

		if cam_box and not game_box:
			self._log("Using fixed cam box (game+cam; gameplay=full)")
			return LayoutDecision(
				layout="game+cam",
				crops={"cam": cam_box, "gameplay": CropBox(0, 0, 0, 0)},
				confidence=0.9,
				reason="fixed_cam",
			)

		# Tier 2a: Molmo face detection (primary path)
		try:
			from .molmo_face_locator import analyze_with_molmo
			molmo_decision = analyze_with_molmo(input_path, enable_logs=True)
			if molmo_decision.layout in {"cam-top-40", "gameplay"}:
				self._log(f"Molmo layout: {molmo_decision.layout} ({molmo_decision.reason})")
				return molmo_decision
		except Exception as e:
			self._log(f"Molmo face detection failed: {e}")

		# Tier 2b: Legacy structural detector (fallback only)
		try:
			struct = StructuralLayoutDetector(enable_logs=True)
			struct_decision = struct.analyze(input_path)
			if struct_decision.layout in {"full-vod", "gameplay", "game+cam"}:
				self._log(f"Structural layout suggests: {struct_decision.layout}")
				if struct_decision.crops:
					return struct_decision
				if struct_decision.layout == "full-vod" and struct_decision.reason in {"multi_large_panes", "portrait_input"} and struct_decision.confidence >= 0.7:
					return struct_decision
		except Exception as e:
			self._log(f"Structural detector failed: {e}")

		# Tier 2c: face detection on first frame (legacy heuristic)
		frame_w, frame_h, faces = self._detect_faces_first_frame(input_path)
		# Portrait input → full-vod immediately
		if frame_w > 0 and frame_h > 0 and frame_w < frame_h:
			self._log("Portrait input detected → full-vod")
			return LayoutDecision(layout="full-vod", crops={}, confidence=0.8, reason="portrait_input")

		if frame_w > 0 and frame_h > 0 and faces:
			cluster = self._merge_faces(faces)
			margin = float(os.getenv("FACE_MARGIN", "0.20"))
			cluster = self._expand_box(cluster, frame_w, frame_h, margin)
			area_ratio = (cluster.width * cluster.height) / float(frame_w * frame_h)
			threshold = float(os.getenv("FACE_MAIN_RATIO", "0.40"))
			self._log(f"Face cluster area ratio: {area_ratio:.2f} (threshold {threshold:.2f})")
			corner_margin = float(os.getenv("CORNER_MARGIN", "0.25"))
			near_corner = (
				cluster.x < corner_margin * frame_w and cluster.y < corner_margin * frame_h
				or cluster.x + cluster.width > (1 - corner_margin) * frame_w and cluster.y < corner_margin * frame_h
				or cluster.x < corner_margin * frame_w and cluster.y + cluster.height > (1 - corner_margin) * frame_h
				or cluster.x + cluster.width > (1 - corner_margin) * frame_w and cluster.y + cluster.height > (1 - corner_margin) * frame_h
			)
			if area_ratio >= threshold and not near_corner:
				self._log("→ full-vod layout")
				return LayoutDecision(layout="full-vod", crops={}, confidence=0.7, reason="large_face")
			# Small face or corner overlay → game+cam
			self._log("→ game+cam layout")
			return LayoutDecision(layout="game+cam", crops={"cam": cluster}, confidence=0.7, reason="small_face_overlay")

		if frame_w > 0 and frame_h > 0 and not faces:
			self._log("No faces → gameplay layout")
			return LayoutDecision(layout="gameplay", crops={}, confidence=0.5, reason="no_faces")

		self._log("No faces detected; returning unknown")
		return LayoutDecision(layout="unknown", crops={}, confidence=0.0, reason="none")


