from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class CropBox:
	x: int
	y: int
	width: int
	height: int


@dataclass
class LayoutDecision:
	"""Represents what to render in a 9:16 short and how.

	layout: one of: "full-vod", "gameplay", "game+cam", "unknown".
	crops: mapping of role -> CropBox. Typical roles: "cam", "gameplay".
	"""

	layout: str
	crops: Dict[str, CropBox]
	confidence: float = 0.0  # 0.0-1.0
	reason: Optional[str] = None
	params: Optional[Dict[str, int]] = None  # optional tunables, e.g., split heights

	def has_crops(self) -> bool:
		return bool(self.crops)


