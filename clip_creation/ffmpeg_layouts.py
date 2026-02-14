from __future__ import annotations

from .models import CropBox, LayoutDecision


def _crop_expr(tag: str, box: CropBox) -> str:
	return f"[0:v]crop={box.width}:{box.height}:{box.x}:{box.y}{tag}"


def build_filter_graph_from_decision(decision: LayoutDecision) -> str:
	"""Translate a LayoutDecision into an FFmpeg filter graph.

	Assumptions:
	- We only use stream 0:v twice when needed
	- Caller maps [vout] and audio separately
	"""
	# New deterministic layouts
	if decision.layout == "full-vod":
		# Fit whole frame into 1080x1920 with padding (no squish)
		return (
			"[0:v]scale=1080:1920:force_original_aspect_ratio=decrease[sv];"
			"[sv]pad=1080:1920:(ow-iw)/2:(oh-ih)/2[vout]"
		)

	if decision.layout == "gameplay":
		# Centre-crop to 9:16 then scale
		return (
			"[0:v]crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale=1080:1920[vout]"
		)

	if decision.layout == "game+cam" and "cam" in decision.crops:
		cam = decision.crops["cam"]
		# Crop gameplay centre 9:16 → scale 1080x1920; crop cam; overlay cam at top-left.
		# We intentionally do NOT upscale cam; if you need shrinking, add a scale step.
		return (
			"[0:v]crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale=1080:1920[game];"
			f"{_crop_expr('[cam]', cam)};"
			"[game][cam]overlay=10:10[vout]"
		)

	# New: cam on top (40% height) and gameplay bottom (60%)
	if decision.layout == "cam-top-40" and "cam" in decision.crops:
		cam = decision.crops["cam"]
		# Static layout - cam gets 40% height, gameplay gets remaining space
		params = decision.params or {}
		top_h = int(params.get("top_h", 768))  # 40% of 1920
		remaining_h = 1920 - top_h
		
		# Gameplay crop should fill the remaining bottom area
		# Use center crop to 9:16 ratio, then scale to fill bottom section
		gameplay_ar = 1080 / float(remaining_h)

		return (
			# Gameplay: center-crop to match bottom aspect ratio, then scale to fill
			f"[0:v]crop=ih*{gameplay_ar:.3f}:ih:(iw-ih*{gameplay_ar:.3f})/2:0[game_c];"
			f"[game_c]scale=1080:{remaining_h}:flags=lanczos[game_s];"
			# Cam: crop → scale-to-fill top band (no borders): increase then center-crop to exact size
			f"[0:v]crop={cam.width}:{cam.height}:{cam.x}:{cam.y}[cam_c];"
			f"[cam_c]scale=1080:{top_h}:force_original_aspect_ratio=increase[cam_sf];"
			f"[cam_sf]crop=1080:{top_h}:(iw-1080)/2:(ih-{top_h})/2[cam_s];"
			# Stack cam on top of gameplay
			"[cam_s][game_s]vstack=inputs=2[vout]"
		)

	# New: cam + chat side-by-side in top 30%, gameplay bottom 70%
	if decision.layout == "cam-chat-top" and {"cam", "chat"}.issubset(decision.crops.keys()):
		cam = decision.crops["cam"]
		chat = decision.crops["chat"]
		params = decision.params or {}
		top_h = int(params.get("top_h", 576))
		remaining_h = 1920 - top_h

		# Target cells for top section (50/50 split): 540x576 each
		cell_w = 540
		cell_h = top_h

		# Gameplay aspect ratio for bottom section
		gameplay_ar = 1080 / float(remaining_h)

		return (
			# Gameplay bottom
			f"[0:v]crop=ih*{gameplay_ar:.3f}:ih:(iw-ih*{gameplay_ar:.3f})/2:0[game_c];"
			f"[game_c]scale=1080:{remaining_h}:flags=lanczos[game_s];"
			# Cam crop → scale-to-FILL left cell (no black borders), then center-crop
			f"[0:v]crop={cam.width}:{cam.height}:{cam.x}:{cam.y}[cam_c];"
			f"[cam_c]scale={cell_w}:{cell_h}:force_original_aspect_ratio=increase:flags=lanczos[cam_sf];"
			f"[cam_sf]crop={cell_w}:{cell_h}:(iw-{cell_w})/2:(ih-{cell_h})/2[cam_s];"
			# Chat crop → scale-to-FILL right cell, then center-crop
			f"[0:v]crop={chat.width}:{chat.height}:{chat.x}:{chat.y}[chat_c];"
			f"[chat_c]scale={cell_w}:{cell_h}:force_original_aspect_ratio=increase:flags=lanczos[chat_sf];"
			f"[chat_sf]crop={cell_w}:{cell_h}:(iw-{cell_w})/2:(ih-{cell_h})/2[chat_s];"
			# HStack cam(left) + chat(right) → top band, then VStack with gameplay
			"[chat_s][cam_s]hstack=inputs=2[top];"
			"[top][game_s]vstack=inputs=2[vout]"
		)

	# Legacy compatibility (if older tags are present)
	if decision.layout == "streamer" and "cam" in decision.crops:
		cam = decision.crops["cam"]
		return (
			f"{_crop_expr('[cam]', cam)};"
			"[cam]scale=1080:1920:force_original_aspect_ratio=decrease[cam_s];"
			"[cam_s]pad=1080:1920:(ow-iw)/2:(oh-ih)/2[vout]"
		)

	if decision.layout == "split" and {"cam", "gameplay"}.issubset(decision.crops.keys()):
		cam = decision.crops["cam"]
		game = decision.crops["gameplay"]
		return (
			f"{_crop_expr('[cam]', cam)};"
			f"{_crop_expr('[game]', game)};"
			"[cam]scale=1080:960:force_original_aspect_ratio=decrease[cam_s];"
			"[cam_s]pad=1080:960:(ow-iw)/2:(oh-ih)/2[cam_p];"
			"[game]scale=1080:960:force_original_aspect_ratio=decrease[game_s];"
			"[game_s]pad=1080:960:(ow-iw)/2:(oh-ih)/2[game_p];"
			"[cam_p][game_p]vstack=inputs=2[vout]"
		)

	# New: two-gameplay split-screen (stacked)
	if decision.layout == "split-gameplay" and {"gameplay_top", "gameplay_bottom"}.issubset(decision.crops.keys()):
		g1 = decision.crops["gameplay_top"]
		g2 = decision.crops["gameplay_bottom"]
		return (
			f"{_crop_expr('[g1]', g1)};"
			f"{_crop_expr('[g2]', g2)};"
			"[g1]scale=1080:960:force_original_aspect_ratio=decrease[g1s];"
			"[g1s]pad=1080:960:(ow-iw)/2:(oh-ih)/2[g1p];"
			"[g2]scale=1080:960:force_original_aspect_ratio=decrease[g2s];"
			"[g2s]pad=1080:960:(ow-iw)/2:(oh-ih)/2[g2p];"
			"[g1p][g2p]vstack=inputs=2[vout]"
		)

	# Unknown → caller should fallback
	return ""


	# New builder: external chat video as [1:v] side-by-side with cam in the top band
def build_cam_chat_top_external(cam: CropBox, top_h: int = 768) -> str:
	"""Build a filter graph using an external chat video input ([1:v]).

	Layout:
	- Top band (height=top_h): cam (left 540xtop_h) + external chat (right 540xtop_h)
	- Bottom band (height=1920-top_h): gameplay center-cropped from [0:v]
	"""
	bottom_pad = 96  # 5% of 1920
	content_h = 1920 - bottom_pad
	remaining_h = content_h - top_h
	cell_w = 540
	cell_h = top_h
	gameplay_ar = 1080 / float(remaining_h)

	return (
		# Background blur to provide bottom padding
		"[0:v]scale=1080:1920:force_original_aspect_ratio=increase[bg_si];"
		"[bg_si]crop=1080:1920:(iw-1080)/2:(ih-1920)/2[bg_s];"
		"[bg_s]boxblur=8:1[bg];"
		# Gameplay bottom sized to remaining content height
		f"[0:v]crop=ih*{gameplay_ar:.3f}:ih:(iw-ih*{gameplay_ar:.3f})/2:0[game_c];"
		f"[game_c]scale=1080:{remaining_h}:flags=lanczos[game_s];"
		# Cam crop → scale-to-FILL left cell (no black borders), then center-crop
		f"[0:v]crop={cam.width}:{cam.height}:{cam.x}:{cam.y}[cam_c];"
		f"[cam_c]scale={cell_w}:{cell_h}:force_original_aspect_ratio=increase:flags=lanczos[cam_sf];"
		f"[cam_sf]crop={cell_w}:{cell_h}:(iw-{cell_w})/2:(ih-{cell_h})/2[cam_s];"
		# External chat from input 1 → scale/crop to cell size to match hstack height
		# We expect upstream to render chat EXACTLY {cell_w}x{cell_h} for layout A.
		# Avoid any scaling/cropping here to prevent text being cut.
		"[1:v]format=yuva420p[chat_s];"
		# Compose on background with bottom pad: top band at y=0, gameplay directly below
		"[cam_s][chat_s]hstack=inputs=2[top];"
		"[bg][top]overlay=0:0[base];"
		f"[base][game_s]overlay=0:{top_h}[vout]"
	)


def build_cam_top30_chat_in_gameplay_external(cam: CropBox, top_h: int = 672, chat_w: int = 450, chat_h: int = 300, margin: int = 0) -> str:
	"""Padded layout: 5% top and bottom blur pads. Cam band ~35% (672px), gameplay ~55% (1056px).

	Chat 450x300 is overlaid fully inside the cam at bottom-left (no boundary crossing).
	Assumes external chat is input [1:v] with black background (no chroma key).
	"""
	top_pad = 0  # removed top padding per request
	bottom_pad = 96  # 5% of 1920
	content_h = 1920 - bottom_pad
	cell_w = 1080
	cell_h = top_h
	remaining_h = content_h - cell_h
	gameplay_ar = 1080 / float(remaining_h)

	return (
		# Blurred background to fill pads (scale-to-cover using increase+crop)
		"[0:v]scale=1080:1920:force_original_aspect_ratio=increase[bg_si];"
		"[bg_si]crop=1080:1920:(iw-1080)/2:(ih-1920)/2[bg_s];"
		"[bg_s]boxblur=8:1[bg];"
		# Prepare gameplay to fit remaining content band height
		f"[0:v]crop=ih*{gameplay_ar:.3f}:ih:(iw-ih*{gameplay_ar:.3f})/2:0[game_c];"
		f"[game_c]scale=1080:{remaining_h}:flags=lanczos[game_s];"
		# Prepare cam to exact cam band height
		f"[0:v]crop={cam.width}:{cam.height}:{cam.x}:{cam.y}[cam_c];"
		f"[cam_c]scale={cell_w}:{cell_h}:force_original_aspect_ratio=increase:flags=lanczos[cam_sf];"
		f"[cam_sf]crop={cell_w}:{cell_h}:(iw-{cell_w})/2:(ih-{cell_h})/2[cam_top];"
		# Compose cam and gameplay inside the content band over blurred bg
		f"[bg][cam_top]overlay=0:0[base1];"
		f"[base1][game_s]overlay=0:{cell_h}[base2];"
		# Chat: preserve alpha, no scaling; assume upstream rendered exact size
		"[1:v]format=yuva420p[chat_s];"
		f"[base2][chat_s]overlay={margin}:{cell_h + margin}[vout]"
	)


def build_insta_padding_external(cam: CropBox, top_pad: int = 444, bottom_pad: int = 96, chat_w: int = 486, chat_h: int = 100) -> str:
	"""Instagram-style paddings: blurred background, gameplay band in middle, cam centered in top pad
	with 10px overlap into gameplay, and a narrow chat bar beneath cam within top padding.

	Assumes external chat is input [1:v] with green background to be keyed.
	"""
	content_h = 1920 - top_pad - bottom_pad
	# Build blurred background from input 0 scaled to full frame
	return (
		# Background (scale-to-cover using increase+crop)
		"[0:v]scale=1080:1920:force_original_aspect_ratio=increase[bg_si];"
		"[bg_si]crop=1080:1920:(iw-1080)/2:(ih-1920)/2[bg_s];"
		"[bg_s]boxblur=8:1[bg];"
		# Gameplay content band
		f"[0:v]crop=ih*{1080/float(content_h):.3f}:ih:(iw-ih*{1080/float(content_h):.3f})/2:0[game_c];"
		f"[game_c]scale=1080:{content_h}:flags=lanczos[game_s];"
		f"[bg][game_s]overlay=0:{top_pad}[base];"
		# Cam: crop, then target width 420 preserving AR, position centered x and stick to bottom of top padding
		f"[0:v]crop={cam.width}:{cam.height}:{cam.x}:{cam.y}[cam_c];"
		"[cam_c]scale=420:-2:flags=lanczos[cam_s];"
		f"[base][cam_s]overlay=(W-w)/2:{top_pad}-h[base2];"
		# Chat: preserve alpha, no scaling
		"[1:v]format=yuva420p[chat_s];"
		f"[base2][chat_s]overlay=0:{top_pad + 8}[vout]"
	)


	# Chat-on-top (20%), full stream bottom (80%), external chat in top pad (bottom-aligned)
def build_chat_top20_full_stream_bottom_external(top_h: int = 384, chat_w: int = 756, chat_h: int = 140, bottom_margin: int = 8) -> str:
	"""Place chat in the top 20% band and full stream in the bottom 80%.

	Assumes external chat is input [1:v]. Chat is keyed to remove background and centered horizontally.
	"""
	# Symmetric pads: top and bottom are both top_h
	content_h = 1920 - 2 * top_h
	gameplay_ar = 1080 / float(content_h)

	return (
		# Background from input 0 to fill frame (scale-to-cover using increase+crop)
		"[0:v]scale=1080:1920:force_original_aspect_ratio=increase[bg_si];"
		"[bg_si]crop=1080:1920:(iw-1080)/2:(ih-1920)/2[bg_s];"
		"[bg_s]boxblur=8:1[bg];"
		# Main content band (60%) between the pads
		f"[0:v]crop=ih*{gameplay_ar:.3f}:ih:(iw-ih*{gameplay_ar:.3f})/2:0[full_c];"
		f"[full_c]scale=1080:{content_h}:flags=lanczos[full_s];"
		f"[bg][full_s]overlay=0:{top_h}[base];"
		# Chat: preserve alpha, no scaling
		"[1:v]format=yuva420p[chat_s];"
		f"[base][chat_s]overlay=0:{top_h + bottom_margin}[vout]"
	)
