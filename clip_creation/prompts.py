#!/usr/bin/env python3
"""
Centralized prompt templates for Molmo vision tasks.

Surgical extraction to avoid duplication across helpers.
"""


def build_face_center_prompt(is_vtuber: bool) -> str:
    vtuber_hint = "There is a VTuber avatar in this image." if is_vtuber else ""
    return f"""
Analyze this livestream/VOD screenshot and locate the streamer's face or VTuber avatar.
{vtuber_hint}

Return the coordinates in this exact format:
x1="<x_coord>" y1="<y_coord>"

Where x_coord and y_coord are normalized values between 0 and 100 representing the center of the face.

Rules:
- Focus on the center of the face/head area
- For VTubers: target the center of the animated character's face
- For humans: target the center of the person's face
- Return ONLY the coordinate format above, no other text
"""


CHAT_EXISTS_PROMPT = """
Look at this livestream/VOD screenshot and determine if there's a chat area visible.

Chat areas typically appear as:
- Text messages on the right side of the screen
- Username: message format
- Emotes and reactions
- Usually in a semi-transparent box or overlay

Respond with only "YES" if you see a chat area, or "NO" if you don't see any chat.
"""


CHAT_CENTER_PROMPT = """
Look at this livestream/VOD screenshot and find the center point of the chat area.

Chat areas typically appear as:
- Text messages on the right side of the screen
- Username: message format
- Emotes and reactions
- Usually in a semi-transparent box or overlay

Return the center coordinates in this exact format:
CENTER_X="<x>" CENTER_Y="<y>"

Where x and y are normalized values between 0 and 100 representing the center point of the chat area.

Rules:
- Find the approximate center of the entire chat area
- Chat is usually on the right side of the screen
- Return ONLY the coordinate format above, no other text
"""


def build_roi_webcam_box_prompt(is_vtuber: bool) -> str:
    vtuber_hint = "There is a VTuber avatar in this image." if is_vtuber else ""
    return f"""
Analyze this livestream/VOD screenshot and locate the streamer’s facecam region.
{vtuber_hint}

Return the coordinates in this exact format:
X1="<x1>" Y1="<y1>" X2="<x2>" Y2="<y2>" X3="<x3>" Y3="<y3>" X4="<x4>" Y4="<y4>"

All coordinates are normalized percentages (0–100) of this image's width/height.

Strict rules (in order):
0) EXCLUDE any solid-color padding/letterbox bars at the image edges (top/bottom/left/right). 
   If a bar is present, the rectangle’s side must align with the seam where real content begins 
   (do not include the padding). Bottom bars are common—set Y2 and Y3 to the seam above the bar.

A) If a WEBCAM WINDOW (PiP) is visible:
   - Select the OUTER edges of that window (axis-aligned). Ignore rounded corner radius; use its tight bounding rectangle.
   - Do not include borders, drop shadows, or padding outside the video window.

B) If NO frame (greenscreen cutout) or a VTuber without a visible window:
   - Create a rectangular box around the face and upper shoulders.
   - Target aspect ratio ≈ 400:350 (~1.14). Small adjustments allowed to tightly fit head+shoulders.
   - Keep size plausible for a facecam (≈1%–30% of the full image area). Avoid chat/HUD.

Output format & geometry:
- X1,Y1 = top-left; X2,Y2 = bottom-left; X3,Y3 = bottom-right; X4,Y4 = top-right.
- Rectangle must be axis-aligned and clean (no rotation).
- Clamp all values to [0, 100].
- Return ONLY the coordinate line in the exact format above—no extra text.
"""


VTUBER_DETECT_PROMPT = """
Look at this livestream/VOD screenshot and determine if the streamer camera shows a VTuber/virtual avatar (2D/3D model, anime-style, PNGTuber, Live2D, reactive image) rather than a human face.

Respond with only "YES" if it is a VTuber/virtual avatar, or "NO" if it is a real human.
"""


