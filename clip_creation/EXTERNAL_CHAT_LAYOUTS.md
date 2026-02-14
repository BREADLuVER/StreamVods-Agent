# External Chat Layouts Guide

This document describes the different external chat layouts available for gameplay+cam clips, their visual ratios, selection logic, and usage examples.

## Overview

All layouts use TwitchDownloaderCLI to render chat segments with transparent backgrounds and grey text. Chat is always treated as present (no more YOLO chat detection).

**Note:** TwitchDownloaderCLI renders chat with transparent backgrounds using mask generation and alpha merge to create true transparency.

## Current Issues to Fix

1. **Chat Background Still Green/Black**: Need to ensure transparent backgrounds are working
2. **Chat Not Pre-rendered**: Need 15-second head start for chat downloads
3. **Chat Clipped/Distorted**: Need proper sizing per layout to avoid squishing
4. **Layout JC Missing Bottom Padding**: Need symmetric top/bottom padding

## Layout A: Side-by-Side Top Band (option 1 when detected cam width is less than 450px)

**Function:** `build_cam_chat_top_external`

### Visual Layout

```
┌─────────────────────────────────────┐
│  Chat (540x768)  │  Cam (540x768)   │ ← Top 40% (768px)
├─────────────────────────────────────┤
│                                     │
│        Gameplay (1080x1152)         │ ← Bottom 60% (1152px)
│                                     │
└─────────────────────────────────────┘
```

### Ratios

- **Top Band:** 40% of frame height (768px)
- **Bottom Band:** 60% of frame height (1152px)
- **Cam:** Left half of top band (540x768)
- **Chat:** Right half of top band (540x768) - **RENDER AT 540x768**
- **Gameplay:** Full width, center-cropped to 9:16 ratio

### Selection

- Manual: `LAYOUT_TEST=A`
- Auto: When cam width < 450px (50% chance, randomly chosen with Layout C)

### Use Case

Traditional side-by-side layout. Good for balanced cam/chat visibility.

---

## Layout B: Padded 35:55

**Function:** `build_cam_top30_chat_in_gameplay_external`

### Visual Layout

```
┌─────────────────────────────────────┐
│ 5% pad (blur)
│        Cam (1080x672)               │ ← Cam band (35% of 1920)
├─────────────────────────────────────┤
│ ┌─────┐                             │
│ │Chat │  Gameplay (1080x1056)       │ ← Gameplay band (55% of 1920)
│ │486x │                             │
│ │100   │                             │
│ 5% pad (blur)
│ └─────┘                             │
└─────────────────────────────────────┘
```

### Ratios

- **Pads:** 5% top (96px) and 5% bottom (96px) blurred background
- **Cam Band:** 35% of frame height (672px)
- **Gameplay Band:** 55% of frame height (1056px)
- **Cam:** Full width of cam band (1080x672)
- **Chat:** 486x90px overlay (45% of 1080px width) fully inside the cam at bottom-left (no margin)
- **Gameplay:** Full width, center-cropped to 9:16 ratio

### Selection

- Manual: `LAYOUT_TEST=B`
- Auto: When cam width ≥ 450px

### Use Case

Large cam streams where you want prominent cam visibility with chat as a floating overlay.

---

## Layout C: Instagram-Style Padding (option 2 when detected cam width is less than 450px)

**Function:** `build_insta_padding_external`

### Visual Layout

```
┌─────────────────────────────────────┐
│        Blurred Background           │
│                                     │
│        Cam (420px wide)             │ ← Top padding (250px)
│                                     │
│        Chat (486x100)                │
├─────────────────────────────────────┤
│                                     │
│      Gameplay (1080x1316)           │ ← Content band
│                                     │
├─────────────────────────────────────┤
│        Blurred Background           │ ← Bottom padding (90px)
└─────────────────────────────────────┘
```

### Ratios

- **Top Padding:** 250px (13% of frame)
- **Bottom Padding:** 90px (4.7% of frame)
- **Content Band:** 1316px (68.5% of frame)
- **Cam:** 420px wide, centered in top padding with 10px overlap into gameplay
- **Chat:** 486x90px (45% of 1080px width), centered under cam
- **Background:** Blurred version of source video

### Selection

- Manual: `LAYOUT_TEST=C`
- Auto: When cam width < 450px (50% chance, randomly chosen with Layout A)

### Use Case

Small cam streams (VTubers, small overlays) with Instagram-style aesthetic.

---

## Layout JC: Just Chat with Blurred Padding

**Function:** `build_chat_top20_full_stream_bottom_external`

### Visual Layout

```
┌─────────────────────────────────────┐
│        Blurred Background           │ ← Top padding (384px)
│                                     │
│        Chat (486x100)                │ ← Centered overlay
│        (centered)                   │
├─────────────────────────────────────┤
│                                     │
│      Main Content (1080x1152)       │ ← 60% of frame
│                                     │
├─────────────────────────────────────┤
│        Blurred Background           │ ← Bottom padding (384px)
└─────────────────────────────────────┘
```

### Ratios

- **Top Padding:** 384px (20% of frame) - blurred background
- **Bottom Padding:** 384px (20% of frame) - blurred background
- **Main Content:** 1152px (60% of frame) - center-cropped to 9:16 ratio
- **Chat:** 486x90px (45% of 1080px width), centered horizontally in top padding
- **Background:** Blurred version of source video

### Selection

- Manual: `LAYOUT_TEST=JC`
- Auto: When no cam detected or just_chat classification

### Use Case

Just chatting streams with a clean, centered layout. Main content takes 60% of the frame with blurred padding on top and bottom. Chat appears as a centered overlay in the top padding.

---

## Selection Logic

### Manual Selection

Set environment variable `LAYOUT_TEST` to force a specific layout:

```bash
export LAYOUT_TEST=A  # Side-by-side top band
export LAYOUT_TEST=B  # 35:55 with chat overlay
export LAYOUT_TEST=C  # Instagram-style padding
export LAYOUT_TEST=JC # Just chat top band
```

### Automatic Selection

When `LAYOUT_TEST` is not set, the system uses smart defaults:

1. **Layout JC (Just Chat):** Selected when no webcam is detected or just_chat classification
2. **Layout B (35:55):** Selected when cam width ≥ 450px
3. **Layout A/C (Small Cam):** 50% chance between A and C when cam width < 450px

### Cam Size Thresholds

- **Large Cam:** ≥ 450px width → Layout B
- **Small Cam:** < 450px width → 50% chance between Layout A and Layout C
- **No Cam:** Not detected → Layout JC

---

## Technical Details

### Chat Rendering Requirements

All layouts use TwitchDownloaderCLI with these settings:

```bash
TwitchDownloaderCLI chatrender \
  -i chat.json \
  -h {chat_h} -w {chat_w} \
  --framerate 30 \
  --update-rate 0 \
  --font-size {calculated} \
  --background-color "#00000000" \
  --alt-background-color "#00000000" \
  --message-color "#BFBFBF" \
  --generate-mask \
  -o chat_raw.mp4
```

### Alpha Channel Creation

After TwitchDownloaderCLI renders with mask, merge to create true transparency:

```bash
ffmpeg -i chat_raw.mp4 -i chat_raw_mask.mp4 \
  -filter_complex "[0:v][1:v]alphamerge,format=yuva420p[v]" \
  -map "[v]" -c:v libx264 -pix_fmt yuva420p chat.mp4
```

### Chat Pre-roll

Chat downloads start 20 seconds before clip start time to pre-populate the chatbox:

```python
chat_start_time = max(0, start_time - 20)
```

### Layout-Specific Chat Sizing

- **Layout A:** 540x768 (matches top band cell)
- **Layout B:** 486x90 (45% of 1080px width)
- **Layout C:** 486x90 (45% of 1080px width)
- **Layout JC:** 486x90 (45% of 1080px width)

### Input Mapping

- **Input 0:** Source gameplay video
- **Input 1:** Rendered chat video (external) with alpha channel

---

## Usage Examples

### For Large Cam Streams

```bash
export LAYOUT_TEST=B
python processing-scripts/create_individual_clips.py <vod_id>
```

### For VTuber/Small Cam Streams

```bash
export LAYOUT_TEST=C
python processing-scripts/create_individual_clips.py <vod_id>
```

### For Just Chatting Content

```bash
export LAYOUT_TEST=JC
python processing-scripts/create_individual_clips.py <vod_id>
```

### Let System Choose Automatically

```bash
# Clear any existing LAYOUT_TEST
unset LAYOUT_TEST
python processing-scripts/create_individual_clips.py <vod_id>
```

---

## Customization

### Adjusting Chat Sizes

Edit the layout functions in `clip_creation/ffmpeg_layouts.py`:

- **Layout A:** Modify `cell_w` and `cell_h` in `build_cam_chat_top_external`
- **Layout B:** Modify `chat_w` and `chat_h` parameters in `build_cam_top30_chat_in_gameplay_external`
- **Layout C:** Modify `chat_w` and `chat_h` parameters in `build_insta_padding_external`
- **Layout JC:** Modify `chat_w` and `chat_h` parameters in `build_chat_top20_full_stream_bottom_external`

### Adjusting Cam Size Thresholds

Edit the selection logic in `processing-scripts/create_individual_clips.py`:

```python
if cam_crop.width >= 450:
    choice = 'B'
else:
    choice = random.choice(['A', 'C'])
```

### Disabling Local Cache

For testing, disable local chat caching by removing cache checks in `render_chat_segment`:

```python
# Always force re-render for local testing
if chat_path.exists():
    chat_path.unlink()
```
