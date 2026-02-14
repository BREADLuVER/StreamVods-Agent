# StreamSniped Video Generation Pipeline Documentation

## Overview
The system operates on a "VOD" (Video on Demand) basis, processing Twitch streams into two main outputs: **Clips** (short, high-engagement moments) and **Arc Videos** (longer, narrative-driven segments).

The pipeline is orchestrated by `aws-scripts/gpu_orchestrator_daemon.py`, which manages job queues and resource allocation.

## 1. Video Pipeline (Arc Videos) - "Top-Down" Approach
This pipeline aims to find coherent stories or gameplay sessions (e.g., a full match, a boss fight, a specific topic discussion).

### A. Detection Phase (`story_archs/gemini_arc_detection.py`)
*   **Method:** The VOD is split into 30-minute chunks (with overlap).
*   **The Brain:** Each chunk's transcript + a list of "Chat Activity Peaks" is sent to Gemini (LLM).
*   **The Prompt:** Gemini is instructed to find arcs fitting a strict narrative structure: `INTRO -> BUILD-UP -> CLIMAX -> RESOLUTION`.
*   **Inputs:**
    *   Transcript (text).
    *   Chat Peaks (top 15 moments of high chat activity).
    *   Previous chunk context (for continuity).
*   **Outputs:** A list of `Arc` objects with start/end times, type, confidence, and scores.

### B. Selection Phase (`story_archs/gemini_to_arc_manifests.py`)
*   **Method:** Detected arcs are rated and filtered.
*   **Scoring System:**
    *   **Hype Score (40%):** Based on `climax_score` (burst data), `peak_chat_rate_z`, and `total_reactions`. Favors loud/high-energy moments.
    *   **Narrative Score (40%):** Based on LLM-assigned `controversy_score` and `narrative_score`. Favors drama and interesting topics.
    *   **Confidence (10%)** & **Resolution (10%)**.
*   **Filtering:** Arcs are selected based on a dynamic threshold or Top-K.

### C. Rendering Phase (`story_archs/create_arch_videos.py`)
*   Selected arcs are rendered into video files using `ffmpeg`.

## 2. Clip Pipeline - "Bottom-Up" Approach
This pipeline finds short, viral moments based purely on audience reaction signals.

### A. Signal Generation (`clip_generation/pipeline.py`)
*   **Method:** Analyzes the VOD second-by-second for signals.
*   **Signals:**
    *   `chat_rate_z`: Statistical deviation of chat speed.
    *   `burst_score`: Sudden spikes in chat.
    *   `reaction_hits`: Specific emotes (LUL, POG, etc.).
    *   `energy`: Audio energy levels.

### B. Grouping & Selection
*   **Seed Groups:** High-signal moments are identified.
*   **Reaction Arcs:** Adjacent signals are grouped together to form a "candidate clip".
*   **Refinement:** Windows are adjusted to capture the "wind-up" and "punchline".
*   **Scoring:** Clips are scored purely on the intensity of the signals.

## 3. Gap Analysis: Missing "Yap" and "Reaction" Content

The current **Video Pipeline** misses "Parasocial Yap" (just chatting/storytime) and "Reaction" (watching videos) content for two reasons:

1.  **Strict Narrative Structure:** The Gemini prompt enforces `INTRO -> BUILD-UP -> CLIMAX -> RESOLUTION`.
    *   *Issue:* A "Yap" session often lacks a clear "Climax" (it's a steady stream of talking).
    *   *Issue:* A "Reaction" segment might be a series of small laughs without a single big narrative arc.
2.  **Signal Mismatch:**
    *   *Issue:* "Yap" content often has *lower* energy and *lower* chat velocity than gameplay, so it gets down-weighted by the "Hype Score" in the selection phase.
    *   *Issue:* The "Chat Peaks" passed to Gemini might miss sustained, moderate engagement (typical of interesting stories) in favor of sudden spikes (typical of gameplay kills).

## 4. Proposed Improvements

To capture this content, we need to:

1.  **Update Arc Definitions:** Modify the Gemini prompt to recognize `PARASOCIAL_YAP` and `REACTION` as valid arc types that *don't* require a traditional climax.
2.  **Adjust Scoring:** Create specific scoring weights for these new types (e.g., value "Narrative Score" much higher than "Hype Score" for Yap content).
3.  **Hybrid Signals:** Inject "Bottom-Up" signals (sustained chat engagement) into the "Top-Down" prompt to help Gemini spot where the "Yap" is happening.
