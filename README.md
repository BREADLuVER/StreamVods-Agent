# StreamSniped Agent

**Automated VOD Processing & Content Generation Pipeline**

StreamSniped Agent is a fully automated, cloud-native service designed to process livestream VODs without human intervention. It operates as a continuous background service that watches for new content, analyzes it using Machine learning + LLM, and generates production-ready clips, thumbnails, and metadata.

## Architecture

The workflow is entirely automated and follows this sequence:

1. **Ingestion (AWS Watcher)**

   * Continuously monitors for new VOD uploads via AWS.
   * Automatically triggers the processing pipeline upon detection of new content.
2. **Analysis Pipeline**

   * **Vector Store Creation**: Builds a semantic search index using LanceDB to understand the context of the VOD.
   * **Speech & Audio**: Uses gemini models for high-fidelity transcription and speech coherence analysis.
3. **Content Generation**

   * **Clip Creation**: Identifies and extracts high-value segments (funny moments, epic gameplay, reactions).
   * **Smart Thumbnails**: Uses MediaPipe and FER (Facial Emotion Recognition) to scan for the best webcam shots, prioritizing high-energy emotions (happiness, surprise) and clear image quality (open eyes, no blur).
   * **Metadata**: Generates detailed titles, descriptions, and tags for each piece of content.

## Key Features

* **Zero-Touch Automation**: Designed to run indefinitely as a background service. No manual trigger or monitoring required.
* **Intelligent VOD Understanding**: Uses vector embeddings to "understand" the content, allowing for semantic queries like "find funny moments" or "chat reaction."
* **Advanced Thumbnail Selection**: Automatically rates and selects the best face cam shots based on emotion and image quality metrics.
* **Scalable Architecture**: Built on a modular pipeline that can handle large volumes of video data.

## Tech Stack

* **Core**: Python 3.10+
* **Cloud & Infrastructure**: AWS (Boto3), Docker
* **AI & ML**:
  * **LLMs**: OpenAI, Google Gemini
  * **Vision**: MediaPipe, OpenCV, FER (Facial Emotion Recognition)
  * **Audio**: OpenAI Whisper, Faster-Whisper
  * **Embeddings**: Sentence-Transformers
* **Data Storage**: LanceDB (Vector Store), Pandas
* **Media Processing**: FFmpeg, MoviePy
