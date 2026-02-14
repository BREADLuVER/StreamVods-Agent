#!/usr/bin/env python3
"""
Document schema and embedding text formatting.

Creates documents with proper schema and formats text for embedding
with structured markers.
"""

import re
from typing import List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    """Document schema for vector store."""
    id: str
    vod_id: str
    start: float
    end: float
    len_s: float
    
    # Chapter metadata
    chapter_id: Optional[str] = None
    category: Optional[str] = None
    excluded: bool = False
    mode: str = 'unknown'  # 'jc' | 'game' | 'unknown'

    
    # Content
    text: str = ""
    chat_text: str = ""
    keywords: List[str] = None
    
    # Burst metrics
    chat_rate: float = 0.0
    chat_rate_z: float = 0.0
    burst_score: float = 0.0
    reaction_hits: dict = None
    
    # Optional section fields (for future use)
    section_id: Optional[str] = None
    section_title: Optional[str] = None
    section_role: Optional[str] = None  # 'topic' | 'encounter'
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.reaction_hits is None:
            self.reaction_hits = {}


def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """
    Extract top noun phrases as keywords from text.
    
    Args:
        text: Input text to analyze
        max_keywords: Maximum number of keywords to return
    
    Returns:
        List of keyword strings
    """
    if not text:
        return []
    
    # Simple keyword extraction using regex patterns
    # Look for noun phrases (2-3 words)
    noun_phrase_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b'
    noun_phrases = re.findall(noun_phrase_pattern, text)
    
    # Look for common gaming/streaming terms
    gaming_terms = [
        'boss fight', 'epic moment', 'funny moment', 'reaction', 'chat',
        'streamer', 'viewer', 'subscriber', 'donation', 'follow',
        'gameplay', 'build', 'craft', 'mine', 'explore', 'battle',
        'victory', 'defeat', 'achievement', 'quest', 'mission',
        'music video', 'song release', 'just chatting', 'streaming',
        'welcome', 'thank you', 'story time', 'gym', 'vocal artist'
    ]
    
    # Look for these terms in text
    found_terms = []
    text_lower = text.lower()
    for term in gaming_terms:
        if term in text_lower:
            found_terms.append(term)
    
    # Also look for single important words (capitalized)
    important_words = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
    
    # Combine and deduplicate
    all_keywords = list(set(noun_phrases + found_terms + important_words))
    
    # Sort by length (prefer longer, more specific terms)
    all_keywords.sort(key=len, reverse=True)
    
    return all_keywords[:max_keywords]


def format_embedding_text(doc: VectorDocument) -> str:
    """
    Format document text for embedding with structured markers.
    
    Args:
        doc: VectorDocument to format
    
    Returns:
        Formatted text string with markers
    """
    parts = []
    
    # Main transcript text
    if doc.text:
        parts.append(f"[TEXT] {doc.text}")
    
    # Chat text
    if doc.chat_text:
        parts.append(f"[CHAT] {doc.chat_text}")

    
    # Chapter category
    if doc.category:
        parts.append(f"[CHAPTER] {doc.category}")
    
    # Mode
    if doc.mode != 'unknown':
        parts.append(f"[MODE] {doc.mode}")
    
    # Keywords
    if doc.keywords:
        keywords_str = "; ".join(doc.keywords)
        parts.append(f"[KEYS] {keywords_str}")
    
    return "\n".join(parts)


def create_document_from_window(window, vod_id: str, chapter_id: Optional[str] = None,
                               category: Optional[str] = None, excluded: bool = False,
                               mode: str = 'unknown') -> VectorDocument:
    """
    Create a VectorDocument from a Window object.
    
    Args:
        window: Window object with segments
        vod_id: VOD identifier
        chapter_id: Chapter ID (if available)
        category: Chapter category (if available)
        excluded: Whether chapter is excluded
        mode: Content mode ('jc', 'game', 'unknown')
    
    Returns:
        VectorDocument object
    """
    # Generate document ID
    doc_id = f"{vod_id}:win:{window.start_time}"
    
    # Combine transcript text from all segments
    transcript_parts = []
    chat_parts = []
    
    total_chat_msgs = 0
    # Check for original chapter information in segments
    original_chapter_type = None
    original_chapter_category = None
    original_chapter_id = None
    
    for segment in window.segments:
        # Add transcript
        transcript = segment.get('transcript', '').strip()
        if transcript:
            transcript_parts.append(transcript)
        
        # Add chat messages
        for msg in segment.get('chat_messages', []):
            message_text = msg.get('content', '').strip()  # Use 'content' field, not 'message'
            if message_text:
                chat_parts.append(message_text)
                total_chat_msgs += 1
        
        # Extract original chapter information from first segment
        if original_chapter_type is None and 'original_chapter_type' in segment:
            original_chapter_type = segment.get('original_chapter_type')
            original_chapter_category = segment.get('original_chapter_category')
            original_chapter_id = segment.get('original_chapter_id')
    
    # Combine text for keyword extraction
    combined_text = " ".join(transcript_parts + chat_parts)
    
    # Extract keywords
    keywords = extract_keywords(combined_text)

    # Use original chapter information if available, otherwise fall back to merged chapter info
    final_chapter_id = original_chapter_id if original_chapter_id else chapter_id
    final_category = original_chapter_category if original_chapter_category else category
    
    # Determine mode based on original chapter type if available
    final_mode = mode
    if original_chapter_type:
        if original_chapter_type.lower() in {'just_chatting', 'irl'}:
            final_mode = 'jc'
        else:
            final_mode = 'game'
    
    # Assemble document
    doc = VectorDocument(
        id=doc_id,
        vod_id=vod_id,
        start=window.start_time,
        end=window.end_time,
        len_s=window.duration,
        chapter_id=final_chapter_id,
        category=final_category,
        excluded=excluded,
        mode=final_mode,
        text=" \n".join(transcript_parts),
        chat_text=" \n".join(chat_parts),
        keywords=keywords,
        # Transfer burst metrics from window
        chat_rate=getattr(window, 'chat_rate', 0.0),
        chat_rate_z=getattr(window, 'chat_rate_z', 0.0),
        burst_score=getattr(window, 'burst_score', 0.0),
        reaction_hits=getattr(window, 'reaction_hits', {}),
    )
    
    return doc


def create_documents_from_windows(windows: List, vod_id: str) -> List[VectorDocument]:
    """
    Create VectorDocument objects from a list of windows.
    
    Args:
        windows: List of Window objects with metadata
        vod_id: VOD identifier
    
    Returns:
        List of VectorDocument objects
    """
    documents = []
    
    for window in windows:
        doc = create_document_from_window(
            window=window,
            vod_id=vod_id,
            chapter_id=getattr(window, 'chapter_id', None),
            category=getattr(window, 'category', None),
            excluded=getattr(window, 'excluded', False),
            mode=getattr(window, 'mode', 'unknown')
        )
        documents.append(doc)
    
    logger.info(f"Created {len(documents)} documents from {len(windows)} windows")
    
    # Log document statistics
    if documents:
        modes = {}
        categories = {}
        for doc in documents:
            modes[doc.mode] = modes.get(doc.mode, 0) + 1
            if doc.category:
                categories[doc.category] = categories.get(doc.category, 0) + 1
        
        logger.info(f"Document modes: {modes}")
        logger.info(f"Document categories: {categories}")
        
        # Log duration statistics
        durations = [doc.len_s for doc in documents]
        logger.info(f"Document durations: min={min(durations):.1f}s, "
                   f"max={max(durations):.1f}s, "
                   f"median={sorted(durations)[len(durations)//2]:.1f}s")
    
    return documents


def validate_document(doc: VectorDocument) -> bool:
    """
    Validate a document for completeness and correctness.
    
    Args:
        doc: VectorDocument to validate
    
    Returns:
        True if valid, False otherwise
    """
    # Check required fields
    if not doc.id or not doc.vod_id:
        logger.warning(f"Document missing required ID fields: {doc.id}")
        return False
    
    # Check time fields
    if doc.start < 0 or doc.end <= doc.start:
        logger.warning(f"Document has invalid time range: {doc.start}-{doc.end}")
        return False
    
    # Check mode
    if doc.mode not in ['jc', 'game', 'unknown']:
        logger.warning(f"Document has invalid mode: {doc.mode}")
        return False
    
    # Check that we have some content
    if not doc.text and not doc.chat_text:
        logger.warning(f"Document has no content: {doc.id}")
        return False
    
    return True


def filter_valid_documents(documents: List[VectorDocument]) -> List[VectorDocument]:
    """
    Filter out invalid documents.
    
    Args:
        documents: List of documents to filter
    
    Returns:
        List of valid documents
    """
    valid_docs = []
    invalid_count = 0
    
    for doc in documents:
        if validate_document(doc):
            valid_docs.append(doc)
        else:
            invalid_count += 1
    
    if invalid_count > 0:
        logger.warning(f"Filtered out {invalid_count} invalid documents")
    
    logger.info(f"Kept {len(valid_docs)} valid documents")
    return valid_docs
