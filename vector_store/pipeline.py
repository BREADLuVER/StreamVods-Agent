#!/usr/bin/env python3
"""
Main vector store pipeline script.

Builds vector store from VOD data with speech-coherent windows,
chapter metadata, and semantic search capabilities.
"""

import json
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vector_store.window_detector import (
    create_speech_windows, 
    apply_chat_latency_correction,
    determine_pause_threshold,
    split_windows_at_chapter_boundaries
)
from vector_store.chapter_metadata import (
    load_chapters,
    attach_chapter_metadata,
    assign_mode_to_windows
)
from vector_store.document_builder import (
    create_documents_from_windows,
    format_embedding_text,
    filter_valid_documents
)
from vector_store.vector_index import VectorIndex
from vector_store.query_system import QuerySystem, QueryFilters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_ai_data(vod_id: str) -> Dict[str, Any]:
    """Load AI data from JSON file - try filtered first, then fall back to raw."""
    ai_data_dir = Path(f"data/ai_data/{vod_id}")
    
    # Try filtered data first
    filtered_path = ai_data_dir / f"{vod_id}_filtered_ai_data.json"
    raw_path = ai_data_dir / f"{vod_id}_ai_data.json"
    
    ai_data_path = None
    data_source = None
    
    if filtered_path.exists():
        ai_data_path = filtered_path
        data_source = "filtered"
    elif raw_path.exists():
        ai_data_path = raw_path
        data_source = "raw"
    else:
        raise FileNotFoundError(f"AI data not found: {filtered_path} or {raw_path}")
    
    with open(ai_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"📄 AI data source: {data_source} ({ai_data_path.name})")
    logger.info(f"Loaded AI data for VOD {vod_id}: {len(data.get('segments', []))} segments")
    return data


def build_vector_store(vod_id: str, index_path: Optional[str] = None,
                      embedding_model: str = "all-MiniLM-L6-v2",
                      chat_lag_seconds: float = 5.0) -> QuerySystem:
    """
    Build vector store for a VOD.
    
    Args:
        vod_id: VOD identifier
        index_path: Path to store vector index
        embedding_model: Sentence transformer model name
        chat_lag_seconds: Chat latency correction in seconds
    
    Returns:
        QuerySystem instance
    """
    logger.info(f"Building vector store for VOD: {vod_id}")
    start_time = time.time()
    
    # Set up index path
    if index_path is None:
        index_path = f"data/vector_stores/{vod_id}"
    
    # Load input data
    logger.info("Loading AI data...")
    ai_data = load_ai_data(vod_id)
    segments = ai_data.get('segments', [])
    
    if not segments:
        raise ValueError(f"No segments found in AI data for VOD {vod_id}")
    
    # Apply chat latency correction
    logger.info(f"Applying {chat_lag_seconds}s chat latency correction...")
    segments = apply_chat_latency_correction(segments, chat_lag_seconds)
    
    # Load chapters
    logger.info("Loading chapters...")
    chapters = load_chapters(vod_id)
    
    # Create speech-coherent windows
    logger.info("Creating speech-coherent windows...")
    windows = create_speech_windows(segments)
    
    if not windows:
        raise ValueError("No windows created from segments")
    
    # Log window statistics
    window_durations = [w.duration for w in windows]
    logger.info(f"Created {len(windows)} windows")
    logger.info(f"Window duration stats: "
                f"min={min(window_durations):.1f}s, "
                f"max={max(window_durations):.1f}s, "
                f"median={sorted(window_durations)[len(window_durations)//2]:.1f}s")
    
    # Split windows at chapter boundaries
    logger.info("Splitting windows at chapter boundaries...")
    windows = split_windows_at_chapter_boundaries(windows, chapters)
    
    # Attach chapter metadata
    logger.info("Attaching chapter metadata...")
    windows = attach_chapter_metadata(windows, chapters)
    
    # Assign modes to windows
    logger.info("Assigning modes to windows...")
    windows = assign_mode_to_windows(windows)
    
    # Create documents
    logger.info("Creating documents...")
    documents = create_documents_from_windows(windows, vod_id)
    
    # Filter valid documents
    documents = filter_valid_documents(documents)
    
    if not documents:
        raise ValueError("No valid documents created")
    
    # Format embedding texts
    logger.info("Formatting embedding texts...")
    embedding_texts = [format_embedding_text(doc) for doc in documents]
    
    # Create vector index
    logger.info("Creating vector index...")
    vector_index = VectorIndex(index_path, embedding_model)
    
    # Add documents to index
    logger.info("Adding documents to vector index...")
    vector_index.add_documents(documents, embedding_texts)
    
    # Create query system
    query_system = QuerySystem(vector_index)
    
    # Log completion statistics
    elapsed_time = time.time() - start_time
    stats = vector_index.get_stats()
    
    logger.info("=" * 60)
    logger.info(f"Vector store build complete for VOD {vod_id}")
    logger.info(f"Processing time: {elapsed_time:.1f} seconds")
    logger.info(f"Documents indexed: {stats.get('document_count', 0)}")
    logger.info(f"Vector count: {stats.get('vector_count', 0)}")
    logger.info(f"Index path: {index_path}")
    logger.info("=" * 60)
    
    return query_system


def test_query_system(query_system: QuerySystem, vod_id: str):
    """Test the query system with sample queries."""
    logger.info("Testing query system...")
    
    test_queries = [
        "funny moments",
        "epic gameplay",
        "chat interaction",
        "reaction moments",
        "gaming highlights"
    ]
    
    for query in test_queries:
        logger.info(f"Testing query: '{query}'")
        results = query_system.search(query, k=3)
        
        for i, result in enumerate(results, 1):
            doc = result.document
            logger.info(f"  {i}. Score: {result.score:.3f}, "
                       f"Time: {doc['start']:.1f}-{doc['end']:.1f}s, "
                       f"Mode: {doc.get('mode', 'unknown')}, "
                       f"Chapter: {doc.get('chapter_id', 'none')}")
    
    logger.info("Query system test complete")


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(description="Build vector store for VOD")
    parser.add_argument("vod_id", help="VOD ID to process")
    parser.add_argument("--index-path", help="Path to store vector index")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                       help="Sentence transformer model name")
    parser.add_argument("--chat-lag", type=float, default=5.0,
                       help="Chat latency correction in seconds")
    parser.add_argument("--test-queries", action="store_true",
                       help="Run test queries after building")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Build vector store
        query_system = build_vector_store(
            vod_id=args.vod_id,
            index_path=args.index_path,
            embedding_model=args.embedding_model,
            chat_lag_seconds=args.chat_lag
        )
        
        # Test queries if requested
        if args.test_queries:
            test_query_system(query_system, args.vod_id)
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
