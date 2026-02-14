#!/usr/bin/env python3
"""
Command-line interface for querying the vector store.

Provides easy access to semantic search functionality.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vector_store.vector_index import VectorIndex
from vector_store.query_system import QuerySystem, QueryFilters


def load_query_system(vod_id: str, index_path: Optional[str] = None) -> QuerySystem:
    """Load existing query system from vector store."""
    if index_path is None:
        index_path = f"data/vector_stores/{vod_id}"
    
    vector_index = VectorIndex(index_path)
    return QuerySystem(vector_index)


def format_search_result(result, show_content: bool = False) -> str:
    """Format a search result for display."""
    doc = result.document
    
    lines = [
        f"Rank {result.rank}: Score {result.score:.3f}",
        f"  Time: {doc['start']:.1f}s - {doc['end']:.1f}s ({doc['len_s']:.1f}s)",
        f"  Mode: {doc.get('mode', 'unknown')}",
        f"  Chapter: {doc.get('chapter_id', 'none')}",
        f"  Category: {doc.get('category', 'none')}"
    ]
    
    if show_content:
        text_preview = doc.get('text', '')[:200]
        if len(doc.get('text', '')) > 200:
            text_preview += "..."
        lines.append(f"  Text: {text_preview}")
        
        if doc.get('keywords'):
            lines.append(f"  Keywords: {', '.join(doc['keywords'][:3])}")
    
    return "\n".join(lines)


def search_command(query_system: QuerySystem, query: str, k: int = 10, 
                  filters: Dict[str, Any] = None, show_content: bool = False):
    """Execute search command."""
    print(f"Searching for: '{query}'")
    print(f"Results requested: {k}")
    
    if filters:
        print(f"Filters: {filters}")
    
    print("-" * 50)
    
    # Convert filters to QueryFilters object
    query_filters = None
    if filters:
        query_filters = QueryFilters(**filters)
    
    # Perform search
    results = query_system.search(query, k=k, filters=query_filters)
    
    if not results:
        print("No results found.")
        return
    
    print(f"Found {len(results)} results:")
    print()
    
    for result in results:
        print(format_search_result(result, show_content))
        print()


def time_range_command(query_system: QuerySystem, start_time: float, end_time: float,
                      vod_id: str = None):
    """Execute time range search command."""
    print(f"Searching time range: {start_time:.1f}s - {end_time:.1f}s")
    if vod_id:
        print(f"VOD ID: {vod_id}")
    
    print("-" * 50)
    
    results = query_system.search_by_time_range(start_time, end_time, vod_id)
    
    if not results:
        print("No results found in time range.")
        return
    
    print(f"Found {len(results)} results in time range:")
    print()
    
    for result in results:
        print(format_search_result(result, show_content=True))
        print()


def chapter_command(query_system: QuerySystem, chapter_id: str, vod_id: str = None,
                   query: str = "content"):
    """Execute chapter search command."""
    print(f"Searching chapter: {chapter_id}")
    if vod_id:
        print(f"VOD ID: {vod_id}")
    print(f"Query: '{query}'")
    
    print("-" * 50)
    
    results = query_system.search_by_chapter(chapter_id, vod_id, query)
    
    if not results:
        print("No results found in chapter.")
        return
    
    print(f"Found {len(results)} results in chapter:")
    print()
    
    for result in results:
        print(format_search_result(result, show_content=True))
        print()


def stats_command(query_system: QuerySystem):
    """Show vector store statistics."""
    stats = query_system.get_stats()
    
    print("Vector Store Statistics:")
    print("-" * 30)
    
    index_stats = stats.get('vector_index_stats', {})
    print(f"Index Path: {index_stats.get('index_path', 'unknown')}")
    print(f"Embedding Model: {index_stats.get('embedding_model', 'unknown')}")
    print(f"Document Count: {index_stats.get('document_count', 0)}")
    print(f"Vector Count: {index_stats.get('vector_count', 0)}")
    print(f"Has FAISS: {index_stats.get('has_faiss', False)}")
    print(f"Has Embedding Model: {index_stats.get('has_embedding_model', False)}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Query vector store")
    parser.add_argument("vod_id", help="VOD ID")
    parser.add_argument("--index-path", help="Path to vector index")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for content')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('-k', type=int, default=10, help='Number of results')
    search_parser.add_argument('--mode', help='Filter by mode (jc/game/unknown)')
    search_parser.add_argument('--chapter-id', help='Filter by chapter ID')
    search_parser.add_argument('--category', help='Filter by category')
    search_parser.add_argument('--min-duration', type=float, help='Minimum duration in seconds')
    search_parser.add_argument('--show-content', action='store_true', help='Show content preview')
    
    # Time range command
    time_parser = subparsers.add_parser('time', help='Search by time range')
    time_parser.add_argument('start', type=float, help='Start time in seconds')
    time_parser.add_argument('end', type=float, help='End time in seconds')
    
    # Chapter command
    chapter_parser = subparsers.add_parser('chapter', help='Search by chapter')
    chapter_parser.add_argument('chapter_id', help='Chapter ID')
    chapter_parser.add_argument('--query', default='content', help='Search query')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Load query system
        query_system = load_query_system(args.vod_id, args.index_path)
        
        # Execute command
        if args.command == 'search':
            filters = {}
            if args.mode:
                filters['mode'] = args.mode
            if args.chapter_id:
                filters['chapter_id'] = args.chapter_id
            if args.category:
                filters['category'] = args.category
            if args.min_duration:
                filters['min_len_s'] = args.min_duration
            
            search_command(query_system, args.query, args.k, filters, args.show_content)
        
        elif args.command == 'time':
            time_range_command(query_system, args.start, args.end, args.vod_id)
        
        elif args.command == 'chapter':
            chapter_command(query_system, args.chapter_id, args.vod_id, args.query)
        
        elif args.command == 'stats':
            stats_command(query_system)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
