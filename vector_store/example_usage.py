#!/usr/bin/env python3
"""
Example usage of the vector store system.

Demonstrates how to build and query the vector store.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vector_store.pipeline import build_vector_store
from vector_store.query_system import QueryFilters


def example_build_and_query():
    """Example of building and querying a vector store."""
    vod_id = "2525077369"  # Use the VOD from your data
    
    print("🚀 Building vector store...")
    
    # Build the vector store
    query_system = build_vector_store(
        vod_id=vod_id,
        embedding_model="all-MiniLM-L6-v2",
        chat_lag_seconds=5.0
    )
    
    print("\n🔍 Testing queries...")
    
    # Example queries
    test_queries = [
        "funny moments and reactions",
        "epic gameplay highlights", 
        "chat interactions",
        "song release discussion",
        "queue setup and planning"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        
        results = query_system.search(query, k=3)
        
        for i, result in enumerate(results, 1):
            doc = result.document
            print(f"{i}. Score: {result.score:.3f}")
            print(f"   Time: {doc['start']:.1f}s - {doc['end']:.1f}s")
            print(f"   Mode: {doc.get('mode', 'unknown')}")
            print(f"   Chapter: {doc.get('chapter_id', 'none')}")
            print(f"   Text preview: {doc.get('text', '')[:100]}...")
            print()
    
    print("\n🎯 Testing filtered queries...")
    
    # Filter by mode
    print("\nJust Chatting content:")
    jc_filter = QueryFilters(mode='jc')
    jc_results = query_system.search("interesting conversation", k=2, filters=jc_filter)
    for result in jc_results:
        doc = result.document
        print(f"  {doc['start']:.1f}s-{doc['end']:.1f}s: {doc.get('text', '')[:80]}...")
    
    # Filter by time range
    print("\nContent from first 30 minutes:")
    time_filter = QueryFilters(time_range=(0, 1800))  # 0-30 minutes
    time_results = query_system.search("content", k=2, filters=time_filter)
    for result in time_results:
        doc = result.document
        print(f"  {doc['start']:.1f}s-{doc['end']:.1f}s: {doc.get('text', '')[:80]}...")
    
    print("\n✅ Example complete!")


if __name__ == "__main__":
    example_build_and_query()
