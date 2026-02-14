#!/usr/bin/env python3
"""
Export full VOD vector store data to JSON for inspection.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vector_store.vector_index import VectorIndex
from vector_store.query_system import QuerySystem


def export_full_vod_to_json(vod_id: str, output_path: str = None):
    """
    Export full VOD vector store data to JSON file.
    
    Args:
        vod_id: VOD identifier
        output_path: Output JSON file path
    """
    if output_path is None:
        output_path = f"data/vector_stores/{vod_id}_full/exported_full_data.json"
    
    print(f"🔍 Exporting FULL VOD vector store data for VOD: {vod_id}")
    
    # Load vector store
    index_path = f"data/vector_stores/{vod_id}_full"
    vector_index = VectorIndex(index_path)
    query_system = QuerySystem(vector_index)
    
    # Get all documents
    print("📊 Getting all documents...")
    # Use a very generic query to get all documents
    all_results = query_system.search("", k=10000, rerank=False)
    
    # Convert to exportable format
    export_data = {
        "vod_id": vod_id,
        "total_documents": len(all_results),
        "documents": []
    }
    
    for result in all_results:
        doc = result.document
        export_doc = {
            "id": doc.get('id'),
            "start_time": doc.get('start'),
            "end_time": doc.get('end'),
            "duration": doc.get('len_s'),
            "chapter_id": doc.get('chapter_id'),
            "category": doc.get('category'),
            "mode": doc.get('mode'),
            "excluded": doc.get('excluded', False),
            "text_preview": doc.get('text', '')[:200] + "..." if len(doc.get('text', '')) > 200 else doc.get('text', ''),
            "chat_preview": doc.get('chat_text', '')[:100] + "..." if len(doc.get('chat_text', '')) > 100 else doc.get('chat_text', ''),
            "keywords": doc.get('keywords', []),
            "search_score": result.score
        }
        export_data["documents"].append(export_doc)
    
    # Sort by start time
    export_data["documents"].sort(key=lambda x: x["start_time"])
    
    # Add summary statistics
    categories = {}
    modes = {}
    chapters = {}
    excluded_count = 0
    total_duration = 0
    
    for doc in export_data["documents"]:
        cat = doc["category"] or "unknown"
        mode = doc["mode"] or "unknown"
        chapter = doc["chapter_id"] or "unknown"
        
        categories[cat] = categories.get(cat, 0) + 1
        modes[mode] = modes.get(mode, 0) + 1
        chapters[chapter] = chapters.get(chapter, 0) + 1
        
        if doc["excluded"]:
            excluded_count += 1
        
        total_duration += doc["duration"]
    
    export_data["statistics"] = {
        "categories": categories,
        "modes": modes,
        "chapters": chapters,
        "excluded_count": excluded_count,
        "total_duration": total_duration,
        "avg_duration": total_duration / len(export_data["documents"]) if export_data["documents"] else 0,
        "coverage_minutes": total_duration / 60
    }
    
    # Write to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Exported {len(all_results)} documents to: {output_file}")
    print(f"📊 Categories: {categories}")
    print(f"📊 Modes: {modes}")
    print(f"📊 Chapters: {chapters}")
    print(f"📊 Excluded: {excluded_count}")
    print(f"📊 Total coverage: {total_duration/60:.1f} minutes")
    
    return output_file


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Export full VOD vector store to JSON")
    parser.add_argument("vod_id", help="VOD ID to export")
    parser.add_argument("--output", help="Output JSON file path")
    
    args = parser.parse_args()
    
    try:
        output_file = export_full_vod_to_json(args.vod_id, args.output)
        print(f"\n🎉 Export complete! Check: {output_file}")
    except Exception as e:
        print(f"❌ Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
