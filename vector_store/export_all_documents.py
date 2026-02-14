#!/usr/bin/env python3
"""
Export ALL documents from vector store directly from the database.
"""

import json
import sys
import sqlite3
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def export_all_documents_direct(vod_id: str, output_path: str = None):
    """
    Export ALL documents directly from the SQLite database.
    
    Args:
        vod_id: VOD identifier
        output_path: Output JSON file path
    """
    if output_path is None:
        output_path = f"data/vector_stores/{vod_id}/exported_all_documents.json"
    
    print(f"🔍 Exporting ALL documents directly from database for VOD: {vod_id}")
    
    # Connect to SQLite database
    db_path = f"data/vector_stores/{vod_id}/metadata.db"
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return None
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Load chapter file_safe_name mapping
    chapter_map = {}
    chapters_path = Path(f"data/ai_data/{vod_id}/{vod_id}_chapters.json")
    if chapters_path.exists():
        try:
            raw = json.loads(chapters_path.read_text(encoding="utf-8"))
            arr = raw.get("chapters") if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
            for ch in arr:
                if isinstance(ch, dict) and ch.get("id"):
                    chapter_map[ch["id"]] = (ch.get("file_safe_name") or "").strip() or None
        except Exception:
            chapter_map = {}
    
    # Get all documents
    print("📊 Getting all documents from database...")
    cursor.execute("""
        SELECT id, vod_id, start_time, end_time, duration,
               chapter_id, category, excluded, mode,
               text, chat_text,
               chat_rate, chat_rate_z, burst_score, reaction_hits,
               summary, topic, energy,
               role, role_confidence,
               same_topic_prev, topic_thread, topic_key,
               link_type, link_evidence, confidence,
               section_id, section_title
        FROM documents
        ORDER BY start_time
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    print(f"📊 Found {len(rows)} documents in database")
    
    # Convert to exportable format
    export_data = {
        "vod_id": vod_id,
        "total_documents": len(rows),
        "documents": []
    }
    
    for row in rows:
        # Prefer chapter file_safe_name as category for consistency
        chapter_id = row[5]
        chapter_category = chapter_map.get(chapter_id) or row[6]

        export_doc = {
            "id": row[0],
            "start_time": row[2],
            "end_time": row[3],
            "duration": row[4],
            "chapter_id": chapter_id,
            "category": chapter_category,
            "mode": row[8],
            "excluded": bool(row[7]),
            "chat_rate": row[11],
            "chat_rate_z": row[12],
            "burst_score": row[13],
            "reaction_hits": json.loads(row[14]) if row[14] else {},
            "summary": row[15],
            "topic": row[16],
            "energy": row[17],
            "role": row[18],
            "role_confidence": row[19],
            "same_topic_prev": bool(row[20]) if row[20] is not None else None,
            "topic_thread": row[21],
            "topic_key": row[22],
            "link_type": row[23],
            "link_evidence": row[24],
            "confidence": row[25],
            "section_id": row[26],
            "section_title": row[27]
        }
        export_data["documents"].append(export_doc)
    
    # Add summary statistics
    categories = {}
    modes = {}
    chapters = {}
    excluded_count = 0
    total_duration = 0
    has_text_count = 0
    has_chat_count = 0
    
    for doc in export_data["documents"]:
        cat = doc["category"] or "unknown"
        mode = doc["mode"] or "unknown"
        chapter = doc["chapter_id"] or "unknown"
        
        categories[cat] = categories.get(cat, 0) + 1
        modes[mode] = modes.get(mode, 0) + 1
        chapters[chapter] = chapters.get(chapter, 0) + 1
        
        if doc["excluded"]:
            excluded_count += 1
        
        if doc.get("summary"):
            has_text_count += 1
            
        if doc.get("chat_rate_z") is not None:
            has_chat_count += 1
        
        total_duration += doc["duration"]
    
    export_data["statistics"] = {
        "categories": categories,
        "modes": modes,
        "chapters": chapters,
        "excluded_count": excluded_count,
        "has_metrics_count": has_text_count,
        "has_chat_count": has_chat_count,
        "total_duration": total_duration,
        "avg_duration": total_duration / len(export_data["documents"]) if export_data["documents"] else 0,
        "coverage_minutes": total_duration / 60
    }
    
    # Write to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Exported {len(rows)} documents to: {output_file}")
    print(f"📊 Categories: {categories}")
    print(f"📊 Modes: {modes}")
    print(f"📊 Chapters: {chapters}")
    print(f"📊 Excluded: {excluded_count}")
    print(f"📊 Has metrics: {has_text_count}")
    print(f"📊 Has chat: {has_chat_count}")
    print(f"📊 Total coverage: {total_duration/60:.1f} minutes")
    
    return output_file


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Export all documents directly from database")
    parser.add_argument("vod_id", help="VOD ID to export")
    parser.add_argument("--output", help="Output JSON file path")
    
    args = parser.parse_args()
    
    try:
        output_file = export_all_documents_direct(args.vod_id, args.output)
        if output_file:
            print(f"\n🎉 Export complete! Check: {output_file}")
    except Exception as e:
        print(f"❌ Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
