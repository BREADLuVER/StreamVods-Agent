#!/usr/bin/env python3
"""
Check Chapters File
Simple script to check the chapters file and understand the format
"""

import json
import sys
from pathlib import Path

def main():
    if len(sys.argv) != 2:
        print("Usage: python check_chapters.py <vod_id>")
        sys.exit(1)
    
    vod_id = sys.argv[1]
    chapters_path = Path(f"data/ai_data/{vod_id}/{vod_id}_chapters.json")
    
    print(f"🔍 Checking chapters file: {chapters_path}")
    
    if not chapters_path.exists():
        print("X Chapters file does not exist")
        print("💡 This explains why the workflow is treating this as a single video")
        return
    
    try:
        with open(chapters_path, 'r', encoding='utf-8') as f:
            chapters_data = json.load(f)
        
        print(f" Chapters file exists and is valid JSON")
        print(f"📊 Data type: {type(chapters_data)}")
        
        if isinstance(chapters_data, list):
            print(f" Number of chapters: {len(chapters_data)}")
            if len(chapters_data) > 1:
                print(" This IS a multi-chapter VOD")
                for i, chapter in enumerate(chapters_data):
                    print(f"   Chapter {i+1}: {chapter.get('category', 'Unknown')}")
            else:
                print("  Only 1 chapter found - will be treated as single video")
        elif isinstance(chapters_data, dict):
            if 'chapters' in chapters_data:
                chapters = chapters_data['chapters']
                print(f" Number of chapters: {len(chapters)}")
                if len(chapters) > 1:
                    print(" This IS a multi-chapter VOD")
                    for i, chapter in enumerate(chapters):
                        print(f"   Chapter {i+1}: {chapter.get('category', 'Unknown')}")
                else:
                    print("  Only 1 chapter found - will be treated as single video")
            else:
                print("  Dict format but no 'chapters' key")
        else:
            print("  Unexpected format")
        
        # Check if focused directory has chapter-specific files
        focused_dir = Path(f"data/ai_data/{vod_id}/focused")
        if focused_dir.exists():
            chapter_files = list(focused_dir.glob(f"{vod_id}_*_sections.json"))
            print(f"\n📁 Found {len(chapter_files)} chapter-specific files in focused/")
            for file in chapter_files:
                print(f"   {file.name}")
        
    except Exception as e:
        print(f"X Error reading chapters file: {e}")

if __name__ == "__main__":
    main() 