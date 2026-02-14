#!/usr/bin/env python3
"""Regenerate filtered AI data with emotes for VOD 2488522748"""

import json
from pathlib import Path
from src.chat_utils import chat_utils
from loguru import logger

def regenerate_filtered_data(vod_id: str = "2488522748"):
    """Regenerate filtered AI data with emotes"""
    try:
        # Setup paths
        ai_data_dir = Path("data/ai_data") / vod_id
        chat_dir = Path("data/chats") / vod_id
        
        # Load raw AI data
        raw_ai_path = ai_data_dir / f"{vod_id}_ai_data.json"
        logger.info(f"Loading raw AI data from {raw_ai_path}")
        with open(raw_ai_path, 'r', encoding='utf-8') as f:
            ai_data = json.load(f)
        
        # Load chat data
        chat_path = chat_dir / f"{vod_id}_chat.json"
        logger.info(f"Loading chat data from {chat_path}")
        with open(chat_path, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
        
        # Parse chat with emotes
        logger.info("Parsing chat messages with emote extraction")
        df = chat_utils.parse_chat_messages(chat_data['comments'])
        
        # Create chat lookup by timestamp
        chat_by_time = {}
        for _, row in df.iterrows():
            timestamp = int(row['timestamp'])
            chat_by_time[timestamp] = {
                'timestamp': timestamp,
                'content': row['content'],
                'username': row['username'],
                'emotes': row['emotes'] if isinstance(row['emotes'], list) else []
            }
        
        # Update chat messages in AI data segments
        segments = ai_data['segments']
        for segment in segments:
            # Update existing chat messages with emotes
            updated_messages = []
            for msg in segment.get('chat_messages', []):
                timestamp = msg['timestamp']
                if timestamp in chat_by_time:
                    updated_messages.append(chat_by_time[timestamp])
                else:
                    # Keep original if not found (shouldn't happen)
                    updated_messages.append(msg)
            segment['chat_messages'] = updated_messages
        
        # Save filtered AI data
        filtered_path = ai_data_dir / f"{vod_id}_filtered_ai_data.json"
        logger.info(f"Saving filtered AI data to {filtered_path}")
        
        filtered_data = {
            'vod_id': vod_id,
            'segments': segments,
            'metadata': {
                'original_segments': len(segments),
                'filtered_start_time': segments[0]['start_time'],
                'filtered_end_time': segments[-1]['end_time'],
                'filtered_duration': segments[-1]['end_time'] - segments[0]['start_time'],
                'source': 'transcript_boundary_filter'
            }
        }
        
        with open(filtered_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        
        # Count emotes for verification
        total_emotes = 0
        messages_with_emotes = 0
        for segment in segments:
            for msg in segment.get('chat_messages', []):
                if msg.get('emotes'):
                    total_emotes += len(msg['emotes'])
                    messages_with_emotes += 1
        
        logger.info(f" Filtered AI data regeneration complete!")
        logger.info(f"   Messages with emotes: {messages_with_emotes}")
        logger.info(f"   Total emotes found: {total_emotes}")
        
        # Show sample emotes
        sample_emotes = []
        for segment in segments:
            for msg in segment.get('chat_messages', []):
                if msg.get('emotes'):
                    sample_emotes.extend(msg['emotes'][:3])
                    if len(sample_emotes) >= 10:
                        break
            if len(sample_emotes) >= 10:
                break
        
        if sample_emotes:
            logger.info(f"   Sample emotes: {sample_emotes[:10]}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to regenerate filtered AI data: {e}")
        return False

if __name__ == "__main__":
    success = regenerate_filtered_data()
    if not success:
        logger.error("X Filtered AI data regeneration failed")
