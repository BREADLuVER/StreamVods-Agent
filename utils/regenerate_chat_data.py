#!/usr/bin/env python3
"""Regenerate chat data with emotes for VOD 2488522748"""

import json
from pathlib import Path
from src.chat_utils import chat_utils
from loguru import logger

def regenerate_chat_data(vod_id: str = "2488522748"):
    """Regenerate chat data with emotes"""
    try:
        # Setup paths
        chat_dir = Path("data/chats") / vod_id
        chat_dir.mkdir(parents=True, exist_ok=True)
        
        raw_chat_path = chat_dir / f"{vod_id}_chat.json"
        clean_chat_path = chat_dir / f"{vod_id}_chat_clean.json"
        
        # Load raw chat data
        logger.info(f"Loading raw chat from {raw_chat_path}")
        with open(raw_chat_path, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
        
        # Parse with updated emote extraction
        logger.info("Parsing chat messages with emote extraction")
        df = chat_utils.parse_chat_messages(chat_data['comments'])
        
        # Convert to clean format
        clean_messages = []
        for _, row in df.iterrows():
            clean_messages.append({
                'timestamp': int(row['timestamp']),
                'content': row['content'],
                'username': row['username'],
                'emotes': row['emotes'] if isinstance(row['emotes'], list) else []
            })
        
        # Save clean chat
        logger.info(f"Saving clean chat to {clean_chat_path}")
        with open(clean_chat_path, 'w', encoding='utf-8') as f:
            json.dump(clean_messages, f, indent=2, ensure_ascii=False)
        
        # Count emotes for verification
        emotes_found = sum(len(msg['emotes']) for msg in clean_messages)
        messages_with_emotes = sum(1 for msg in clean_messages if msg['emotes'])
        
        logger.info(f" Chat regeneration complete!")
        logger.info(f"   Messages with emotes: {messages_with_emotes}")
        logger.info(f"   Total emotes found: {emotes_found}")
        
        # Show sample emotes
        sample_emotes = []
        for msg in clean_messages:
            if msg['emotes']:
                sample_emotes.extend(msg['emotes'][:3])
                if len(sample_emotes) >= 10:
                    break
        
        if sample_emotes:
            logger.info(f"   Sample emotes: {sample_emotes[:10]}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to regenerate chat data: {e}")
        return False

if __name__ == "__main__":
    success = regenerate_chat_data()
    if not success:
        logger.error("X Chat regeneration failed")
