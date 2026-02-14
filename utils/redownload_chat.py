#!/usr/bin/env python3
"""Re-download chat messages for VOD 2488522748 and create clean version with emotes"""

import subprocess
import sys
import json
from pathlib import Path
from loguru import logger
from src.chat_utils import chat_utils

def redownload_chat(vod_id: str = "2488522748"):
    """Re-download chat with emotes embedded and create clean version"""
    
    # Setup paths
    chat_dir = Path("data/chats") / vod_id
    chat_dir.mkdir(parents=True, exist_ok=True)
    
    raw_chat_path = chat_dir / f"{vod_id}_chat.json"
    clean_chat_path = chat_dir / f"{vod_id}_chat_clean.json"
    
    # TwitchDownloaderCLI command with emotes embedded
    cmd = [
        "TwitchDownloaderCLI",
        "chatdownload",
        "--id", vod_id,
        "--embed-images",  # Embed emote images
        "--bttv=true",     # Include BTTV emotes
        "--ffz=true",      # Include FFZ emotes  
        "--stv=true",      # Include 7TV emotes
        "-o", str(raw_chat_path)
    ]
    
    logger.info(f"🔄 Re-downloading chat for VOD {vod_id}")
    logger.info(f"📁 Raw output: {raw_chat_path}")
    logger.info(f"📁 Clean output: {clean_chat_path}")
    logger.info(f"🔧 Command: {' '.join(cmd)}")
    
    try:
        # Run the download
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        logger.info(" Raw chat download completed successfully!")
        
        # Verify raw file exists and has content
        if not raw_chat_path.exists():
            logger.error("X Raw chat file was not created!")
            return False
            
        file_size = raw_chat_path.stat().st_size
        logger.info(f"📄 Raw file size: {file_size:,} bytes")
        
        if file_size == 0:
            logger.error("X Raw chat file is empty!")
            return False
        
        # Process raw chat to create clean version
        logger.info("🔄 Processing raw chat to create clean version...")
        
        # Load raw chat data
        with open(raw_chat_path, 'r', encoding='utf-8') as f:
            raw_chat_data = json.load(f)
        
        # Parse with our updated chat utils (includes emote extraction)
        df = chat_utils.parse_chat_messages(raw_chat_data['comments'])
        
        # Convert to clean format
        clean_messages = []
        for _, row in df.iterrows():
            clean_message = {
                "timestamp": int(row['timestamp']),
                "content": row['content'],
                "username": row['username'],
                "emotes": row['emotes']  # This now contains properly extracted emote names
            }
            clean_messages.append(clean_message)
        
        # Save clean version
        with open(clean_chat_path, 'w', encoding='utf-8') as f:
            json.dump(clean_messages, f, indent=2, ensure_ascii=False)
        
        # Verify clean file
        clean_file_size = clean_chat_path.stat().st_size
        logger.info(f"📄 Clean file size: {clean_file_size:,} bytes")
        
        # Count emotes in clean version
        total_emotes = sum(len(msg['emotes']) for msg in clean_messages if msg['emotes'])
        messages_with_emotes = sum(1 for msg in clean_messages if msg['emotes'])
        
        logger.info(f"📊 Clean chat stats:")
        logger.info(f"   - Total messages: {len(clean_messages)}")
        logger.info(f"   - Messages with emotes: {messages_with_emotes}")
        logger.info(f"   - Total emotes: {total_emotes}")
        
        if total_emotes > 0:
            # Show sample emotes
            sample_emotes = [msg['emotes'] for msg in clean_messages if msg['emotes']][:3]
            logger.info(f"   - Sample emotes: {sample_emotes}")
        
        logger.info(" Clean chat file created successfully!")
        return True
            
    except subprocess.CalledProcessError as e:
        logger.error(f"X Chat download failed!")
        logger.error(f"Exit code: {e.returncode}")
        logger.error(f"Error: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("X TwitchDownloaderCLI not found! Make sure it's installed and in PATH")
        return False
    except Exception as e:
        logger.error(f"X Unexpected error: {e}")
        return False

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
    
    # Re-download chat
    success = redownload_chat()
    
    if success:
        print("\n🎉 Chat re-download and processing completed successfully!")
        print("📁 Raw chat: data/chats/2488522748/2488522748_chat.json")
        print("📁 Clean chat: data/chats/2488522748/2488522748_chat_clean.json")
    else:
        print("\n💥 Chat re-download failed!")
        sys.exit(1)
