"""
Shared chat utilities for StreamSniped
Provides chat parsing and analysis functions used by both phases
"""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from .config import config


def local_zscore(data: np.ndarray, window: int = 60) -> np.ndarray:
    """Calculate local z-score using rolling window"""
    if len(data) < window:
        return np.zeros_like(data)
    
    # Calculate rolling mean and std
    rolling_mean = pd.Series(data).rolling(window=window, center=True, min_periods=1).mean().values
    rolling_std = pd.Series(data).rolling(window=window, center=True, min_periods=1).std().values
    
    # Avoid division by zero
    rolling_std = np.where(rolling_std == 0, 1, rolling_std)
    
    # Calculate z-score
    z_scores = (data - rolling_mean) / rolling_std
    
    return z_scores


class ChatUtils:
    """Shared chat utilities for both Phase 1 and Phase 2"""
    
    def __init__(self):
        pass
    
    def load_chat(self, chat_path: Path) -> List[dict]:
        """Load chat data from JSON file"""
        try:
            with open(chat_path, 'r', encoding='utf-8') as f:
                chat_data = json.load(f)
            
            # Handle TwitchDownloaderCLI format
            if isinstance(chat_data, dict) and 'comments' in chat_data:
                comments = chat_data['comments']
                logger.info(f"Loaded {len(comments)} chat messages from TwitchDownloaderCLI format: {chat_path}")
                return comments
            # Handle simple array format
            elif isinstance(chat_data, list):
                logger.info(f"Loaded {len(chat_data)} chat messages from simple array format: {chat_path}")
                return chat_data
            else:
                logger.warning(f"Unknown chat format in {chat_path}")
                return []
            
        except Exception as e:
            logger.error(f"Failed to load chat from {chat_path}: {e}")
            raise
    
    def parse_chat_messages(self, chat_data: List[dict]) -> pd.DataFrame:
        """Parse chat messages into a DataFrame with timing information"""
        if not chat_data:
            logger.warning("No chat data provided")
            return pd.DataFrame()
        
        # Extract relevant fields
        messages = []
        for msg in chat_data:
            try:
                # Handle different chat formats
                if 'content_offset_seconds' in msg:
                    # TwitchDownloaderCLI format
                    timestamp = msg.get('content_offset_seconds', 0)
                    content = msg.get('message', {}).get('body', '') if msg.get('message') else ''
                    # Extract emotes from TwitchDownloaderCLI fragments structure
                    emotes = []
                    if msg.get('message', {}).get('fragments'):
                        for fragment in msg['message']['fragments']:
                            # Each fragment with an emoticon is an emote
                            if fragment.get('emoticon'):
                                # The fragment text is the emote name
                                emote_text = fragment.get('text', '')
                                if emote_text:
                                    emotes.append(emote_text)
                    
                    # Fallback to legacy emoticons array
                    if not emotes and msg.get('message', {}).get('emoticons'):
                        for emote in msg['message']['emoticons']:
                            if isinstance(emote, dict) and 'emoticon_id' in emote:
                                emotes.append(f"emote_{emote['emoticon_id']}")
                    username = msg.get('commenter', {}).get('display_name', '') if msg.get('commenter') else ''
                else:
                    # Simple format
                    timestamp = msg.get('timestamp', 0)
                    if isinstance(timestamp, str):
                        timestamp = float(timestamp)
                    content = msg.get('content', '')
                    emotes = msg.get('emotes', [])
                    username = msg.get('username', '')
                
                # Ensure timestamp is numeric
                if isinstance(timestamp, str):
                    timestamp = float(timestamp)
                
                messages.append({
                    'timestamp': timestamp,
                    'content': content,
                    'emotes': emotes,
                    'username': username,
                    'second': int(timestamp)  # Round to nearest second
                })
                
            except Exception as e:
                logger.warning(f"Failed to parse message: {e}")
                continue
        
        if not messages:
            logger.warning("No valid messages found in chat data")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(messages)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Parsed {len(df)} chat messages")
        return df
    
    def calculate_chat_activity(self, df: pd.DataFrame, window_size: float = 1.0) -> pd.DataFrame:
        """
        Calculate chat activity metrics over time
        
        Args:
            df: DataFrame with chat messages
            window_size: Time window size in seconds for smoothing
            
        Returns:
            DataFrame with activity metrics per second
        """
        if df.empty:
            logger.warning("No chat data to analyze")
            return pd.DataFrame()
        
        # Group by second and count messages
        activity_df = df.groupby('second').agg({
            'content': 'count',
            'emotes': lambda x: sum(len(emotes) for emotes in x if emotes)
        }).reset_index()
        
        activity_df.columns = ['second', 'message_count', 'emote_count']
        
        # Fill missing seconds with zeros
        min_second = activity_df['second'].min()
        max_second = activity_df['second'].max()
        
        all_seconds = pd.DataFrame({'second': range(min_second, max_second + 1)})
        activity_df = all_seconds.merge(activity_df, on='second', how='left').fillna(0)
        
        # Calculate total activity (messages + emotes)
        activity_df['total_activity'] = activity_df['message_count'] + activity_df['emote_count']
        
        # Apply smoothing using rolling window
        window = max(1, int(window_size))
        activity_df['smoothed_count'] = activity_df['total_activity'].rolling(
            window=window, center=True, min_periods=1
        ).mean()
        
        # Calculate local z-score for reaction detection
        activity_df['reaction_score'] = local_zscore(activity_df['smoothed_count'].values)
        
        # Normalize activity for scoring
        max_activity = activity_df['smoothed_count'].max()
        if max_activity > 0:
            activity_df['normalized_activity'] = activity_df['smoothed_count'] / max_activity
        else:
            activity_df['normalized_activity'] = 0
        
        logger.info(f"Calculated chat activity for {len(activity_df)} seconds")
        return activity_df
    
    def get_chat_snippet(self, chat_df: pd.DataFrame, start_time: int, end_time: int) -> List[dict]:
        """
        Get chat messages for a specific time window
        
        Args:
            chat_df: DataFrame with raw chat messages
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            List of chat messages in the time window
        """
        # Filter messages in the time window
        window_messages = chat_df[
            (chat_df['second'] >= start_time) & 
            (chat_df['second'] <= end_time)
        ]
        
        # Convert to message format
        messages = []
        for _, row in window_messages.iterrows():
            messages.append({
                'timestamp': row['second'],
                'content': row['content'],
                'emotes': row['emotes'] if 'emotes' in row else []
            })
        
        return messages
    
    def calculate_composite_score(self, activity_df: pd.DataFrame) -> pd.Series:
        """Calculate composite score combining multiple metrics"""
        if activity_df.empty:
            return pd.Series()
        
        # Normalize each metric to 0-1 range
        max_count = activity_df['smoothed_count'].max()
        max_reaction = activity_df['reaction_score'].max()
        
        normalized_count = activity_df['smoothed_count'] / max_count if max_count > 0 else 0
        normalized_reaction = activity_df['reaction_score'] / max_reaction if max_reaction > 0 else 0
        
        # Combine metrics (weighted average)
        composite_score = (0.6 * normalized_count + 0.4 * normalized_reaction)
        
        return composite_score


# Global chat utils instance
chat_utils = ChatUtils() 