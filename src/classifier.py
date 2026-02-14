#!/usr/bin/env python3
"""
Content classification module for StreamSniped
Uses GPT to analyze clips/chunks and generate metadata
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

from .config import config


class ContentClassifier:
    """Handles content classification using GPT for both clips and chunks"""
    
    def __init__(self):
        self.api_key = config.openai_api_key
        self.model = config.gpt_model
        
    def classify_clip(self, 
                     clip_path: Path,
                     transcript: str,
                     chat_snippet: List[Dict],
                     clip_start: int,
                     clip_end: int,
                     vod_id: str) -> Dict:
        """
        Classify a single clip using GPT (Phase 1)
        
        Args:
            clip_path: Path to the clip file
            transcript: Whisper transcript text
            chat_snippet: List of chat messages during clip time
            clip_start: Start time in VOD (seconds)
            clip_end: End time in VOD (seconds)
            vod_id: Source VOD ID
            
        Returns:
            Classification result with metadata
        """
        
        # Format chat snippet for analysis
        chat_text = self._format_chat_snippet(chat_snippet)
        
        # Create prompt for GPT
        prompt = self._create_clip_classification_prompt(
            transcript=transcript,
            chat_text=chat_text,
            clip_start=clip_start,
            clip_end=clip_end,
            duration=clip_end - clip_start
        )
        
        # Call GPT
        classification = self._call_gpt(prompt)
        
        # Parse and validate response
        result = self._parse_classification(classification, clip_path, vod_id)
        
        return result
    
    def classify_chunk(self,
                      chunk_path: Path,
                      transcript: str,
                      chat_snippet: List[Dict],
                      chunk_start: int,
                      chunk_end: int,
                      vod_id: str) -> Dict:
        """
        Classify a single chunk using GPT (Phase 2)
        
        Args:
            chunk_path: Path to the chunk file
            transcript: Whisper transcript text
            chat_snippet: List of chat messages during chunk time
            chunk_start: Start time in VOD (seconds)
            chunk_end: End time in VOD (seconds)
            vod_id: Source VOD ID
            
        Returns:
            Classification result with narrative-focused metadata
        """
        
        # Format chat snippet for analysis
        chat_text = self._format_chat_snippet(chat_snippet)
        
        # Create prompt for GPT
        prompt = self._create_chunk_classification_prompt(
            transcript=transcript,
            chat_text=chat_text,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            duration=chunk_end - chunk_start
        )
        
        # Call GPT
        classification = self._call_gpt(prompt)
        
        # Parse and validate response
        result = self._parse_classification(classification, chunk_path, vod_id)
        
        return result
    
    def _format_chat_snippet(self, chat_snippet: List[Dict]) -> str:
        """Format chat messages for analysis"""
        if not chat_snippet:
            return "No chat activity during this segment."
            
        formatted = []
        for msg in chat_snippet:
            timestamp = msg.get('timestamp', 0)
            content = msg.get('content', '')
            emotes = msg.get('emotes', [])
            
            # Format emotes
            emote_text = f" [{', '.join(emotes)}]" if emotes else ""
            
            formatted.append(f"[{timestamp}s] {content}{emote_text}")
        
        return "\n".join(formatted)
    
    def _create_clip_classification_prompt(self, 
                                         transcript: str,
                                         chat_text: str,
                                         clip_start: int,
                                         clip_end: int,
                                         duration: int) -> str:
        """Create GPT prompt for clip classification (Phase 1)"""
        
        start_time = f"{clip_start // 3600:02d}:{(clip_start % 3600) // 60:02d}:{clip_start % 60:02d}"
        end_time = f"{clip_end // 3600:02d}:{(clip_end % 3600) // 60:02d}:{clip_end % 60:02d}"
        
        prompt = f"""
            Analyze this Twitch clip like a brutally honest Twitch enjoyer. Be ruthless about what's actually entertaining. Rate it as if you're curating for a highlight reel where boring clips get roasted in the comments. Respond in the following JSON format:

            {{
                "start": "{start_time}",
                "end": "{end_time}",
                "label": "reaction|banter|setup|low_energy|gameplay|other",
                "score": 0.0-10.0,
                "clip_title": "Brief descriptive title",
                "transcript": "{transcript}",
                "keep": true/false,
                "tags": ["tag1", "tag2", "tag3"],
                "reasoning": "Brief explanation of classification"
            }}

            CLIP INFO:
            - Duration: {duration} seconds
            - Time: {start_time} - {end_time}

            TRANSCRIPT:
            {transcript}

            🔥 CHAT REACTIONS (THIS IS CRUCIAL FOR SCORING):
            {chat_text}

            LABEL GUIDELINES:
            - reaction: Streamer reacting to something (laugh, shock, rage, excitement)
            - banter: Casual conversation, jokes, chat interaction, entertaining commentary
            - setup: Building up to something, preparation, anticipation
            - low_energy: Quiet, boring, filler content, minimal engagement
            - gameplay: Pure gameplay without much commentary
            - other: Doesn't fit other categories

            🔥 CHAT CONTEXT IS THE KEY TO SCORING:
            - If chat is spamming "LMAO", "KEKW", "🤣", "AHAHAHA" = THIS IS HILARIOUS (7-9/10)
            - If chat is going wild with reactions = streamer did something worth watching
            - Chat reactions are MORE important than transcript content for entertainment value
            - A simple "Oh" with chat going crazy = better than long commentary with no reactions

            SCORING GUIDELINES - Focus on ENTERTAINMENT VALUE:
            - 0–2 Dead clip: no voice, silence, irrelevant chatter
            - 3–4 Low energy, filler clip, thanking gifted subs, thanking subscribers, thanking viewers, could be cut
            - 5–6 Mildly entertaining, filler but passable
            - 7	Solid moment, has either charm, narrative, or funny interaction
            - 8	Strong clip: funny joke, memorable interaction, engaging, possibly standalone
            - 9–10	Gold-tier: iconic line, huge moment, heavy emotion or laughter, clipped by viewers

            keep = true/false. Should this clip survive the purge and go to final output? Be ruthless. You're the bouncer.
            CONTEXTUAL TIPS (When in doubt, be mean):
            Ask: "Would I actually share this with a friend?"
            If streamer is thanking subs, yawning, or just walking in-game, it's a no.
            A weird rant, hilarious fail, or spicy chat moment? That's the gold.
            Chat going wild, laughing, emoting, or reacting to something = streamer probably did something worth watching
            Think like you're curating for a Twitch recap YouTube channel with real standards.
            Any clip with a score above 5 is a keeper.
            Respond with ONLY the JSON object, no other text.
            """
        return prompt
    
    def _create_chunk_classification_prompt(self,
                                          transcript: str,
                                          chat_text: str,
                                          chunk_start: int,
                                          chunk_end: int,
                                          duration: int) -> str:
        """Create GPT prompt for chunk classification (Phase 2)"""
        
        start_time = f"{chunk_start // 3600:02d}:{(chunk_start % 3600) // 60:02d}:{chunk_start % 60:02d}"
        end_time = f"{chunk_end // 3600:02d}:{(chunk_end % 3600) // 60:02d}:{chunk_end % 60:02d}"
        
        prompt = f"""
            Analyze this VOD chunk like a brutally honest video editor. Be ruthless about what's actually worth keeping for a narrative. Rate it as if you're curating for a story where boring chunks get cut immediately. Respond in the following JSON format:

            {{
                "start": "{start_time}",
                "end": "{end_time}",
                "label": "setup|payoff|transition|climax|character_development|banter|low_energy|other",
                "score": 0.0-10.0,
                "narrative_role": "opening|rising_action|climax|falling_action|closing|filler",
                "story_arc": "beginning|middle|end|standalone",
                "character_moments": ["moment1", "moment2"],
                "transcript": "{transcript}",
                "keep": true/false,
                "tags": ["tag1", "tag2", "tag3"],
                "reasoning": "Brief explanation of classification"
            }}

            CHUNK INFO:
            - Duration: {duration} seconds
            - Time: {start_time} - {end_time}

            TRANSCRIPT:
            {transcript}

            🔥 CHAT REACTIONS (THIS IS CRUCIAL FOR SCORING):
            {chat_text}

            LABEL GUIDELINES:
            - setup: Building tension, introducing conflict, preparing for something
            - payoff: Resolution, climax, satisfying conclusion to setup
            - transition: Moving between topics, mood shifts, scene changes
            - climax: Peak emotional moment, highest tension, most dramatic
            - character_development: Streamer personality, growth, memorable traits
            - banter: Casual conversation, jokes, chat interaction
            - low_energy: Filler content, minimal engagement, skippable
            - other: Doesn't fit other categories

            NARRATIVE ROLE GUIDELINES:
            - opening: Sets the scene, introduces characters, establishes mood
            - rising_action: Builds tension, develops conflict, moves toward climax
            - climax: Peak moment, highest emotional intensity
            - falling_action: Resolves tension, provides closure
            - closing: Wraps up story, provides conclusion
            - filler: Doesn't advance narrative, can be cut

            STORY ARC GUIDELINES:
            - beginning: Start of a story arc or narrative thread
            - middle: Development of ongoing story or conflict
            - end: Conclusion of a story arc or narrative thread
            - standalone: Self-contained moment that doesn't need context

            🔥 CHAT CONTEXT IS THE KEY TO SCORING:
            - If chat is spamming "LMAO", "KEKW", "🤣", "AHAHAHA" = THIS IS HILARIOUS (7-9/10)
            - If chat is going wild with reactions = streamer did something worth watching
            - Chat reactions are MORE important than transcript content for entertainment value
            - A simple "Oh" with chat going crazy = better than long commentary with no reactions

            SCORING GUIDELINES - Focus on ENTERTAINMENT VALUE:
            - 0–2 Dead chunk: no voice, silence, irrelevant chatter, thanking subs
            - 3–4 Low energy, filler chunk, thanking gifted subs, thanking subscribers, thanking viewers, could be cut
            - 5–6 Mildly entertaining, filler but passable
            - 7	Solid moment, has either charm, narrative, or funny interaction
            - 8	Strong chunk: funny joke, memorable interaction, engaging, possibly standalone
            - 9–10	Gold-tier: iconic line, huge moment, heavy emotion or laughter, clipped by viewers

            keep = true/false. Should this chunk survive the purge and go to final output? Be ruthless. You're the bouncer.
            CONTEXTUAL TIPS (When in doubt, be mean):
            Ask: "Would I actually share this with a friend?"
            If streamer is thanking subs, yawning, or just walking in-game, it's a no.
            A weird rant, hilarious fail, or spicy chat moment? That's the gold.
            Chat going wild, laughing, emoting, or reacting to something = streamer probably did something worth watching
            Think like you're curating for a Twitch recap YouTube channel with real standards.
            Any chunk with a score above 5 is a keeper.
            Respond with ONLY the JSON object, no other text.
            """
        return prompt
    
    def _call_gpt(self, prompt: str) -> str:
        """Call GPT API with real OpenAI integration"""
        try:
            from openai import OpenAI
            
            # Create client
            client = OpenAI(api_key=self.api_key)
            
            # Call GPT-3.5-turbo
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes Twitch content and provides classification metadata in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent output
                max_tokens=1000
            )
            
            # Extract response content
            content = response.choices[0].message.content.strip()
            
            # Clean up response (remove markdown if present)
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            return content.strip()
            
        except ImportError:
            logger.error("OpenAI library not installed. Install with: pip install openai")
            return self._get_fallback_response()
        except Exception as e:
            logger.error(f"GPT API call failed: {e}")
            return self._get_fallback_response()
    
    def _get_fallback_response(self) -> str:
        """Fallback response when API fails"""
        return json.dumps({
            "start": "00:00:00",
            "end": "00:00:20", 
            "label": "other",
            "score": 0.0,
            "clip_title": "Classification Failed",
            "transcript": "",
            "keep": False,
            "tags": ["error"],
            "reasoning": "API call failed - using fallback classification"
        })
    
    def _parse_classification(self, 
                            classification: str, 
                            content_path: Path,
                            vod_id: str) -> Dict:
        """Parse and validate GPT classification response"""
        try:
            data = json.loads(classification)
            
            # Validate required fields
            required_fields = ["start", "end", "label", "score", "keep"]
            for field in required_fields:
                if field not in data:
                    data[field] = None
            
            # Add metadata
            data.update({
                "content_path": str(content_path),
                "vod_id": vod_id,
                "classification_model": self.model
            })
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse GPT response: {e}")
            return {
                "start": "00:00:00",
                "end": "00:00:00",
                "label": "other",
                "score": 0.0,
                "clip_title": "Classification Failed",
                "transcript": "",
                "keep": False,
                "tags": ["error"],
                "reasoning": f"Failed to parse GPT response: {e}",
                "content_path": str(content_path),
                "vod_id": vod_id,
                "classification_model": self.model
            }
    
    def classify_clips(self, 
                      clips_data: List[Dict],
                      chat_contexts: Optional[Dict[Path, List[Dict]]] = None) -> List[Dict]:
        """
        Classify multiple clips (Phase 1)
        
        Args:
            clips_data: List of clip data with transcript and chat info
            chat_contexts: Optional dict mapping clip paths to chat snippets
            
        Returns:
            List of classification results
        """
        results = []
        
        for i, clip_data in enumerate(clips_data, 1):
            logger.info(f"Classifying clip {i}/{len(clips_data)}: {clip_data['clip_path']}")
            
            try:
                # Get chat context if available
                chat_snippet = []
                if chat_contexts:
                    clip_path_str = clip_data['clip_path']
                    chat_snippet = chat_contexts.get(clip_path_str, [])
                
                classification = self.classify_clip(
                    clip_path=Path(clip_data['clip_path']),
                    transcript=clip_data['transcript'],
                    chat_snippet=chat_snippet,
                    clip_start=clip_data['start_time'],
                    clip_end=clip_data['end_time'],
                    vod_id=clip_data['vod_id']
                )
                results.append(classification)
                
            except Exception as e:
                logger.error(f"Failed to classify clip: {e}")
                results.append({
                    "start": "00:00:00",
                    "end": "00:00:00", 
                    "label": "other",
                    "score": 0.0,
                    "clip_title": "Classification Error",
                    "transcript": "",
                    "keep": False,
                    "tags": ["error"],
                    "reasoning": f"Classification failed: {e}",
                    "clip_path": clip_data.get('clip_path', ''),
                    "vod_id": clip_data.get('vod_id', ''),
                    "classification_model": self.model
                })
        
        return results
    
    def classify_chunks(self,
                       chunks_data: List[Dict],
                       chat_contexts: Optional[Dict[Path, List[Dict]]] = None) -> List[Dict]:
        """
        Classify multiple chunks (Phase 2)
        
        Args:
            chunks_data: List of chunk data with transcript and chat info
            chat_contexts: Optional dict mapping chunk paths to chat snippets
            
        Returns:
            List of classification results
        """
        results = []
        
        for i, chunk_data in enumerate(chunks_data, 1):
            logger.info(f"Classifying chunk {i}/{len(chunks_data)}: {chunk_data['chunk_path']}")
            
            try:
                # Get chat context if available
                chat_snippet = []
                if chat_contexts:
                    chunk_path_str = chunk_data['chunk_path']
                    chat_snippet = chat_contexts.get(chunk_path_str, [])
                
                classification = self.classify_chunk(
                    chunk_path=Path(chunk_data['chunk_path']),
                    transcript=chunk_data['transcript'],
                    chat_snippet=chat_snippet,
                    chunk_start=chunk_data['start_time'],
                    chunk_end=chunk_data['end_time'],
                    vod_id=chunk_data['vod_id']
                )
                results.append(classification)
                
            except Exception as e:
                logger.error(f"Failed to classify chunk: {e}")
                results.append({
                    "start": "00:00:00",
                    "end": "00:00:00", 
                    "label": "other",
                    "score": 0.0,
                    "narrative_role": "filler",
                    "story_arc": "standalone",
                    "character_moments": [],
                    "transcript": "",
                    "keep": False,
                    "tags": ["error"],
                    "reasoning": f"Classification failed: {e}",
                    "chunk_path": chunk_data.get('chunk_path', ''),
                    "vod_id": chunk_data.get('vod_id', ''),
                    "classification_model": self.model
                })
        
        return results


# Backward compatibility alias
ClipClassifier = ContentClassifier 