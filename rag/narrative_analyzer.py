#!/usr/bin/env python3
"""
Narrative Analysis System

Processes video content in 1-hour chunks with context linking between chunks
to understand story arcs, narrative purposes, and content quality.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import os
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai_client import call_llm


class NarrativeAnalyzer:
    def __init__(self, vod_id: str):
        self.vod_id = vod_id
        self.chunk_duration = 1200
        self.previous_context = None
        
    def load_ai_data(self) -> List[Dict]:
        """Load the AI data for analysis."""
        data_file = Path(f"data/ai_data/{self.vod_id}/{self.vod_id}_ai_data.json")
        if not data_file.exists():
            raise FileNotFoundError(f"AI data not found: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data.get("segments", [])
    
    def split_into_chunks(self, segments: List[Dict]) -> List[List[Dict]]:
        """Split segments into 2-hour chunks."""
        chunks = []
        current_chunk = []
        chunk_start_time = None
        
        for segment in segments:
            start_time = segment.get("start_time", 0)
            
            # Start new chunk if this is the first segment or we've hit 2 hours
            if chunk_start_time is None or (start_time - chunk_start_time) >= self.chunk_duration:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [segment]
                chunk_start_time = start_time
            else:
                current_chunk.append(segment)
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def create_chunk_transcript(self, chunk: List[Dict]) -> str:
        """Create a readable transcript for the chunk."""
        transcript_parts = []
        
        for segment in chunk:
            start_time = segment.get("start_time", 0)
            transcript = segment.get("transcript", "").strip()
            chat_activity = segment.get("chat_activity", 0)
            
            if transcript:
                # Format: [timestamp] transcript (chat_activity: X)
                time_str = f"[{start_time//60:02d}:{start_time%60:02d}]"
                transcript_parts.append(f"{time_str} {transcript} (chat_activity: {chat_activity})")
        
        return "\n".join(transcript_parts)
    
    def analyze_chunk_narrative(self, chunk: List[Dict], chunk_index: int, total_chunks: int) -> Dict:
        """Analyze a single chunk for narrative understanding."""
        
        # Create transcript
        transcript = self.create_chunk_transcript(chunk)
        
        # Get chunk time range
        start_time = chunk[0].get("start_time", 0) if chunk else 0
        end_time = chunk[-1].get("end_time", 0) if chunk else 0
        
        # Build context prompt
        context_info = f"Chunk {chunk_index + 1}/{total_chunks} (Time: {start_time//60:02d}:{start_time%60:02d} - {end_time//60:02d}:{end_time%60:02d})"
        
        if self.previous_context:
            context_info += f"\n\nPrevious Context:\n{self.previous_context}"
        
        prompt = f"""
        You are an expert content analyst analyzing a 1-hour video chunk for GRANULAR moment-by-moment understanding.

        {context_info}

        TRANSCRIPT:
        {transcript}

        Analyze this chunk and provide DETAILED, SPECIFIC information:

        1. CONTENT TRANSITIONS: When does content switch between different modes (sponsor, gameplay, chat, just chatting, technical setup, AFK/break, music-only)? Provide exact timestamps.
        2. GAMEPLAY EVENTS: What specific game events happen (deaths, achievements, failures, level changes, boss fights, clutch plays, funny bugs, rage quits)?
        3. EMOTIONAL STATE CHANGES: When do emotions shift (frustration, excitement, humor, sarcasm, boredom, hype)? What triggers them?
        4. HIGH POINTS: What are the most engaging engaging moments, based on streamer energy or narrative flow?
        5. SPONSOR SEGMENTS: When do sponsor segments start/end? What’s being promoted? How is it delivered (casual mention, scripted ad, integrated demo)?
        6. GAMEPLAY SEGMENTS: Break down key stretches of play (exploration, grinding, combat, boss fight, tutorial section). Highlight big pivots in the game state.
        7. CHAT INTERACTIONS: When does chat engagement spike (spam, emotes, subs, raids, donations)? What triggered it?
        8. TECHNICAL/STREAM ISSUES: Note moments with audio/video problems, overlays breaking, streamer adjusting mic/camera, game crashes, OBS restarts.
        9. SOCIAL/COMMUNITY MOMENTS: Raids, streamer thanking subs/donors, community meme participation, streamer addressing drama or community updates.
        10. REACTION SEGMENTS: When the streamer watches or reacts to external content (YouTube clips, other streams, trailers). Provide timestamps and what’s being reacted to.
        11. PERSONAL/IRL MOMENTS: Streamer talks about personal life, reacts to news, shares stories, or goes off-topic.

        IMPORTANT: Return ONLY valid JSON. Be SPECIFIC and DETAILED. Include exact timestamps and concrete descriptions.

        {{
        "content_transitions": [
            {{"timestamp": "MM:SS", "from": "just chatting", "to": "gameplay", "description": "streamer wraps up story and loads into game"}}
        ],
        "gameplay_events": [
            {{"timestamp": "MM:SS", "event": "death", "description": "fell off cliff while distracted by chat", "context": "had just switched weapons"}},
            {{"timestamp": "MM:SS", "event": "achievement", "description": "unlocked rare mount", "context": "completed dungeon run"}}
        ],
        "emotional_changes": [
            {{"timestamp": "MM:SS", "emotion": "excitement", "trigger": "rare loot drop", "duration": "00:45"}}
        ],
        "high_points": [
            {{"timestamp": "MM:SS", "description": "streamer clutches a 1v3 fight", "reason": "chat explodes with pog emotes"}}
        ],
        "sponsor_segments": [
            {{"start": "MM:SS", "end": "MM:SS", "content": "promo", "quality": "medium"}}
        ],
        "gameplay_segments": [
            {{"start": "MM:SS", "end": "MM:SS", "game_state": "boss fight", "events": ["dodge roll", "final blow"]}}
        ],
        "chat_interactions": [
            {{"timestamp": "MM:SS", "activity": "high", "trigger": "big donation with funny message"}}
        ],
        "technical_issues": [
            {{"timestamp": "MM:SS", "issue": "audio cuts out", "duration": "00:30", "resolution": "streamer restarts mic"}}
        ],
        "community_moments": [
            {{"timestamp": "MM:SS", "event": "raid", "description": "another streamer raids with 200 viewers"}}
        ],
        "reaction_segments": [
            {{"timestamp": "MM:SS", "content": "YouTube clip", "description": "streamer laughs at meme video"}}
        ],
        "personal_moments": [
            {{"timestamp": "MM:SS", "topic": "personal story", "description": "talks about dog while waiting for matchmaking"}}
        ],
        "summary": "Detailed summary of this chunk with specific moments, transitions, streamer emotions, and community interactions."
        }}
        """
        
        try:
            response = call_llm(prompt, max_tokens=4000, temperature=0.3, request_tag=f"narrative_chunk_{chunk_index}")
            
            # Debug: print response
            print(f"  LLM Response: {response[:200]}...")
            
            if not response or response.strip() == "":
                raise ValueError("Empty response from LLM")
            
            # Try to extract JSON from response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            
            # Try to find the first complete JSON object
            try:
                # Find the first { and try to parse from there
                start_idx = response.find('{')
                if start_idx != -1:
                    # Try to find a complete JSON object
                    brace_count = 0
                    end_idx = start_idx
                    for i, char in enumerate(response[start_idx:], start_idx):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                    
                    if brace_count == 0:
                        response = response[start_idx:end_idx]
                    else:
                        # If incomplete, try to complete it
                        response = response[start_idx:] + '}'
                
                analysis = json.loads(response)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Response length: {len(response)}")
                print(f"Response: {response[:500]}...")
                raise
            
            # Add metadata
            analysis["chunk_index"] = chunk_index
            analysis["start_time"] = start_time
            analysis["end_time"] = end_time
            analysis["duration"] = end_time - start_time
            analysis["segment_count"] = len(chunk)
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing chunk {chunk_index}: {e}")
            print(f"Response was: {response[:500] if 'response' in locals() else 'No response'}")
            return {
                "chunk_index": chunk_index,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "segment_count": len(chunk),
                "error": str(e)
            }
    
    def link_chunks(self, chunk_analysis: Dict) -> str:
        """Create context for linking chunks."""
        if not chunk_analysis:
            return ""
        
        return f"""
Previous Chunk Summary:
- Narrative Arc: {chunk_analysis.get('narrative_arc', 'N/A')}
- Content Types: {', '.join(chunk_analysis.get('content_types', []))}
- Key Moments: {len(chunk_analysis.get('key_moments', []))} important moments
- Content Quality: {chunk_analysis.get('content_quality', 'N/A')}/10
- Summary: {chunk_analysis.get('summary', 'N/A')}
"""
    
    def analyze_full_vod(self) -> Dict:
        """Analyze the entire VOD with chunk linking."""
        print(f"Loading AI data for VOD {self.vod_id}...")
        segments = self.load_ai_data()
        
        print(f"Found {len(segments)} segments")
        print("Splitting into 1-hour chunks...")
        chunks = self.split_into_chunks(segments)
        
        print(f"Created {len(chunks)} chunks")
        
        analyses: List[Dict] = [None] * len(chunks)  # type: ignore

        # Concurrency controls
        concurrent = (os.getenv("NARRATIVE_CONCURRENT", "true").lower() in ("1", "true", "yes"))
        max_workers = max(1, int(os.getenv("NARRATIVE_MAX_WORKERS", os.getenv("LLM_MAX_PARALLEL", "4"))))

        if not concurrent or len(chunks) <= 1:
            self.previous_context = None
            for i, chunk in enumerate(chunks):
                print(f"\nAnalyzing chunk {i+1}/{len(chunks)}...")
                analysis = self.analyze_chunk_narrative(chunk, i, len(chunks))
                analyses[i] = analysis
                self.previous_context = self.link_chunks(analysis)
                print(f"  Chunk {i+1} analyzed: {analysis.get('summary', 'No summary')}")
        else:
            # Parallelize independent chunk analyses; link contexts after in order
            def _worker(args: Tuple[int, List[Dict]]):
                idx, ch = args
                return (idx, self.analyze_chunk_narrative(ch, idx, len(chunks)))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_worker, (i, chunk)) for i, chunk in enumerate(chunks)]
                for fut in futures:
                    idx, result = fut.result()
                    analyses[idx] = result

            # Now compute previous_context serially for deterministic linking
            self.previous_context = None
            for i in range(len(analyses)):
                print(f"\nLinking context for chunk {i+1}/{len(analyses)}...")
                # Update previous_context based on ordered analyses
                self.previous_context = self.link_chunks(analyses[i])
                print(f"  Chunk {i+1} analyzed: {analyses[i].get('summary', 'No summary')}")
        
        # Create final analysis
        final_analysis = {
            "vod_id": self.vod_id,
            "total_chunks": len(chunks),
            "total_duration": sum(a.get("duration", 0) for a in analyses),
            "chunks": analyses,
            "overall_narrative": self._create_overall_narrative(analyses)
        }
        
        return final_analysis
    
    def _create_overall_narrative(self, analyses: List[Dict]) -> Dict:
        """Create overall narrative understanding from all chunks."""
        if not analyses:
            return {}
        
        # Extract key information
        all_content_types = set()
        all_key_moments = []
        quality_scores = []
        
        for analysis in analyses:
            if "content_types" in analysis:
                all_content_types.update(analysis["content_types"])
            if "key_moments" in analysis:
                all_key_moments.extend(analysis["key_moments"])
            if "content_quality" in analysis:
                quality_scores.append(analysis["content_quality"])
        
        return {
            "total_content_types": list(all_content_types),
            "total_key_moments": len(all_key_moments),
            "average_quality": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "narrative_arcs": [a.get("narrative_arc", "") for a in analyses],
            "quality_distribution": {
                "high_quality_chunks": len([a for a in analyses if a.get("content_quality", 0) >= 8]),
                "medium_quality_chunks": len([a for a in analyses if 5 <= a.get("content_quality", 0) < 8]),
                "low_quality_chunks": len([a for a in analyses if a.get("content_quality", 0) < 5])
            }
        }


def main():
    """Test the narrative analysis system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze VOD narrative structure")
    parser.add_argument("vod_id", help="VOD ID to analyze")
    args = parser.parse_args()
    
    analyzer = NarrativeAnalyzer(args.vod_id)
    
    try:
        print(f"Starting narrative analysis for VOD {args.vod_id}...")
        analysis = analyzer.analyze_full_vod()
        
        # Save results
        output_file = Path(f"data/ai_data/{args.vod_id}/{args.vod_id}_narrative_analysis.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print("\n✅ Analysis complete!")
        print(f"Total chunks: {analysis['total_chunks']}")
        print(f"Total duration: {analysis['total_duration']//3600:.1f} hours")
        print(f"Results saved to: {output_file}")
        
        # Print summary
        overall = analysis.get("overall_narrative", {})
        print("\nOverall Narrative:")
        print(f"  Content types: {', '.join(overall.get('total_content_types', []))}")
        print(f"  Key moments: {overall.get('total_key_moments', 0)}")
        print(f"  Average quality: {overall.get('average_quality', 0):.1f}/10")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
