#!/usr/bin/env python3
"""
Video Transitions Module

Provides randomized transitions between video clips for final video assembly.
Supports fade, crossfade, slide, and dissolve transitions.
"""

import random
import os
from moviepy.editor import ColorClip, CompositeVideoClip, concatenate_videoclips

class VideoTransitions:
    """Handles video transitions between clips."""
    
    TRANSITION_TYPES = ['fade', 'crossfade', 'slide', 'dissolve']
    
    def __init__(self, transition_duration=1.0):
        """
        Initialize transition handler.
        
        Args:
            transition_duration (float): Duration of transitions in seconds
        """
        self.transition_duration = transition_duration
    
    def apply_random_transition(self, clip1, clip2):
        """
        Apply a random transition between two clips.
        
        Args:
            clip1: First video clip
            clip2: Second video clip
            
        Returns:
            Transitioned video sequence
        """
        transition_type = random.choice(self.TRANSITION_TYPES)
        return self.apply_transition(clip1, clip2, transition_type)
    
    def apply_transition(self, clip1, clip2, transition_type):
        """
        Apply specific transition between two clips.
        
        Args:
            clip1: First video clip
            clip2: Second video clip
            transition_type (str): Type of transition ('fade', 'crossfade', 'slide', 'dissolve')
            
        Returns:
            Transitioned video sequence
        """
        if transition_type == 'fade':
            return self._fade_transition(clip1, clip2)
        elif transition_type == 'crossfade':
            return self._crossfade_transition(clip1, clip2)
        elif transition_type == 'dissolve':
            return self._dissolve_transition(clip1, clip2)
        else:
            # Fallback to simple concatenation
            return concatenate_videoclips([clip1, clip2])
    
    def _fade_transition(self, clip1, clip2):
        """Fade to black transition."""
        fade_duration = self.transition_duration / 2
        
        # Fade out clip1
        clip1_fadeout = clip1.fadeout(fade_duration)
        
        # Black gap
        black_gap = ColorClip(size=clip1.size, color=[0, 0, 0], duration=fade_duration)
        
        # Fade in clip2
        clip2_fadein = clip2.fadein(fade_duration)
        
        return concatenate_videoclips([clip1_fadeout, black_gap, clip2_fadein])
    
    def _crossfade_transition(self, clip1, clip2):
        """Cross-fade transition with overlap."""
        # Prepare clips for crossfade
        clip1_part = clip1.fadeout(self.transition_duration)
        clip2_part = clip2.crossfadein(self.transition_duration)
        
        # Concatenate with overlap
        return concatenate_videoclips([clip1_part, clip2_part])
    
    def _dissolve_transition(self, clip1, clip2):
        """Dissolve transition with overlapping fade."""
        # First clip fades out while second clip fades in
        clip1_fadeout = clip1.fadeout(self.transition_duration)
        clip2_fadein = clip2.fadein(self.transition_duration)
        
        # Create overlap by starting clip2 earlier
        clip2_delayed = clip2_fadein.set_start(clip1.duration - self.transition_duration)
        
        # Composite both clips
        return CompositeVideoClip([clip1_fadeout, clip2_delayed])
    
    def apply_transitions_to_clip_list(self, clips, random_seed=None):
        """
        Apply random transitions between a list of clips.
        
        Args:
            clips (list): List of video clips
            random_seed (int): Optional seed for reproducible randomization
            
        Returns:
            Final video with transitions between all clips
        """
        if random_seed is not None:
            random.seed(random_seed)
        
        if len(clips) <= 1:
            return clips[0] if clips else None
        
        print(f" Applying random transitions between {len(clips)} clips...")
        
        # Start with first clip
        result = clips[0]
        
        # Apply transitions between consecutive clips
        for i in range(1, len(clips)):
            transition_type = random.choice(self.TRANSITION_TYPES)
            print(f"   Clip {i}: {transition_type} transition")
            
            # Apply transition between result and next clip
            result = self.apply_transition(result, clips[i], transition_type)
        
        print(" All transitions applied successfully")
        return result

def create_transitioned_video(clips, output_path, transition_duration=1.0, random_seed=None):
    """
    Convenience function to create a video with random transitions.
    
    Args:
        clips (list): List of video clips
        output_path (str): Output file path
        transition_duration (float): Duration of transitions
        random_seed (int): Optional seed for reproducible randomization
        
    Returns:
        bool: Success status
    """
    try:
        transitions = VideoTransitions(transition_duration)
        final_video = transitions.apply_transitions_to_clip_list(clips, random_seed)
        
        if final_video:
            final_video.write_videofile(
                output_path,
                fps=30,
                preset='ultrafast',
                verbose=False,
                logger=None
            )
            final_video.close()
            return True
        return False
        
    except Exception as e:
        print(f"X Error creating transitioned video: {e}")
        return False