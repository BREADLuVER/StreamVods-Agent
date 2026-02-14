"""
Main ClipPipeline class for orchestrating clip generation.
"""

from pathlib import Path
from typing import List, Optional

from .types import ClipCandidate, FinalClip, ClipManifestMeta, WindowDoc, format_hms
from .config import ClipConfig, DEFAULT_CONFIG
from .loader import load_docs, load_retriever, load_sponsor_spans
from .seeding import create_seed_groups, reaction_total
from .grouping import build_reaction_arcs, extend_groups
from .windowing import (
    apply_dynamic_padding, 
    snap_to_transcript_boundaries, 
    left_pad_to_sentence_start,
    apply_final_padding,
    bounds_of_indices
)
from .scoring import (
    compute_quality_score,
    looks_like_goodbye,
    low_energy_reject,
    long_windup_guard,
    anchor_center_for_group,
    build_preview_text
)
from .selection import deduplicate_and_select, append_sequence_numbers_to_adjacent_titles
from .title_llm import generate_titles_for_clips


class ClipPipeline:
    """Main pipeline for generating clips using non-LLM selection + LLM titles."""
    
    def __init__(self, vod_id: str, retriever=None, config: Optional[ClipConfig] = None):
        self.vod_id = vod_id
        self.retriever = retriever
        self.config = config or DEFAULT_CONFIG
        self.docs: Optional[List[WindowDoc]] = None
        self.sponsor_spans: List = []
    
    def load_data(self):
        """Load required data for processing."""
        if self.docs is None:
            self.docs = load_docs(self.vod_id)
        if not self.retriever:
            self.retriever = load_retriever(self.vod_id)
        if not self.sponsor_spans:
            self.sponsor_spans = load_sponsor_spans(self.vod_id)
    
    def generate_candidates(
        self, 
        top_k: int = 0, 
        use_semantics: bool = True
    ) -> List[ClipCandidate]:
        """Generate clip candidates using non-LLM selection."""
        self.load_data()
        
        # Calculate VOD duration
        vod_start = self.docs[0].start if self.docs else 0.0
        vod_end = self.docs[-1].end if self.docs else 0.0
        vod_duration_s = max(0.0, vod_end - vod_start)
        
        # Create seed groups
        seed_groups = create_seed_groups(self.docs, self.config, vod_duration_s)
        
        # Convert to base groups for arc building
        base_groups = [sg.indices for sg in seed_groups]
        
        # Build reaction arcs
        groups = build_reaction_arcs(
            self.docs, 
            base_groups,
            max_gap_s=self.config.max_gap_seconds,
            max_arc_dur=self.config.max_arc_duration
        )
        
        # Extend groups
        extended_groups = extend_groups(
            self.docs, 
            groups, 
            self.retriever, 
            self.config, 
            use_semantics
        )
        
        # Build candidates
        candidates: List[ClipCandidate] = []
        
        for g in extended_groups:
            if not g:
                continue
            
            g_sorted = sorted(g)
            g_docs = [self.docs[i] for i in g_sorted]
            chapter_id = g_docs[0].chapter_id if g_docs else "chapter_001"
            
            # Initial bounds
            start_guess, end_guess = bounds_of_indices(self.docs, g_sorted)
            
            # Apply generous padding
            pad_lo = max(0.0, start_guess - 120.0)
            pad_hi = end_guess + 120.0
            start_guess, end_guess = pad_lo, pad_hi
            
            # Dynamic padding
            start, end = apply_dynamic_padding(self.docs, chapter_id, start_guess, end_guess)
            
            # Compute anchor center
            anchor_center = anchor_center_for_group(self.docs, g_sorted)
            
            # Ensure anchor is inside and not too late
            win_len = min(180.0, max(30.0, end - start))
            if not (start <= anchor_center <= end):
                desired_start = anchor_center - (2.0 / 3.0) * win_len
                start = max(pad_lo, min(desired_start, pad_hi - win_len))
                end = start + win_len
            
            # Tighten window dynamically (target 35–120s) using signals
            ctx_docs = [d for d in self.docs if d.start < end and d.end > start]
            inner = ctx_docs
            if inner:
                # Mode majority inside window
                chat_count = sum(1 for d in inner if (d.mode or "").lower() == "chat")
                game_count = sum(1 for d in inner if (d.mode or "").lower() == "game")
                mode_major = "chat" if chat_count >= game_count else "game"
                high_energy_frac = sum(1 for d in inner if (d.energy or "").lower() == "high") / float(len(inner))
                mean_chat_z = sum(max(0.0, d.chat_rate_z) for d in inner) / max(1, len(inner))
                max_chat_z = max(max(0.0, d.chat_rate_z) for d in inner)
                total_reacts = sum(reaction_total(d) for d in inner)

                # Base targets by mode
                desired_len = self.config.expected_clip_chat if mode_major == "chat" else self.config.expected_clip_game
                
                # Boost for strong signals (shorter is usually punchier)
                if mean_chat_z >= 1.0 and max_chat_z >= 2.0:
                    desired_len -= self.config.high_signal_shorten
                if high_energy_frac >= 0.5:
                    desired_len -= self.config.energy_shorten
                if total_reacts >= 20:
                    desired_len -= self.config.reaction_shorten
                
                # Clamp within 35..120s
                if desired_len < 35.0:
                    desired_len = 35.0
                if desired_len > 120.0:
                    desired_len = 120.0

                # Pre/post allocation by mode
                pre_ratio = self.config.chat_pre_ratio if mode_major == "chat" else self.config.game_pre_ratio
                pre_len = pre_ratio * desired_len
                
                # Center around anchor
                vstart = anchor_center - pre_len
                vstart = max(start, min(vstart, end - desired_len))
                vend = vstart + desired_len

                # Snap to transcript boundaries and allow slight left pad
                vstart, vend = snap_to_transcript_boundaries(vstart, vend, ctx_docs, start, end, anchor_center)
                vstart, vend = left_pad_to_sentence_start(vstart, vend, ctx_docs, start, max_left_pad=20.0)

                # Final containment and duration
                if vend - vstart >= 30.0:
                    start, end = vstart, min(end, vstart + max(30.0, min(179.0, vend - vstart)))

            # Quality gates
            if looks_like_goodbye([d for d in self.docs if d.start < end and d.end > start]):
                continue
            if low_energy_reject(self.docs, start, end, self.config):
                continue

            score, mean_chat_z, total_reacts = compute_quality_score(self.docs, start, end)
            if score < self.config.min_score_threshold:
                continue

            # Long windup guard
            if long_windup_guard(anchor_center, start, end, self.config):
                continue

            preview = build_preview_text(self.docs, start, end, limit=160)
            candidates.append(ClipCandidate(
                vod_id=self.vod_id,
                start=round(float(start), 3),
                end=round(float(end), 3),
                duration=round(float(end - start), 3),
                start_hms=format_hms(start),
                end_hms=format_hms(end),
                anchor_time=round(float(max(start, min(anchor_center, end))), 3),
                anchor_time_hms=format_hms(max(start, min(anchor_center, end))),
                score=round(float(score), 4),
                mean_chat_z=round(float(mean_chat_z), 4),
                total_reactions=int(total_reacts),
                preview=preview,
            ))

        # Deduplicate and select
        dedup_spacing = max(45.0, min(90.0, int(round(0.7 * 60.0))))  # Use default expected length
        selected = deduplicate_and_select(
            candidates, 
            top_k=top_k, 
            iou_thr=self.config.dedup_iou_threshold, 
            min_spacing=dedup_spacing
        )
        
        return selected
    
    def finalize_clips(
        self, 
        candidates: List[ClipCandidate], 
        front_pad_s: float = 1.0, 
        back_pad_s: float = 1.0, 
        min_score: float = 2.0
    ) -> List[FinalClip]:
        """Finalize clips with padding and score filtering."""
        final_clips: List[FinalClip] = []
        
        for candidate in candidates:
            if candidate.score < min_score:
                continue
            
            # Apply final padding
            vstart, vend = apply_final_padding(
                candidate.start,
                candidate.end,
                front_pad_s,
                back_pad_s,
                hard_cap=179.0
            )
            
            # Re-snap to transcript boundaries if needed
            ctx_docs = [d for d in self.docs if d.start < vend and d.end > vstart]
            if ctx_docs:
                vstart, vend = snap_to_transcript_boundaries(
                    vstart, vend, ctx_docs, candidate.start, candidate.end, candidate.anchor_time
                )
            
            # Ensure anchor is still inside
            if not (vstart <= candidate.anchor_time <= vend):
                # Recenter if needed
                anchor_center = candidate.anchor_time
                win_len = min(179.0, max(30.0, vend - vstart))
                desired_start = anchor_center - (2.0 / 3.0) * win_len
                vstart = max(candidate.start, min(desired_start, candidate.end - win_len))
                vend = vstart + win_len
            
            final_clips.append(FinalClip(
                vod_id=candidate.vod_id,
                start=round(vstart, 3),
                end=round(vend, 3),
                duration=round(vend - vstart, 3),
                start_hms=format_hms(vstart),
                end_hms=format_hms(vend),
                anchor_time=round(candidate.anchor_time, 3),
                anchor_time_hms=candidate.anchor_time_hms,
                title="Highlight Clip",  # Will be filled by LLM
                score=candidate.score,
                rationale="",
                anchor_burst_id=candidate.anchor_burst_id,
            ))
        
        return final_clips
    
    def title_with_llm(self, final_clips: List[FinalClip], concurrent: bool = False) -> List[FinalClip]:
        """Generate titles for clips using LLM."""
        if not final_clips:
            return final_clips
        
        return generate_titles_for_clips(
            final_clips, 
            self.docs, 
            self.vod_id,
            concurrent=concurrent,
            max_workers=4
        )
    
    def write(self, final_clips: List[FinalClip], meta: ClipManifestMeta) -> Path:
        """Write manifest to file."""
        out_dir = Path(f"data/vector_stores/{self.vod_id}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "clips_manifest.json"
        
        # Number adjacent titles
        numbered_clips = append_sequence_numbers_to_adjacent_titles(list(final_clips), proximity_s=30.0)
        
        # Convert to dict format for JSON
        clips_data = []
        for clip in numbered_clips:
            clips_data.append({
                "vod_id": clip.vod_id,
                "start": clip.start,
                "end": clip.end,
                "duration": clip.duration,
                "start_hms": clip.start_hms,
                "end_hms": clip.end_hms,
                "anchor_time": clip.anchor_time,
                "anchor_time_hms": clip.anchor_time_hms,
                "title": clip.title,
                "score": clip.score,
                "rationale": clip.rationale,
                "anchor_burst_id": clip.anchor_burst_id,
            })
        
        manifest = {
            "vod_id": meta.vod_id,
            "total_candidates": meta.total_candidates,
            "total_selected": meta.total_selected,
            "min_score": meta.min_score,
            "front_pad_s": meta.front_pad_s,
            "back_pad_s": meta.back_pad_s,
            "clips": clips_data,
        }
        
        import json
        out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return out_path
    
    def run(
        self, 
        top_k: int = 0, 
        front_pad_s: float = 1.0, 
        back_pad_s: float = 1.0, 
        min_score: float = 2.0, 
        use_semantics: bool = True, 
        concurrent: bool = False
    ) -> Path:
        """Run the full pipeline."""
        # Generate candidates
        candidates = self.generate_candidates(top_k=top_k, use_semantics=use_semantics)
        
        # Finalize clips
        final_clips = self.finalize_clips(candidates, front_pad_s, back_pad_s, min_score)
        
        # Generate titles
        titled_clips = self.title_with_llm(final_clips, concurrent=concurrent)
        
        # Create metadata
        meta = ClipManifestMeta(
            vod_id=self.vod_id,
            total_candidates=len(candidates),
            total_selected=len(titled_clips),
            min_score=min_score,
            front_pad_s=front_pad_s,
            back_pad_s=back_pad_s,
        )
        
        # Write manifest
        return self.write(titled_clips, meta)
