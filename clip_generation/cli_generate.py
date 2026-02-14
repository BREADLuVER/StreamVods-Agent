#!/usr/bin/env python3
"""
CLI for generating clips using the new OOP pipeline.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from clip_generation.pipeline import ClipPipeline
from clip_generation.config import ClipConfig


def main():
    parser = argparse.ArgumentParser(description="Generate clips using OOP pipeline")
    parser.add_argument("vod_id", help="VOD ID")
    parser.add_argument("--top-k", type=int, default=0, help="Max number of clips to select (0 for dynamic)")
    parser.add_argument("--min-score", type=float, default=4.0, help="Minimum score threshold")
    parser.add_argument("--front-pad", type=float, default=10.0, help="Front padding in seconds")
    parser.add_argument("--back-pad", type=float, default=1.0, help="Back padding in seconds")
    parser.add_argument("--no-semantics", action="store_true", help="Disable semantic extension")
    parser.add_argument("--concurrent", action="store_true", help="Enable concurrent processing")
    parser.add_argument("--dry-run", action="store_true", help="Print candidates without writing manifest")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Create pipeline
    config = ClipConfig()
    pipeline = ClipPipeline(args.vod_id, config=config)
    
    try:
        if args.dry_run:
            # Just generate and print candidates
            candidates = pipeline.generate_candidates(
                top_k=args.top_k,
                use_semantics=not args.no_semantics
            )
            
            print(f"Generated {len(candidates)} candidates:")
            for c in candidates:
                print(
                    f"{c.start_hms} -> {c.end_hms}  (dur={int(c.duration)}s, score={c.score}) | "
                    f"anchor={c.anchor_time_hms} | {c.preview}"
                )
        else:
            # Run full pipeline
            manifest_path = pipeline.run(
                top_k=args.top_k,
                front_pad_s=args.front_pad,
                back_pad_s=args.back_pad,
                min_score=args.min_score,
                use_semantics=not args.no_semantics,
                concurrent=args.concurrent
            )
            
            print(f"Clips manifest written: {manifest_path}")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
