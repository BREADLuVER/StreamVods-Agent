#!/usr/bin/env python3
"""
Preview LLM-generated clip titles for one or more VOD IDs.

Usage examples:
  python processing-scripts/preview_titles.py 2587445207
  python processing-scripts/preview_titles.py 2587445207 2587445208 --topk 8 --concurrent
  TITLE_PREVIEW_VODS=2587445207,2587445208 python processing-scripts/preview_titles.py --limit 2

Environment vars:
  TITLE_PREVIEW_VODS           Comma-separated VOD IDs (overrides discovery)
  TITLE_PREVIEW_LIMIT          Limit number of discovered VOD IDs (default 3)
  TITLE_PREVIEW_TOPK           Number of candidates to generate (default 5)
  TITLE_PREVIEW_CONCURRENT     1/true to title concurrently (default false)
  TITLE_PREVIEW_USE_SEMANTICS  0/false to disable semantic extension (default true)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List


# Ensure project root is importable when run from anywhere
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clip_generation.pipeline import ClipPipeline  # noqa: E402


def discover_vod_ids(limit: int) -> List[str]:
    env_ids = os.getenv("TITLE_PREVIEW_VODS")
    if env_ids:
        ids = [vid.strip() for vid in env_ids.split(",") if vid.strip()]
        return ids[:limit] if limit > 0 else ids

    base = Path("data/vector_stores")
    if not base.exists():
        return []
    vod_ids = [p.name for p in sorted(base.iterdir()) if p.is_dir()]
    return vod_ids[:limit] if limit > 0 else vod_ids


def bool_flag(value: str, default: bool) -> bool:
    if value is None:
        return default
    v = str(value).strip().lower()
    return v not in ("0", "false", "no", "")


def preview_for_vod(vod_id: str, top_k: int, concurrent: bool, use_semantics: bool) -> None:
    pipeline = ClipPipeline(vod_id)
    candidates = pipeline.generate_candidates(top_k=top_k, use_semantics=use_semantics)
    final_clips = pipeline.finalize_clips(
        candidates,
        front_pad_s=1.0,
        back_pad_s=1.0,
        min_score=2.0,
    )
    titled_clips = pipeline.title_with_llm(final_clips, concurrent=concurrent)

    print(f"\n=== VOD {vod_id} ({len(titled_clips)} clips) ===")
    for idx, clip in enumerate(titled_clips, 1):
        print(f"{idx:02d}. {clip.start_hms}-{clip.end_hms} | {int(clip.anchor_time)}s | {clip.title}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview LLM titles for VODs")
    parser.add_argument("vod_ids", nargs="*", help="VOD IDs to preview; if omitted, discover from data/vector_stores")
    parser.add_argument("--limit", type=int, default=int(os.getenv("TITLE_PREVIEW_LIMIT", "3")), help="Limit number of discovered VODs")
    parser.add_argument("--topk", type=int, default=int(os.getenv("TITLE_PREVIEW_TOPK", "5")), help="Number of candidates to generate")
    parser.add_argument("--concurrent", action="store_true", help="Generate titles concurrently")
    parser.add_argument("--no-semantics", dest="use_semantics", action="store_false", help="Disable semantic extension")
    parser.set_defaults(use_semantics=bool_flag(os.getenv("TITLE_PREVIEW_USE_SEMANTICS", "1"), True))

    args = parser.parse_args()

    # If --concurrent not passed, read from env
    concurrent_env = os.getenv("TITLE_PREVIEW_CONCURRENT")
    if not args.concurrent and concurrent_env is not None:
        args.concurrent = bool_flag(concurrent_env, False)

    vod_ids: List[str] = args.vod_ids or discover_vod_ids(args.limit)
    if not vod_ids:
        print("No VOD IDs provided or discovered. Provide IDs or set TITLE_PREVIEW_VODS.")
        sys.exit(1)

    print(f"Previewing titles for {len(vod_ids)} VOD(s). top_k={args.topk}, concurrent={args.concurrent}, semantics={args.use_semantics}")

    for vod_id in vod_ids:
        try:
            preview_for_vod(vod_id, top_k=args.topk, concurrent=args.concurrent, use_semantics=args.use_semantics)
        except Exception as exc:
            print(f"[ERROR] Failed for VOD {vod_id}: {exc}")


if __name__ == "__main__":
    main()


