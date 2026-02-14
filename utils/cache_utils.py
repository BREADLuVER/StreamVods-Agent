#!/usr/bin/env python3
"""
Cache utilities for StreamSniped workflow
Provides helper functions for cache management
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is on sys.path for absolute imports
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import factory via package path; fallback to shim if needed
try:
    from utils.cache_manager import create_cache_manager  # type: ignore
except Exception:
    from cache_manager import create_cache_manager  # type: ignore


def add_cache_arguments(parser: argparse.ArgumentParser):
    """Add cache-related command line arguments to parser"""
    cache_group = parser.add_argument_group('Cache Options')
    cache_group.add_argument(
        '--force-regenerate',
        action='store_true',
        help='Force regeneration of all cached data'
    )
    cache_group.add_argument(
        '--skip-cache',
        action='store_true',
        help='Skip cache checks entirely'
    )
    cache_group.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear all cache for this VOD before processing'
    )
    # Cache max age removed - cache is valid as long as files exist
    # S3 cleanup is managed externally by design
    cache_group.add_argument(
        '--partial-regenerate',
        nargs='+',
        help='Force regenerate specific steps only'
    )


def should_skip_with_cache(
    vod_id: str,
    step_name: str,
    output_files: List[str],
    dependencies: List[str] | None = None,
    args: Optional[argparse.Namespace] = None,
    debug: bool = False,
    **kwargs,
) -> bool:
    """
    Check if step should be skipped based on cache and command line arguments
    
    Args:
        vod_id: VOD ID
        step_name: Name of the workflow step
        output_files: List of expected output files
        dependencies: List of dependency files
        args: Command line arguments namespace
        debug: Enable debug logging
    
    Returns:
        True if step should be skipped, False otherwise
    """
    # Resolve flags from either argparse Namespace or explicit keyword args
    skip_cache_flag = (getattr(args, 'skip_cache', False) if args else False) or kwargs.get('skip_cache', False)
    force_regenerate_flag = (getattr(args, 'force_regenerate', False) if args else False) or kwargs.get('force_regenerate', False)
    partial_regenerate = (getattr(args, 'partial_regenerate', []) if args else kwargs.get('partial_regenerate', [])) or []
    # Max age no longer used - cache is valid as long as files exist

    # If user explicitly asked to skip cache, do not skip the step (i.e., re-run)
    if skip_cache_flag:
        return False

    # Respect force or partial regenerate requests
    if force_regenerate_flag or step_name in partial_regenerate:
        return False
    

    
    # Enable debug mode if requested
    debug = debug or kwargs.get('debug', False)
    
    if debug:
        print(f"🔍 DEBUG: Cache check for step '{step_name}' with {len(output_files)} output files:")
        for i, file in enumerate(output_files):
            print(f"   {i+1}. {file}")
    
    # Check cache
    cache_manager = create_cache_manager(vod_id)

    # Ensure we receive a (bool, reason) tuple from should_skip_step
    result = cache_manager.should_skip_step(
        step_name=step_name,
        output_files=output_files,
        dependencies=dependencies,
        force_regenerate=force_regenerate_flag
    )

    # Backward safety: normalize to (bool, str)
    if isinstance(result, tuple) and len(result) == 2:
        should_skip, reason = result
    else:
        # If older versions returned only a bool, coerce to tuple
        should_skip = bool(result)
        reason = ""

    if debug:
        print(f"🔍 DEBUG: Cache check result: should_skip={should_skip}, reason='{reason}'")

    if should_skip:
        print(f"⏭️ Skipping {step_name}: {reason}")
        return True
    else:
        if debug:
            print(f"🚫 DEBUG: NOT skipping {step_name}: {reason}")
    
    return False


def mark_step_completed(vod_id: str, step_name: str, output_files: List[str],
                       dependencies: List[str] = None, **kwargs):
    """Mark step as completed in cache"""
    cache_manager = create_cache_manager(vod_id)
    cache_manager.mark_step_completed(
        step_name=step_name,
        output_files=output_files,
        dependencies=dependencies,
        **kwargs
    )


def get_cached_outputs(vod_id: str, step_name: str, **kwargs) -> List[str]:
    """Get cached output files for a step"""
    cache_manager = create_cache_manager(vod_id)
    return cache_manager.get_cached_outputs(step_name, **kwargs)


def clear_cache(vod_id: str, step_name: str = None):
    """Clear cache for VOD"""
    cache_manager = create_cache_manager(vod_id)
    cache_manager.clear_cache(step_name)


def get_cache_status(vod_id: str) -> dict:
    """Get cache status for VOD"""
    cache_manager = create_cache_manager(vod_id)
    return cache_manager.get_cache_status()


def main():
    """Command line interface for cache management"""
    parser = argparse.ArgumentParser(description='StreamSniped Cache Management')
    parser.add_argument('vod_id', help='VOD ID to manage cache for')
    parser.add_argument('action', choices=['status', 'clear', 'clear-step'], 
                       help='Action to perform')
    parser.add_argument('--step', help='Step name for clear-step action')
    
    args = parser.parse_args()
    
    if args.action == 'status':
        status = get_cache_status(args.vod_id)
        print(f"Cache Status for VOD {args.vod_id}:")
        print(f"  Total cached steps: {status['total_cached_steps']}")
        print(f"  Cache directory: {status['cache_dir']}")
        if status['cached_steps']:
            print("  Cached steps:")
            for step in status['cached_steps']:
                print(f"    - {step}")
        else:
            print("  No cached steps")
    
    elif args.action == 'clear':
        clear_cache(args.vod_id)
        print(f"Cleared all cache for VOD {args.vod_id}")
    
    elif args.action == 'clear-step':
        if not args.step:
            print("Error: --step required for clear-step action")
            sys.exit(1)
        clear_cache(args.vod_id, args.step)
        print(f"Cleared cache for step '{args.step}' in VOD {args.vod_id}")


if __name__ == "__main__":
    main() 