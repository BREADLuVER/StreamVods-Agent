#!/usr/bin/env python3
"""
Generate and save a Director's Cut title for a VOD.

Usage:
  python -m directors_cut.generate_director_cut_name <vod_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path as _Path

# Allow running as a script (python directors_cut/generate_director_cut_name.py) or module (-m directors_cut.generate_director_cut_name)
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from directors_cut.title import generate_title, save_title


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Director's Cut title")
    parser.add_argument("vod_id")
    args = parser.parse_args()

    title = generate_title(args.vod_id)
    out = save_title(args.vod_id, title)
    print(f"✅ Director's Cut title saved: {out}\n✨ {title}")


if __name__ == "__main__":
    main()


