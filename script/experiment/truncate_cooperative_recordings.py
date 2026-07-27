#!/usr/bin/env python3
"""将 cooperative monostatic 每次采样截断为前 N 帧 CPI。

示例::

    python script/experiment/truncate_cooperative_recordings.py \\
        --input-dir data/experiment/cooperative_monostatic \\
        --max-frames 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isac_imp.record_target_metadata import (
    DEFAULT_COOPERATIVE_VLEN,
    truncate_cooperative_recordings,
)


def _default_input_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "experiment" / "cooperative_monostatic"


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Truncate cooperative monostatic divide_profiles to first N CPI frames"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_default_input_dir(),
        help="directory with target_positions.csv and dev0/dev1/",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=50,
        help="number of CPI frames to keep per session (default: 50)",
    )
    parser.add_argument(
        "--vlen",
        type=int,
        default=DEFAULT_COOPERATIVE_VLEN,
        help=f"complex profile length per CPI frame (default: {DEFAULT_COOPERATIVE_VLEN})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report files that would be truncated without modifying",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable tqdm progress bar",
    )
    return parser.parse_args()


def main() -> None:
    args = argument_parser()
    sessions, files = truncate_cooperative_recordings(
        args.input_dir.resolve(),
        max_frames=args.max_frames,
        vlen=args.vlen,
        dry_run=args.dry_run,
        show_progress=not args.no_progress,
    )
    action = "would process" if args.dry_run else "processed"
    print(f"{action} {sessions} sessions, {files} files truncated to {args.max_frames} frames")


if __name__ == "__main__":
    main()
