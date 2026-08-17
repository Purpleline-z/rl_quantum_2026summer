#!/usr/bin/env python3
"""Validate one or more compact RHEED growth-session metadata ZIP archives."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from process_metadata_fusion_core import align_heartbeat_frames_causally, preflight_summary, read_session_bundle, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-archive", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = []
    for archive in args.session_archive:
        bundle = read_session_bundle(archive)
        aligned = align_heartbeat_frames_causally(bundle)
        summaries.append(preflight_summary(bundle, aligned))
        print(f"Validated {archive.name}: {len(aligned)} heartbeat frames, {len(bundle.sensor_rows)} sensor rows.", flush=True)
    write_json({"session_count": len(summaries), "sessions": summaries, "training_started": False}, args.output)


if __name__ == "__main__":
    main()

