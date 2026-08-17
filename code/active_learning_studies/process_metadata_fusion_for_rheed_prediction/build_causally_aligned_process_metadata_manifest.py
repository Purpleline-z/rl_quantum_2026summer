#!/usr/bin/env python3
"""Build a compact, causal frame-to-process-metadata manifest from session ZIPs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from process_metadata_fusion_core import align_heartbeat_frames_causally, read_session_bundle, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-archive", type=Path, action="append", required=True)
    parser.add_argument("--frame-path-mapping", type=Path, help="Explicit CSV with source_frame_path,resolved_image_path.")
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    mapping = pd.read_csv(args.frame_path_mapping) if args.frame_path_mapping else None
    frames = [align_heartbeat_frames_causally(read_session_bundle(path), mapping) for path in args.session_archive]
    manifest = pd.concat(frames, ignore_index=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest_output, index=False)
    write_json({"session_count": len(frames), "frame_count": int(len(manifest)),
                "unresolved_frame_count": int(manifest.resolved_image_path.isna().sum()),
                "future_sensor_match_count": int((manifest.source_sensor_timestamp > manifest.frame_timestamp).fillna(False).sum()),
                "training_started": False}, args.audit_output)
    print(f"Wrote causal manifest with {len(manifest)} frames; unresolved={manifest.resolved_image_path.isna().sum()}.", flush=True)


if __name__ == "__main__":
    main()

