#!/usr/bin/env python3
"""Fail closed unless a causal manifest can support a run/image-disjoint study."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from process_metadata_fusion_core import MetadataSafetyError, build_run_disjoint_split, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--split-output", type=Path)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    try:
        split = build_run_disjoint_split(manifest)
    except MetadataSafetyError as exc:
        write_json({"eligible": False, "reason": str(exc), "training_started": False}, args.audit_output)
        raise SystemExit(f"Experiment capacity blocked: {exc}")
    if args.split_output is None:
        raise SystemExit("A successful capacity check requires --split-output.")
    args.split_output.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(args.split_output, index=False)
    write_json({"eligible": True, "run_count": int(split.run_id.nunique()), "training_started": False}, args.audit_output)
    print("Run- and image-disjoint capacity passed; no training is implemented by this scaffold.", flush=True)


if __name__ == "__main__":
    main()

