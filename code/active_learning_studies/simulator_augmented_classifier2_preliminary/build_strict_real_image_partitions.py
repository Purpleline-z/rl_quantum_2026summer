#!/usr/bin/env python3
"""Build the fixed real image-ID train/validation/outer-test partition."""
from __future__ import annotations
import argparse
from pathlib import Path
from simulator_study_common import build_real_partitions, resolve_data_root
def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--drive-output",required=True,type=Path); parser.add_argument("--data-root")
    args=parser.parse_args(); print(build_real_partitions(resolve_data_root(args.data_root), args.drive_output / "preflight_audits")["status"])
if __name__ == "__main__": main()
