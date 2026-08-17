#!/usr/bin/env python3
"""Validate the p656 synthetic archive and write a compact Drive audit JSON."""
from __future__ import annotations
import argparse
from pathlib import Path
from simulator_study_common import atomic_json_write, validate_synthetic_archive

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-archive", required=True, type=Path)
    parser.add_argument("--drive-output", required=True, type=Path)
    parser.add_argument("--skip-image-hash-verification", action="store_true")
    args = parser.parse_args()
    result = validate_synthetic_archive(args.synthetic_archive, not args.skip_image_hash_verification)
    atomic_json_write(result, args.drive_output / "preflight_audits" / "synthetic_archive_validation.json")
    print(result["status"])
if __name__ == "__main__": main()
