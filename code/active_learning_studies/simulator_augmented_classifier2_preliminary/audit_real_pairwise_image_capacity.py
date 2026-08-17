#!/usr/bin/env python3
"""Audit real pairwise rows after the strict image-ID partition is fixed."""
from __future__ import annotations
import argparse
from pathlib import Path
from simulator_study_common import atomic_json_write, load_partition_table, load_real_training_inputs, resolve_data_root
def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--drive-output",required=True,type=Path); parser.add_argument("--data-root")
    args=parser.parse_args(); root=resolve_data_root(args.data_root); pairs, anchors, bad, trajectories=load_real_training_inputs(root, load_partition_table(args.drive_output))
    result={"status":"pass","pairwise_training_rows":len(pairs),"pairwise_training_images":len(trajectories),"training_anchor_counts":{key:len(value) for key,value in anchors.items()},"bad_training_images":len(bad),"pairwise_outer_holdout_status":"not_available: real outer test is ideal-image-only to retain strict image disjointness"}
    atomic_json_write(result,args.drive_output/"preflight_audits"/"real_pairwise_capacity_audit.json"); print(result["status"])
if __name__ == "__main__": main()
