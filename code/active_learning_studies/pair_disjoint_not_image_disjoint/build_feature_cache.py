#!/usr/bin/env python3
"""Encode every image used by the frozen-encoder studies once and save the features (SimCLR checkpoint, no training)."""
import sys
import tempfile
from pathlib import Path

import torch

import frozen_encoder_reward_head as frozen
import run_frozen_encoder_task3 as harness

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "data"
OUT = HERE / "results" / "frozen_encoder_task3" / "simclr_feature_cache.pt"


def main() -> None:
    torch.set_num_threads(2); cache: dict = {}; scratch = Path(tempfile.mkdtemp())
    for seed in harness.ALL_SEEDS:
        exp = harness.make_experiment(seed, str(DATA), scratch); exp.load_and_split(); features = frozen.FrozenFeatures(exp, cache)
        paths = {p for g in exp.groups.values() for p in (g.iloc[0].resolved_img1, g.iloc[0].resolved_img2)}
        for table in (exp.references, exp.test_images, exp.utility_images): paths |= {str(p) for ps in table.values() for p in ps}
        paths |= {str(p) for p in exp.bad_paths}; features.get(sorted(paths)); print(seed, len(cache), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True); frozen.save_feature_cache(cache, OUT, DATA); print("saved", len(cache), OUT)


if __name__ == "__main__":
    main()
