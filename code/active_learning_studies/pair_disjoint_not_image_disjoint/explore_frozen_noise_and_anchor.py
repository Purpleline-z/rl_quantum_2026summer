#!/usr/bin/env python3
"""Cheap CPU diagnostics on cached frozen features: (1) residual run-to-run noise, (2) effect of the reference-anchor weight."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import frozen_encoder_reward_head as frozen
import run_frozen_encoder_task3 as harness

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "data"; CACHE = harness.OUT / "simclr_feature_cache.pt"


def main() -> None:
    torch.set_num_threads(1); scratch = Path(tempfile.mkdtemp()); cache = frozen.load_feature_cache(CACHE, DATA); rows = []
    for seed in harness.ALL_SEEDS:
        exp = harness.make_experiment(seed, str(DATA), scratch); initial, candidates = exp.load_and_split(); features = frozen.FrozenFeatures(exp, cache)
        ids = initial + harness.reference_batch(candidates, seed, 100)
        for anchor in (.25, 1.0, 4.0):
            for head_seed in range(8):
                model = frozen.train_model(exp, features, ids, 3e-3, 300, head_seed=head_seed, anchor_weight=anchor)
                rows.append({"seed": seed, "anchor_weight": anchor, "head_seed": head_seed,
                             "validation": frozen.evaluate_model(exp, features, model, "utility_validation")["test_accuracy"],
                             "outer": frozen.evaluate_model(exp, features, model)["test_accuracy"]})
        print("seed", seed, flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(harness.OUT / "noise_and_anchor_diagnostic.csv", index=False)
    print(frame.groupby("anchor_weight")[["validation", "outer"]].agg(["mean", "std"]).round(3))
    print("within-seed sd of outer accuracy across head initialisations (anchor .25):",
          frame[frame.anchor_weight == .25].groupby("seed").outer.std().round(3).to_dict())
    print("between-seed sd of the within-seed mean:", round(frame[frame.anchor_weight == .25].groupby("seed").outer.mean().std(), 3))


if __name__ == "__main__":
    main()
