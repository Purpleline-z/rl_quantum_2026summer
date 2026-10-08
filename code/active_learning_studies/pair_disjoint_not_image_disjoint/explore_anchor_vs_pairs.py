#!/usr/bin/env python3
"""Where does the accuracy come from?  Decompose the frozen-head classifier into reference anchors and pairwise preferences."""
import tempfile
from pathlib import Path

import pandas as pd
import torch

import frozen_encoder_reward_head as frozen
import run_frozen_encoder_task3 as harness

HERE = Path(__file__).resolve().parent; DATA = HERE.parents[2] / "data"; CACHE = harness.OUT / "simclr_feature_cache.pt"


def main() -> None:
    torch.set_num_threads(1); scratch = Path(tempfile.mkdtemp()); cache = frozen.load_feature_cache(CACHE, DATA); rows = []
    for seed in harness.ALL_SEEDS:
        exp = harness.make_experiment(seed, str(DATA), scratch); initial, candidates = exp.load_and_split(); features = frozen.FrozenFeatures(exp, cache)
        settings = {"anchors only (0 pair groups)": ([], .25), "pairs only, 10 groups": (initial, 0.0), "pairs only, 110 groups": (initial + candidates[:100], 0.0),
                    "pairs only, all 168 groups": (initial + candidates, 0.0), "anchors + 10 groups": (initial, .25), "anchors + 110 groups": (initial + candidates[:100], .25)}
        for name, (ids, anchor) in settings.items():
            for head_seed in range(4):
                model = frozen.train_model(exp, features, ids, 3e-3, 300, head_seed=head_seed, anchor_weight=anchor)
                rows.append({"seed": seed, "setting": name, "head_seed": head_seed, "outer": frozen.evaluate_model(exp, features, model)["test_accuracy"],
                             "validation": frozen.evaluate_model(exp, features, model, "utility_validation")["test_accuracy"]})
        print("seed", seed, flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(harness.OUT / "anchor_vs_pairs_decomposition.csv", index=False)
    print(frame.groupby("setting", sort=False)[["validation", "outer"]].agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
