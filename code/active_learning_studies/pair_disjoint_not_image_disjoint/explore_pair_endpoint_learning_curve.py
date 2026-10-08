#!/usr/bin/env python3
"""Does the held-out pair-preference endpoint respond to the number of acquired groups (random selection)?"""
import tempfile
from pathlib import Path

import pandas as pd
import torch

import frozen_encoder_reward_head as frozen
import pair_preference_endpoint as endpoint
import run_frozen_encoder_task3 as harness

HERE = Path(__file__).resolve().parent; DATA = HERE.parents[2] / "data"; CACHE = harness.OUT / "simclr_feature_cache.pt"


def main() -> None:
    torch.set_num_threads(1); scratch = Path(tempfile.mkdtemp()); cache = frozen.load_feature_cache(CACHE, DATA); rows = []
    for seed in harness.ALL_SEEDS:
        exp = harness.make_experiment(seed, str(DATA), scratch); initial, candidates = exp.load_and_split(); features = frozen.FrozenFeatures(exp, cache)
        initial, pool, heldout = endpoint.split_heldout(exp, initial, candidates)
        print(f"seed {seed}: initial {len(initial)}, pool {len(pool)}, held-out {len(heldout)}", flush=True)
        order = harness.reference_batch(pool, seed, len(pool))
        for n in (0, 10, 25, 50, 75, len(pool)):
            for anchor in (.25, 0.0):
                for head_seed in range(3):
                    model = frozen.train_model(exp, features, initial + order[:n], 3e-3, 300, head_seed=head_seed, anchor_weight=anchor)
                    rows.append({"seed": seed, "acquired": n, "anchor_weight": anchor, "head_seed": head_seed, **endpoint.evaluate_preferences(exp, features, model, heldout),
                                 "type_accuracy": frozen.evaluate_model(exp, features, model)["test_accuracy"]})
    frame = pd.DataFrame(rows); frame.to_csv(harness.OUT / "pair_endpoint_learning_curve.csv", index=False)
    print(frame.groupby(["anchor_weight", "acquired"])[["decisive_accuracy", "decisive_log_loss", "type_accuracy"]].mean().round(3))


if __name__ == "__main__":
    main()
