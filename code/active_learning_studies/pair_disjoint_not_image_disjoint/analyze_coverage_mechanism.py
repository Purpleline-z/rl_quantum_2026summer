#!/usr/bin/env python3
"""Does coverage of the held-out pairs explain the held-out log-loss of a selected set?  (single-shot cells, five study seeds)"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import frozen_encoder_reward_head as frozen
import new_pair_strategies as nps
import pair_preference_endpoint as endpoint
import run_frozen_encoder_task3 as harness
import run_pair_endpoint_study as study


def main() -> None:
    scratch = Path(tempfile.mkdtemp()); cache = frozen.load_feature_cache(study.CACHE, study.DATA); records = []
    for seed in harness.ALL_SEEDS:
        exp, features, initial, pool, validation, test = study.setup(seed, scratch, cache)
        item = lambda i: {"pair_id": i, "img1": exp.groups[i].iloc[0].resolved_img1, "img2": exp.groups[i].iloc[0].resolved_img2}
        cache_full = features.embedding_cache([item(i) for i in initial + pool + test])
        everything = [item(i) for i in initial + pool + test]; vectors, _ = nps.pair_features(everything, cache_full, "relation")
        index = {x["pair_id"]: k for k, x in enumerate(everything)}; test_vectors = vectors[[index[i] for i in test]]
        for path in sorted((study.OUT / "cells").glob(f"seed{seed}_budget*.json")):
            d = json.loads(path.read_text())
            if d["budget"] == 0: continue
            chosen = vectors[[index[i] for i in initial + d["selected_pair_ids"]]]
            nearest = np.sqrt(((test_vectors[:, None] - chosen[None]) ** 2).sum(2)).min(1).mean()
            records.append({"seed": seed, "budget": d["budget"], "strategy": "random" if d["strategy"].startswith("random_r") else d["strategy"],
                            "test_coverage_distance": nearest, "log_loss": d["test_decisive_log_loss"]})
    frame = pd.DataFrame(records); frame.to_csv(study.OUT / "coverage_mechanism_cells.csv", index=False)
    # standardise within (seed, budget) so budget and split difficulty do not drive the correlation
    for column in ("test_coverage_distance", "log_loss"): frame[column + "_z"] = frame.groupby(["seed", "budget"])[column].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    print("within (seed, budget) correlation between mean distance of test pairs to the nearest selected pair and held-out log-loss:",
          round(float(frame.test_coverage_distance_z.corr(frame.log_loss_z)), 3), f"(n={len(frame)} cells)")
    print(frame.groupby("strategy")[["test_coverage_distance", "log_loss"]].mean().sort_values("log_loss").round(3).to_string())


if __name__ == "__main__":
    main()
