#!/usr/bin/env python3
"""What do the strategies select?  Outcome mix and image/type coverage of the acquired pair groups (single-shot cells)."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import run_frozen_encoder_task3 as harness
import run_pair_endpoint_study as study

OUT = study.OUT


def main() -> None:
    scratch = Path(tempfile.mkdtemp()); records = []
    for seed in harness.ALL_SEEDS:
        exp = harness.make_experiment(seed, str(study.DATA), scratch); exp.load_and_split()
        for path in sorted((OUT / "cells").glob(f"seed{seed}_budget*.json")):
            d = json.loads(path.read_text())
            if d["budget"] == 0: continue
            rows = pd.concat([exp.groups[i] for i in d["selected_pair_ids"]], ignore_index=True)
            images = {x for i in d["selected_pair_ids"] for x in (exp.groups[i].iloc[0].resolved_img1, exp.groups[i].iloc[0].resolved_img2)}
            records.append({"strategy": "random" if d["strategy"].startswith("random_r") else d["strategy"], "seed": seed, "budget": d["budget"],
                            "rows_per_group": len(rows) / d["budget"], "decisive": rows.Winner.isin(["1", "2"]).mean(), "tie": (rows.Winner == "tie").mean(),
                            "not_apply": (rows.Winner == "not_apply").mean(), "htr_share": (rows.canonical_type == "HTR").mean(),
                            "distinct_images_per_group": len(images) / d["budget"], "log_loss": d["test_decisive_log_loss"]})
    frame = pd.DataFrame(records); frame.to_csv(OUT / "selected_pair_composition.csv", index=False)
    table = frame.groupby("strategy")[["rows_per_group", "decisive", "tie", "not_apply", "htr_share", "log_loss"]].mean().sort_values("log_loss")
    print(table.round(3).to_string())
    print("\ncorrelation across (strategy x seed x budget) cells between selected decisive fraction and held-out log-loss:",
          round(float(frame.decisive.corr(frame.log_loss)), 3), " rows per group:", round(float(frame.rows_per_group.corr(frame.log_loss)), 3))


if __name__ == "__main__":
    main()
