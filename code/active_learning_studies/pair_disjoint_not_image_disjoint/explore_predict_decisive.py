#!/usr/bin/env python3
"""Can image features predict whether a judgment is decisive (winner 1/2) or not (tie / not_apply)?  Group-wise cross-validated AUC."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import frozen_encoder_reward_head as frozen
import run_frozen_encoder_task3 as harness

HERE = Path(__file__).resolve().parent; DATA = HERE.parents[2] / "data"; CACHE = harness.OUT / "simclr_feature_cache.pt"


def main() -> None:
    exp = harness.make_experiment(42, str(DATA), Path(tempfile.mkdtemp())); exp.load_and_split(); cache = frozen.load_feature_cache(CACHE, DATA)
    rows = pd.concat(list(exp.groups.values()), ignore_index=True); features = frozen.FrozenFeatures(exp, cache)
    a, b = features.get(rows.resolved_img1).numpy(), features.get(rows.resolved_img2).numpy()
    distance = np.linalg.norm(a - b, axis=1); cosine = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    typ = np.eye(5)[rows.type_idx.to_numpy()]; y = rows.Winner.isin(["1", "2"]).to_numpy().astype(int); tie = (rows.Winner == "tie").to_numpy().astype(int); na = (rows.Winner == "not_apply").to_numpy().astype(int)
    groups = rows.pair_id.to_numpy()
    print(f"rows {len(rows)}: decisive {y.mean():.2f}, tie {tie.mean():.2f}, not_apply {na.mean():.2f}")
    sets = {"type only": typ, "type + distance + cosine": np.column_stack([typ, distance, cosine]),
            "mean + |a-b| (1024-d, z-scored)": np.column_stack([typ, (np.hstack([(a + b) / 2, np.abs(a - b)]) - 0) / 1.0])}
    for target_name, target in (("decisive", y), ("not_apply", na), ("tie", tie)):
        for name, X in sets.items():
            X = (X - X.mean(0)) / (X.std(0) + 1e-6); scores = np.zeros(len(rows))
            for train, test in GroupKFold(5).split(X, target, groups):
                scores[test] = LogisticRegression(C=0.05, max_iter=2000).fit(X[train], target[train]).predict_proba(X[test])[:, 1]
            print(f"  predict {target_name:9s} from {name:34s} AUC {roc_auc_score(target, scores):.3f}")
    print("outcome by type:"); print(pd.crosstab(rows.canonical_type, rows.Winner, normalize="index").round(2))


if __name__ == "__main__":
    main()
