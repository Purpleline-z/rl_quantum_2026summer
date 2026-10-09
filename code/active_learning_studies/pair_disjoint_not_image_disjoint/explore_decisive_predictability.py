#!/usr/bin/env python3
"""How predictable is the outcome of a judgment (decisive vs tie / not_apply) from information available before asking?
Features: type, pair distance / cosine in the frozen feature space, and the head's own scores s_k(image) of both images for the queried type
(the head is trained on the initial judgments + anchors only, i.e. the information available at the start).  Group-wise cross-validated AUC on the pool.
"""
import sys, json, tempfile
from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import frozen_encoder_reward_head as frozen, judgment_unit_study as ju, run_pair_endpoint_study as ss
torch.set_num_threads(1)
cache = frozen.load_feature_cache(ss.CACHE, ss.DATA); schedule = json.loads((ss.OUT / "schedule.json").read_text()); table = ju.load_schedule("results/pair_endpoint_study/schedule_judgment_unit.json")
rows = []
with tempfile.TemporaryDirectory() as scratch:
    for split in "AB":
        for seed in range(600, 610):
            ctx = ju.Context(seed, Path(scratch), cache, split, "groups", schedule["learning_rate"], schedule["steps"], table)
            model = ctx.fit(ctx.initial); items = [ctx.item(j) for j in ctx.pool]; head = model.reward_head.eval()
            feats = ctx.features.embedding_cache(items)
            def s(img):
                with torch.no_grad(): return head[3](head[1](head[0](feats[img]))).numpy().astype(float)
            k = np.array([x["type_idx"] for x in items]); y = np.array([ctx.info[j]["decisive"] for j in ctx.pool]); grp = np.array([hash(j[0]) % 10**9 for j in ctx.pool])
            a = np.stack([s(x["img1"]) for x in items])[np.arange(len(k)), k]; b = np.stack([s(x["img2"]) for x in items])[np.arange(len(k)), k]
            f1 = np.stack([feats[x["img1"]].numpy() for x in items]); f2 = np.stack([feats[x["img2"]].numpy() for x in items])
            dist = np.linalg.norm(f1 - f2, axis=1); cos = (f1 * f2).sum(1) / (np.linalg.norm(f1, axis=1) * np.linalg.norm(f2, axis=1))
            onehot = np.eye(5)[k]
            sets = {"type only": onehot, "type + distance/cosine": np.column_stack([onehot, dist, cos]),
                    "type + head scores": np.column_stack([onehot, np.maximum(a, b), np.minimum(a, b), np.abs(a - b)]),
                    "all": np.column_stack([onehot, dist, cos, np.maximum(a, b), np.minimum(a, b), np.abs(a - b)])}
            for name, X in sets.items():
                X = (X - X.mean(0)) / (X.std(0) + 1e-6); pred = np.zeros(len(y))
                for tr, te in GroupKFold(5).split(X, y, grp): pred[te] = LogisticRegression(C=0.3, max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
                rows.append((split, seed, name, roc_auc_score(y, pred), y.mean()))
import pandas as pd
d = pd.DataFrame(rows, columns=["split", "seed", "features", "auc", "decisive_rate"])
print(d.groupby(["split", "features"]).auc.agg(["mean", "std"]).round(3)); print("decisive rate", d.decisive_rate.mean().round(3))
d.to_csv("results/new_methods/explore_decisive_predictability.csv", index=False)
