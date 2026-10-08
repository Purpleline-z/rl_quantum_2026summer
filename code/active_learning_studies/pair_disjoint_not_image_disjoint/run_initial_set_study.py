#!/usr/bin/env python3
"""Cold start: does the choice of the first 10 labelled pair groups matter?  (label-free selection, frozen features, held-out pair endpoint)

Initial sets are chosen from all groups that do not touch the validation/test images, WITHOUT looking at any label:
  default     the study's original rule (random draw with greedy reconstruction-type coverage; uses labels' types)
  random      10 groups drawn uniformly
  kmeans      pair nearest to each of 10 k-means centres in the relation-aware pair space
  typiclust   TypiClust on pair vectors (densest pair of each of the 10 largest clusters)
  farthest    farthest-first traversal from the pair-space centroid
Each initial set is then used alone (budget 0) and extended by 10 / 30 random pool groups.
"""
from __future__ import annotations

import json
import random as pyrandom
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

import frozen_encoder_reward_head as frozen
import new_pair_strategies as nps
import pair_preference_endpoint as endpoint
import run_pair_endpoint_study as study

OUT = study.OUT; N = 10


def choose(method, items, vectors, seed, default_initial):
    rng = pyrandom.Random(seed * 31 + 7)
    if method == "default": return list(default_initial)
    if method == "random": return [items[i]["pair_id"] for i in rng.sample(range(len(items)), N)]
    if method == "kmeans":
        centres = KMeans(n_clusters=N, n_init=5, random_state=seed).fit(vectors).cluster_centers_
        chosen = []
        for c in centres:
            for i in np.argsort(((vectors - c) ** 2).sum(1)):
                if items[int(i)]["pair_id"] not in chosen: chosen.append(items[int(i)]["pair_id"]); break
        return chosen
    if method == "typiclust":
        labels = KMeans(n_clusters=N, n_init=5, random_state=seed).fit_predict(vectors); chosen = []
        for c in np.argsort(-np.bincount(labels))[:N]:
            members = np.where(labels == c)[0]; sub = vectors[members]; k = min(10, len(members) - 1)
            density = 1 / (np.sort(np.sqrt(((sub[:, None] - sub[None]) ** 2).sum(2)), 1)[:, 1:k + 1].mean(1) + 1e-9) if k > 0 else np.ones(len(members))
            chosen.append(items[int(members[int(np.argmax(density))])]["pair_id"])
        return chosen
    if method == "farthest":
        picked = [int(np.argmin(((vectors - vectors.mean(0)) ** 2).sum(1)))]
        while len(picked) < N: picked.append(int(np.argmax(np.sqrt(((vectors[:, None] - vectors[picked][None]) ** 2).sum(2)).min(1))))
        return [items[i]["pair_id"] for i in picked]
    raise ValueError(method)


def main() -> None:
    torch.set_num_threads(2); cache = frozen.load_feature_cache(study.CACHE, study.DATA); schedule = json.loads((OUT / "schedule.json").read_text()); records = []
    with tempfile.TemporaryDirectory() as scratch:
        for seed in study.seeds():
            exp, features, default_initial, pool, validation, test = study.setup(seed, Path(scratch), cache)
            item = lambda i: {"pair_id": i, "img1": exp.groups[i].iloc[0].resolved_img1, "img2": exp.groups[i].iloc[0].resolved_img2}
            universe = list(default_initial) + list(pool); items = [item(i) for i in universe]
            vectors, _ = nps.pair_features(items, features.embedding_cache(items), "relation")
            for method in ("default", "random", "kmeans", "typiclust", "farthest"):
                initial = choose(method, items, vectors, seed, default_initial); rest = [i for i in universe if i not in initial]
                extra = pyrandom.Random(seed * 17 + 3).sample(rest, 30)
                for added in (0, 10, 30):
                    model = frozen.train_model(exp, features, initial + extra[:added], schedule["learning_rate"], schedule["steps"])
                    records.append({"seed": seed, "method": method, "added_random_groups": added, **endpoint.evaluate_full(exp, features, model, validation, test)})
            print("finished", seed, flush=True)
    frame = pd.DataFrame(records); frame.to_csv(OUT / "initial_set_study.csv", index=False)
    for metric in ("test_decisive_log_loss", "test_calibrated_log_loss", "test_decisive_auc", "test_decisive_accuracy"):
        print("\n" + metric); print(frame.pivot_table(index="method", columns="added_random_groups", values=metric).round(3).to_string())


if __name__ == "__main__":
    main()
