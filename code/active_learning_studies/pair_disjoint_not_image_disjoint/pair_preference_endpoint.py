"""Held-out pair-preference endpoint.

Type accuracy on ideal images is driven by the reference anchors (anchors alone reach ~0.85), so it cannot tell acquisition rules apart.
The quantity pairwise labels actually teach is the *preference*: which of two images better shows a reconstruction.  This module builds an
image-disjoint set of held-out pair groups and scores a reward model on its decisive judgments.

Split (per seed): of the groups left after the 10 initial ones, 20 validation and 40 test groups are drawn at random (mutually image-disjoint);
every other group that shares an image with them is removed from training/candidate use, so no evaluation image is ever seen in training.
"""
from __future__ import annotations

import random

import numpy as np
import torch

from pairwise_active_learning_pipeline import Experiment


def split_heldout(exp: Experiment, initial: list[str], candidates: list[str], n_validation: int = 20, n_test: int = 40, salt: int = 17):
    """Return (initial, pool, validation, test).  No image is shared between {validation, test} and (initial + pool), and none between validation and test."""
    images = {pid: {exp.groups[pid].iloc[0].resolved_img1, exp.groups[pid].iloc[0].resolved_img2} for pid in initial + candidates}
    initial_images = {x for i in initial for x in images[i]}
    rng = random.Random(exp.cfg.seed * 1000 + salt); shuffled = candidates[:]; rng.shuffle(shuffled)
    chosen: list[str] = []; used: set[str] = set()
    for pid in shuffled:
        if len(chosen) >= n_validation + n_test: break
        if images[pid] & initial_images or images[pid] & used: continue
        chosen.append(pid); used |= images[pid]
    validation, test = chosen[:n_validation], chosen[n_validation:]
    pool = [pid for pid in candidates if pid not in chosen and not (images[pid] & used)]
    return initial, pool, validation, test


@torch.no_grad()
def evaluate_preferences(exp: Experiment, features, model, heldout: list[str]) -> dict:
    """Decisive-row accuracy and Bradley--Terry log-loss on the held-out groups, using each row's own reconstruction-type head."""
    rows = exp.rows_for(heldout); rows = rows[rows.Winner.isin(["1", "2"])]
    if rows.empty: return {"decisive_accuracy": float("nan"), "decisive_log_loss": float("nan"), "decisive_rows": 0}
    head = model.reward_head.eval(); a, b = head(features.get(rows.resolved_img1)), head(features.get(rows.resolved_img2))
    index = torch.arange(len(rows)); typ = torch.as_tensor(rows.type_idx.to_numpy()); d = (a - b)[index, typ]
    sign = torch.as_tensor((rows.Winner == "1").to_numpy()) * 2 - 1.0; weight = torch.as_tensor(rows.confidence_weight.to_numpy(), dtype=torch.float32)
    correct = ((d * sign) > 0).float(); loss = torch.nn.functional.softplus(-d * sign)
    return {"decisive_accuracy": float((correct * weight).sum() / weight.sum()), "decisive_log_loss": float((loss * weight).sum() / weight.sum()),
            "decisive_rows": int(len(rows))}
