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
from sklearn.metrics import roc_auc_score

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


def split_classifier2_style(exp: Experiment, initial: list[str], candidates: list[str], test_fraction: float = .2, salt: int = 29):
    """Pair-level hold-out as in classifier2 (``train_unified.load_data``: shuffle the unique pairs, hold out 20%), keeping the identity-safe rule.

    20% of all usable pair groups form the test set (every judgment of a pair stays on the same side).  Groups that share an image with a test group are
    dropped from training, so no test image is seen in training.  The initial groups are re-chosen from what remains with the study's rule (a random draw
    that covers every reconstruction type first); there is no validation set, which classifier2 did not have either.  Returns (initial, pool, [], test)."""
    all_ids = list(dict.fromkeys(list(initial) + list(candidates))); images = {pid: {exp.groups[pid].iloc[0].resolved_img1, exp.groups[pid].iloc[0].resolved_img2} for pid in all_ids}
    rng = random.Random(exp.cfg.seed * 1000 + salt); shuffled = all_ids[:]; rng.shuffle(shuffled)
    test = shuffled[:int(round(test_fraction * len(all_ids)))]; test_images = set().union(*(images[i] for i in test))
    remaining = [i for i in shuffled[len(test):] if not (images[i] & test_images)]
    chosen, covered = [], set()
    for pid in remaining:  # greedy reconstruction-type coverage, then fill with random groups (same rule as Experiment.load_and_split)
        types = set(exp.groups[pid].canonical_type)
        if types - covered: chosen.append(pid); covered |= types
    for pid in remaining:
        if len(chosen) >= len(initial): break
        if pid not in chosen: chosen.append(pid)
    chosen = chosen[:len(initial)]
    return chosen, [i for i in remaining if i not in chosen], [], test


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


@torch.no_grad()
def preference_scores(exp: Experiment, features, model, groups: list[str]):
    """Signed logit gap d = r_a - r_b (own type head) for every decisive row, its orientation (+1 if image 1 won) and confidence weight."""
    rows = exp.rows_for(groups); rows = rows[rows.Winner.isin(["1", "2"])]
    head = model.reward_head.eval(); a, b = head(features.get(rows.resolved_img1)), head(features.get(rows.resolved_img2))
    d = (a - b)[torch.arange(len(rows)), torch.as_tensor(rows.type_idx.to_numpy())].numpy()
    return d, np.where((rows.Winner == "1").to_numpy(), 1.0, -1.0), rows.confidence_weight.to_numpy().astype(float)


def _log_loss(d, sign, weight, temperature=1.0):
    return float((weight * np.logaddexp(0.0, -sign * d / temperature)).sum() / weight.sum())


def evaluate_full(exp: Experiment, features, model, validation: list[str], test: list[str]) -> dict:
    """Test metrics on the held-out groups: accuracy, log-loss, AUC (scale-free) and log-loss after fitting one temperature on the validation groups."""
    dt, st, wt = preference_scores(exp, features, model, test)
    if validation:
        dv, sv, wv = preference_scores(exp, features, model, validation)
        temperatures = np.exp(np.linspace(np.log(.05), np.log(20), 120)); best = temperatures[int(np.argmin([_log_loss(dv, sv, wv, t) for t in temperatures]))]
    else: dv = None; best = float("nan")
    return {"test_decisive_accuracy": float((wt * ((dt * st) > 0)).sum() / wt.sum()), "test_decisive_log_loss": _log_loss(dt, st, wt),
            "test_decisive_auc": float(roc_auc_score(st > 0, dt)) if len(set(st)) == 2 else float("nan"),
            "test_calibrated_log_loss": _log_loss(dt, st, wt, best) if validation else float("nan"), "fitted_temperature": float(best), "test_decisive_rows": int(len(dt)),
            "validation_decisive_log_loss": _log_loss(dv, sv, wv) if validation else float("nan"),
            "validation_decisive_accuracy": float((wv * ((dv * sv) > 0)).sum() / wv.sum()) if validation else float("nan")}
