"""Shared harness for the new-direction screens.

Everything reuses the judgment-unit protocol of the repository (``judgment_unit_study.Context``): frozen SimCLR features, one (pair, type) judgment per query, budgets of
10/20/40/60 judgments on top of the 10 initial groups, Splits A and B, the per-budget re-tuned head schedule, held-out image-disjoint test groups.  What this module adds:

* ``make_ctx``                    one seed's context with the repository's own parameters;
* ``fit_generic``                 the reward-head fit of ``frozen_encoder_reward_head.fit_head`` on arbitrary feature maps, optional extra (pseudo-labelled) rows;
* ``Scorer`` / ``evaluate``       held-out decisive-judgment metrics (accuracy, log-loss, AUC, per type) for any model that can produce a logit gap d = r_k(a) - r_k(b);
* ``random_labelled``             the shared random label sets on which learners are compared pairwise (same labels for every learner).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent / "pair_disjoint_not_image_disjoint"
sys.path.insert(0, str(STUDY)); sys.path.insert(0, str(HERE.parents[1] / "active_learning_program"))

import frozen_encoder_reward_head as frozen  # noqa: E402
import judgment_unit_study as ju_study  # noqa: E402
import run_pair_endpoint_study as single_study  # noqa: E402

BUDGETS = (10, 20, 40, 60)
TYPE_NAMES = {0: "t1x1", 2: "tc6x2", 3: "t13", 4: "htr"}
DEV_SEEDS = tuple(range(2000, 2020)); CONFIRM_SEEDS = tuple(range(3000, 3035))
USED_SEEDS_BY_EARLIER_WORK = set(range(400, 430)) | set(range(500, 535)) | set(range(600, 630)) | set(range(700, 1400)) | {42, 79, 123, 202, 303}
assert not (set(DEV_SEEDS) | set(CONFIRM_SEEDS)) & USED_SEEDS_BY_EARLIER_WORK


ALL_CACHE = HERE / "results" / "feature_cache_all.pt"


def feature_cache() -> dict:
    """SimCLR features of the images used by the studies (committed protocol cache) plus, if built, every trajectory and ideal image (``nd_temporal.py build``)."""
    cache = frozen.load_feature_cache(single_study.CACHE, single_study.DATA)
    if ALL_CACHE.exists(): cache.update(frozen.load_feature_cache(ALL_CACHE, single_study.DATA))
    return cache


def make_ctx(seed: int, split: str, scratch, cache: dict):
    """Context of the judgment-unit study with the repository's own head schedule (per budget, re-tuned)."""
    schedule = json.loads((single_study.OUT / "schedule.json").read_text()); table = ju_study.load_schedule(single_study.OUT / "schedule_judgment_unit.json")
    return ju_study.Context(seed, Path(scratch), cache, split, os.environ.get("ND_INITIAL", "groups"), schedule["learning_rate"], schedule["steps"], table)


def random_labelled(ctx, budget: int, draw: int) -> list:
    """Initial judgments plus ``budget`` judgments drawn uniformly from the pool; identical for every learner (seeded by seed, budget, draw)."""
    rng = np.random.default_rng(ctx.seed * 100003 + budget * 101 + draw)
    picks = [ctx.pool[i] for i in rng.choice(len(ctx.pool), size=budget, replace=False)]
    return list(ctx.initial) + picks


# ------------------------------------------------------------------------------------------ held-out evaluation
class TestSet:
    """Decisive judgments of the held-out groups: image paths, own type, orientation (+1 if image 1 won) and confidence weight."""
    def __init__(self, ctx, groups=None):
        rows = ctx.exp.rows_for(ctx.test if groups is None else groups); rows = rows[rows.Winner.isin(["1", "2"])].reset_index(drop=True)
        self.img1, self.img2 = list(rows.resolved_img1), list(rows.resolved_img2); self.k = rows.type_idx.to_numpy().astype(int)
        self.sign = np.where((rows.Winner == "1").to_numpy(), 1.0, -1.0); self.weight = rows.confidence_weight.to_numpy().astype(float)


def metrics_from_d(d: np.ndarray, test: TestSet) -> dict:
    d = np.asarray(d, dtype=float); s, w = test.sign, test.weight
    out = {"acc": float((w * ((d * s) > 0)).sum() / w.sum()), "ll": float((w * np.logaddexp(0.0, -s * d)).sum() / w.sum()),
           "auc": float(roc_auc_score(s > 0, d)) if len(set(s)) == 2 else float("nan"), "n_test": int(len(d))}
    for k, name in TYPE_NAMES.items():
        m = test.k == k; out[f"{name}_acc"] = float(((d[m] * s[m]) > 0).mean()) if m.any() else float("nan")
        out[f"{name}_ll"] = float(np.logaddexp(0.0, -s[m] * d[m]).mean()) if m.any() else float("nan")
    return out


def head_d(head: nn.Module, featfn, test: TestSet) -> np.ndarray:
    """Logit gap of the own-type output of a feed-forward head on arbitrary features."""
    head.eval()
    with torch.no_grad(): a, b = head(featfn(test.img1)), head(featfn(test.img2))
    return (a - b)[torch.arange(len(test.k)), torch.as_tensor(test.k)].numpy()


# ------------------------------------------------------------------------------------------ generic head fit (same loss as the repository)
def new_head(dim: int, seed: int, hidden: int = 256, dropout: float = .2) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden, 5))


def anchors(ctx, featfn):
    refs = {c: featfn(ps) for c, ps in ctx.exp.references.items()}; bad = featfn(ctx.exp.bad_paths) if ctx.exp.bad_paths else None
    return refs, bad


def fit_generic(ctx, judgments, featfn, dim: int, lr: float, steps: int, seed: int = 0, extra=None, rows=None, hidden: int = 256, anchor_weight: float = .25) -> nn.Module:
    """Train a head with exactly the repository's loss (``frozen.fit_head``) on ``featfn``-features of the rows of ``judgments`` (or of ``rows``).

    ``extra`` = optional (img1 paths, img2 paths, type_idx array, weight array, winner array) of additional (e.g. pseudo-labelled) rows."""
    rows = ju_study.rows_of(ctx.exp, judgments) if rows is None else rows
    head = new_head(dim, seed, hidden); refs, bad = anchors(ctx, featfn)
    if len(rows):
        xa, xb = featfn(list(rows.resolved_img1)), featfn(list(rows.resolved_img2)); typ = torch.as_tensor(rows.type_idx.to_numpy())
        weight, winner = torch.as_tensor(rows.confidence_weight.to_numpy(), dtype=torch.float32), rows.Winner.to_numpy()
    else:
        xa = xb = torch.empty(0, dim); typ = torch.empty(0, dtype=torch.long); weight = torch.empty(0); winner = np.array([], dtype=object)
    if extra is not None:
        ea, eb = featfn(list(extra[0])), featfn(list(extra[1]))
        xa, xb = torch.cat([xa, ea]), torch.cat([xb, eb]); typ = torch.cat([typ, torch.as_tensor(extra[2])])
        weight = torch.cat([weight, torch.as_tensor(extra[3], dtype=torch.float32)]); winner = np.concatenate([winner, extra[4]])
    frozen.fit_head(head, xa, xb, typ, weight, winner, refs, bad, lr, steps, ctx.exp.cfg.weight_decay, anchor_weight=anchor_weight, bad_weight=ctx.exp.cfg.bad_anchor_weight)
    return head.eval()


def identity_features(ctx):
    return lambda paths: ctx.features.get(paths)


def schedule_for_labelled(ctx, labelled) -> tuple[float, int]:
    return ctx.params_for(labelled)


def paired_table(frame: pd.DataFrame, baseline: str, metrics=("acc", "ll", "auc", "cal_ll")) -> pd.DataFrame:
    """Per (split, learner): mean gain over ``baseline`` (accuracy and AUC: learner - baseline; log-loss: baseline - learner, so positive = better) pooled over budgets and draws,
    then averaged over seeds, with the share of seeds that are better and a paired-bootstrap 95% interval over seeds."""
    rows = []
    for (split, learner), g in frame.groupby(["split", "learner"]):
        if learner == baseline: continue
        base = frame[(frame.split == split) & (frame.learner == baseline)]
        merged = g.merge(base, on=["seed", "budget", "draw"], suffixes=("", "_b"))
        row = {"split": split, "learner": learner, "n_seeds": merged.seed.nunique()}
        for m in metrics:
            if m not in merged or merged[m].isna().all(): continue
            sign = -1.0 if m in ("ll", "cal_ll") else 1.0
            per_seed = (sign * (merged[m] - merged[f"{m}_b"])).groupby(merged.seed).mean().to_numpy()
            rng = np.random.default_rng(0); boots = rng.choice(per_seed, size=(2000, len(per_seed))).mean(1)
            row[f"{m}_gain"] = per_seed.mean(); row[f"{m}_lo"], row[f"{m}_hi"] = np.percentile(boots, [2.5, 97.5]); row[f"{m}_share_better"] = float((per_seed > 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def calibrated_log_loss(d_val, val: "TestSet", d_test, test: "TestSet") -> float:
    """Test log-loss after fitting a single temperature on the validation groups (Split A only)."""
    temps = np.exp(np.linspace(np.log(.05), np.log(20), 120))
    losses = [float((val.weight * np.logaddexp(0.0, -val.sign * np.asarray(d_val) / t)).sum() / val.weight.sum()) for t in temps]; t = temps[int(np.argmin(losses))]
    return float((test.weight * np.logaddexp(0.0, -test.sign * np.asarray(d_test) / t)).sum() / test.weight.sum())
