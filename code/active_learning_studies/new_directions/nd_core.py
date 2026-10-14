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
DEV_SEEDS = tuple(range(2000, 2020)); CONFIRM_SEEDS = tuple(range(3000, 3035)) + tuple(range(4000, 4035)) + tuple(range(5000, 5035)) + tuple(range(6000, 6035))
USED_SEEDS_BY_EARLIER_WORK = set(range(400, 430)) | set(range(500, 535)) | set(range(600, 630)) | set(range(700, 1400)) | {42, 79, 123, 202, 303}
assert not (set(DEV_SEEDS) | set(CONFIRM_SEEDS)) & USED_SEEDS_BY_EARLIER_WORK

# graph-learner study (branch claude/graph-learner-*): its own seed blocks; everything used before (including this module's 2000-6034 blocks and 400-459, 700-1739 of the graph / judgment-unit work) is excluded
ND_SEEDS_ALL = set(DEV_SEEDS) | set(CONFIRM_SEEDS)
USED_BEFORE_GRAPH_LEARNER = USED_SEEDS_BY_EARLIER_WORK | ND_SEEDS_ALL | set(range(400, 460)) | set(range(700, 1740)) | set(range(2000, 2020)) | set(range(3000, 6035))
GL_DEV_SEEDS = tuple(range(7000, 7020)); GL_CONFIRM1 = tuple(range(7100, 7135)); GL_CONFIRM2 = tuple(range(7200, 7235)); GL_COLD = tuple(range(7300, 7335))
GL_SEEDS_ALL = set(GL_DEV_SEEDS) | set(GL_CONFIRM1) | set(GL_CONFIRM2) | set(GL_COLD)
assert not GL_SEEDS_ALL & USED_BEFORE_GRAPH_LEARNER and len(GL_SEEDS_ALL) == 20 + 3 * 35
USED_SEEDS_BY_EARLIER_WORK = USED_SEEDS_BY_EARLIER_WORK | GL_SEEDS_ALL   # later studies must not reuse the graph-learner blocks
# --- graph-mechanism study (claude/graph-mechanism-*): own seed blocks; other parallel sessions use 8200-8599
GM_DEV_SEEDS = tuple(range(8000, 8020)); GM_CONFIRM1 = tuple(range(8100, 8135)); GM_CONFIRM2 = tuple(range(8150, 8185)); GM_SEEDS_ALL = set(GM_DEV_SEEDS) | set(GM_CONFIRM1) | set(GM_CONFIRM2)
assert not GM_SEEDS_ALL & USED_SEEDS_BY_EARLIER_WORK and not GM_SEEDS_ALL & set(range(7000, 7336)) and len(GM_SEEDS_ALL) == 20 + 2 * 35 and not GM_SEEDS_ALL & set(range(8200, 8600))
USED_SEEDS_BY_EARLIER_WORK = USED_SEEDS_BY_EARLIER_WORK | GM_SEEDS_ALL   # graph-mechanism blocks (8000-8019, 8100-8134, 8150-8184)

# selection x graph-learner x HTR study (branch claude/selection-htr-*): dev 8200-8219, confirmation 1 = 8300-8334, confirmation 2 = 8350-8384.  The parallel studies of the same round own 8000-8199 and 8400-8599.
SH_DEV_SEEDS = tuple(range(8200, 8220)); SH_CONFIRM1 = tuple(range(8300, 8335)); SH_CONFIRM2 = tuple(range(8350, 8385))
SH_COLD = tuple(range(8385, 8400))   # cold-start confirmation of the learner family (15 seeds; pre-registered in PREREGISTRATION_SH2 before it was run)
SH_SEEDS_ALL = set(SH_DEV_SEEDS) | set(SH_CONFIRM1) | set(SH_CONFIRM2) | set(SH_COLD)
assert not SH_SEEDS_ALL & USED_SEEDS_BY_EARLIER_WORK and len(SH_SEEDS_ALL) == 20 + 2 * 35 + 15
USED_SEEDS_BY_EARLIER_WORK = USED_SEEDS_BY_EARLIER_WORK | SH_SEEDS_ALL

# red-team audit (branch claude/redteam-*): development 8800-8819, confirmation 8850-8884 (the other blocks of 8000-8999 belong to parallel studies)
RT_DEV_SEEDS = tuple(range(8800, 8820)); RT_CONFIRM = tuple(range(8850, 8885)); RT_SEEDS_ALL = set(RT_DEV_SEEDS) | set(RT_CONFIRM)
assert not RT_SEEDS_ALL & USED_SEEDS_BY_EARLIER_WORK and len(RT_SEEDS_ALL) == 20 + 35
USED_SEEDS_BY_EARLIER_WORK = USED_SEEDS_BY_EARLIER_WORK | RT_SEEDS_ALL

# regulariser study G2 (branch claude/regularisers-*): development 9000-9019, confirmation 9050-9084.  8000-8199, 8400-8799 and 8885-8999 belong to parallel studies and are not touched here.
G2_DEV_SEEDS = tuple(range(9000, 9020)); G2_CONFIRM = tuple(range(9050, 9085)); G2_SEEDS_ALL = set(G2_DEV_SEEDS) | set(G2_CONFIRM)
assert not G2_SEEDS_ALL & USED_SEEDS_BY_EARLIER_WORK and len(G2_SEEDS_ALL) == 20 + 35
assert not G2_SEEDS_ALL & (set(range(8000, 8200)) | set(range(8400, 8800)) | set(range(8885, 9000)))
USED_SEEDS_BY_EARLIER_WORK = USED_SEEDS_BY_EARLIER_WORK | G2_SEEDS_ALL
# recipe search (branch claude/recipe-*): development 9300-9319, confirmation 9350-9384 (nothing else in 9000-9999 is claimed by this study; 8000-8199, 8400-8799 and 8885-8999 stay reserved for parallel studies)
RC_DEV_SEEDS = tuple(range(9300, 9320)); RC_CONFIRM = tuple(range(9350, 9385)); RC_SEEDS_ALL = set(RC_DEV_SEEDS) | set(RC_CONFIRM)
assert not RC_SEEDS_ALL & USED_SEEDS_BY_EARLIER_WORK and len(RC_SEEDS_ALL) == 20 + 35
assert not RC_SEEDS_ALL & (set(range(8000, 8200)) | set(range(8400, 8800)) | set(range(8885, 9000)))
USED_SEEDS_BY_EARLIER_WORK = USED_SEEDS_BY_EARLIER_WORK | RC_SEEDS_ALL
# temporal-graph study (branch claude/temporal-graph-*): development 9200-9219, confirmation 9250-9284 (block 9200-9299 is this study's; spare seeds 9220-9249 / 9285-9299 are for its own follow-ups only).
# Blocks of the parallel conversations of the same round (reserved, never used here): 8000-8199, 8400-8799, 8885-8999.  Already used by earlier studies: 7000-7335, 8200-8399, 8800-8884 (and everything above).
TG_DEV_SEEDS = tuple(range(9200, 9220)); TG_CONFIRM = tuple(range(9250, 9285)); TG_SEEDS_ALL = set(TG_DEV_SEEDS) | set(TG_CONFIRM)
TG_RESERVED_PARALLEL = set(range(8000, 8200)) | set(range(8400, 8800)) | set(range(8885, 9000))
assert not TG_SEEDS_ALL & USED_SEEDS_BY_EARLIER_WORK and not TG_SEEDS_ALL & TG_RESERVED_PARALLEL and len(TG_SEEDS_ALL) == 20 + 35
USED_SEEDS_BY_EARLIER_WORK = USED_SEEDS_BY_EARLIER_WORK | TG_SEEDS_ALL
# calibration study (branch claude/calibration-*): development 9100-9119, confirmation 9150-9184 (the blocks 8000-8199, 8400-8799, 8885-8999 belong to parallel studies)
CAL_DEV_SEEDS = tuple(range(9100, 9120)); CAL_CONFIRM = tuple(range(9150, 9185)); CAL_SEEDS_ALL = set(CAL_DEV_SEEDS) | set(CAL_CONFIRM)
assert not CAL_SEEDS_ALL & USED_SEEDS_BY_EARLIER_WORK and len(CAL_SEEDS_ALL) == 20 + 35
assert not CAL_SEEDS_ALL & (set(range(8000, 8200)) | set(range(8400, 8800)) | set(range(8885, 9000)))
USED_SEEDS_BY_EARLIER_WORK = USED_SEEDS_BY_EARLIER_WORK | CAL_SEEDS_ALL


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


def anchors(ctx, featfn, all_ideal: bool = False):
    """Reference (ideal-image) anchors per class; ``all_ideal`` also adds the utility-validation and outer-test ideal images of the seed's split (their class labels are absolute labels the lab
    owns; the preference endpoint evaluates held-out PAIR groups, never ideal-image classification, and ideal images are excluded from the pair universe by content identity)."""
    refs = {c: featfn(list(ps) + (list(ctx.exp.utility_images.get(c, [])) + list(ctx.exp.test_images.get(c, [])) if all_ideal else [])) for c, ps in ctx.exp.references.items()}; bad = featfn(ctx.exp.bad_paths) if ctx.exp.bad_paths else None
    return refs, bad


def fit_generic(ctx, judgments, featfn, dim: int, lr: float, steps: int, seed: int = 0, extra=None, rows=None, hidden: int = 256, anchor_weight: float = .25, bad_weight=None, all_ideal: bool = False, weight_decay=None) -> nn.Module:
    """Train a head with exactly the repository's loss (``frozen.fit_head``) on ``featfn``-features of the rows of ``judgments`` (or of ``rows``).

    ``extra`` = optional (img1 paths, img2 paths, type_idx array, weight array, winner array) of additional (e.g. pseudo-labelled) rows."""
    rows = ju_study.rows_of(ctx.exp, judgments) if rows is None else rows
    head = new_head(dim, seed, hidden); refs, bad = anchors(ctx, featfn, all_ideal)
    if len(rows):
        xa, xb = featfn(list(rows.resolved_img1)), featfn(list(rows.resolved_img2)); typ = torch.as_tensor(rows.type_idx.to_numpy())
        weight, winner = torch.as_tensor(rows.confidence_weight.to_numpy(), dtype=torch.float32), rows.Winner.to_numpy()
    else:
        xa = xb = torch.empty(0, dim); typ = torch.empty(0, dtype=torch.long); weight = torch.empty(0); winner = np.array([], dtype=object)
    if extra is not None:
        ea, eb = featfn(list(extra[0])), featfn(list(extra[1]))
        xa, xb = torch.cat([xa, ea]), torch.cat([xb, eb]); typ = torch.cat([typ, torch.as_tensor(extra[2])])
        weight = torch.cat([weight, torch.as_tensor(extra[3], dtype=torch.float32)]); winner = np.concatenate([winner, extra[4]])
    frozen.fit_head(head, xa, xb, typ, weight, winner, refs, bad, lr, steps, ctx.exp.cfg.weight_decay if weight_decay is None else weight_decay, anchor_weight=anchor_weight, bad_weight=ctx.exp.cfg.bad_anchor_weight if bad_weight is None else bad_weight)
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


# ------------------------------------------------------------------------------------------ head fit with an extra mixture-consistency anchor loss
def fit_head_ext(head, xa, xb, type_index, weight, winner, refs, bad, lr, steps, weight_decay=1e-4, anchor_weight=.25, bad_weight=.10, mix_weight=0.0, mix_pairs=64, mix_seed=0, lam_hi=.8, lam_lo=.3, ce_weight=0.0):
    """``frozen.fit_head`` plus a mixture-consistency term: real RHEED patterns are close to linear mixtures of ideal patterns, so for two ideal images a (type A) and b (type B) the score of type A must
    increase with the share of a in the mixture.  Mixtures are formed in feature space; loss = -log sigmoid(s_A(lam_hi a + (1-lam_hi) b) - s_A(lam_lo a + (1-lam_lo) b)) and the same for B with the roles reversed."""
    import torch.nn.functional as F
    head.eval(); optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    index = torch.arange(len(type_index)); masks = {label: torch.as_tensor(winner == label) for label in frozen.LABELS}; gen = torch.Generator().manual_seed(mix_seed)
    names = list(refs); cols = {c: frozen.TYPE_TO_INDEX[c] for c in names}
    for _ in range(steps):
        a, b = head(xa)[index, type_index], head(xb)[index, type_index]
        terms = (-F.logsigmoid(a - b) * masks["1"] - F.logsigmoid(b - a) * masks["2"] + (a - b).abs() * masks["tie"] + (F.relu(a) + F.relu(b)) * masks["not_apply"])
        loss = (terms * weight).sum() / max(len(type_index), 1)
        if refs and len(refs) > 1:
            scores = {n: head(f) for n, f in refs.items()}; values = []
            for p_, sp in scores.items():
                for o_, so in scores.items():
                    if o_ != p_: values.append(-F.logsigmoid(sp[:, cols[p_]][:, None] - so[:, cols[p_]][None, :]).mean())
            loss = loss + anchor_weight * torch.stack(values).mean()
        if mix_weight > 0 and refs and len(refs) > 1:
            ca = torch.randint(len(names), (mix_pairs,), generator=gen); cb = (ca + torch.randint(1, len(names), (mix_pairs,), generator=gen)) % len(names)
            xa_m = torch.stack([refs[names[i]][int(torch.randint(len(refs[names[i]]), (1,), generator=gen))] for i in ca]); xb_m = torch.stack([refs[names[i]][int(torch.randint(len(refs[names[i]]), (1,), generator=gen))] for i in cb])
            hi, lo = lam_hi * xa_m + (1 - lam_hi) * xb_m, lam_lo * xa_m + (1 - lam_lo) * xb_m; sh, sl = head(hi), head(lo); ia = torch.as_tensor([cols[names[i]] for i in ca]); ib = torch.as_tensor([cols[names[i]] for i in cb]); r = torch.arange(mix_pairs)
            loss = loss + mix_weight * (-F.logsigmoid(sh[r, ia] - sl[r, ia]) - F.logsigmoid(sl[r, ib] - sh[r, ib])).mean()
        if ce_weight > 0 and refs and len(refs) > 1:
            ce = [F.cross_entropy(head(f)[:, [cols[n] for n in names]], torch.full((len(f),), i)) for i, (n, f) in enumerate(refs.items())]; loss = loss + ce_weight * torch.stack(ce).mean()
        if bad is not None and len(bad): loss = loss + bad_weight * F.relu(head(bad) + 1.0).mean()
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); optimizer.step()
    return head.eval()


# ------------------------------------------------------------------------------------------ graph-endpoint study (branch claude/graph-endpoint-1013): own seed block, separate from every earlier line
# NOTE: the user assigned dev 8400-8419 although the sibling graph-selection session had already used 8000-8479 (its S3 blocks were 8400-8479, registered only on that branch).  Dev screens may reuse
# 8400-8419 (different question: classification / dose-response / harsh splits; no dev number is used for confirmation); the CONFIRMATION blocks must be disjoint from everything used anywhere.
GE_DEV_SEEDS = tuple(range(8400, 8420)); GE_CONFIRM1 = tuple(range(8500, 8535)); GE_CONFIRM2 = tuple(range(8550, 8585))
GE_SEEDS_ALL = set(GE_DEV_SEEDS) | set(GE_CONFIRM1) | set(GE_CONFIRM2)
USED_BY_SIBLING_SELECTION_SESSION = set(range(8000, 8480))
assert not (set(GE_CONFIRM1) | set(GE_CONFIRM2)) & (USED_SEEDS_BY_EARLIER_WORK | USED_BY_SIBLING_SELECTION_SESSION), "confirmation seeds must be unused anywhere"
assert not set(GE_DEV_SEEDS) & USED_SEEDS_BY_EARLIER_WORK, "dev seeds must be unused on this branch"
assert not set(GE_CONFIRM1) & set(GE_CONFIRM2) and len(GE_DEV_SEEDS) == 20 and len(GE_CONFIRM1) == len(GE_CONFIRM2) == 35
USED_SEEDS_BY_EARLIER_WORK = USED_SEEDS_BY_EARLIER_WORK | GE_SEEDS_ALL
# --- representation study (claude/representation-*): own seed blocks (development 8600-8619, confirmation 1 = 8650-8684); 8000-8199, 8400-8599 and the rest of 8600-8999 belong to parallel sessions
RP_DEV_SEEDS = tuple(range(8600, 8620)); RP_CONFIRM1 = tuple(range(8650, 8685)); RP_SEEDS_ALL = set(RP_DEV_SEEDS) | set(RP_CONFIRM1)
assert len(RP_SEEDS_ALL) == 55 and not RP_SEEDS_ALL & USED_SEEDS_BY_EARLIER_WORK and not RP_SEEDS_ALL & set(range(7000, 7336)) and not RP_SEEDS_ALL & set(range(8000, 8600)) and not set(RP_DEV_SEEDS) & set(RP_CONFIRM1)
USED_SEEDS_BY_EARLIER_WORK = USED_SEEDS_BY_EARLIER_WORK | RP_SEEDS_ALL   # representation blocks (8600-8619, 8650-8684)

# ------------------------------------------------------------------------------------------ anchor-exchange study (branch claude/anchor-exchange-*): own seed block, separate lines
AE_DEV_SEEDS = tuple(range(8700, 8720)); AE_CONFIRM1 = tuple(range(8750, 8785)); AE_SEEDS_ALL = set(AE_DEV_SEEDS) | set(AE_CONFIRM1)
RESERVED_FOR_PARALLEL_STUDIES = set(range(8000, 8200)) | set(range(8400, 8600)) | (set(range(8600, 9000)) - AE_SEEDS_ALL)
assert not AE_SEEDS_ALL & USED_SEEDS_BY_EARLIER_WORK and not AE_SEEDS_ALL & RESERVED_FOR_PARALLEL_STUDIES and len(AE_SEEDS_ALL) == 20 + 35
USED_SEEDS_BY_EARLIER_WORK = USED_SEEDS_BY_EARLIER_WORK | AE_SEEDS_ALL

# ------------------------------------------------------------------------------------------ cross-run transfer study (branch claude/cross-run-*): development 10900-10919, confirmation 1 = 10930-10964, confirmation 2 = 10965-10999
CR_DEV_SEEDS = tuple(range(10900, 10920)); CR_CONFIRM1 = tuple(range(10930, 10965)); CR_CONFIRM2 = tuple(range(10965, 11000)); CR_SEEDS_ALL = set(CR_DEV_SEEDS) | set(CR_CONFIRM1) | set(CR_CONFIRM2)
assert len(CR_SEEDS_ALL) == 20 + 35 + 35 and not CR_SEEDS_ALL & USED_SEEDS_BY_EARLIER_WORK and not set(CR_CONFIRM1) & set(CR_CONFIRM2) and not set(CR_DEV_SEEDS) & (set(CR_CONFIRM1) | set(CR_CONFIRM2))
USED_SEEDS_BY_EARLIER_WORK = USED_SEEDS_BY_EARLIER_WORK | CR_SEEDS_ALL
