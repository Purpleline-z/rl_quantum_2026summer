#!/usr/bin/env python3
"""Score-propagation runner: head learners, free-score models, hybrids and distilled heads on identical random label sets (paired).

  python3 sp_run.py --variants lap3_k10_st4,fs_lam30_aw8_rho0.01,hy_lamr10_rhor0.1,ds_mu1_lam30 --seeds 11200-11209 --splits A,B --draws 2 --out results/sp/dev1 [--shard 0/4]
  splits: A, B (the repository's splits) or ``blocked`` (time-blocked contexts of rt_blocked.py); cold start = ``ND_INITIAL=random`` in the environment.

Variant kinds (names -> nd_propagate.parse_name):
  <head name>      any learner of nd_graph_learner / nd_recipe (``lap3_k10_st4`` and ``aw8_all_st2lr0.5`` use the cell's cached head fit)
  fs / kn / ls     free scores / nearest-scored-node baseline / label spreading of the anchors (no head)
  hy               hybrid (a): detached head scores H of ``lap3_k10_st4`` + free residual R   (lamr, rhor, penS, kt, tau)
  ds               hybrid (b): ``lap3_k10_st4`` head + pair-difference distillation (weight mu) of the free-score model given by the remaining tokens (lam, aw, rho, ...) on unlabelled nodes
One JSON per (seed, split): {"meta": {anc_min, k, sign, weight}, "records": [...]} exactly as rc_run.py (per-judgment scores ``d`` are stored); resumable.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import nd_core as core
import nd_graph_learner as G
import nd_propagate as P
import nd_recipe as R
import rc_run
import rt_core as rt
import rt_blocked as rb

FAR = rc_run.FAR
HEAD_NAMES = {"lap3_k10_st4"}


def _fit_head(ctx, labelled, seed, spec, teacher=None, lr_steps=None):
    gd = G.graph_for(ctx, spec.k, spec.shuffle, 0.0); featfn = core.identity_features(ctx)
    return R.fit_recipe(ctx, labelled, seed, gd, featfn, gd.x, gd.x.shape[1], spec, teacher, lr_steps), featfn


class Cell:
    """Everything shared by the variants of one (context, label set): the cached incumbent head and its scores."""
    def __init__(self, ctx, labelled, seed, test):
        self.ctx, self.labelled, self.seed, self.test = ctx, labelled, seed, test; self._head = None; self._fs = {}; self.extra = {}

    def head(self):
        if self._head is None:
            head, featfn = _fit_head(self.ctx, self.labelled, self.seed, R.BASE); self._head = (head, featfn)
        return self._head

    def head_scores(self, feats: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(): return self.head()[0](feats)[:, list(P.ACTIVE)]

    def teacher(self, name_tokens: str):
        if name_tokens not in self._fs:
            p = {**P.DEFAULTS, **P.parse_name("fs_" + name_tokens)}; pd_ = P.prop_for(self.ctx, int(p["k"]), bool(p["sh"]), bool(p["ex"])); rows = P.rows_of(pd_, self.labelled)
            self._fs[name_tokens] = P.solve(pd_, rows, p["aw"], p["bad"], p["lam"], p["rho"])[:pd_.N]
        return self._fs[name_tokens]


STACK_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def stack_weight(cell: Cell, tokens: str, n_folds: int = 3) -> float:
    """In-budget cross-validation of the weight w in d = d_head + w * d_free: grouped folds over the non-initial judgments (the initial groups always stay in training), both models refitted on the training
    folds with the same (lr, steps) as the final fit; w minimises the pooled Bradley-Terry log-loss of the held-out decisive judgments."""
    ctx, labelled = cell.ctx, cell.labelled; p = {**P.DEFAULTS, **P.parse_name("fs_" + tokens)}; pd_ = P.prop_for(ctx, int(p["k"]), bool(p["sh"]), bool(p["ex"]))
    init = set(ctx.initial); extra = sorted({j[0] for j in labelled if j not in init}); rng = np.random.default_rng(cell.seed * 13 + 7); rng.shuffle(extra); fold = {g: i % n_folds for i, g in enumerate(extra)}
    lr_steps = ctx.params_for(labelled); gd = G.graph_for(ctx, 10, False, 0.0); featfn = core.identity_features(ctx); dh, df, sg = [], [], []
    for f in range(n_folds):
        te = [j for j in labelled if j not in init and fold[j[0]] == f]; tr = [j for j in labelled if j not in te]
        rows_te = P.ju_study.rows_of(ctx.exp, te); rows_te = rows_te[rows_te.Winner.isin(["1", "2"])]
        if not len(rows_te): continue
        head, _ = _fit_head(ctx, tr, cell.seed, R.BASE, lr_steps=lr_steps); Ff = P.solve(pd_, P.rows_of(pd_, tr), p["aw"], p["bad"], p["lam"], p["rho"])
        t = torch.as_tensor(rows_te.type_idx.to_numpy()); col = torch.as_tensor([P.COL[int(x)] for x in t.tolist()]); r = torch.arange(len(t))
        with torch.no_grad(): s1, s2 = head(featfn(list(rows_te.resolved_img1))), head(featfn(list(rows_te.resolved_img2)))
        dh.append((s1 - s2)[r, t].numpy()); ia = torch.as_tensor([pd_.index[str(x)] for x in rows_te.resolved_img1]); ib = torch.as_tensor([pd_.index[str(x)] for x in rows_te.resolved_img2])
        df.append((Ff[ia, col] - Ff[ib, col]).numpy()); sg.append(np.where((rows_te.Winner == "1").to_numpy(), 1.0, -1.0))
    if not dh: return 0.0
    dh, df, sg = np.concatenate(dh), np.concatenate(df), np.concatenate(sg)
    return min(STACK_GRID, key=lambda w: float(np.logaddexp(0.0, -sg * (dh + w * df)).mean()))


def predict(cell: Cell, name: str, registry: dict) -> np.ndarray:
    ctx, labelled, seed, test = cell.ctx, cell.labelled, cell.seed, cell.test
    kind = name.split("_")[0]
    if name == "lap3_k10_st4":
        head, featfn = cell.head(); return core.head_d(head, featfn, test)
    if kind in ("fs", "kn", "ls"): return P.fit_variant(ctx, labelled, name)(test)
    if kind == "hy":
        p = {**P.DEFAULTS, **P.parse_name(name)}; pd_ = P.prop_for(ctx, int(p["k"]), bool(p["sh"]), bool(p["ex"])); rows = P.rows_of(pd_, labelled)
        H = cell.head_scores(pd_.feat); R_ = P.solve(pd_, rows, p["aw"], p["bad"], p["lamr"], p["rhor"], H=H, pen_total=bool(p["penS"]))
        cache = ctx.features.cache; f = lambda paths: torch.stack([cache[str(x)] for x in paths]).float()
        offset = (cell.head_scores(f(test.img1)), cell.head_scores(f(test.img2)))
        return P.predict_d(pd_, test, R_, int(p["kt"]), p["tau"], offset=offset)
    if kind == "st":
        tokens = name[3:]; w = stack_weight(cell, tokens); cell.extra[name] = {"stack_w": w}; head, featfn = cell.head()
        return core.head_d(head, featfn, test) + w * P.fit_variant(ctx, labelled, "fs_" + tokens)(test)
    if kind == "ds":
        toks = name.split("_")[1:]; mu = float([t for t in toks if t.startswith("mu")][0][2:]); rest = "_".join(t for t in toks if not t.startswith("mu"))
        T = cell.teacher(rest); head, featfn = _fit_head(ctx, labelled, seed, replace(R.BASE, distill=mu), teacher=T.float()); return core.head_d(head, featfn, test)
    return np.asarray(registry[name](ctx, labelled, seed)(test), dtype=float)


def make_contexts(seed: int, split: str, scratch, cache):
    return rb.make_blocked_ctx(seed, scratch, cache) if split == "blocked" else core.make_ctx(seed, split, scratch, cache)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--variants", required=True); ap.add_argument("--seeds", default="11200-11209"); ap.add_argument("--splits", default="A,B")
    ap.add_argument("--draws", type=int, default=3); ap.add_argument("--budgets", default="10,20,40,60"); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1")
    args = ap.parse_args(); torch.set_num_threads(1)
    names = args.variants.split(","); registry = {**G.LEARNERS, **R.LEARNERS}; budgets = [int(b) for b in args.budgets.split(",")]
    shard, nshards = (int(x) for x in args.shard.split("/")); args.out.mkdir(parents=True, exist_ok=True); cache = core.feature_cache(); sim = rt.Sim(cache)
    seeds = [s for i, s in enumerate(rc_run.parse_seeds(args.seeds)) if i % nshards == shard]
    with tempfile.TemporaryDirectory() as scratch:
        for seed in seeds:
            for split in args.splits.split(","):
                target = args.out / f"seed{seed}_{split}.json"; blob = json.loads(target.read_text()) if target.exists() else {"meta": None, "records": []}
                have = {(r["budget"], r["draw"], r["learner"]) for r in blob["records"]}
                if all((b, d, n) in have for b in budgets for d in range(args.draws) for n in names): continue
                started = time.monotonic(); ctx = make_contexts(seed, split, scratch, cache); test = core.TestSet(ctx); anc = rc_run.anchor_distance(ctx, test, sim); far = anc > FAR
                blob["meta"] = {"anc_min": np.round(anc, 6).tolist(), "k": test.k.tolist(), "sign": test.sign.tolist(), "weight": np.round(test.weight, 6).tolist()}
                for budget in budgets:
                    for draw in range(args.draws):
                        todo = [n for n in names if (budget, draw, n) not in have]
                        if not todo: continue
                        labelled = core.random_labelled(ctx, budget, draw); cell = Cell(ctx, labelled, seed * 7 + draw, test)
                        for name in todo:
                            d = predict(cell, name, registry)
                            rec = {"seed": seed, "split": split, "budget": budget, "draw": draw, "learner": name, **core.metrics_from_d(d, test), "n_far": int(far.sum())}
                            if far.sum() >= 2:
                                m = core.metrics_from_d(d[far], rc_run.subset(test, far)); rec.update({f"far_{k}": v for k, v in m.items() if k in ("acc", "ll", "auc")})
                            rec.update(cell.extra.get(name, {})); rec["d"] = np.round(d, 4).tolist(); blob["records"].append(rec)
                    target.write_text(json.dumps(blob))
                print(f"seed {seed} split {split}: {len(names)} variants in {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__": main()
