"""Selection-level screen (single shot: one batch of b judgments chosen from the model trained on the 10 initial groups), paired with random label sets.

Selectors: random (3 draws), vopt_u (the repository's pool-wide I-optimal rule, the strongest earlier rule), vopt_amb (goal-oriented: target weights = class ambiguity of the pool
judgments' images), consistency / vopt_cons (disagreement of the preference between the image and its mirror image), gp_var (greedy posterior variance of the preference under the GP learner).
The learner that is trained on the selected labels is a second factor.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

import judgment_unit_strategies as ju
import judgment_unit_study as ju_study
import new_methods_strategies as nm
import nd_core as core
import nd_learners as L

CTX_REF: dict = {}


# ------------------------------------------------------------------------------------------ ambiguity of an image between the reconstruction types
def anchor_head(ctx):
    key = ("anchor", ctx.seed, ctx.__class__.__name__, id(ctx))
    if CTX_REF.get("anchor_key") != key:
        lr, steps = ctx.params_for([]); CTX_REF["anchor"] = core.fit_generic(ctx, [], core.identity_features(ctx), 512, lr, steps, ctx.seed); CTX_REF["anchor_key"] = key
    return CTX_REF["anchor"]


def image_ambiguity(ctx, paths) -> np.ndarray:
    """Normalised entropy (0..1) of the type posterior of each image: win-rate of each active type's head against the references of the other types, normalised to sum to 1."""
    head = anchor_head(ctx); fn = core.identity_features(ctx)
    with torch.no_grad():
        s = head(fn(paths)).numpy(); refs = {c: head(fn(ps)).numpy() for c, ps in ctx.exp.references.items()}
    win = []
    for c in refs:
        k = core.frozen.TYPE_TO_INDEX[c]; opp = np.concatenate([v for o, v in refs.items() if o != c])
        win.append((1 / (1 + np.exp(-np.clip(s[:, [k]] - opp[:, k][None, :], -50, 50)))).mean(1))
    p = np.stack(win, 1); p = p / p.sum(1, keepdims=True); return -(p * np.log(np.clip(p, 1e-9, 1))).sum(1) / np.log(p.shape[1])


def judgment_ambiguity(ctx, items) -> np.ndarray:
    paths = sorted({x for it in items for x in (it["img1"], it["img2"])}); amb = dict(zip(paths, image_ambiguity(ctx, paths)))
    return np.array([.5 * (amb[it["img1"]] + amb[it["img2"]]) for it in items])


def amb_target(cands, labeled, cache):
    a = judgment_ambiguity(CTX_REF["ctx"], cands); m = max(float(a.mean()), 1e-6); return (a + .1 * m) / (1.1 * m)


def vopt_amb(*a, **kw): return nm.vopt(*a, sensitivity=False, target_decisive=amb_target, **kw)


# ------------------------------------------------------------------------------------------ mirror consistency
def consistency_scores(cands, model, cache) -> np.ndarray:
    ctx = CTX_REF["ctx"]; head = model.reward_head.eval(); k = torch.as_tensor([int(c["type_idx"]) for c in cands]); i = torch.arange(len(cands))
    fa = torch.stack([cache[c["img1"]] for c in cands]); fb = torch.stack([cache[c["img2"]] for c in cands])
    ma, mb = L.mirror_features(ctx, [c["img1"] for c in cands]), L.mirror_features(ctx, [c["img2"] for c in cands])
    with torch.no_grad():
        d = (head(fa) - head(fb))[i, k]; dm = (head(ma) - head(mb))[i, k]
    return (d - dm).abs().numpy()


def consistency(cands, labeled, model, cache, budget, seed=0):
    s = consistency_scores(cands, model, cache); return [cands[i] for i in np.argsort(-s, kind="stable")[:budget]]


def vopt_cons(cands, labeled, model, cache, budget, seed=0):
    s = consistency_scores(cands, model, cache); factor = 1 + s / max(float(s.mean()), 1e-9)
    return nm.vopt(cands, labeled, model, cache, budget, seed, sensitivity=False, informative=lambda c, l, ca: factor)


# ------------------------------------------------------------------------------------------ GP posterior variance
def gp_var(cands, labeled, model, cache, budget, seed=0, lam: float = 0.03, factor: float = 0.5):
    """Greedy pick of the judgment with the largest posterior variance of its preference f(a) - f(b) under the GP (Laplace/Gauss-Newton precision from the labelled judgments of
    its type), with a rank-one update after every pick so the batch does not repeat itself."""
    import nd_gp
    ctx = CTX_REF["ctx"]; s = nd_gp._nodes(ctx); idx = s["index"]; K = nd_gp.kernel(ctx, factor).numpy(); n = len(K)
    ia = np.array([idx[str(c["img1"])] for c in cands]); ib = np.array([idx[str(c["img2"])] for c in cands]); k = np.array([int(c["type_idx"]) for c in cands])
    p = np.full(len(cands), .25)  # preference probability prior weight w = p(1-p) at p = 1/2
    chosen = []
    cov = {}
    for kk in nd_gp.ACTIVE:
        Sigma = K / lam   # prior covariance K / lam
        for it in labeled:
            if int(it["type_idx"]) != kk: continue
            u = np.zeros(n); u[idx[str(it["img1"])]] += 1; u[idx[str(it["img2"])]] -= 1; Su = Sigma @ u; Sigma = Sigma - np.outer(Su, Su) * .25 / (1 + .25 * (u @ Su))
        cov[kk] = Sigma
    alive = np.ones(len(cands), bool)
    def gain(i):
        u = np.zeros(n); u[ia[i]] += 1; u[ib[i]] -= 1; return u @ cov[k[i]] @ u
    score = np.array([gain(i) for i in range(len(cands))])
    for _ in range(min(budget, len(cands))):
        i = int(np.argmax(np.where(alive, score, -np.inf))); chosen.append(i); alive[i] = False
        u = np.zeros(n); u[ia[i]] += 1; u[ib[i]] -= 1; Su = cov[k[i]] @ u; cov[k[i]] = cov[k[i]] - np.outer(Su, Su) * .25 / (1 + .25 * (u @ Su))
        for j in np.flatnonzero(alive & (k == k[i])): score[j] = gain(j)
    return [cands[i] for i in chosen]


SELECTORS = {"vopt_u": nm.NEW["vopt_u"], "vopt_amb": vopt_amb, "consistency": consistency, "vopt_cons": vopt_cons, "gp_var": gp_var}


def parse_seeds(text):
    if "-" in text: lo, hi = text.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def amb_metrics(ctx, test, d) -> dict:
    """Metrics on the half of the held-out decisive judgments whose images are most ambiguous between types (the goal-oriented target)."""
    paths = sorted(set(test.img1) | set(test.img2)); amb = dict(zip(paths, image_ambiguity(ctx, paths))); a = np.array([.5 * (amb[x] + amb[y]) for x, y in zip(test.img1, test.img2)])
    hi = a >= np.median(a); s, w = test.sign[hi], test.weight[hi]; dd = np.asarray(d)[hi]
    return {"acc_amb": float((w * ((dd * s) > 0)).sum() / w.sum()), "ll_amb": float((w * np.logaddexp(0.0, -s * dd)).sum() / w.sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selectors", required=True); parser.add_argument("--learners", default="baseline"); parser.add_argument("--seeds", default="2000-2019"); parser.add_argument("--splits", default="A,B")
    parser.add_argument("--draws", type=int, default=3); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--shard", default="0/1")
    args = parser.parse_args(); torch.set_num_threads(1)
    reg = L.LEARNERS.copy()
    import nd_bayes, nd_gp, nd_temporal
    reg.update(nd_bayes.LEARNERS); reg.update(nd_gp.LEARNERS); reg.update(nd_temporal.LEARNERS)
    sel_names = args.selectors.split(","); learner_names = args.learners.split(","); shard, nshards = (int(x) for x in args.shard.split("/")); args.out.mkdir(parents=True, exist_ok=True); cache = core.feature_cache()
    with tempfile.TemporaryDirectory() as scratch:
        for seed in [s for i, s in enumerate(parse_seeds(args.seeds)) if i % nshards == shard]:
            for split in args.splits.split(","):
                target = args.out / f"seed{seed}_{split}.json"; done = json.loads(target.read_text()) if target.exists() else []; have = {(r["budget"], r["selector"], r["draw"], r["learner"]) for r in done}
                need = [(b, s, (range(args.draws) if s == "random" else [0]), n) for b in core.BUDGETS for s in sel_names for n in learner_names]
                if all((b, s, d, n) in have for b, s, ds, n in need for d in ds): continue
                started = time.monotonic(); ctx = core.make_ctx(seed, split, scratch, cache); test = core.TestSet(ctx); CTX_REF["ctx"] = ctx; nm.CTX["ctx"] = ctx
                model = ctx.fit(ctx.initial); cands, labeled_items, ccache = ju_study.make_candidates(ctx, ctx.pool, ctx.initial, model, seed)
                for budget in core.BUDGETS:
                    for sel in sel_names:
                        if sel == "random": label_sets = {d: core.random_labelled(ctx, budget, d) for d in range(args.draws)}
                        else:
                            picks = SELECTORS[sel](cands, labeled_items, model, ccache, budget, seed); ids = {f"{j[0]}#{j[1]}": j for j in ctx.pool}
                            label_sets = {0: list(ctx.initial) + [ids[x["pair_id"]] for x in picks]}
                        for draw, labelled in label_sets.items():
                            for name in learner_names:
                                if (budget, sel, draw, name) in have: continue
                                d = reg[name](ctx, labelled, seed * 7 + draw)(test)
                                done.append({"seed": seed, "split": split, "budget": budget, "selector": sel, "draw": draw, "learner": name, **core.metrics_from_d(d, test), **amb_metrics(ctx, test, d),
                                             "n_pairs": len({p for p, _ in labelled if (p, _) not in set(ctx.initial)})})
                    target.write_text(json.dumps(done))
                print(f"seed {seed} split {split}: {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
