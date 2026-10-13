"""Lambda selected by internal validation (grouped 3-fold over the labelled pair groups; criterion = decisive log-loss) vs fixed lambda = 3 vs aw8_all, preference endpoint, identical random label sets.
Also reports the best fixed lambda in hindsight ('oracle', uses test metrics, diagnostic only) and the chosen lambda."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

import nd_core as core
import ge_heads as gh
import judgment_unit_study as ju_study

LAMS = (1.0, 3.0, 10.0)


def parse_seeds(text):
    if "-" in text: lo, hi = text.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def _val_logloss(ctx, head, held, fn) -> float:
    rows = ju_study.rows_of(ctx.exp, held); rows = rows[rows.Winner.isin(["1", "2"])]
    if not len(rows): return float("nan")
    with torch.no_grad(): a, b = head(fn(list(rows.resolved_img1))), head(fn(list(rows.resolved_img2)))
    k = torch.as_tensor(rows.type_idx.to_numpy()); d = (a - b)[torch.arange(len(k)), k].numpy(); s = np.where((rows.Winner == "1").to_numpy(), 1.0, -1.0); w = rows.confidence_weight.to_numpy().astype(float)
    return float((w * np.logaddexp(0.0, -s * d)).sum() / w.sum())


def choose_lambda(ctx, labelled, seed, gd, fn, folds: int = 3) -> float:
    groups = sorted({j[0] for j in labelled}); rng = np.random.default_rng(seed * 997 + len(labelled)); order = [groups[i] for i in rng.permutation(len(groups))]; scores = {lam: [] for lam in LAMS}
    for f in range(folds):
        held_groups = set(order[f::folds]); held = [j for j in labelled if j[0] in held_groups]; train = [j for j in labelled if j[0] not in held_groups]
        for lam in LAMS:
            head = gh.fit_head(ctx, train, seed, gd, fn, gd.x, 512, lap=lam, all_ideal=True); v = _val_logloss(ctx, head, held, fn)
            if not np.isnan(v): scores[lam].append(v)
    means = {lam: (np.mean(v) if v else float("inf")) for lam, v in scores.items()}; best = min(means.values())
    return 3.0 if means[3.0] <= best + 1e-12 else min(l for l, v in means.items() if v == best)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", required=True); ap.add_argument("--splits", default="A,B"); ap.add_argument("--budgets", default="20,60"); ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1"); a = ap.parse_args()
    torch.set_num_threads(1); shard, n = (int(x) for x in a.shard.split("/")); a.out.mkdir(parents=True, exist_ok=True); cache = core.feature_cache(); budgets = [int(b) for b in a.budgets.split(",")]
    with tempfile.TemporaryDirectory() as scratch:
        for i, seed in enumerate(parse_seeds(a.seeds)):
            if i % n != shard: continue
            for split in a.splits.split(","):
                target = a.out / f"seed{seed}_{split}.json"
                if target.exists(): continue
                t0 = time.monotonic(); ctx = core.make_ctx(seed, split, scratch, cache); test = core.TestSet(ctx); fn = core.identity_features(ctx); gd = gh.make_graph(ctx, 10); rows = []
                for budget in budgets:
                    for draw in range(a.draws):
                        lab = core.random_labelled(ctx, budget, draw); s = seed * 7 + draw; res = {}
                        res["base"] = core.metrics_from_d(core.head_d(gh.fit_head(ctx, lab, s, None, fn, None, 512, 0.0, True), fn, test), test)
                        for lam in LAMS: res[f"lap{lam:g}"] = core.metrics_from_d(core.head_d(gh.fit_head(ctx, lab, s, gd, fn, gd.x, 512, lam, True), fn, test), test)
                        lam = choose_lambda(ctx, lab, s, gd, fn); res["lap_auto"] = dict(res[f"lap{lam:g}"]); res["lap_auto"]["chosen_lambda"] = lam
                        best = max(LAMS, key=lambda l: res[f"lap{l:g}"]["auc"]); res["lap_oracle"] = dict(res[f"lap{best:g}"])
                        for arm, m in res.items(): rows.append({"seed": seed, "split": split, "budget": budget, "draw": draw, "arm": arm, **m})
                target.write_text(json.dumps(rows)); print(f"seed {seed} split {split}: {time.monotonic() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
