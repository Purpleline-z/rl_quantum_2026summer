#!/usr/bin/env python3
"""Which observable candidate features correlate with the TRUE single-judgment utility (decrease of test decisive log-loss when the judgment alone is added to the initial
set; hindsight, uses test labels)?  Spearman correlation per seed, averaged.  Dev seeds only.  usage: oracle_utility_features.py SPLIT FIRST LAST [n_labeled_extra]"""
import json, sys, tempfile, random
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
import frozen_encoder_reward_head as frozen, judgment_unit_study as ju, judgment_unit_strategies as jus, run_pair_endpoint_study as ss, pair_preference_endpoint as endpoint, new_methods_strategies as nm
torch.set_num_threads(1)
split, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); extra = int(sys.argv[4]) if len(sys.argv) > 4 else 0
cache = frozen.load_feature_cache(ss.CACHE, ss.DATA); schedule = json.loads((ss.OUT / "schedule.json").read_text()); table = ju.load_schedule("results/pair_endpoint_study/schedule_judgment_unit.json")
rows = []
with tempfile.TemporaryDirectory() as scratch:
    for seed in range(first, last + 1):
        ctx = ju.Context(seed, Path(scratch), cache, split, "groups", schedule["learning_rate"], schedule["steps"], table); nm.CTX["ctx"] = ctx
        pre = random.Random(seed).sample(ctx.pool, extra) if extra else []; labeled = ctx.initial + pre; pool = [j for j in ctx.pool if j not in set(pre)]
        ev = lambda js: endpoint.evaluate_preferences(ctx.exp, ctx.features, ctx.fit(js), ctx.test)["decisive_log_loss"]
        model = ctx.fit(labeled); base = ev(labeled); util = np.array([base - ev(labeled + [j]) for j in pool])
        cands, lab, c = ju.make_candidates(ctx, pool, labeled, model, seed)
        h, p, k = jus._own(model, cands + lab, c); n = len(cands); w = (p * (1 - p)); inverse = jus._inverses(h, w, k, n, 1.0)
        lev = jus._leverage(h, k, inverse, range(n)); pdec1 = nm.decisive_v1(cands, lab, c); pdec2 = nm.decisive_v2(cands, lab, c)
        a, b = nm._anchor_scores(cands, c)[:, 0], nm._anchor_scores(cands, c)[:, 1]
        judg = [(x["base_pair"], x["pos"]) for x in cands]; truth = np.array([ctx.info[j]["decisive"] for j in judg]); tie = np.array([ctx.info[j]["winner"] == "tie" for j in judg]); na = np.array([ctx.info[j]["winner"] == "not_apply" for j in judg])
        seen = {x["img1"] for x in lab} | {x["img2"] for x in lab}; unseen = np.array([(x["img1"] not in seen) + (x["img2"] not in seen) for x in cands])
        feats = {"phi_norm": np.linalg.norm(h[:n], axis=1), "p(1-p)": w[:n], "leverage": lev, "EGL phi*p(1-p)": np.linalg.norm(h[:n], axis=1) * w[:n], "pdec_v1": pdec1, "pdec_v2": pdec2, "anchor_max": a, "anchor_min": b,
                 "unseen_images": unseen, "TRUE decisive": truth.astype(float), "TRUE tie": tie.astype(float), "TRUE not_apply": na.astype(float), **{f"type{t}": (k[:n] == t).astype(float) for t in (0, 2, 3, 4)}}
        for name, f in feats.items(): rows.append({"seed": seed, "feature": name, "rho": spearmanr(f, util).statistic if np.std(f) > 0 else np.nan})
        print(seed, "done", flush=True)
d = pd.DataFrame(rows).groupby("feature").rho.agg(["mean", "std"]).round(3); print(d.sort_values("mean").to_string()); d.to_csv(f"results/new_methods/oracle_utility_features_{split}_{extra}.csv")
