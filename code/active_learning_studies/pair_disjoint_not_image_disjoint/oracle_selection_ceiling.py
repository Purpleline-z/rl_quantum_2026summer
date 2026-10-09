#!/usr/bin/env python3
"""How much could ANY selection rule gain?  Oracle selection that uses the TEST labels (never usable in practice): the utility of a judgment is the decrease of
test decisive log-loss when it alone is added to the initial set; the oracle batch of size b is the top-b by utility.  Compared with Random (mean of 20 random
batches) on the same seed/split.  Dev seeds only.  usage: oracle_selection_ceiling.py SPLIT FIRST LAST"""
import json, sys, tempfile, random
from pathlib import Path
import numpy as np, pandas as pd, torch
import frozen_encoder_reward_head as frozen, judgment_unit_study as ju, run_pair_endpoint_study as ss, pair_preference_endpoint as endpoint
torch.set_num_threads(1)
split, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cache = frozen.load_feature_cache(ss.CACHE, ss.DATA); schedule = json.loads((ss.OUT / "schedule.json").read_text()); table = ju.load_schedule("results/pair_endpoint_study/schedule_judgment_unit.json")
out = Path("results/new_methods/oracle_ceiling"); out.mkdir(parents=True, exist_ok=True); rows = []
with tempfile.TemporaryDirectory() as scratch:
    for seed in range(first, last + 1):
        ctx = ju.Context(seed, Path(scratch), cache, split, "groups", schedule["learning_rate"], schedule["steps"], table)
        ev = lambda judgments: endpoint.evaluate_preferences(ctx.exp, ctx.features, ctx.fit(judgments), ctx.test)
        base = ev(ctx.initial); util = np.array([base["decisive_log_loss"] - ev(ctx.initial + [j])["decisive_log_loss"] for j in ctx.pool]); order = np.argsort(-util, kind="stable")
        for b in (10, 20, 40, 60):
            pick = [ctx.pool[i] for i in order[:b]]; orc = ev(ctx.initial + pick); rnd = [ev(ctx.initial + random.Random(seed * 100 + r).sample(ctx.pool, b)) for r in range(20)]
            rows.append({"split": split, "seed": seed, "budget": b, "oracle_ll": orc["decisive_log_loss"], "oracle_acc": orc["decisive_accuracy"], "random_ll": np.mean([r["decisive_log_loss"] for r in rnd]),
                         "random_acc": np.mean([r["decisive_accuracy"] for r in rnd]), "oracle_decisive_share": float(np.mean([ctx.info[j]["decisive"] for j in pick]))})
        print(seed, "done", flush=True)
d = pd.DataFrame(rows); d.to_csv(out / f"oracle_{split}_{first}_{last}.csv", index=False)
g = d.groupby("budget").mean(numeric_only=True); g["ll_gain"] = g.random_ll - g.oracle_ll; g["acc_gain"] = g.oracle_acc - g.random_acc
print(g[["ll_gain", "acc_gain", "oracle_decisive_share"]].round(4))
