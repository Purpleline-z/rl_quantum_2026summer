#!/usr/bin/env python3
"""How much could ANY selection rule gain?  Oracle selection that uses the TEST labels (never usable in practice): the utility of a judgment is the
decrease of test decisive log-loss when it alone is added to the initial set; the oracle batch of size b is the top-b by utility.  Compared with Random
(mean of 20 random batches) on the same seed/split.  Dev seeds only.  usage: oracle_selection_ceiling.py SPLIT FIRST LAST"""
import json, sys, tempfile, random
from pathlib import Path
import numpy as np, torch

import frozen_encoder_reward_head as frozen, judgment_unit_study as ju, run_pair_endpoint_study as ss
torch.set_num_threads(1)
split, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cache = frozen.load_feature_cache(ss.CACHE, ss.DATA); schedule = json.loads((ss.OUT / "schedule.json").read_text()); table = ju.load_schedule("results/pair_endpoint_study/schedule_judgment_unit.json")
out = Path("results/new_methods/oracle_ceiling"); out.mkdir(parents=True, exist_ok=True); rows = []
with tempfile.TemporaryDirectory() as scratch:
    for seed in range(first, last + 1):
        ctx = ju.Context(seed, Path(scratch), cache, split, "groups", schedule["learning_rate"], schedule["steps"], table)
        base = ctx.evaluate(ctx.fit(ctx.initial)); util = []
        for j in ctx.pool:
            m = ctx.evaluate(ctx.fit(ctx.initial + [j])); util.append((base["test_decisive_log_loss"] - m["test_decisive_log_loss"], base["test_decisive_auc"] - m["test_decisive_auc"] * -1 if False else m["test_decisive_auc"] - base["test_decisive_auc"]))
        util = np.array(util); order = np.argsort(-util[:, 0], kind="stable")
        for b in (10, 20, 40, 60):
            pick = [ctx.pool[i] for i in order[:b]]; orc = ctx.evaluate(ctx.fit(ctx.initial + pick))
            rnd = [ctx.evaluate(ctx.fit(ctx.initial + random.Random(seed * 100 + r).sample(ctx.pool, b))) for r in range(20)]
            rows.append({"split": split, "seed": seed, "budget": b, **{f"oracle_{k}": orc[k] for k in ("test_decisive_log_loss", "test_decisive_auc", "test_decisive_accuracy")},
                         **{f"random_{k}": float(np.mean([r[k] for r in rnd])) for k in ("test_decisive_log_loss", "test_decisive_auc", "test_decisive_accuracy")},
                         "oracle_decisive_share": float(np.mean([ctx.info[j]["decisive"] for j in pick]))})
        print(seed, "done", flush=True)
import pandas as pd
d = pd.DataFrame(rows); d.to_csv(out / f"oracle_{split}_{first}_{last}.csv", index=False)
g = d.groupby("budget").mean(numeric_only=True)
g["logloss_gain"] = g.random_test_decisive_log_loss - g.oracle_test_decisive_log_loss; g["auc_gain"] = g.oracle_test_decisive_auc - g.random_test_decisive_auc; g["acc_gain"] = g.oracle_test_decisive_accuracy - g.random_test_decisive_accuracy
print(g[["logloss_gain", "auc_gain", "acc_gain", "oracle_decisive_share"]].round(4))
