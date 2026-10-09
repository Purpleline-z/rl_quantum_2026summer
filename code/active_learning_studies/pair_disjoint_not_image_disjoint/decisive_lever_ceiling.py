#!/usr/bin/env python3
"""Value of knowing (or predicting) whether a judgment will be decisive.  Single-shot, initial set -> batch of b judgments -> evaluate.
Policies: random | random among truly decisive judgments (hidden knowledge, upper bound for the outcome lever) | random among the top-q fraction of the pool ranked by the
predicted P(decisive) (decisive_v2 fitted on the initial judgments) for q in {0.25, 0.5}.  usage: decisive_lever_ceiling.py SPLIT FIRST LAST"""
import json, sys, tempfile, random
from pathlib import Path
import numpy as np, pandas as pd, torch
import frozen_encoder_reward_head as frozen, judgment_unit_study as ju, run_pair_endpoint_study as ss, new_methods_strategies as nm
torch.set_num_threads(1)
split, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cache = frozen.load_feature_cache(ss.CACHE, ss.DATA); schedule = json.loads((ss.OUT / "schedule.json").read_text()); table = ju.load_schedule("results/pair_endpoint_study/schedule_judgment_unit.json")
rows = []
with tempfile.TemporaryDirectory() as scratch:
    for seed in range(first, last + 1):
        ctx = ju.Context(seed, Path(scratch), cache, split, "groups", schedule["learning_rate"], schedule["steps"], table); nm.CTX["ctx"] = ctx
        model = ctx.fit(ctx.initial); cands, lab, c = ju.make_candidates(ctx, ctx.pool, ctx.initial, model, seed)
        pdec = nm.decisive_v2(cands, lab, c); by_id = {x["pair_id"]: j for x, j in zip(cands, [(x["base_pair"], x["pos"]) for x in cands])}
        judg = [(x["base_pair"], x["pos"]) for x in cands]; truth = np.array([ctx.info[j]["decisive"] for j in judg]); order = np.argsort(-pdec)
        pools = {"random": list(range(len(judg))), "oracle_decisive": list(np.flatnonzero(truth)), "top25_pdec": list(order[: len(order) // 4]), "top50_pdec": list(order[: len(order) // 2])}
        for b in (10, 20, 40, 60):
            for name, idx in pools.items():
                vals = []
                for r in range(10):
                    rng = random.Random(seed * 997 + b * 13 + r); chosen = rng.sample(idx, min(b, len(idx)))
                    if len(chosen) < b: chosen += rng.sample([i for i in range(len(judg)) if i not in set(chosen)], b - len(chosen))
                    m = ctx.evaluate(ctx.fit(ctx.initial + [judg[i] for i in chosen])); vals.append((m["test_decisive_log_loss"], m["test_decisive_auc"], m["test_decisive_accuracy"], truth[chosen].mean()))
                v = np.mean(vals, axis=0); rows.append({"split": split, "seed": seed, "budget": b, "policy": name, "logloss": v[0], "auc": v[1], "acc": v[2], "decisive_share": v[3]})
        print(seed, "done", "predicted P(dec) AUC on pool:", flush=True)
d = pd.DataFrame(rows); d.to_csv(f"results/new_methods/decisive_lever_{split}_{first}_{last}.csv", index=False)
p = d.groupby(["policy", "budget"])[["logloss", "auc", "acc", "decisive_share"]].mean().round(4)
r = p.xs("random", level="policy")
for pol in ("oracle_decisive", "top25_pdec", "top50_pdec"):
    q = p.xs(pol, level="policy"); print(pol, "gain over random (logloss, auc, acc), decisive share"); print(pd.DataFrame({"ll": r.logloss - q.logloss, "auc": q.auc - r.auc, "acc": q.acc - r.acc, "dec": q.decisive_share}).round(4))
