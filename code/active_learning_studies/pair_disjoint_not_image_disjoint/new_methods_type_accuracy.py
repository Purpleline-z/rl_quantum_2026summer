#!/usr/bin/env python3
"""Downstream 4-class reconstruction-type accuracy (win-rate against the ideal references, as in the earlier studies) of the models trained on the judgments selected in a stored run.
usage: new_methods_type_accuracy.py <run> <split A|B> <first seed> <last seed> <budgets comma list>   (single-shot cells only)"""
import json, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd, torch
import new_methods_study  # registers the patches
import frozen_encoder_reward_head as frozen, judgment_unit_study as ju, run_pair_endpoint_study as ss
torch.set_num_threads(1)
run, split, first, last, budgets = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), [int(b) for b in sys.argv[5].split(",")]
root = Path("results/new_methods") / run / split; cache = frozen.load_feature_cache(ss.CACHE, ss.DATA); sch = json.loads((ss.OUT / "schedule.json").read_text()); table = ju.load_schedule("results/pair_endpoint_study/schedule_judgment_unit.json")
rows = []
with tempfile.TemporaryDirectory() as scratch:
    for seed in range(first, last + 1):
        ctx = ju.Context(seed, Path(scratch), cache, split, "groups", sch["learning_rate"], sch["steps"], table); man = json.loads((root / f"manifest_seed{seed}.json").read_text())
        pool = [(p, pos) for p, pos, _, _ in man["pool"]]
        rows.append({"seed": seed, "strategy": "initial_only", "budget": 0, "type_acc": frozen.evaluate_model(ctx.exp, ctx.features, ctx.fit(ctx.initial))["test_accuracy"]})
        for path in sorted((root / "single").glob(f"seed{seed}_*.json")):
            d = json.loads(path.read_text()); name = d["strategy"]
            if name == "initial_only": continue
            fam = name.rsplit("_r", 1)[0] if name.startswith("random") and name[-1].isdigit() and "_r" in name else name
            for b in budgets:
                picks = [pool[i] for i in d["checkpoints"][str(b)]["selected"]]; model = ctx.fit(ctx.initial + picks)
                rows.append({"seed": seed, "strategy": fam, "budget": b, "type_acc": frozen.evaluate_model(ctx.exp, ctx.features, model)["test_accuracy"]})
        print(seed, "done", flush=True)
d = pd.DataFrame(rows); d.to_csv(f"results/new_methods/type_accuracy_{run}_{split}_{first}_{last}.csv", index=False)
g = d.groupby(["strategy", "budget", "seed"]).type_acc.mean().groupby(["strategy", "budget"]).agg(["mean", "std", "count"]).round(3); print(g.to_string())
