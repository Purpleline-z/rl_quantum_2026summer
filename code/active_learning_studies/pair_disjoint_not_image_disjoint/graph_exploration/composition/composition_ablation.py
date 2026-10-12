"""Why does the log-loss endpoint reward / punish the outcome mix of the labelled set?  fit_head normalises the loss by the NUMBER of rows and gives tie rows an |a-b| penalty and
not_apply rows a relu(a)+relu(b) penalty, so non-decisive rows shrink the scores (they are a regulariser, not wasted budget).
Clean test: for the SAME randomly chosen labelled set, train the head on (all rows) / (decisive only) / (decisive + tie) / (decisive + not_apply) and evaluate the unchanged endpoint.
usage: PAIR_STUDY_SEEDS=400-419 JU_SPLIT=A python3 composition_ablation.py  (run from the study folder with PYTHONPATH=../../active_learning_program)"""
import json, os, sys, tempfile
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import judgment_unit_study as J
import frozen_encoder_reward_head as frozen
import run_pair_endpoint_study as single_study

split = os.environ.get("JU_SPLIT", "A"); torch.set_num_threads(2)
schedule = json.loads((single_study.OUT / "schedule.json").read_text()); lr, steps = schedule["learning_rate"], schedule["steps"]
cache = frozen.load_feature_cache(single_study.CACHE, single_study.DATA); out_path = Path(__file__).parent / f"composition_{split}.jsonl"
done = {(json.loads(l)["seed"]) for l in out_path.read_text().splitlines()} if out_path.exists() else set()
with tempfile.TemporaryDirectory() as scratch, out_path.open("a") as f:
    for seed in single_study.seeds():
        if seed in done: continue
        ctx = J.Context(seed, Path(scratch), cache, split, "groups", lr, steps); kind = lambda j: ctx.info[j]["winner"]
        rec = {"seed": seed, "split": split, "budgets": {}}
        for budget in J.BUDGETS:
            picks = J.random_picks(ctx, "random", ctx.pool, budget, seed * 7919 + budget * 31); labeled = ctx.initial + picks; res = {}
            sets = {"all": labeled, "decisive_only": [j for j in labeled if kind(j) in ("1", "2")],
                    "decisive+tie": [j for j in labeled if kind(j) in ("1", "2", "tie")], "decisive+not_apply": [j for j in labeled if kind(j) in ("1", "2", "not_apply")]}
            for name, js in sets.items():
                m = ctx.evaluate(ctx.fit(js)); res[name] = {k: m[k] for k in ("test_decisive_log_loss", "test_decisive_auc", "test_calibrated_log_loss", "test_decisive_accuracy")} | {"n_rows": len(js), "n_decisive": sum(kind(j) in ("1", "2") for j in js)}
            rec["budgets"][str(budget)] = res
        f.write(json.dumps(rec) + "\n"); f.flush(); print("seed", seed, "done", flush=True)
