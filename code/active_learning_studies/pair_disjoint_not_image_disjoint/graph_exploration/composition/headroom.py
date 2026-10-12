"""How much ranking headroom does the endpoint have?  Test AUC (all and per type) of heads trained on: initial 10 groups only / + 60 random judgments / + ALL pool judgments (the ceiling any selector could reach).
usage: JU_SPLIT=A PAIR_STUDY_SEEDS=400-409 python3 headroom.py"""
import json, os, sys, tempfile
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import judgment_unit_study as J
import frozen_encoder_reward_head as frozen
import pair_preference_endpoint as endpoint
import run_pair_endpoint_study as single_study

split = os.environ.get("JU_SPLIT", "A"); torch.set_num_threads(1)
schedule = json.loads((single_study.OUT / "schedule.json").read_text()); lr, steps = schedule["learning_rate"], schedule["steps"]
cache = frozen.load_feature_cache(single_study.CACHE, single_study.DATA); out = Path(__file__).parent / f"headroom_{split}.jsonl"
with tempfile.TemporaryDirectory() as scratch, out.open("a") as f:
    for seed in single_study.seeds():
        ctx = J.Context(seed, Path(scratch), cache, split, "groups", lr, steps)
        rows = ctx.exp.rows_for(ctx.test); rows = rows[rows.Winner.isin(["1", "2"])]; typ = rows.type_idx.to_numpy(); y = (rows.Winner == "1").to_numpy()
        res = {"seed": seed, "split": split, "n_pool": len(ctx.pool)}
        for name, js in {"initial": ctx.initial, "random60": ctx.initial + J.random_picks(ctx, "random", ctx.pool, 60, seed * 7919 + 60 * 31), "full_pool": ctx.initial + ctx.pool}.items():
            m = ctx.fit(js); d, s, w = endpoint.preference_scores(ctx.exp, ctx.features, m, ctx.test)
            per = {int(t): (float(roc_auc_score(y[typ == t], d[typ == t])) if len(set(y[typ == t])) == 2 else None) for t in sorted(set(typ))}
            res[name] = {"auc": float(roc_auc_score(y, d)), "acc": float(((d * s) > 0).mean()), "per_type_auc": per, "n_labeled": len(js)}
        res["n_test_by_type"] = {int(t): int((typ == t).sum()) for t in sorted(set(typ))}; f.write(json.dumps(res) + "\n"); f.flush(); print(seed, {k: round(v["auc"], 3) for k, v in res.items() if isinstance(v, dict) and "auc" in v}, flush=True)
