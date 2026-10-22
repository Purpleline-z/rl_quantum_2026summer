#!/usr/bin/env python3
"""V1a main experiment: sequential judgment-unit acquisition (rounds of 10 judgments, retrain every round), classifier2-style split B, V1a loss, frozen (lr, steps) from FROZEN.md.

  python3 run_v1a.py SEED [SEED ...]          # one CSV per seed in results/by_seed/ (idempotent); rows = seed x rule x budget (+ the two no-selection references)
Rules: random (5 replicates: random, random_r1..r4 -> averaged in the analysis), uncertainty, core_set, typiclust_pairs, laplace_bald, bald_decisive, ensemble_bald, probcover_pairs.
"""
import json
import sys
import tempfile
import time

import pandas as pd
import torch

import v1a_core as V

ju_study = V.ju_study

OUT = V.HERE / "results" / "by_seed"; FROZEN = V.HERE / "frozen_setting.json"
NAMES = [f"random_r{r}" if r else "random" for r in range(V.RANDOM_REPLICATES)] + [r for r in V.RULES if r != "random"]


def run_seed(seed: int, cache: dict, lr: float, steps: int) -> None:
    target = OUT / f"seed{seed}.csv"
    if target.exists(): return
    rows = []; started = time.monotonic()
    with tempfile.TemporaryDirectory() as scratch:
        ctx = V.V1aContext(seed, scratch, cache, lr, steps); base = {"seed": seed, "n_pool_judgments": len(ctx.pool), "n_initial_judgments": len(ctx.initial), "n_test_groups": len(ctx.test), "T": ctx.t_info["T"], "lr": lr, "steps": steps}
        for label, labelled in (("initial_only", ctx.initial), ("all_pool", ctx.initial + ctx.pool)):
            rows.append({**base, "rule": label, "budget": label, **ctx.evaluate(ctx.fit(labelled)), "selected": ""})
        for name in NAMES:
            t0 = time.monotonic(); result = ju_study.trajectory(ctx, name)
            for budget, m in result.items():
                rows.append({**base, "rule": name, "budget": int(budget), **{k: v for k, v in m.items() if k != "selected"}, "selected": ";".join(map(str, m["selected"])), "rule_seconds": round(time.monotonic() - t0, 1)})
    OUT.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(target, index=False); print(f"seed {seed}: {len(rows)} rows in {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1); setting = json.loads(FROZEN.read_text()); cache = V.load_cache()
    for s in sys.argv[1:]: run_seed(int(s), cache, setting["lr"], setting["steps"])
