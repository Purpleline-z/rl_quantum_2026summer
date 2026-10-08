#!/usr/bin/env python3
"""Hand-check of stored judgment-unit cells: re-derive pool/initial sets from the manifest, recount the revealed judgments, confirm that no validation/test image
is in the labelled set, and retrain the head from the stored selection to reproduce the stored metrics.
usage: judgment_unit_verify.py <run> <condition> <seed> <strategy> <budget> [<split A|B>]"""
import json, sys, tempfile
from pathlib import Path
import torch
import frozen_encoder_reward_head as frozen
import judgment_unit_study as study
import run_pair_endpoint_study as single

run, condition, seed, strategy, budget = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
split, initial_mode = run.split("/")[-1].split("_")[:2]; torch.set_num_threads(1)  # run may be a sub-path such as replication/A_groups
schedule = json.loads((single.OUT / "schedule.json").read_text()); cache = frozen.load_feature_cache(single.CACHE, single.DATA)
with tempfile.TemporaryDirectory() as t:
    ctx = study.Context(seed, Path(t), cache, split, initial_mode, schedule["learning_rate"], schedule["steps"])
    manifest = json.loads((study.OUT / run / f"manifest_seed{seed}.json").read_text()); cell = json.loads((study.OUT / run / condition / f"seed{seed}_{strategy}.json").read_text())
    assert [tuple(x[:2]) for x in manifest["pool"]] == ctx.pool and [tuple(x) for x in manifest["initial"]] == ctx.initial
    values = cell["checkpoints"][budget]; chosen = [ctx.pool[i] for i in values["selected"]]
    groups_pool = len(ctx.pool_groups); judgments_per_group = len(ctx.pool) / groups_pool
    print(f"pool: {len(ctx.pool)} judgments from {groups_pool} groups ({judgments_per_group:.2f} per group), {len({p for p, _ in ctx.pool})} distinct pairs; initial: {len(ctx.initial)} judgments")
    print(f"revealed {len(chosen)} (budget {budget}), distinct {len(set(chosen))}, all in pool {set(chosen) <= set(ctx.pool)}, touches {len({p for p, _ in chosen})} pairs; stored n_revealed={values['n_revealed']}")
    images = lambda js: {x for j in js for x in (ctx.info[j]["img1"], ctx.info[j]["img2"])}
    held = {x for g in ctx.validation + ctx.test for r in ctx.exp.groups[g].itertuples() for x in (r.resolved_img1, r.resolved_img2)}
    print(f"validation+test images: {len(held)}; shared with labelled+selected: {len(held & images(ctx.initial + chosen))}; shared with the whole pool: {len(held & images(ctx.pool))}")
    rows = study.rows_of(ctx.exp, ctx.initial + chosen); print(f"training rows: {len(rows)} = {len(ctx.initial)} + {len(chosen)}: {len(rows) == len(ctx.initial) + len(chosen)}")
    hidden = len({(p, q) for p in {p for p, _ in chosen} for q in range(len(ctx.exp.groups[p]))} - set(ctx.initial + chosen)); print(f"unselected judgments of selected pairs: {hidden}")
    keys = set(zip(rows.pair_id, rows.type_idx)); outside = [(p, ctx.info[(p, q)]['type_idx']) for p in {p for p, _ in chosen} for q in range(len(ctx.exp.groups[p])) if (p, q) not in set(ctx.initial + chosen) and (p, ctx.info[(p, q)]['type_idx']) in keys]
    print(f"hidden (pair, type) judgments present in the training rows: {len(outside)}")
    metrics = ctx.evaluate(ctx.fit(ctx.initial + chosen))
    print({k: (round(metrics[k], 6), round(values[k], 6)) for k in ("test_decisive_log_loss", "test_decisive_auc", "test_decisive_accuracy")}, "(recomputed, stored)")
