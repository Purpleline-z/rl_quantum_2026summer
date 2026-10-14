#!/usr/bin/env python3
"""Zero-pair-label diagnostic of what typed anchors alone teach the head (no judgment is labelled; the head sees only anchors + the Laplacian term): held-out preference AUC / accuracy / anchor-far strata for
real anchors, synthetic anchors REPLACING the real ones (simulator banks, permuted types, augmented real images), and a real + synthetic mixture.  One JSON per (seed, split); dev seeds only.
  python3 syn_anchor_only.py --seeds 11100-11119 --splits A,B --out results/syn/anchor_only [--shard i/n]"""
import argparse, json, tempfile
from pathlib import Path
import numpy as np, torch
import nd_core as core, nd_graph_learner as G, nd_synthetic as Y, rt_core as rt, tg_run
import syn_specs  # noqa: F401

T4 = syn_specs.T4; T3 = syn_specs.T3
for name, kw in {
    "ao_syn_v1": dict(syn=dict(bank="v1", types=T4, share=0.999, nodes=0, per_step=16, replace=True)), "ao_syn_v1_T3": dict(syn=dict(bank="v1", types=T3, share=0.999, nodes=0, per_step=16, replace=True)),
    "ao_syn_v1_perm": dict(syn=dict(bank="v1", types=T4, share=0.999, nodes=0, per_step=16, replace=True, perm=True)), "ao_syn_v2": dict(syn=dict(bank="v2", types=T4, share=0.999, nodes=0, per_step=16, replace=True)),
    "ao_aug": dict(syn=dict(bank="augreal1", types=T4, share=0.999, nodes=0, per_step=16, replace=True)), "ao_aug_perm": dict(syn=dict(bank="augreal1", types=T4, share=0.999, nodes=0, per_step=16, replace=True, perm=True)),
    "ao_mix_v1": dict(syn=dict(bank="v1", types=T4, share=0.5, nodes=0, per_step=16)), "ao_real_mask": dict(ideal="mask"),
}.items(): Y.register(name, **kw)
NAMES = ["ao_real", "ao_real_mask", "ao_syn_v1", "ao_syn_v1_T3", "ao_syn_v1_perm", "ao_syn_v2", "ao_aug", "ao_aug_perm", "ao_mix_v1"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="11100-11119"); ap.add_argument("--splits", default="A,B"); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1"); a = ap.parse_args(); torch.set_num_threads(1)
    shard, n = (int(x) for x in a.shard.split("/")); a.out.mkdir(parents=True, exist_ok=True); cache = core.feature_cache(); sim = rt.Sim(cache)
    with tempfile.TemporaryDirectory() as scratch:
        for seed in [s for i, s in enumerate(tg_run.seeds_of(a.seeds)) if i % n == shard]:
            assert seed in core.SA_DEV_SEEDS
            for split in a.splits.split(","):
                target = a.out / f"seed{seed}_{split}.json"
                if target.exists(): continue
                ctx = core.make_ctx(seed, split, scratch, cache); test = core.TestSet(ctx); far = tg_run.anchor_distance(sim, ctx, test) > tg_run.FAR; recs = []
                for name in NAMES:
                    fn = G.LEARNERS["lap3_k10_st4"] if name == "ao_real" else Y.LEARNERS[name]; d = np.asarray(fn(ctx, [], seed * 7)(test), dtype=float)
                    recs.append({"seed": seed, "split": split, "budget": 0, "draw": 0, "learner": name, **core.metrics_from_d(d, test), **tg_run.stratum_metrics(d, test, far, None)})
                target.write_text(json.dumps(recs)); print(seed, split, flush=True)


if __name__ == "__main__": main()
