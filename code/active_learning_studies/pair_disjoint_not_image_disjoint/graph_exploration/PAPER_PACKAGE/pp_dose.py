"""Dose-response tables for the graph (gain over the tuned graph-free baseline aw8_all, paired over seeds; 95% bootstrap interval over seeds):
  noise    topology noise sigma in {0, .5, 1, 2, 3} for lambda = 3 and 10 + the shuffled-graph control (graph-mechanism confirmations GM1 8100-8134 and GM2 8150-8184, pre-registered) + mean graph quality (ideal-ideal same-class share)
  rewire   random / local rewiring of a fraction of the edges, lambda = 10 (graph-mechanism development rounds 16-17, seeds 8000-8019, NOT evidence)
  nodes    fraction of the trajectory frames in the graph (graph-endpoint confirmations GE1 8500-8534 and GE2 8550-8584, pre-registered); 'out' = frames outside every pair group; shuffled graph at 100 %
Outputs tables/dose_noise.csv, dose_rewire.csv, dose_nodes.csv, numbers_dose.json."""
import json

import numpy as np
import pandas as pd

import pp_lib as L
from pp_lib import RES, Numbers

R = "random"; SIG = [("n0", 0.0), ("n0.5", 0.5), ("n1", 1.0), ("n2", 2.0), ("n3", 3.0)]
METRICS = ("auc", "acc", "ll")


def quality(seed, split, name):
    p = RES / "gm" / "quality" / f"seed{seed}_{split}.json"
    return json.loads(p.read_text()).get(name, {}).get("quality", np.nan) if p.exists() else np.nan


def one(f, split, m, a, ref):
    per = L.per_seed(f, split, m, (R, a), (R, ref)); return per, L.summarize(per)


def main():
    N = Numbers("dose"); noise, rew, nodes = [], [], []
    for blk, rel in (("GM1", "gm/confirm1"), ("GM2", "gm/confirm2")):
        f = L.load_dir(rel)
        for lam in (3, 10):
            for suf, sg in SIG:
                name = f"gm_lap{lam}_{suf}"
                for split in ("A", "B"):
                    q = float(np.nanmean([quality(s, split, name) for s in sorted(f.seed.unique())]))
                    for m in METRICS:
                        _, s = one(f, split, m, name, "aw8_all"); noise.append(dict(block=blk, lam=lam, sigma=sg, learner=name, split=split, metric=m, quality=q, **{k: s[k] for k in ("gain", "lo", "hi", "p", "n")}))
                        for fld in ("gain", "lo", "hi"): N.put(f"dn.{blk}.lam{lam}.sig{suf[1:]}.{split}.{m}.{fld}", s[fld])
                    N.put(f"dn.{blk}.lam{lam}.sig{suf[1:]}.{split}.quality", q)
        for name, tagn in (("gm_lap10_n0_shuf", "shuf10"), ("gmc_t_no_rt13_shuf", "shuf_rt13off")):
            for split in ("A", "B"):
                for m in METRICS:
                    _, s = one(f, split, m, name, "aw8_all"); noise.append(dict(block=blk, lam=10, sigma=np.nan, learner=name, split=split, metric=m, quality=np.nan, **{k: s[k] for k in ("gain", "lo", "hi", "p", "n")}))
                    for fld in ("gain", "lo", "hi"): N.put(f"dn.{blk}.{tagn}.{split}.{m}.{fld}", s[fld])
    # graph-learner confirmations: shuffled graphs of the lambda = 1, 3, 10 learners (pre-registered controls) vs aw8_all
    for blk, rel in (("GL1", "gl/confirm1"), ("GL2", "gl/confirm2")):
        f = L.load_dir(rel)
        for lam in (1, 3, 10):
            for split in ("A", "B"):
                for m in METRICS:
                    for kind, name in (("shuf", f"lap{lam}_k10_shuf"), ("n3", f"lap{lam}_k10_n3"), ("real", f"lap{lam}_k10")):
                        if name not in set(f.learner): continue
                        _, s = one(f, split, m, name, "aw8_all"); noise.append(dict(block=blk, lam=lam, sigma=(3.0 if kind == "n3" else (0.0 if kind == "real" else np.nan)), learner=name, split=split, metric=m, quality=np.nan, **{k: s[k] for k in ("gain", "lo", "hi", "p", "n")}))
                        for fld in ("gain", "lo", "hi"): N.put(f"dn.{blk}.lam{lam}.{kind}.{split}.{m}.{fld}", s[fld])
    pd.DataFrame(noise).to_csv(L.TABLES / "dose_noise.csv", index=False)
    # rewiring (development)
    f = pd.concat([L.load_dir(r) for r in ("gm/dev1", "gm/dev16", "gm/dev17")], ignore_index=True)
    for kind, frac, name in [("clean", 0, "gm_lap10_n0")] + [("random", x, f"gm10_rw{x}") for x in (10, 25, 50, 75)] + [("local", x, f"gm10_rwl{x}") for x in (25, 50)]:
        for split in ("A", "B"):
            for m in METRICS:
                _, s = one(f, split, m, name, "aw8_all"); rew.append(dict(kind=kind, frac=frac, learner=name, split=split, metric=m, **{k: s[k] for k in ("gain", "lo", "hi", "p", "n")}))
                for fld in ("gain", "lo", "hi"): N.put(f"dr.{kind}{frac}.{split}.{m}.{fld}", s[fld])
    pd.DataFrame(rew).to_csv(L.TABLES / "dose_rewire.csv", index=False)
    # unlabelled-node fraction (graph-endpoint confirmations)
    for blk, rel in (("GE1", "ge/conf1_dose"), ("GE2", "ge/conf2_dose")):
        f = L.load_dir(rel)
        for var in ("all_p0", "all_p10", "all_p25", "all_p50", "all_p100", "out_p0", "out_p25", "out_p50", "all_p100_shuf"):
            for split in ("A", "B"):
                g = f[f.split == split]; per = g.groupby(["seed", "variant"])[["auc", "acc", "ll"]].mean()
                for m in METRICS:
                    d = (L.sign_of(m) * (per.xs(var, level="variant")[m] - per.xs("base", level="variant")[m])).dropna(); s = L.summarize(d)
                    nodes.append(dict(block=blk, variant=var, split=split, metric=m, n_nodes=float(g[g.variant == var].n_nodes.mean()), **{k: s[k] for k in ("gain", "lo", "hi", "p", "n")}))
                    for fld in ("gain", "lo", "hi"): N.put(f"dg.{blk}.{var}.{split}.{m}.{fld}", s[fld])
    pd.DataFrame(nodes).to_csv(L.TABLES / "dose_nodes.csv", index=False); N.save(); print(len(noise), len(rew), len(nodes))


if __name__ == "__main__":
    main()
