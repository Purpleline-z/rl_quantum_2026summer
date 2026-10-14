"""Honest-inference analysis driver (frozen by PREREGISTRATION_HI1.md).  numpy / pandas / scipy only.

  python3 hi_analyze.py main  --blocks conf=results/hi/conf1,results/hi/conf2  [--out results/hi/analysis] [--B 10000]       group-level inference for the comparison family (standard endpoint, all judgments)
  python3 hi_analyze.py strata --blocks ...                                                                                 anchor strata / HTR / clustered units sensitivities
  python3 hi_analyze.py ...        (see main())
Every row is one (block, comparison, split, metric, stratum) with the estimate, percentile intervals (group units; clustered units), seed-level inference of the same cells, randomisation p-values.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import hi_infer as H

HERE = Path(__file__).resolve().parent
CAL_T = {0: 1.352, 2: 0.682, 3: 0.293, 4: 0.630}      # per-type temperatures (types 0 (1x1), 2 c(6x2), 3 RT13, 4 HTR) of the calibration branch (CAL1_spec.json priors)
COMPS = [  # id, candidate, reference, label
    ("C1", "aw8", "baseline", "aw8 vs default head"), ("C2", "aw8_all", "baseline", "aw8_all vs default head"), ("C2b", "aw8_all", "aw8", "aw8_all vs aw8"), ("C3", "lap3_k10", "aw8_all", "lap3_k10 vs aw8_all"),
    ("C4", "lap3_k10_st4", "lap3_k10", "lap3_k10_st4 vs lap3_k10"), ("C5", "lap3_k10_st4", "aw8_all_st2lr0.5", "lap3_k10_st4 vs tuned graph-free control (primary)"), ("C6", "lap3_k10_st4", "aw8_all", "lap3_k10_st4 vs aw8_all"),
    ("C7", "aw8_all_st2lr0.5", "aw8_all", "graph-free schedule tuning vs aw8_all"), ("C9", "gmc_t_no_rt13", "lap3_k10", "RT13-off (lambda 10) vs lap3_k10"), ("C11", "gmc_t_no_rt13", "lap3_k10_st4", "RT13-off vs lap3_k10_st4"),
    ("C8", "lap3_k10_st4", "lap3_k10_st4_shuf", "lap3_k10_st4 vs its shuffled-graph control"), ("C10", "lap3_k10_st4_cal", "lap3_k10_st4", "per-type temperature calibration of lap3_k10_st4")]
FAMILY1 = ["C1", "C2", "C2b", "C3", "C4", "C5", "C6", "C7", "C9", "C11"]
STRATA = {"all": None, "near": lambda b: b.anc <= 0.001, "far03": lambda b: b.anc > 0.03, "far01": lambda b: b.anc > 0.01, "mid": lambda b: (b.anc > 0.001) & (b.anc <= 0.03), "htr": lambda b: b.k == 4}


def load(dirs) -> H.Block:
    blk = H.Block([HERE / d for d in dirs]) if all(isinstance(d, str) for d in dirs) else H.Block(dirs)
    if "lap3_k10_st4" in blk.D:      # post-hoc calibrated logits (no extra training)
        t = np.array([CAL_T[int(k)] for k in blk.k]); blk.D["lap3_k10_st4_cal"] = blk.D["lap3_k10_st4"] / t
    return blk


def unit_of(blk: H.Block, col: str) -> np.ndarray:
    return blk.clusters[col].to_numpy()


def analyze(blk: H.Block, name: str, comps, splits, metrics, strata, units=("unit_g",), B=10000, B_alt=4000, perm=20000, seed=0, verbose=True) -> pd.DataFrame:
    rows = []
    for cid, cand, ref, label in comps:
        if cand not in blk.D or ref not in blk.D: continue
        for split in splits:
            cell_mask = (blk.cells.split == split).to_numpy() if split != "all" else None
            for st in strata:
                mask = None if STRATA[st] is None else STRATA[st](blk)
                for m in metrics:
                    if st == "htr" and m == "auc": continue
                    t0 = time.monotonic(); des = H.Design(blk, cand, ref, m, mask=mask, cell_mask=cell_mask)
                    if des.nc == 0: continue
                    est = des.estimate(); pc = des.perm_contributions(); sl = H.seed_level(des, B=2000, seed=seed); row = dict(block=name, comp=cid, cand=cand, ref=ref, split=split, metric=m, stratum=st, est=est, n_cells=des.nc, n_seeds=sl["seed_n"],
                                                                                                                        seed_lo=sl["seed_lo"], seed_hi=sl["seed_hi"], seed_wilcoxon_p=sl["seed_wilcoxon_p"], seed_sd=sl["seed_sd"], n_groups=len(np.unique(des.gidx[des.gidx >= 0])))
                    for u in units:
                        ug = unit_of(blk, u); Bu = B if u == "unit_g" else B_alt; boots = H.group_bootstrap(des, ug, B=Bu, seed=seed); lo, hi = H.ci(boots); p, se_rand = H.signflip_p(pc, ug, n=perm if u == "unit_g" else perm // 4, seed=seed)
                        tag = "" if u == "unit_g" else "_" + u.replace("unit_", ""); row.update({f"lo{tag}": lo, f"hi{tag}": hi, f"se{tag}": float(np.nanstd(boots)), f"perm_p{tag}": p, f"rand_se{tag}": se_rand, f"n_units{tag}": len(np.unique(ug[np.unique(des.gidx[des.gidx >= 0])]))})
                    rows.append(row)
                    if verbose: print(f"{name} {cid} {split} {st} {m}: {time.monotonic() - t0:.1f}s", flush=True)
    return pd.DataFrame(rows)


def holm(p) -> np.ndarray:
    p = np.asarray(p, float); order = np.argsort(p); out = np.empty_like(p); run = 0.0
    for rank, i in enumerate(order): run = max(run, (len(p) - rank) * p[i]); out[i] = min(1.0, run)
    return out


def assess(df: pd.DataFrame, family=FAMILY1) -> pd.DataFrame:
    """Apply the pre-registered rule to the 'all' / pooled rows: Holm over family F1 (AUC and accuracy x splits A, B); label per comparison."""
    f1 = df[(df.comp.isin(family)) & (df.stratum == "all") & (df.metric.isin(["auc", "acc"])) & (df.split.isin(["A", "B"]))].copy(); f1["perm_p_holm"] = holm(f1.perm_p.to_numpy()); rows = []
    for cid, g in f1.groupby("comp", sort=False):
        a = {s: g[(g.split == s) & (g.metric == "auc")].iloc[0] for s in ("A", "B")}; c = {s: g[(g.split == s) & (g.metric == "acc")].iloc[0] for s in ("A", "B")}
        sup = all(a[s].lo > 0 and a[s].perm_p_holm < 0.05 and c[s].est > 0 for s in a); weaker = (not sup) and all(a[s].est > 0 for s in a)
        label = "SUPPORTED" if sup else ("WEAKER" if weaker else "UNSUPPORTED")
        rows.append(dict(comp=cid, label=label, **{f"auc_{s}": a[s].est for s in a}, **{f"auc_{s}_lo": a[s].lo for s in a}, **{f"auc_{s}_hi": a[s].hi for s in a}, **{f"auc_{s}_holm_p": a[s].perm_p_holm for s in a}, **{f"acc_{s}": c[s].est for s in c},
                         **{f"acc_{s}_lo": c[s].lo for s in c}, **{f"acc_{s}_hi": c[s].hi for s in c}, **{f"auc_{s}_c0.03_lo": a[s].get("lo_c0.03", np.nan) for s in a}, **{f"auc_{s}_b42_lo": a[s].get("lo_b42", np.nan) for s in a}))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("what"); ap.add_argument("--blocks", nargs="+", required=True, help="name=dir[,dir...]"); ap.add_argument("--out", type=Path, default=HERE / "results" / "hi" / "analysis"); ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--comps", default=None); ap.add_argument("--splits", default="A,B"); ap.add_argument("--metrics", default="auc,acc,ll"); ap.add_argument("--strata", default="all"); ap.add_argument("--units", default="unit_g"); ap.add_argument("--B_alt", type=int, default=4000)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); comps = [c for c in COMPS if a.comps is None or c[0] in a.comps.split(",")]
    for spec in a.blocks:
        name, dirs = spec.split("="); blk = load(dirs.split(",")); df = analyze(blk, name, comps, a.splits.split(","), a.metrics.split(","), a.strata.split(","), tuple(a.units.split(",")), B=a.B, B_alt=a.B_alt)
        df.to_csv(a.out / f"{a.what}_{name}.csv", index=False); print("wrote", a.out / f"{a.what}_{name}.csv", len(df))


if __name__ == "__main__": main()
