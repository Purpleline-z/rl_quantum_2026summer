"""HTR failure analysis: paired gains of a candidate learner over a baseline learner from stored rt_a1-format predictions (identical label sets, so every row is paired).

  python3 hf_gain.py --dir results/hf/stage2_dev --baseline lap3_k10_st4 [--cands a,b,c] [--out results/hf/stage2_dev_gain.md]

Unit of analysis: the seed.  For every (seed, split) the HTR gain is the confidence-weighted mean over all HTR test rows of all (budget, draw) cells (pooled HTR judgments), the paired difference candidate - baseline;
accuracy gain = + (higher is better), log-loss gain = baseline - candidate.  Other types: pooled weighted accuracy change per type.  All-type AUC change: AUC per (seed, split, budget, draw) cell, averaged per seed.
Inference: bootstrap over seeds (20000 resamples) for the 95% interval and a two-sided bootstrap p (sign-flip randomisation p is reported as a cross-check); Holm over the family declared by the caller.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import nd_core as core

TYPES = {0: "1x1", 2: "c6x2", 3: "RT13", 4: "HTR"}


def read(dirs) -> pd.DataFrame:
    frames = []
    for d in dirs: frames += [pd.read_csv(f, usecols=["seed", "split", "budget", "draw", "learner", "j", "d", "sign", "weight", "k"]) for f in sorted(Path(d).glob("seed*_*.csv.gz"))]
    d = pd.concat(frames, ignore_index=True); d["margin"] = d.sign * d.d; d["correct"] = (d.margin > 0).astype(float); d["ll"] = np.logaddexp(0.0, -d.margin); return d


def per_seed(d: pd.DataFrame, learner: str) -> pd.DataFrame:
    """One row per (seed, split) and learner: weighted HTR / per-type accuracy and log-loss, mean per-cell all-type AUC."""
    g = d[d.learner == learner]; rows = []
    for (seed, split), x in g.groupby(["seed", "split"]):
        r = {"seed": seed, "split": split}
        for k, name in TYPES.items():
            y = x[x.k == k]
            if len(y): r[f"acc_{name}"] = (y.correct * y.weight).sum() / y.weight.sum(); r[f"ll_{name}"] = (y.ll * y.weight).sum() / y.weight.sum()
        r["acc_all"] = (x.correct * x.weight).sum() / x.weight.sum(); r["ll_all"] = (x.ll * x.weight).sum() / x.weight.sum()
        aucs = [roc_auc_score(c.sign > 0, c.d) for _, c in x.groupby(["budget", "draw"]) if c.sign.nunique() == 2]; r["auc_all"] = float(np.mean(aucs))
        rows.append(r)
    return pd.DataFrame(rows)


def boot_mean(v: np.ndarray, n: int = 20000, seed: int = 0):
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(v), size=(n, len(v))); m = v[idx].mean(1)
    p = 2 * min((m <= 0).mean(), (m >= 0).mean()); p = max(p, 1.0 / n)
    signs = rng.choice([-1.0, 1.0], size=(n, len(v))); null = (signs * v).mean(1); p_flip = max((np.abs(null) >= abs(v.mean()) - 1e-15).mean(), 1.0 / n)
    return float(v.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975)), float(p), float(p_flip)


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps); out = [0.0] * len(ps); run = 0.0
    for rank, i in enumerate(order): run = max(run, min(1.0, (len(ps) - rank) * ps[i])); out[i] = run
    return out


def compare(d: pd.DataFrame, base: str, cands: list[str]) -> pd.DataFrame:
    rows = []
    for c in cands:
        keys = d[d.learner == c][["seed", "split", "budget", "draw"]].drop_duplicates(); dd = d.merge(keys, on=["seed", "split", "budget", "draw"])   # compare on the (budget, draw) cells both learners have
        b = per_seed(dd, base); x = per_seed(dd, c); m = b.merge(x, on=["seed", "split"], suffixes=("_b", "_c"))
        for split, g in m.groupby("split"):
            rec = {"cand": c, "split": split, "n_seeds": g.seed.nunique()}
            for metric, sgn in (("acc_HTR", 1), ("ll_HTR", -1), ("acc_1x1", 1), ("acc_c6x2", 1), ("acc_RT13", 1), ("auc_all", 1), ("acc_all", 1), ("ll_all", -1)):
                v = sgn * (g[f"{metric}_c"] - g[f"{metric}_b"]).to_numpy(); mean, lo, hi, p, pf = boot_mean(v); rec[f"{metric}"] = mean; rec[f"{metric}_lo"] = lo; rec[f"{metric}_hi"] = hi; rec[f"{metric}_p"] = p; rec[f"{metric}_pflip"] = pf
            rec["base_acc_HTR"] = g.acc_HTR_b.mean(); rec["cand_acc_HTR"] = g.acc_HTR_c.mean(); rec["base_ll_HTR"] = g.ll_HTR_b.mean(); rec["cand_ll_HTR"] = g.ll_HTR_c.mean(); rows.append(rec)
    return pd.DataFrame(rows)


def verdict(r: pd.DataFrame, family: int | None = None) -> pd.DataFrame:
    """Pre-registered success rule per candidate: HTR accuracy and log-loss gains Holm-significant (alpha .05) in BOTH splits, other types lose <= 0.01 accuracy, all-type AUC not worse by > 0.002."""
    out = []
    for c, g in r.groupby("cand"):
        ps = [g.loc[g.split == s, f"{m}_p"].iloc[0] for s in ("A", "B") for m in ("acc_HTR", "ll_HTR")]; hp = holm(ps); gains = [g.loc[g.split == s, m].iloc[0] for s in ("A", "B") for m in ("acc_HTR", "ll_HTR")]
        sig = all(h < 0.05 and gv > 0 for h, gv in zip(hp, gains)); others = all(g.loc[g.split == s, f"acc_{t}"].iloc[0] >= -0.01 for s in ("A", "B") for t in ("1x1", "c6x2", "RT13")); auc = all(g.loc[g.split == s, "auc_all"].iloc[0] >= -0.002 for s in ("A", "B"))
        out.append({"cand": c, "holm_A_acc": hp[0], "holm_A_ll": hp[1], "holm_B_acc": hp[2], "holm_B_ll": hp[3], "HTR gains significant both splits": sig, "other types lose <= 0.01": others, "AUC not worse by > 0.002": auc, "SUCCESS": bool(sig and others and auc)})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", nargs="+", required=True); ap.add_argument("--baseline", default="lap3_k10_st4"); ap.add_argument("--cands", default=""); ap.add_argument("--out", type=Path); a = ap.parse_args()
    d = read(a.dir); cands = a.cands.split(",") if a.cands else sorted(set(d.learner) - {a.baseline}); r = compare(d, a.baseline, cands)
    show = ["cand", "split", "n_seeds", "base_acc_HTR", "cand_acc_HTR", "acc_HTR", "acc_HTR_lo", "acc_HTR_hi", "acc_HTR_p", "base_ll_HTR", "cand_ll_HTR", "ll_HTR", "ll_HTR_lo", "ll_HTR_hi", "ll_HTR_p", "acc_1x1", "acc_c6x2", "acc_RT13", "auc_all", "acc_all", "ll_all"]
    text = "## gains over `%s` (positive = better; bootstrap over seeds)\n\n" % a.baseline + r[show].round(4).to_markdown(index=False) + "\n\n## verdict\n\n" + verdict(r).round(4).to_markdown(index=False) + "\n"
    print(text); (a.out and a.out.write_text(text))


if __name__ == "__main__": main()
