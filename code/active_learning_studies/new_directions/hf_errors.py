"""HTR failure analysis, prediction side: per-judgment error profile of the stored predictions (rt_a1.py output) pooled over seeds / splits / budgets / draws.

  python3 hf_errors.py --a1 results/hf/a1 --testmap results/hf/testmap_dev.csv --out results/hf/errors_dev   ->  errors_dev.md + errors_dev_*.csv

All rates are confidence-weighted (weight 1.0 'Confident'/none, 0.7 'Somewhat sure') exactly like the harness endpoint.  Intervals are bootstrap-by-seed (resampling seeds, both splits kept together).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import nd_core as core

HERE = core.HERE; HF = HERE / "results" / "hf"
CAND = "lap3_k10_st4"


def load(a1: Path, testmap: Path) -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in sorted(a1.glob("seed*_*.csv.gz"))]; d = pd.concat(frames, ignore_index=True)
    tm = pd.read_csv(testmap); j = pd.read_csv(HF / "judgments.csv")
    d = d.merge(tm, on=["seed", "split", "j"], how="left", validate="m:1"); assert d.jid.notna().all(); d["jid"] = d.jid.astype(int)
    keep = ["jid", "pair", "type", "labeller", "confidence", "run1", "run2", "idx1", "idx2", "temp1", "temp2", "same_run", "frame_gap", "cls_pair", "nn_class1", "nn_class2", "d_ideal_min1", "d_ideal_min2",
            "d_HTR1", "d_HTR2", "d_RT131", "d_RT132", "d_1x11", "d_1x12", "d_c6x21", "d_c6x22", "d_pair", "img1", "img2", "i1", "i2"]
    d = d.merge(j[keep], on="jid", how="left"); d["margin"] = d.sign * d.d; d["correct"] = d.margin > 0; d["ll"] = np.logaddexp(0.0, -d.margin); d["is_htr"] = d.k == 4
    # the judgment's winner / loser classes (1-NN to the ideal images)
    win1 = d.sign > 0; d["win_class"] = np.where(win1, d.nn_class1, d.nn_class2); d["lose_class"] = np.where(win1, d.nn_class2, d.nn_class1)
    d["d_HTR_win"] = np.where(win1, d.d_HTR1, d.d_HTR2); d["d_HTR_lose"] = np.where(win1, d.d_HTR2, d.d_HTR1)
    d["rule_htr_correct"] = d.d_HTR_win < d.d_HTR_lose            # "the frame closer to the ideal HTR images wins" (no pairwise labels, no learner)
    return d


def wmean(x, w): return float((x * w).sum() / w.sum())


def boot(df: pd.DataFrame, value: str, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Weighted mean of ``value`` with a bootstrap-by-seed 95% interval."""
    seeds = df.seed.unique(); g = {s: (float((df.loc[df.seed == s, value] * df.loc[df.seed == s, "weight"]).sum()), float(df.loc[df.seed == s, "weight"].sum())) for s in seeds}
    arr = np.array([g[s] for s in seeds]); rng = np.random.default_rng(seed); idx = rng.integers(0, len(seeds), size=(n, len(seeds)))
    est = arr[idx, 0].sum(1) / arr[idx, 1].sum(1); return arr[:, 0].sum() / arr[:, 1].sum(), float(np.quantile(est, .025)), float(np.quantile(est, .975))


def by(df: pd.DataFrame, col: str, value: str = "correct", min_n: int = 1) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(col, dropna=False):
        if g.jid.nunique() < min_n: continue
        m, lo, hi = boot(g, value); rows.append({col: key, "n_judgments": g.jid.nunique(), "n_rows": len(g), "rate": m, "lo": lo, "hi": hi, "ll": wmean(g.ll, g.weight)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--a1", type=Path, default=HF / "a1"); ap.add_argument("--testmap", type=Path, default=HF / "testmap_dev.csv"); ap.add_argument("--out", type=Path, default=HF / "errors_dev"); a = ap.parse_args()
    d = load(a.a1, a.testmap); L = [f"# HTR error analysis of `{CAND}` (dev seeds; {d.seed.nunique()} seeds x splits {sorted(d.split.unique())}, {len(d)} stored predictions)\n"]
    c = d[d.learner == CAND]; h = c[c.is_htr]
    L.append("## 1. Accuracy / log-loss by type and split (row-weighted, bootstrap-by-seed 95% interval)\n")
    rows = []
    for (split, k), g in c.groupby(["split", "k"]):
        m, lo, hi = boot(g, "correct"); rows.append({"split": split, "type": core.TYPE_NAMES[k], "n_unique_judgments": g.jid.nunique(), "acc": m, "lo": lo, "hi": hi, "ll": wmean(g.ll, g.weight)})
    L.append(pd.DataFrame(rows).round(3).to_markdown(index=False) + "\n")
    L.append("Comparison learner `aw8_all_st2lr0.5` (same label sets), HTR only:\n"); r2 = []
    for lrn, g in d[d.is_htr].groupby("learner"):
        for split, gg in g.groupby("split"): m, lo, hi = boot(gg, "correct"); r2.append({"learner": lrn, "split": split, "acc": m, "lo": lo, "hi": hi, "ll": wmean(gg.ll, gg.weight)})
    L.append(pd.DataFrame(r2).round(3).to_markdown(index=False) + "\n")
    L.append("## 2. The no-learner rule 'the frame nearer to the ideal HTR images wins' on the same pooled HTR test rows\n")
    L.append(f"rule accuracy {wmean(h.rule_htr_correct.astype(float), h.weight):.3f} vs learner {wmean(h.correct.astype(float), h.weight):.3f} (same rows, both splits, all budgets).  By budget / split:\n")
    t = h.assign(rule=h.rule_htr_correct.astype(float), learner=h.correct.astype(float)).groupby(["split", "budget"]).apply(lambda g: pd.Series({"rule": wmean(g.rule, g.weight), "learner": wmean(g.learner, g.weight), "n": len(g)}), include_groups=False)
    L.append(t.round(3).to_markdown() + "\n")
    L.append("## 3. Per-judgment profile (decisive HTR judgments that appear in at least one test set; sorted by learner error rate)\n")
    pj = h.groupby("jid").apply(lambda g: pd.Series({"n_rows": len(g), "n_cells": g[["seed", "split"]].drop_duplicates().shape[0], "err_rate": 1 - wmean(g.correct.astype(float), g.weight), "mean_margin": g.margin.mean(), "mean_ll": wmean(g.ll, g.weight),
                                                     "rule_ok": bool(g.rule_htr_correct.iloc[0]), "sign": g.sign.iloc[0], "labeller": g.labeller.iloc[0], "conf": g.confidence.iloc[0], "cls_pair": g.cls_pair.iloc[0], "win": g.win_class.iloc[0], "lose": g.lose_class.iloc[0],
                                                     "same_run": bool(g.same_run.iloc[0]), "temp1": g.temp1.iloc[0], "temp2": g.temp2.iloc[0], "d_pair": g.d_pair.iloc[0]}), include_groups=False).sort_values("err_rate", ascending=False)
    pj.to_csv(a.out.with_name(a.out.name + "_per_judgment.csv")); L.append(f"{len(pj)} distinct HTR judgments seen; error-rate distribution: " + ", ".join(f"{lab} {int(m.sum())}" for lab, m in (("always right (<5%)", pj.err_rate < .05), ("mostly right (5-35%)", (pj.err_rate >= .05) & (pj.err_rate < .35)), ("mixed (35-65%)", (pj.err_rate >= .35) & (pj.err_rate <= .65)), ("mostly wrong (65-95%)", (pj.err_rate > .65) & (pj.err_rate <= .95)), ("always wrong (>95%)", pj.err_rate > .95))) + ".\n")
    L.append(pj.head(20).round(3).to_markdown() + "\n")
    L.append("Share of all HTR errors contributed by the 5 / 10 worst judgments: %.2f / %.2f.\n" % tuple((pj.err_rate * pj.n_rows).sort_values(ascending=False).head(n).sum() / (pj.err_rate * pj.n_rows).sum() for n in (5, 10)))
    L.append("## 4. Where do failures live? (HTR; accuracy by stratum)\n")
    h = h.copy(); h["temp_hi"] = np.maximum(h.temp1, h.temp2); h["temp_bin"] = pd.cut(h.temp_hi, [0, 600, 800, 900, 2000]); h["anchor_bin"] = pd.cut(np.minimum(h.d_ideal_min1, h.d_ideal_min2), [-1, 0.001, 0.01, 0.03, 9])
    h["far_anchor_bin"] = pd.cut(np.maximum(h.d_ideal_min1, h.d_ideal_min2), [-1, 0.001, 0.01, 0.03, 9]); h["d_htr_gap"] = pd.cut((h.d_HTR_lose - h.d_HTR_win).abs(), [-1, 0.05, 0.1, 0.2, 0.4, 9])
    h["rule_agree"] = np.where(h.rule_htr_correct, "rule right", "rule wrong"); h["winner_is"] = h.win_class + " beats " + h.lose_class; h["pair_dist"] = pd.cut(h.d_pair * -1 + 1, [-1, .1, .3, .6, 2])
    h["budget_s"] = h.budget.astype(str)
    for col in ("cls_pair", "winner_is", "labeller", "confidence", "same_run", "temp_bin", "anchor_bin", "far_anchor_bin", "d_htr_gap", "rule_agree", "budget_s", "split"):
        t = by(h, col); L.append(f"### by `{col}`\n" + t.round(3).to_markdown(index=False) + "\n")
    # failure/rule contingency
    ct = pd.crosstab(h.correct, h.rule_htr_correct, rownames=["learner correct"], colnames=["rule correct"]); L.append("### learner vs rule (rows)\n" + ct.to_markdown() + "\n")
    # label-neighbour conflicts: do the failing judgments have a near-identical comparison with the opposite label?
    nb = pd.read_csv(HF / "label_neighbours.csv"); nb = nb[nb.type == "HTR"].set_index("jid"); pj2 = pj.join(nb[["nn_D", "nn_agree"]]); pj2["nn_conflict"] = pj2.nn_agree < 0
    L.append("### label-neighbour conflict of the judgments (nearest other HTR comparison, oriented): conflict = opposite label\n" + pj2.groupby(pd.cut(pj2.err_rate, [-.01, .05, .35, .65, 1.0]), observed=True).agg(n=("nn_D", "size"), conflicts=("nn_conflict", "sum"), median_nn_D=("nn_D", "median")).round(3).to_markdown() + "\n")
    out = a.out.with_suffix(".md"); out.write_text("\n".join(L)); print("\n".join(L))


if __name__ == "__main__": main()
