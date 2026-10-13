"""Tables of the recipe-search study (all numbers come from the result JSONs of ``new_directions/rc_run.py``; nothing is typed by hand).

  python3 rc_tables.py dev DIR [DIR ...] --ref lap3_k10_st4 [--out FILE]      development screen (every learner vs the reference, Holm over the screened family)
  python3 rc_tables.py types DIR [DIR ...] --ref ...                          per-type accuracy gains
  python3 rc_tables.py strata DIR [DIR ...] --ref ...                         anchor-near / anchor-far AUC and accuracy gains (needs the stored per-judgment scores)

Gain = learner minus reference (log-loss: reference minus learner, positive = better), paired on identical label sets, mean over budgets and draws, then over seeds; p = two-sided Wilcoxon over seeds,
interval = paired bootstrap over seeds.  ``far`` = both test images more than 0.03 (1 - cos) from every typed ideal-image anchor.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

NEW = Path(__file__).resolve().parents[2] / "new_directions"
FAR = 0.03
NEG = ("ll", "far_ll", "htr_ll", "t1x1_ll", "tc6x2_ll", "t13_ll")        # lower is better
TYPES = ("t1x1", "tc6x2", "t13", "htr")


def holm(p) -> np.ndarray:
    p = np.asarray(p, float); order = np.argsort(p); out = np.empty_like(p); run = 0.0
    for rank, i in enumerate(order): run = max(run, (len(p) - rank) * p[i]); out[i] = min(1.0, run)
    return out


def load(dirs, keep_d: bool = False) -> pd.DataFrame:
    rows = []
    for d in dirs:
        for f in sorted(glob.glob(str(Path(d) / "seed*.json"))):
            blob = json.loads(Path(f).read_text())
            for r in blob["records"]:
                r = dict(r)
                if keep_d: r["anc_min"] = blob["meta"]["anc_min"]; r["sign"] = blob["meta"]["sign"]; r["weight"] = blob["meta"]["weight"]; r["k"] = blob["meta"]["k"]
                else: r.pop("d", None)
                rows.append(r)
    df = pd.DataFrame(rows)
    return df.drop_duplicates(["seed", "split", "budget", "draw", "learner"], keep="last") if len(df) else df


def per_seed_gain(df: pd.DataFrame, cand: str, ref: str, split: str, metric: str) -> pd.Series:
    g = df[df.split == split]; a = g[g.learner == cand]; b = g[g.learner == ref]
    m = a.merge(b, on=["seed", "budget", "draw"], suffixes=("", "_r")); sign = -1.0 if metric in NEG else 1.0
    if metric not in m: return pd.Series(dtype=float)
    diff = sign * (m[metric] - m[f"{metric}_r"]); return diff.groupby(m.seed).mean().dropna()


def summarize(per_seed: pd.Series) -> dict:
    x = per_seed.to_numpy(); n = len(x)
    if n < 3: return {"n": n, "gain": float(np.mean(x)) if n else np.nan, "lo": np.nan, "hi": np.nan, "share": np.nan, "p": np.nan}
    rng = np.random.default_rng(0); boots = rng.choice(x, size=(4000, n)).mean(1)
    try: p = float(wilcoxon(x).pvalue) if np.any(x != 0) else 1.0
    except ValueError: p = 1.0
    return {"n": n, "gain": float(x.mean()), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)), "share": float((x > 0).mean()), "p": p}


def md(df: pd.DataFrame, fmt="+.4f") -> str:
    if not len(df): return "(no rows)"
    cols = list(df.columns); lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else (f"{v:.4f}" if c in ("p", "holm") else f"{v:{fmt}}"))
            else: cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def dev_table(df: pd.DataFrame, ref: str, learners=None, family_metrics=("auc", "acc")) -> pd.DataFrame:
    learners = [l for l in sorted(df.learner.unique()) if l != ref] if learners is None else learners; rows = []
    for l in learners:
        for split in ("A", "B"):
            row = {"learner": l, "split": split}
            for m in ("auc", "acc", "ll", "far_auc", "far_acc", "htr_acc", "htr_ll"):
                s = summarize(per_seed_gain(df, l, ref, split, m)); row[m] = s["gain"]
                if m in family_metrics: row[f"{m}_p"] = s["p"]; row["n"] = s["n"]
                if m in ("auc", "acc"): row[f"{m}_lo"], row[f"{m}_hi"] = s["lo"], s["hi"]
            rows.append(row)
    out = pd.DataFrame(rows)
    for m in family_metrics:   # Holm over the whole screened family (all learners x splits x metrics)
        pass
    ps = np.concatenate([out[f"{m}_p"].to_numpy() for m in family_metrics]); h = holm(np.nan_to_num(ps, nan=1.0)); n = len(out)
    for i, m in enumerate(family_metrics): out[f"{m}_holm"] = h[i * n:(i + 1) * n]
    return out


def shortlist_view(tab: pd.DataFrame) -> pd.DataFrame:
    """One row per learner: AUC / accuracy gain in A and B side by side and the number of Holm-significant (< 0.05) of the 4 tests."""
    rows = []
    for l, g in tab.groupby("learner"):
        a, b = g[g.split == "A"].iloc[0], g[g.split == "B"].iloc[0]
        rows.append({"learner": l, "n_seeds": int(a.n), "auc_A": a.auc, "auc_B": b.auc, "acc_A": a.acc, "acc_B": b.acc,
                     "all_positive": bool(min(a.auc, b.auc, a.acc, b.acc) > 0), "holm<0.05 (of 4)": int(sum(x < 0.05 for x in (a.auc_holm, b.auc_holm, a.acc_holm, b.acc_holm))),
                     "far_auc_A": a.far_auc, "far_auc_B": b.far_auc, "far_acc_A": a.far_acc, "far_acc_B": b.far_acc, "ll_A": a.ll, "ll_B": b.ll, "htr_acc_A": a.htr_acc, "htr_acc_B": b.htr_acc})
    return pd.DataFrame(rows).sort_values("learner").reset_index(drop=True)


def types_table(df: pd.DataFrame, ref: str, learners=None) -> pd.DataFrame:
    learners = [l for l in sorted(df.learner.unique()) if l != ref] if learners is None else learners; rows = []
    for l in learners:
        for split in ("A", "B"):
            row = {"learner": l, "split": split}
            for t in TYPES: row[f"{t}_acc"] = summarize(per_seed_gain(df, l, ref, split, f"{t}_acc"))["gain"]
            for t in TYPES: row[f"{t}_ll"] = summarize(per_seed_gain(df, l, ref, split, f"{t}_ll"))["gain"]
            rows.append(row)
    return pd.DataFrame(rows)


def strata_gain(dfd: pd.DataFrame, cand: str, ref: str, split: str, lo: float, hi: float, metric: str) -> pd.Series:
    """Per-seed gain on judgments with lo < anchor distance <= hi (pooled AUC / weighted accuracy of the stratum in each cell), from the stored scores."""
    g = dfd[dfd.split == split]; a = g[g.learner == cand].set_index(["seed", "budget", "draw"]); b = g[g.learner == ref].set_index(["seed", "budget", "draw"]); out = {}
    for key in a.index.intersection(b.index):
        ra, rb = a.loc[key], b.loc[key]; anc = np.asarray(ra["anc_min"]); m = (anc > lo) & (anc <= hi); s = np.asarray(ra["sign"])[m]
        if m.sum() < 4 or len(set(s)) < 2 and metric == "auc": continue
        w = np.asarray(ra["weight"])[m]; da, db = np.asarray(ra["d"])[m], np.asarray(rb["d"])[m]
        f = (lambda d: roc_auc_score(s > 0, d)) if metric == "auc" else (lambda d: float((w * ((d * s) > 0)).sum() / w.sum()))
        out.setdefault(key[0], []).append(f(da) - f(db))
    return pd.Series({k: float(np.mean(v)) for k, v in out.items()})


def strata_table(dfd: pd.DataFrame, ref: str, learners=None, bins=((-1, 0.001, "near (<=0.001)"), (0.001, FAR, "0.001-0.03"), (FAR, 9, "far (>0.03)"), (0.01, 9, "far (>0.01, RT1 threshold)"))) -> pd.DataFrame:
    learners = [l for l in sorted(dfd.learner.unique()) if l != ref] if learners is None else learners; rows = []
    for l in learners:
        for split in ("A", "B"):
            for lo, hi, name in bins:
                row = {"learner": l, "split": split, "stratum": name}
                for m in ("auc", "acc"):
                    s = summarize(strata_gain(dfd, l, ref, split, lo, hi, m)); row[m] = s["gain"]; row[f"{m}_lo"], row[f"{m}_hi"] = s["lo"], s["hi"]; row["n_seeds"] = s["n"]
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("mode", choices=["dev", "types", "strata", "counts"]); ap.add_argument("dirs", nargs="+"); ap.add_argument("--ref", default="lap3_k10_st4")
    ap.add_argument("--out", type=Path); ap.add_argument("--learners"); args = ap.parse_args(); dirs = [str((NEW / d) if not Path(d).exists() else d) for d in args.dirs]
    learners = args.learners.split(",") if args.learners else None; text = ""
    if args.mode == "counts":
        df = load(dirs); text = md(df.groupby(["learner", "split"]).agg(cells=("auc", "size"), seeds=("seed", "nunique")).reset_index(), fmt=".0f")
    elif args.mode == "dev":
        df = load(dirs); tab = dev_table(df, args.ref, learners); text = "### Shortlist view\n\n" + md(shortlist_view(tab)) + "\n\n### Full screen (per split)\n\n" + md(tab)
    elif args.mode == "types": text = md(types_table(load(dirs), args.ref, learners))
    else: text = md(strata_table(load(dirs, keep_d=True), args.ref, learners))
    if args.out: args.out.write_text(text + "\n")
    else: print(text)


if __name__ == "__main__": main()
