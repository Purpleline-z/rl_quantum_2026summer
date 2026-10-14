#!/usr/bin/env python3
"""Tables of the transductive / forward-in-time study (generated from the result JSONs of new_directions/results/td/<dir>; nothing is typed by hand).

  python3 td_tables.py dev   <dir> [--base BASE] [--learners a,b]      development table: every learner vs the baseline, windows FA / FB (forward) or budgets-pooled (standard)
  python3 td_tables.py confirm <dir> <spec.json>                       pre-registered test family with Holm, verdicts, bootstrap intervals (seed level and group level)

Conventions: per seed the metric difference is averaged over the cuts (forward; every cut equal weight) or over the budgets 10/20/40/60 (standard); accuracy and AUC: learner - baseline,
log-loss: baseline - learner (positive = better); intervals = paired bootstrap over seeds (95 %, 4000 resamples); p = two-sided Wilcoxon signed-rank over seeds (uncorrected unless Holm is stated).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

NI = Path(__file__).resolve().parents[2] / "new_directions"
RESULTS = NI / "results" / "td"
METRICS = ("auc", "acc", "ll", "far_auc", "far_acc", "far_ll")


def load(dirs, with_d=False) -> pd.DataFrame:
    rows = []
    for d in ([dirs] if isinstance(dirs, str) else dirs):
        for f in sorted((RESULTS / d).glob("*.json")):
            for r in json.loads(f.read_text()):
                if r["unit"] == "meta": continue
                if not with_d: r.pop("d", None)
                rows.append(r)
    return pd.DataFrame(rows)


def per_seed(df: pd.DataFrame, learner: str, base: str, window: str, metric: str, split=None) -> pd.Series:
    """per-seed mean over units (cuts / budgets) of the paired difference; units must exist for both learners."""
    sel = df[(df.window == window)] if split is None else df[(df.window == window) & (df.split == split)]
    a = sel[sel.learner == learner]; b = sel[sel.learner == base]
    key = ["seed", "unit"] + (["split"] if "split" in sel and sel.split.nunique() > 1 else []) + (["cut"] if "cut" in sel and sel.cut.notna().any() else [])
    key = [k for k in dict.fromkeys(key)]
    m = a.merge(b, on=key, suffixes=("", "_b")); sign = -1.0 if metric.endswith("ll") else 1.0
    diff = sign * (m[metric] - m[f"{metric}_b"]); diff = diff.where(np.isfinite(diff))
    return diff.groupby(m["seed"]).mean().dropna()


def summarize(x: pd.Series, seed_boot=4000) -> dict:
    v = x.to_numpy(); rng = np.random.default_rng(0)
    if len(v) < 3: return {"n": len(v), "gain": float(np.mean(v)) if len(v) else float("nan"), "lo": float("nan"), "hi": float("nan"), "share": float("nan"), "p": float("nan")}
    boots = rng.choice(v, size=(seed_boot, len(v))).mean(1); lo, hi = np.percentile(boots, [2.5, 97.5])
    try: p = float(wilcoxon(v).pvalue) if np.any(v != 0) else 1.0
    except ValueError: p = 1.0
    return {"n": len(v), "gain": float(v.mean()), "lo": float(lo), "hi": float(hi), "share": float((v > 0).mean()), "p": p}


def dev_table(df: pd.DataFrame, base: str, learners=None, windows=None, metrics=("auc", "acc", "ll", "far_auc")) -> pd.DataFrame:
    windows = windows or sorted(df.window.unique()); learners = learners or [l for l in sorted(df.learner.unique()) if l != base]; out = []
    for w in windows:
        for l in learners:
            if l == base or l not in set(df.learner): continue
            row = {"window": w, "learner": l}
            for m in metrics:
                s = summarize(per_seed(df, l, base, w, m)); row[f"{m}"] = s["gain"]; row[f"{m}_ci"] = f"[{s['lo']:+.3f},{s['hi']:+.3f}]"; row[f"{m}_p"] = s["p"]; row["n_seeds"] = s["n"]
            out.append(row)
    return pd.DataFrame(out)


def fmt(t: pd.DataFrame) -> str:
    cols = list(t.columns); lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in t.iterrows(): lines.append("| " + " | ".join((f"{v:+.4f}" if c in ("auc", "acc", "ll", "far_auc") else f"{v:.3f}" if isinstance(v, float) else str(v)) for c, v in r.items()) + " |")
    return "\n".join(lines)


def holm(ps: dict) -> dict:
    items = sorted(ps.items(), key=lambda kv: kv[1]); m = len(items); out = {}; running = 0.0
    for i, (k, p) in enumerate(items): running = max(running, min(1.0, (m - i) * p)); out[k] = running
    return out


def load_with_meta(dirs):
    """records (with d) and meta[(seed, cut|split, window)] = dict(group, anc_dist, sign, k, weight)."""
    rows, meta = [], {}
    for d in ([dirs] if isinstance(dirs, str) else dirs):
        for f in sorted((RESULTS / d).glob("*.json")):
            for r in json.loads(f.read_text()):
                if r["unit"] == "meta":
                    for w, m in r["windows"].items(): meta[(r["seed"], f.stem.split("_", 1)[1], w)] = m
                else: rows.append(r)
    return pd.DataFrame(rows), meta


def wauc(score, pos, w):
    """Weighted AUC (ties count 1/2)."""
    o = np.argsort(score, kind="mergesort"); s, p, w = score[o], pos[o], w[o]
    # group equal scores
    uniq, start = np.unique(s, return_index=True); wp = np.add.reduceat(w * p, start); wn = np.add.reduceat(w * (1 - p), start)
    cn = np.cumsum(wn) - wn; auc_num = (wp * (cn + 0.5 * wn)).sum(); den = wp.sum() * wn.sum()
    return auc_num / den if den > 0 else np.nan


def group_bootstrap(df, meta, learner, base, window, metric, n_boot=1000, seed=0):
    """Test-group bootstrap: resample the labelled pair groups (the units of the held-out sets) with replacement, recompute the metric of both learners on the judgments of the resampled groups for every
    (seed, cut | split, budget) unit, take the per-seed mean of the paired difference, then the mean over seeds.  Returns (lo, hi) of the 95 % interval.  AUC unweighted (as the endpoint), accuracy weighted by
    the confidence weight, log-loss not offered."""
    sel = df[df.window == window]; a = sel[sel.learner == learner]; b = sel[sel.learner == base]; units = []
    key = ["seed", "unit"] + [c for c in ("split", "cut") if c in sel and sel[c].notna().any()]
    mm = a.merge(b, on=key, suffixes=("", "_b"))
    for _, r in mm.iterrows():
        m = meta.get((r["seed"], f"cut{r['cut']}" if r["split"] == "F" else str(r["split"]), window))
        if m is None or not m["group"]: continue
        units.append((r["seed"], np.array(r["d"]), np.array(r["d_b"]), np.array(m["sign"]), np.array(m["weight"]) if "weight" in m else np.ones(len(m["sign"])), np.array(m["group"])))
    groups = sorted({g for u in units for g in u[5]}); gi = {g: i for i, g in enumerate(groups)}; rng = np.random.default_rng(seed); seeds = sorted({u[0] for u in units}); out = []
    idx = [np.array([gi[g] for g in u[5]]) for u in units]
    for _ in range(n_boot):
        mult = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups)); per = {}
        for u, ix in zip(units, idx):
            mw = mult[ix].astype(float)
            if mw.sum() == 0: continue
            sign, conf = u[3], u[4]; pos = (sign > 0).astype(float)
            if metric == "auc": v = wauc(u[1], pos, mw) - wauc(u[2], pos, mw)
            else: w = mw * conf; v = (w * ((u[1] * sign) > 0)).sum() / w.sum() - (w * ((u[2] * sign) > 0)).sum() / w.sum()
            if np.isfinite(v): per.setdefault(u[0], []).append(v)
        out.append(np.mean([np.mean(v) for v in per.values()]))
    return tuple(np.percentile(out, [2.5, 97.5]))


def test_family(df, meta, base, candidates, windows, metrics=("auc", "acc"), n_boot=1000):
    """rows for every (candidate, window, metric): gain vs ``base``, seed bootstrap interval, group bootstrap interval, share of seeds better, Wilcoxon p; Holm adjusted over the whole family."""
    rows = []
    for c in candidates:
        for w in windows:
            for m in metrics:
                s = summarize(per_seed(df, c, base, w, m)); gb = group_bootstrap(df, meta, c, base, w, m, n_boot) if meta else (float("nan"),) * 2
                rows.append({"learner": c, "window": w, "metric": m, "n_seeds": s["n"], "gain": s["gain"], "seed_ci": f"[{s['lo']:+.4f},{s['hi']:+.4f}]", "group_ci": f"[{gb[0]:+.4f},{gb[1]:+.4f}]", "share_better": s["share"], "p": s["p"]})
    t = pd.DataFrame(rows); adj = holm({i: p for i, p in enumerate(t.p)}); t["holm_p"] = [adj[i] for i in range(len(t))]; return t


def md(t: pd.DataFrame, floats=4) -> str:
    cols = list(t.columns); lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in t.iterrows(): lines.append("| " + " | ".join((f"{v:+.{floats}f}" if c in ("gain",) else f"{v:.4f}" if isinstance(v, float) else str(v)) for c, v in r.items()) + " |")
    return "\n".join(lines)


def main() -> None:
    cmd = sys.argv[1]
    if cmd == "dev":
        d = sys.argv[2]; base = "BASE"; learners = None
        for i, a in enumerate(sys.argv):
            if a == "--base": base = sys.argv[i + 1]
            if a == "--learners": learners = sys.argv[i + 1].split(",")
        df = load(d.split(",")); print(f"# {d}: {df.seed.nunique()} seeds, {len(df)} records, baseline {base}\n"); print(fmt(dev_table(df, base, learners)))
    else: raise SystemExit("see docstring")


if __name__ == "__main__": main()
