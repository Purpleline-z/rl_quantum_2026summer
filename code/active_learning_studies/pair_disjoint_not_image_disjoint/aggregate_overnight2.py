#!/usr/bin/env python3
"""Overnight run 2 analysis (graph_exploration/OVERNIGHT2_PLAN.md).  usage: aggregate_overnight2.py P1|P2|P3 [--candidates a,b,c] [--markdown]
Per block (split x condition), per seed: gain over Random (mean of 5 draws) averaged over budgets 10/20/40/60, on log-loss, AUC and decisive accuracy; seed-level Wilcoxon,
Holm over the --candidates within each block (raw p for everything else); paired differences to controls / references listed in PAIRS."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent / "results" / "new_methods"
METRICS = {"test_decisive_log_loss": ("log-loss", True), "test_decisive_auc": ("AUC", False), "test_decisive_accuracy": ("accuracy", False)}
BLOCKS = [("A", "single"), ("A", "sequential"), ("B", "single"), ("B", "sequential")]
DIRS = {"P1": "graph_p1", "P2": "graph_p2", "P3": "graph_p3", "P4": "graph_p4", "P5": "graph_p5"}


def load(out: str, split: str, cond: str) -> pd.DataFrame:
    rows = []
    for path in sorted((ROOT / out / split / cond).glob("seed*_*.json")):
        d = json.loads(path.read_text())
        for budget, v in d.get("checkpoints", {}).items(): rows.append({"seed": d["seed"], "strategy": d["strategy"], "budget": int(budget), **{m: v.get(m, np.nan) for m in METRICS}})
    f = pd.DataFrame(rows)
    if len(f): f["family"] = f.strategy.str.replace(r"^random_r\d+$", "random", regex=True)
    return f


def holm(p: dict) -> dict:
    order = sorted(p, key=p.get); out = {}; run = 0.0
    for r, k in enumerate(order): run = max(run, min(1.0, (len(order) - r) * p[k])); out[k] = run
    return out


def wp(v):
    return float(wilcoxon(v).pvalue) if len(v) >= 6 and np.any(v != 0) else float("nan")


def fp(p):
    return "n/a" if np.isnan(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def gains(out, split, cond, metric):
    f = load(out, split, cond)
    if f.empty: return None
    lower = METRICS[metric][1]; sign = -1.0 if lower else 1.0
    per = f.groupby(["seed", "budget", "family"], as_index=False)[metric].mean()
    rnd = per[per.family == "random"][["seed", "budget", metric]].rename(columns={metric: "random"}); p = per.merge(rnd, on=["seed", "budget"]); p["gain"] = sign * (p[metric] - p.random)
    return p[p.family != "random"].groupby(["family", "seed"]).gain.mean().unstack(0)


def main(phase, candidates, pairs, markdown=False):
    out = DIRS[phase]; lines = []
    for split, cond in BLOCKS:
        res = {m: gains(out, split, cond, m) for m in METRICS}
        if res["test_decisive_log_loss"] is None: continue
        ll = res["test_decisive_log_loss"]; tests = {c: wp(ll[c].dropna().to_numpy()) for c in ll.columns}
        hp = holm({c: tests[c] for c in candidates if c in tests and not np.isnan(tests[c])})
        print(f"\n=== {phase} split {split}, {cond}: gain over Random (mean over seeds; share of seeds better; raw p; Holm over candidates) ===")
        print(f"{'rule':44s}{'n':>3s} {'log-loss':>9s} {'raw p':>7s} {'Holm':>6s} {'better':>7s}  {'AUC':>8s} {'acc':>8s}")
        for c in ll.columns:
            v = ll[c].dropna().to_numpy(); tag = "*" if c in candidates else " "
            print(f"{tag}{c:43s}{len(v):3d} {v.mean():+9.3f} {fp(tests[c]):>7s} {fp(hp[c]) if c in hp else '':>6s} {(v > 0).mean():7.0%}  {res['test_decisive_auc'][c].mean():+8.4f} {res['test_decisive_accuracy'][c].mean():+8.4f}")
        for a, b in pairs:
            if a in ll and b in ll:
                for m, (lab, _) in METRICS.items():
                    d = (res[m][a] - res[m][b]).dropna().to_numpy(); print(f"    {a} minus {b} [{lab}]: {d.mean():+.4f} (raw p {fp(wp(d))}, n={len(d)})")
    return lines


if __name__ == "__main__":
    phase = sys.argv[1]; cand = {"P1": "typed_decisive_coverage_unc,typed_decisive_coverage,typed_decisive_bald", "P2": "gvopt_lap,gvopt_type,gvopt_prop,gvopt_sigma,gvopt_lapsigma", "P3": "gvopt_lap3,typed_decisive_coverage", "P4": "gvopt_lap3,typed_decisive_coverage", "P5": "typed_decisive_coverage"}[phase]
    pairs = {"P1": [("typed_decisive_coverage", "typed_decisive_coverage_typeonly"), ("typed_decisive_coverage", "typed_decisive_coverage_shuffled"), ("typed_decisive_coverage", "vopt_u"), ("typed_decisive_coverage", "core_set_relation")],
             "P2": [(g, "vopt_u") for g in ("gvopt_lap", "gvopt_type", "gvopt_prop", "gvopt_sigma", "gvopt_lapsigma")], "P3": [("gvopt_lap3", "gvopt_lap3_shuffled"), ("gvopt_lap3", "vopt_u"), ("typed_decisive_coverage", "vopt_u"), ("gvopt_lap3", "typed_decisive_coverage")], "P4": [("gvopt_lap3", "vopt_u"), ("typed_decisive_coverage", "vopt_u")], "P5": [("typed_decisive_coverage", "typed_decisive_coverage_typeonly"), ("typed_decisive_coverage", "typed_decisive_coverage_shuffled"), ("typed_decisive_coverage", "vopt_u")]}[phase]
    for i, a in enumerate(sys.argv):
        if a == "--candidates": cand = sys.argv[i + 1]
        if a == "--pairs": pairs = [tuple(x.split(":")) for x in sys.argv[i + 1].split(",")]
    main(phase, [c for c in cand.split(",") if c], pairs)
