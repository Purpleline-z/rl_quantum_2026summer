#!/usr/bin/env python3
"""Graph-aware strategies against Random and against their shuffled-graph controls, judgment-unit study.

usage: aggregate_graph_vs_random.py [A|B] [single|sequential]   (reads results/judgment_unit_study/<split>_groups/<condition>)
Per seed: gain over Random (mean of the five Random draws) averaged over budgets 10/20/40/60; Wilcoxon signed-rank over seeds, Holm over the non-random
strategies in the table (so the correction depends on how many strategies were run).  Also the paired difference real graph minus shuffled graph, and per-budget means.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent / "results" / "judgment_unit_study"
METRICS = {"test_decisive_log_loss": ("log-loss", True), "test_decisive_auc": ("AUC", False)}
CONTROLS = {"graph_centrality_uncertainty": "graph_centrality_uncertainty_shuffled", "graph_bridge_uncertainty": "graph_bridge_uncertainty_shuffled", "graph_core_set": "graph_core_set_shuffled"}


def load(split: str, condition: str) -> pd.DataFrame:
    rows = []
    for path in sorted((ROOT / f"{split}_groups" / condition).glob("seed*_*.json")):
        d = json.loads(path.read_text())
        for budget, values in d.get("checkpoints", {}).items(): rows.append({"seed": d["seed"], "strategy": d["strategy"], "budget": int(budget), **{k: values[k] for k in METRICS if k in values}})
    frame = pd.DataFrame(rows); frame["family"] = frame.strategy.str.replace(r"^random_r\d+$", "random", regex=True); return frame


def holm(p: dict) -> dict:
    order = sorted(p, key=p.get); out = {}; running = 0.0
    for rank, key in enumerate(order): running = max(running, min(1.0, (len(order) - rank) * p[key])); out[key] = running
    return out


def wilcoxon_p(v: np.ndarray) -> float:
    return float(wilcoxon(v).pvalue) if len(v) >= 6 and np.any(v != 0) else float("nan")


def report(split: str, condition: str) -> None:
    cells = load(split, condition)
    if cells.empty: print(f"no cells for split {split} {condition}"); return
    for metric, (label, lower) in METRICS.items():
        sign = -1.0 if lower else 1.0; per = cells.groupby(["seed", "budget", "family"], as_index=False)[metric].mean()
        random = per[per.family == "random"][["seed", "budget", metric]].rename(columns={metric: "random"}); paired = per.merge(random, on=["seed", "budget"]); paired["gain"] = sign * (paired[metric] - paired.random)
        seed_level = paired[paired.family != "random"].groupby(["family", "seed"]).gain.mean().reset_index(); tests = {f: wilcoxon_p(g.gain.to_numpy()) for f, g in seed_level.groupby("family")}
        adjusted = holm({k: v for k, v in tests.items() if not np.isnan(v)})
        table = seed_level.groupby("family").gain.agg(mean_gain="mean", sd="std", n="count", frac_better=lambda x: (x > 0).mean()); table["wilcoxon_p"] = pd.Series(tests); table["holm_p"] = pd.Series(adjusted)
        print(f"\n=== split {split}, {condition}, {label} ({'lower' if lower else 'higher'} is better): gain over Random, per seed averaged over budgets ===")
        print(table.sort_values("mean_gain", ascending=False).round(4).to_string())
        by_budget = paired[paired.family != "random"].groupby(["family", "budget"]).gain.mean().unstack(); print("per-budget mean gain:"); print(by_budget.round(4).to_string())
        wide = seed_level.pivot(index="seed", columns="family", values="gain"); rows = []
        for real, fake in CONTROLS.items():
            if real in wide and fake in wide:
                d = (wide[real] - wide[fake]).dropna().to_numpy(); rows.append({"real graph minus shuffled": real, "mean_diff": d.mean(), "n": len(d), "wilcoxon_p": wilcoxon_p(d)})
        if rows: print("real graph minus shuffled-graph control (positive = the graph information helps):"); print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    splits = [sys.argv[1]] if len(sys.argv) > 1 else ["A", "B"]; conditions = [sys.argv[2]] if len(sys.argv) > 2 else ["single", "sequential"]
    for s in splits:
        for c in conditions: report(s, c)
