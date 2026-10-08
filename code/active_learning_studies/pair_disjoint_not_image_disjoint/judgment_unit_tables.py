#!/usr/bin/env python3
"""Markdown tables for JUDGMENT_UNIT_RESULTS.md, generated from the aggregate CSVs (no hand-copied numbers).

usage: judgment_unit_tables.py <run, e.g. A_groups> <old tag prefix: single_all | sequential_all | single_all_cells_c2split | sequential_all_sequential_cells_c2split> <condition>
       judgment_unit_tables.py pools            (pool sizes per split)
       judgment_unit_tables.py budget <run> <condition> <metric label>   (per-budget top-3 table)
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import judgment_unit_study as study
from make_endpoint_report_tables import NAMES

OLD = study.HERE / "results" / "pair_endpoint_study"
NAMES = {**NAMES, "random_pair_type": "Random pair, random type"}


def fmt_p(v) -> str:
    return "–" if pd.isna(v) else ("<0.001" if v < .001 else f"{v:.3f}")


def significance(run: str, old_prefix: str, condition: str) -> str:
    new = {m: pd.read_csv(study.OUT / run / "analysis" / f"{condition}_{m}_vs_random.csv", index_col=0) for m in ("test_decisive_log_loss", "test_calibrated_log_loss", "test_decisive_auc")
           if (study.OUT / run / "analysis" / f"{condition}_{m}_vs_random.csv").exists()}
    old = {m: pd.read_csv(OLD / f"{old_prefix}_{m}_vs_random.csv", index_col=0) for m in ("test_decisive_log_loss", "test_decisive_auc") if (OLD / f"{old_prefix}_{m}_vs_random.csv").exists()}
    base = new["test_decisive_log_loss"]; cal = "test_calibrated_log_loss" in new and not new["test_calibrated_log_loss"].empty
    head = "| Strategy | log-loss gain | Holm p | " + ("cal. log-loss gain | Holm p | " if cal else "") + "AUC gain | Holm p | seeds better (log-loss) | OLD group-unit: log-loss gain | Holm p | AUC gain | Holm p |"
    lines = [head, "|---|" + "---:|" * (head.count("|") - 2)]
    for family in base.index:
        row = [NAMES.get(family, family), f"{base.loc[family, 'mean_gain']:+.3f}", fmt_p(base.loc[family, "holm_p"])]
        if cal: row += [f"{new['test_calibrated_log_loss'].loc[family, 'mean_gain']:+.3f}", fmt_p(new["test_calibrated_log_loss"].loc[family, "holm_p"])]
        row += [f"{new['test_decisive_auc'].loc[family, 'mean_gain']:+.3f}", fmt_p(new["test_decisive_auc"].loc[family, "holm_p"]), f"{base.loc[family, 'frac_better']:.0%}"]
        for m in ("test_decisive_log_loss", "test_decisive_auc"):
            o = old.get(m); row += [f"{o.loc[family, 'mean_gain']:+.3f}", fmt_p(o.loc[family, "holm_p"])] if o is not None and family in o.index else ["–", "–"]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def pools() -> str:
    lines = ["| Split / initial set | seeds | initial judgments (mean, range) | pool judgments (mean, range) | pool pairs (mean, range) | judgments per pool pair | validation decisive judgments | test decisive judgments (mean, range) |", "|---|---:|---|---|---|---:|---:|---|"]
    import run_pair_endpoint_study as single
    import frozen_encoder_reward_head as frozen
    for run in sorted(p.name for p in study.OUT.iterdir() if p.is_dir()):
        manifests = [json.loads(p.read_text()) for p in sorted((study.OUT / run).glob("manifest_seed*.json"))]
        if not manifests: continue
        cells = [json.loads(p.read_text()) for p in sorted((study.OUT / run / "single").glob("seed*_initial_only.json"))]
        ini = [len(m["initial"]) for m in manifests]; pj = [len(m["pool"]) for m in manifests]; pp = [len({x[0] for x in m["pool"]}) for m in manifests]
        test = [c["test_decisive_rows"] for c in cells]; val = [json.loads(p.read_text()).get("validation_decisive_rows") for p in []]
        rng = lambda v: f"{np.mean(v):.1f} ({min(v)}–{max(v)})"
        lines.append(f"| {run} | {len(manifests)} | {rng(ini)} | {rng(pj)} | {rng(pp)} | {np.mean(pj) / np.mean(pp):.2f} | {'20 groups' if run.startswith('A') else 'none'} | {rng(test)} |")
    return "\n".join(lines)


def per_budget(run: str, condition: str, metric: str, k: int = 3) -> str:
    table = pd.read_csv(study.OUT / run / "analysis" / f"by_budget_gain_vs_random_{condition}.csv"); table = table[(table.metric == metric)]
    lines = [f"| Budget (judgments) | top {k} by mean gain over Random (Holm p within the budget) | variants with Holm p < 0.05 |", "|---:|---|---|"]
    for budget, h in table.groupby("budget"):
        top = h.sort_values("mean_gain", ascending=False).head(k); sig = h[h.holm_p < .05].sort_values("mean_gain", ascending=False)
        lines.append(f"| {budget} | " + "; ".join(f"{NAMES.get(r.family, r.family)} {r.mean_gain:+.3f} ({r.holm_p:.2f})" for r in top.itertuples()) + " | " + (", ".join(f"{NAMES.get(r.family, r.family)} {r.mean_gain:+.3f}" for r in sig.itertuples()) or "none") + " |")
    return "\n".join(lines)


def random_levels(run: str, condition: str, old_prefix: str) -> str:
    new = pd.read_csv(study.OUT / run / "analysis" / f"{condition}_test_decisive_log_loss_by_strategy_and_budget.csv"); newa = pd.read_csv(study.OUT / run / "analysis" / f"{condition}_test_decisive_auc_by_strategy_and_budget.csv")
    old = pd.read_csv(OLD / f"{old_prefix}_test_decisive_log_loss_by_strategy_and_budget.csv"); olda = pd.read_csv(OLD / f"{old_prefix}_test_decisive_auc_by_strategy_and_budget.csv")
    f = lambda t, b: t[(t.family == "random") & (t.budget == b)]["mean"].iloc[0]
    lines = ["| Budget | NEW: judgments | Random log-loss | Random AUC | OLD: groups | Random log-loss | Random AUC |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for b in study.BUDGETS: lines.append(f"| {b} | {b} | {f(new, b):.3f} | {f(newa, b):.3f} | {b} | {f(old, b):.3f} | {f(olda, b):.3f} |")
    return "\n".join(lines)


def main() -> None:
    if sys.argv[1] == "pools": print(pools())
    elif sys.argv[1] == "budget": print(per_budget(sys.argv[2], sys.argv[3], sys.argv[4]))
    elif sys.argv[1] == "levels": print(random_levels(sys.argv[2], sys.argv[3], sys.argv[4]))
    else: print(significance(sys.argv[1], sys.argv[2], sys.argv[3]))


if __name__ == "__main__":
    main()
