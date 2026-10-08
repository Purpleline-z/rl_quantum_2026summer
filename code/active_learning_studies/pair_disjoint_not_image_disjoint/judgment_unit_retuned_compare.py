#!/usr/bin/env python3
"""Old schedule (results/judgment_unit_study) versus per-budget re-tuned schedule (results/judgment_unit_study_retuned): everything generated from the aggregate CSVs.

usage: judgment_unit_retuned_compare.py            -> writes results/judgment_unit_study_retuned/comparison/*.csv and comparison_tables.md (and prints the markdown)
Pieces: Random's levels per budget (and the initial-rows-only model) under both schedules; side-by-side Holm table (variants with Holm p < 0.05 under either schedule);
Spearman correlation of the 32 variants' mean gains over Random between the schedules, per table and metric; a short count of nominal / Holm significance per table."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from make_endpoint_report_tables import NAMES

HERE = Path(__file__).resolve().parent
OLD = HERE / "results" / "judgment_unit_study"; NEW = HERE / "results" / "judgment_unit_study_retuned"
NAMES = {**NAMES, "random_pair_type": "Random pair, random type"}
RUNS = (("A_groups", "Split A"), ("B_groups", "Split B")); CONDITIONS = ("single", "sequential")
METRICS = (("test_decisive_log_loss", "log-loss"), ("test_calibrated_log_loss", "cal. log-loss"), ("test_decisive_auc", "AUC"), ("test_decisive_accuracy", "accuracy (secondary)"))
BUDGETS = (10, 20, 40, 60)


def vs_random(root: Path, run: str, condition: str, metric: str):
    path = root / run / "analysis" / f"{condition}_{metric}_vs_random.csv"
    return pd.read_csv(path, index_col=0) if path.exists() and path.stat().st_size > 5 else None


def by_budget(root: Path, run: str, condition: str, metric: str):
    path = root / run / "analysis" / f"{condition}_{metric}_by_strategy_and_budget.csv"
    return pd.read_csv(path) if path.exists() else None


def p(v) -> str:
    return "<0.001" if v < .001 else f"{v:.3f}"


def build() -> tuple[str, dict]:
    out = {}; md = []
    # 1. Random's levels
    levels = []
    for run, split in RUNS:
        for condition in CONDITIONS:
            for metric, label in METRICS[:1] + METRICS[2:]:
                for tag, root in (("old", OLD), ("new", NEW)):
                    t = by_budget(root, run, condition, metric)
                    if t is None: continue
                    for b in BUDGETS: levels.append({"split": split, "condition": condition, "metric": label, "schedule": tag, "budget": b, "random_mean": float(t[(t.family == "random") & (t.budget == b)]["mean"].iloc[0])})
    levels = pd.DataFrame(levels); out["random_levels"] = levels
    md.append("### Random's level by budget (mean over 35 seeds; Random = mean of its 5 draws)\n")
    md.append("| Split | condition | metric | schedule | " + " | ".join(f"{b} judgments" for b in BUDGETS) + " |"); md.append("|---|---|---|---|" + "---:|" * len(BUDGETS))
    for (split, condition, metric), g in levels.groupby(["split", "condition", "metric"], sort=False):
        for tag in ("old", "new"):
            h = g[g.schedule == tag].sort_values("budget")
            if len(h): md.append(f"| {split} | {condition} | {metric} | {tag} | " + " | ".join(f"{v:.3f}" for v in h.random_mean) + " |")
    initial = []
    for run, split in RUNS:
        for tag, root in (("old", OLD), ("new", NEW)):
            cells = [json.loads(q.read_text()) for q in sorted((root / run / "single").glob("seed*_initial_only.json"))] if (root / run / "single").exists() else []
            if cells: initial.append({"split": split, "schedule": tag, "n_seeds": len(cells), **{m: float(np.nanmean([c[m] for c in cells])) for m in ("test_decisive_log_loss", "test_decisive_auc", "test_decisive_accuracy")}})
    initial = pd.DataFrame(initial); out["initial_only"] = initial
    md.append("\n### Initial rows only (no acquired judgment), test metrics, mean over seeds\n"); md.append("| Split | schedule | seeds | log-loss | AUC | accuracy |"); md.append("|---|---|---:|---:|---:|---:|")
    for r in initial.itertuples(): md.append(f"| {r.split} | {r.schedule} | {r.n_seeds} | {r.test_decisive_log_loss:.3f} | {r.test_decisive_auc:.3f} | {r.test_decisive_accuracy:.3f} |")
    # 2. side by side
    rows = []
    for run, split in RUNS:
        for condition in CONDITIONS:
            for metric, label in METRICS:
                o, n = vs_random(OLD, run, condition, metric), vs_random(NEW, run, condition, metric)
                if o is None or n is None: continue
                for family in o.index.intersection(n.index):
                    rows.append({"split": split, "condition": condition, "metric": label, "family": family, "old_gain": o.loc[family, "mean_gain"], "old_p": o.loc[family, "wilcoxon_p"], "old_holm": o.loc[family, "holm_p"], "old_frac_better": o.loc[family, "frac_better"],
                                 "new_gain": n.loc[family, "mean_gain"], "new_p": n.loc[family, "wilcoxon_p"], "new_holm": n.loc[family, "holm_p"], "new_frac_better": n.loc[family, "frac_better"]})
    table = pd.DataFrame(rows); out["side_by_side"] = table
    md.append("\n### Holm-significant variants (p < 0.05 within a table of 32) under the old and the re-tuned schedule\n")
    md.append("Gain = improvement over Random averaged over budgets 10/20/40/60 (positive is better, also for log-loss). `n.s.` = Holm p >= 0.05 (value shown).\n")
    md.append("| Table | metric | variant | old gain (Holm p) | re-tuned gain (Holm p) | status |"); md.append("|---|---|---|---|---|---|")
    summary = []
    for (split, condition, metric), g in table.groupby(["split", "condition", "metric"], sort=False):
        so, sn = set(g[g.old_holm < .05].family), set(g[g.new_holm < .05].family)
        summary.append({"split": split, "condition": condition, "metric": metric, "holm_old": len(so), "holm_new": len(sn), "holm_both": len(so & sn), "nominal_old": int((g.old_p < .05).sum()), "nominal_new": int((g.new_p < .05).sum()),
                        "n_variants": len(g), "spearman_rho": spearmanr(g.old_gain, g.new_gain).statistic, "spearman_p": spearmanr(g.old_gain, g.new_gain).pvalue,
                        "mean_abs_gain_old": g.old_gain.abs().mean(), "mean_abs_gain_new": g.new_gain.abs().mean()})
        for family in sorted(so | sn, key=lambda f: -max(abs(g.set_index("family").loc[f, "old_gain"]), abs(g.set_index("family").loc[f, "new_gain"]))):
            r = g.set_index("family").loc[family]; status = "significant under both" if family in so & sn else ("only old schedule" if family in so else "only re-tuned schedule")
            md.append(f"| {split}, {condition} | {metric} | {NAMES.get(family, family)} | {r.old_gain:+.3f} ({p(r.old_holm)}) | {r.new_gain:+.3f} ({p(r.new_holm)}) | {status} |")
    summary = pd.DataFrame(summary); out["summary"] = summary
    md.append("\n### Per table: significance counts and rank agreement of the 32 variants' gains (old vs re-tuned schedule)\n")
    md.append("| Table | metric | Holm<0.05 old | Holm<0.05 re-tuned | in both | nominal p<0.05 old / re-tuned | Spearman rho (p) | mean abs gain old / re-tuned |"); md.append("|---|---|---:|---:|---:|---:|---|---|")
    for r in summary.itertuples(): md.append(f"| {r.split}, {r.condition} | {r.metric} | {r.holm_old} | {r.holm_new} | {r.holm_both} | {r.nominal_old} / {r.nominal_new} | {r.spearman_rho:+.2f} ({p(r.spearman_p)}) | {r.mean_abs_gain_old:.3f} / {r.mean_abs_gain_new:.3f} |")
    # pooled Spearman per metric
    md.append("\n### Spearman correlation pooled over the four tables (128 variant-table gains per metric)\n"); md.append("| metric | n | Spearman rho | p |"); md.append("|---|---:|---:|---|")
    pooled = []
    for metric, g in table.groupby("metric", sort=False):
        s = spearmanr(g.old_gain, g.new_gain); pooled.append({"metric": metric, "n": len(g), "rho": s.statistic, "p": s.pvalue}); md.append(f"| {metric} | {len(g)} | {s.statistic:+.2f} | {p(s.pvalue)} |")
    out["pooled_spearman"] = pd.DataFrame(pooled)
    return "\n".join(md), out


def main() -> None:
    md, out = build(); folder = NEW / "comparison"; folder.mkdir(parents=True, exist_ok=True)
    for key, frame in out.items(): frame.to_csv(folder / f"{key}.csv", index=False)
    (folder / "comparison_tables.md").write_text(md); print(md)


if __name__ == "__main__":
    main()
