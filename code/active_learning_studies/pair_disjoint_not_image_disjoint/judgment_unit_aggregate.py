#!/usr/bin/env python3
"""Aggregate the judgment-unit study with the statistics of ``aggregate_pair_endpoint.py`` (imported, not copied).

usage: judgment_unit_aggregate.py <run folder name, e.g. A_groups>
For both conditions (single, sequential) and every metric: per-seed gain over Random averaged over budgets, Wilcoxon signed-rank with Holm correction over the
strategy variants, Friedman omnibus, mean rank; plus the per-budget gain tables of ``analyze_by_budget.py``.  Random = mean of its five draws within a seed.
The extra baseline ``random_pair_type`` is compared with Random separately and is not part of the Holm family.
Outputs go to results/judgment_unit_study/<run>/analysis/.
"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import aggregate_pair_endpoint as agg
import judgment_unit_strategies as ju
import judgment_unit_study as study

ROOT = study.OUT
SEEDS = (42, 79, 123, 202, 303) + tuple(range(400, 430))
BUDGET_METRICS = (("test_decisive_log_loss", True, "log-loss"), ("test_decisive_auc", False, "AUC"), ("test_calibrated_log_loss", True, "cal. log-loss"))
EXPECTED_STRATEGY_FILES = len(study.strategy_names())


def load(run: Path, condition: str) -> pd.DataFrame:
    rows = []
    for path in sorted((run / condition).glob("seed*_*.json")):
        d = json.loads(path.read_text())
        if d["strategy"] == "initial_only": continue
        for budget, values in d["checkpoints"].items():
            rows.append({"seed": d["seed"], "strategy": d["strategy"], "budget": int(budget), **{k: v for k, v in values.items() if k != "selected"}, "pool_judgments": d["pool_judgments"]})
    frame = pd.DataFrame(rows); frame["family"] = frame.strategy.map(study.family_of); return frame


def check_counts(frame: pd.DataFrame, label: str) -> None:
    """Recount the cells instead of trusting a hard-coded number."""
    seeds = frame.seed.nunique(); per_seed = frame.groupby("seed").strategy.nunique()
    expected = len(study.strategy_names())
    print(f"[{label}] {len(frame)} cells = {seeds} seeds x {frame.strategy.nunique()} strategy names x {frame.budget.nunique()} budgets; "
          f"expected {seeds * expected * len(study.BUDGETS)}; seeds with all {expected} strategies: {int((per_seed == expected).sum())}/{seeds}; families: {frame.family.nunique()} "
          f"({frame[~frame.family.isin(['random', ju.EXTRA_BASELINE])].family.nunique()} non-random variants + random + {ju.EXTRA_BASELINE})")
    assert len(frame) == seeds * expected * len(study.BUDGETS), "cell count does not match the design"
    assert (frame.n_revealed == frame.budget).all(), "a cell revealed a number of judgments different from its budget"


def per_budget(frame: pd.DataFrame, run_out: Path, condition: str) -> pd.DataFrame:
    rows = []
    for metric, lower, label in BUDGET_METRICS:
        if frame[metric].isna().all(): continue
        ps = frame.groupby(["seed", "budget", "family"], as_index=False)[metric].mean(); rnd = ps[ps.family == "random"][["seed", "budget", metric]].rename(columns={metric: "random"})
        paired = ps.merge(rnd, on=["seed", "budget"]); paired["gain"] = (-1.0 if lower else 1.0) * (paired[metric] - paired.random)
        paired = paired[~paired.family.isin(["random", ju.EXTRA_BASELINE])]
        for budget, g in paired.groupby("budget"):
            tests = {fam: (float(wilcoxon(h.gain.to_numpy()).pvalue) if np.any(h.gain.to_numpy() != 0) else 1.0) for fam, h in g.groupby("family")}; adj = agg.holm(tests)
            for fam, h in g.groupby("family"): rows.append({"condition": condition, "metric": label, "budget": budget, "family": fam, "mean_gain": h.gain.mean(), "p": tests[fam], "holm_p": adj[fam], "n": len(h)})
    out = pd.DataFrame(rows); out.to_csv(run_out / f"by_budget_gain_vs_random_{condition}.csv", index=False); return out


def describe_best(table: pd.DataFrame) -> None:
    for (condition, label), g in table.groupby(["condition", "metric"]):
        print(f"\n=== {condition}, {label}: top 3 at each budget (mean gain over Random; Holm p within that budget across {g.family.nunique()} variants)")
        for budget, h in g.groupby("budget"):
            top = h.sort_values("mean_gain", ascending=False).head(3); sig = h[h.holm_p < .05].sort_values("mean_gain", ascending=False)
            print(f"  budget {budget:>2}: " + "; ".join(f"{r.family} {r.mean_gain:+.3f} (p={r.holm_p:.2f})" for r in top.itertuples()) + (f" | Holm<0.05: {', '.join(f'{r.family} {r.mean_gain:+.3f}' for r in sig.itertuples())}" if len(sig) else " | none with Holm<0.05"))


def touched(frame: pd.DataFrame, run_out: Path, condition: str) -> None:
    summary = frame.groupby(["family", "budget"], as_index=False).agg(n_pairs_touched=("n_pairs_touched", "mean"), n_revealed=("n_revealed", "mean"))
    summary.to_csv(run_out / f"pairs_touched_{condition}.csv", index=False)


def main() -> None:
    name = sys.argv[1]; run = ROOT / name; run_out = run / "analysis"; run_out.mkdir(exist_ok=True); agg.OUT = run_out
    report = []; budget_tables = []
    for condition in ("single", "sequential"):
        frame = load(run, condition)
        if frame.empty: continue
        report.append(f"\n##### condition: {condition}, run: {name}\n")
        buffer = []
        class Tee:
            def write(self, s): buffer.append(s); sys.__stdout__.write(s)
            def flush(self): sys.__stdout__.flush()
        with contextlib.redirect_stdout(Tee()):
            check_counts(frame, f"{name}/{condition}")
            main_frame = frame[frame.family != ju.EXTRA_BASELINE]
            for metric, (_, lower) in agg.METRICS.items(): agg.analyse(main_frame, metric, lower, condition)
            print("\n--- extra baseline: random pair then random type, against Random (uniform judgment) ---")
            extra = frame[frame.family.isin(["random", ju.EXTRA_BASELINE])]
            for metric, (_, lower) in agg.METRICS.items(): agg.analyse(extra, metric, lower, f"{condition}_extra_baseline")
            table = per_budget(frame, run_out, condition); budget_tables.append(table); describe_best(table); touched(frame, run_out, condition)
            initial = [json.loads(p.read_text()) for p in sorted((run / "single").glob("seed*_initial_only.json"))]
            print(f"\ninitial labelled set only ({len(initial)} seeds): " + ", ".join(f"{m}={np.nanmean([d[m] for d in initial]):.3f}" for m in ("test_decisive_log_loss", "test_decisive_auc", "test_decisive_accuracy")),
                  f"; mean initial judgments {np.mean([d['initial_judgments'] for d in initial]):.1f}; mean pool judgments {np.mean([d['pool_judgments'] for d in initial]):.1f}")
        report.append("".join(buffer))
    (run_out / "report.txt").write_text("\n".join(report))


if __name__ == "__main__":
    main()
