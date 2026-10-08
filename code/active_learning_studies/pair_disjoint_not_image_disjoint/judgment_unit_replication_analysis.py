#!/usr/bin/env python3
"""Analysis of the pre-registered replication (JUDGMENT_UNIT_REPLICATION_PREREGISTRATION.md).

usage: judgment_unit_replication_analysis.py            (writes results/judgment_unit_study/replication/analysis/*.csv and report.txt)

Confirmatory: seeds 500-534 only.  Exploratory: the 35 main seeds pooled with the 35 new ones; full variant tables; ranking consistency.
Gain over Random = (Random - strategy) for log-loss, (strategy - Random) for AUC / accuracy, Random = mean of its five draws within a seed, per seed averaged
over the budgets 10/20/40/60.  One-sided Wilcoxon signed-rank (scipy default zero handling); Holm over the pre-registered families only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

import aggregate_pair_endpoint as agg
import judgment_unit_aggregate as ja
import judgment_unit_study as study

ROOT = study.OUT; REP = ROOT / "replication"; OUT = REP / "analysis"
NEW_SEEDS = tuple(range(500, 535)); BUDGETS = study.BUDGETS
LL, AUC, ACC = "test_decisive_log_loss", "test_decisive_auc", "test_decisive_accuracy"
LOWER = {LL: True, AUC: False, ACC: False}
CELLS = [("A", "single"), ("A", "sequential"), ("B", "single"), ("B", "sequential")]
VARIANTS = ["cluster_quota_uncertainty_lf", "cluster_quota_uncertainty", "ensemble_bald_decisive", "ensemble_bald", "fisher_dopt", "bald_decisive", "laplace_bald",
            "core_set", "core_set_relation", "dpp_pairs", "uncertainty", "uncertainty_all_heads", "delta_gap"]
EXTRA = "random_pair_type"
# (hypothesis, key, split, condition, metric, direction)   direction "greater": gain > 0 ; "less": gain < 0
H = [("H1", "cluster_quota_uncertainty_lf", "B", "sequential", LL, "greater"), ("H2", "ensemble_bald_decisive", "B", "sequential", AUC, "greater")]
H += [("H3", k, s, c, ACC, "greater") for k in ("ensemble_bald_decisive", "ensemble_bald", "fisher_dopt") for s, c in CELLS]
H += [("H4", "core_set", "B", c, LL, "greater") for c in ("single", "sequential")]
H += [("H5", "dpp_pairs", "B", "single", LL, "less"), ("H5", "uncertainty_all_heads", "B", "single", AUC, "less")]
H6 = [("H6", "uncertainty", s, c, m, "greater") for s, c in CELLS for m in (LL, AUC)]
NAMES = {"cluster_quota_uncertainty_lf": "Cluster-quota uncertainty, all heads (original code)", "cluster_quota_uncertainty": "Cluster-quota uncertainty (own type)",
         "ensemble_bald_decisive": "Deep-ensemble BALD x P(decisive)", "ensemble_bald": "Deep-ensemble BALD", "fisher_dopt": "Fisher D-optimal", "bald_decisive": "BALD x P(decisive)",
         "laplace_bald": "Laplace BALD", "core_set": "Core-set", "core_set_relation": "Core-set, relation-aware pairs", "dpp_pairs": "DPP", "uncertainty": "Uncertainty (own type)",
         "uncertainty_all_heads": "Uncertainty, all heads", "delta_gap": "Largest predicted gap", EXTRA: "Random pair, random type"}
METRIC_LABEL = {LL: "log-loss", AUC: "AUC", ACC: "accuracy"}


def load_cell(run: Path, condition: str, seeds) -> pd.DataFrame:
    frame = ja.load(run, condition); return frame[frame.seed.isin(seeds)]


def per_seed_gain(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """seed x family table of the gain over Random averaged over the four budgets."""
    ps = frame.groupby(["seed", "budget", "family"], as_index=False)[metric].mean()
    rnd = ps[ps.family == "random"][["seed", "budget", metric]].rename(columns={metric: "random"}); p = ps.merge(rnd, on=["seed", "budget"])
    p["gain"] = (-1.0 if LOWER[metric] else 1.0) * (p[metric] - p.random)
    return p[p.family != "random"].groupby(["seed", "family"]).gain.mean().unstack()


def one_test(gains: np.ndarray, direction: str) -> float:
    return float(wilcoxon(gains, alternative=direction).pvalue) if np.any(gains != 0) else 1.0


def run_tests(data: dict, specs) -> pd.DataFrame:
    """data[(split, condition)] = frame.  Returns one row per spec with estimate, p, Holm p (over ``specs``), share of seeds better."""
    rows = []
    for h, key, split, cond, metric, direction in specs:
        g = per_seed_gain(data[(split, cond)], metric)[key].dropna().to_numpy()
        rows.append({"hypothesis": h, "variant": key, "split": split, "condition": cond, "metric": METRIC_LABEL[metric], "direction": direction, "n_seeds": len(g), "mean_gain": g.mean(),
                     "median_gain": float(np.median(g)), "sd": g.std(ddof=1), "share_seeds_gain_gt0": float((g > 0).mean()), "share_seeds_gain_lt0": float((g < 0).mean()), "p_one_sided": one_test(g, direction)})
    out = pd.DataFrame(rows); out["holm_p"] = [agg.holm(dict(enumerate(out.p_one_sided)))[i] for i in range(len(out))]
    right = np.where(out.direction == "greater", out.mean_gain > 0, out.mean_gain < 0)
    out["right_sign"] = right; out["test_replicated"] = right & (out.holm_p < .05)
    return out


def verdicts(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (h, key), g in t.groupby(["hypothesis", "variant"], sort=False):
        rows.append({"hypothesis": h, "claim": key, "tests": len(g), "replicated_tests": int(g.test_replicated.sum()), "right_sign_tests": int(g.right_sign.sum()),
                     "nominal_p<0.05": int((g.p_one_sided < .05).sum()), "verdict": "replicated" if g.test_replicated.all() else "not replicated"})
    v = pd.DataFrame(rows)
    for h, g in t.groupby("hypothesis", sort=False):  # whole hypothesis: all of its tests must be replicated
        rows.append({"hypothesis": h, "claim": "ALL (whole hypothesis)", "tests": len(g), "replicated_tests": int(g.test_replicated.sum()), "right_sign_tests": int(g.right_sign.sum()),
                     "nominal_p<0.05": int((g.p_one_sided < .05).sum()), "verdict": "replicated" if g.test_replicated.all() else "not replicated"})
    return pd.DataFrame(rows)


def two_sided_table(data: dict, split: str, cond: str, metric: str) -> pd.DataFrame:
    g = per_seed_gain(data[(split, cond)], metric); rows = {}
    for k in VARIANTS + [EXTRA]:
        v = g[k].dropna().to_numpy(); rows[k] = {"mean_gain": v.mean(), "share_better": float((v > 0).mean()), "p_two_sided": float(wilcoxon(v).pvalue) if np.any(v != 0) else 1.0}
    out = pd.DataFrame(rows).T; main = {k: out.loc[k, "p_two_sided"] for k in VARIANTS}; adj = agg.holm(main); out["holm_p_over_13"] = pd.Series(adj)
    return out.loc[VARIANTS + [EXTRA]]


def md(df: pd.DataFrame, floatfmt: dict | None = None) -> str:
    cols = list(df.columns); lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("<0.001" if c in ("p_one_sided", "holm_p", "p", "p_two_sided", "holm_p_over_13") and v < .001 else (f"{v:+.3f}" if "gain" in c else f"{v:.3f}" if "p" in c.lower() or "share" in c else f"{v:.3f}"))
            else: cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def counts(data: dict, tag: str) -> list[str]:
    out = []
    for (split, cond), f in data.items():
        expected = len(study.BUDGETS) * f.seed.nunique() * 23
        per_seed = f.groupby("seed").strategy.nunique()
        out.append(f"[{tag}] split {split} {cond}: {len(f)} cells = {f.seed.nunique()} seeds x {f.strategy.nunique()} strategy names x {f.budget.nunique()} budgets (expected {expected}); seeds with all 23 names: {int((per_seed == 23).sum())}/{f.seed.nunique()}; n_revealed == budget everywhere: {bool((f.n_revealed == f.budget).all())}")
        assert len(f) == expected and (f.n_revealed == f.budget).all() and (per_seed == 23).all(), "cell count / design mismatch"
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); log: list[str] = []
    new = {(s, c): load_cell(REP / f"{s}_groups", c, NEW_SEEDS) for s, c in CELLS}
    main_runs = {(s, c): load_cell(ROOT / f"{s}_groups", c, ja.SEEDS) for s, c in CELLS}
    log += counts(new, "replication seeds 500-534")
    # main-run cells restricted to the replicated variants (the main run holds 42 names per seed)
    keep = lambda f: f[f.family.isin(["random", EXTRA] + VARIANTS)]
    main_runs = {k: keep(f) for k, f in main_runs.items()}
    for k, f in main_runs.items(): log.append(f"[main seeds, restricted to the 15 replicated families] split {k[0]} {k[1]}: {f.seed.nunique()} seeds, {len(f)} cells")
    pooled = {k: pd.concat([main_runs[k], new[k]], ignore_index=True) for k in new}

    # ---- confirmatory
    conf = run_tests(new, H); conf.to_csv(OUT / "confirmatory_tests.csv", index=False)
    conf_v = verdicts(conf); conf_v.to_csv(OUT / "confirmatory_verdicts.csv", index=False)
    null = run_tests(new, H6); null.to_csv(OUT / "h6_null_check_tests.csv", index=False)
    # ---- main-run values for the same tests, for comparison (the hypotheses were formed on them: not a test)
    mainv = run_tests(main_runs, H); mainv.to_csv(OUT / "main_seeds_same_tests.csv", index=False)
    # ---- exploratory pooled
    pool = run_tests(pooled, H); pool.to_csv(OUT / "exploratory_pooled_tests.csv", index=False); pool_v = verdicts(pool); pool_v.to_csv(OUT / "exploratory_pooled_verdicts.csv", index=False)
    pool_null = run_tests(pooled, H6); pool_null.to_csv(OUT / "exploratory_pooled_h6_tests.csv", index=False)
    cols = ["hypothesis", "variant", "split", "condition", "metric", "direction", "mean_gain", "share_seeds_gain_gt0", "share_seeds_gain_lt0", "p_one_sided", "holm_p", "right_sign", "test_replicated"]
    for title, t in (("CONFIRMATORY (seeds 500-534)", conf), ("MAIN SEEDS, same tests (hypotheses were formed here)", mainv), ("EXPLORATORY POOLED (70 seeds)", pool)):
        log += ["", f"=== {title}: 18 one-sided tests, Holm over the 18 ===", t[cols].to_string(index=False)]
    for title, v in (("CONFIRMATORY verdicts", conf_v), ("EXPLORATORY pooled verdicts", pool_v)): log += ["", f"=== {title} ===", v.to_string(index=False)]
    for title, t in (("H6 null check (seeds 500-534), alternative gain > 0, Holm over 8", null), ("H6 pooled 70 seeds (exploratory)", pool_null)):
        log += ["", f"=== {title} ===", t[cols].to_string(index=False), f"no test rejected: {bool((t.holm_p >= .05).all())}"]

    # ---- exploratory full tables
    sections = []
    for (s, c) in CELLS:
        for metric in (LL, AUC, ACC):
            tn = two_sided_table(new, s, c, metric); tm = two_sided_table(main_runs, s, c, metric)
            joined = tn.join(tm, lsuffix="_new", rsuffix="_main"); joined.to_csv(OUT / f"full_table_{s}_{c}_{METRIC_LABEL[metric]}.csv")
            rho = spearmanr(tn.loc[VARIANTS, "mean_gain"], tm.loc[VARIANTS, "mean_gain"])
            sections.append({"split": s, "condition": c, "metric": METRIC_LABEL[metric], "spearman_mean_gain_main_vs_new": rho.statistic})
    pd.DataFrame(sections).to_csv(OUT / "ranking_consistency.csv", index=False)
    log += ["", "=== exploratory: Spearman correlation (13 variants) of mean gains, main seeds vs new seeds ===", pd.DataFrame(sections).round(2).to_string(index=False)]
    # ---- random_pair_type vs random
    rows = []
    for (s, c) in CELLS:
        for metric in (LL, AUC, ACC):
            v = per_seed_gain(new[(s, c)], metric)[EXTRA].to_numpy(); rows.append({"split": s, "condition": c, "metric": METRIC_LABEL[metric], "mean_gain": v.mean(), "p_two_sided": float(wilcoxon(v).pvalue)})
    rpt = pd.DataFrame(rows); rpt.to_csv(OUT / "random_pair_type_vs_random.csv", index=False)
    log += ["", "=== random_pair_type vs Random (two-sided) ===", rpt.round(3).to_string(index=False)]
    # ---- Random levels
    lv = []
    for (s, c) in CELLS:
        for label, f in (("new", new[(s, c)]), ("main", main_runs[(s, c)])):
            r = f[f.family == "random"].groupby(["seed", "budget"])[[LL, AUC, ACC]].mean().groupby("budget").mean()
            for b, row in r.iterrows(): lv.append({"split": s, "condition": c, "seeds": label, "budget": b, "random_log_loss": row[LL], "random_auc": row[AUC], "random_accuracy": row[ACC]})
    pd.DataFrame(lv).to_csv(OUT / "random_levels.csv", index=False)
    (OUT / "report.txt").write_text("\n".join(log)); print("\n".join(log))


if __name__ == "__main__":
    main()
