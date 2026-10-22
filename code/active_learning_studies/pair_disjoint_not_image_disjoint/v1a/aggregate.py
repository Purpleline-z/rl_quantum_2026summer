#!/usr/bin/env python3
"""Aggregate the V1a per-seed CSVs into the tables of REPORT_V1a.md (all numbers are generated here; nothing is typed by hand).

Primary: test ideal-image accuracy (28 images).  Per rule and seed the accuracy is averaged over the four budgets (10/20/40/60); the paired difference to Random (= mean of the 5 Random replicates of the
same seed) is averaged over seeds with a 95 % bootstrap interval over seeds (10 000 resamples, seed 0), a two-sided Wilcoxon signed-rank p and Holm correction over the 7 non-random rules.
Secondary (reported only): macro-F1 and the held-out pair-group preference AUC / accuracy / log-loss.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent; RES = HERE / "results"
REPLICATES = ("random", "random_r1", "random_r2", "random_r3", "random_r4")
RULES = ("uncertainty", "core_set", "typiclust_pairs", "laplace_bald", "bald_decisive", "ensemble_bald", "probcover_pairs")
BUDGETS = (10, 20, 40, 60)


def holm(p) -> np.ndarray:
    p = np.asarray(p, float); order = np.argsort(p); out = np.empty_like(p); run = 0.0
    for rank, i in enumerate(order): run = max(run, (len(p) - rank) * p[i]); out[i] = min(1.0, run)
    return out


def md(df: pd.DataFrame, fmt: str = ".4f") -> str:
    cols = list(df.columns); lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows(): lines.append("| " + " | ".join((f"{v:{fmt}}" if isinstance(v, (float, np.floating)) and not np.isnan(v) else ("" if isinstance(v, float) else str(v))) for v in r) + " |")
    return "\n".join(lines)


def load() -> pd.DataFrame:
    frame = pd.concat([pd.read_csv(f) for f in sorted((RES / "by_seed").glob("seed*.csv"))], ignore_index=True); frame.to_csv(RES / "all_rows.csv", index=False); return frame


def per_seed(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Rows: seed; columns: rule (Random = mean of the replicates) = metric averaged over the four budgets."""
    d = frame[frame.budget.isin([str(b) for b in BUDGETS]) | frame.budget.isin(BUDGETS)].copy(); d["budget"] = d.budget.astype(int)
    d["rule_family"] = np.where(d.rule.isin(REPLICATES), "random", d.rule)
    cell = d.groupby(["seed", "rule_family", "budget"], as_index=False)[metric].mean()          # Random: mean over replicates within (seed, budget)
    return cell.groupby(["seed", "rule_family"])[metric].mean().unstack("rule_family")


def paired(frame: pd.DataFrame, metric: str, sign: float = 1.0) -> pd.DataFrame:
    table = per_seed(frame, metric); rows = []
    for r in RULES:
        diff = (sign * (table[r] - table["random"])).dropna().to_numpy(); rng = np.random.default_rng(0); boots = rng.choice(diff, size=(10000, len(diff))).mean(1)
        try: p = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
        except ValueError: p = 1.0
        rows.append({"rule": r, "n_seeds": len(diff), "mean_diff_vs_random": diff.mean(), "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5)), "share_seeds_better": float((diff > 0).mean()), "wilcoxon_p": p})
    out = pd.DataFrame(rows); out["holm_p"] = holm(out.wilcoxon_p.to_numpy()); out["ci_excludes_0"] = (out.ci_lo > 0) | (out.ci_hi < 0); return out


def by_budget(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    d = frame[~frame.budget.isin(["initial_only", "all_pool"])].copy(); d["budget"] = d.budget.astype(int); d["rule_family"] = np.where(d.rule.isin(REPLICATES), "random", d.rule)
    cell = d.groupby(["seed", "rule_family", "budget"], as_index=False)[metric].mean()
    return cell.groupby(["rule_family", "budget"])[metric].mean().unstack("budget").reindex(["random", *RULES])


def references(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in ("initial_only", "all_pool"):
        g = frame[frame.rule == label]; rows.append({"reference": label, "n_seeds": g.seed.nunique(), "test_ideal_accuracy": g.test_ideal_accuracy.mean(), "test_ideal_macro_f1": g.test_ideal_macro_f1.mean(), "pref_auc": g.pref_auc.mean(), "pref_accuracy": g.pref_accuracy.mean(), "pref_log_loss": g.pref_log_loss.mean()})
    return pd.DataFrame(rows)


def main() -> None:
    frame = load(); frame["budget"] = frame.budget.astype(str); text = [f"Seeds: {frame.seed.nunique()}; rows: {len(frame)}\n"]
    tables = per_seed(frame, "test_ideal_accuracy"); text.append("### Mean test ideal accuracy per rule (average over seeds and the four budgets)\n\n" + md(tables.mean().rename("mean_accuracy").reset_index().rename(columns={"rule_family": "rule"})))
    text.append("\n### Primary: test ideal-image accuracy, paired difference to Random (budgets averaged per seed; Holm over the 7 non-random rules)\n\n" + md(paired(frame, "test_ideal_accuracy")))
    text.append("\n### Accuracy per budget (mean over seeds)\n\n" + md(by_budget(frame, "test_ideal_accuracy").reset_index().rename(columns={"rule_family": "rule"})))
    text.append("\n### Secondary: macro-F1, paired difference to Random\n\n" + md(paired(frame, "test_ideal_macro_f1")))
    for metric, sign, name in (("pref_auc", 1.0, "AUC"), ("pref_accuracy", 1.0, "accuracy"), ("pref_log_loss", -1.0, "log-loss (positive = better)")):
        text.append(f"\n### Secondary: held-out pair-group preference {name}, paired difference to Random\n\n" + md(paired(frame, metric, sign)))
    text.append("\n### No-selection references (initial 10 groups only; all candidate-pool groups)\n\n" + md(references(frame)))
    (RES / "tables.md").write_text("\n".join(text) + "\n"); print("\n".join(text))


if __name__ == "__main__": main()
