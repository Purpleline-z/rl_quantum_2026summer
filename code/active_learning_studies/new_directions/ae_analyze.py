"""Tables for the anchor-exchange study (generated; no hand-copied numbers).
Gains: AUC, accuracy, and -log-loss ('utility' metrics, higher = better).  Tests: two-sided Wilcoxon over seeds (seed = mean over budgets and subsets), Holm within the family passed in.
Exchange rate: per bootstrap resample of seeds, the metric surface M(n_anchors, n_pairwise) is averaged over seeds; the budget axis is modelled as M(b) = alpha + beta log(30 + b) per anchor level
(30 = judgments of the 10 initial groups, so the axis is the total number of pairwise judgments); b* solves M(b*; a0) = M(b0; a1); extra judgments equivalent to the extra anchors = b* - b0."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
R = HERE / "results" / "ae"
UTIL = {"auc": 1.0, "acc": 1.0, "ll": -1.0, "htr_acc": 1.0, "htr_ll": -1.0, "ext_acc": 1.0, "t1x1_acc": 1.0, "tc6x2_acc": 1.0, "t13_acc": 1.0, "htr_auc": 1.0, "ext_acc_3": 1.0}
def sgn(m): return UTIL.get(m, -1.0 if m.endswith("ll") else 1.0)


N0 = 30.0   # judgments of the 10 initial groups (approximate; only shifts the log axis)


def load(dirpath) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(dirpath).glob("seed*_*.json")): rows += json.loads(f.read_text())
    return pd.DataFrame(rows)


def wil(x) -> float:
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return float("nan") if len(x) < 6 or np.allclose(x, 0) else float(wilcoxon(x).pvalue)


def holm(p: dict) -> dict:
    items = sorted((k for k, v in p.items() if not np.isnan(v)), key=lambda k: p[k]); out = {k: float("nan") for k in p}; run = 0.0
    for i, k in enumerate(items): run = max(run, min(1.0, (len(items) - i) * p[k])); out[k] = run
    return out


def md(t: pd.DataFrame, digits=3) -> str:
    if t is None or len(t) == 0: return "(no data)\n"
    t = t.copy()
    for c in t.columns:
        if t[c].dtype.kind == "f": t[c] = t[c].map(lambda v: "" if np.isnan(v) else (f"{v:.2e}" if c in ("p", "holm", "p_uncorrected") and 0 < abs(v) < 10 ** -digits else (f"{v:+.{digits}f}" if c in ("gain", "diff") else f"{v:.{digits}f}")))
    return "| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns) + "\n" + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False)) + "\n"


# ------------------------------------------------------------------------------------------ active anchor rules
def rand_reference(frame: pd.DataFrame) -> pd.DataFrame:
    """Random-anchor reference = mean of the three random subsets per (split, seed, budget)."""
    r = frame[frame.arm.str.startswith("rand")]; cols = [c for c in r.columns if r[c].dtype.kind == "f" and c not in ("lam",)]
    return r.groupby(["split", "seed", "budget"], as_index=False)[cols].mean()


def seed_gains(frame: pd.DataFrame, arm: str, ref: str | pd.DataFrame, metrics) -> pd.DataFrame:
    """Per (split, seed) gain of ``arm`` over ``ref`` (an arm name or the rand_reference frame), mean over budgets."""
    a = frame[frame.arm == arm].set_index(["split", "seed", "budget"]); b = (ref.set_index(["split", "seed", "budget"]) if isinstance(ref, pd.DataFrame) else frame[frame.arm == ref].set_index(["split", "seed", "budget"])); common = a.index.intersection(b.index)
    d = pd.DataFrame({m: sgn(m) * (a.loc[common, m] - b.loc[common, m]) for m in metrics}); return d.groupby(level=[0, 1]).mean().reset_index()


def active_table(frame: pd.DataFrame, rules, metrics=("auc", "acc"), holm_metrics=("auc", "acc")) -> pd.DataFrame:
    ref = rand_reference(frame); rows = []
    for r in rules:
        g = seed_gains(frame, r, ref, metrics + tuple(m for m in ("ll", "htr_acc", "htr_ll", "ext_acc") if m not in metrics))
        for split, gg in g.groupby("split"):
            for m in g.columns[2:]: rows.append({"rule": r, "split": split, "metric": m, "n_seeds": len(gg), "gain": float(gg[m].mean()), "share_better": float((gg[m] > 0).mean()), "p": wil(gg[m])})
    t = pd.DataFrame(rows); fam = t[t.metric.isin(holm_metrics)]; hp = holm({i: p for i, p in zip(fam.index, fam.p)}); t["holm"] = [hp.get(i, float("nan")) for i in t.index]; return t


def control_table(frame: pd.DataFrame, pairs, metrics=("auc", "acc")) -> pd.DataFrame:
    rows = []
    for a, b in pairs:
        g = seed_gains(frame, a, b, metrics)
        for split, gg in g.groupby("split"):
            for m in metrics: rows.append({"comparison": f"{a} - {b}", "split": split, "metric": m, "n_seeds": len(gg), "diff": float(gg[m].mean()), "p_uncorrected": wil(gg[m])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ dose-response and exchange rate
def surfaces(frame: pd.DataFrame, metric: str, split: str | None = None):
    """array M[seed, n_idx, b_idx] (mean over subsets), plus the n and budget axes.  arm 'all144' is excluded."""
    f = frame[(frame.arm != "all144")] if split is None else frame[(frame.arm != "all144") & (frame.split == split)]
    g = f.groupby(["seed", "split", "n_anchors", "budget"], as_index=False)[metric].mean(); g = g.groupby(["seed", "n_anchors", "budget"], as_index=False)[metric].mean()
    ns = sorted(g.n_anchors.unique()); bs = sorted(g.budget.unique()); seeds = sorted(g.seed.unique()); M = np.full((len(seeds), len(ns), len(bs)), np.nan)
    for r in g.itertuples(): M[seeds.index(r.seed), ns.index(r.n_anchors), bs.index(r.budget)] = getattr(r, metric)
    return M * sgn(metric), ns, bs


def dose_table(frame: pd.DataFrame, metric: str, split: str | None = None) -> pd.DataFrame:
    M, ns, bs = surfaces(frame, metric, split); base = M[:, 0, :]; rows = []
    for j, n in enumerate(ns):
        row = {"anchors": n}
        for k, b in enumerate(bs): row[f"b={b}"] = float(np.nanmean(M[:, j, k])); row[f"gain b={b}"] = float(np.nanmean(M[:, j, k] - base[:, k])) if j else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _fit_log(bs, y):
    x = np.log(N0 + np.asarray(bs, float)); beta, alpha = np.polyfit(x, y, 1); return alpha, beta


def exchange_once(Mbar: np.ndarray, ns, bs, a0: int, a1: int, b0: int) -> dict:
    i0, i1, kb = ns.index(a0), ns.index(a1), bs.index(b0); gain = Mbar[i1, kb] - Mbar[i0, kb]; alpha, beta = _fit_log(bs, Mbar[i0, :])
    if beta <= 1e-9: return {"gain": float(gain), "extra_judgments": float("nan"), "beyond": False}
    bstar = float(np.exp((Mbar[i0, kb] + gain - alpha) / beta) - N0); return {"gain": float(gain), "extra_judgments": bstar - b0, "beyond": bstar > max(bs)}


def exchange_table(frame: pd.DataFrame, metric: str, transitions, b0s, split: str | None = None, reps: int = 1000, seed: int = 0) -> pd.DataFrame:
    M, ns, bs = surfaces(frame, metric, split); ok = ~np.isnan(M).any(axis=(1, 2)); M = M[ok]; rng = np.random.default_rng(seed); rows = []
    for (a0, a1) in transitions:
        for b0 in b0s:
            est = exchange_once(np.nanmean(M, axis=0), ns, bs, a0, a1, b0); boots = []
            for _ in range(reps): pick = rng.integers(0, len(M), len(M)); boots.append(exchange_once(M[pick].mean(axis=0), ns, bs, a0, a1, b0)["extra_judgments"])
            boots = np.array(boots, float); lo, hi = np.nanpercentile(boots, [2.5, 97.5]) if np.isfinite(boots).any() else (np.nan, np.nan); share_pos = float(np.nanmean(boots > 0))
            per_label = (est["extra_judgments"] / (a1 - a0)) if est["extra_judgments"] > 0 else float("nan")
            rows.append({"anchors": f"{a0}->{a1}", "from_budget": b0, "metric_gain": est["gain"], "extra_judgments_equiv": est["extra_judgments"], "ci_lo": float(lo), "ci_hi": (float(hi) if hi < 1000 else float("inf")), "judgments_per_typed_label": per_label, "beyond_range_60": est["beyond"], "share_resamples_positive": share_pos})
    return pd.DataFrame(rows)
