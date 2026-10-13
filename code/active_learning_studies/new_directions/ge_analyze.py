"""Tables for the graph-endpoint study (all generated; no hand-copied numbers).

Gain conventions: accuracy and AUC = arm - reference (positive = arm better); tests = two-sided Wilcoxon signed-rank over seeds (zero differences dropped, as scipy's default); Holm within the family
that the caller passes; trend test of the dose-response = Wilcoxon over seeds of the per-seed least-squares slope of the gain on the unlabelled fraction."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
R = HERE / "results" / "ge"


def wil(x) -> float:
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return float("nan") if len(x) < 6 or np.allclose(x, 0) else float(wilcoxon(x).pvalue)


def holm(p: dict) -> dict:
    items = sorted((k for k, v in p.items() if not np.isnan(v)), key=lambda k: p[k]); out = {k: float("nan") for k in p}; run = 0.0
    for i, k in enumerate(items): run = max(run, min(1.0, (len(items) - i) * p[k])); out[k] = run
    return out


def load(dirpath, pattern="seed*.json") -> pd.DataFrame:
    rows = []
    for f in sorted(Path(dirpath).glob(pattern)): rows += json.loads(f.read_text())
    return pd.DataFrame(rows)


def md(t: pd.DataFrame, digits=3) -> str:
    if t is None or t.empty: return "(no data)\n"
    t = t.copy()
    for c in t.columns:
        if t[c].dtype.kind == "f": t[c] = t[c].map(lambda v: "" if np.isnan(v) else (f"{v:.2e}" if c in ("p", "holm", "p_uncorrected") and 0 < abs(v) < 10 ** -digits else (f"{v:+.{digits}f}" if c in ("gain", "diff", "slope") else f"{v:.{digits}f}")))
    return "| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns) + "\n" + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False)) + "\n"


# ------------------------------------------------------------------------------------------ classification endpoint
def class_table(frame: pd.DataFrame, ref=("nn", "none", -1), holm_family=True) -> pd.DataFrame:
    """Per (method, protocol, budget): mean accuracy over seeds, mean paired difference to the reference (NN by default), share of seeds better/worse, Wilcoxon p, Holm."""
    per = frame.groupby(["seed", "method", "protocol", "budget"], as_index=False).acc.mean(); base = per[(per.method == ref[0]) & (per.protocol == ref[1]) & (per.budget == ref[2])].set_index("seed").acc; rows = []
    for (m, pr, b), g in per.groupby(["method", "protocol", "budget"]):
        a = g.set_index("seed").acc; d = (a - base.reindex(a.index)).dropna(); rows.append({"method": m, "protocol": pr, "budget": b, "n_seeds": len(a), "acc": a.mean(), "diff_vs_nn": d.mean(), "better": int((d > 0).sum()), "worse": int((d < 0).sum()), "p": wil(d)})
    t = pd.DataFrame(rows).sort_values("acc", ascending=False)
    if holm_family:
        mask = ~((t.method == ref[0]) & (t.protocol == ref[1]) & (t.budget == ref[2])); hp = holm({i: p for i, p in zip(t.index[mask], t.p[mask])}); t["holm"] = [hp.get(i, float("nan")) for i in t.index]
    return t.reset_index(drop=True)


def class_vs(frame: pd.DataFrame, a, b) -> dict:
    """Paired accuracy difference a - b (each (method, protocol, budget)) over seeds."""
    per = frame.groupby(["seed", "method", "protocol", "budget"], as_index=False).acc.mean()
    get = lambda s: per[(per.method == s[0]) & (per.protocol == s[1]) & (per.budget == s[2])].set_index("seed").acc
    d = (get(a) - get(b)).dropna(); return {"diff": float(d.mean()), "p": wil(d), "n": len(d)}


# ------------------------------------------------------------------------------------------ dose-response
def dose_gains(frame: pd.DataFrame) -> pd.DataFrame:
    """Per (split, seed, variant): gain over the aw8_all arm ('base') of auc and acc, mean over budgets and draws."""
    per = frame.groupby(["split", "seed", "variant"], as_index=False)[["auc", "acc"]].mean(); out = []
    for (split, seed), g in per.groupby(["split", "seed"]):
        b = g[g.variant == "base"].iloc[0]
        for r in g.itertuples():
            if r.variant != "base": out.append({"split": split, "seed": seed, "variant": r.variant, "auc": r.auc - b.auc, "acc": r.acc - b.acc})
    return pd.DataFrame(out)


def dose_table(gains: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, v), g in gains.groupby(["split", "variant"]):
        rows.append({"split": split, "variant": v, "n_seeds": len(g), "gain_auc": g.auc.mean(), "p_auc": wil(g.auc), "gain_acc": g.acc.mean(), "p_acc": wil(g.acc)})
    return pd.DataFrame(rows)


def dose_trend(gains: pd.DataFrame, series: str, metric: str) -> pd.DataFrame:
    """Per split: slope of gain on fraction p (series 'all' uses all_p0..all_p100; 'out' uses out_p0, out_p25, out_p50 and all_p100); Wilcoxon of the per-seed slopes; pooled Spearman (descriptive)."""
    names = {"all": [f"all_p{p}" for p in (0, 25, 50, 100)], "out": ["out_p0", "out_p25", "out_p50", "all_p100"]}[series]; fr = np.array([0, .25, .5, 1.0]); rows = []
    for split, g in gains.groupby("split"):
        piv = g.pivot(index="seed", columns="variant", values=metric)
        if not set(names) <= set(piv.columns): continue
        y = piv[names].dropna(); slopes = np.array([np.polyfit(fr, row.to_numpy(), 1)[0] for _, row in y.iterrows()]); rho = spearmanr(np.tile(fr, len(y)), y.to_numpy().ravel()).statistic
        rows.append({"series": series, "split": split, "metric": metric, "n_seeds": len(y), "slope": float(slopes.mean()), "share_pos": float((slopes > 0).mean()), "p": wil(slopes), "pooled_spearman": float(rho)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ harsh splits and lambda
def arm_gains(frame: pd.DataFrame, group_cols, arm_col: str, ref: str, arms=None, metrics=("auc", "acc")) -> pd.DataFrame:
    per = frame.groupby(group_cols + ["seed", arm_col], as_index=False)[list(metrics)].mean(); rows = []
    for key, g in per.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,); piv = {m: g.pivot(index="seed", columns=arm_col, values=m) for m in metrics}
        for arm in (arms or [a for a in g[arm_col].unique() if a != ref]):
            for m in metrics:
                d = (piv[m][arm] - piv[m][ref]).dropna(); rows.append({**dict(zip(group_cols, key)), "arm": arm, "metric": m, "n_seeds": len(d), "gain": float(d.mean()), "share_better": float((d > 0).mean()), "p": wil(d)})
    return pd.DataFrame(rows)


def arm_vs(frame: pd.DataFrame, group_cols, arm_col: str, a: str, b: str, metrics=("auc", "acc")) -> pd.DataFrame:
    per = frame.groupby(group_cols + ["seed", arm_col], as_index=False)[list(metrics)].mean(); rows = []
    for key, g in per.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        for m in metrics:
            piv = g.pivot(index="seed", columns=arm_col, values=m); d = (piv[a] - piv[b]).dropna(); rows.append({**dict(zip(group_cols, key)), "arm": f"{a} - {b}", "metric": m, "n_seeds": len(d), "diff": float(d.mean()), "p_uncorrected": wil(d)})
    return pd.DataFrame(rows)
