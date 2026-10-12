"""Analysis of the graph-selection screens/confirmations (gsel_select.py output).  Every table is generated here from the result JSONs.

Gain of a selector over Random = per-seed mean over budgets of (selector - mean of the random draws) (log-loss: random - selector, so positive = better).  Tests: seed-level Wilcoxon
signed-rank (two-sided); Holm over the family given by ``--holm`` (list of selector names x split x metric); paired differences to a control selector are reported uncorrected."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = ("acc", "auc", "ll", "htr_acc", "htr_auc", "htr_ll", "other_acc")
OTHER = ("t1x1_acc", "tc6x2_acc", "t13_acc")


def load(dirs, learner="aw8_all") -> pd.DataFrame:
    rows = []
    for d in dirs:
        for f in sorted(Path(d).glob("seed*_*.json")): rows += json.loads(f.read_text())
    frame = pd.DataFrame(rows)
    if frame.empty: return frame
    frame = frame[frame.learner == learner].copy(); frame["other_acc"] = frame[list(OTHER)].mean(axis=1, skipna=True); return frame


def gains(frame: pd.DataFrame, selector: str, base: str = "random", budgets=None) -> pd.DataFrame:
    """Per (split, seed) gain of ``selector`` over ``base`` for every metric (mean over budgets)."""
    out = []
    for split, g in frame.groupby("split"):
        for seed, gs in g.groupby("seed"):
            if budgets is not None: gs = gs[gs.budget.isin(budgets)]
            a = gs[gs.selector == selector].groupby("budget").mean(numeric_only=True); b = gs[gs.selector == base].groupby("budget").mean(numeric_only=True)
            common = a.index.intersection(b.index)
            if not len(common): continue
            row = {"split": split, "seed": seed, "n_budgets": len(common)}
            for m in METRICS:
                if m not in a: continue
                sign = -1.0 if m.endswith("ll") else 1.0; row[m] = float(np.nanmean(sign * (a.loc[common, m] - b.loc[common, m])))
            out.append(row)
    return pd.DataFrame(out)


def wilcoxon_p(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 6 or np.allclose(x, 0): return float("nan")
    return float(wilcoxon(x).pvalue)


def holm(p: dict) -> dict:
    items = sorted((k for k, v in p.items() if not np.isnan(v)), key=lambda k: p[k]); out = {k: float("nan") for k in p}; running = 0.0
    for i, k in enumerate(items): running = max(running, min(1.0, (len(items) - i) * p[k])); out[k] = running
    return out


def table(frame, selectors, base="random", metrics=("acc", "auc", "ll"), holm_family=None, budgets=None) -> pd.DataFrame:
    """Rows: (selector, split, metric) with mean gain, share of seeds better, Wilcoxon p and Holm p over ``holm_family`` (default all rows of the table)."""
    rows = []
    for s in selectors:
        g = gains(frame, s, base, budgets)
        if g.empty: continue
        for split, gg in g.groupby("split"):
            for m in metrics:
                if m not in gg: continue
                v = gg[m].to_numpy(); rows.append({"selector": s, "split": split, "metric": m, "n_seeds": len(v), "gain": float(np.nanmean(v)), "share_better": float((v > 0).mean()), "p": wilcoxon_p(v)})
    t = pd.DataFrame(rows)
    if t.empty: return t
    fam = t if holm_family is None else t[t.apply(lambda r: holm_family(r), axis=1)]
    hp = holm({i: r.p for i, r in fam.iterrows()}); t["holm"] = [hp.get(i, float("nan")) for i in t.index]; return t


def vs_control(frame, selector, control, metrics=("acc", "auc", "ll")) -> pd.DataFrame:
    """Selector minus control, paired by seed and budget (uncorrected p)."""
    rows = []
    for split, g in frame.groupby("split"):
        a = g[g.selector == selector].groupby(["seed", "budget"]).mean(numeric_only=True); b = g[g.selector == control].groupby(["seed", "budget"]).mean(numeric_only=True); common = a.index.intersection(b.index)
        if not len(common): continue
        for m in metrics:
            sign = -1.0 if m.endswith("ll") else 1.0; d = (sign * (a.loc[common, m] - b.loc[common, m])).groupby(level=0).mean().to_numpy()
            rows.append({"selector": selector, "control": control, "split": split, "metric": m, "n_seeds": len(d), "diff": float(np.nanmean(d)), "p_uncorrected": wilcoxon_p(d)})
    return pd.DataFrame(rows)


def composition(frame, selectors) -> pd.DataFrame:
    cols = [c for c in frame.columns if c.startswith("pick_") and c not in ("pick_ids", "pick_decisive")]; rows = []
    for s in selectors:
        g = frame[frame.selector == s]
        for b, gb in g.groupby("budget"):
            tot = gb[[c for c in cols if c != "pick_images"]].sum(axis=1).mean(); row = {"selector": s, "budget": b, "n_picks": float(tot), "unique_images": float(gb.pick_images.mean()), "decisive_share": float(gb.pick_decisive.mean() / max(tot, 1e-9)) if "pick_decisive" in gb and gb.pick_decisive.notna().any() else float("nan")}
            for c in cols:
                if c != "pick_images": row["share_" + c[5:]] = float(gb[c].mean() / max(tot, 1e-9))
            rows.append(row)
    return pd.DataFrame(rows)


def md(t: pd.DataFrame, digits=3) -> str:
    if t.empty: return "(no data)"
    t = t.copy()
    for c in t.columns:
        if t[c].dtype.kind == "f": t[c] = t[c].map(lambda v: "" if np.isnan(v) else (f"{v:.2e}" if (abs(v) < 10 ** -digits and v != 0 and c in ("p", "holm", "p_uncorrected")) else f"{v:+.{digits}f}" if c in ("gain", "diff") else f"{v:.{digits}f}"))
    head = "| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns) + "\n"; return head + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dirs", required=True); ap.add_argument("--selectors", default=None); ap.add_argument("--learner", default="aw8_all"); ap.add_argument("--metrics", default="acc,auc,ll")
    ap.add_argument("--controls", default=""); ap.add_argument("--budgets", default=None); ap.add_argument("--out", default=None); a = ap.parse_args()
    frame = load(a.dirs.split(","), a.learner); sels = a.selectors.split(",") if a.selectors else [s for s in sorted(frame.selector.unique()) if s != "random"]
    budgets = [int(b) for b in a.budgets.split(",")] if a.budgets else None
    text = f"# gsel analysis of {a.dirs}\nseeds: {frame.seed.nunique()}, rows: {len(frame)}\n\n## Gain over Random (mean over budgets)\n\n" + md(table(frame, sels, metrics=tuple(a.metrics.split(",")), budgets=budgets)) + "\n"
    for pair in [p for p in a.controls.split(",") if p]:
        s, c = pair.split(":"); text += f"\n## {s} minus {c} (uncorrected)\n\n" + md(vs_control(frame, s, c, tuple(a.metrics.split(",")))) + "\n"
    text += "\n## Type composition of the batches\n\n" + md(composition(frame, ["random"] + sels)) + "\n"
    print(text)
    if a.out: Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
