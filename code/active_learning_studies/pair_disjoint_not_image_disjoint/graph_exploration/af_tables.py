"""Tables of the anchor-far study, generated from the per-judgment prediction files (rt_a1.py CSVs) and the anchor-distance files (rt_a1b.py).

  python3 af_tables.py dev A1DIR A1BDIR [--ref=af_base] [--lo=0.01 --hi=9]     gain of every learner over the reference on the standard endpoint and on the anchor-far stratum
  python3 af_tables.py dose A1DIR A1BDIR --cand=X --ref=Y                       AUC / accuracy gain by distance to the nearest typed anchor
Strata: distance = anc_all_min (1 - cos of the closer image of the judgment to the nearest typed anchor: references, utility and outer-test ideal images); far = both images > 0.01 from every anchor.
"""
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

BINS = [("<=0.001", -1, 0.001), ("0.001-0.01", 0.001, 0.01), ("0.01-0.03", 0.01, 0.03), ("0.03-0.1", 0.03, 0.1), (">0.1", 0.1, 9)]


def load(a1, a1b):
    df = pd.concat([pd.read_csv(f) for d in a1.split(",") for f in sorted(glob.glob(str(Path(d) / "seed*.csv.gz")))], ignore_index=True)       # a1 may be a comma-separated list of directories (learners of different rounds)
    df = df.drop_duplicates(["seed", "split", "budget", "draw", "learner", "j"])
    b = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(Path(a1b) / "seed*.csv")))], ignore_index=True)[["seed", "split", "j", "anc_all_min"]]
    return df.merge(b, on=["seed", "split", "j"], how="left")


def metric_of(g, metric):
    if metric == "auc": return roc_auc_score(g.sign > 0, g.d) if len(set(g.sign)) == 2 else np.nan
    return float((g.weight * ((g.d * g.sign) > 0)).sum() / g.weight.sum())


def per_seed(df, split, metric, cand, ref, lo=-1.0, hi=9.0):
    g = df[(df.split == split) & (df.anc_all_min > lo) & (df.anc_all_min <= hi)]; out = {}
    for (seed, budget, draw), gg in g.groupby(["seed", "budget", "draw"]):
        a, r = gg[gg.learner == cand].sort_values("j"), gg[gg.learner == ref].sort_values("j")
        if len(a) and len(a) == len(r): out.setdefault(seed, []).append(metric_of(a, metric) - metric_of(r, metric))
    return pd.Series({k: np.nanmean(v) for k, v in out.items()}).dropna()


def summ(per):
    if len(per) < 3: return dict(gain=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n=len(per))
    rng = np.random.default_rng(0); bs = rng.choice(per.to_numpy(), (2000, len(per))).mean(1)
    try: p = float(wilcoxon(per).pvalue) if np.any(per.to_numpy() != 0) else 1.0
    except ValueError: p = 1.0
    return dict(gain=per.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), p=p, n=len(per))


def to_md(df, fmt="{:+.4f}"):
    cols = list(df.columns); o = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows(): o.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else (fmt.format(v) if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(o)


def dev_table(df, ref="af_base", far=(0.01, 9.0)):
    rows = []
    for l in [x for x in sorted(df.learner.unique()) if x != ref]:
        for split in "AB":
            r = {"learner": l, "split": split}
            for tag, (lo, hi) in (("all", (-1, 9)), ("far", far)):
                for m in ("auc", "acc"):
                    s = summ(per_seed(df, split, m, l, ref, lo, hi)); r[f"{tag}_{m}"] = s["gain"]; r[f"{tag}_{m}_p"] = s["p"]
            rows.append(r)
    return pd.DataFrame(rows)


def dose_table(df, cand, ref):
    rows = []
    for label, lo, hi in BINS + [("all", -1, 9)]:
        for split in "AB":
            r = {"stratum": label, "split": split}
            for m in ("auc", "acc"):
                s = summ(per_seed(df, split, m, cand, ref, lo, hi)); r[m] = s["gain"]; r[m + " CI"] = f"[{s['lo']:+.4f}, {s['hi']:+.4f}]"
            r["n_seeds"] = s["n"]; r["share of judgments"] = float(((df.anc_all_min > lo) & (df.anc_all_min <= hi) & (df.learner == cand) & (df.split == split)).sum() / ((df.learner == cand) & (df.split == split)).sum()); rows.append(r)
    return pd.DataFrame(rows)


def absolute_table(df, learners, bins=None):
    """Absolute AUC / accuracy of each learner by distance to the nearest typed anchor (per-seed mean over budgets and draws, then mean over seeds) and the share of judgments."""
    rows = []
    for label, lo, hi in (bins or BINS) + [("all", -1, 9)]:
        for split in "AB":
            r = {"stratum": label, "split": split}; sub = df[(df.split == split) & (df.anc_all_min > lo) & (df.anc_all_min <= hi)]
            r["share"] = len(sub[sub.learner == learners[0]]) / max(len(df[(df.split == split) & (df.learner == learners[0])]), 1)
            r["judgments/test set"] = r["share"] * df[(df.split == split) & (df.learner == learners[0])].groupby(["seed", "budget", "draw"]).size().mean()
            for l in learners:
                for m in ("auc", "acc"):
                    v = [metric_of(g, m) for _, g in sub[sub.learner == l].groupby(["seed", "budget", "draw"]) if len(g) > 1]
                    r[f"{l} {m}"] = float(np.nanmean(v)) if v else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    mode = sys.argv[1]; opts = dict(a[2:].split("=", 1) for a in sys.argv[2:] if a.startswith("--") and "=" in a); dirs = [a for a in sys.argv[2:] if not a.startswith("--")]; df = load(dirs[0], dirs[1])
    if mode == "dev": print(to_md(dev_table(df, opts.get("ref", "af_base"), (float(opts.get("lo", 0.01)), float(opts.get("hi", 9))))))
    if mode == "abs": print(to_md(absolute_table(df, opts["learners"].split(","))))
    if mode == "dose": print(to_md(dose_table(df, opts["cand"], opts.get("ref", "af_base"))))
