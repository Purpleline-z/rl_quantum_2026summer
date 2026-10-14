"""Tables of the cross-run transfer study (CR), generated from the result JSONs (nothing typed by hand).

  python3 cr_tables.py counts DIR                                  fold counts per run and split kind (groups, judgments, test judgments, anchor-far share)
  python3 cr_tables.py curve DIR [DIR..] [--metric=auc]            learner x b: pooled over runs (mean of the three run means) and per run; seeds and split kinds averaged
  python3 cr_tables.py gaps DIR [DIR..] [--ref=cr_tgt] [--b=10,20] gain of every learner over a reference at the same b, pooled and per run, uncorrected Wilcoxon over (seed, kind) cells
  python3 cr_tables.py confirm SPEC.json DIR [DIR..]               the pre-registered tests with a group-level cluster bootstrap (see PREREGISTRATION_CR.md)
Unit of aggregation: per (seed, split kind, run) cell the metric of a learner at b; the *pooled* estimate is the mean over the three runs of the per-run means over (seed, kind) cells.
cr_src has no b (stored at b = 0 and used for every b); pooled / weighted learners at b = 0 equal cr_src."""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

RUNS = ("RR220204A", "RR220206A", "RR220411A")
SHORT = {"RR220204A": "0204A", "RR220206A": "0206A", "RR220411A": "0411A"}


def load(dirs):
    rows, metas = [], []
    for d in dirs:
        for f in sorted(glob.glob(str(Path(d) / "seed*.json"))):
            for r in json.loads(Path(f).read_text()):
                (metas if r.get("meta") else rows).append(r)
    return pd.DataFrame(rows), pd.DataFrame(metas)


B_FREE = ("cr_src", "cr_ind_src", "cr_nolap_src", "cr_anchor0", "cr_align_src")   # learners without target labels: stored once at b = 0


def resolve(learner, b):
    if learner.startswith(B_FREE) or (learner.startswith("cr_lapR") and not learner.endswith("_pool")): return learner, 0
    if learner.startswith("cr_lapR") and b == 0: return learner.replace("_pool", ""), 0
    if b == 0 and learner == "cr_align_pool": return "cr_align_src", 0
    if b == 0 and learner.startswith(("cr_pool", "cr_iw")): return "cr_src", 0
    return learner, b


def cells(df, learner, b, metric):
    l, bb = resolve(learner, b); g = df[(df.learner == l) & (df.budget == bb)]
    return g.set_index(["seed", "split", "run"])[metric]


def sign(metric): return -1.0 if metric.startswith("ll") else 1.0


def pooled(series_by_run: dict) -> float:
    return float(np.mean([v for v in series_by_run.values()]))


def pooled_mean(s: pd.Series) -> tuple[float, dict]:
    per = {r: float(s.xs(r, level="run").mean()) for r in RUNS if r in s.index.get_level_values("run")}; return float(np.mean(list(per.values()))), per


def gain_cells(df, a, b_a, ref, b_ref, metric):
    x, y = cells(df, a, b_a, metric), cells(df, ref, b_ref, metric); return (sign(metric) * (x - y)).dropna()


def wil(x):
    x = np.asarray(x, float)
    try: return float(wilcoxon(x).pvalue) if np.any(x != 0) else 1.0
    except ValueError: return 1.0


def fmt_md(df, floatfmt="{:+.4f}"):
    cols = list(df.columns); out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows(): out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else (floatfmt.format(v) if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)


def curve_table(df, metric="auc", learners=None, budgets=(0, 5, 10, 20, 40), kinds=None):
    learners = learners or sorted(set(df.learner)); rows = []
    for kind in (kinds or ["A", "B", "A+B"]):
        d = df if kind == "A+B" else df[df.split == kind]
        for l in learners:
            r = {"split": kind, "learner": l}
            for b in budgets:
                try: v, per = pooled_mean(cells(d, l, b, metric)); r[f"b={b}"] = v
                except Exception: r[f"b={b}"] = np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def gaps_table(df, ref="cr_tgt", metric="auc", budgets=(10, 20), learners=None, kinds=("A", "B", "A+B")):
    rows = []
    learners = [l for l in (learners or sorted(set(df.learner))) if l != ref]
    for kind in kinds:
        d = df if kind == "A+B" else df[df.split == kind]
        for l in learners:
            for b in budgets:
                try: g = gain_cells(d, l, b, ref, b if ref != "cr_src" else 0, metric)
                except Exception: continue
                if not len(g): continue
                pm, per = pooled_mean(g); row = {"split": kind, "learner": l, "b": b, "pooled": pm, **{SHORT[r]: per.get(r, np.nan) for r in RUNS}, "cells": len(g), "p_wilcoxon_cells": wil(g.to_numpy())}
                rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ group-level bootstrap (confirmation)
def judgment_table(df, meta, learner, b, base_cells=None):
    """per (seed, split, run) cell: arrays of d, sign, weight, group, far for a learner at b."""
    l, bb = resolve(learner, b); out = {}
    g = df[(df.learner == l) & (df.budget == bb)]
    mm = meta.set_index(["seed", "split", "run"])
    for _, r in g.iterrows():
        key = (r["seed"], r["split"], r["run"]); m = mm.loc[key]; out[key] = {"d": np.asarray(r["d"]), "sign": np.asarray(m["sign"]), "weight": np.asarray(m["weight"]), "group": np.asarray(m["test_group"]), "far": np.asarray(m["far"])}
    return out


def metric_weighted(d, s, w, mult, which):
    """weighted accuracy / AUC with integer group multiplicities ``mult`` (cluster bootstrap)."""
    ww = w * mult
    if ww.sum() == 0: return np.nan
    if which == "acc": return float((ww * ((d * s) > 0)).sum() / ww.sum())
    if which == "auc":
        pos = s > 0
        if (ww[pos].sum() == 0) or (ww[~pos].sum() == 0): return np.nan
        return float(roc_auc_score(pos, d, sample_weight=mult.astype(float)))
    raise ValueError(which)


def bootstrap_pooled_gain(df, meta, a, ref, b, which, n_boot=400, seed=0, subset=None):
    """Pooled (mean over runs) gain of learner ``a`` over ``ref`` at budget b for metric ``which`` (auc|acc) with a cluster bootstrap over the TEST GROUPS of each run: the groups that occur in any
    cell of a run are resampled with replacement (same multiplicities in every cell of that run).  Returns (estimate, boot array).  subset: 'far' restricts to anchor-far judgments."""
    A, B = judgment_table(df, meta, a, b), judgment_table(df, meta, ref, b if ref != "cr_src" else 0); keys = sorted(set(A) & set(B)); rng = np.random.default_rng(seed)
    per_run_groups = {r: sorted({g for k in keys if k[2] == r for g in A[k]["group"]}) for r in RUNS}; gidx = {r: {g: i for i, g in enumerate(per_run_groups[r])} for r in RUNS}
    def gain(mults):
        vals = {r: [] for r in RUNS}
        for k in keys:
            ma, mb = A[k], B[k]; gi = np.array([gidx[k[2]][g] for g in ma["group"]]); mult = mults[k[2]][gi].astype(float)
            if subset == "far": mult = mult * ma["far"]
            x = metric_weighted(ma["d"], ma["sign"], ma["weight"], mult, which); y = metric_weighted(mb["d"], mb["sign"], mb["weight"], mult, which)
            if not (np.isnan(x) or np.isnan(y)): vals[k[2]].append(x - y)
        return float(np.mean([np.mean(v) for v in vals.values() if v]))
    ones = {r: np.ones(len(per_run_groups[r])) for r in RUNS}; est = gain(ones); boots = []
    for _ in range(n_boot):
        mults = {r: np.bincount(rng.integers(0, len(per_run_groups[r]), len(per_run_groups[r])), minlength=len(per_run_groups[r])) for r in RUNS}; boots.append(gain(mults))
    return est, np.asarray(boots)


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); out = np.empty_like(p); run = 0.0
    for rank, i in enumerate(order): run = max(run, (len(p) - rank) * p[i]); out[i] = min(1.0, run)
    return out


if __name__ == "__main__":
    mode = sys.argv[1]; dirs = [a for a in sys.argv[2:] if not a.startswith("--") and not a.endswith(".json")]; opts = dict(a[2:].split("=", 1) for a in sys.argv[2:] if a.startswith("--") and "=" in a)
    df, meta = load(dirs)
    if mode == "counts":
        m = meta.drop_duplicates(["split", "run", "seed"]); t = m.groupby(["split", "run"])[["touching", "test", "source", "source_dropped", "pool", "pool_dropped", "n_test", "n_far", "n_source_judgments", "n_pool_judgments", "n_frames"]].mean().reset_index(); print(fmt_md(t, "{:.1f}"))
    elif mode == "curve":
        for metric in opts.get("metric", "auc,acc").split(","): print(f"\n#### {metric} (pooled over the three runs; mean of run means)\n"); print(fmt_md(curve_table(df, metric)))
    elif mode == "gaps":
        bs = tuple(int(x) for x in opts.get("b", "10,20").split(","))
        for metric in opts.get("metric", "auc,acc").split(","):
            print(f"\n#### gain in {metric} over {opts.get('ref', 'cr_tgt')} at the same b\n"); print(fmt_md(gaps_table(df, opts.get("ref", "cr_tgt"), metric, bs)))


# ------------------------------------------------------------------------------------------ fast cluster bootstrap over all learners at once (confirmation)
def wauc(d, pos, mult):
    """AUC with integer multiplicities (ties count 1/2) via the pairwise matrix; NaN when a class is empty."""
    dp, dn, mp, mn = d[pos], d[~pos], mult[pos], mult[~pos]
    if mp.sum() == 0 or mn.sum() == 0: return np.nan
    diff = dp[:, None] - dn[None, :]; w = mp[:, None] * mn[None, :]; return float((w * ((diff > 0) + 0.5 * (diff == 0))).sum() / w.sum())


def wacc(d, s, w, mult):
    ww = w * mult; return np.nan if ww.sum() == 0 else float((ww * ((d * s) > 0)).sum() / ww.sum())


def prepare(df, meta, specs):
    """specs: list of (learner, b).  Returns cells[(seed, split, run)] = {"sign", "weight", "group_idx", "far", "d": {spec: array}} and the per-run group lists."""
    mm = meta.set_index(["seed", "split", "run"]); cells = {}; groups = {r: sorted({g for key, row in mm.iterrows() if key[2] == r for g in row["test_group"]}) for r in RUNS}; gidx = {r: {g: i for i, g in enumerate(groups[r])} for r in RUNS}
    for spec in specs:
        l, bb = resolve(*spec); g = df[(df.learner == l) & (df.budget == bb)]
        for _, r in g.iterrows():
            key = (r["seed"], r["split"], r["run"]); m = mm.loc[key]
            c = cells.setdefault(key, {"sign": np.asarray(m["sign"]), "weight": np.asarray(m["weight"]), "gi": np.array([gidx[key[2]][x] for x in m["test_group"]]), "far": np.asarray(m["far"]), "d": {}}); c["d"][spec] = np.asarray(r["d"])
    return cells, groups


def pooled_metrics(cells, groups, mults, specs, metrics=("auc", "acc"), subset=None, kinds=None):
    """pooled[spec][metric] = mean over (split kind, run) of the mean over seeds of the metric (restricted to anchor-far if subset == 'far'), with group multiplicities ``mults[run]``."""
    acc = {(s, m): {} for s in specs for m in metrics}
    for key, c in cells.items():
        if kinds and key[1] not in kinds: continue
        mult = mults[key[2]][c["gi"]].astype(float)
        if subset == "far": mult = mult * c["far"]
        pos = c["sign"] > 0
        for s in specs:
            if s not in c["d"]: continue
            for m in metrics:
                v = wauc(c["d"][s], pos, mult) if m == "auc" else wacc(c["d"][s], c["sign"], c["weight"], mult)
                if not np.isnan(v): acc[(s, m)].setdefault((key[1], key[2]), []).append(v)
    return {k: float(np.mean([np.mean(v) for v in d.values()])) if d else np.nan for k, d in acc.items()}


def cluster_bootstrap(df, meta, specs, n_boot=1000, seed=0, subset=None, kinds=None):
    """Returns (point, boots): point[(spec, metric)] and boots[(spec, metric)] = array over bootstrap draws (groups of each run resampled with replacement, shared by all learners)."""
    cells, groups = prepare(df, meta, specs); ones = {r: np.ones(len(groups[r])) for r in RUNS}; point = pooled_metrics(cells, groups, ones, specs, subset=subset, kinds=kinds); rng = np.random.default_rng(seed)
    boots = {k: [] for k in point}
    for _ in range(n_boot):
        mults = {r: np.bincount(rng.integers(0, len(groups[r]), len(groups[r])), minlength=len(groups[r])) for r in RUNS}; res = pooled_metrics(cells, groups, mults, specs, subset=subset, kinds=kinds)
        for k, v in res.items(): boots[k].append(v)
    return point, {k: np.asarray(v) for k, v in boots.items()}
