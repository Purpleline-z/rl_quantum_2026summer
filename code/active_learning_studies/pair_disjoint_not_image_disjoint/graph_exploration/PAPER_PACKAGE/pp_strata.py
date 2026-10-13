"""Anchor-proximity ("anchor-near" / "anchor-far") and novelty strata from the red-team per-judgment predictions (new_directions/results/rt/{a1,a1b,a1c}; development block seeds 8800-8819;
the red-team confirmation block 8850-8884, rt/c_a1 + rt/c_a1b, is used automatically when it exists).

Stratum of a held-out judgment = min over its two images of the distance (1 - cos, centred+normalised frozen SimCLR features, the graph's geometry) to the nearest typed anchor (`anc_all_min` of rt_a1b.py:
references + utility + outer-test ideal images = the anchors of aw8_all).  anchor-near <= 0.001, 0.001-0.01, 0.01-0.03, anchor-far > 0.03 (this package's secondary endpoint: BOTH images > 0.03 from every anchor);
the red team's pre-registered 'far' threshold 0.01 is reported as sensitivity.  Per (seed, split, budget, draw) the metric is computed inside the stratum (AUC needs both outcomes present), the gain is
cand - ref on identical label sets, averaged over cells per seed, then bootstrapped over seeds (same as rt_tables.auc_by_stratum, which this script is cross-checked against).
Outputs tables/strata.csv, tables/anchor_fractions.csv, tables/novelty.csv, numbers_strata.json."""
import glob

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import pp_lib as L
from pp_lib import RES, Numbers

BINS = [("all", -1, 9), ("near<=0.001", -1, 0.001), ("0.001-0.01", 0.001, 0.01), ("0.01-0.03", 0.01, 0.03), ("far>0.03", 0.03, 9), ("far>0.01 (red-team threshold)", 0.01, 9)]
PAIRS = {"a1": [("lap3_k10_st4", "aw8_all_st2lr0.5"), ("lap3_k10_st4", "aw8_all"), ("lap3_k10", "aw8_all"), ("lap3_k10_st4", "lap3_k10"), ("aw8_all_st2lr0.5", "aw8_all")],
         "a1c": [("aw8_all", "baseline"), ("aw8", "baseline"), ("aw8_all", "aw8")]}


def load_csvs(d, pat):
    fs = sorted(glob.glob(str(RES / d / pat)))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def cell_table(df, b, learners, bins=BINS, col="anc_all_min"):
    """Metric of every learner inside every stratum for every (seed, split, budget, draw) cell: rows seed, split, budget, draw, learner, stratum, n, auc, acc, ll (computed once, reused by every pair)."""
    j = df[df.learner.isin(learners)].merge(b[["seed", "split", "j", col]], on=["seed", "split", "j"]); rows = []
    for label, lo, hi in bins:
        sub = j[(j[col] > lo) & (j[col] <= hi)]
        for (seed, split, budget, draw, learner), g in sub.groupby(["seed", "split", "budget", "draw", "learner"]):
            g = g.sort_values("j"); z = g.d.to_numpy() * g.sign.to_numpy(); w = g.weight.to_numpy(); s = g.sign.to_numpy()
            auc = roc_auc_score(s > 0, g.d.to_numpy()) if len(set(s)) == 2 else np.nan
            rows.append((seed, split, budget, draw, learner, label, len(g), auc, float((w * (z > 0)).sum() / w.sum()), float((w * np.logaddexp(0, -z)).sum() / w.sum()), tuple(g.j)))
    return pd.DataFrame(rows, columns=["seed", "split", "budget", "draw", "learner", "stratum", "n", "auc", "acc", "ll", "jset"])


def stratum_gains(ct, cand, ref):
    a = ct[ct.learner == cand].set_index(["seed", "split", "budget", "draw", "stratum"]); r = ct[ct.learner == ref].set_index(["seed", "split", "budget", "draw", "stratum"]); common = a.index.intersection(r.index)
    a, r = a.loc[common], r.loc[common]; assert (a.jset == r.jset).all(), "learners were evaluated on different judgments"
    d = pd.DataFrame({"auc": a.auc - r.auc, "acc": a.acc - r.acc, "ll": r.ll - a.ll, "n": a.n}); rows = []
    for (split, label), g in d.groupby(level=["split", "stratum"]):
        per = g.groupby(level="seed")[["auc", "acc", "ll"]].mean()
        for m in ("auc", "acc", "ll"):
            x = per[m].dropna(); lo_, hi_ = L.boot_ci(x.to_numpy(), 2000)
            rows.append(dict(cand=cand, ref=ref, split=split, stratum=label, metric=m, gain=float(x.mean()), lo=lo_, hi=hi_, n_seeds=int(len(x)), judgments_per_cell=float(g.n.mean())))
    return pd.DataFrame(rows)


def main():
    N = Numbers("strata"); out = []
    sources = [("development (rt 8800-8819)", "a1", "a1b", PAIRS["a1"]), ("development (rt 8800-8819)", "a1c", "a1b", PAIRS["a1c"])]
    complete = lambda d, pat: len(glob.glob(str(RES / "rt" / d / pat))) == 70      # 35 seeds x splits A, B; a partially finished red-team confirmation is NOT analysed
    if complete("c_a1", "seed*.csv.gz") and complete("c_a1b", "seed*.csv"): sources += [("confirmation (rt 8850-8884)", "c_a1", "c_a1b", PAIRS["a1"])]
    else: print("red-team confirmation block (rt/c_a1, rt/c_a1b) not complete: not included")
    for src, a1, a1b, pairs in sources:
        df = load_csvs(f"rt/{a1}", "seed*.csv.gz"); b = load_csvs(f"rt/{a1b}", "seed*.csv")
        ct = cell_table(df, b, sorted({x for p in pairs for x in p}))
        for cand, ref in pairs:
            t = stratum_gains(ct, cand, ref); t.insert(0, "source", src); out.append(t)
            for _, r in t.iterrows():
                for fld in ("gain", "lo", "hi"): N.put(f"st.{'conf' if src.startswith('conf') else 'dev'}.{cand}.{ref}.{r.split}.{r.stratum}.{r.metric}.{fld}", r[fld])
    t = pd.concat(out, ignore_index=True); t.to_csv(L.TABLES / "strata.csv", index=False)
    # cross-check against the red team's own function (AUC gain, lap3_k10_st4 - aw8_all_st2lr0.5, development block)
    import rt_tables as RT
    df = load_csvs("rt/a1", "seed*.csv.gz"); b = load_csvs("rt/a1b", "seed*.csv"); chk = RT.auc_by_stratum(df, b, "lap3_k10_st4", "aw8_all_st2lr0.5", [("all", -1, 9), ("far>0.03", 0.03, 9)])
    for _, r in chk.iterrows():
        mine = t[(t.source.str.startswith("dev")) & (t.cand == "lap3_k10_st4") & (t.ref == "aw8_all_st2lr0.5") & (t.split == r.split) & (t.stratum == r.stratum) & (t.metric == "auc")].gain.iloc[0]
        assert abs(mine - r.auc_gain) < 1e-9, (r.split, r.stratum, mine, r.auc_gain)
    # fractions of held-out judgments by distance to the nearest typed anchor (unique judgments of a1b: seed x split x j)
    rows = []
    for col, name in (("anc_all_min", "any typed anchor (references + utility + outer-test ideal images)"), ("anc_ref_min", "reference images only")):
        for lab, lo, hi in BINS[:-1] + [("<=0.003", -1, 0.003)]:
            sel = b[(b[col] > lo) & (b[col] <= hi)]; frac = len(sel) / len(b); rows.append(dict(anchor_set=name, stratum=lab, share_of_judgments=frac, n_judgments=len(sel), n_total=len(b)))
            N.put(f"af.{col}.{lab}", frac)
    for col in ("anc_all_min", "anc_ref_min"):
        N.put(f"af.{col}.le_0.001", float((b[col] <= 0.001).mean())); N.put(f"af.{col}.le_0.01", float((b[col] <= 0.01).mean()))
    pd.DataFrame(rows).to_csv(L.TABLES / "anchor_fractions.csv", index=False); N.put("af.n_total", len(b))
    # novelty of the test judgment with respect to the training images (distance to the nearest labelled image), learner-independent
    nov = df[df.learner == "aw8_all"]; rows = []
    for budget, g in nov.groupby("budget"):
        for col in ("nov_train_min", "nov_train_max", "nov_node_min", "nov_node_max"):
            q = g[col].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_numpy(); rows.append(dict(budget=int(budget), column=col, q10=q[0], q25=q[1], median=q[2], q75=q[3], q90=q[4], share_le_0001=float((g[col] <= 0.001).mean()), share_le_001=float((g[col] <= 0.01).mean()), share_gt_003=float((g[col] > 0.03).mean())))
            for k, nm in zip(("q10", "q25", "median", "q75", "q90"), q): N.put(f"nov.b{int(budget)}.{col}.{k}", float(nm))
            N.put(f"nov.b{int(budget)}.{col}.share_le_001", float((g[col] <= 0.01).mean()))
    pd.DataFrame(rows).to_csv(L.TABLES / "novelty.csv", index=False)
    # compact summary table for the documents: AUC gain by stratum, split A / split B, per pair, per source
    cols = ["near<=0.001", "0.001-0.01", "0.01-0.03", "far>0.03", "far>0.01 (red-team threshold)", "all"]; lines = ["| source | candidate - reference | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for src in t.source.unique():
        for (cand, ref), g in t[(t.source == src) & (t.metric == "auc")].groupby(["cand", "ref"], sort=False):
            cells = []
            for c in cols:
                x = g[g.stratum == c].set_index("split"); cells.append(" / ".join(L.fmt_gain(x.loc[s].gain, x.loc[s].lo, x.loc[s].hi) for s in ("A", "B")) if len(x) == 2 else "")
            lines.append(f"| {src} | {cand} - {ref} | " + " | ".join(cells) + " |")
    L.write(L.TABLES / "strata_summary.md", "\n".join(lines) + "\n"); N.save(); print(len(t), "strata rows")


if __name__ == "__main__":
    main()
