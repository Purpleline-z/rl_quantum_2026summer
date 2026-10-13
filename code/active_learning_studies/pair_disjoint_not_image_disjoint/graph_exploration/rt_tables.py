"""Tables of the red-team audits, generated from the result files (nothing typed by hand).

  python3 rt_tables.py a1 DIR [--cand=lap3_k10_st4 --ref=aw8_all_st2lr0.5 --by=nov_train_max --bins=4]   gain by novelty stratum of the test judgment
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd

KEYS = ["seed", "split", "budget", "draw", "j"]


def load_a1(d) -> pd.DataFrame:
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(Path(d) / "seed*.csv.gz")))], ignore_index=True)


def per_judgment(df, cand, ref) -> pd.DataFrame:
    a = df[df.learner == cand].set_index(KEYS); b = df[df.learner == ref].set_index(KEYS); j = a.join(b[["d"]], rsuffix="_ref", how="inner")
    z = j.d * j.sign; zr = j.d_ref * j.sign
    j["acc_c"] = (z > 0).astype(float); j["acc_r"] = (zr > 0).astype(float); j["ll_c"] = np.logaddexp(0, -z); j["ll_r"] = np.logaddexp(0, -zr)
    return j.reset_index()


def stratified(j, by="nov_train_max", bins=4, edges=None) -> pd.DataFrame:
    edges = np.quantile(j[by], np.linspace(0, 1, bins + 1)) if edges is None else edges
    j = j.assign(stratum=pd.cut(j[by], np.unique(edges), include_lowest=True, labels=False)); rows = []
    for (split, s), g in j.groupby(["split", "stratum"]):
        per = g.groupby("seed").apply(lambda x: pd.Series({"acc_gain": (x.weight * (x.acc_c - x.acc_r)).sum() / x.weight.sum(), "ll_gain": (x.weight * (x.ll_r - x.ll_c)).sum() / x.weight.sum()}))
        rng = np.random.default_rng(0); out = {"split": split, "stratum": int(s), f"{by}_range": f"{g[by].min():.4f}-{g[by].max():.4f}", "n_judgments": len(g), "n_seeds": len(per)}
        for m in ("acc_gain", "ll_gain"):
            b = rng.choice(per[m].to_numpy(), (2000, len(per))).mean(1); out[m] = per[m].mean(); out[m + "_lo"] = np.percentile(b, 2.5); out[m + "_hi"] = np.percentile(b, 97.5)
        rows.append(out)
    return pd.DataFrame(rows)


def load_a1b(d) -> pd.DataFrame:
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(Path(d) / "seed*.csv")))], ignore_index=True)


def anchor_bins(j, col="anc_all_min", edges=(0.001, 0.01, 0.03)):
    j = j.copy(); j["stratum"] = np.digitize(j[col], edges); labels = [f"<={edges[0]}"] + [f"{edges[i]}-{edges[i + 1]}" for i in range(len(edges) - 1)] + [f">{edges[-1]}"]; j["bin"] = [labels[i] for i in j["stratum"]]; return j


def stratified_custom(j, by_cols=("split", "bin")) -> pd.DataFrame:
    rows = []
    for key, g in j.groupby(list(by_cols)):
        per = g.groupby("seed").apply(lambda x: pd.Series({"acc_gain": (x.weight * (x.acc_c - x.acc_r)).sum() / x.weight.sum(), "ll_gain": (x.weight * (x.ll_r - x.ll_c)).sum() / x.weight.sum(), "acc_ref": (x.weight * x.acc_r).sum() / x.weight.sum(), "acc_cand": (x.weight * x.acc_c).sum() / x.weight.sum()}))
        rng = np.random.default_rng(0); out = dict(zip(by_cols, key if isinstance(key, tuple) else (key,))); out["n_judgments"] = len(g); out["n_seeds"] = len(per)
        for m in ("acc_gain", "ll_gain"):
            b = rng.choice(per[m].to_numpy(), (2000, len(per))).mean(1); out[m] = per[m].mean(); out[m + "_lo"] = np.percentile(b, 2.5); out[m + "_hi"] = np.percentile(b, 97.5)
        out["acc_ref"], out["acc_cand"] = per["acc_ref"].mean(), per["acc_cand"].mean(); rows.append(out)
    return pd.DataFrame(rows)


def load_json_dir(d) -> pd.DataFrame:
    rows = [r for f in sorted(glob.glob(str(Path(d) / "seed*.json"))) for r in json.loads(Path(f).read_text()) if "learner" in r]; return pd.DataFrame(rows)


SIGN = {"auc": 1, "acc": 1, "ll": -1}


def gain_table(df, cand, ref, group_cols=("split",), metrics=("auc", "acc", "ll", "htr_acc")) -> pd.DataFrame:
    """Per group: mean over seeds of the per-seed mean paired gain (cand - ref; log-loss: ref - cand), paired bootstrap CI, share of seeds positive, Wilcoxon p."""
    from scipy.stats import wilcoxon
    rows = []
    for key, g in df.groupby(list(group_cols)):
        a = g[g.learner == cand].set_index(["seed", "budget", "draw"]); b = g[g.learner == ref].set_index(["seed", "budget", "draw"]); common = a.index.intersection(b.index)
        if not len(common): continue
        out = dict(zip(group_cols, key if isinstance(key, tuple) else (key,))); out["n_seeds"] = len({i[0] for i in common})
        for m in metrics:
            if m not in a: continue
            per = (SIGN.get(m, 1) * (a.loc[common, m] - b.loc[common, m])).groupby(level=0).mean().dropna(); rng = np.random.default_rng(0); bs = rng.choice(per.to_numpy(), (2000, len(per))).mean(1)
            try: p = float(wilcoxon(per).pvalue)
            except ValueError: p = 1.0
            out[m] = per.mean(); out[m + "_lo"] = np.percentile(bs, 2.5); out[m + "_hi"] = np.percentile(bs, 97.5); out[m + "_share+"] = float((per > 0).mean()); out[m + "_p"] = p
        rows.append(out)
    return pd.DataFrame(rows)


def auc_by_stratum(df, b, cand, ref, bins, col="anc_all_min") -> pd.DataFrame:
    """AUC gain (cand - ref) within strata of a distance column, computed per (seed, split, budget, draw) and averaged; bins = [(label, lo, hi)] meaning lo < value <= hi."""
    from sklearn.metrics import roc_auc_score
    j = df[df.learner.isin([cand, ref])].merge(b[["seed", "split", "j", col]], on=["seed", "split", "j"]); rows = []
    for label, lo, hi in bins:
        sub = j[(j[col] > lo) & (j[col] <= hi)]
        for split, g in sub.groupby("split"):
            per = {}
            for (seed, budget, draw), gg in g.groupby(["seed", "budget", "draw"]):
                a, r = gg[gg.learner == cand].sort_values("j"), gg[gg.learner == ref].sort_values("j")
                if len(a) == len(r) and len(set(a.sign)) == 2: per.setdefault(seed, []).append(roc_auc_score(a.sign > 0, a.d) - roc_auc_score(r.sign > 0, r.d))
            per = pd.Series({k: np.mean(v) for k, v in per.items()}); rng = np.random.default_rng(0); bs = rng.choice(per.to_numpy(), (2000, len(per))).mean(1)
            rows.append({"split": split, "stratum": label, "n_judgments": len(g) // 2, "n_seeds": len(per), "auc_gain": per.mean(), "CI": f"[{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]"})
    return pd.DataFrame(rows)


def to_md(df, fmt="{:+.4f}"):
    cols = list(df.columns); out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows(): out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else (fmt.format(v) if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    mode = sys.argv[1]; opts = dict(a[2:].split("=", 1) for a in sys.argv[2:] if a.startswith("--") and "=" in a); dirs = [a for a in sys.argv[2:] if not a.startswith("--")]
    if mode == "a1":
        df = load_a1(dirs[0]); j = per_judgment(df, opts.get("cand", "lap3_k10_st4"), opts.get("ref", "aw8_all_st2lr0.5"))
        print(to_md(stratified(j, opts.get("by", "nov_train_max"), int(opts.get("bins", 4)))))
    if mode in ("a2", "a3", "a1c"):   # aggregated-metric audits: gains of cand over ref per delta / per split
        df = load_json_dir(dirs[0]); cols = {"a2": ("split", "delta"), "a3": ("split",), "a1c": ("split",)}[mode]
        for cand, ref in [x.split(":") for x in opts.get("pairs", "lap3_k10_st4:aw8_all_st2lr0.5,lap3_k10:aw8_all_st2lr0.5,lap3_k10_st4:lap3_k10").split(",")]:
            t = gain_table(df, cand, ref, cols); print(f"\n#### {cand} - {ref}\n"); print(to_md(t[[c for c in t.columns if not c.endswith(("_lo", "_hi"))]] if "full" not in opts else t))
        if mode == "a2": print("\nnodes kept (mean) by delta:"); print(to_md(df.groupby(["split", "delta"]).n_nodes.mean().reset_index(), "{:.0f}"))
    if mode == "a4":   # independent re-implementation vs main implementation
        ref = load_a1(dirs[1]); recs = [r for f in sorted(glob.glob(str(Path(dirs[0]) / "seed*.json"))) for r in json.loads(Path(f).read_text())]; rows = []
        for r in recs:
            m = ref[(ref.seed == r["seed"]) & (ref.split == r["split"]) & (ref.budget == r["budget"]) & (ref.draw == r["draw"]) & (ref.learner == r["learner"])].sort_values("j")
            if len(m) == len(r["d"]): rows.append({"learner": r["learner"], "split": r["split"], "max_abs_d_diff": float(np.abs(np.array(r["d"]) - m.d.to_numpy()).max())})
        t = pd.DataFrame(rows).groupby(["learner", "split"]).max_abs_d_diff.agg(["count", "median", "max"]).reset_index(); print(to_md(t, "{:.2e}"))
    if mode == "a5":   # variance components of the paired gain
        recs = [r for f in sorted(glob.glob(str(Path(dirs[0]) / "seed*.json"))) for r in json.loads(Path(f).read_text())]; chk = [r for r in recs if "thread_check_max_abs_diff" in r]; df = pd.DataFrame([r for r in recs if "learner" in r])
        if chk: print(f"thread-count check (1 vs 4 threads, one fit): max |d diff| = {chk[0]['thread_check_max_abs_diff']:.2e}\n")
        rows = []
        for cand, ref in (("lap3_k10_st4", "aw8_all_st2lr0.5"), ("lap3_k10_st4", "lap3_k10"), ("lap3_k10", "aw8_all_st2lr0.5")):
            for m in ("auc", "acc"):
                a = df[df.learner == cand].set_index(["seed", "split", "budget", "init"])[m]; b = df[df.learner == ref].set_index(["seed", "split", "budget", "init"])[m]; g = (a - b).dropna().reset_index(name="g")
                cell = g.groupby(["seed", "split", "budget"]).g; within = float(cell.var(ddof=1).mean()); between = float(cell.mean().var(ddof=1)); rows.append({"cand - ref": f"{cand} - {ref}", "metric": m, "mean gain": float(g.g.mean()), "sd of the gain across inits (same label set)": within ** .5,
                                                                         "sd of the cell mean gain across seeds/splits/budgets": between ** .5, "share of inits with gain > 0": float((g.g > 0).mean())})
        print(to_md(pd.DataFrame(rows)))
    if mode == "a8":   # endpoint sensitivity from the per-judgment predictions
        df = load_a1(dirs[0]); cand, ref = opts.get("cand", "lap3_k10_st4"), opts.get("ref", "aw8_all_st2lr0.5"); j = per_judgment(df, cand, ref); rows = []
        for split, g in j.groupby("split"):
            for name, sub in (("all judgments", g), ("HTR only (k=4)", g[g.k == 4]), ("(1x1) only (k=0)", g[g.k == 0]), ("c(6x2) only (k=2)", g[g.k == 2]), ("RT13 only (k=3)", g[g.k == 3])):
                per = sub.groupby("seed").apply(lambda x: pd.Series({"acc_weighted": (x.weight * (x.acc_c - x.acc_r)).sum() / x.weight.sum(), "acc_unweighted": (x.acc_c - x.acc_r).mean(), "ll_gain": (x.ll_r - x.ll_c).mean()}))
                row = {"split": split, "subset": name, "n_judgments": len(sub), "n_seeds": len(per)}; rng = np.random.default_rng(0)
                for m in per:
                    bs = rng.choice(per[m].to_numpy(), (2000, len(per))).mean(1); row[m] = per[m].mean(); row[m + " CI"] = f"[{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]"
                rows.append(row)
        print(to_md(pd.DataFrame(rows)))
    if mode == "a1auc":
        df = load_a1(dirs[0]); b = load_a1b(dirs[1]); cand, ref = opts.get("cand", "lap3_k10_st4"), opts.get("ref", "aw8_all_st2lr0.5")
        print(to_md(auc_by_stratum(df, b, cand, ref, [("all", -1, 9), ("anchor-near (<=0.001)", -1, 0.001), ("0.001-0.01", 0.001, 0.01), ("0.01-0.03", 0.01, 0.03), ("anchor-far (>0.03)", 0.03, 9), ("far from anchors (>0.01)", 0.01, 9)])))
    if mode == "a1b":   # gain by proximity of the test judgment's images to a typed anchor (ideal image)
        df = load_a1(dirs[0]); j = per_judgment(df, opts.get("cand", "lap3_k10_st4"), opts.get("ref", "aw8_all_st2lr0.5")); b = load_a1b(dirs[1]); j = j.merge(b, on=["seed", "split", "j"], how="inner")
        print(to_md(stratified_custom(anchor_bins(j, opts.get("col", "anc_all_min")))))
