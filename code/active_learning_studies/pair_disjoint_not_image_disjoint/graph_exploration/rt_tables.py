"""Tables of the red-team audits, generated from the result files (nothing typed by hand).

  python3 rt_tables.py a1 DIR [--cand=lap3_k10_st4 --ref=aw8_all_st2lr0.5 --by=nov_train_max --bins=4]   gain by novelty stratum of the test judgment
"""
import glob, sys
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


def to_md(df, fmt="{:+.4f}"):
    cols = list(df.columns); out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows(): out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else (fmt.format(v) if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    mode = sys.argv[1]; opts = dict(a[2:].split("=", 1) for a in sys.argv[2:] if a.startswith("--") and "=" in a); dirs = [a for a in sys.argv[2:] if not a.startswith("--")]
    if mode == "a1":
        df = load_a1(dirs[0]); j = per_judgment(df, opts.get("cand", "lap3_k10_st4"), opts.get("ref", "aw8_all_st2lr0.5"))
        print(to_md(stratified(j, opts.get("by", "nov_train_max"), int(opts.get("bins", 4)))))
    if mode == "a1b":   # gain by proximity of the test judgment's images to a typed anchor (ideal image)
        df = load_a1(dirs[0]); j = per_judgment(df, opts.get("cand", "lap3_k10_st4"), opts.get("ref", "aw8_all_st2lr0.5")); b = load_a1b(dirs[1]); j = j.merge(b, on=["seed", "split", "j"], how="inner")
        print(to_md(stratified_custom(anchor_bins(j, opts.get("col", "anc_all_min")))))
