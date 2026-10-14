"""Tables of the latent-type study, generated from the result directories of ``new_directions/lt_run.py`` (nothing typed by hand).
  python3 lt_tables.py gains DIR --cand=a,b --ref=lap3_k10_st4 [--b=A1B_DIR] [--thr=0.01]
Per candidate and split: mean paired gain over seeds (cand - ref; log-loss: ref - cand), bootstrap CI, share of seeds > 0, two-sided Wilcoxon p (uncorrected) for the standard endpoint (all judgments) and, if --b is given,
the anchor-far stratum (both images of the test judgment farther than --thr cosine distance from every typed ideal image, ``anc_all_min`` of rt_a1b)."""
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "new_directions"))
import rt_tables as rt

KEYS = ["seed", "split", "budget", "draw", "j"]


def load(d): return rt.load_json_dir(d)
def load_long(d): return rt.load_a1(d)


def seed_gains_standard(df, cand, ref, split):
    a = df[(df.learner == cand) & (df.split == split)].set_index(["seed", "budget", "draw"]); b = df[(df.learner == ref) & (df.split == split)].set_index(["seed", "budget", "draw"]); common = a.index.intersection(b.index)
    out = {}
    for m, sign in (("auc", 1), ("acc", 1), ("ll", -1)): out[m] = (sign * (a.loc[common, m] - b.loc[common, m])).groupby(level=0).mean().dropna()
    return out


def seed_gains_far(long, anc, cand, ref, split, thr=0.01):
    j = long[(long.split == split) & long.learner.isin([cand, ref])].merge(anc[["seed", "split", "j", "anc_all_min"]], on=["seed", "split", "j"]); j = j[j.anc_all_min > thr]; per_auc, per_acc, per_ll = {}, {}, {}
    for (seed, budget, draw), g in j.groupby(["seed", "budget", "draw"]):
        a, r = g[g.learner == cand].sort_values("j"), g[g.learner == ref].sort_values("j")
        if len(a) != len(r) or not len(a): continue
        w = a.weight.to_numpy(); za, zr = a.d.to_numpy() * a.sign.to_numpy(), r.d.to_numpy() * r.sign.to_numpy()
        per_acc.setdefault(seed, []).append((w * ((za > 0).astype(float) - (zr > 0).astype(float))).sum() / w.sum()); per_ll.setdefault(seed, []).append((w * (np.logaddexp(0, -zr) - np.logaddexp(0, -za))).sum() / w.sum())
        if len(set(a.sign)) == 2: per_auc.setdefault(seed, []).append(roc_auc_score(a.sign > 0, a.d) - roc_auc_score(r.sign > 0, r.d))
    f = lambda dct: pd.Series({k: np.mean(v) for k, v in dct.items()}); return {"auc": f(per_auc), "acc": f(per_acc), "ll": f(per_ll)}, int(len(j[j.learner == cand]) / max(1, j[j.learner == cand].groupby(["seed", "budget", "draw"]).ngroups))


def summarise(per: pd.Series):
    per = per.dropna(); rng = np.random.default_rng(0); bs = rng.choice(per.to_numpy(), (2000, len(per))).mean(1)
    try: p = float(wilcoxon(per).pvalue)
    except ValueError: p = 1.0
    return dict(gain=per.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), share=float((per > 0).mean()), p=p, n=len(per))


def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order): run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def table(df, long, anc, cands, ref, splits=("A", "B"), thr=0.01):
    rows = []
    for cand in cands:
        for split in splits:
            s = seed_gains_standard(df, cand, ref, split)
            if not len(s["auc"]): continue
            row = {"cand": cand, "split": split, "stratum": "all"}
            for m in ("auc", "acc", "ll"): r = summarise(s[m]); row.update({f"{m}": r["gain"], f"{m}_lo": r["lo"], f"{m}_hi": r["hi"], f"{m}_share": r["share"], f"{m}_p": r["p"]})
            row["n_seeds"] = r["n"]; rows.append(row)
            if long is not None:
                s, nj = seed_gains_far(long, anc, cand, ref, split, thr); row = {"cand": cand, "split": split, "stratum": f"far>{thr}"}
                for m in ("auc", "acc", "ll"): r = summarise(s[m]); row.update({f"{m}": r["gain"], f"{m}_lo": r["lo"], f"{m}_hi": r["hi"], f"{m}_share": r["share"], f"{m}_p": r["p"]})
                row["n_seeds"] = r["n"]; rows.append(row)
    return pd.DataFrame(rows)


def to_md(t, cols=("cand", "split", "stratum", "auc", "auc_p", "acc", "acc_p", "ll", "ll_p", "auc_share", "n_seeds")):
    t = t[[c for c in cols if c in t.columns]]; out = ["| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
    for _, r in t.iterrows(): out.append("| " + " | ".join(f"{v:+.4f}" if (isinstance(v, float) and not c.endswith(("_p", "share"))) else (f"{v:.3f}" if isinstance(v, float) else str(v)) for c, v in zip(t.columns, r)) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    mode = sys.argv[1]; opts = dict(a[2:].split("=", 1) for a in sys.argv[2:] if a.startswith("--") and "=" in a); dirs = [a for a in sys.argv[2:] if not a.startswith("--")]
    if mode == "gains":
        df = load(dirs[0]); cands = opts["cand"].split(","); ref = opts.get("ref", "lap3_k10_st4"); thr = float(opts.get("thr", 0.01))
        long = anc = None
        if "b" in opts: long = load_long(dirs[0]); anc = rt.load_a1b(opts["b"])
        print(to_md(table(df, long, anc, cands, ref, thr=thr)))
