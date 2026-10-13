"""Pre-registered tests of the red-team study (PREREGISTRATION_RT1.md): gains of a candidate learner over a reference under stricter evaluations, Holm-corrected as one family.

  python3 rt_confirm.py A1DIR A1BDIR A3DIR A2DIR [--out FILE]
Evaluations (all per seed, mean over budgets and draws; two-sided Wilcoxon over the confirmation seeds; Holm over the whole family; paired bootstrap CI over seeds):
  all     every held-out judgment (standard endpoint)                                   a1 cells, splits A and B, AUC and accuracy
  far     judgments whose two images are both > 0.01 (cosine distance) from every typed anchor    a1 + a1b cells
  blocked time-blocked split (100-frame blocks, 25-frame buffer, graph nodes in the zone removed)   a3 cells (single 'blocked' split)
  dedup   graph nodes within cosine distance 0.003 of any held-out image removed          a2 cells (delta = 0.003)
  aw8far  anchor-weight check: aw8_all vs baseline on 'far' judgments                     a1 cells
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

FAR = 0.01
CAND, REF1, REF2, BASE = "lap3_k10_st4", "aw8_all_st2lr0.5", "lap3_k10", "baseline"


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); out = np.empty_like(p); run = 0.0
    for rank, i in enumerate(order): run = max(run, (len(p) - rank) * p[i]); out[i] = min(1.0, run)
    return out


def load_a1(d, b):
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(Path(d) / "seed*.csv.gz")))], ignore_index=True)
    if b: df = df.merge(pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(Path(b) / "seed*.csv")))])[["seed", "split", "j", "anc_all_min"]], on=["seed", "split", "j"], how="left")
    return df


def metric_of(g, metric):
    if metric == "auc": return roc_auc_score(g.sign > 0, g.d) if len(set(g.sign)) == 2 else np.nan
    return float((g.weight * ((g.d * g.sign) > 0)).sum() / g.weight.sum())


def per_seed_a1(df, split, metric, cand, ref, far=False):
    g = df[df.split == split]
    if far: g = g[g.anc_all_min > FAR]
    out = {}
    for (seed, budget, draw), gg in g.groupby(["seed", "budget", "draw"]):
        a, r = gg[gg.learner == cand].sort_values("j"), gg[gg.learner == ref].sort_values("j")
        if len(a) and len(a) == len(r): out.setdefault(seed, []).append(metric_of(a, metric) - metric_of(r, metric))
    return pd.Series({k: np.nanmean(v) for k, v in out.items()})


def per_seed_json(df, split, metric, cand, ref, **flt):
    g = df[df.split == split] if "split" in df and split else df
    for k, v in flt.items(): g = g[g[k] == v]
    a = g[g.learner == cand].set_index(["seed", "budget", "draw"])[metric]; b = g[g.learner == ref].set_index(["seed", "budget", "draw"])[metric]; return (a - b).dropna().groupby(level=0).mean()


def load_json(d):
    return pd.DataFrame([r for f in sorted(glob.glob(str(Path(d) / "seed*.json"))) for r in json.loads(Path(f).read_text()) if "learner" in r])


def run(a1, a1b, a3, a2):
    df = load_a1(a1, a1b); j3 = load_json(a3); j2 = load_json(a2); tests = []
    def add(name, split, metric, ref, per, cand=CAND): tests.append({"evaluation": name, "split": split, "metric": metric, "candidate": cand, "reference": ref, "per": per})
    for split in "AB":
        for m in ("auc", "acc"):
            add("all", split, m, REF1, per_seed_a1(df, split, m, CAND, REF1)); add("far", split, m, REF1, per_seed_a1(df, split, m, CAND, REF1, far=True)); add("dedup", split, m, REF1, per_seed_json(j2, split, m, CAND, REF1, delta=0.003))
            add("aw8far", split, m, BASE, per_seed_a1(df, split, m, "aw8_all", BASE, far=True), cand="aw8_all")
    for m in ("auc", "acc"): add("blocked", "blocked", m, REF1, per_seed_json(j3, None, m, CAND, REF1))
    rows = []
    for t in tests:
        per = t.pop("per").dropna(); rng = np.random.default_rng(0); bs = rng.choice(per.to_numpy(), (3000, len(per))).mean(1)
        try: p = float(wilcoxon(per).pvalue)
        except ValueError: p = 1.0
        rows.append({**t, "n_seeds": len(per), "gain": float(per.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "share_pos": float((per > 0).mean()), "p": p})
    out = pd.DataFrame(rows); out["holm"] = holm(out.p.to_numpy()); return out


def verdicts(out):
    rows = []
    for ev, g in out.groupby("evaluation", sort=False):
        pos = bool((g.gain > 0).all()); sig = int((g.holm < 0.05).sum()); n = len(g); need = 2 if n == 2 else 3
        v = "robust" if pos and sig >= need else ("weakened (all gains positive, too few Holm-significant)" if pos else "not supported (some gain <= 0)")
        if ev == "far" and not (pos and sig >= need): v = v + "; anchor-far gain small or absent"
        rows.append({"evaluation": ev, "tests": n, "all_positive": pos, "n_holm_sig": sig, "verdict": v})
    return pd.DataFrame(rows)


def md(df, fmt="{:+.4f}"):
    cols = list(df.columns); o = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows(): o.append("| " + " | ".join(fmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(o)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]; out = run(*a); print(md(out)); print(); print(md(verdicts(out), "{}"))
