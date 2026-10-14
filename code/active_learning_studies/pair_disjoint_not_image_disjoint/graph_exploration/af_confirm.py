"""Pre-registered tests of the anchor-far study (PREREGISTRATION_AF1.md): candidates vs the baseline lap3_k10_st4 (= af_base) on the anchor-far stratum, the standard endpoint and the time-blocked split.

  python3 af_confirm.py SPEC.json A1DIRS A1BDIR A3DIRS [--groups=CSV] [--out=FILE]
SPEC.json: {"reference": "af_base", "candidates": [{"name": ..., "control": ...}, ...], "far": 0.01, "margin": 0.002}
A1DIRS / A3DIRS may be comma-separated directory lists.

Primary family (Holm over all of it): candidates x {A, B} x {AUC, accuracy} on the anchor-far stratum (both images > `far` cosine distance from every typed anchor).
Blocked family (Holm over it, reported with equal weight): candidates x {AUC, accuracy} on the time-blocked split.
Standard endpoint: candidates x {A, B} AUC and accuracy, all held-out judgments (not tested; the success rule only needs "AUC not worse than -margin").
Success of a candidate: all 4 far gains > 0, >= 3 of the 4 Holm-adjusted p < 0.05, standard-endpoint AUC gain >= -margin in A and in B, and the far gain of the candidate larger than the far gain of its
shuffled / permuted control in all 4 cells.  Per-seed values are means over budgets and draws; two-sided Wilcoxon signed-rank over seeds; paired bootstrap CI over seeds.
Cluster-aware add-on (not part of the Holm decision): two-way bootstrap over seeds and pair groups (af_power.py) when --groups is given.
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

import af_tables as T


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); out = np.empty_like(p); run = 0.0
    for rank, i in enumerate(order): run = max(run, (len(p) - rank) * p[i]); out[i] = min(1.0, run)
    return out


def load_json(dirs):
    return pd.DataFrame([r for d in dirs.split(",") for f in sorted(glob.glob(str(Path(d) / "seed*.json"))) for r in json.loads(Path(f).read_text()) if "learner" in r])


def per_seed_json(df, metric, cand, ref):
    a = df[df.learner == cand].set_index(["seed", "budget", "draw"])[metric]; b = df[df.learner == ref].set_index(["seed", "budget", "draw"])[metric]
    return (a - b).dropna().groupby(level=0).mean()


def test(per, seed=0):
    per = per.dropna()
    if len(per) < 3: return dict(n_seeds=len(per), gain=np.nan, lo=np.nan, hi=np.nan, share_pos=np.nan, p=np.nan)
    rng = np.random.default_rng(seed); bs = rng.choice(per.to_numpy(), (3000, len(per))).mean(1)
    try: p = float(wilcoxon(per).pvalue) if np.any(per.to_numpy() != 0) else 1.0
    except ValueError: p = 1.0
    return dict(n_seeds=len(per), gain=float(per.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), share_pos=float((per > 0).mean()), p=p)


def run(spec, a1, a1b, a3):
    ref, far, margin = spec["reference"], spec.get("far", 0.01), spec.get("margin", 0.002); df = T.load(a1, a1b); j3 = load_json(a3); rows = []
    for c in spec["candidates"]:
        for who, name in (("candidate", c["name"]), ("control", c.get("control"))):
            if not name: continue
            for split in "AB":
                for m in ("auc", "acc"):
                    rows.append(dict(candidate=c["name"], role=who, learner=name, evaluation="far", split=split, metric=m, **test(T.per_seed(df, split, m, name, ref, far, 9.0))))
                    rows.append(dict(candidate=c["name"], role=who, learner=name, evaluation="all", split=split, metric=m, **test(T.per_seed(df, split, m, name, ref))))
            for m in ("auc", "acc"): rows.append(dict(candidate=c["name"], role=who, learner=name, evaluation="blocked", split="blocked", metric=m, **test(per_seed_json(j3, m, name, ref))))
    out = pd.DataFrame(rows)
    for ev in ("far", "blocked"):
        sel = (out.evaluation == ev) & (out.role == "candidate"); out.loc[sel, "holm"] = holm(out.loc[sel, "p"].to_numpy())
    return out


def verdicts(out, spec):
    margin = spec.get("margin", 0.002); rows = []
    for c in spec["candidates"]:
        g = out[(out.candidate == c["name"]) & (out.role == "candidate")]; f = g[g.evaluation == "far"]; a = g[(g.evaluation == "all") & (g.metric == "auc")]; b = g[g.evaluation == "blocked"]
        pos = bool((f.gain > 0).all()); sig = int((f.holm < 0.05).sum()); std_ok = bool((a.gain >= -margin).all()); ctrl_ok = None
        if c.get("control"):
            cg = out[(out.candidate == c["name"]) & (out.role == "control") & (out.evaluation == "far")]; ctrl_ok = bool(all(f.set_index(["split", "metric"]).gain > cg.set_index(["split", "metric"]).gain))
        ok = pos and sig >= 3 and std_ok and (ctrl_ok is not False)
        rows.append(dict(candidate=c["name"], far_all_positive=pos, far_holm_sig=f"{sig}/4", standard_auc_not_worse=std_ok, beats_control=ctrl_ok, blocked_both_positive=bool((b.gain > 0).all()), blocked_holm_sig=f"{int((b.holm < 0.05).sum())}/2", SUCCESS=ok))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]; opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    spec = json.loads(Path(pos[0]).read_text()); out = run(spec, pos[1], pos[2], pos[3])
    text = "### Tests\n\n" + T.to_md(out) + "\n\n### Verdicts\n\n" + T.to_md(verdicts(out, spec), "{}")
    print(text)
    if "out" in opts: Path(opts["out"]).write_text(text + "\n")
