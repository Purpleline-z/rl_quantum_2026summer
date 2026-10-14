"""Cluster-aware uncertainty and power for the anchor-far stratum (generated numbers only).

  python3 af_power.py groups SEEDS                      write results/rt/af_jgroups_<seeds>.csv (seed, split, j, pair_id); no model is fitted
  python3 af_power.py power A1DIR A1BDIR GROUPS CAND REF [--lo=0.01 --hi=9 --nboot=400]

Why: the seed-level Wilcoxon treats the seeds as independent, but within a split every seed is evaluated on (almost) the same held-out judgments, so the effective number of
independent test units is the number of pair groups, not the number of seeds.  The two-way bootstrap resamples seeds AND pair groups (judgments of one pair group stay together); a
metric is computed on the resampled judgments of each (seed, budget, draw), averaged, and the paired gain cand - ref is formed on the same resample.
MDE80 = 2.8 x SE (two-sided 5 %, 80 % power, normal approximation) is the smallest true gain the design would detect.
"""
import sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ND = HERE.parents[1] / "new_directions"


def groups_for(seeds):
    sys.path.insert(0, str(ND)); import nd_core as core
    cache = core.feature_cache(); out = []
    with tempfile.TemporaryDirectory() as s:
        for seed in seeds:
            for split in "AB":
                ctx = core.make_ctx(seed, split, s, cache); rows = ctx.exp.rows_for(ctx.test); rows = rows[rows.Winner.isin(["1", "2"])].reset_index(drop=True)
                out.append(pd.DataFrame({"seed": seed, "split": split, "j": np.arange(len(rows)), "pair_id": rows.pair_id.to_numpy()}))
    return pd.concat(out, ignore_index=True)


def _auc(sign, d):
    pos, neg = d[sign > 0], d[sign < 0]
    if len(pos) == 0 or len(neg) == 0: return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def two_way(df, cand, ref, split, lo, hi, metric, nboot=400, seed=0):
    """df: merged per-judgment frame with pair_id and anc_all_min.  Returns dict(gain, se_seed, se_two_way, share, n_groups_far)."""
    g = df[(df.split == split) & (df.learner.isin([cand, ref]))].copy(); g["far"] = (g.anc_all_min > lo) & (g.anc_all_min <= hi)
    seeds = sorted(g.seed.unique()); pairs = sorted(g.pair_id.unique()); rng = np.random.default_rng(seed)
    cells = {}                                        # (seed, budget, draw) -> arrays
    for (s, b, dr), cc in g.groupby(["seed", "budget", "draw"]):
        a = cc[cc.learner == cand].sort_values("j"); r = cc[cc.learner == ref].sort_values("j")
        if len(a) != len(r) or len(a) == 0: continue
        cells.setdefault(s, []).append(dict(far=a.far.to_numpy(), pid=a.pair_id.to_numpy(), sign=a.sign.to_numpy(), w=a.weight.to_numpy(), dc=a.d.to_numpy(), dr=r.d.to_numpy()))

    def cell_gain(c, pair_w, use_far=True):
        m = c["far"] if use_far else np.ones(len(c["far"]), bool); w = pair_w[c["pid_idx"]] * m
        if metric == "acc":
            ww = c["w"] * w
            return np.nan if ww.sum() <= 0 else float((ww * (((c["dc"] * c["sign"]) > 0).astype(float) - ((c["dr"] * c["sign"]) > 0))).sum() / ww.sum())
        # AUC on a weighted resample: expand pair multiplicities
        rep = np.repeat(np.arange(len(w)), np.round(w).astype(int))
        if len(rep) == 0: return np.nan
        return _auc(c["sign"][rep], c["dc"][rep]) - _auc(c["sign"][rep], c["dr"][rep])

    pidx = {p: i for i, p in enumerate(pairs)}
    for s in cells:
        for c in cells[s]: c["pid_idx"] = np.array([pidx[p] for p in c["pid"]])
    ones = np.ones(len(pairs))
    per_seed = np.array([np.nanmean([cell_gain(c, ones) for c in cells[s]]) for s in seeds if s in cells]); gain = float(np.nanmean(per_seed))
    seed_only, two = [], []
    for _ in range(nboot):
        ss = rng.choice(list(cells), len(cells)); seed_only.append(np.nanmean([np.nanmean([cell_gain(c, ones) for c in cells[s]]) for s in ss]))
        pw = np.bincount(rng.integers(0, len(pairs), len(pairs)), minlength=len(pairs)).astype(float)
        two.append(np.nanmean([np.nanmean([cell_gain(c, pw) for c in cells[s]]) for s in ss]))
    share = float(np.mean([c["far"].mean() for s in cells for c in cells[s]]))
    return dict(split=split, metric=metric, gain=gain, se_seed=float(np.nanstd(seed_only)), se_two_way=float(np.nanstd(two)), mde80_two_way=2.8 * float(np.nanstd(two)),
                far_share=share, n_seeds=len(cells), n_pairs=len(pairs))


def main():
    mode = sys.argv[1]
    if mode == "groups":
        lo, hi = sys.argv[2].split("-"); df = groups_for(range(int(lo), int(hi) + 1)); out = ND / "results/rt" / f"af_jgroups_{sys.argv[2]}.csv"; df.to_csv(out, index=False); print(out, len(df)); return
    opts = dict(a[2:].split("=", 1) for a in sys.argv[2:] if a.startswith("--") and "=" in a); pos = [a for a in sys.argv[2:] if not a.startswith("--")]
    sys.path.insert(0, str(HERE)); import af_tables as T
    df = T.load(pos[0], pos[1]).merge(pd.read_csv(pos[2]), on=["seed", "split", "j"], how="left"); cand, ref = pos[3], pos[4]
    lo, hi = float(opts.get("lo", 0.01)), float(opts.get("hi", 9)); rows = []
    for split in "AB":
        for m in ("auc", "acc"): rows.append(two_way(df, cand, ref, split, lo, hi, m, int(opts.get("nboot", 400))))
    print(T.to_md(pd.DataFrame(rows)))


if __name__ == "__main__": main()
